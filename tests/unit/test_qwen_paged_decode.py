import pytest
import torch

from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from pagedserve.attention.qwen_paged_attention import qwen_paged_attention
from pagedserve.engine.paged_kv_cache import PagedKVCache


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_qwen_paged_prefill_and_decode_match_reference():
    device = "cuda:0"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device).eval()

    cache = None

    try:
        attention = model.model.layers[0].self_attn

        sequence_length = 6
        prefill_length = 5
        hidden_size = model.config.hidden_size
        head_dim = attention.head_dim

        hidden_states = torch.randn(
            1,
            sequence_length,
            hidden_size,
            device=device,
            dtype=dtype,
        )

        position_ids = torch.arange(
            sequence_length,
            device=device,
        ).unsqueeze(0)

        with torch.inference_mode():
            position_embeddings = model.model.rotary_emb(
                hidden_states,
                position_ids,
            )

            # Reference: process all six tokens together.
            projection_shape = (1, sequence_length, -1, head_dim)

            query = (
                attention.q_proj(hidden_states)
                .view(projection_shape)
                .transpose(1, 2)
            )
            key = (
                attention.k_proj(hidden_states)
                .view(projection_shape)
                .transpose(1, 2)
            )
            value = (
                attention.v_proj(hidden_states)
                .view(projection_shape)
                .transpose(1, 2)
            )

            query, key = apply_rotary_pos_emb(
                query,
                key,
                *position_embeddings,
            )

            causal_mask = torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                dtype=dtype,
                device=device,
            ).triu(diagonal=1)

            reference, _ = eager_attention_forward(
                attention,
                query,
                key,
                value,
                attention_mask=causal_mask[None, None, :, :],
                scaling=attention.scaling,
                dropout=0.0,
            )

            reference = reference.reshape(
                1, sequence_length, hidden_size
            ).contiguous()

            reference = attention.o_proj(reference)

            # PagedServe cache for prefill plus one decode token.
            cache = PagedKVCache(
                num_blocks=4,
                num_layers=model.config.num_hidden_layers,
                num_kv_heads=model.config.num_key_value_heads,
                head_dim=head_dim,
                block_size=2,
                dtype=dtype,
                device=device,
            )

            cache.allocate("decode-test", sequence_length)

            prefill_embeddings = tuple(
                tensor[:, :prefill_length]
                for tensor in position_embeddings
            )

            # Stage 1: prefill five tokens.
            prefill_output = qwen_paged_attention(
                attention=attention,
                hidden_states=hidden_states[:, :prefill_length],
                position_embeddings=prefill_embeddings,
                cache=cache,
                request_id="decode-test",
                query_start_position=0,
            )

            # Stage 2: decode the sixth token using the existing cache.
            decode_embeddings = tuple(
                tensor[:, prefill_length:sequence_length]
                for tensor in position_embeddings
            )

            decode_output = qwen_paged_attention(
                attention=attention,
                hidden_states=hidden_states[:, prefill_length:],
                position_embeddings=decode_embeddings,
                cache=cache,
                request_id="decode-test",
                query_start_position=prefill_length,
            )

            actual = torch.cat(
                [prefill_output, decode_output],
                dim=1,
            )

            assert actual.shape == reference.shape

            # Compare prefill and decode independently.
            torch.testing.assert_close(
                prefill_output,
                reference[:, :prefill_length],
                rtol=2e-2,
                atol=2e-2,
            )

            torch.testing.assert_close(
                decode_output,
                reference[:, prefill_length:],
                rtol=2e-2,
                atol=2e-2,
            )

            torch.testing.assert_close(
                actual,
                reference,
                rtol=2e-2,
                atol=2e-2,
            )

        print("PASS: Prefill matches Hugging Face.")
        print("PASS: Single-token decode matches Hugging Face.")
        print("PASS: Combined prefill/decode output matches reference.")

    finally:
        if cache is not None:
            cache.free("decode-test")

        del model
        torch.cuda.empty_cache()
