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
def test_qwen_paged_attention_matches_reference():
    device = "cuda:0"
    dtype = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    try:
        attention = model.model.layers[0].self_attn
        hidden_size = model.config.hidden_size
        num_kv_heads = model.config.num_key_value_heads
        head_dim = model.config.hidden_size // model.config.num_attention_heads
        sequence_length = 5

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

            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(
                query,
                key,
                cos,
                sin,
            )

            # Reference attention must use causal masking.
            causal_mask = torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                dtype=query.dtype,
                device=query.device,
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask = causal_mask[None, None, :, :]

            reference, _ = eager_attention_forward(
                attention,
                query,
                key,
                value,
                attention_mask=causal_mask,
                scaling=attention.scaling,
                dropout=0.0,
            )
            reference = reference.reshape(1, sequence_length, hidden_size).contiguous()
            reference = attention.o_proj(reference)

            cache = PagedKVCache(
                num_blocks=4,
                num_layers=model.config.num_hidden_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                block_size=2,
                dtype=dtype,
                device=device,
            )
            cache.allocate("test-request", sequence_length)

            actual = qwen_paged_attention(
                attention=attention,
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                cache=cache,
                request_id="test-request",
                query_start_position=0,
            )

            assert actual.shape == reference.shape
            assert cache.token_count("test-request") == sequence_length

            torch.testing.assert_close(
                actual,
                reference,
                rtol=2e-2,
                atol=2e-2,
            )

            cache.free("test-request")

    finally:
        del model
        torch.cuda.empty_cache()
