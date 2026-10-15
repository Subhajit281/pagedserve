import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pagedserve.engine.paged_kv_cache import PagedKVCache


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)


def test_qwen_kv_round_trip():
    device = "cuda:0"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(device)

    model.eval()

    messages = [
        {
            "role": "user",
            "content": "Explain paged KV caching briefly.",
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )

    reference_cache = outputs.past_key_values
    sequence_length = reference_cache.get_seq_length()

    num_layers = model.config.num_hidden_layers
    num_kv_heads = model.config.num_key_value_heads
    head_dim = (
        model.config.hidden_size
        // model.config.num_attention_heads
    )
    block_size = 16

    num_blocks = (
        sequence_length + block_size - 1
    ) // block_size

    cache = PagedKVCache(
        num_blocks=num_blocks,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )

    cache.allocate("request-A", sequence_length)

    for layer in range(num_layers):
        key = reference_cache.layers[layer].keys[0]
        value = reference_cache.layers[layer].values[0]

        assert key.shape == (
            num_kv_heads,
            sequence_length,
            head_dim,
        )
        assert value.shape == key.shape

        for position in range(sequence_length):
            cache.write_token(
                request_id="request-A",
                layer=layer,
                token_position=position,
                key=key[:, position, :],
                value=value[:, position, :],
            )

    for layer in range(num_layers):
        key = reference_cache.layers[layer].keys[0]
        value = reference_cache.layers[layer].values[0]

        for position in range(sequence_length):
            actual_key, actual_value = cache.read_token(
                request_id="request-A",
                layer=layer,
                token_position=position,
            )

            torch.testing.assert_close(
                actual_key,
                key[:, position, :],
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                actual_value,
                value[:, position, :],
                rtol=0,
                atol=0,
            )

    print(
        f"\nQwen K/V round-trip passed: "
        f"{num_layers} layers, "
        f"{sequence_length} tokens, "
        f"{num_blocks} blocks."
    )

    cache.free("request-A")

    del model, outputs, reference_cache, cache
    torch.cuda.empty_cache()
