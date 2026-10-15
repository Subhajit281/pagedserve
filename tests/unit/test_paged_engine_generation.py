import pytest
import torch

from pagedserve.model.engine import PagedServeEngine


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_paged_engine_generates_and_releases_cache():
    engine = PagedServeEngine(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=32,
        block_size=16,
    )

    prompt = "The capital of France is"
    free_blocks_before = engine.cache.free_blocks

    try:
        generated_text = engine.generate(
            prompt=prompt,
            max_new_tokens=8,
        )

        print(f"\nPrompt: {prompt}")
        print(f"Generated text: {generated_text}")

        assert isinstance(generated_text, str)
        assert generated_text.strip()
        assert engine.cache.free_blocks == free_blocks_before
        assert engine.cache.used_blocks == 0

    finally:
        del engine
        torch.cuda.empty_cache()
