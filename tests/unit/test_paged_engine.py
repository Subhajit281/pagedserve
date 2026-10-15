import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pagedserve.model.engine import PagedServeEngine


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_model_paged_logits_match_reference():
    device = "cuda:0"
    dtype = torch.bfloat16

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device=device,
        dtype=dtype,
        num_blocks=16,
        block_size=16,
    )

    request_id = "full-model-test"

    try:
        encoded = engine.tokenizer(
            "The capital of France is",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        sequence_length = input_ids.shape[1]

        engine.cache.allocate(request_id, sequence_length)

        with torch.inference_mode():
            # Original Hugging Face forward pass.
            reference_logits = engine.model(
                input_ids=input_ids,
                use_cache=False,
            ).logits

            # PagedServe forward pass through every transformer layer.
            paged_logits = engine._forward_paged(
                input_ids=input_ids,
                request_id=request_id,
                position_start=0,
            )

        assert paged_logits.shape == reference_logits.shape

        max_diff = (
            paged_logits.float() - reference_logits.float()
        ).abs().max().item()

        mean_diff = (
            paged_logits.float() - reference_logits.float()
        ).abs().mean().item()

        print(f"\nMaximum logit difference: {max_diff:.6f}")
        print(f"Mean logit difference:    {mean_diff:.6f}")

        torch.testing.assert_close(
            paged_logits,
            reference_logits,
            rtol=2e-2,
            atol=2e-2,
        )

    finally:
        if request_id in engine.cache._block_tables:
            engine.cache.free(request_id)

        del engine
        torch.cuda.empty_cache()
