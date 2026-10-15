import pytest
import torch

from pagedserve.model.engine import PagedServeEngine


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.xfail(
    strict=True,
    reason="Known BF16 full-model logit divergence in experimental Triton backend",
)
def test_triton_engine_batched_decode_matches_pytorch():
    """Compare real Qwen batched decode using equivalent KV-cache state."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=64,
        block_size=16,
        attention_backend="pytorch",
    )

    request_ids = ["triton-integration-a", "triton-integration-b"]
    lengths = [17, 33]
    positions = [length - 1 for length in lengths]

    try:
        # Allocate identical logical capacities for both requests.
        for request_id, length in zip(request_ids, lengths):
            engine.cache.allocate(request_id, length)

        # Fill the cache with deterministic, nonzero K/V data for every layer.
        # This isolates the batched decode attention backend from prefill.
        generator = torch.Generator(device="cuda")
        generator.manual_seed(12345)

        for request_id, length in zip(request_ids, lengths):
            for layer in range(engine.model.config.num_hidden_layers):
                for position in range(length):
                    key = torch.randn(
                        engine.cache.kv_cache.num_kv_heads,
                        engine.cache.kv_cache.head_dim,
                        generator=generator,
                        device="cuda",
                        dtype=engine.dtype,
                    )
                    value = torch.randn_like(key)
                    engine.cache.write_token(
                        request_id, layer, position, key, value
                    )

        # Snapshot cache storage so both paths see exactly the same K/V.
        keys_before = engine.cache.kv_cache.k_cache.clone()
        values_before = engine.cache.kv_cache.v_cache.clone()

        input_ids = torch.tensor(
            [[12095], [12095]],
            device=engine.device,
            dtype=torch.long,
        )

        with torch.inference_mode():
            pytorch_logits = engine._forward_paged_batch(
                input_ids=input_ids,
                request_ids=request_ids,
                positions=positions,
            )

            # Restore the cache because each forward writes the current K/V.
            engine.cache.kv_cache.k_cache.copy_(keys_before)
            engine.cache.kv_cache.v_cache.copy_(values_before)

            engine.attention_backend = "triton"
            triton_logits = engine._forward_paged_batch(
                input_ids=input_ids,
                request_ids=request_ids,
                positions=positions,
            )

        assert pytorch_logits.shape == triton_logits.shape
        assert torch.isfinite(pytorch_logits).all()
        assert torch.isfinite(triton_logits).all()

        max_diff = (
            pytorch_logits.float() - triton_logits.float()
        ).abs().max().item()
        mean_diff = (
            pytorch_logits.float() - triton_logits.float()
        ).abs().mean().item()

        print(f"\nTriton integration max logit diff:  {max_diff:.6f}")
        print(f"Triton integration mean logit diff: {mean_diff:.6f}")

        # Attention outputs are close within BF16 tolerance. The full model's
        # logits can amplify small attention differences across layers.
        torch.testing.assert_close(
            triton_logits.float(),
            pytorch_logits.float(),
            rtol=5e-2,
            atol=5e-2,
        )

    finally:
        engine.attention_backend = "pytorch"
        for request_id in request_ids:
            if request_id in engine.cache._block_tables:
                engine.cache.free(request_id)
        assert engine.cache.used_blocks == 0
        del engine
        torch.cuda.empty_cache()
