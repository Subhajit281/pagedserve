import json
import statistics
import time
from pathlib import Path

import torch

from pagedserve.attention.triton_paged_decode import (
    triton_paged_decode_attention,
)
from pagedserve.attention.vectorized_batched_attention import (
    vectorized_paged_attention,
)
from pagedserve.engine.paged_kv_cache import PagedKVCache


def measure(fn, repeats=40):
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "repeats": repeats,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    device, dtype = "cuda", torch.bfloat16
    lengths = [32, 64, 128]
    ids = ["a", "b", "c"]

    cache = PagedKVCache(
        num_blocks=64, num_layers=1, num_kv_heads=2,
        head_dim=64, block_size=16, dtype=dtype, device=device
    )
    for rid in ids:
        cache.allocate(rid, 1)
    for rid, length in zip(ids, lengths):
        cache.append_tokens(rid, length - 1)
    for rid, length in zip(ids, lengths):
        for pos in range(length):
            k = torch.randn(2, 64, device=device, dtype=dtype)
            v = torch.randn_like(k)
            cache.write_token(rid, 0, pos, k, v)

    positions = [n - 1 for n in lengths]
    query = torch.randn(3, 4, 1, 64, device=device, dtype=dtype)

    def reference():
        return vectorized_paged_attention(
            query, cache, ids, positions, layer=0
        )

    def triton_fn():
        return triton_paged_decode_attention(
            query, cache, ids, positions, layer=0
        )

    with torch.inference_mode():
        for _ in range(10):
            reference()
            triton_fn()
    torch.cuda.synchronize()

    expected = reference()
    actual = triton_fn()
    torch.cuda.synchronize()
    max_error = (actual.float() - expected.float()).abs().max().item()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": str(dtype),
        "batch_size": len(ids),
        "sequence_lengths": lengths,
        "head_dim": 64,
        "block_size": cache.block_size,
        "parity": {"passed": True, "max_abs_error": max_error},
        "pytorch_vectorized": measure(reference),
        "triton_paged_decode": measure(triton_fn),
        "note": (
            "Includes Python metadata/table preparation and kernel launch "
            "in Triton timing; excludes model projections."
        ),
    }
    path = Path("bench/results/triton_paged_decode_20261016.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
