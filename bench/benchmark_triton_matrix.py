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


BATCH_SIZES = [1, 2, 4, 8]
SCENARIOS = {
    "short_boundary": [1, 15, 16, 17, 31, 32, 33, 65],
    "mixed": [5, 32, 65, 129, 17, 96, 257, 48],
    "long": [65, 127, 128, 129, 256, 257, 384, 512],
}
REPEATS = 30
WARMUP = 8
BLOCK_SIZE = 16
HEAD_DIM = 64
Q_HEADS = 4
KV_HEADS = 2


def make_case(batch_size, lengths, scenario):
    device = "cuda"
    dtype = torch.bfloat16
    ids = [f"{scenario}-{i}" for i in range(batch_size)]

    # Ample capacity for the longest tested batch.
    cache = PagedKVCache(
        num_blocks=512,
        num_layers=2,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        block_size=BLOCK_SIZE,
        dtype=dtype,
        device=device,
    )

    # Interleaved allocations create non-contiguous physical mappings.
    for rid in ids:
        cache.allocate(rid, 1)

    for rid, length in zip(ids, lengths):
        if length > 1:
            cache.append_tokens(rid, length - 1)

    # Initialize both layers independently; benchmark layer 0.
    for rid, length in zip(ids, lengths):
        for layer in range(2):
            for pos in range(length):
                key = torch.randn(
                    KV_HEADS, HEAD_DIM, device=device, dtype=dtype
                )
                value = torch.randn_like(key)
                cache.write_token(rid, layer, pos, key, value)

    query = torch.randn(
        batch_size, Q_HEADS, 1, HEAD_DIM, device=device, dtype=dtype
    )
    positions = [n - 1 for n in lengths]

    def reference():
        return vectorized_paged_attention(
            query, cache, ids, positions, layer=0
        )

    def triton_fn():
        return triton_paged_decode_attention(
            query, cache, ids, positions, layer=0
        )

    return cache, reference, triton_fn


def measure(fn):
    samples = []
    for _ in range(REPEATS):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)

    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "min_ms": min(samples),
        "repeats": REPEATS,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(20261016)
    results = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": "torch.bfloat16",
        "block_size": BLOCK_SIZE,
        "head_dim": HEAD_DIM,
        "warmup": WARMUP,
        "repeats": REPEATS,
        "cases": [],
    }

    for scenario, all_lengths in SCENARIOS.items():
        for batch_size in BATCH_SIZES:
            lengths = all_lengths[:batch_size]
            print(
                f"Running scenario={scenario}, batch={batch_size}, "
                f"lengths={lengths}",
                flush=True,
            )

            cache, reference, triton_fn = make_case(
                batch_size, lengths, scenario
            )

            with torch.inference_mode():
                for _ in range(WARMUP):
                    reference()
                    triton_fn()
                torch.cuda.synchronize()

                expected = reference()
                actual = triton_fn()
                torch.cuda.synchronize()

                max_error = (
                    actual.float() - expected.float()
                ).abs().max().item()

                try:
                    torch.testing.assert_close(
                        actual, expected, rtol=2e-2, atol=2e-2
                    )
                    parity_passed = True
                except AssertionError:
                    parity_passed = False
                    raise

                # Reset peak statistics after cache allocation and warmup,
                # so the measurement reflects temporary execution overhead.
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

                reference_timing = measure(reference)
                triton_timing = measure(triton_fn)

                peak_extra_mib = (
                    torch.cuda.max_memory_allocated()
                    - torch.cuda.memory_allocated()
                ) / (1024 ** 2)

            speedup = (
                reference_timing["median_ms"]
                / triton_timing["median_ms"]
            )

            case = {
                "scenario": scenario,
                "batch_size": batch_size,
                "sequence_lengths": lengths,
                "total_tokens_attended": sum(lengths),
                "parity": {
                    "passed": parity_passed,
                    "max_abs_error": max_error,
                },
                "pytorch_vectorized": reference_timing,
                "triton_paged_decode": triton_timing,
                "median_speedup": speedup,
                "peak_extra_allocated_mib_approx": peak_extra_mib,
            }
            results["cases"].append(case)

            print(
                f"  parity=PASS, max_error={max_error:.6g}, "
                f"PyTorch={reference_timing['median_ms']:.3f} ms, "
                f"Triton={triton_timing['median_ms']:.3f} ms, "
                f"speedup={speedup:.2f}x",
                flush=True,
            )

            del cache, reference, triton_fn
            torch.cuda.empty_cache()

    output = Path("bench/results/triton_matrix_20261016.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\nSaved results to: {output}")
    print("\nSUMMARY")
    for case in results["cases"]:
        print(
            f"{case['scenario']:14} "
            f"batch={case['batch_size']} "
            f"median speedup={case['median_speedup']:.2f}x "
            f"Triton={case['triton_paged_decode']['median_ms']:.3f} ms"
        )


if __name__ == "__main__":
    main()
