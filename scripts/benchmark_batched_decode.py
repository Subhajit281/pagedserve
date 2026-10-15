import statistics
import time

import torch

from pagedserve.model.engine import PagedServeEngine


PROMPTS = [
    "The capital of France is",
    "A CPU is responsible for",
    "The result of 15 multiplied by 12 is",
]

MAX_NEW_TOKENS = 32
WARMUP_RUNS = 1
BENCHMARK_RUNS = 3


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    engine = PagedServeEngine(
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=512,
        block_size=16,
    )

    print("\n========== WARMUP ==========")

    for _ in range(WARMUP_RUNS):
        engine.generate_many(PROMPTS, max_new_tokens=MAX_NEW_TOKENS)

    synchronize()

    print("\n========== BATCHED GENERATION ==========")

    batched_times = []

    for run in range(BENCHMARK_RUNS):
        synchronize()
        start = time.perf_counter()

        results = engine.generate_many(
            PROMPTS,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        synchronize()
        elapsed = time.perf_counter() - start
        batched_times.append(elapsed)

        print(f"Run {run + 1}: {elapsed:.3f}s")

        if run == 0:
            for index, result in enumerate(results, start=1):
                print(f"\nRequest {index}: {result}")

    print("\n========== INDEPENDENT GENERATION ==========")

    independent_times = []

    for run in range(BENCHMARK_RUNS):
        synchronize()
        start = time.perf_counter()

        results = [
            engine.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
            for prompt in PROMPTS
        ]

        synchronize()
        elapsed = time.perf_counter() - start
        independent_times.append(elapsed)

        print(f"Run {run + 1}: {elapsed:.3f}s")

    batched_median = statistics.median(batched_times)
    independent_median = statistics.median(independent_times)

    print("\n========== SUMMARY ==========")
    print(f"Requests per workload: {len(PROMPTS)}")
    print(f"Maximum new tokens/request: {MAX_NEW_TOKENS}")
    print(f"Batched median: {batched_median:.3f}s")
    print(f"Independent median: {independent_median:.3f}s")

    if batched_median > 0:
        print(
            "Relative speedup: "
            f"{independent_median / batched_median:.2f}x"
        )

    print(f"KV-cache blocks remaining: {engine.cache.used_blocks}")

    assert engine.cache.used_blocks == 0, "KV-cache blocks leaked."


if __name__ == "__main__":
    main()
