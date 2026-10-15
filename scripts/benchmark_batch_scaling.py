import gc
import statistics
import time

import torch

from pagedserve.model.engine import PagedServeEngine


PROMPTS = [
    "The capital of France is",
    "A CPU is responsible for",
    "The result of 15 multiplied by 12 is",
    "Python is a programming language used for",
]

BATCH_SIZES = [1, 2, 3, 4]
MAX_NEW_TOKENS = 32
WARMUP_RUNS = 1
BENCHMARK_RUNS = 5


def synchronize():
    torch.cuda.synchronize()


def benchmark_batch(engine, prompts):
    latencies = []
    generated_tokens = 0
    peak_memory_mb = 0.0

    for run in range(WARMUP_RUNS + BENCHMARK_RUNS):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        synchronize()
        start = time.perf_counter()

        results = engine.generate_many(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        synchronize()
        elapsed = time.perf_counter() - start

        assert len(results) == len(prompts)
        assert all(isinstance(result, str) for result in results)

        if engine.cache.used_blocks != 0:
            raise AssertionError("KV-cache allocations leaked.")

        if run >= WARMUP_RUNS:
            latencies.append(elapsed)

            # Count generated tokens using the tokenizer rather than
            # assuming every request generated MAX_NEW_TOKENS.
            total_tokens = 0

            for prompt, result in zip(prompts, results):
                prompt_tokens = len(
                    engine.tokenizer(
                        prompt,
                        add_special_tokens=True,
                    )["input_ids"]
                )

                result_tokens = len(
                    engine.tokenizer(
                        result,
                        add_special_tokens=True,
                    )["input_ids"]
                )

                total_tokens += max(0, result_tokens - prompt_tokens)

            generated_tokens = total_tokens
            peak_memory_mb = max(
                peak_memory_mb,
                torch.cuda.max_memory_allocated() / (1024 ** 2),
            )

    median_latency = statistics.median(latencies)

    return {
        "batch_size": len(prompts),
        "latency": median_latency,
        "throughput": generated_tokens / median_latency,
        "per_request_latency": median_latency / len(prompts),
        "peak_memory_mb": peak_memory_mb,
        "runs": latencies,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    engine = PagedServeEngine(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=512,
        block_size=16,
    )

    results = []

    try:
        print("========== PAGEDSERVE BATCH SCALING ==========")
        print(f"Maximum new tokens/request: {MAX_NEW_TOKENS}")
        print(f"Timed runs per batch size: {BENCHMARK_RUNS}")

        for batch_size in BATCH_SIZES:
            prompts = PROMPTS[:batch_size]

            print(f"\nBenchmarking batch size {batch_size}...")

            result = benchmark_batch(engine, prompts)
            results.append(result)

            print(
                f"Median latency:       {result['latency']:.3f}s\n"
                f"Aggregate throughput: {result['throughput']:.2f} tokens/s\n"
                f"Latency/request:      "
                f"{result['per_request_latency']:.3f}s\n"
                f"Peak GPU memory:      "
                f"{result['peak_memory_mb']:.1f} MiB\n"
                f"Cache blocks remaining: {engine.cache.used_blocks}"
            )

        print("\n========== SUMMARY ==========")
        print(
            f"{'Batch':>5} {'Latency(s)':>11} "
            f"{'Tokens/s':>10} {'Per-request(s)':>15} "
            f"{'Peak MiB':>10}"
        )

        for result in results:
            print(
                f"{result['batch_size']:>5} "
                f"{result['latency']:>11.3f} "
                f"{result['throughput']:>10.2f} "
                f"{result['per_request_latency']:>15.3f} "
                f"{result['peak_memory_mb']:>10.1f}"
            )

        print("\nPASS: All benchmark requests released their KV caches.")

    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
