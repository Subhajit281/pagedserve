import argparse
import json
import statistics
import time
import sys
from collections import Counter
from pathlib import Path

# Allow direct execution from the bench/ directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from pagedserve.engine.continuous_batcher import ContinuousBatcher
from pagedserve.model.engine import PagedServeEngine


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "The capital of France is",
    (
        "Explain how a hash table works, including hash functions, "
        "collisions, and average lookup complexity."
    ),
    (
        "Explain operating systems in detail. Discuss processes, "
        "threads, CPU scheduling, virtual memory, page tables, "
        "caching, synchronization, deadlocks, file systems, and "
        "input output management. Include examples and explain "
        "how these components interact during program execution."
    ),
]


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark PAGEDSERVE online continuous batching."
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Concurrent requests per wave.",
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output",
        default="bench/results/continuous_batcher_20261016.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")
    if args.repeats < 1 or args.max_new_tokens < 1:
        parser.error("--repeats and --max-new-tokens must be positive.")
    if any(n < 1 or n > 8 for n in args.concurrency):
        parser.error("--concurrency values must be between 1 and 8.")

    device = torch.device("cuda:0")

    print(f"Loading {MODEL_ID}...")
    synchronize()
    load_start = time.perf_counter()

    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=128,
        block_size=16,
    )

    synchronize()
    load_seconds = time.perf_counter() - load_start

    results = {
        "benchmark": "pagedserve_online_continuous_batching",
        "model": MODEL_ID,
        "device": torch.cuda.get_device_name(device),
        "dtype": str(engine.dtype),
        "block_size": engine.block_size,
        "num_blocks": engine.cache.num_blocks,
        "model_load_seconds": load_seconds,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "benchmarks": {},
        "notes": [
            "Each wave submits requests concurrently to the online batcher.",
            "Prefill is performed independently for newly admitted requests.",
            "Decode batch sizes record actual _forward_paged_batch calls.",
            "Throughput counts generated tokens from completed Request objects.",
            "This benchmark measures performance; it does not assert exact "
            "token parity against Hugging Face.",
        ],
    }

    original_forward_batch = engine._forward_paged_batch
    observed_batch_sizes = []

    def record_batch(input_ids, request_ids, positions):
        observed_batch_sizes.append(len(request_ids))
        return original_forward_batch(
            input_ids=input_ids,
            request_ids=request_ids,
            positions=positions,
        )

    engine._forward_paged_batch = record_batch

    try:
        for concurrency in args.concurrency:
            print(f"\n=== Concurrency {concurrency} ===")
            observed_batch_sizes.clear()

            with ContinuousBatcher(
                engine,
                max_running_requests=concurrency,
            ) as batcher:
                # Warm up model execution and the batcher worker.
                warmup_id = batcher.submit(
                    PROMPTS[0],
                    max_new_tokens=min(args.max_new_tokens, 4),
                )
                batcher.result(warmup_id, timeout=120)
                synchronize()
                observed_batch_sizes.clear()
                torch.cuda.reset_peak_memory_stats(device)

                wave_times = []
                request_latencies = []
                wave_throughputs = []
                generated_counts = []
                expected_request_count = concurrency * args.repeats
                completed_request_count = 0
                all_requests_completed = True
                all_token_counts_valid = True
                all_cache_clean = True

                for repeat in range(args.repeats):
                    wave_start = time.perf_counter()
                    submitted_at = {}
                    request_ids = []

                    for index in range(concurrency):
                        prompt = PROMPTS[index % len(PROMPTS)]
                        request_id = batcher.submit(
                            prompt,
                            max_new_tokens=args.max_new_tokens,
                        )
                        request_ids.append(request_id)
                        submitted_at[request_id] = time.perf_counter()

                    # Observe completion times without waiting serially
                    # on the first request in the submission list.
                    completion_times = {}
                    deadline = time.monotonic() + 300

                    while len(completion_times) < len(request_ids):
                        now = time.perf_counter()

                        for request_id in request_ids:
                            if request_id in completion_times:
                                continue

                            job = batcher._jobs[request_id]
                            if job.done.is_set():
                                completion_times[request_id] = now
                                request_latencies.append(
                                    now - submitted_at[request_id]
                                )

                        if len(completion_times) == len(request_ids):
                            break

                        if time.monotonic() > deadline:
                            raise TimeoutError(
                                f"Timed out at concurrency={concurrency}, "
                                f"repeat={repeat + 1}"
                            )
                        time.sleep(0.001)

                    # Propagate errors and validate every completed request.
                    wave_generated = 0
                    for request_id in request_ids:
                        batcher.result(request_id, timeout=1)
                        job = batcher._jobs[request_id]

                        if not job.done.is_set():
                            all_requests_completed = False
                            raise RuntimeError(
                                f"Request did not complete: {request_id}"
                            )

                        if job.request is None:
                            all_requests_completed = False
                            raise RuntimeError(
                                f"Missing request state for {request_id}"
                            )

                        token_count = len(job.request.generated_token_ids)
                        if not 1 <= token_count <= args.max_new_tokens:
                            all_token_counts_valid = False
                            raise RuntimeError(
                                f"Invalid generated-token count for "
                                f"{request_id}: {token_count}; expected "
                                f"between 1 and {args.max_new_tokens}"
                            )

                        completed_request_count += 1
                        wave_generated += token_count

                    if len(request_ids) != concurrency:
                        all_requests_completed = False
                        raise RuntimeError(
                            f"Expected {concurrency} requests in the wave, "
                            f"submitted {len(request_ids)}"
                        )

                    synchronize()
                    wave_elapsed = time.perf_counter() - wave_start
                    wave_times.append(wave_elapsed)
                    generated_counts.append(wave_generated)
                    wave_throughputs.append(
                        wave_generated / wave_elapsed
                    )

                    cache_clean = engine.cache.used_blocks == 0
                    all_cache_clean &= cache_clean
                    if not cache_clean:
                        raise RuntimeError(
                            "KV-cache leak detected after a benchmark wave."
                        )

                    print(
                        f"  Repeat {repeat + 1}: "
                        f"{wave_generated} tokens in {wave_elapsed:.3f}s "
                        f"({wave_generated / wave_elapsed:.2f} tok/s); "
                        f"cache clean={cache_clean}"
                    )

                batch_sizes = Counter(observed_batch_sizes)
                total_generated = sum(generated_counts)
                total_elapsed = sum(wave_times)

                entry = {
                    "concurrency": concurrency,
                    "generated_tokens_per_wave": generated_counts,
                    "wave_seconds": wave_times,
                    "wave_throughput_tokens_per_second": wave_throughputs,
                    "aggregate_throughput_tokens_per_second": (
                        total_generated / total_elapsed
                    ),
                    "request_latency_seconds_mean": (
                        statistics.mean(request_latencies)
                        if request_latencies else None
                    ),
                    "request_latency_seconds_p50": (
                        statistics.median(request_latencies)
                        if request_latencies else None
                    ),
                    "request_latency_seconds_max": (
                        max(request_latencies)
                        if request_latencies else None
                    ),
                    "decode_batch_size_histogram": {
                        str(size): count
                        for size, count in sorted(batch_sizes.items())
                    },
                    "decode_forward_calls": len(observed_batch_sizes),
                    "peak_gpu_memory_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    ),
                    "expected_request_count": expected_request_count,
                    "completed_request_count": completed_request_count,
                    "all_requests_completed": (
                        all_requests_completed
                        and completed_request_count == expected_request_count
                    ),
                    "all_token_counts_valid": all_token_counts_valid,
                    "cache_clean_after_every_wave": all_cache_clean,
                }

                results["benchmarks"][str(concurrency)] = entry

                print(
                    "  Aggregate throughput: "
                    f"{entry['aggregate_throughput_tokens_per_second']:.2f} "
                    "generated tokens/s"
                )
                print(
                    "  Mean request latency: "
                    f"{entry['request_latency_seconds_mean'] * 1000:.2f} ms"
                )
                print(
                    "  Decode batch-size histogram: "
                    f"{entry['decode_batch_size_histogram']}"
                )
                print(
                    "  Peak GPU memory: "
                    f"{entry['peak_gpu_memory_mb']:.2f} MiB"
                )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )
        print(f"\nResults saved to: {output_path}")

    finally:
        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
