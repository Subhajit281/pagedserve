import json
import statistics
import time
from pathlib import Path

import torch

from pagedserve.model.engine import PagedServeEngine

PROMPTS = [
    "The capital of France is",
    "Explain what a CPU does in one sentence.",
    "What is the difference between RAM and storage?",
    "Describe how a transformer language model generates text.",
    "Why are data structures important in computer science?",
    "Explain the purpose of an operating system.",
    "What is an algorithm?",
    "Describe how a GPU accelerates neural network inference.",
]
CONCURRENCIES = [1, 2, 4, 8]
MAX_NEW_TOKENS = 32
REPEATS = 3
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_wave(engine, prompts, concurrency):
    sync()
    started = time.perf_counter()
    outputs = engine.generate_scheduled(
        prompts,
        max_new_tokens=MAX_NEW_TOKENS,
        max_running_requests=concurrency,
    )
    sync()
    elapsed = time.perf_counter() - started

    if engine.cache.used_blocks != 0:
        raise RuntimeError(
            f"Cache leak: {engine.cache.used_blocks} blocks still used"
        )

    generated_tokens = sum(
        max(
            0,
            len(engine.tokenizer(text, add_special_tokens=True)["input_ids"])
            - len(engine.tokenizer(prompt, add_special_tokens=True)["input_ids"]),
        )
        for prompt, text in zip(prompts, outputs)
    )

    return {
        "elapsed_seconds": elapsed,
        "requests": len(prompts),
        "generated_tokens_estimate": generated_tokens,
        "requests_per_second": len(prompts) / elapsed,
        "aggregate_generated_tokens_per_second": generated_tokens / elapsed,
        "cache_clean": engine.cache.used_blocks == 0,
        "outputs": outputs,
    }


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=256,
        block_size=16,
        attention_backend="pytorch",
    )

    results = {
        "model": MODEL_ID,
        "dtype": "bfloat16",
        "max_new_tokens": MAX_NEW_TOKENS,
        "repeats": REPEATS,
        "note": (
            "Experimental Triton integration has a known full-model logit "
            "parity failure. Performance does not establish correctness."
        ),
        "runs": [],
    }

    try:
        # Warm both paths before collecting measurements.
        for backend in ("pytorch", "triton"):
            engine.attention_backend = backend
            try:
                run_wave(engine, PROMPTS[:2], 2)
            except Exception as exc:
                print(f"Warmup failed for {backend}: {exc}")
                raise

        for concurrency in CONCURRENCIES:
            prompts = PROMPTS[:concurrency]

            for backend in ("pytorch", "triton"):
                engine.attention_backend = backend
                timings = []
                outputs = []
                token_rates = []
                peak_memory = None

                for repeat in range(REPEATS):
                    torch.cuda.reset_peak_memory_stats()
                    run = run_wave(engine, prompts, concurrency)
                    timings.append(run["elapsed_seconds"])
                    outputs.append(run.pop("outputs"))
                    token_rates.append(
                        run["aggregate_generated_tokens_per_second"]
                    )
                    peak_memory = max(
                        peak_memory or 0,
                        torch.cuda.max_memory_allocated() / (1024 ** 2),
                    )

                record = {
                    "backend": backend,
                    "concurrency": concurrency,
                    "median_elapsed_seconds": statistics.median(timings),
                    "median_requests_per_second": statistics.median(
                        concurrency / t for t in timings
                    ),
                    "median_aggregate_generated_tokens_per_second": (
                        statistics.median(token_rates)
                    ),
                    "peak_allocated_mib": peak_memory,
                    "cache_clean_each_run": engine.cache.used_blocks == 0,
                    "outputs": outputs[-1],
                }
                results["runs"].append(record)

                print(
                    f"{backend:7s} concurrency={concurrency} "
                    f"latency={record['median_elapsed_seconds']:.3f}s "
                    f"requests/s={record['median_requests_per_second']:.2f} "
                    f"tokens/s={record['median_aggregate_generated_tokens_per_second']:.2f} "
                    f"peak={peak_memory:.1f}MiB"
                )

        # Compare final-repeat outputs for each concurrency level.
        for concurrency in CONCURRENCIES:
            pair = [
                r for r in results["runs"]
                if r["concurrency"] == concurrency
            ]
            pytorch = next(r for r in pair if r["backend"] == "pytorch")
            triton = next(r for r in pair if r["backend"] == "triton")
            equality = pytorch["outputs"] == triton["outputs"]
            results.setdefault("output_comparison", []).append({
                "concurrency": concurrency,
                "exact_output_match": equality,
            })
            print(
                f"output equality concurrency={concurrency}: {equality}"
            )

        output_path = Path("bench/results/integrated_backends_20261016.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nSaved results: {output_path}")

    finally:
        engine.attention_backend = "pytorch"
        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
