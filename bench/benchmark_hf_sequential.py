import argparse
import gc
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

# Exact prompts from bench/benchmark_pagedserve.py
PROMPTS = {
    "short": "The capital of France is",
    "medium": (
        "Explain how a hash table works, including hash functions, "
        "collisions, and average lookup complexity."
    ),
    "long": (
        "Explain operating systems in detail. Discuss processes, "
        "threads, CPU scheduling, virtual memory, page tables, "
        "caching, synchronization, deadlocks, file systems, and "
        "input output management. Include examples and explain "
        "how these components interact during program execution."
    ),
}


def synchronize():
    torch.cuda.synchronize()


def percentile_nearest_rank(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(
        description="S0: Hugging Face sequential generation benchmark"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output",
        default="bench/results/hf_sequential_matched_20261015.json",
    )
    args = parser.parse_args()

    if args.repeats < 1 or args.warmups < 0 or args.max_new_tokens < 1:
        parser.error("repeats and max-new-tokens must be positive; warmups cannot be negative")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    output_path = Path(args.output)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing results: {output_path}"
        )

    device = torch.device("cuda:0")
    print("Loading standalone Hugging Face model...")
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()

    synchronize()
    model_load_seconds = time.perf_counter() - load_start

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model load: {model_load_seconds:.3f}s")
    print(f"Warmups: {args.warmups}; measured repetitions: {args.repeats}")
    print(f"Maximum new tokens: {args.max_new_tokens}")

    results = {}

    try:
        for name, prompt in PROMPTS.items():
            encoded = tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            prompt_length = input_ids.shape[1]

            print(
                f"\nBenchmarking {name}: {prompt_length} prompt tokens, "
                f"{args.max_new_tokens} requested new tokens"
            )

            generation_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "max_new_tokens": args.max_new_tokens,
                "do_sample": False,
                "use_cache": True,
                "num_return_sequences": 1,
                "pad_token_id": (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                ),
            }

            with torch.inference_mode():
                for _ in range(args.warmups):
                    warmup_output = model.generate(**generation_kwargs)
                    del warmup_output
                synchronize()

                latencies = []
                generated_counts = []
                peak_memory_mib = 0.0
                sample_completion = None

                for _ in range(args.repeats):
                    gc.collect()
                    torch.cuda.empty_cache()
                    synchronize()
                    torch.cuda.reset_peak_memory_stats(device)

                    start = time.perf_counter()
                    output_ids = model.generate(**generation_kwargs)
                    synchronize()
                    elapsed = time.perf_counter() - start

                    completion_ids = output_ids[0, prompt_length:]
                    generated_count = int(completion_ids.numel())

                    latencies.append(elapsed)
                    generated_counts.append(generated_count)
                    peak_memory_mib = max(
                        peak_memory_mib,
                        torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                    )
                    sample_completion = tokenizer.decode(
                        completion_ids,
                        skip_special_tokens=True,
                    )
                    del output_ids

            if any(count < 1 for count in generated_counts):
                raise RuntimeError(
                    f"No completion tokens generated for {name}: {generated_counts}"
                )

            mean_latency = statistics.mean(latencies)
            mean_generated = statistics.mean(generated_counts)
            throughput = mean_generated / mean_latency

            results[name] = {
                "prompt": prompt,
                "prompt_tokens": prompt_length,
                "requested_new_tokens": args.max_new_tokens,
                "generated_tokens_per_repeat": generated_counts,
                "latency_seconds_runs": latencies,
                "total_seconds_mean": mean_latency,
                "total_seconds_p50": statistics.median(latencies),
                "total_seconds_p95_nearest_rank": percentile_nearest_rank(
                    latencies, 0.95
                ),
                "generation_throughput_tokens_per_second": throughput,
                "peak_gpu_memory_mb": peak_memory_mib,
                "sample_completion": sample_completion,
            }

            print(f"  Generated tokens: {generated_counts}")
            print(f"  Mean latency:     {mean_latency * 1000:.2f} ms")
            print(f"  Median latency:   {statistics.median(latencies) * 1000:.2f} ms")
            print(f"  Throughput:       {throughput:.2f} tokens/s")
            print(f"  Peak GPU memory:  {peak_memory_mib:.2f} MiB")

        payload = {
            "benchmark": "S0_huggingface_sequential_generate_matched",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "device": torch.cuda.get_device_name(device),
            "dtype": "torch.bfloat16",
            "attention_implementation": "eager",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "max_new_tokens": args.max_new_tokens,
            "generation": {
                "do_sample": False,
                "use_cache": True,
                "num_return_sequences": 1,
            },
            "model_load_seconds": model_load_seconds,
            "benchmarks": results,
            "notes": [
                "Standalone Hugging Face model; PagedServe KV cache is not constructed.",
                "Prompts match bench/benchmark_pagedserve.py.",
                "Latency includes model.generate prefill and decoding.",
                "Throughput is mean generated completion tokens divided by mean end-to-end latency.",
                "Peak memory is torch.cuda.max_memory_allocated after resetting peak stats.",
                "TTFT and TPOT are not measured by this benchmark.",
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nResults saved to: {output_path}")

    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
