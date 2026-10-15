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


def sync():
    torch.cuda.synchronize()


def completion_tokens(sequence, eos_token_ids):
    """Return completion IDs through the first EOS token, inclusive."""
    ids = sequence.tolist()
    if isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]

    for i, token_id in enumerate(ids):
        if token_id in eos_token_ids:
            return ids[: i + 1]
    return ids


def percentile(values, q):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser(
        description="S1: Hugging Face static-batch generation benchmark"
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--output",
        default="bench/results/hf_static_batch_20261015.json",
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
    print("Loading standalone Hugging Face model for S1...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # Decoder-only batched generation must use left padding.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_start = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device).eval()
    sync()
    model_load_seconds = time.perf_counter() - load_start

    prompts = list(PROMPTS.values())
    names = list(PROMPTS.keys())
    results = {}

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Model load: {model_load_seconds:.3f}s")
    print(f"Warmups: {args.warmups}; measured runs: {args.repeats}")
    print(f"Maximum new tokens/request: {args.max_new_tokens}")

    try:
        # Individual reference outputs are used only for correctness checks,
        # never included in the timed batch measurements.
        reference_ids = []
        prompt_token_counts = []

        with torch.inference_mode():
            for prompt in prompts:
                item = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                )
                input_ids = item["input_ids"].to(device)
                prompt_token_counts.append(int(input_ids.shape[1]))

                ref = model.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    num_return_sequences=1,
                    pad_token_id=tokenizer.pad_token_id,
                )
                eos_ids = model.generation_config.eos_token_id
                completion = completion_tokens(
                    ref[0, input_ids.shape[1]:], eos_ids
                )
                reference_ids.append(completion)
                del ref

            sync()

            for batch_size in (1, 2, 3):
                batch_prompts = prompts[:batch_size]
                batch_names = names[:batch_size]

                encoded = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    add_special_tokens=True,
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)
                padded_prompt_length = int(input_ids.shape[1])

                print(
                    f"\nBatch size {batch_size}: "
                    f"prompt lengths={prompt_token_counts[:batch_size]}, "
                    f"padded width={padded_prompt_length}"
                )

                kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": args.max_new_tokens,
                    "do_sample": False,
                    "use_cache": True,
                    "num_return_sequences": 1,
                    "pad_token_id": tokenizer.pad_token_id,
                }

                for _ in range(args.warmups):
                    warmup_output = model.generate(**kwargs)
                    del warmup_output
                sync()

                latencies = []
                peak_memory_mib = 0.0
                final_completions = None
                parity_by_run = []

                for _ in range(args.repeats):
                    gc.collect()
                    torch.cuda.empty_cache()
                    sync()
                    torch.cuda.reset_peak_memory_stats(device)

                    start = time.perf_counter()
                    output = model.generate(**kwargs)
                    sync()
                    elapsed = time.perf_counter() - start

                    completions = [
                        completion_tokens(
                            output[row, padded_prompt_length:],
                            model.generation_config.eos_token_id,
                        )
                        for row in range(batch_size)
                    ]

                    latencies.append(elapsed)
                    peak_memory_mib = max(
                        peak_memory_mib,
                        torch.cuda.max_memory_allocated(device) / (1024 ** 2),
                    )
                    final_completions = completions
                    parity_by_run.append([
                        completion == reference_ids[i]
                        for i, completion in enumerate(completions)
                    ])
                    del output

                # Every request must match its standalone greedy reference.
                # Compare batched greedy output with standalone references.
                # Log differences instead of aborting the benchmark.
                parity = []
                common_prefix_lengths = []
                for i, completion in enumerate(final_completions):
                    reference = reference_ids[i]
                    matches = completion == reference
                    parity.append(matches)
                    prefix_length = 0
                    for actual_id, reference_id in zip(completion, reference):
                        if actual_id != reference_id:
                            break
                        prefix_length += 1
                    common_prefix_lengths.append(prefix_length)
                    if not matches:
                        print(
                            f"  WARNING: {batch_names[i]} differs from its standalone reference; "
                            f"common prefix={prefix_length}/{min(len(completion), len(reference))} tokens, "
                            f"batched length={len(completion)}, reference length={len(reference)}"
                        )
                counts = [len(x) for x in final_completions]
                mean_latency = statistics.mean(latencies)
                mean_total_tokens = statistics.mean(
                    [sum(len(x) for x in final_completions)]
                    * len(latencies)
                )
                aggregate_throughput = mean_total_tokens / mean_latency

                entry = {
                    "batch_size": batch_size,
                    "prompt_names": batch_names,
                    "prompt_tokens": prompt_token_counts[:batch_size],
                    "padded_prompt_width": padded_prompt_length,
                    "requested_new_tokens_per_request": args.max_new_tokens,
                    "generated_tokens_per_request": counts,
                    "aggregate_generated_tokens": sum(counts),
                    "token_parity_with_individual_generation": parity,
                    "token_parity_all_repetitions": parity_by_run,
                    "all_repetitions_match_reference": all(
                        all(run_parity) for run_parity in parity_by_run
                    ),
                    "common_prefix_tokens_vs_individual": common_prefix_lengths,
                    "latency_seconds_runs": latencies,
                    "batch_latency_seconds_mean": mean_latency,
                    "batch_latency_seconds_p50": statistics.median(latencies),
                    "batch_latency_seconds_p95": percentile(latencies, 0.95),
                    "aggregate_throughput_tokens_per_second": aggregate_throughput,
                    "latency_per_request_seconds": mean_latency / batch_size,
                    "peak_gpu_memory_mib": peak_memory_mib,
                }
                results[str(batch_size)] = entry

                print(f"  Generated tokens/request: {counts}")
                print(f"  Token parity:             {parity}")
                print(f"  Mean batch latency:       {mean_latency * 1000:.2f} ms")
                print(f"  Aggregate throughput:     {aggregate_throughput:.2f} tokens/s")
                print(f"  Latency/request:          {mean_latency / batch_size * 1000:.2f} ms")
                print(f"  Peak GPU memory:          {peak_memory_mib:.2f} MiB")

        payload = {
            "benchmark": "S1_huggingface_static_batch",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "device": torch.cuda.get_device_name(device),
            "dtype": "torch.bfloat16",
            "attention_implementation": "eager",
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "max_new_tokens_per_request": args.max_new_tokens,
            "padding_side": tokenizer.padding_side,
            "generation": {
                "do_sample": False,
                "use_cache": True,
                "num_return_sequences": 1,
            },
            "model_load_seconds": model_load_seconds,
            "benchmarks": results,
            "notes": [
                "Hugging Face model.generate with left-padded static batches.",
                "Batch latency includes prefill and completion generation for the whole batch.",
                "Aggregate throughput counts completion tokens across all requests.",
                "Token IDs are checked against individual greedy generation.",
                "Peak memory is torch.cuda.max_memory_allocated after resetting peak stats.",
                "This benchmark uses a fixed batch; it does not implement continuous batching.",
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nResults saved to: {output_path}")
        all_parity_passed = all(
            item.get("all_repetitions_match_reference", False)
            for item in results.values()
        )
        if all_parity_passed:
            print("PASS: All static-batch token sequences match standalone references.")
        else:
            print("COMPLETED WITH TOKEN DIFFERENCES: do not treat S1 as correctness-passing.")

    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
