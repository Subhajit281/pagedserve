import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from pagedserve.model.engine import PagedServeEngine


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


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile(values, fraction):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--prompt",
        choices=["short", "medium", "long", "all"],
        default="all",
    )
    parser.add_argument("--output", default="bench/results/baseline.json")
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be at least 1")

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    device = torch.device("cuda:0")

    print("Loading PagedServeEngine...")
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

    selected_prompts = (
        PROMPTS
        if args.prompt == "all"
        else {args.prompt: PROMPTS[args.prompt]}
    )

    results = {
        "model": MODEL_ID,
        "device": torch.cuda.get_device_name(device),
        "dtype": str(engine.dtype),
        "block_size": engine.block_size,
        "num_blocks": engine.cache.num_blocks,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "model_load_seconds": load_seconds,
        "benchmarks": {},
    }

    try:
        print(f"Model loaded in {load_seconds:.3f} seconds")
        print(f"GPU: {results['device']}")
        print()

        for prompt_name, prompt in selected_prompts.items():
            encoded = engine.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(device)
            prompt_length = input_ids.shape[1]

            print(
                f"Benchmarking {prompt_name}: "
                f"{prompt_length} prompt tokens, "
                f"{args.max_new_tokens} requested new tokens"
            )

            # Warm-up: exclude initial CUDA/kernel overhead.
            warmup_id = f"benchmark-warmup-{prompt_name}"
            warmup_capacity = prompt_length + min(
                args.max_new_tokens, 4
            )
            engine.cache.allocate(warmup_id, warmup_capacity)

            try:
                with torch.inference_mode():
                    engine._forward_paged(
                        input_ids=input_ids,
                        request_id=warmup_id,
                        position_start=0,
                    )
            finally:
                engine.cache.free(warmup_id)

            synchronize()
            torch.cuda.reset_peak_memory_stats(device)

            prefill_times = []
            ttft_times = []
            tpot_times = []
            total_times = []
            generated_counts = []

            for repeat in range(args.repeats):
                request_id = (
                    f"benchmark-{prompt_name}-{repeat}"
                )
                capacity = prompt_length + args.max_new_tokens
                engine.cache.allocate(request_id, capacity)

                try:
                    with torch.inference_mode():
                        synchronize()
                        total_start = time.perf_counter()

                        prefill_start = time.perf_counter()
                        logits = engine._forward_paged(
                            input_ids=input_ids,
                            request_id=request_id,
                            position_start=0,
                        )
                        synchronize()
                        prefill_end = time.perf_counter()

                        next_token = logits[:, -1, :].argmax(
                            dim=-1, keepdim=True
                        )

                        # The first token is available immediately
                        # after prefill and greedy token selection.
                        synchronize()
                        first_token_time = time.perf_counter()

                        generated_count = 1
                        decode_step_times = []

                        eos_id = engine.model.generation_config.eos_token_id

                        for step in range(1, args.max_new_tokens):
                            if (
                                eos_id is not None
                                and next_token.item() == eos_id
                            ):
                                break

                            decode_start = time.perf_counter()

                            logits = engine._forward_paged(
                                input_ids=next_token,
                                request_id=request_id,
                                position_start=prompt_length + step - 1,
                            )

                            next_token = logits[:, -1, :].argmax(
                                dim=-1, keepdim=True
                            )

                            synchronize()
                            decode_end = time.perf_counter()

                            decode_step_times.append(
                                decode_end - decode_start
                            )
                            generated_count += 1

                        total_end = time.perf_counter()

                    prefill_times.append(prefill_end - prefill_start)
                    ttft_times.append(
                        first_token_time - total_start
                    )
                    total_times.append(total_end - total_start)
                    generated_counts.append(generated_count)

                    if decode_step_times:
                        tpot_times.append(
                            statistics.mean(decode_step_times)
                        )

                finally:
                    engine.cache.free(request_id)

                if engine.cache.used_blocks != 0:
                    raise RuntimeError(
                        "Cache leak detected after request cleanup."
                    )

            peak_memory_mb = (
                torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            )

            mean_prefill = statistics.mean(prefill_times)
            mean_ttft = statistics.mean(ttft_times)
            mean_total = statistics.mean(total_times)
            mean_generated = statistics.mean(generated_counts)

            # Total elapsed time includes prefill and decode.
            mean_throughput = mean_generated / mean_total

            entry = {
                "prompt_tokens": prompt_length,
                "requested_new_tokens": args.max_new_tokens,
                "generated_tokens_per_repeat": generated_counts,
                "prefill_seconds_mean": mean_prefill,
                "prefill_seconds_p50": statistics.median(prefill_times),
                "ttft_seconds_mean": mean_ttft,
                "ttft_seconds_p50": statistics.median(ttft_times),
                "tpot_seconds_mean": (
                    statistics.mean(tpot_times)
                    if tpot_times else None
                ),
                "total_seconds_mean": mean_total,
                "generation_throughput_tokens_per_second": (
                    mean_throughput
                ),
                "peak_gpu_memory_mb": peak_memory_mb,
                "cache_clean_after_requests": True,
            }

            results["benchmarks"][prompt_name] = entry

            print(f"  Prefill mean:  {mean_prefill * 1000:.2f} ms")
            print(f"  TTFT mean:     {mean_ttft * 1000:.2f} ms")

            if entry["tpot_seconds_mean"] is not None:
                print(
                    "  TPOT mean:     "
                    f"{entry['tpot_seconds_mean'] * 1000:.2f} ms"
                )

            print(f"  Throughput:    {mean_throughput:.2f} tokens/s")
            print(f"  Peak GPU RAM:  {peak_memory_mb:.2f} MiB")
            print()

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

        print(f"Results saved to: {output_path}")

    finally:
        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
