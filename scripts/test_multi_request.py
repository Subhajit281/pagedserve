import torch

from pagedserve.model.engine import PagedServeEngine


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke test.")

    engine = PagedServeEngine(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=128,
        block_size=16,
    )

    try:
        prompts = [
            "The capital of France is",
            "A CPU is responsible for",
            "The result of 15 multiplied by 12 is",
        ]

        print("Running multiple requests...")

        results = engine.generate_many(
            prompts,
            max_new_tokens=12,
        )

        assert len(results) == len(prompts)
        assert all(isinstance(result, str) for result in results)
        assert all(result.strip() for result in results)

        for index, (prompt, result) in enumerate(
            zip(prompts, results),
            start=1,
        ):
            print(f"\nRequest {index}")
            print("Prompt:", prompt)
            print("Output:", result)

        assert engine.cache.used_blocks == 0, (
            "KV-cache blocks leaked after generation."
        )

        print("\nPASS: All requests completed.")
        print("PASS: KV-cache allocations were released.")

    finally:
        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
