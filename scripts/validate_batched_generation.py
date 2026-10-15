import gc

import torch

from pagedserve.model.engine import PagedServeEngine


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 16

PROMPTS = [
    "The capital of France is",
    "A CPU is responsible for",
    "The result of 15 multiplied by 12 is",
    "Explain what an operating system does.",
]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=128,
        block_size=16,
    )

    try:
        print("\nRunning sequential generation...")
        sequential = [
            engine.generate(prompt, max_new_tokens=MAX_NEW_TOKENS)
            for prompt in PROMPTS
        ]

        assert engine.cache.used_blocks == 0, (
            "Cache leak after sequential generation."
        )

        print("Running batched generation...")
        batched = engine.generate_many(
            PROMPTS,
            max_new_tokens=MAX_NEW_TOKENS,
        )

        assert engine.cache.used_blocks == 0, (
            "Cache leak after batched generation."
        )

        matches = 0

        print("\n" + "=" * 65)

        for i, (seq, batch) in enumerate(
            zip(sequential, batched), start=1
        ):
            exact = seq == batch
            matches += int(exact)

            print(f"\nRequest {i}: {PROMPTS[i - 1]}")
            print(f"Exact output match: {'PASS' if exact else 'FAIL'}")

            if not exact:
                print("Sequential:", repr(seq))
                print("Batched:   ", repr(batch))

        print("\n" + "=" * 65)
        print(f"Exact matches: {matches}/{len(PROMPTS)}")
        print(f"Cache blocks remaining: {engine.cache.used_blocks}")

        if matches != len(PROMPTS):
            raise AssertionError(
                "Batched outputs differ from sequential outputs."
            )

        print("\nPASS: Batched generation matches sequential generation.")

    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
