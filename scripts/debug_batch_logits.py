import gc
import torch

from pagedserve.model.engine import PagedServeEngine

PROMPTS = [
    "The capital of France is",
    "A CPU is responsible for",
    "The result of 15 multiplied by 12 is",
    "Explain what an operating system does.",
]

MAX_NEW_TOKENS = 16


def main():
    engine = PagedServeEngine(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=128,
        block_size=16,
    )

    try:
        # Compare outputs under the actual four-request workload.
        sequential = [
            engine.generate(p, max_new_tokens=MAX_NEW_TOKENS)
            for p in PROMPTS
        ]
        batched = engine.generate_many(
            PROMPTS, max_new_tokens=MAX_NEW_TOKENS
        )

        print("Sequential output:", repr(sequential[0]))
        print("Batched output:   ", repr(batched[0]))
        print("Exact match:", sequential[0] == batched[0])

        if sequential[0] != batched[0]:
            print(
                "\nThe first request diverges only in the four-request "
                "workload. The next step is to instrument per-step logits "
                "inside the sequential and batched decode loops."
            )
        else:
            print("\nNo mismatch reproduced in this run.")

        print("Cache blocks remaining:", engine.cache.used_blocks)
        assert engine.cache.used_blocks == 0

    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
