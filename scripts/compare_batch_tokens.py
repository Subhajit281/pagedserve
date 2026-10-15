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
        sequential = [
            engine.generate(p, MAX_NEW_TOKENS) for p in PROMPTS
        ]
        batched = engine.generate_many(PROMPTS, MAX_NEW_TOKENS)

        prompt_ids = [
            engine.tokenizer(p, add_special_tokens=True)["input_ids"]
            for p in PROMPTS
        ]

        for i, (seq_text, batch_text) in enumerate(
            zip(sequential, batched)
        ):
            # Recover token IDs from the output text is not reliable
            # because decoding can normalize whitespace. Report text here.
            print(f"\nRequest {i + 1}: {PROMPTS[i]}")
            print("Sequential:", repr(seq_text))
            print("Batched:   ", repr(batch_text))
            print("Exact text match:", seq_text == batch_text)

        print("\nThis test identifies the affected request; token-level")
        print("logit instrumentation is required to isolate the cause.")
        print("KV-cache blocks remaining:", engine.cache.used_blocks)
        assert engine.cache.used_blocks == 0

    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
