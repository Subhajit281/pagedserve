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


def capture(engine, prompts, batched):
    captured = []
    original_decode = engine.tokenizer.decode

    def wrapped_decode(token_ids, *args, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.detach().cpu().flatten().tolist()
        else:
            ids = list(token_ids)
        captured.append(ids)
        return original_decode(token_ids, *args, **kwargs)

    engine.tokenizer.decode = wrapped_decode
    try:
        if batched:
            outputs = engine.generate_many(
                prompts, max_new_tokens=MAX_NEW_TOKENS
            )
        else:
            outputs = [
                engine.generate(p, max_new_tokens=MAX_NEW_TOKENS)
                for p in prompts
            ]
    finally:
        engine.tokenizer.decode = original_decode

    return outputs, captured


def main():
    engine = PagedServeEngine(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=128,
        block_size=16,
    )

    try:
        for label, prompts in [
            ("SINGLE REQUEST", [PROMPTS[0]]),
            ("FOUR REQUESTS", PROMPTS),
        ]:
            print(f"\n===== {label} =====")

            sequential_text, sequential_ids = capture(
                engine, prompts, False
            )
            batched_text, batched_ids = capture(
                engine, prompts, True
            )

            for i, prompt in enumerate(prompts):
                input_ids = engine.tokenizer(
                    prompt, add_special_tokens=True
                )["input_ids"]

                seq = sequential_ids[i][len(input_ids):]
                batch = batched_ids[i][len(input_ids):]

                mismatch = next(
                    (j for j, (a, b) in enumerate(zip(seq, batch))
                     if a != b),
                    None,
                )

                print(f"\nPrompt {i + 1}: {prompt}")
                print("Token sequences equal:", seq == batch)
                print("Sequential IDs:", seq)
                print("Batched IDs:   ", batch)

                if mismatch is not None:
                    print("First mismatch index:", mismatch)
                    print("Sequential token:", seq[mismatch])
                    print("Batched token:", batch[mismatch])
                elif len(seq) != len(batch):
                    print("Different sequence lengths:", len(seq), len(batch))

            print("Cache blocks remaining:", engine.cache.used_blocks)
            assert engine.cache.used_blocks == 0

    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
