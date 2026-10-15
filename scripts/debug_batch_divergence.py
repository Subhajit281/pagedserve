import gc
import torch
from pagedserve.model.engine import PagedServeEngine

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 16


def main():
    engine = PagedServeEngine(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=128,
        block_size=16,
    )

    original_decode = engine.tokenizer.decode
    captured = []

    def capture_decode(token_ids, *args, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.detach().cpu().flatten().tolist()
        else:
            ids = list(token_ids)
        captured.append(ids)
        return original_decode(token_ids, *args, **kwargs)

    try:
        engine.tokenizer.decode = capture_decode

        sequential_text = engine.generate(
            PROMPT, max_new_tokens=MAX_NEW_TOKENS
        )
        sequential_ids = captured[-1]

        batched_text = engine.generate_many(
            [PROMPT], max_new_tokens=MAX_NEW_TOKENS
        )[0]
        batched_ids = captured[-1]

        prompt_ids = engine.tokenizer(
            PROMPT, add_special_tokens=True
        )["input_ids"]

        sequential_new = sequential_ids[len(prompt_ids):]
        batched_new = batched_ids[len(prompt_ids):]

        print("Sequential generated IDs:", sequential_new)
        print("Batched generated IDs:   ", batched_new)

        limit = min(len(sequential_new), len(batched_new))
        mismatch = next(
            (i for i in range(limit)
             if sequential_new[i] != batched_new[i]),
            None,
        )

        if mismatch is None and len(sequential_new) != len(batched_new):
            mismatch = limit

        if mismatch is None:
            print("\nEXACT TOKEN MATCH")
        else:
            print(f"\nFirst mismatch at generated token index: {mismatch}")
            for i in range(max(0, mismatch - 2),
                           min(max(len(sequential_new), len(batched_new)),
                               mismatch + 3)):
                a = sequential_new[i] if i < len(sequential_new) else None
                b = batched_new[i] if i < len(batched_new) else None
                at = engine.tokenizer.decode([a]) if a is not None else "<END>"
                bt = engine.tokenizer.decode([b]) if b is not None else "<END>"
                print(f"  [{i}] sequential={a} {at!r} | batched={b} {bt!r}")

        print("\nSequential:", repr(sequential_text))
        print("Batched:   ", repr(batched_text))
        print("Cache blocks remaining:", engine.cache.used_blocks)
        assert engine.cache.used_blocks == 0

    finally:
        engine.tokenizer.decode = original_decode
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
