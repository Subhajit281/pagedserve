import gc

import torch

from pagedserve.model.engine import PagedServeEngine


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 12

PROMPTS = [
    "The capital of France is",
    "A CPU is responsible for",
    "The result of 15 multiplied by 12 is",
]


def reference_generation(engine, prompt):
    encoded = engine.tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = encoded["input_ids"].to(engine.device)

    with torch.inference_mode():
        output = engine.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            use_cache=True,
            pad_token_id=(
                engine.tokenizer.pad_token_id
                if engine.tokenizer.pad_token_id is not None
                else engine.tokenizer.eos_token_id
            ),
        )

    return output[0, input_ids.shape[1]:].tolist()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=128,
        block_size=16,
    )

    try:
        for index, prompt in enumerate(PROMPTS, start=1):
            reference_ids = reference_generation(engine, prompt)

            # Temporarily observe the internal generated token IDs.
            # No production source files are modified.
            captured = {}
            original_decode = engine.tokenizer.decode

            def capture_decode(token_ids, *args, **kwargs):
                captured["decoded_ids"] = (
                    token_ids.detach().cpu().tolist()
                    if isinstance(token_ids, torch.Tensor)
                    else list(token_ids)
                )
                return original_decode(token_ids, *args, **kwargs)

            engine.tokenizer.decode = capture_decode
            try:
                paged_text = engine.generate(
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                )
            finally:
                engine.tokenizer.decode = original_decode

            full_ids = captured.get("decoded_ids", [])
            prompt_ids = engine.tokenizer(
                prompt,
                add_special_tokens=True,
            )["input_ids"]

            paged_ids = full_ids[len(prompt_ids):]

            print(f"\n{'=' * 60}")
            print(f"Request {index}: {prompt}")
            print("Reference IDs:", reference_ids)
            print("PagedServe IDs:", paged_ids)

            mismatch = next(
                (
                    i for i, (a, b) in enumerate(
                        zip(reference_ids, paged_ids)
                    )
                    if a != b
                ),
                None,
            )

            if mismatch is None and len(reference_ids) != len(paged_ids):
                mismatch = min(len(reference_ids), len(paged_ids))

            if mismatch is None:
                print("Result: EXACT TOKEN MATCH")
            else:
                print("First mismatch index:", mismatch)
                print(
                    "Reference token:",
                    reference_ids[mismatch]
                    if mismatch < len(reference_ids)
                    else "<END>",
                )
                print(
                    "PagedServe token:",
                    paged_ids[mismatch]
                    if mismatch < len(paged_ids)
                    else "<END>",
                )

            print("PagedServe output:", repr(paged_text))
            print("Cache blocks remaining:", engine.cache.used_blocks)

            assert engine.cache.used_blocks == 0

        print("\nPASS: All KV-cache allocations released.")

    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
