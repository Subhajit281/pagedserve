import gc

import torch
from transformers import AutoModelForCausalLM

from pagedserve.model.engine import PagedServeEngine


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 16

PROMPTS = [
    "The capital of France is",
    "A CPU is responsible for",
    "The result of 15 multiplied by 12 is",
]


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this validation.")

    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=128,
        block_size=16,
    )

    reference = None

    try:
        print("Loading the Hugging Face reference model...")

        # Reuse the exact model weights already loaded by PagedServe.
        # This avoids downloading/loading a second copy of the model.
        reference = engine.model

        for index, prompt in enumerate(PROMPTS, start=1):
            print(f"\n{'=' * 60}")
            print(f"Request {index}: {prompt}")

            encoded = engine.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = encoded["input_ids"].to(engine.device)

            # Hugging Face reference generation.
            with torch.inference_mode():
                reference_ids = reference.generate(
                    input_ids=input_ids,
                    attention_mask=torch.ones_like(input_ids),
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=engine.tokenizer.pad_token_id
                    if engine.tokenizer.pad_token_id is not None
                    else engine.tokenizer.eos_token_id,
                )

            reference_new_ids = reference_ids[0, input_ids.shape[1]:].tolist()

            # PagedServe generation.
            # The public API returns decoded text, so this initial
            # comparison checks output text rather than exact token IDs.
            paged_text = engine.generate(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            reference_text = engine.tokenizer.decode(
                reference_ids[0],
                skip_special_tokens=True,
            )

            print("Reference output:", repr(reference_text))
            print("PagedServe output:", repr(paged_text))
            print("Reference generated token IDs:", reference_new_ids)
            print(
                "Text matches:",
                reference_text == paged_text,
            )

        print("\nKV-cache blocks remaining:", engine.cache.used_blocks)

        if engine.cache.used_blocks != 0:
            raise AssertionError("KV-cache allocations leaked.")

        print("PASS: KV-cache allocations were released.")

    finally:
        del reference
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
