import gc

import torch

from pagedserve.model.engine import PagedServeEngine


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The result of 15 multiplied by 12 is"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")

    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=32,
        block_size=16,
    )

    request_id = "prefill-debug"

    try:
        encoded = engine.tokenizer(
            PROMPT,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(engine.device)

        print("Prompt:", PROMPT)
        print("Input token IDs:", input_ids[0].tolist())

        # Reference prefill: standard Hugging Face model.
        with torch.inference_mode():
            reference_logits = engine.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=False,
            ).logits[:, -1, :].float()

        # PagedServe prefill: custom attention and paged KV cache.
        engine.cache.allocate(request_id, input_ids.shape[1] + 2)

        try:
            with torch.inference_mode():
                paged_logits = engine._forward_paged(
                    input_ids=input_ids,
                    request_id=request_id,
                    position_start=0,
                )[:, -1, :].float()

            difference = (reference_logits - paged_logits).abs()

            ref_top = torch.topk(reference_logits.squeeze(0), k=10)
            paged_top = torch.topk(paged_logits.squeeze(0), k=10)

            print("\n===== LOGIT COMPARISON =====")
            print("Reference argmax:", reference_logits.argmax(-1).item())
            print("PagedServe argmax:", paged_logits.argmax(-1).item())
            print("Exact argmax match:",
                  reference_logits.argmax(-1).item()
                  == paged_logits.argmax(-1).item())
            print("Maximum absolute logit difference:",
                  difference.max().item())
            print("Mean absolute logit difference:",
                  difference.mean().item())

            print("\n===== REFERENCE TOP 10 =====")
            for token_id, score in zip(
                ref_top.indices.tolist(),
                ref_top.values.tolist(),
            ):
                token = engine.tokenizer.decode([token_id])
                print(f"{token_id:6d} {score:12.5f} {token!r}")

            print("\n===== PAGEDSERVE TOP 10 =====")
            for token_id, score in zip(
                paged_top.indices.tolist(),
                paged_top.values.tolist(),
            ):
                token = engine.tokenizer.decode([token_id])
                print(f"{token_id:6d} {score:12.5f} {token!r}")

        finally:
            engine.cache.free(request_id)

        print("\nKV-cache blocks remaining:", engine.cache.used_blocks)

    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
