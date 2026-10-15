import gc

import torch

from pagedserve.model.engine import PagedServeEngine


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The result of 15 multiplied by 12 is"


def compare_top_tokens(engine, reference_logits, paged_logits):
    reference_top = torch.topk(reference_logits.squeeze(), k=10)
    paged_top = torch.topk(paged_logits.squeeze(), k=10)

    print("\nReference top 10:")
    for token_id, score in zip(
        reference_top.indices.tolist(),
        reference_top.values.tolist(),
    ):
        print(
            f"{token_id:6d} {score:12.5f} "
            f"{engine.tokenizer.decode([token_id])!r}"
        )

    print("\nPagedServe top 10:")
    for token_id, score in zip(
        paged_top.indices.tolist(),
        paged_top.values.tolist(),
    ):
        print(
            f"{token_id:6d} {score:12.5f} "
            f"{engine.tokenizer.decode([token_id])!r}"
        )


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

    request_id = "decode-debug"

    try:
        encoded = engine.tokenizer(
            PROMPT,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"].to(engine.device)
        prompt_length = input_ids.shape[1]

        # Reference prefill.
        with torch.inference_mode():
            reference_prefill = engine.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                use_cache=True,
            )

        reference_cache = reference_prefill.past_key_values

        # PagedServe prefill.
        engine.cache.allocate(request_id, prompt_length + 4)

        try:
            with torch.inference_mode():
                paged_prefill = engine._forward_paged(
                    input_ids=input_ids,
                    request_id=request_id,
                    position_start=0,
                )

            reference_first = reference_prefill.logits[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )
            paged_first = paged_prefill[:, -1, :].argmax(
                dim=-1,
                keepdim=True,
            )

            print("Reference first token:", reference_first.item())
            print("PagedServe first token:", paged_first.item())

            # Use the same token for both decode paths.
            shared_token = reference_first

            with torch.inference_mode():
                reference_decode = engine.model(
                    input_ids=shared_token,
                    past_key_values=reference_cache,
                    use_cache=True,
                    cache_position=torch.tensor(
                        [prompt_length],
                        device=engine.device,
                    ),
                )

                paged_decode = engine._forward_paged(
                    input_ids=shared_token,
                    request_id=request_id,
                    position_start=prompt_length,
                )

            reference_logits = reference_decode.logits[:, -1, :].float()
            paged_logits = paged_decode[:, -1, :].float()

            difference = (reference_logits - paged_logits).abs()

            print("\n===== DECODE COMPARISON =====")
            print(
                "Reference next token:",
                reference_logits.argmax(-1).item(),
            )
            print(
                "PagedServe next token:",
                paged_logits.argmax(-1).item(),
            )
            print(
                "Exact argmax match:",
                reference_logits.argmax(-1).item()
                == paged_logits.argmax(-1).item(),
            )
            print("Maximum absolute logit difference:", difference.max().item())
            print("Mean absolute logit difference:", difference.mean().item())

            compare_top_tokens(engine, reference_logits, paged_logits)

        finally:
            engine.cache.free(request_id)

        print("\nKV-cache blocks remaining:", engine.cache.used_blocks)

    finally:
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
