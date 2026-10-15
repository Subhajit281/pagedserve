import torch
from pagedserve.model.engine import PagedServeEngine

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 8


def main():
    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=32,
        block_size=16,
    )

    request_id = "generation-comparison"

    try:
        input_ids = engine.tokenizer(
            PROMPT,
            return_tensors="pt",
        )["input_ids"].to(engine.device)

        prompt_length = input_ids.shape[1]
        engine.cache.allocate(
            request_id,
            prompt_length + MAX_NEW_TOKENS,
        )

        prefix = input_ids.clone()
        paged_input = input_ids
        paged_position = 0

        with torch.inference_mode():
            for step in range(MAX_NEW_TOKENS):
                # Reference: recompute the complete prefix without a cache.
                reference_logits = engine.model(
                    input_ids=prefix,
                    use_cache=False,
                ).logits[:, -1, :]

                # PagedServe: prefill first, then process only the newest token.
                paged_logits = engine._forward_paged(
                    input_ids=paged_input,
                    request_id=request_id,
                    position_start=paged_position,
                )[:, -1, :]

                max_diff = (
                    reference_logits.float() - paged_logits.float()
                ).abs().max().item()

                reference_token = reference_logits.argmax(
                    dim=-1, keepdim=True
                )
                paged_token = paged_logits.argmax(
                    dim=-1, keepdim=True
                )

                print(
                    f"Step {step + 1}: "
                    f"max_logit_diff={max_diff:.6f}, "
                    f"HF={reference_token.item()}, "
                    f"PagedServe={paged_token.item()}"
                )

                if reference_token.item() != paged_token.item():
                    print("FIRST TOKEN MISMATCH")
                    break

                if (
                    engine.tokenizer.eos_token_id is not None
                    and reference_token.item()
                    == engine.tokenizer.eos_token_id
                ):
                    break

                prefix = torch.cat(
                    [prefix, reference_token],
                    dim=1,
                )

                paged_input = reference_token
                paged_position = prefix.shape[1] - 1

    finally:
        if request_id in engine.cache._block_tables:
            engine.cache.free(request_id)

        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
