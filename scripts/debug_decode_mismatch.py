import torch

from pagedserve.model.engine import PagedServeEngine

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 8


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=32,
        block_size=16,
    )

    request_id = "decode-debug"

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

        with torch.inference_mode():
            # PagedServe prefill.
            paged_logits = engine._forward_paged(
                input_ids=input_ids,
                request_id=request_id,
                position_start=0,
            )

            for step in range(MAX_NEW_TOKENS):
                # Reference logits for the complete prefix.
                reference_logits = engine.model(
                    input_ids=prefix,
                    use_cache=False,
                ).logits[:, -1, :]

                current_paged_logits = paged_logits[:, -1, :]

                reference_token = reference_logits.argmax(
                    dim=-1, keepdim=True
                )
                paged_token = current_paged_logits.argmax(
                    dim=-1, keepdim=True
                )

                max_diff = (
                    reference_logits.float()
                    - current_paged_logits.float()
                ).abs().max().item()

                mean_diff = (
                    reference_logits.float()
                    - current_paged_logits.float()
                ).abs().mean().item()

                reference_id = reference_token.item()
                paged_id = paged_token.item()

                print(f"\nGeneration step {step + 1}")
                print(
                    "Reference token:",
                    repr(engine.tokenizer.decode([reference_id])),
                    f"(ID {reference_id})",
                )
                print(
                    "PagedServe token:",
                    repr(engine.tokenizer.decode([paged_id])),
                    f"(ID {paged_id})",
                )
                print(f"Maximum logit difference: {max_diff:.6f}")
                print(f"Mean logit difference:    {mean_diff:.6f}")

                if reference_id != paged_id:
                    print("FIRST TOKEN MISMATCH FOUND")
                    break

                if (
                    engine.tokenizer.eos_token_id is not None
                    and reference_id == engine.tokenizer.eos_token_id
                ):
                    print("EOS reached.")
                    break

                # Append the agreed token to the prefix.
                prefix = torch.cat(
                    [prefix, reference_token],
                    dim=1,
                )

                # Process the newly generated token through PagedServe.
                paged_logits = engine._forward_paged(
                    input_ids=reference_token,
                    request_id=request_id,
                    position_start=prefix.shape[1] - 2,
                )

    finally:
        if request_id in engine.cache._block_tables:
            engine.cache.free(request_id)

        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
