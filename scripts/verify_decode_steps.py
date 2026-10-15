import torch
from pagedserve.model.engine import PagedServeEngine

PROMPT = "The capital of France is"
STEPS = 5


def main():
    engine = PagedServeEngine(
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=32,
        block_size=16,
    )

    request_id = "verify-decode"

    try:
        input_ids = engine.tokenizer(
            PROMPT, return_tensors="pt"
        )["input_ids"].to(engine.device)

        prompt_len = input_ids.shape[1]
        engine.cache.allocate(request_id, prompt_len + STEPS)

        prefix = input_ids.clone()

        with torch.inference_mode():
            # Prefill PagedServe and Hugging Face independently.
            paged_logits = engine._forward_paged(
                input_ids,
                request_id,
                0,
            )

            hf_cache = None

            for step in range(STEPS):
                hf_out = engine.model(
                    input_ids=prefix if step == 0 else prefix[:, -1:],
                    past_key_values=hf_cache,
                    use_cache=True,
                    cache_position=torch.arange(
                        0 if step == 0 else prefix.shape[1] - 1,
                        prefix.shape[1],
                        device=engine.device,
                    ),
                )
                hf_cache = hf_out.past_key_values
                hf_logits = hf_out.logits[:, -1, :]

                if step == 0:
                    # Both sides have processed the prompt.
                    paged_current = paged_logits[:, -1, :]
                else:
                    paged_current = paged_logits[:, -1, :]

                diff = (
                    hf_logits.float() - paged_current.float()
                ).abs().max().item()

                hf_token = hf_logits.argmax(dim=-1, keepdim=True)
                paged_token = paged_current.argmax(dim=-1, keepdim=True)

                print(
                    f"Step {step + 1}: "
                    f"max_logit_diff={diff:.6f}, "
                    f"HF={hf_token.item()}, "
                    f"PagedServe={paged_token.item()}"
                )

                if hf_token.item() != paged_token.item():
                    print("FIRST DIVERGENCE")
                    break

                prefix = torch.cat([prefix, hf_token], dim=1)

                if step + 1 < STEPS:
                    paged_logits = engine._forward_paged(
                        hf_token,
                        request_id,
                        prefix.shape[1] - 1,
                    )

    finally:
        if request_id in engine.cache._block_tables:
            engine.cache.free(request_id)
        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
