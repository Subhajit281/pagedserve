import torch
from transformers import DynamicCache

from pagedserve.model.engine import PagedServeEngine

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "The capital of France is"


def main():
    engine = PagedServeEngine(
        model_id=MODEL_ID,
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=32,
        block_size=16,
    )

    request_id = "cache-debug"

    try:
        input_ids = engine.tokenizer(
            PROMPT, return_tensors="pt"
        )["input_ids"].to(engine.device)

        prompt_length = input_ids.shape[1]

        engine.cache.allocate(request_id, prompt_length + 8)

        with torch.inference_mode():
            # Reference prefill and KV cache.
            reference_prefill = engine.model(
                input_ids=input_ids,
                use_cache=True,
            )

            # PagedServe prefill.
            engine._forward_paged(
                input_ids=input_ids,
                request_id=request_id,
                position_start=0,
            )

            print("\nPREFILL CACHE COMPARISON")

            for layer_idx in (0, 12, 23):
                ref_key = reference_prefill.past_key_values.layers[
                    layer_idx
                ].keys[0]
                ref_value = reference_prefill.past_key_values.layers[
                    layer_idx
                ].values[0]

                for position in range(prompt_length):
                    key, value = engine.cache.read_token(
                        request_id, layer_idx, position
                    )

                    key_diff = (
                        key - ref_key[:, position, :]
                    ).abs().max().item()

                    value_diff = (
                        value - ref_value[:, position, :]
                    ).abs().max().item()

                    if key_diff or value_diff:
                        print(
                            f"Layer {layer_idx}, position {position}: "
                            f"K diff={key_diff}, V diff={value_diff}"
                        )

            print("Reference cache layout:", ref_key.shape)

            # First generated token.
            first_token = reference_prefill.logits[:, -1, :].argmax(
                dim=-1, keepdim=True
            )

            next_position = prompt_length

            # Reference decode with the existing HF cache.
            reference_decode = engine.model(
                input_ids=first_token,
                past_key_values=reference_prefill.past_key_values,
                use_cache=True,
                cache_position=torch.tensor(
                    [next_position], device=engine.device
                ),
            )

            # PagedServe decode.
            paged_decode = engine._forward_paged(
                input_ids=first_token,
                request_id=request_id,
                position_start=next_position,
            )

            print("\nDECODE COMPARISON")
            print(
                "Maximum logit difference:",
                (
                    reference_decode.logits[:, -1, :].float()
                    - paged_decode[:, -1, :].float()
                ).abs().max().item(),
            )

    finally:
        if request_id in engine.cache._block_tables:
            engine.cache.free(request_id)

        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
