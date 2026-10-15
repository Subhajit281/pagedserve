import gc
import torch

from pagedserve.model.engine import PagedServeEngine

PROMPTS = [
    "The capital of France is",
    "A CPU is responsible for",
    "The result of 15 multiplied by 12 is",
    "Explain what an operating system does.",
]

TARGET_POSITION = 15
MAX_NEW_TOKENS = 16


def main():
    engine = PagedServeEngine(
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        device="cuda:0",
        dtype=torch.bfloat16,
        num_blocks=128,
        block_size=16,
    )

    original_single = engine._forward_paged
    original_batch = engine._forward_paged_batch

    captured = {"sequential": None, "batched": None}

    def capture_single(*args, **kwargs):
        logits = original_single(*args, **kwargs)

        position = kwargs.get("position_start")
        input_ids = kwargs.get("input_ids")

        if (
            position == TARGET_POSITION
            and input_ids is not None
            and input_ids.shape[1] == 1
            and kwargs.get("request_id") is not None
        ):
            captured["sequential"] = {
                "logits": logits[:, -1, :].detach().float().cpu(),
                "input_id": input_ids[0, 0].item(),
            }

        return logits

    def capture_batch(*args, **kwargs):
        logits = original_batch(*args, **kwargs)

        positions = kwargs.get("positions")
        input_ids = kwargs.get("input_ids")

        if positions is not None and TARGET_POSITION in positions:
            row = positions.index(TARGET_POSITION)

            if input_ids is not None:
                captured["batched"] = {
                    "logits": logits[row, -1, :].detach().float().cpu(),
                    "input_id": input_ids[row, 0].item(),
                }

        return logits

    engine._forward_paged = capture_single
    engine._forward_paged_batch = capture_batch

    try:
        print("Running sequential generation...")
        engine.generate(PROMPTS[0], max_new_tokens=MAX_NEW_TOKENS)

        assert engine.cache.used_blocks == 0

        print("Running four-request batched generation...")
        engine.generate_many(PROMPTS, max_new_tokens=MAX_NEW_TOKENS)

        assert engine.cache.used_blocks == 0

        seq = captured["sequential"]
        bat = captured["batched"]

        if seq is None or bat is None:
            print("\nCould not capture both logits at the target position.")
            print("Sequential captured:", seq is not None)
            print("Batched captured:", bat is not None)
            return

        print("\n===== DIRECT LOGIT COMPARISON =====")
        print("Target position:", TARGET_POSITION)
        print("Sequential input token:", seq["input_id"])
        print("Batched input token:   ", bat["input_id"])

        a = seq["logits"].squeeze(0)
        b = bat["logits"]

        print("Sequential argmax:", a.argmax().item())
        print("Batched argmax:   ", b.argmax().item())
        print("Maximum absolute difference:", (a - b).abs().max().item())
        print("Mean absolute difference:   ", (a - b).abs().mean().item())

        for label, values in [
            ("SEQUENTIAL", a),
            ("BATCHED", b),
        ]:
            top = torch.topk(values, 5)
            print(f"\n{label} top 5:")
            for token, score in zip(top.indices.tolist(), top.values.tolist()):
                text = engine.tokenizer.decode([token])
                print(f"  ID={token:<7} logit={score:.8f} token={text!r}")

        difference = (a - b).abs()
        print("\nArgmax predictions match:", a.argmax().item() == b.argmax().item())
        print("Cache blocks remaining:", engine.cache.used_blocks)

    finally:
        engine._forward_paged = original_single
        engine._forward_paged_batch = original_batch
        del engine
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
