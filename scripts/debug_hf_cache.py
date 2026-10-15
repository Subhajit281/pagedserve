import torch
from transformers import DynamicCache
from pagedserve.model.engine import PagedServeEngine

PROMPT = "The capital of France is"
STEPS = 6

engine = PagedServeEngine()
model = engine.model
tokenizer = engine.tokenizer

input_ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"].to(
    engine.device
)

past_key_values = DynamicCache(config=model.config)
prefix = input_ids.clone()

with torch.inference_mode():
    for step in range(STEPS):
        if step == 0:
            current_input = input_ids
            cache_position = torch.arange(
                input_ids.shape[1], device=input_ids.device
            )
        else:
            current_input = prefix[:, -1:]
            cache_position = torch.tensor(
                [prefix.shape[1] - 1], device=input_ids.device
            )

        cached_output = model(
            input_ids=current_input,
            past_key_values=past_key_values,
            cache_position=cache_position,
            use_cache=True,
        )

        cached_logits = cached_output.logits[:, -1, :].float()

        uncached_output = model(
            input_ids=prefix,
            use_cache=False,
        )
        uncached_logits = uncached_output.logits[:, -1, :].float()

        diff = (cached_logits - uncached_logits).abs()
        cached_token = cached_logits.argmax(dim=-1).item()
        uncached_token = uncached_logits.argmax(dim=-1).item()

        print(
            f"Step {step + 1}: "
            f"max_diff={diff.max().item():.6f}, "
            f"mean_diff={diff.mean().item():.6f}, "
            f"cached_token={cached_token}, "
            f"uncached_token={uncached_token}"
        )

        if cached_token != uncached_token:
            print("FIRST TOKEN DIVERGENCE")
            break

        next_token = torch.tensor(
            [[uncached_token]], device=input_ids.device
        )
        prefix = torch.cat([prefix, next_token], dim=1)

print("Prefix:", tokenizer.decode(prefix[0]))
