import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    device = "cuda:0"
    dtype = torch.bfloat16

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
    ).to(device)

    model.eval()

    messages = [
        {
            "role": "user",
            "content": "Explain paged KV caching briefly.",
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    print("\nRunning forward pass...")

    with torch.inference_mode():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True,
        )

    cache = outputs.past_key_values

    print("\n--- Cache inspection ---")
    print("Cache type:", type(cache).__name__)

    if hasattr(cache, "get_seq_length"):
        print("Cached sequence length:", cache.get_seq_length())

    if hasattr(cache, "layers"):
        print("Number of cache layers:", len(cache.layers))

        first_layer = cache.layers[0]

        key = getattr(first_layer, "keys", None)
        value = getattr(first_layer, "values", None)

        if key is not None and value is not None:
            print("First-layer key shape:", tuple(key.shape))
            print("First-layer value shape:", tuple(value.shape))
            print("Key dtype:", key.dtype)
            print("Value dtype:", value.dtype)
            print("Key device:", key.device)
            print("Value device:", value.device)
        else:
            print("Could not find keys/values on the first cache layer.")
            print("Layer type:", type(first_layer).__name__)

    elif isinstance(cache, (tuple, list)):
        print("Number of cache layers:", len(cache))

        first_key, first_value = cache[0][:2]

        print("First-layer key shape:", tuple(first_key.shape))
        print("First-layer value shape:", tuple(first_value.shape))
        print("Key dtype:", first_key.dtype)
        print("Value dtype:", first_value.dtype)
        print("Key device:", first_key.device)
        print("Value device:", first_value.device)

    else:
        print("Unknown cache representation.")
        print("Available attributes:", dir(cache))

    print("\nKV inspection completed.")


if __name__ == "__main__":
    main()
