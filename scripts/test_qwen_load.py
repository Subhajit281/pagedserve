import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    device = "cuda:0"
    dtype = torch.bfloat16

    print(f"Loading {MODEL_ID}...")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Target dtype: {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    print("\n--- Model loaded ---")
    print(f"Model type: {model.config.model_type}")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"KV heads: {model.config.num_key_value_heads}")
    print(f"Head dimension: {model.config.hidden_size // model.config.num_attention_heads}")
    print(f"Parameter dtype: {next(model.parameters()).dtype}")
    print(f"Parameter device: {next(model.parameters()).device}")

    allocated_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)

    print(f"GPU memory allocated: {allocated_mb:.2f} MiB")
    print(f"GPU memory reserved: {reserved_mb:.2f} MiB")

    messages = [
        {
            "role": "user",
            "content": "Explain paged KV caching in two simple sentences.",
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    print("\nGenerating a short response...")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            use_cache=True,
        )

    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    )

    print("\n--- Generated response ---")
    print(response)

    torch.cuda.synchronize()

    print("\nReference inference passed.")


if __name__ == "__main__":
    main()
