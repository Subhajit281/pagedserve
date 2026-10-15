import torch
from transformers import AutoConfig

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def main():
    print(f"Model: {MODEL_ID}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {torch.cuda.is_bf16_supported()}")

    print("\nLoading model configuration...")

    config = AutoConfig.from_pretrained(MODEL_ID)

    print("\n--- Actual Model Configuration ---")
    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Number of layers: {config.num_hidden_layers}")
    print(f"Attention heads: {config.num_attention_heads}")
    print(f"KV heads: {config.num_key_value_heads}")
    head_dim = getattr(
        config,
        "head_dim",
        config.hidden_size // config.num_attention_heads,
    )

    print(f"Head dimension: {head_dim}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Maximum positions: {config.max_position_embeddings}")

    print("\nConfiguration loaded successfully.")


if __name__ == "__main__":
    main()
