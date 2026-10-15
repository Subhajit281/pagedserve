import json
from pathlib import Path

from transformers import AutoConfig


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
BLOCK_SIZE = 16
DTYPE_BYTES = 2  # BF16


def main():
    print("=" * 60)
    print("        PagedServe Model Configuration")
    print("=" * 60)

    print("\nLoading config:")
    print(MODEL_ID)

    config = AutoConfig.from_pretrained(MODEL_ID)

    # Core architecture parameters
    num_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    num_kv_heads = getattr(
        config,
        "num_key_value_heads",
        num_attention_heads,
    )

    hidden_size = config.hidden_size

    # Prefer explicit head_dim if the config provides it.
    head_dim = getattr(
        config,
        "head_dim",
        hidden_size // num_attention_heads,
    )

    intermediate_size = getattr(
        config,
        "intermediate_size",
        None,
    )

    max_position_embeddings = getattr(
        config,
        "max_position_embeddings",
        None,
    )

    rope_theta = getattr(
        config,
        "rope_theta",
        None,
    )

    torch_dtype = getattr(
        config,
        "torch_dtype",
        None,
    )

    # KV cache calculation
    kv_bytes_per_token = (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * DTYPE_BYTES
    )

    block_bytes = kv_bytes_per_token * BLOCK_SIZE

    print("\n--- Architecture ---")
    print("Model ID             :", MODEL_ID)
    print("Model type           :", config.model_type)
    print("Layers               :", num_layers)
    print("Hidden size          :", hidden_size)
    print("Attention heads      :", num_attention_heads)
    print("KV heads             :", num_kv_heads)
    print("Head dimension       :", head_dim)
    print("Intermediate size    :", intermediate_size)

    print("\n--- Context / RoPE ---")
    print("Max position         :", max_position_embeddings)
    print("RoPE theta            :", rope_theta)

    print("\n--- Dtype ---")
    print("Config torch dtype   :", torch_dtype)
    print("Assumed KV dtype     : BF16")
    print("Bytes per element    :", DTYPE_BYTES)

    print("\n--- KV Cache ---")
    print(
        "KV bytes/token       :",
        f"{kv_bytes_per_token:,}",
        "B",
    )

    print(
        "KV KiB/token         :",
        f"{kv_bytes_per_token / 1024:.2f}",
        "KiB",
    )

    print(
        "Block size           :",
        BLOCK_SIZE,
        "tokens",
    )

    print(
        "Block bytes          :",
        f"{block_bytes:,}",
        "B",
    )

    print(
        "Block KiB            :",
        f"{block_bytes / 1024:.2f}",
        "KiB",
    )

    # Save a compact machine-readable record.
    output = {
        "model_id": MODEL_ID,
        "model_type": config.model_type,
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_kv_heads,
        "head_dim": head_dim,
        "intermediate_size": intermediate_size,
        "max_position_embeddings": max_position_embeddings,
        "rope_theta": rope_theta,
        "torch_dtype": str(torch_dtype),
        "kv_dtype": "bfloat16",
        "dtype_bytes": DTYPE_BYTES,
        "block_size": BLOCK_SIZE,
        "kv_bytes_per_token": kv_bytes_per_token,
        "block_bytes": block_bytes,
    }

    output_path = Path("bench/baselines/model_config.json")

    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print("\nSaved:")
    print(output_path)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
