import math


# ---------------------------------------------------------
# Qwen2.5-0.5B planning configuration
# ---------------------------------------------------------

NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64

# BF16 = 2 bytes
DTYPE_BYTES = 2

# Paged KV block size
BLOCK_SIZE = 16

# Initial planning KV pool
POOL_GIB = 1.5


def kv_bytes_per_token():
    """
    KV memory required for one token.

    Each token stores:
        K + V
    """
    return (
        2
        * NUM_LAYERS
        * NUM_KV_HEADS
        * HEAD_DIM
        * DTYPE_BYTES
    )


def block_bytes():
    """Memory required by one KV block."""
    return kv_bytes_per_token() * BLOCK_SIZE


def num_blocks(pool_gib):
    """Number of complete KV blocks in the pool."""
    pool_bytes = pool_gib * (1024 ** 3)
    return int(pool_bytes // block_bytes())


def blocks_needed(num_tokens):
    """Number of blocks required for a sequence."""
    return math.ceil(num_tokens / BLOCK_SIZE)


def main():
    token_bytes = kv_bytes_per_token()
    bytes_per_block = block_bytes()
    blocks = num_blocks(POOL_GIB)

    print("=" * 55)
    print("              PagedServe KV Budget")
    print("=" * 55)

    print("\n--- Model KV Configuration ---")
    print("Layers           :", NUM_LAYERS)
    print("KV heads         :", NUM_KV_HEADS)
    print("Head dimension   :", HEAD_DIM)
    print("Dtype bytes      :", DTYPE_BYTES)
    print("Block size       :", BLOCK_SIZE, "tokens")

    print("\n--- KV Memory ---")

    print(
        "KV bytes/token   :",
        f"{token_bytes:,}",
        "B",
    )

    print(
        "KV KiB/token     :",
        f"{token_bytes / 1024:.2f}",
        "KiB",
    )

    print(
        "Block bytes      :",
        f"{bytes_per_block:,}",
        "B",
    )

    print(
        "Block KiB        :",
        f"{bytes_per_block / 1024:.2f}",
        "KiB",
    )

    print("\n--- KV Pool ---")

    print("Pool size        :", POOL_GIB, "GiB")
    print("Number of blocks :", f"{blocks:,}")
    print(
        "Token capacity   :",
        f"{blocks * BLOCK_SIZE:,}",
        "tokens",
    )

    print("\n--- Sequence Examples ---")

    for tokens in [16, 32, 100, 256, 400, 512, 1024, 2048]:

        required = blocks_needed(tokens)

        memory_mib = (
            required * bytes_per_block / (1024 ** 2)
        )

        print(
            f"{tokens:4d} tokens -> "
            f"{required:4d} blocks -> "
            f"{memory_mib:8.2f} MiB"
        )

    print("=" * 55)


if __name__ == "__main__":
    main()
