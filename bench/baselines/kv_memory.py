import gc

import torch


NUM_LAYERS = 24
NUM_KV_HEADS = 2
HEAD_DIM = 64

DTYPE = torch.bfloat16


def gib(value):
    return value / (1024 ** 3)


def allocate_kv(num_tokens):
    keys = []
    values = []

    for _ in range(NUM_LAYERS):
        k = torch.empty(
            1,
            NUM_KV_HEADS,
            num_tokens,
            HEAD_DIM,
            dtype=DTYPE,
            device="cuda",
        )

        v = torch.empty(
            1,
            NUM_KV_HEADS,
            num_tokens,
            HEAD_DIM,
            dtype=DTYPE,
            device="cuda",
        )

        keys.append(k)
        values.append(v)

    return keys, values


def main():

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=" * 60)
    print("             PagedServe KV Memory")
    print("=" * 60)

    print("GPU:", torch.cuda.get_device_name(0))

    for tokens in [16, 32, 64, 128, 256, 512, 1024]:

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        before = torch.cuda.memory_allocated()

        keys, values = allocate_kv(tokens)

        torch.cuda.synchronize()

        after = torch.cuda.memory_allocated()

        actual = after - before

        theoretical = (
            2
            * NUM_LAYERS
            * NUM_KV_HEADS
            * HEAD_DIM
            * 2
            * tokens
        )

        print(
            f"{tokens:4d} tokens | "
            f"actual: {gib(actual):8.4f} GiB | "
            f"theoretical: {gib(theoretical):8.4f} GiB"
        )

        del keys
        del values

    print("=" * 60)


if __name__ == "__main__":
    main()
