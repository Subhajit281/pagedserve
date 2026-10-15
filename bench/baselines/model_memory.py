import gc
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def gib(bytes_value):
    return bytes_value / (1024 ** 3)


def print_gpu_memory(label):
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()

    free, total = torch.cuda.mem_get_info()

    print(f"\n--- {label} ---")
    print(f"Allocated : {gib(allocated):.3f} GiB")
    print(f"Reserved  : {gib(reserved):.3f} GiB")
    print(f"Free      : {gib(free):.3f} GiB")
    print(f"Total     : {gib(total):.3f} GiB")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("=" * 60)
    print("       PagedServe Model Memory Baseline")
    print("=" * 60)

    print("\nModel:", MODEL_ID)
    print("GPU  :", torch.cuda.get_device_name(0))

    torch.cuda.empty_cache()
    gc.collect()

    print_gpu_memory("Before loading model")

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    print("Tokenizer loaded.")

    print("\nLoading model in BF16...")

    start = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
    )

    model = model.to("cuda")
    model.eval()

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"Model loaded in {elapsed:.2f} seconds.")

    print_gpu_memory("After loading model")

    print("\n--- Model ---")
    print("Parameters :", sum(p.numel() for p in model.parameters()))
    print("Dtype      :", next(model.parameters()).dtype)
    print("Device     :", next(model.parameters()).device)

    print("\n--- Estimated weight memory ---")

    parameter_bytes = sum(
        p.numel() * p.element_size()
        for p in model.parameters()
    )

    print(
        "Parameter memory:",
        f"{gib(parameter_bytes):.3f} GiB",
    )

    print("\n--- GPU State ---")

    print(
        "Max allocated:",
        f"{gib(torch.cuda.max_memory_allocated()):.3f} GiB",
    )

    print(
        "Max reserved:",
        f"{gib(torch.cuda.max_memory_reserved()):.3f} GiB",
    )

    print("\nModel successfully loaded.")

    # Keep model alive until measurements are printed.


if __name__ == "__main__":
    main()
