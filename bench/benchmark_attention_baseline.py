from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from pagedserve.attention.vectorized_batched_attention import (
    vectorized_paged_attention,
)
from pagedserve.engine.paged_kv_cache import PagedKVCache


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA.")

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    batch_size = 3
    num_kv_heads = 2
    num_query_heads = 4
    head_dim = 64
    block_size = 16
    lengths = [32, 64, 128]

    cache = PagedKVCache(
        num_blocks=64,
        num_layers=1,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )

    request_ids = ["attn-A", "attn-B", "attn-C"]

    # Interleave allocations so requests occupy non-contiguous physical
    # blocks. The attention implementation must use the block table.
    for request_id in request_ids:
        cache.allocate(request_id, 1)

    for request_id, length in zip(request_ids, lengths):
        cache.append_tokens(request_id, length - 1)

    for request_id, length in zip(request_ids, lengths):
        for position in range(length):
            key = torch.randn(
                num_kv_heads, head_dim, device=device, dtype=dtype
            )
            value = torch.randn_like(key)
            cache.write_token(
                request_id,
                layer=0,
                token_position=position,
                key=key,
                value=value,
            )

    positions = [length - 1 for length in lengths]
    query = torch.randn(
        batch_size,
        num_query_heads,
        1,
        head_dim,
        device=device,
        dtype=dtype,
    )

    def reference():
        outputs = []
        for request_id, position in zip(request_ids, positions):
            keys, values = [], []
            for token_position in range(position + 1):
                key, value = cache.read_token(
                    request_id, layer=0, token_position=token_position
                )
                keys.append(key)
                values.append(value)

            k = torch.stack(keys, dim=1).repeat_interleave(
                num_query_heads // num_kv_heads, dim=0
            )
            v = torch.stack(values, dim=1).repeat_interleave(
                num_query_heads // num_kv_heads, dim=0
            )
            q = query[len(outputs)].unsqueeze(0)
            outputs.append(
                F.scaled_dot_product_attention(
                    q, k.unsqueeze(0), v.unsqueeze(0), is_causal=False
                )
            )
        return torch.cat(outputs, dim=0)

    def current():
        return vectorized_paged_attention(
            query=query,
            cache=cache,
            request_ids=request_ids,
            positions=positions,
            layer=0,
        )

    # Warm up compilation/caches and CUDA execution.
    with torch.inference_mode():
        for _ in range(5):
            current()
            reference()
    synchronize()

    expected = reference()
    actual = current()
    synchronize()

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)

    def measure(fn, repeats=30):
        samples_ms = []
        with torch.inference_mode():
            for _ in range(repeats):
                synchronize()
                start = time.perf_counter()
                fn()
                synchronize()
                samples_ms.append((time.perf_counter() - start) * 1000)
        return {
            "median_ms": statistics.median(samples_ms),
            "mean_ms": statistics.mean(samples_ms),
            "min_ms": min(samples_ms),
            "repeats": repeats,
        }

    result = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": str(dtype),
        "batch_size": batch_size,
        "num_query_heads": num_query_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "sequence_lengths": lengths,
        "block_size": block_size,
        "physical_blocks": {
            request_id: cache.get_block_table(request_id).get_blocks()
            for request_id in request_ids
        },
        "parity": {
            "max_abs_error": (actual.float() - expected.float())
            .abs()
            .max()
            .item(),
            "passed": True,
        },
        "current_vectorized_pytorch": measure(current),
        "pytorch_sdpa_reference": measure(reference),
    }

    output_path = Path("bench/results/attention_baseline_20261016.json")
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
