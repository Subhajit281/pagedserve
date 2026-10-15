import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

from pagedserve.attention.triton_paged_decode import (
    triton_paged_decode_attention,
)
from pagedserve.attention.vectorized_batched_attention import (
    vectorized_paged_attention,
)
from pagedserve.engine.paged_kv_cache import PagedKVCache


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_triton_decode_matches_pytorch_with_fragmented_blocks(dtype):
    torch.manual_seed(7)
    device = "cuda"
    block_size = 16
    lengths = [17, 32, 65]
    request_ids = ["a", "b", "c"]

    cache = PagedKVCache(
        num_blocks=32,
        num_layers=1,
        num_kv_heads=2,
        head_dim=64,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )

    # Interleave allocations, then grow them to produce fragmented mappings.
    for rid in request_ids:
        cache.allocate(rid, 1)
    for rid, length in zip(request_ids, lengths):
        cache.append_tokens(rid, length - 1)

    for rid, length in zip(request_ids, lengths):
        for pos in range(length):
            key = torch.randn(2, 64, device=device, dtype=dtype)
            value = torch.randn_like(key)
            cache.write_token(rid, 0, pos, key, value)

    query = torch.randn(3, 4, 1, 64, device=device, dtype=dtype)
    positions = [length - 1 for length in lengths]

    expected = vectorized_paged_attention(
        query, cache, request_ids, positions, layer=0
    )
    actual = triton_paged_decode_attention(
        query, cache, request_ids, positions, layer=0
    )
    torch.cuda.synchronize()

    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-4
    torch.testing.assert_close(
        actual, expected, rtol=tolerance, atol=tolerance
    )


def test_triton_decode_rejects_cpu_query():
    cache = PagedKVCache(
        num_blocks=4, num_layers=1, num_kv_heads=1,
        head_dim=4, block_size=2, dtype=torch.float32, device="cpu"
    )
    cache.allocate("a", 1)
    with pytest.raises(ValueError, match="requires CUDA"):
        triton_paged_decode_attention(
            torch.randn(1, 1, 1, 4), cache, ["a"], [0], 0
        )


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_triton_decode_multiple_layers_and_block_boundaries(
    layer, batch_size
):
    """Validate layer addressing and sequences around block boundaries."""
    torch.manual_seed(1000 + layer * 100 + batch_size)
    device = "cuda"
    dtype = torch.bfloat16
    block_size = 16
    boundary_lengths = [1, 15, 16, 17, 31, 32, 33, 65]
    lengths = boundary_lengths[:batch_size]
    request_ids = [f"boundary-{i}" for i in range(batch_size)]

    cache = PagedKVCache(
        num_blocks=128,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        block_size=block_size,
        dtype=dtype,
        device=device,
    )

    # Interleaved allocation creates non-contiguous physical block mappings.
    for rid in request_ids:
        cache.allocate(rid, 1)

    for rid, length in zip(request_ids, lengths):
        if length > 1:
            cache.append_tokens(rid, length - 1)

    # Populate every layer independently so an incorrect layer stride
    # cannot accidentally pass by reading equivalent data.
    for rid, length in zip(request_ids, lengths):
        for cache_layer in range(2):
            for position in range(length):
                key = torch.randn(
                    2, 64, device=device, dtype=dtype
                )
                value = torch.randn_like(key)
                cache.write_token(
                    rid, cache_layer, position, key, value
                )

    query = torch.randn(
        batch_size, 4, 1, 64, device=device, dtype=dtype
    )
    positions = [length - 1 for length in lengths]

    expected = vectorized_paged_attention(
        query, cache, request_ids, positions, layer=layer
    )
    actual = triton_paged_decode_attention(
        query, cache, request_ids, positions, layer=layer
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual, expected, rtol=2e-2, atol=2e-2
    )


@pytest.mark.parametrize("layer", [0, 1])
def test_triton_decode_layers_have_independent_cache_values(layer):
    """Changing one layer's cache must not affect another layer's output."""
    torch.manual_seed(2026)
    device = "cuda"
    dtype = torch.float32

    cache = PagedKVCache(
        num_blocks=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        block_size=16,
        dtype=dtype,
        device=device,
    )
    request_ids = ["layer-check"]
    length = 17
    cache.allocate(request_ids[0], 1)
    cache.append_tokens(request_ids[0], length - 1)

    for position in range(length):
        key0 = torch.randn(2, 64, device=device, dtype=dtype)
        value0 = torch.randn_like(key0)
        key1 = torch.randn(2, 64, device=device, dtype=dtype)
        value1 = torch.randn_like(key1)
        cache.write_token(request_ids[0], 0, position, key0, value0)
        cache.write_token(request_ids[0], 1, position, key1, value1)

    query = torch.randn(1, 4, 1, 64, device=device, dtype=dtype)
    positions = [length - 1]

    expected = vectorized_paged_attention(
        query, cache, request_ids, positions, layer=layer
    )
    actual = triton_paged_decode_attention(
        query, cache, request_ids, positions, layer=layer
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        actual, expected, rtol=2e-4, atol=2e-4
    )
