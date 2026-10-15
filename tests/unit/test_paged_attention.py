import pytest
import torch
import torch.nn.functional as F

from pagedserve.attention.paged_attention import paged_attention
from pagedserve.engine.paged_kv_cache import PagedKVCache


def create_cache(num_blocks=4, block_size=2):
    return PagedKVCache(
        num_blocks=num_blocks,
        num_layers=1,
        num_kv_heads=2,
        head_dim=4,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
    )


def write_reference(cache, request_id, keys, values):
    """Write tensors shaped [kv_heads, sequence_length, head_dim]."""
    sequence_length = keys.shape[1]

    cache.allocate(request_id, sequence_length)

    for position in range(sequence_length):
        cache.write_token(
            request_id,
            layer=0,
            token_position=position,
            key=keys[:, position, :],
            value=values[:, position, :],
        )


def test_paged_attention_matches_gqa_reference():
    torch.manual_seed(42)

    num_query_heads = 4
    num_kv_heads = 2
    sequence_length = 5
    head_dim = 4

    keys = torch.randn(num_kv_heads, sequence_length, head_dim)
    values = torch.randn_like(keys)
    queries = torch.randn(num_query_heads, sequence_length, head_dim)

    cache = create_cache(num_blocks=3, block_size=2)
    write_reference(cache, "request-A", keys, values)

    actual = paged_attention(
        query=queries,
        cache=cache,
        request_id="request-A",
        layer=0,
        query_start_position=0,
    )

    # Repeat each KV head for its two corresponding query heads.
    expanded_keys = keys.repeat_interleave(2, dim=0)
    expanded_values = values.repeat_interleave(2, dim=0)

    expected = torch.empty_like(queries)

    for position in range(sequence_length):
        expected[:, position, :] = F.scaled_dot_product_attention(
            queries[:, position : position + 1, :].unsqueeze(0),
            expanded_keys[:, : position + 1, :].unsqueeze(0),
            expanded_values[:, : position + 1, :].unsqueeze(0),
            is_causal=False,
        ).squeeze(0).squeeze(1)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_paged_attention_supports_single_token_decode():
    torch.manual_seed(123)

    sequence_length = 5
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 4

    keys = torch.randn(num_kv_heads, sequence_length, head_dim)
    values = torch.randn_like(keys)

    cache = create_cache(num_blocks=3, block_size=2)
    write_reference(cache, "request-A", keys, values)

    query = torch.randn(num_query_heads, 1, head_dim)

    actual = paged_attention(
        query=query,
        cache=cache,
        request_id="request-A",
        layer=0,
        query_start_position=sequence_length - 1,
    )

    expected = F.scaled_dot_product_attention(
        query.unsqueeze(0),
        keys.repeat_interleave(2, dim=0).unsqueeze(0),
        values.repeat_interleave(2, dim=0).unsqueeze(0),
        is_causal=False,
    ).squeeze(0)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_paged_attention_rejects_invalid_query_shape():
    cache = create_cache()
    cache.allocate("request-A", 1)

    with pytest.raises(ValueError, match="query must have shape"):
        paged_attention(
            query=torch.randn(2, 4),
            cache=cache,
            request_id="request-A",
            layer=0,
            query_start_position=0,
        )


def test_paged_attention_supports_noncontiguous_physical_blocks():
    torch.manual_seed(456)

    sequence_length = 5
    num_query_heads = 4
    num_kv_heads = 2
    head_dim = 4
    block_size = 2

    cache = create_cache(num_blocks=5, block_size=block_size)

    # Allocate three requests to create fragmentation.
    cache.allocate("request-A", 2)  # Physical block [0]
    cache.allocate("request-B", 2)  # Physical block [1]
    cache.allocate("request-C", 2)  # Physical block [2]

    # Free the middle allocation.
    cache.free("request-B")

    # Allocate five tokens.
    # The allocator should assign physical blocks [3, 4, 1].
    cache.allocate("request-D", sequence_length)

    block_table = cache.get_block_table("request-D")

    assert block_table.get_blocks() == [3, 4, 1]

    keys = torch.randn(num_kv_heads, sequence_length, head_dim)
    values = torch.randn_like(keys)
    queries = torch.randn(num_query_heads, sequence_length, head_dim)

    # Write tokens through the logical-to-physical block mapping.
    for position in range(sequence_length):
        cache.write_token(
            request_id="request-D",
            layer=0,
            token_position=position,
            key=keys[:, position, :],
            value=values[:, position, :],
        )

    actual = paged_attention(
        query=queries,
        cache=cache,
        request_id="request-D",
        layer=0,
        query_start_position=0,
    )

    # Reference implementation: attention over the original
    # logical sequence, independent of physical block placement.
    expanded_keys = keys.repeat_interleave(2, dim=0)
    expanded_values = values.repeat_interleave(2, dim=0)

    expected = torch.empty_like(queries)

    for position in range(sequence_length):
        expected[:, position, :] = F.scaled_dot_product_attention(
            queries[:, position : position + 1, :].unsqueeze(0),
            expanded_keys[:, : position + 1, :].unsqueeze(0),
            expanded_values[:, : position + 1, :].unsqueeze(0),
            is_causal=False,
        ).squeeze(0).squeeze(1)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-5,
        atol=1e-5,
    )
