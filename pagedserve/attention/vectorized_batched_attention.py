from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from pagedserve.engine.paged_kv_cache import PagedKVCache


def vectorized_paged_attention(
    query: torch.Tensor,
    cache: PagedKVCache,
    request_ids: Sequence[str],
    positions: Sequence[int],
    layer: int,
) -> torch.Tensor:
    """
    Compute single-token attention for multiple independent requests.

    Args:
        query:
            Query states with shape [B, Hq, 1, D].
        cache:
            Paged KV cache containing each request's history.
        request_ids:
            Request ID for each batch row.
        positions:
            Logical position of the query token for each request.
        layer:
            Transformer layer index.

    Returns:
        Attention output with shape [B, Hq, 1, D].

    Note:
        KV gathering still uses Python loops. The attention score,
        masking, softmax, and weighted-value calculations are batched.
        This is a correctness-oriented PyTorch implementation, not
        a fused GPU attention kernel.
    """
    if query.ndim != 4:
        raise ValueError("query must have shape [B, Hq, 1, D]")

    batch_size, num_query_heads, query_length, head_dim = query.shape

    if batch_size == 0:
        raise ValueError("Batch must not be empty")

    if query_length != 1:
        raise NotImplementedError(
            "Only single-token decoding is supported"
        )

    if len(request_ids) != batch_size or len(positions) != batch_size:
        raise ValueError(
            "request_ids and positions must match the batch size"
        )

    if len(set(request_ids)) != batch_size:
        raise ValueError("Request IDs must be unique")

    if head_dim != cache.kv_cache.head_dim:
        raise ValueError("Query head dimension does not match the cache")

    num_kv_heads = cache.kv_cache.num_kv_heads

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "Query heads must be divisible by KV heads"
        )

    if query.dtype != cache.kv_cache.dtype:
        raise ValueError("Query dtype does not match the cache dtype")

    if query.device != cache.kv_cache.k_cache.device:
        raise ValueError("Query and KV cache must be on the same device")

    lengths = []

    for request_id, position in zip(request_ids, positions):
        capacity = cache.token_count(request_id)

        if not 0 <= position < capacity:
            raise ValueError(
                f"Invalid position {position} for request {request_id!r}; "
                f"allocated capacity is {capacity}"
            )

        lengths.append(position + 1)

    max_length = max(lengths)

    # [B, Hkv, Smax, D]
    keys = torch.zeros(
        batch_size,
        num_kv_heads,
        max_length,
        head_dim,
        dtype=query.dtype,
        device=query.device,
    )

    values = torch.zeros_like(keys)

    # True for valid tokens, False for padded positions.
    valid_mask = (
        torch.arange(max_length, device=query.device).unsqueeze(0)
        < torch.tensor(lengths, device=query.device).unsqueeze(1)
    )

    # Gather each request through its logical-to-physical block mapping.
    # Requests can have different lengths and different physical blocks.
    for row, (request_id, position) in enumerate(
        zip(request_ids, positions)
    ):
        for token_position in range(position + 1):
            key, value = cache.read_token(
                request_id=request_id,
                layer=layer,
                token_position=token_position,
            )

            keys[row, :, token_position, :] = key
            values[row, :, token_position, :] = value

    # Grouped-query attention: map each query head to its KV head.
    queries_per_kv_head = num_query_heads // num_kv_heads

    keys = keys.repeat_interleave(
        queries_per_kv_head,
        dim=1,
    )

    values = values.repeat_interleave(
        queries_per_kv_head,
        dim=1,
    )

    # [B, Hq, 1, Smax]
    scores = torch.matmul(
        query,
        keys.transpose(-1, -2),
    ) / math.sqrt(head_dim)

    # Mask padded cache positions before softmax.
    scores = scores.masked_fill(
        ~valid_mask[:, None, None, :],
        float("-inf"),
    )

    probabilities = torch.softmax(
        scores,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)

    # [B, Hq, 1, D]
    return torch.matmul(probabilities, values)
