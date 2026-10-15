from __future__ import annotations

import math

import torch

from pagedserve.engine.paged_kv_cache import PagedKVCache


def paged_attention(
    query: torch.Tensor,
    cache: PagedKVCache,
    request_id: str,
    layer: int,
    query_start_position: int,
) -> torch.Tensor:
    """
    Compute causal attention using K/V entries stored in PagedKVCache.

    Args:
        query:
            Tensor shaped [num_query_heads, query_length, head_dim].
        cache:
            Paged KV cache containing the request's K/V entries.
        request_id:
            Request whose cached K/V tensors will be read.
        layer:
            Transformer layer index.
        query_start_position:
            Logical position of the first query token.

    Returns:
        Tensor shaped [num_query_heads, query_length, head_dim].

    Notes:
        - Supports grouped-query attention when num_query_heads is
          divisible by num_kv_heads.
        - Assumes K/V entries for all positions up to the query
          positions have already been written to the cache.
        - Uses a simple PyTorch implementation for correctness,
          not optimized GPU performance.
    """
    if query.ndim != 3:
        raise ValueError(
            "query must have shape "
            "[num_query_heads, query_length, head_dim]"
        )

    num_query_heads, query_length, head_dim = query.shape

    if query_length <= 0:
        raise ValueError("query_length must be positive")

    if head_dim != cache.kv_cache.head_dim:
        raise ValueError(
            f"Expected head_dim {cache.kv_cache.head_dim}, "
            f"got {head_dim}"
        )

    num_kv_heads = cache.kv_cache.num_kv_heads

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "num_query_heads must be divisible by num_kv_heads"
        )

    if query.dtype != cache.kv_cache.dtype:
        raise ValueError(
            f"Query dtype must be {cache.kv_cache.dtype}"
        )

    if query.device != cache.kv_cache.k_cache.device:
        raise ValueError(
            "Query must be on the same device as the KV cache"
        )

    if query_start_position < 0:
        raise ValueError("query_start_position cannot be negative")

    cached_tokens = cache.token_count(request_id)
    query_end_position = query_start_position + query_length

    if query_end_position > cached_tokens:
        raise ValueError(
            f"Query positions end at {query_end_position - 1}, "
            f"but request has only {cached_tokens} allocated tokens"
        )

    queries_per_kv_head = num_query_heads // num_kv_heads
    scale = 1.0 / math.sqrt(head_dim)

    output = torch.empty_like(query)

    for query_index in range(query_length):
        position = query_start_position + query_index

        keys = []
        values = []

        # Read only positions visible to this query under causal attention.
        for token_position in range(position + 1):
            key, value = cache.read_token(
                request_id=request_id,
                layer=layer,
                token_position=token_position,
            )
            keys.append(key)
            values.append(value)

        # [num_kv_heads, visible_tokens, head_dim]
        key_tensor = torch.stack(keys, dim=1)
        value_tensor = torch.stack(values, dim=1)

        # Expand each KV head across its associated query heads.
        key_tensor = key_tensor.repeat_interleave(
            queries_per_kv_head,
            dim=0,
        )
        value_tensor = value_tensor.repeat_interleave(
            queries_per_kv_head,
            dim=0,
        )
	# [num_query_heads, visible_tokens]
        # Match Hugging Face eager attention arithmetic.
        query_token = query[:, query_index, :].unsqueeze(1)

        scores = torch.matmul(
            query_token,
            key_tensor.transpose(1, 2),
        ).squeeze(1) * scale

        # Softmax in FP32, then cast probabilities to the query dtype.
        probabilities = torch.softmax(
            scores,
            dim=-1,
            dtype=torch.float32,
        ).to(query.dtype)

        # Weighted-value multiplication in the original dtype.
        attended = torch.matmul(
            probabilities.unsqueeze(1),
            value_tensor,
        ).squeeze(1)

        output[:, query_index, :] = attended
    return output

