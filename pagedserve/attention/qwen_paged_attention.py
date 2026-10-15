from __future__ import annotations

import torch
from torch import nn

from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from pagedserve.attention.paged_attention import paged_attention
from pagedserve.engine.paged_kv_cache import PagedKVCache


def qwen_paged_attention(
    attention: nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    cache: PagedKVCache,
    request_id: str,
    query_start_position: int,
) -> torch.Tensor:
    """
    Execute Qwen2 attention using PagedServe's KV cache.

    Supports batch size 1, prompt prefill, and single-token decoding.

    Args:
        attention: Original pretrained Qwen2Attention module.
        hidden_states: [1, query_length, hidden_size].
        position_embeddings: Qwen's (cos, sin) rotary embeddings.
        cache: PagedServe KV cache.
        request_id: Active request identifier.
        query_start_position: Position of the first query token.

    Returns:
        Attention output shaped [1, query_length, hidden_size].
    """
    if hidden_states.ndim != 3:
        raise ValueError(
            "hidden_states must have shape "
            "[batch_size, query_length, hidden_size]"
        )

    batch_size, query_length, hidden_size = hidden_states.shape

    if batch_size != 1:
        raise NotImplementedError(
            "Only batch_size=1 is supported currently"
        )

    if query_length <= 0:
        raise ValueError("query_length must be positive")

    if query_start_position < 0:
        raise ValueError("query_start_position cannot be negative")

    if hidden_size != attention.config.hidden_size:
        raise ValueError("Hidden size does not match Qwen configuration")

    num_query_heads = attention.config.num_attention_heads
    num_kv_heads = attention.config.num_key_value_heads
    head_dim = attention.head_dim

    if num_query_heads % num_kv_heads != 0:
        raise ValueError(
            "Query heads must be divisible by KV heads"
        )

    query_end_position = query_start_position + query_length

    if query_end_position > cache.token_count(request_id):
        raise ValueError(
            "Query positions exceed the request's allocated token count"
        )

    # 1. Original Qwen projections.
    projection_shape = (batch_size, query_length, -1, head_dim)

    query_states = (
        attention.q_proj(hidden_states)
        .view(projection_shape)
        .transpose(1, 2)
    )

    key_states = (
        attention.k_proj(hidden_states)
        .view(projection_shape)
        .transpose(1, 2)
    )

    value_states = (
        attention.v_proj(hidden_states)
        .view(projection_shape)
        .transpose(1, 2)
    )

    # 2. Apply Qwen's original rotary positional embeddings.
    cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
    )

    query = query_states[0]
    key = key_states[0]
    value = value_states[0]

    # 3. Write the new K/V entries into paged storage.
    for token_index in range(query_length):
        token_position = query_start_position + token_index

        cache.write_token(
            request_id=request_id,
            layer=attention.layer_idx,
            token_position=token_position,
            key=key[:, token_index, :],
            value=value[:, token_index, :],
        )

    # 4. Compute attention using the paged KV cache.
    attended = paged_attention(
        query=query,
        cache=cache,
        request_id=request_id,
        layer=attention.layer_idx,
        query_start_position=query_start_position,
    )

    # 5. Restore Qwen's expected tensor layout.
    attended = attended.transpose(0, 1).contiguous()
    attended = attended.reshape(batch_size, query_length, hidden_size)

    # 6. Original Qwen output projection.
    return attention.o_proj(attended)
