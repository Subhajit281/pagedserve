"""Experimental Triton kernel for single-token paged GQA decode."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_decode_kernel(
    Q, K, V, TABLE, LENGTHS, OUT,
    q_sb: tl.constexpr, q_sh: tl.constexpr, q_sd: tl.constexpr,
    k_sl: tl.constexpr, k_sb: tl.constexpr, k_sh: tl.constexpr,
    k_st: tl.constexpr, k_sd: tl.constexpr,
    v_sl: tl.constexpr, v_sb: tl.constexpr, v_sh: tl.constexpr,
    v_st: tl.constexpr, v_sd: tl.constexpr,
    t_sb: tl.constexpr, t_sm: tl.constexpr,
    o_sb: tl.constexpr, o_sh: tl.constexpr, o_sd: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    LAYER: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    q_head = tl.program_id(1)
    kv_head = q_head // (NUM_Q_HEADS // NUM_KV_HEADS)

    seq_len = tl.load(LENGTHS + row)
    s = tl.arange(0, BLOCK_S)
    d = tl.arange(0, BLOCK_D)

    q = tl.load(
        Q + row * q_sb + q_head * q_sh + d * q_sd,
        mask=d < HEAD_DIM,
        other=0.0,
    ).to(tl.float32)

    logical_block = s // BLOCK_SIZE
    block_offset = s % BLOCK_SIZE

    physical_block = tl.load(
        TABLE + row * t_sb + logical_block * t_sm,
        mask=(s < seq_len) & (logical_block < MAX_BLOCKS),
        other=0,
    )

    k_offsets = (
        physical_block[:, None] * k_sb
        + kv_head * k_sh
        + block_offset[:, None] * k_st
        + d[None, :] * k_sd
    )
    v_offsets = (
        physical_block[:, None] * v_sb
        + kv_head * v_sh
        + block_offset[:, None] * v_st
        + d[None, :] * v_sd
    )

    valid = (s[:, None] < seq_len) & (d[None, :] < HEAD_DIM)

    k = tl.load(
        K + LAYER * k_sl + k_offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    scores = tl.sum(k * q[None, :], axis=1)
    scores = scores * (1.0 / tl.sqrt(float(HEAD_DIM)))
    scores = tl.where(s < seq_len, scores, float("-inf"))

    # Numerically stable softmax.
    max_score = tl.max(scores, axis=0)
    weights = tl.exp(scores - max_score)
    weights = tl.where(s < seq_len, weights, 0.0)
    denominator = tl.sum(weights, axis=0)

    values = tl.load(
        V + LAYER * v_sl + v_offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    output = tl.sum(weights[:, None] * values, axis=0) / denominator

    tl.store(
        OUT + row * o_sb + q_head * o_sh + d * o_sd,
        output,
        mask=d < HEAD_DIM,
    )


def triton_paged_decode_attention(
    query: torch.Tensor,
    cache,
    request_ids,
    positions,
    layer: int,
) -> torch.Tensor:
    """Decode attention for query [B, Hq, 1, D], returning the same shape."""
    if not query.is_cuda:
        raise ValueError("Triton paged attention requires CUDA")
    if query.ndim != 4 or query.shape[2] != 1:
        raise ValueError("query must have shape [B, Hq, 1, D]")
    if query.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("Unsupported query dtype")

    batch, q_heads, _, head_dim = query.shape
    kv = cache.kv_cache
    kv_heads = kv.num_kv_heads

    if batch == 0:
        raise ValueError("Batch must not be empty")
    if len(request_ids) != batch or len(positions) != batch:
        raise ValueError("Request IDs and positions must match batch size")
    if len(set(request_ids)) != batch:
        raise ValueError("Request IDs must be unique")
    if head_dim != kv.head_dim:
        raise ValueError("Head dimension mismatch")
    if q_heads % kv_heads:
        raise ValueError("Query heads must be divisible by KV heads")
    if query.dtype != kv.dtype:
        raise ValueError("Query dtype must match cache dtype")
    if query.device != kv.k_cache.device:
        raise ValueError("Query and cache must be on the same device")
    if not 0 <= layer < kv.k_cache.shape[0]:
        raise ValueError("Invalid layer")

    lengths = []
    tables = []
    for request_id, position in zip(request_ids, positions):
        count = cache.token_count(request_id)
        if not 0 <= position < count:
            raise ValueError(f"Invalid position {position} for {request_id}")
        lengths.append(position + 1)
        tables.append(cache.get_block_table(request_id).get_blocks())

    max_blocks = max(len(table) for table in tables)
    table_tensor = torch.zeros(
        (batch, max_blocks), device=query.device, dtype=torch.int32
    )
    for row, table in enumerate(tables):
        table_tensor[row, :len(table)] = torch.tensor(
            table, device=query.device, dtype=torch.int32
        )

    length_tensor = torch.tensor(
        lengths, device=query.device, dtype=torch.int32
    )
    output = torch.empty_like(query)

    k = kv.k_cache
    v = kv.v_cache
    block_size = cache.block_size

    _paged_decode_kernel[(batch, q_heads)](
        query, k, v, table_tensor, length_tensor, output,
        query.stride(0), query.stride(1), query.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3), k.stride(4),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3), v.stride(4),
        table_tensor.stride(0), table_tensor.stride(1),
        output.stride(0), output.stride(1), output.stride(3),
        q_heads, kv_heads, layer, head_dim, block_size, max_blocks,
        triton.next_power_of_2(max(lengths)),
        triton.next_power_of_2(head_dim),
        num_warps=4,
    )
    return output
