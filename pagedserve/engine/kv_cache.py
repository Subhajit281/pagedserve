from __future__ import annotations

import torch


class KVCache:
    """
    GPU storage for the paged KV cache.

    The cache is organized as:

        [layer, physical_block, kv_head, token_in_block, head_dim]

    A physical block therefore contains KV data for all transformer
    layers for a fixed number of token positions.
    """

    def __init__(
        self,
        num_blocks: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if num_kv_heads <= 0:
            raise ValueError("num_kv_heads must be positive")

        if head_dim <= 0:
            raise ValueError("head_dim must be positive")

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = torch.device(device)

        shape = (
            num_layers,
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
        )

        self.k_cache = torch.empty(
            shape,
            dtype=dtype,
            device=self.device,
        )

        self.v_cache = torch.empty(
            shape,
            dtype=dtype,
            device=self.device,
        )

    @property
    def block_shape(self) -> tuple[int, ...]:
        """Shape of one physical block for one layer."""
        return (
            self.num_kv_heads,
            self.block_size,
            self.head_dim,
        )

    @property
    def block_bytes(self) -> int:
        """Total K+V memory used by one physical block."""
        return (
            self.num_layers
            * 2  # K + V
            * self.num_kv_heads
            * self.block_size
            * self.head_dim
            * torch.tensor([], dtype=self.dtype).element_size()
        )

    @property
    def total_bytes(self) -> int:
        """Total K+V memory allocated by the cache."""
        return self.num_blocks * self.block_bytes

    def write_block(
        self,
        layer: int,
        block_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write a complete physical block for one layer."""

        self._validate_layer(layer)
        self._validate_block(block_id)

        expected_shape = self.block_shape

        if key.shape != expected_shape:
            raise ValueError(
                f"Invalid key shape: expected {expected_shape}, "
                f"got {tuple(key.shape)}"
            )

        if value.shape != expected_shape:
            raise ValueError(
                f"Invalid value shape: expected {expected_shape}, "
                f"got {tuple(value.shape)}"
            )

        if key.dtype != self.dtype:
            raise ValueError(
                f"Invalid key dtype: expected {self.dtype}, "
                f"got {key.dtype}"
            )

        if value.dtype != self.dtype:
            raise ValueError(
                f"Invalid value dtype: expected {self.dtype}, "
                f"got {value.dtype}"
            )

        self.k_cache[layer, block_id].copy_(key)
        self.v_cache[layer, block_id].copy_(value)

    def read_block(
        self,
        layer: int,
        block_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the K and V tensors for one physical block."""

        self._validate_layer(layer)
        self._validate_block(block_id)

        return (
            self.k_cache[layer, block_id],
            self.v_cache[layer, block_id],
        )

    def _validate_layer(self, layer: int) -> None:
        if not 0 <= layer < self.num_layers:
            raise IndexError(
                f"Invalid layer: {layer}; "
                f"valid range is [0, {self.num_layers})"
            )

    def _validate_block(self, block_id: int) -> None:
        if not 0 <= block_id < self.num_blocks:
            raise IndexError(
                f"Invalid block ID: {block_id}; "
                f"valid range is [0, {self.num_blocks})"
            )