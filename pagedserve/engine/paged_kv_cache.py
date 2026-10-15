from __future__ import annotations

import torch

from pagedserve.engine.block_manager import BlockManager
from pagedserve.engine.block_table import BlockTable
from pagedserve.engine.kv_cache import KVCache


class PagedKVCache:
    """
    Coordinates physical block allocation, block tables,
    and KV-cache tensor storage.
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
        self.block_manager = BlockManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )

        self.kv_cache = KVCache(
            num_blocks=num_blocks,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

        # Request ID -> logical-to-physical block mapping.
        self._block_tables: dict[str, BlockTable] = {}

        # Request ID -> actual number of tokens.
        self._token_counts: dict[str, int] = {}

    def allocate(self, request_id: str, num_tokens: int) -> None:
        """Allocate blocks for a new request."""

        if request_id in self._block_tables:
            raise ValueError(f"Request {request_id!r} already exists")

        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")

        physical_blocks = self.block_manager.allocate(
            request_id=request_id,
            num_tokens=num_tokens,
        )

        table = BlockTable()

        for physical_block in physical_blocks:
            table.append(physical_block)

        self._block_tables[request_id] = table
        self._token_counts[request_id] = num_tokens

    def append_tokens(self, request_id: str, num_tokens: int) -> None:
        """
        Extend a request's token count and allocate additional
        physical blocks whenever a block boundary is crossed.

        This updates allocation metadata only; it does not write
        token K/V values into the cache tensors.
        """

        self._validate_request(request_id)

        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")

        if num_tokens == 0:
            return

        old_count = self._token_counts[request_id]
        new_count = old_count + num_tokens

        old_blocks = self.block_manager.blocks_needed(old_count)
        new_blocks = self.block_manager.blocks_needed(new_count)

        blocks_to_add = new_blocks - old_blocks

        if blocks_to_add > 0:
            physical_blocks = self.block_manager.allocate_blocks(
                request_id=request_id,
                num_blocks=blocks_to_add,
            )

            table = self._block_tables[request_id]

            for physical_block in physical_blocks:
                table.append(physical_block)

        self._token_counts[request_id] = new_count

    def get_block_table(self, request_id: str) -> BlockTable:
        """Return a request's block table."""

        self._validate_request(request_id)
        return self._block_tables[request_id]

    def token_count(self, request_id: str) -> int:
        """Return the actual token count for a request."""

        self._validate_request(request_id)
        return self._token_counts[request_id]

    def free(self, request_id: str) -> None:
        """Release all blocks belonging to a request."""

        self._validate_request(request_id)

        del self._block_tables[request_id]
        del self._token_counts[request_id]

        self.block_manager.free(request_id)

    @property
    def free_blocks(self) -> int:
        return self.block_manager.free_blocks

    @property
    def used_blocks(self) -> int:
        return self.block_manager.used_blocks

    @property
    def num_blocks(self) -> int:
        return self.block_manager.num_blocks

    @property
    def block_size(self) -> int:
        return self.block_manager.block_size

    def _validate_request(self, request_id: str) -> None:
        if request_id not in self._block_tables:
            raise KeyError(f"Unknown request: {request_id!r}")
    def write_token(
        self,
        request_id: str,
        layer: int,
        token_position: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Write one token's K/V data using its logical position."""

        self._validate_request(request_id)

        if not 0 <= token_position < self._token_counts[request_id]:
            raise IndexError(
                f"Invalid token position: {token_position}; "
                f"valid range is [0, {self._token_counts[request_id]})"
            )

        expected_shape = (self.kv_cache.num_kv_heads, self.kv_cache.head_dim)

        if key.shape != expected_shape or value.shape != expected_shape:
            raise ValueError(
                f"Expected K/V shape {expected_shape}; "
                f"got key={tuple(key.shape)}, value={tuple(value.shape)}"
            )

        if key.dtype != self.kv_cache.dtype or value.dtype != self.kv_cache.dtype:
            raise ValueError(
                f"K/V tensors must have dtype {self.kv_cache.dtype}"
            )

        if (
    	    key.device != self.kv_cache.k_cache.device
    	    or value.device != self.kv_cache.v_cache.device
	):
    	    raise ValueError(
                "K/V tensors must be on the same devices as their respective cache tensors"
    	    )

        logical_block = token_position // self.block_size
        offset = token_position % self.block_size

        physical_block = self._block_tables[
            request_id
        ].get_physical_block(logical_block)

        self.kv_cache.k_cache[
            layer, physical_block, :, offset, :
        ].copy_(key)

        self.kv_cache.v_cache[
            layer, physical_block, :, offset, :
        ].copy_(value)

    def read_token(
        self,
        request_id: str,
        layer: int,
        token_position: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read one token's K/V data using its logical position."""

        self._validate_request(request_id)

        if not 0 <= token_position < self._token_counts[request_id]:
            raise IndexError(
                f"Invalid token position: {token_position}; "
                f"valid range is [0, {self._token_counts[request_id]})"
            )

        logical_block = token_position // self.block_size
        offset = token_position % self.block_size

        physical_block = self._block_tables[
            request_id
        ].get_physical_block(logical_block)

        key = self.kv_cache.k_cache[
            layer, physical_block, :, offset, :
        ]

        value = self.kv_cache.v_cache[
            layer, physical_block, :, offset, :
        ]

        return key, value
