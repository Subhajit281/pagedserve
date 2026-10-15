from __future__ import annotations

import math


class BlockManager:
    """
    Manages physical KV-cache blocks.

    This component handles block ownership only.
    It does not allocate GPU tensors.

    Example:

        physical blocks:
        [0, 1, 2, 3, 4, 5, 6, 7]

        request A -> [0, 1, 2]
        request B -> [3, 4]

    Later, these physical block IDs will be used
    by the actual KV cache.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int = 16,
    ):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.num_blocks = num_blocks
        self.block_size = block_size

        # Physical blocks currently available.
        self._free_blocks = list(range(num_blocks))

        # request_id -> physical block IDs
        self._request_blocks: dict[str, list[int]] = {}

    @property
    def free_blocks(self) -> int:
        """Number of currently free physical blocks."""
        return len(self._free_blocks)

    @property
    def used_blocks(self) -> int:
        """Number of currently allocated physical blocks."""
        return self.num_blocks - self.free_blocks

    def blocks_needed(self, num_tokens: int) -> int:
        """
        Calculate how many physical blocks are required
        for a sequence containing num_tokens.
        """

        if num_tokens < 0:
            raise ValueError("num_tokens cannot be negative")

        if num_tokens == 0:
            return 0

        return math.ceil(
            num_tokens / self.block_size
        )

    def allocate(
        self,
        request_id: str,
        num_tokens: int,
    ) -> list[int]:
        """
        Allocate enough physical blocks for a request.

        Returns:
            List of physical block IDs.
        """

        if request_id in self._request_blocks:
            raise ValueError(
                f"Request {request_id!r} already has blocks"
            )

        required = self.blocks_needed(num_tokens)

        if required > self.free_blocks:
            raise RuntimeError(
                "Not enough KV blocks: "
                f"required={required}, "
                f"free={self.free_blocks}"
            )

        blocks = self._free_blocks[:required]

        del self._free_blocks[:required]

        self._request_blocks[request_id] = blocks

        return list(blocks)

    def free(self, request_id: str) -> None:
        """
        Release every block belonging to a request.
        """

        if request_id not in self._request_blocks:
            raise KeyError(
                f"Unknown request: {request_id!r}"
            )

        blocks = self._request_blocks.pop(request_id)

        self._free_blocks.extend(blocks)

    def allocate_blocks(self, request_id: str, num_blocks: int) -> list[int]:
    	"""Allocate additional physical blocks to an existing request."""

    	if request_id not in self._request_blocks:
        	raise KeyError(f"Unknown request: {request_id!r}")

    	if num_blocks <= 0:
        	raise ValueError("num_blocks must be positive")

    	if num_blocks > self.free_blocks:
        	raise RuntimeError(
            		f"Not enough KV blocks: required={num_blocks}, "
            		f"free={self.free_blocks}"
        	)

    	blocks = self._free_blocks[:num_blocks]
    	del self._free_blocks[:num_blocks]

    	self._request_blocks[request_id].extend(blocks)

    	return list(blocks)

    def get_blocks(self, request_id: str) -> list[int]:
        """
        Return physical blocks belonging to a request.
        """

        if request_id not in self._request_blocks:
            raise KeyError(
                f"Unknown request: {request_id!r}"
            )

        return list(self._request_blocks[request_id])

    def has_request(self, request_id: str) -> bool:
        """Return whether a request currently owns blocks."""
        return request_id in self._request_blocks
