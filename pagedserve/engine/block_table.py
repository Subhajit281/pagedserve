from __future__ import annotations


class BlockTable:
    """
    Maps logical block indices to physical KV-cache block IDs
    for a single request.

    Example:
        logical block 0 -> physical block 7
        logical block 1 -> physical block 2
        logical block 2 -> physical block 9

    Internally:
        [7, 2, 9]
    """

    def __init__(self) -> None:
        self._blocks: list[int] = []

    @property
    def num_blocks(self) -> int:
        """Number of logical blocks currently mapped."""
        return len(self._blocks)

    def append(self, physical_block_id: int) -> None:
        """Append a new physical block to the next logical position."""
        if physical_block_id < 0:
            raise ValueError("physical_block_id cannot be negative")

        self._blocks.append(physical_block_id)

    def get_physical_block(self, logical_block_id: int) -> int:
        """Return the physical block mapped to a logical block."""
        if not 0 <= logical_block_id < self.num_blocks:
            raise IndexError(
                f"Invalid logical block: {logical_block_id}; "
                f"valid range is [0, {self.num_blocks})"
            )

        return self._blocks[logical_block_id]

    def get_blocks(self) -> list[int]:
        """Return a copy of the logical-to-physical mapping."""
        return list(self._blocks)

    def clear(self) -> None:
        """Remove all logical-to-physical mappings."""
        self._blocks.clear()