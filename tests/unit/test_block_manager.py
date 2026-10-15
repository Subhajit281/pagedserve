import pytest

from pagedserve.engine.block_manager import BlockManager


def test_initial_state():
    manager = BlockManager(
        num_blocks=8,
        block_size=16,
    )

    assert manager.free_blocks == 8
    assert manager.used_blocks == 0


def test_blocks_needed():
    manager = BlockManager(
        num_blocks=8,
        block_size=16,
    )

    assert manager.blocks_needed(0) == 0
    assert manager.blocks_needed(1) == 1
    assert manager.blocks_needed(16) == 1
    assert manager.blocks_needed(17) == 2
    assert manager.blocks_needed(32) == 2
    assert manager.blocks_needed(33) == 3


def test_allocate():
    manager = BlockManager(
        num_blocks=8,
        block_size=16,
    )

    blocks = manager.allocate(
        "request-A",
        40,
    )

    assert len(blocks) == 3
    assert manager.used_blocks == 3
    assert manager.free_blocks == 5

    assert manager.get_blocks("request-A") == blocks


def test_multiple_requests():
    manager = BlockManager(
        num_blocks=8,
        block_size=16,
    )

    a = manager.allocate("request-A", 20)
    b = manager.allocate("request-B", 30)

    assert len(a) == 2
    assert len(b) == 2

    assert set(a).isdisjoint(b)

    assert manager.used_blocks == 4
    assert manager.free_blocks == 4


def test_free():
    manager = BlockManager(
        num_blocks=8,
        block_size=16,
    )

    manager.allocate("request-A", 40)

    manager.free("request-A")

    assert manager.free_blocks == 8
    assert manager.used_blocks == 0

    assert not manager.has_request("request-A")


def test_out_of_blocks():
    manager = BlockManager(
        num_blocks=2,
        block_size=16,
    )

    manager.allocate("request-A", 32)

    with pytest.raises(RuntimeError):
        manager.allocate("request-B", 1)


def test_duplicate_request():
    manager = BlockManager(
        num_blocks=8,
        block_size=16,
    )

    manager.allocate("request-A", 16)

    with pytest.raises(ValueError):
        manager.allocate("request-A", 16)


def test_unknown_request():
    manager = BlockManager(
        num_blocks=8,
        block_size=16,
    )

    with pytest.raises(KeyError):
        manager.get_blocks("does-not-exist")

    with pytest.raises(KeyError):
        manager.free("does-not-exist")
