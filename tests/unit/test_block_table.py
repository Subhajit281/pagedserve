import pytest

from pagedserve.engine.block_table import BlockTable


def test_empty_table():
    table = BlockTable()

    assert table.num_blocks == 0
    assert table.get_blocks() == []


def test_append_block():
    table = BlockTable()

    table.append(7)

    assert table.num_blocks == 1
    assert table.get_physical_block(0) == 7


def test_multiple_blocks():
    table = BlockTable()

    table.append(7)
    table.append(2)
    table.append(9)

    assert table.num_blocks == 3
    assert table.get_blocks() == [7, 2, 9]


def test_logical_to_physical_mapping():
    table = BlockTable()

    table.append(7)
    table.append(2)
    table.append(9)

    assert table.get_physical_block(0) == 7
    assert table.get_physical_block(1) == 2
    assert table.get_physical_block(2) == 9


def test_invalid_logical_block():
    table = BlockTable()

    table.append(7)

    with pytest.raises(IndexError):
        table.get_physical_block(1)


def test_negative_physical_block_rejected():
    table = BlockTable()

    with pytest.raises(ValueError):
        table.append(-1)


def test_get_blocks_returns_copy():
    table = BlockTable()

    table.append(7)
    table.append(2)

    blocks = table.get_blocks()
    blocks.append(99)

    assert table.get_blocks() == [7, 2]


def test_clear():
    table = BlockTable()

    table.append(7)
    table.append(2)

    table.clear()

    assert table.num_blocks == 0
    assert table.get_blocks() == []
