import torch
import pytest
from pagedserve.engine.paged_kv_cache import PagedKVCache


def create_cache():
    return PagedKVCache(
        num_blocks=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        block_size=16,
        dtype=torch.float32,
        device="cpu",
    )


def test_allocate_creates_block_table():
    cache = create_cache()

    cache.allocate("request-1", 40)

    table = cache.get_block_table("request-1")

    assert table.num_blocks == 3
    assert table.get_blocks() == [0, 1, 2]


def test_allocation_updates_block_counts():
    cache = create_cache()

    cache.allocate("request-1", 40)

    assert cache.used_blocks == 3
    assert cache.free_blocks == 1


def test_multiple_requests():
    cache = create_cache()

    cache.allocate("request-1", 16)
    cache.allocate("request-2", 16)

    table1 = cache.get_block_table("request-1")
    table2 = cache.get_block_table("request-2")

    assert table1.get_blocks() == [0]
    assert table2.get_blocks() == [1]

    assert cache.used_blocks == 2
    assert cache.free_blocks == 2


def test_free_releases_blocks():
    cache = create_cache()

    cache.allocate("request-1", 40)

    assert cache.used_blocks == 3

    cache.free("request-1")

    assert cache.used_blocks == 0
    assert cache.free_blocks == 4


def test_free_then_reuse_blocks():
    cache = create_cache()

    cache.allocate("request-1", 16)
    cache.free("request-1")

    cache.allocate("request-2", 16)

    table = cache.get_block_table("request-2")

    assert table.get_blocks() == [1] or table.get_blocks() == [0]
    assert cache.used_blocks == 1

@pytest.mark.parametrize(
    ("initial_tokens", "appended_tokens", "expected_tokens", "expected_blocks"),
    [
        (15, 1, 16, 1),
        (16, 1, 17, 2),
        (31, 1, 32, 2),
        (32, 1, 33, 3),
        (40, 8, 48, 3),
        (48, 1, 49, 4),
    ],
)
def test_dynamic_block_growth(
    initial_tokens,
    appended_tokens,
    expected_tokens,
    expected_blocks,
):
    cache = create_cache()

    cache.allocate("request-A", initial_tokens)
    cache.append_tokens("request-A", appended_tokens)

    assert cache.token_count("request-A") == expected_tokens

    table = cache.get_block_table("request-A")
    assert table.num_blocks == expected_blocks

    assert cache.used_blocks == expected_blocks
    assert cache.free_blocks == 4 - expected_blocks


def test_append_zero_tokens_does_not_allocate_blocks():
    cache = create_cache()

    cache.allocate("request-A", 16)
    original_blocks = cache.get_block_table("request-A").get_blocks()

    cache.append_tokens("request-A", 0)

    assert cache.token_count("request-A") == 16
    assert cache.get_block_table("request-A").get_blocks() == original_blocks
    assert cache.used_blocks == 1


def test_negative_append_is_rejected():
    cache = create_cache()
    cache.allocate("request-A", 16)

    with pytest.raises(ValueError):
        cache.append_tokens("request-A", -1)

    assert cache.token_count("request-A") == 16
    assert cache.used_blocks == 1


def test_append_unknown_request_is_rejected():
    cache = create_cache()

    with pytest.raises(KeyError):
        cache.append_tokens("unknown", 1)


def test_insufficient_blocks_preserves_request_state():
    cache = create_cache()

    cache.allocate("request-A", 64)

    original_blocks = cache.get_block_table("request-A").get_blocks()

    # All four physical blocks are now allocated.
    with pytest.raises(RuntimeError):
        cache.append_tokens("request-A", 1)

    assert cache.token_count("request-A") == 64
    assert cache.get_block_table("request-A").get_blocks() == original_blocks
    assert cache.used_blocks == 4
    assert cache.free_blocks == 0


def test_multiple_requests_grow_independently():
    cache = create_cache()

    cache.allocate("request-A", 16)
    cache.allocate("request-B", 16)

    cache.append_tokens("request-A", 1)

    assert cache.token_count("request-A") == 17
    assert cache.token_count("request-B") == 16

    assert cache.get_block_table("request-A").num_blocks == 2
    assert cache.get_block_table("request-B").num_blocks == 1

    assert cache.used_blocks == 3


def test_free_after_growth_releases_all_blocks():
    cache = create_cache()

    cache.allocate("request-A", 16)
    cache.append_tokens("request-A", 33)

    assert cache.used_blocks == 4

    cache.free("request-A")

    assert cache.used_blocks == 0
    assert cache.free_blocks == 4


def test_blocks_are_reusable_after_growth_and_free():
    cache = create_cache()

    cache.allocate("request-A", 16)
    cache.append_tokens("request-A", 17)

    assert cache.used_blocks == 3

    cache.free("request-A")
    cache.allocate("request-B", 48)

    assert cache.get_block_table("request-B").num_blocks == 3
    assert cache.used_blocks == 3
    assert cache.free_blocks == 1

def test_write_and_read_token():
    cache = create_cache()
    cache.allocate("request-A", 17)

    key = torch.full((2, 4), 1.5, dtype=torch.float32)
    value = torch.full((2, 4), 2.5, dtype=torch.float32)

    cache.write_token(
        request_id="request-A",
        layer=0,
        token_position=0,
        key=key,
        value=value,
    )

    actual_key, actual_value = cache.read_token(
        request_id="request-A",
        layer=0,
        token_position=0,
    )

    assert torch.equal(actual_key, key)
    assert torch.equal(actual_value, value)

def test_write_and_read_across_block_boundary():
    cache = create_cache()
    cache.allocate("request-A", 17)

    key = torch.full((2, 4), 3.0, dtype=torch.float32)
    value = torch.full((2, 4), 4.0, dtype=torch.float32)

    cache.write_token("request-A", 1, 15, key, value)
    cache.write_token("request-A", 1, 16, key * 2, value * 2)

    key_15, value_15 = cache.read_token("request-A", 1, 15)
    key_16, value_16 = cache.read_token("request-A", 1, 16)

    assert torch.equal(key_15, key)
    assert torch.equal(value_15, value)
    assert torch.equal(key_16, key * 2)
    assert torch.equal(value_16, value * 2)

def test_read_token_rejects_invalid_position():
    cache = create_cache()
    cache.allocate("request-A", 16)

    with pytest.raises(IndexError):
        cache.read_token("request-A", 0, 16)

    with pytest.raises(IndexError):
        cache.read_token("request-A", 0, -1)

def test_read_token_rejects_unknown_request():
    cache = create_cache()

    with pytest.raises(KeyError):
        cache.read_token("unknown", 0, 0)


