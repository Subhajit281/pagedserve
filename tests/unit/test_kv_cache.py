import pytest
import torch

from pagedserve.engine.kv_cache import KVCache


@pytest.fixture
def cache():
    return KVCache(
        num_blocks=4,
        num_layers=2,
        num_kv_heads=2,
        head_dim=4,
        block_size=16,
        dtype=torch.float32,
        device="cpu",
    )


def test_cache_shape(cache):
    assert cache.k_cache.shape == (2, 4, 2, 16, 4)
    assert cache.v_cache.shape == (2, 4, 2, 16, 4)


def test_cache_dtype(cache):
    assert cache.k_cache.dtype == torch.float32
    assert cache.v_cache.dtype == torch.float32


def test_cache_device(cache):
    assert cache.k_cache.device.type == "cpu"
    assert cache.v_cache.device.type == "cpu"


def test_block_shape(cache):
    assert cache.block_shape == (2, 16, 4)


def test_block_bytes(cache):
    # 2 layers × 2(K/V) × 2 KV heads × 16 tokens × 4 dims × 4 bytes
    expected = 2 * 2 * 2 * 16 * 4 * 4
    assert cache.block_bytes == expected


def test_total_bytes(cache):
    assert cache.total_bytes == 4 * cache.block_bytes


def test_write_and_read_block(cache):
    key = torch.randn(cache.block_shape)
    value = torch.randn(cache.block_shape)

    cache.write_block(
        layer=0,
        block_id=1,
        key=key,
        value=value,
    )

    stored_key, stored_value = cache.read_block(
        layer=0,
        block_id=1,
    )

    assert torch.equal(stored_key, key)
    assert torch.equal(stored_value, value)


def test_invalid_layer(cache):
    key = torch.randn(cache.block_shape)
    value = torch.randn(cache.block_shape)

    with pytest.raises(IndexError):
        cache.write_block(
            layer=99,
            block_id=0,
            key=key,
            value=value,
        )


def test_invalid_block(cache):
    key = torch.randn(cache.block_shape)
    value = torch.randn(cache.block_shape)

    with pytest.raises(IndexError):
        cache.write_block(
            layer=0,
            block_id=99,
            key=key,
            value=value,
        )


def test_invalid_key_shape(cache):
    key = torch.randn(2, 16, 3)
    value = torch.randn(cache.block_shape)

    with pytest.raises(ValueError):
        cache.write_block(
            layer=0,
            block_id=0,
            key=key,
            value=value,
        )