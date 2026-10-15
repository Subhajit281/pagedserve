import pytest
import torch

from pagedserve.engine.paged_kv_cache import PagedKVCache


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)


def test_paged_kv_cache_cuda_write_read():
    cache = PagedKVCache(
        num_blocks=8,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        block_size=16,
        dtype=torch.bfloat16,
        device="cuda",
    )

    cache.allocate("request-A", 17)

    assert cache.kv_cache.k_cache.is_cuda
    assert cache.kv_cache.v_cache.is_cuda
    assert cache.kv_cache.k_cache.dtype == torch.bfloat16
    assert cache.kv_cache.v_cache.dtype == torch.bfloat16

    key = torch.randn(
        2, 64,
        dtype=torch.bfloat16,
        device="cuda",
    )

    value = torch.randn(
        2, 64,
        dtype=torch.bfloat16,
        device="cuda",
    )

    cache.write_token(
        request_id="request-A",
        layer=0,
        token_position=16,
        key=key,
        value=value,
    )

    actual_key, actual_value = cache.read_token(
        request_id="request-A",
        layer=0,
        token_position=16,
    )

    torch.cuda.synchronize()

    assert torch.equal(actual_key, key)
    assert torch.equal(actual_value, value)

    assert cache.get_block_table("request-A").num_blocks == 2

    print(
        "\nCUDA integration passed:",
        torch.cuda.get_device_name(0),
    )
