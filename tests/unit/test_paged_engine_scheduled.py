import pytest
import torch
from types import SimpleNamespace

from pagedserve.model.engine import PagedServeEngine
from pagedserve.engine.paged_kv_cache import PagedKVCache


class FakeTokenizer:
    def __init__(self):
        self.mapping = {
            "alpha": [3],
            "beta": [4],
            "gamma": [5],
        }

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = self.mapping[text]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(str(int(token)) for token in ids)


def make_engine(monkeypatch, *, eos_token_id=7, fail_prefill=False):
    engine = PagedServeEngine.__new__(PagedServeEngine)
    engine.device = torch.device("cpu")
    engine.tokenizer = FakeTokenizer()
    engine.model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=64),
        generation_config=SimpleNamespace(eos_token_id=eos_token_id),
    )
    engine.cache = PagedKVCache(
        num_blocks=32,
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        block_size=4,
        dtype=torch.float32,
        device="cpu",
    )

    engine._scheduled_prefill_ids = {}
    engine._scheduled_batch_sizes = []

    def fake_forward_paged(input_ids, request_id, position_start=0, **kwargs):
        if fail_prefill:
            raise RuntimeError("synthetic prefill failure")

        token = int(input_ids[0, 0].item())
        engine._scheduled_prefill_ids[request_id] = token

        # beta emits EOS immediately; other prompts emit token 2.
        next_token = 7 if token == 4 else 2
        logits = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1], 8),
            dtype=torch.float32,
        )
        logits[:, -1, next_token] = 10.0
        return logits

    def fake_forward_paged_batch(input_ids, request_ids, positions, **kwargs):
        batch_size = len(request_ids)
        engine._scheduled_batch_sizes.append(batch_size)

        logits = torch.zeros((batch_size, 1, 8), dtype=torch.float32)
        for row, request_id in enumerate(request_ids):
            prompt_token = engine._scheduled_prefill_ids[request_id]
            next_token = 7 if prompt_token == 4 else 2
            logits[row, 0, next_token] = 10.0
        return logits

    engine._forward_paged = fake_forward_paged
    engine._forward_paged_batch = fake_forward_paged_batch
    return engine


def test_scheduled_generation_preserves_input_order_and_releases_cache(monkeypatch):
    engine = make_engine(monkeypatch)

    results = engine.generate_scheduled(
        ["alpha", "beta", "gamma"],
        max_new_tokens=2,
        max_running_requests=2,
    )

    assert len(results) == 3
    assert results[0].startswith("3 ")
    assert results[1] == "4 7"
    assert results[2].startswith("5 ")
    assert engine.cache.used_blocks == 0


def test_scheduled_generation_respects_running_request_limit(monkeypatch):
    engine = make_engine(monkeypatch)

    results = engine.generate_scheduled(
        ["alpha", "gamma"],
        max_new_tokens=3,
        max_running_requests=1,
    )

    assert len(results) == 2
    assert engine._scheduled_batch_sizes
    assert max(engine._scheduled_batch_sizes) <= 1
    assert engine.cache.used_blocks == 0


def test_scheduled_generation_zero_tokens_returns_prompts(monkeypatch):
    engine = make_engine(monkeypatch)

    results = engine.generate_scheduled(
        ["alpha", "beta"],
        max_new_tokens=0,
        max_running_requests=1,
    )

    assert results == ["3", "4"]
    assert engine.cache.used_blocks == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": -1},
        {"max_running_requests": 0},
    ],
)
def test_scheduled_generation_rejects_invalid_limits(monkeypatch, kwargs):
    engine = make_engine(monkeypatch)

    with pytest.raises(ValueError):
        engine.generate_scheduled(["alpha"], **kwargs)

    assert engine.cache.used_blocks == 0


def test_scheduled_generation_rejects_non_string_prompt(monkeypatch):
    engine = make_engine(monkeypatch)

    with pytest.raises((TypeError, ValueError)):
        engine.generate_scheduled([123], max_new_tokens=2)

    assert engine.cache.used_blocks == 0


def test_scheduled_generation_releases_cache_after_prefill_failure(monkeypatch):
    engine = make_engine(monkeypatch, fail_prefill=True)

    with pytest.raises(RuntimeError, match="synthetic prefill failure"):
        engine.generate_scheduled(
            ["alpha", "gamma"],
            max_new_tokens=2,
            max_running_requests=2,
        )

    assert engine.cache.used_blocks == 0
