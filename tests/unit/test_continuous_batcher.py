import threading
import time
from types import SimpleNamespace

import pytest
import torch

from pagedserve.engine.continuous_batcher import (
    ContinuousBatcher,
    RequestCancelledError,
)
from pagedserve.engine.paged_kv_cache import PagedKVCache


class FakeTokenizer:
    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        mapping = {"alpha": [3], "beta": [4], "gamma": [5]}
        ids = mapping.get(text, [6])
        result = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": result}

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(str(int(token)) for token in ids)


class FakeEngine:
    def __init__(self, *, batch_hook=None, fail_batch=False, fail_prefill=False):
        self.device = torch.device("cpu")
        self.tokenizer = FakeTokenizer()
        self.model = SimpleNamespace(
            config=SimpleNamespace(max_position_embeddings=64),
            generation_config=SimpleNamespace(eos_token_id=7),
        )
        self.cache = PagedKVCache(
            num_blocks=64,
            num_layers=1,
            num_kv_heads=1,
            head_dim=2,
            block_size=4,
            dtype=torch.float32,
            device="cpu",
        )
        self.batch_sizes = []
        self.batch_hook = batch_hook
        self.fail_batch = fail_batch
        self.fail_prefill = fail_prefill

    def _logits(self, batch_size, token_id=2):
        logits = torch.zeros((batch_size, 1, 8), dtype=torch.float32)
        logits[:, 0, token_id] = 10.0
        return logits

    def _forward_paged(self, input_ids, request_id, position_start=0):
        if self.fail_prefill:
            raise RuntimeError("synthetic prefill failure")
        return self._logits(input_ids.shape[0])

    def _forward_paged_batch(self, input_ids, request_ids, positions):
        self.batch_sizes.append(len(request_ids))
        if self.batch_hook is not None:
            self.batch_hook(request_ids)
        if self.fail_batch:
            raise RuntimeError("synthetic batch failure")
        return self._logits(len(request_ids))


def wait_until(predicate, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return bool(predicate())


def test_online_requests_complete_and_worker_can_be_reused():
    engine = FakeEngine()
    with ContinuousBatcher(engine, max_running_requests=2) as batcher:
        first = batcher.submit("alpha", max_new_tokens=2)
        assert batcher.result(first, timeout=3).startswith("3 ")

        # The worker remains alive and accepts a later request after going idle.
        second = batcher.submit("beta", max_new_tokens=2)
        assert batcher.result(second, timeout=3).startswith("4 ")

    assert engine.cache.used_blocks == 0


def test_new_arrival_joins_active_decode_batch():
    first_batch_entered = threading.Event()
    release_first_batch = threading.Event()
    observed_batches = []

    def hook(request_ids):
        observed_batches.append(tuple(request_ids))
        if len(observed_batches) == 1:
            first_batch_entered.set()
            assert release_first_batch.wait(timeout=3)

    engine = FakeEngine(batch_hook=hook)
    batcher = ContinuousBatcher(
        engine, max_running_requests=3, poll_interval=0.002
    )

    try:
        first = batcher.submit("alpha", max_new_tokens=4)
        assert first_batch_entered.wait(timeout=3)

        second = batcher.submit("beta", max_new_tokens=4)
        release_first_batch.set()

        assert batcher.result(first, timeout=4).startswith("3 ")
        assert batcher.result(second, timeout=4).startswith("4 ")

        assert any(len(batch) >= 2 for batch in observed_batches)
    finally:
        release_first_batch.set()
        batcher.close()

    assert engine.cache.used_blocks == 0


def test_running_request_limit_is_respected():
    engine = FakeEngine()
    with ContinuousBatcher(engine, max_running_requests=1) as batcher:
        ids = [
            batcher.submit("alpha", max_new_tokens=3),
            batcher.submit("beta", max_new_tokens=3),
            batcher.submit("gamma", max_new_tokens=3),
        ]
        outputs = [batcher.result(rid, timeout=4) for rid in ids]

    assert len(outputs) == 3
    assert engine.batch_sizes
    assert max(engine.batch_sizes) <= 1
    assert engine.cache.used_blocks == 0


def test_cancelling_one_active_request_does_not_cancel_another():
    entered = threading.Event()
    release = threading.Event()
    first_call = [True]

    def hook(request_ids):
        if first_call[0]:
            first_call[0] = False
            entered.set()
            assert release.wait(timeout=3)

    engine = FakeEngine(batch_hook=hook)
    batcher = ContinuousBatcher(engine, max_running_requests=2)

    try:
        cancelled_id = batcher.submit("alpha", max_new_tokens=5)
        survivor_id = batcher.submit("beta", max_new_tokens=5)
        assert entered.wait(timeout=3)

        # Capture the cancelled request's token history while its first
        # batched forward pass is deliberately blocked.
        cancelled_job = batcher._jobs[cancelled_id]
        assert cancelled_job.request is not None
        tokens_before_cancel = list(
            cancelled_job.request.generated_token_ids
        )

        batcher.cancel(cancelled_id)
        release.set()

        with pytest.raises(RequestCancelledError):
            batcher.result(cancelled_id, timeout=4)

        # Cancellation must be terminal and must not append a token from
        # the in-flight batch after the cancellation command is processed.
        assert cancelled_job.done.is_set()
        assert cancelled_job.request.is_terminal
        assert cancelled_job.request.generated_token_ids == tokens_before_cancel

        # Cancellation of one request must not disrupt the other request.
        survivor_output = batcher.result(survivor_id, timeout=4)
        assert survivor_output.startswith("4 ")
        assert batcher._jobs[survivor_id].request.status.name == "FINISHED"
    finally:
        release.set()
        batcher.close()

    assert engine.cache.used_blocks == 0


def test_prefill_failure_is_reported_and_cache_is_released():
    engine = FakeEngine(fail_prefill=True)
    with ContinuousBatcher(engine, max_running_requests=2) as batcher:
        request_id = batcher.submit("alpha", max_new_tokens=3)

        with pytest.raises(RuntimeError, match="synthetic prefill failure"):
            batcher.result(request_id, timeout=3)

    assert engine.cache.used_blocks == 0


def test_batched_decode_failure_fails_active_requests_and_releases_cache():
    engine = FakeEngine(fail_batch=True)
    with ContinuousBatcher(engine, max_running_requests=2) as batcher:
        request_ids = [
            batcher.submit("alpha", max_new_tokens=4),
            batcher.submit("beta", max_new_tokens=4),
        ]

        for request_id in request_ids:
            with pytest.raises(RuntimeError, match="synthetic batch failure"):
                batcher.result(request_id, timeout=3)

    assert engine.cache.used_blocks == 0


def test_close_cancels_unfinished_requests_and_releases_cache():
    entered = threading.Event()
    release = threading.Event()

    def hook(request_ids):
        entered.set()
        release.wait(timeout=3)

    engine = FakeEngine(batch_hook=hook)
    batcher = ContinuousBatcher(engine, max_running_requests=1)

    request_id = batcher.submit("alpha", max_new_tokens=10)
    assert entered.wait(timeout=3)

    # Allow the in-flight forward pass to finish, then shut down the worker.
    release.set()
    batcher.close(wait=True)

    with pytest.raises((RequestCancelledError, RuntimeError)):
        batcher.result(request_id, timeout=1)

    assert engine.cache.used_blocks == 0


def test_invalid_submission_is_rejected():
    engine = FakeEngine()
    with ContinuousBatcher(engine) as batcher:
        with pytest.raises(TypeError):
            batcher.submit(123, max_new_tokens=2)
        with pytest.raises(ValueError):
            batcher.submit("alpha", max_new_tokens=-1)
