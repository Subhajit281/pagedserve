import pytest
import torch

from pagedserve.engine.paged_kv_cache import PagedKVCache
from pagedserve.engine.request import Request, RequestStatus
from pagedserve.engine.scheduler import Scheduler


def make_cache(num_blocks=4, block_size=4):
    return PagedKVCache(
        num_blocks=num_blocks,
        num_layers=1,
        num_kv_heads=1,
        head_dim=2,
        block_size=block_size,
        dtype=torch.float32,
        device="cpu",
    )


def make_request(request_id, prompt_length=3, max_new_tokens=2):
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_length)),
        max_new_tokens=max_new_tokens,
    )


def test_admission_allocates_capacity_and_completion_frees_it():
    cache = make_cache()
    scheduler = Scheduler(max_running_requests=2, kv_cache=cache)
    request = make_request("r1", prompt_length=3, max_new_tokens=2)

    scheduler.submit(request)
    assert scheduler.schedule() == [request]

    assert request.status == RequestStatus.RUNNING
    assert cache.token_count("r1") == 5
    assert cache.used_blocks == 2

    scheduler.complete("r1")

    assert request.status == RequestStatus.FINISHED
    assert cache.used_blocks == 0
    assert cache.free_blocks == cache.num_blocks


def test_request_waits_when_cache_capacity_is_insufficient():
    cache = make_cache(num_blocks=2, block_size=4)
    scheduler = Scheduler(kv_cache=cache)

    # Needs 3 blocks: ceil((9 + 1) / 4).
    request = make_request("large", prompt_length=9, max_new_tokens=1)
    scheduler.submit(request)

    assert scheduler.schedule() == []
    assert request.status == RequestStatus.WAITING
    assert scheduler.waiting_count == 1
    assert scheduler.running_count == 0
    assert cache.used_blocks == 0


def test_fifo_head_blocks_later_request_that_would_fit():
    cache = make_cache(num_blocks=2, block_size=4)
    scheduler = Scheduler(kv_cache=cache)

    too_large = make_request("large", prompt_length=9, max_new_tokens=1)
    small = make_request("small", prompt_length=2, max_new_tokens=1)

    scheduler.submit(too_large)
    scheduler.submit(small)

    assert scheduler.schedule() == []
    assert scheduler.waiting_requests == [too_large, small]
    assert cache.used_blocks == 0


def test_failure_releases_cache():
    cache = make_cache()
    scheduler = Scheduler(kv_cache=cache)
    request = make_request("r1")

    scheduler.submit(request)
    scheduler.schedule()
    scheduler.fail("r1", "inference failed")

    assert request.status == RequestStatus.FAILED
    assert request.error == "inference failed"
    assert cache.used_blocks == 0


def test_running_cancellation_releases_cache():
    cache = make_cache()
    scheduler = Scheduler(kv_cache=cache)
    request = make_request("r1")

    scheduler.submit(request)
    scheduler.schedule()
    assert cache.used_blocks > 0

    scheduler.cancel("r1", "client disconnected")

    assert request.status == RequestStatus.FAILED
    assert cache.used_blocks == 0
    assert scheduler.running_count == 0


def test_preemption_releases_cache_and_resume_reallocates():
    cache = make_cache()
    scheduler = Scheduler(max_running_requests=1, kv_cache=cache)
    request = make_request("r1")

    scheduler.submit(request)
    scheduler.schedule()
    request.add_generated_token(99)

    scheduler.preempt("r1")

    assert request.status == RequestStatus.PREEMPTED
    assert cache.used_blocks == 0

    scheduler.resume("r1")
    assert request.status == RequestStatus.PREEMPTED

    assert scheduler.schedule() == [request]
    assert request.status == RequestStatus.RUNNING
    assert cache.used_blocks == 2
    assert request.generated_token_ids == [99]


def test_cancelling_waiting_request_does_not_allocate_cache():
    cache = make_cache()
    scheduler = Scheduler(kv_cache=cache)
    request = make_request("r1")

    scheduler.submit(request)
    scheduler.cancel("r1")

    assert request.status == RequestStatus.FAILED
    assert scheduler.waiting_count == 0
    assert cache.used_blocks == 0


def test_running_capacity_is_respected_with_cache():
    cache = make_cache(num_blocks=8, block_size=4)
    scheduler = Scheduler(max_running_requests=1, kv_cache=cache)

    first = make_request("r1")
    second = make_request("r2")

    scheduler.submit(first)
    scheduler.submit(second)

    assert scheduler.schedule() == [first]
    assert scheduler.running_count == 1
    assert scheduler.waiting_requests == [second]

    scheduler.complete("r1")
    assert scheduler.schedule() == [second]
    assert scheduler.running_count == 1


def test_cache_reservation_uses_prompt_plus_maximum_output():
    cache = make_cache(num_blocks=3, block_size=4)
    scheduler = Scheduler(kv_cache=cache)

    # 5 prompt + 4 output = 9 tokens, requiring 3 blocks.
    request = make_request("r1", prompt_length=5, max_new_tokens=4)
    scheduler.submit(request)

    assert scheduler.schedule() == [request]
    assert cache.token_count("r1") == 9
    assert cache.used_blocks == 3

    scheduler.complete("r1")
    assert cache.used_blocks == 0


def test_resumed_request_that_cannot_fit_remains_preempted_and_queued():
    cache = make_cache(num_blocks=2, block_size=4)
    scheduler = Scheduler(kv_cache=cache)

    request = make_request("r1", prompt_length=9, max_new_tokens=1)
    request.transition(RequestStatus.RUNNING)
    request.transition(RequestStatus.PREEMPTED)

    scheduler._requests[request.request_id] = request
    scheduler.resume("r1")

    assert scheduler.schedule() == []
    assert request.status == RequestStatus.PREEMPTED
    assert scheduler.waiting_count == 1
    assert cache.used_blocks == 0
