import pytest

from pagedserve.engine.request import Request, RequestStatus
from pagedserve.engine.scheduler import Scheduler


def make_request(request_id):
    return Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        max_new_tokens=5,
    )


def test_submit_adds_request_to_waiting_queue():
    scheduler = Scheduler()

    request = make_request("r1")
    scheduler.submit(request)

    assert scheduler.waiting_count == 1
    assert scheduler.running_count == 0
    assert scheduler.waiting_requests == [request]


def test_fifo_scheduling_respects_capacity():
    scheduler = Scheduler(max_running_requests=2)

    requests = [make_request(f"r{i}") for i in range(3)]

    for request in requests:
        scheduler.submit(request)

    admitted = scheduler.schedule()

    assert [r.request_id for r in admitted] == ["r0", "r1"]
    assert scheduler.running_count == 2
    assert scheduler.waiting_count == 1


def test_schedule_admits_more_after_completion():
    scheduler = Scheduler(max_running_requests=1)

    first = make_request("r1")
    second = make_request("r2")

    scheduler.submit(first)
    scheduler.submit(second)

    assert scheduler.schedule() == [first]

    scheduler.complete("r1")

    assert scheduler.schedule() == [second]
    assert second.status == RequestStatus.RUNNING


def test_duplicate_request_ids_are_rejected():
    scheduler = Scheduler()

    scheduler.submit(make_request("r1"))

    with pytest.raises(ValueError, match="Duplicate request ID"):
        scheduler.submit(make_request("r1"))


def test_non_waiting_request_cannot_be_submitted():
    scheduler = Scheduler()
    request = make_request("r1")
    request.transition(RequestStatus.RUNNING)

    with pytest.raises(ValueError, match="WAITING"):
        scheduler.submit(request)


def test_completion_releases_running_slot():
    scheduler = Scheduler(max_running_requests=1)
    request = make_request("r1")

    scheduler.submit(request)
    scheduler.schedule()
    result = scheduler.complete("r1")

    assert result.status == RequestStatus.FINISHED
    assert scheduler.running_count == 0

    with pytest.raises(KeyError, match="not running"):
        scheduler.complete("r1")


def test_failure_releases_running_slot():
    scheduler = Scheduler()
    request = make_request("r1")

    scheduler.submit(request)
    scheduler.schedule()

    scheduler.fail("r1", "inference failed")

    assert request.status == RequestStatus.FAILED
    assert request.error == "inference failed"
    assert scheduler.running_count == 0


def test_preempted_request_can_be_resumed():
    scheduler = Scheduler(max_running_requests=1)

    first = make_request("r1")
    second = make_request("r2")

    scheduler.submit(first)
    scheduler.submit(second)
    scheduler.schedule()

    first.add_generated_token(10)
    scheduler.preempt("r1")

    assert first.status == RequestStatus.PREEMPTED
    assert scheduler.running_count == 0

    scheduler.resume("r1")

    assert scheduler.waiting_requests[0] == first
    assert first.status == RequestStatus.PREEMPTED

    admitted = scheduler.schedule()

    assert admitted == [first]
    assert first.status == RequestStatus.RUNNING
    assert first.generated_token_ids == [10]


def test_cancelling_waiting_request_removes_it_from_queue():
    scheduler = Scheduler()

    first = make_request("r1")
    second = make_request("r2")

    scheduler.submit(first)
    scheduler.submit(second)

    scheduler.cancel("r1")

    assert first.status == RequestStatus.FAILED
    assert scheduler.waiting_requests == [second]


def test_cancelling_running_request_releases_slot():
    scheduler = Scheduler()

    request = make_request("r1")
    scheduler.submit(request)
    scheduler.schedule()

    scheduler.cancel("r1", "client disconnected")

    assert request.status == RequestStatus.FAILED
    assert request.error == "client disconnected"
    assert scheduler.running_count == 0


def test_unknown_request_is_rejected():
    scheduler = Scheduler()

    with pytest.raises(KeyError, match="Unknown request"):
        scheduler.resume("missing")


def test_invalid_capacity_is_rejected():
    with pytest.raises(ValueError, match="must be positive"):
        Scheduler(max_running_requests=0)


def test_running_request_cannot_be_resumed():
    scheduler = Scheduler()
    request = make_request("r1")

    scheduler.submit(request)
    scheduler.schedule()

    with pytest.raises(ValueError, match="status is RUNNING"):
        scheduler.resume("r1")


def test_terminal_request_cannot_be_cancelled():
    scheduler = Scheduler()
    request = make_request("r1")

    scheduler.submit(request)
    scheduler.schedule()
    scheduler.complete("r1")

    with pytest.raises(ValueError, match="terminal request"):
        scheduler.cancel("r1")
