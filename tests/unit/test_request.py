import pytest

from pagedserve.engine.request import Request, RequestStatus


def make_request():
    return Request(
        request_id="req-1",
        prompt_token_ids=[10, 20, 30],
        max_new_tokens=3,
    )


def test_request_initial_state():
    request = make_request()

    assert request.status == RequestStatus.WAITING
    assert request.num_prompt_tokens == 3
    assert request.num_generated_tokens == 0
    assert request.total_tokens == 3
    assert request.remaining_tokens == 3
    assert not request.is_terminal


def test_request_normal_lifecycle():
    request = make_request()

    request.transition(RequestStatus.RUNNING)
    request.add_generated_token(40)
    request.add_generated_token(50)

    assert request.num_generated_tokens == 2
    assert request.total_tokens == 5
    assert request.remaining_tokens == 1
    assert request.last_scheduled_time is not None

    request.finish()

    assert request.status == RequestStatus.FINISHED
    assert request.is_terminal


def test_request_can_be_preempted_and_resumed():
    request = make_request()

    request.transition(RequestStatus.RUNNING)
    request.add_generated_token(40)
    request.transition(RequestStatus.PREEMPTED)

    assert request.status == RequestStatus.PREEMPTED

    request.transition(RequestStatus.RUNNING)
    request.add_generated_token(50)

    assert request.status == RequestStatus.RUNNING
    assert request.generated_token_ids == [40, 50]


def test_invalid_transition_is_rejected():
    request = make_request()

    with pytest.raises(ValueError, match="Invalid request transition"):
        request.transition(RequestStatus.FINISHED)


def test_terminal_request_cannot_resume():
    request = make_request()
    request.transition(RequestStatus.RUNNING)
    request.finish()

    with pytest.raises(ValueError, match="Invalid request transition"):
        request.transition(RequestStatus.RUNNING)


def test_generation_limit_is_enforced():
    request = make_request()
    request.transition(RequestStatus.RUNNING)

    request.add_generated_token(40)
    request.add_generated_token(50)
    request.add_generated_token(60)

    assert request.remaining_tokens == 0

    with pytest.raises(RuntimeError, match="generation limit"):
        request.add_generated_token(70)


def test_tokens_cannot_be_generated_while_waiting():
    request = make_request()

    with pytest.raises(RuntimeError, match="RUNNING"):
        request.add_generated_token(40)


def test_failure_records_error():
    request = make_request()
    request.transition(RequestStatus.RUNNING)
    request.fail("CUDA allocation failed")

    assert request.status == RequestStatus.FAILED
    assert request.error == "CUDA allocation failed"
    assert request.is_terminal


@pytest.mark.parametrize(
    ("request_id", "prompt_ids", "max_new_tokens"),
    [
        ("", [1], 1),
        ("req", [], 1),
        ("req", [1], -1),
    ],
)
def test_invalid_request_arguments(
    request_id,
    prompt_ids,
    max_new_tokens,
):
    with pytest.raises(ValueError):
        Request(
            request_id=request_id,
            prompt_token_ids=prompt_ids,
            max_new_tokens=max_new_tokens,
        )
