from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from pagedserve.server.app import create_app


class FakeBatcher:
    """Deterministic stand-in for the real inference batcher."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counter = 0
        self._results: dict[str, str] = {}
        self.closed = False

    def submit(self, prompt: str, max_new_tokens: int = 32) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")

        with self._lock:
            self._counter += 1
            request_id = f"fake-{self._counter}"
            self._results[request_id] = (
                prompt[:max_new_tokens] if max_new_tokens else ""
            )
            return request_id

    def result(self, request_id: str, timeout: float | None = None) -> str:
        with self._lock:
            if request_id not in self._results:
                raise KeyError(f"Unknown request: {request_id!r}")
            return self._results[request_id]

    def cancel(self, request_id: str) -> None:
        with self._lock:
            if request_id not in self._results:
                raise KeyError(f"Unknown request: {request_id!r}")
            raise ValueError("Cannot cancel a completed request")

    def close(self) -> None:
        self.closed = True


def test_health_is_ready_with_injected_batcher():
    batcher = FakeBatcher()

    with TestClient(create_app(batcher=batcher)) as client:
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "service": "pagedserve",
        }

    # The application must not close a batcher supplied by its caller.
    assert batcher.closed is False


def test_submit_then_poll_completed_request():
    with TestClient(create_app(batcher=FakeBatcher())) as client:
        submitted = client.post(
            "/v1/requests",
            json={"prompt": "hello world", "max_new_tokens": 5},
        )

        assert submitted.status_code == 202
        request_id = submitted.json()["request_id"]

        response = client.get(f"/v1/requests/{request_id}")

        assert response.status_code == 200
        assert response.json() == {
            "request_id": request_id,
            "status": "completed",
            "text": "hello",
        }


def test_generate_convenience_endpoint():
    with TestClient(create_app(batcher=FakeBatcher())) as client:
        response = client.post(
            "/v1/generate",
            json={"prompt": "pagedserve", "max_new_tokens": 4},
        )

        assert response.status_code == 200
        assert response.json()["text"] == "page"
        assert response.json()["request_id"].startswith("fake-")


def test_empty_prompt_is_rejected():
    with TestClient(create_app(batcher=FakeBatcher())) as client:
        response = client.post(
            "/v1/requests",
            json={"prompt": "", "max_new_tokens": 4},
        )

        assert response.status_code == 422


def test_negative_token_count_is_rejected():
    with TestClient(create_app(batcher=FakeBatcher())) as client:
        response = client.post(
            "/v1/requests",
            json={"prompt": "hello", "max_new_tokens": -1},
        )

        assert response.status_code == 422


def test_unknown_request_returns_404():
    with TestClient(create_app(batcher=FakeBatcher())) as client:
        response = client.get("/v1/requests/missing")

        assert response.status_code == 404


def test_factories_run_once_and_owned_batcher_is_closed():
    batcher = FakeBatcher()
    engine_calls = []
    batcher_calls = []

    def engine_factory():
        engine_calls.append(True)
        return object()

    def batcher_factory(engine):
        batcher_calls.append(engine)
        return batcher

    application = create_app(
        engine_factory=engine_factory,
        batcher_factory=batcher_factory,
    )

    with TestClient(application) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/healthz").json()["status"] == "ok"

    assert len(engine_calls) == 1
    assert len(batcher_calls) == 1
    assert batcher.closed is True
