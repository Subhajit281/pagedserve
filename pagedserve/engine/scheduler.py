from __future__ import annotations

from collections import deque

from pagedserve.engine.request import Request, RequestStatus
from pagedserve.engine.paged_kv_cache import PagedKVCache


class Scheduler:
    """
    FIFO request scheduler with optional KV-cache admission control.

    When kv_cache is supplied, requests reserve enough blocks for
    their prompt and maximum output length before entering RUNNING.

    This component does not execute model inference.
    """

    def __init__(
        self,
        max_running_requests: int = 8,
        kv_cache: PagedKVCache | None = None,
    ):
        if max_running_requests <= 0:
            raise ValueError("max_running_requests must be positive")

        self.max_running_requests = max_running_requests
        self.kv_cache = kv_cache

        self._waiting: deque[Request] = deque()
        self._running: dict[str, Request] = {}
        self._requests: dict[str, Request] = {}

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def requests(self) -> dict[str, Request]:
        return dict(self._requests)

    @property
    def waiting_requests(self) -> list[Request]:
        return list(self._waiting)

    @property
    def running_requests(self) -> list[Request]:
        return list(self._running.values())

    def submit(self, request: Request) -> None:
        if request.request_id in self._requests:
            raise ValueError(
                f"Duplicate request ID: {request.request_id!r}"
            )

        if request.status != RequestStatus.WAITING:
            raise ValueError(
                "Only requests in WAITING state can be submitted"
            )

        self._requests[request.request_id] = request
        self._waiting.append(request)

    def schedule(self) -> list[Request]:
        """
        Admit FIFO requests while running capacity and KV blocks permit.

        A request remains queued if its cache reservation cannot fit.
        """
        admitted = []

        while (
            self._waiting
            and self.running_count < self.max_running_requests
        ):
            request = self._waiting[0]

            if self.kv_cache is not None:
                capacity = (
                    request.num_prompt_tokens + request.max_new_tokens
                )
                blocks_needed = (
                    self.kv_cache.block_manager.blocks_needed(capacity)
                )

                # Strict FIFO: do not skip the queue head.
                if blocks_needed > self.kv_cache.free_blocks:
                    break

                try:
                    self.kv_cache.allocate(
                        request.request_id,
                        capacity,
                    )
                except RuntimeError:
                    # Allocation may fail if capacity changed unexpectedly.
                    # Leave the request queued and its state unchanged.
                    break

            self._waiting.popleft()
            request.transition(RequestStatus.RUNNING)
            self._running[request.request_id] = request
            admitted.append(request)

        return admitted

    def complete(self, request_id: str) -> Request:
        request = self._get_running(request_id)

        # Release cache before changing request state. If cleanup fails,
        # keep the request registered as running for diagnosis/retry.
        self._release_cache(request_id)

        request.finish()
        del self._running[request_id]
        return request

    def fail(self, request_id: str, error: str) -> Request:
        request = self._get_running(request_id)

        self._release_cache(request_id)

        request.fail(error)
        del self._running[request_id]
        return request

    def preempt(self, request_id: str) -> Request:
        """
        Preempt a running request and release its KV-cache allocation.

        Resuming the request requires the inference layer to recompute
        its prompt and generated-token history into a fresh cache.
        """
        request = self._get_running(request_id)

        self._release_cache(request_id)

        request.transition(RequestStatus.PREEMPTED)
        del self._running[request_id]
        return request

    def resume(self, request_id: str) -> None:
        """Requeue a preempted request at the front of the FIFO queue."""
        request = self._get_registered(request_id)

        if request.status != RequestStatus.PREEMPTED:
            raise ValueError(
                f"Cannot resume request {request_id!r}: "
                f"status is {request.status.name}"
            )

        self._waiting.appendleft(request)

    def cancel(
        self,
        request_id: str,
        error: str = "Request cancelled",
    ) -> Request:
        request = self._get_registered(request_id)

        if request.is_terminal:
            raise ValueError(
                f"Cannot cancel terminal request {request_id!r}"
            )

        if not error:
            raise ValueError("An error message is required")

        if request.status == RequestStatus.RUNNING:
            self._release_cache(request_id)
            del self._running[request_id]

        elif request.status == RequestStatus.WAITING:
            self._remove_waiting(request_id)

        # PREEMPTED requests have already released their cache.
        request.fail(error)
        return request

    def _release_cache(self, request_id: str) -> None:
        if (
            self.kv_cache is not None
            and self.kv_cache.block_manager.has_request(request_id)
        ):
            self.kv_cache.free(request_id)

    def _get_registered(self, request_id: str) -> Request:
        try:
            return self._requests[request_id]
        except KeyError:
            raise KeyError(
                f"Unknown request: {request_id!r}"
            ) from None

    def _get_running(self, request_id: str) -> Request:
        try:
            return self._running[request_id]
        except KeyError:
            raise KeyError(
                f"Request {request_id!r} is not running"
            ) from None

    def _remove_waiting(self, request_id: str) -> None:
        for request in self._waiting:
            if request.request_id == request_id:
                self._waiting.remove(request)
                return

        raise RuntimeError(
            f"Waiting request {request_id!r} is missing from the queue"
        )
