from __future__ import annotations

import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from pagedserve.engine.request import Request, RequestStatus
from pagedserve.engine.scheduler import Scheduler


class RequestCancelledError(RuntimeError):
    """Raised when a submitted generation request is cancelled."""


@dataclass
class _Job:
    request_id: str
    prompt: str
    max_new_tokens: int
    done: threading.Event = field(default_factory=threading.Event)
    result: str | None = None
    error: BaseException | None = None
    request: Request | None = None


class ContinuousBatcher:
    """Thread-backed online request queue for a PagedServeEngine.

    Requests can be submitted while other requests are decoding. Model
    inference is serialized on one worker thread; active decode steps are
    combined into a single batched forward pass.
    """

    def __init__(
        self,
        engine: Any,
        max_running_requests: int = 8,
        poll_interval: float = 0.01,
    ) -> None:
        if max_running_requests <= 0:
            raise ValueError("max_running_requests must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")

        self.engine = engine
        self.max_running_requests = max_running_requests
        self.poll_interval = poll_interval

        self._commands: queue.Queue[tuple] = queue.Queue()
        self._jobs: dict[str, _Job] = {}
        self._jobs_lock = threading.Lock()
        self._closed = False
        self._close_lock = threading.Lock()

        self._worker = threading.Thread(
            target=self._run,
            name="pagedserve-continuous-batcher",
            daemon=True,
        )
        self._worker.start()

    def submit(self, prompt: str, max_new_tokens: int = 32) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")

        with self._close_lock:
            if self._closed:
                raise RuntimeError("ContinuousBatcher is closed")

            request_id = f"online-{uuid.uuid4().hex}"
            job = _Job(request_id, prompt, max_new_tokens)

            with self._jobs_lock:
                self._jobs[request_id] = job

            self._commands.put(("submit", job))
            return request_id

    def result(self, request_id: str, timeout: float | None = None) -> str:
        with self._jobs_lock:
            job = self._jobs.get(request_id)

        if job is None:
            raise KeyError(f"Unknown request: {request_id!r}")

        if not job.done.wait(timeout):
            raise TimeoutError(f"Request {request_id!r} is not complete")

        if job.error is not None:
            raise job.error
        if job.result is None:
            raise RuntimeError("Request completed without a result")
        return job.result

    def cancel(self, request_id: str) -> None:
        with self._jobs_lock:
            job = self._jobs.get(request_id)

        if job is None:
            raise KeyError(f"Unknown request: {request_id!r}")
        if job.done.is_set():
            raise ValueError("Cannot cancel a completed request")

        self._commands.put(("cancel", request_id))

    def close(self, wait: bool = True) -> None:
        with self._close_lock:
            if not self._closed:
                self._closed = True
                self._commands.put(("stop",))

        if wait and threading.current_thread() is not self._worker:
            self._worker.join()

    def __enter__(self) -> ContinuousBatcher:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _finish_error(self, job: _Job, error: BaseException) -> None:
        job.error = error
        job.done.set()

    def _run(self) -> None:
        scheduler = Scheduler(
            max_running_requests=self.max_running_requests,
            kv_cache=self.engine.cache,
        )
        jobs: dict[str, _Job] = {}
        inputs: dict[str, torch.Tensor] = {}
        next_tokens: dict[str, torch.Tensor] = {}
        stop_requested = False

        eos = self.engine.model.generation_config.eos_token_id
        eos_ids = (
            set() if eos is None
            else set(eos) if isinstance(eos, (list, tuple, set))
            else {eos}
        )
        max_positions = getattr(
            self.engine.model.config, "max_position_embeddings", 32768
        )

        def fail_request(request_id: str, error: BaseException) -> None:
            job = jobs.get(request_id)
            if job is None or job.done.is_set():
                return

            request = job.request
            try:
                if request is not None and not request.is_terminal:
                    if request.status == RequestStatus.RUNNING:
                        scheduler.fail(request_id, str(error) or type(error).__name__)
                    else:
                        scheduler.cancel(request_id, str(error) or type(error).__name__)
            except Exception as cleanup_error:
                error = RuntimeError(
                    f"{error}; additionally failed to clean up request: "
                    f"{cleanup_error}"
                )

            next_tokens.pop(request_id, None)
            inputs.pop(request_id, None)
            self._finish_error(job, error)

        def process_commands() -> None:
            nonlocal stop_requested

            while True:
                try:
                    command = self._commands.get_nowait()
                except queue.Empty:
                    return

                kind = command[0]

                if kind == "stop":
                    stop_requested = True
                    continue

                if kind == "cancel":
                    request_id = command[1]
                    job = jobs.get(request_id)
                    if job is None or job.done.is_set():
                        continue

                    try:
                        if job.request is None:
                            self._finish_error(
                                job, RequestCancelledError("Request cancelled")
                            )
                        else:
                            scheduler.cancel(request_id)
                            next_tokens.pop(request_id, None)
                            inputs.pop(request_id, None)
                            self._finish_error(
                                job, RequestCancelledError("Request cancelled")
                            )
                    except Exception as error:
                        fail_request(request_id, error)
                    continue

                if kind == "submit":
                    job = command[1]
                    jobs[job.request_id] = job

                    try:
                        encoded = self.engine.tokenizer(
                            job.prompt,
                            return_tensors="pt",
                            add_special_tokens=True,
                        )
                        input_ids = encoded["input_ids"].to(self.engine.device)

                        if input_ids.shape[1] == 0:
                            raise ValueError("Prompt must contain at least one token")

                        if input_ids.shape[1] + job.max_new_tokens > max_positions:
                            raise ValueError(
                                "Prompt plus requested output exceeds the model's "
                                f"maximum position count ({max_positions})"
                            )

                        if job.max_new_tokens == 0:
                            job.result = self.engine.tokenizer.decode(
                                input_ids[0], skip_special_tokens=True
                            )
                            job.done.set()
                            continue

                        request = Request(
                            request_id=job.request_id,
                            prompt_token_ids=input_ids[0].tolist(),
                            max_new_tokens=job.max_new_tokens,
                        )
                        job.request = request
                        inputs[job.request_id] = input_ids
                        scheduler.submit(request)
                    except Exception as error:
                        self._finish_error(job, error)

        try:
            while True:
                process_commands()

                if stop_requested:
                    for request_id, job in list(jobs.items()):
                        if not job.done.is_set():
                            fail_request(
                                request_id,
                                RequestCancelledError(
                                    "Batcher shut down before request completed"
                                ),
                            )
                    break

                admitted = scheduler.schedule()

                # Prefill newly admitted requests independently.
                for request in admitted:
                    rid = request.request_id
                    job = jobs[rid]
                    try:
                        with torch.inference_mode():
                            logits = self.engine._forward_paged(
                                input_ids=inputs[rid],
                                request_id=rid,
                                position_start=0,
                            )
                        token = logits[:, -1, :].argmax(
                            dim=-1, keepdim=True
                        )
                        token_id = int(token.item())
                        request.add_generated_token(token_id)

                        if token_id in eos_ids or request.remaining_tokens == 0:
                            scheduler.complete(rid)
                            job.result = self.engine.tokenizer.decode(
                                request.prompt_token_ids + request.generated_token_ids,
                                skip_special_tokens=True,
                            )
                            job.done.set()
                            inputs.pop(rid, None)
                        else:
                            next_tokens[rid] = token
                    except Exception as error:
                        fail_request(rid, error)

                # Drain arrivals/cancellations before forming the decode batch.
                process_commands()
                if stop_requested:
                    continue

                active = [
                    request for request in scheduler.running_requests
                    if request.request_id in next_tokens
                ]

                if active:
                    active_ids = [request.request_id for request in active]
                    batch_ids = torch.cat(
                        [next_tokens[rid] for rid in active_ids], dim=0
                    )
                    positions = [
                        request.num_prompt_tokens + request.num_generated_tokens - 1
                        for request in active
                    ]

                    try:
                        with torch.inference_mode():
                            logits = self.engine._forward_paged_batch(
                                input_ids=batch_ids,
                                request_ids=active_ids,
                                positions=positions,
                            )

                        # Process cancellations and shutdown requests that
                        # arrived while the batched forward pass was running.
                        # Cancelled requests must not consume this batch's
                        # output token.
                        process_commands()
                        if stop_requested:
                            continue

                        tokens = logits[:, -1, :].argmax(
                            dim=-1, keepdim=True
                        )

                        for row, request in enumerate(active):
                            rid = request.request_id
                            if rid not in next_tokens:
                                continue

                            token = tokens[row:row + 1]
                            token_id = int(token.item())
                            request.add_generated_token(token_id)

                            if (
                                token_id in eos_ids
                                or request.remaining_tokens == 0
                            ):
                                scheduler.complete(rid)
                                next_tokens.pop(rid, None)
                                inputs.pop(rid, None)
                                jobs[rid].result = self.engine.tokenizer.decode(
                                    request.prompt_token_ids
                                    + request.generated_token_ids,
                                    skip_special_tokens=True,
                                )
                                jobs[rid].done.set()
                            else:
                                next_tokens[rid] = token
                    except Exception as error:
                        # A failed batched forward affects every request in it.
                        for request in active:
                            fail_request(request.request_id, error)

                # Reject only requests that cannot fit even when the cache
                # is completely free. Temporary shortages must remain queued.
                if scheduler.waiting_count:
                    cache = scheduler.kv_cache
                    if cache is not None:
                        impossible = []
                        for request in scheduler.waiting_requests:
                            capacity = (
                                request.num_prompt_tokens
                                + request.max_new_tokens
                            )
                            blocks_needed = (
                                cache.block_manager.blocks_needed(capacity)
                            )
                            if blocks_needed > cache.num_blocks:
                                impossible.append(request)

                        for request in impossible:
                            fail_request(
                                request.request_id,
                                RuntimeError(
                                    "Request exceeds total KV-cache capacity"
                                ),
                            )

                if not jobs or all(job.done.is_set() for job in jobs.values()):
                    # Keep the worker alive for future arrivals.
                    try:
                        command = self._commands.get(timeout=self.poll_interval)
                    except queue.Empty:
                        continue
                    self._commands.put(command)
                else:
                    time.sleep(self.poll_interval)

        except BaseException as error:
            for request_id, job in list(jobs.items()):
                if not job.done.is_set():
                    fail_request(request_id, error)
        finally:
            # Final defensive cleanup if shutdown or an unexpected error
            # leaves any scheduler-owned cache allocations.
            for request in list(scheduler.running_requests):
                try:
                    scheduler.fail(request.request_id, "Batcher stopped")
                except Exception:
                    pass
            for request in scheduler.waiting_requests:
                try:
                    scheduler.cancel(request.request_id, "Batcher stopped")
                except Exception:
                    pass
