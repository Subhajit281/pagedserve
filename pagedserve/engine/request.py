from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto


class RequestStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    PREEMPTED = auto()
    FINISHED = auto()
    FAILED = auto()


TERMINAL_STATUSES = {
    RequestStatus.FINISHED,
    RequestStatus.FAILED,
}


ALLOWED_TRANSITIONS = {
    RequestStatus.WAITING: {
        RequestStatus.RUNNING,
        RequestStatus.FAILED,
    },
    RequestStatus.RUNNING: {
        RequestStatus.PREEMPTED,
        RequestStatus.FINISHED,
        RequestStatus.FAILED,
    },
    RequestStatus.PREEMPTED: {
        RequestStatus.RUNNING,
        RequestStatus.FAILED,
    },
    RequestStatus.FINISHED: set(),
    RequestStatus.FAILED: set(),
}


@dataclass
class Request:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int

    generated_token_ids: list[int] = field(default_factory=list)
    status: RequestStatus = RequestStatus.WAITING

    arrival_time: float = field(default_factory=time.monotonic)
    last_scheduled_time: float | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")

        if not self.prompt_token_ids:
            raise ValueError("prompt_token_ids cannot be empty")

        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative")

        if len(self.generated_token_ids) > self.max_new_tokens:
            raise ValueError("Generated tokens exceed max_new_tokens")

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_generated_tokens

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def remaining_tokens(self) -> int:
        return self.max_new_tokens - self.num_generated_tokens

    def transition(self, new_status: RequestStatus) -> None:
        if new_status not in ALLOWED_TRANSITIONS[self.status]:
            raise ValueError(
                f"Invalid request transition: "
                f"{self.status.name} -> {new_status.name}"
            )

        self.status = new_status

        if new_status == RequestStatus.RUNNING:
            self.last_scheduled_time = time.monotonic()

    def add_generated_token(self, token_id: int) -> None:
        if self.status != RequestStatus.RUNNING:
            raise RuntimeError(
                "Tokens can only be generated for a RUNNING request"
            )

        if self.remaining_tokens <= 0:
            raise RuntimeError("Request has reached its generation limit")

        self.generated_token_ids.append(token_id)

    def finish(self) -> None:
        self.transition(RequestStatus.FINISHED)

    def fail(self, error: str) -> None:
        if not error:
            raise ValueError("An error message is required")

        self.error = error
        self.transition(RequestStatus.FAILED)
