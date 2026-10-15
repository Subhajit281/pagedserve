from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from pagedserve.engine.continuous_batcher import (
    ContinuousBatcher,
    RequestCancelledError,
)


class GenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=100_000)
    max_new_tokens: int = Field(default=32, ge=0, le=4096)


class SubmissionResponse(BaseModel):
    request_id: str
    status: str = "queued"


class RequestResultResponse(BaseModel):
    request_id: str
    status: str
    text: str | None = None


def create_app(
    engine: Any | None = None,
    batcher: Any | None = None,
    *,
    engine_factory: Callable[[], Any] | None = None,
    batcher_factory: Callable[[Any], Any] | None = None,
    max_running_requests: int = 8,
) -> FastAPI:
    """Create an API app with optional injectable inference dependencies."""
    if max_running_requests <= 0:
        raise ValueError("max_running_requests must be positive")

    if batcher is not None and engine is not None:
        raise ValueError("Pass either batcher or engine, not both")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        created_batcher = False

        if app.state.batcher is None:
            selected_engine = engine

            if selected_engine is None:
                if engine_factory is not None:
                    selected_engine = engine_factory()
                else:
                    from pagedserve.model.engine import PagedServeEngine

                    device = os.environ.get("PAGEDSERVE_DEVICE", "cuda:0")
                    selected_engine = PagedServeEngine(device=device)

            factory = batcher_factory or (
                lambda model_engine: ContinuousBatcher(
                    model_engine,
                    max_running_requests=max_running_requests,
                )
            )

            app.state.batcher = factory(selected_engine)
            created_batcher = True

        app.state.ready = True
        app.state.owns_batcher = created_batcher

        try:
            yield
        finally:
            app.state.ready = False

            if app.state.owns_batcher and app.state.batcher is not None:
                await run_in_threadpool(app.state.batcher.close)

            # Keep externally supplied batchers externally owned.
            app.state.batcher = batcher
            app.state.owns_batcher = False

    app = FastAPI(
        title="PagedServe Inference API",
        version="0.1.0",
        description="HTTP interface for PagedServe continuous-batched inference.",
        lifespan=lifespan,
    )

    app.state.batcher = batcher
    app.state.ready = False
    app.state.owns_batcher = False

    def get_batcher() -> Any:
        active = app.state.batcher

        if not app.state.ready or active is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="PagedServe inference engine is not ready",
            )

        return active

    @app.get("/health")
    @app.get("/healthz")
    async def health() -> dict[str, str]:
        ready = bool(app.state.ready and app.state.batcher is not None)
        return {
            "status": "ok" if ready else "starting",
            "service": "pagedserve",
        }

    @app.post(
        "/v1/requests",
        response_model=SubmissionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    async def submit_request(
        payload: GenerationRequest,
    ) -> SubmissionResponse:
        active = get_batcher()

        try:
            request_id = await run_in_threadpool(
                active.submit,
                payload.prompt,
                payload.max_new_tokens,
            )
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return SubmissionResponse(request_id=request_id)

    @app.get(
        "/v1/requests/{request_id}",
        response_model=RequestResultResponse,
    )
    async def get_request(request_id: str) -> RequestResultResponse:
        active = get_batcher()

        try:
            text = await run_in_threadpool(active.result, request_id, 0)

            return RequestResultResponse(
                request_id=request_id,
                status="completed",
                text=text,
            )
        except TimeoutError:
            return RequestResultResponse(
                request_id=request_id,
                status="processing",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RequestCancelledError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.delete(
        "/v1/requests/{request_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def cancel_request(request_id: str) -> Response:
        active = get_batcher()

        try:
            await run_in_threadpool(active.cancel, request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.post("/v1/generate")
    async def generate(payload: GenerationRequest) -> dict[str, str]:
        """Convenience endpoint that waits for the complete generated text."""
        active = get_batcher()

        try:
            request_id = await run_in_threadpool(
                active.submit,
                payload.prompt,
                payload.max_new_tokens,
            )
            text = await run_in_threadpool(active.result, request_id, None)

            return {
                "request_id": request_id,
                "text": text,
            }
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RequestCancelledError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


# Importing this module does not load the model.
# The default engine is initialized when the ASGI application starts.
app = create_app()
