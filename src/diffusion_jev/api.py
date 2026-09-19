"""Small HTTP service with bounded admission and serialized inference."""

import asyncio
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Literal, Protocol

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from diffusion_jev.canvas import PROMPT_TEMPLATE_VERSION
from diffusion_jev.compat import (
    COMPAT_PATH,
    CompatRequest,
    CompatResponse,
    to_compat_response,
    to_decision_request,
)
from diffusion_jev.schemas import DecisionRequest, DecisionResponse, StrictModel

MODEL_ID = "diffusiongemma-local"
MAX_BODY_BYTES = 1024 * 1024
MAX_ADMITTED_REQUESTS = 9


class DecisionEngine(Protocol):
    def decide(self, request: DecisionRequest) -> DecisionResponse: ...


class BodyLimitMiddleware:
    def __init__(self, app: ASGIApp, limit: int = MAX_BODY_BYTES) -> None:
        self.app = app
        self.limit = limit

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > self.limit:
                # Match the path's error shape; the handlers below never see this.
                oversized: dict[str, object] = (
                    {"error_code": 413, "error_summary": "Request too large"}
                    if scope.get("path") == COMPAT_PATH
                    else {"detail": "Request too large"}
                )
                response = JSONResponse(status_code=413, content=oversized)
                await response(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        delivered = False

        async def bounded_receive() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}

        await self.app(scope, bounded_receive, send)


class HealthResponse(StrictModel):
    status: Literal["ok"] = "ok"
    model: str = MODEL_ID
    prompt_template: str = PROMPT_TEMPLATE_VERSION


class ModelEntry(StrictModel):
    id: str = MODEL_ID
    object: Literal["model"] = "model"
    owned_by: str = "local"


class ModelList(StrictModel):
    object: Literal["list"] = "list"
    data: list[ModelEntry]


def create_app(engine_factory: Callable[[], DecisionEngine]) -> FastAPI:
    executor: ThreadPoolExecutor | None = None
    engine: DecisionEngine | None = None
    admitted = 0

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal executor, engine
        worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jev-inference")
        executor = worker
        try:
            # MLX lazy buffers retain thread-local streams from model construction.
            engine = await asyncio.get_running_loop().run_in_executor(worker, engine_factory)
            yield
        finally:
            engine = None
            executor = None
            await asyncio.to_thread(worker.shutdown, wait=True, cancel_futures=True)

    app = FastAPI(title="Local JEV decisions", version="0.1.0", lifespan=lifespan)
    app.add_middleware(BodyLimitMiddleware)

    def error_response(request: Request, status: int, detail: str) -> JSONResponse:
        # Neither shape echoes the request: details can contain private source code.
        if request.url.path == COMPAT_PATH:
            return JSONResponse(
                status_code=400 if status == 422 else status,
                content={"error_code": 400 if status == 422 else status, "error_summary": detail},
            )
        return JSONResponse(status_code=status, content={"detail": detail})

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error_response(request, 422, "Invalid request")

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return error_response(request, exc.status_code, str(exc.detail))

    @app.get("/health")
    async def health() -> HealthResponse:
        return HealthResponse()

    @app.get("/v1/models")
    async def models() -> ModelList:
        return ModelList(data=[ModelEntry()])

    async def run_decision(request: DecisionRequest) -> DecisionResponse:
        nonlocal admitted
        if engine is None or executor is None:
            raise HTTPException(status_code=503, detail="Model is not ready")
        if request.model != MODEL_ID:
            raise HTTPException(status_code=400, detail="Unknown model")
        if admitted >= MAX_ADMITTED_REQUESTS:
            raise HTTPException(status_code=503, detail="Inference queue is full")
        admitted += 1
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(executor, engine.decide, request)

        def release(completed: asyncio.Future[DecisionResponse]) -> None:
            nonlocal admitted
            admitted -= 1
            if not completed.cancelled():
                completed.exception()

        future.add_done_callback(release)
        try:
            # Disconnected clients must not free admission while GPU work is still running.
            return await asyncio.shield(future)
        except ValueError:
            raise HTTPException(status_code=422, detail="Request cannot be processed") from None
        except Exception:
            raise HTTPException(status_code=500, detail="Inference failed") from None

    @app.post("/v1/systemone")
    async def decide(request: DecisionRequest) -> DecisionResponse:
        return await run_decision(request)

    @app.post(COMPAT_PATH)
    async def decisions(request: CompatRequest) -> CompatResponse:
        return to_compat_response(await run_decision(to_decision_request(request)))

    return app
