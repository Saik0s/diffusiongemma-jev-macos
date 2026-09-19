import asyncio
import threading
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from diffusion_jev.api import MAX_ADMITTED_REQUESTS, MAX_BODY_BYTES, create_app
from diffusion_jev.canvas import PROMPT_TEMPLATE_VERSION
from diffusion_jev.compat import COMPAT_PATH, CompatResponse
from diffusion_jev.schemas import DecisionRequest, DecisionResponse, NoulAnswer, Usage

BODY = '{"state":null,"questions":{"q":{"type":"noul","instructions":"Check"}}}'
HEADERS = {"content-type": "application/json"}


class FakeEngine:
    def __init__(self, failure: Exception | None = None) -> None:
        self.failure = failure

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        if self.failure is not None:
            raise self.failure
        return DecisionResponse(
            model=request.model,
            answers={"q": NoulAnswer(noul=0.5)},
            usage=Usage(
                prompt_tokens=10, questions=1, decoder_passes=1, canvas_tokens=16,
                prefill_ms=1.0, decode_ms=1.0, total_ms=2.0, peak_memory_gb=0.0,
            ),
        )


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(create_app(FakeEngine)) as client:
        yield client


def test_routes(client: TestClient) -> None:
    assert client.get("/health").json() == {
        "status": "ok",
        "model": "diffusiongemma-local",
        "prompt_template": PROMPT_TEMPLATE_VERSION,
    }
    assert client.get("/v1/models").json()["data"][0]["id"] == "diffusiongemma-local"
    response = client.post("/v1/systemone", content=BODY, headers=HEADERS)
    assert response.status_code == 200
    parsed = DecisionResponse.model_validate_json(response.content)
    assert parsed.answers["q"] == NoulAnswer(noul=0.5)
    assert client.get("/docs").status_code == 200


def test_validation_does_not_echo_input(client: TestClient) -> None:
    response = client.post("/v1/systemone", content='{"secret":"PRIVATE"}', headers=HEADERS)
    assert response.status_code == 422
    assert response.json() == {"detail": "Invalid request"}
    assert "PRIVATE" not in response.text


def test_model_mismatch(client: TestClient) -> None:
    body = BODY.replace('"state":null', '"state":null,"model":"other"')
    assert client.post("/v1/systemone", content=body, headers=HEADERS).status_code == 400


def test_body_limit(client: TestClient) -> None:
    response = client.post("/v1/systemone", content=b"x" * (MAX_BODY_BYTES + 1))
    assert response.status_code == 413


def test_chunked_body_limit() -> None:
    async def scenario() -> None:
        async def chunks() -> AsyncIterator[bytes]:
            yield b"x" * (MAX_BODY_BYTES // 2)
            yield b"x" * (MAX_BODY_BYTES // 2 + 1)

        app = create_app(FakeEngine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/v1/systemone", content=chunks())
                assert response.status_code == 413

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "error,status", [(ValueError("PRIVATE"), 422), (RuntimeError("PRIVATE"), 500)]
)
def test_sanitized_engine_errors(error: Exception, status: int) -> None:
    with TestClient(create_app(lambda: FakeEngine(error))) as client:
        response = client.post("/v1/systemone", content=BODY, headers=HEADERS)
    assert response.status_code == status
    assert "PRIVATE" not in response.text


class BlockingEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.finish = threading.Event()
        self.thread_ids: set[int] = set()

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.thread_ids.add(threading.get_ident())
        self.started.set()
        if not self.finish.wait(timeout=10):
            raise RuntimeError("Test worker timed out")
        return super().decide(request)


def test_backpressure_and_responsive_health() -> None:
    async def scenario() -> None:
        engine = BlockingEngine()
        app = create_app(lambda: engine)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                requests = [
                    asyncio.create_task(client.post("/v1/systemone", content=BODY, headers=HEADERS))
                    for _ in range(MAX_ADMITTED_REQUESTS)
                ]
                try:
                    assert await asyncio.to_thread(engine.started.wait, 2)
                    # Let each queued HTTP task reach the executor submission point.
                    await asyncio.sleep(0)
                    health = await asyncio.wait_for(client.get("/health"), timeout=1)
                    assert health.status_code == 200
                    rejected = await client.post("/v1/systemone", content=BODY, headers=HEADERS)
                    assert rejected.status_code == 503
                    cancelled = requests.pop(0)
                    cancelled.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await cancelled
                    still_full = await client.post("/v1/systemone", content=BODY, headers=HEADERS)
                    assert still_full.status_code == 503
                finally:
                    engine.finish.set()
                    results = await asyncio.gather(*requests)
                assert all(response.status_code == 200 for response in results)
                assert len(engine.thread_ids) == 1
                recovered = await client.post("/v1/systemone", content=BODY, headers=HEADERS)
                assert recovered.status_code == 200

    asyncio.run(scenario())


class ThreadAffineEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.owner = threading.current_thread()

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        if threading.current_thread() is not self.owner:
            raise RuntimeError("Model called outside its loading thread")
        return super().decide(request)


def test_model_load_and_inference_share_worker_thread() -> None:
    loaded: list[ThreadAffineEngine] = []

    def factory() -> ThreadAffineEngine:
        engine = ThreadAffineEngine()
        loaded.append(engine)
        return engine

    app = create_app(factory)
    assert not loaded
    with TestClient(app) as client:
        assert len(loaded) == 1
        assert loaded[0].owner is not threading.current_thread()
        for _ in range(2):
            response = client.post("/v1/systemone", content=BODY, headers=HEADERS)
            assert response.status_code == 200
    assert not loaded[0].owner.is_alive()


def test_startup_failure_stops_loading_thread() -> None:
    threads: list[threading.Thread] = []

    def factory() -> FakeEngine:
        threads.append(threading.current_thread())
        raise RuntimeError("Model loading failed")

    with pytest.raises(RuntimeError, match="Model loading failed"), TestClient(create_app(factory)):
        pytest.fail("Server started after model loading failed")
    assert len(threads) == 1
    assert not threads[0].is_alive()


def test_inference_requires_completed_startup() -> None:
    async def scenario() -> None:
        app = create_app(FakeEngine)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/v1/systemone", content=BODY, headers=HEADERS)
            assert response.status_code == 503
            assert response.json() == {"detail": "Model is not ready"}

    asyncio.run(scenario())


def test_compat_route_returns_the_hosted_envelope(client: TestClient) -> None:
    body = BODY.replace('"state":null', '"state":{"note":"ok"}')
    response = client.post(COMPAT_PATH, content=body, headers=HEADERS)
    assert response.status_code == 200
    parsed = CompatResponse.model_validate_json(response.content)
    assert parsed.model == "diffusiongemma-local"
    assert parsed.provider == "local"
    assert parsed.answers["q"] == NoulAnswer(noul=0.5)
    # Local inference is not billed, and one label position is read per question.
    assert parsed.usage.cost == 0.0
    assert parsed.usage.output_tokens == 1
    assert parsed.usage.input_tokens == 10
    # Hosted responses carry an opaque id; the local one is fresh per response.
    assert parsed.id.startswith("decision-")
    second = CompatResponse.model_validate_json(
        client.post(COMPAT_PATH, content=body, headers=HEADERS).content
    )
    assert second.id != parsed.id


def test_compat_route_rejects_a_scalar_state(client: TestClient) -> None:
    # Hosted Jev refuses a bare scalar root, so the compat shape refuses it too.
    response = client.post(COMPAT_PATH, content=BODY, headers=HEADERS)
    assert response.status_code == 400


@pytest.mark.parametrize(
    "content,status",
    [('{"secret":"PRIVATE"}', 400), (b"x" * (MAX_BODY_BYTES + 1), 413)],
)
def test_compat_errors_use_the_hosted_shape(
    client: TestClient, content: str | bytes, status: int
) -> None:
    response = client.post(COMPAT_PATH, content=content, headers=HEADERS)
    assert response.status_code == status
    payload = response.json()
    assert payload["error_code"] == status
    assert isinstance(payload["error_summary"], str)
    assert "PRIVATE" not in response.text


def test_compat_route_reports_engine_failures_as_400() -> None:
    body = BODY.replace('"state":null', '"state":{"note":"ok"}')
    with TestClient(create_app(lambda: FakeEngine(ValueError("PRIVATE")))) as client:
        response = client.post(COMPAT_PATH, content=body, headers=HEADERS)
    assert response.status_code == 400
    assert response.json()["error_code"] == 400
    assert "PRIVATE" not in response.text
