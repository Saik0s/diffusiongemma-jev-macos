"""Small synchronous client for the local decision API."""

from types import TracebackType

import httpx
from pydantic import ValidationError

from diffusion_jev.schemas import DecisionRequest, DecisionResponse


class DecisionClientError(RuntimeError):
    """A request failed without exposing its input or response payload."""


class DecisionClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8017") -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=120.0)

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        try:
            response = self._client.post(
                "/v1/systemone", json=request.model_dump(mode="json")
            )
            response.raise_for_status()
            return DecisionResponse.model_validate_json(response.content)
        except (httpx.HTTPError, ValidationError, ValueError):
            raise DecisionClientError("Local decision request failed.") from None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "DecisionClient":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
