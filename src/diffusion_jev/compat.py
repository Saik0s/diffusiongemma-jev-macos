"""OpenRouter-shaped decisions envelope over the same local engine.

Hosted Jev answers `POST /api/alpha/decisions` with `id`, `model`, `provider`,
`answers` and a token/cost `usage` block. The answer objects themselves already match this
project's schema field for field, so only the envelope and the request's accepted
`state` shapes differ. See docs/jev-differences.md for the remaining gaps.
"""

from typing import Annotated
from uuid import uuid4

from pydantic import Field, JsonValue

from diffusion_jev.schemas import (
    Answer,
    DecisionOptions,
    DecisionRequest,
    DecisionResponse,
    Identifier,
    Question,
    StrictModel,
)

PROVIDER = "local"
COMPAT_PATH = "/api/alpha/decisions"


class CompatRequest(StrictModel):
    """Hosted request shape. A scalar `state` root is rejected there and here."""

    model: Identifier = "diffusiongemma-local"
    state: dict[str, JsonValue] | list[JsonValue] | str
    questions: Annotated[dict[Identifier, Question], Field(min_length=1, max_length=32)]
    options: DecisionOptions = Field(default_factory=DecisionOptions)


class CompatUsage(StrictModel):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]
    cost: Annotated[float, Field(ge=0, allow_inf_nan=False)]


class CompatResponse(StrictModel):
    id: str
    model: str
    provider: str
    answers: dict[str, Answer]
    usage: CompatUsage


def to_decision_request(request: CompatRequest) -> DecisionRequest:
    return DecisionRequest(
        model=request.model,
        state=request.state,
        questions=request.questions,
        options=request.options,
    )


def to_compat_response(response: DecisionResponse) -> CompatResponse:
    return CompatResponse(
        # Hosted responses carry an opaque id. Nothing correlates it here: the
        # server keeps no request log, so a fresh random value is the honest one.
        id=f"decision-{uuid4().hex}",
        model=response.model,
        provider=PROVIDER,
        answers=response.answers,
        usage=CompatUsage(
            input_tokens=response.usage.prompt_tokens,
            # One verified label position is read per question. This is not the
            # hosted count, which includes structure around the same decisions.
            output_tokens=len(response.answers),
            cost=0.0,
        ),
    )
