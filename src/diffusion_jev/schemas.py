"""Bounded public request and response contracts."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints

Identifier = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Description = Annotated[
    str, StringConstraints(min_length=1, max_length=4096, strip_whitespace=True)
]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Description


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Description
    criteria: Annotated[dict[Identifier, Description], Field(min_length=2, max_length=26)]


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Description
    criteria: Annotated[list[Description], Field(min_length=2, max_length=10)]


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class DecisionOptions(StrictModel):
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)] = 0
    samples: Annotated[int, Field(ge=1, le=8)] = 1
    mode: Literal["packed", "independent"] = "packed"
    projection: Literal["labels", "labels-fp32", "full"] = "labels"
    canvas_length: Annotated[int, Field(ge=16, le=256, multiple_of=16)] | None = None


class DecisionRequest(StrictModel):
    model: Identifier = "diffusiongemma-local"
    state: JsonValue
    questions: Annotated[dict[Identifier, Question], Field(min_length=1, max_length=32)]
    options: DecisionOptions = Field(default_factory=DecisionOptions)


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: Probability


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, Probability]
    confidence: Probability


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    legend: dict[str, str]
    probabilities: dict[str, Probability]
    confidence: Probability


Answer = Annotated[NoulAnswer | ChoiceAnswer | ScoreAnswer, Field(discriminator="type")]
NonnegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Usage(StrictModel):
    prompt_tokens: Annotated[int, Field(ge=0)]
    questions: Annotated[int, Field(ge=1)]
    decoder_passes: Annotated[int, Field(ge=0)]
    canvas_tokens: Annotated[int, Field(ge=0)]
    prefill_ms: NonnegativeFloat
    decode_ms: NonnegativeFloat
    total_ms: NonnegativeFloat
    peak_memory_gb: NonnegativeFloat
    reasoning_tokens: Annotated[int, Field(ge=0)] = 0
    reasoning_ms: NonnegativeFloat = 0.0
    reasoning_passes: Annotated[int, Field(ge=0)] = 0
    reasoning_forced_closures: Annotated[int, Field(ge=0)] = 0
    trajectory_accepted_slots: Annotated[int, Field(ge=0)] = 0


class DecisionResponse(StrictModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage
