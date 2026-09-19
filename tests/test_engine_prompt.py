"""What the engine puts in front of the model, and which sampling profile it applies."""

from typing import Literal

import pytest
from test_canvas import LexemeTokenizer
from test_engine_math import FakeModel

from diffusion_jev.engine import LocalEngine
from diffusion_jev.schemas import (
    DecisionOptions,
    DecisionRequest,
    NoulCriteria,
    NoulQuestion,
    Question,
)


class RecordingTokenizer(LexemeTokenizer):
    """Captures the chat turns so a test can read the exact rendered state."""

    def __init__(self) -> None:
        super().__init__()
        self.conversations: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: Literal[False],
        add_generation_prompt: bool, enable_thinking: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt and not enable_thinking
        self.conversations.append(conversation)
        return " ".join(turn["content"] for turn in conversation)


def build_engine() -> LocalEngine:
    # No weights, no GPU buffers: _prepare only needs the tokenizer and the pad id.
    engine = object.__new__(LocalEngine)
    engine.model = FakeModel()
    engine.tokenizer = RecordingTokenizer()
    engine.max_prompt_tokens = 8192
    engine.reasoning_tokens = 0
    engine.trajectory_steps = 1
    return engine


def test_state_keeps_the_caller_field_order() -> None:
    # Sorting moved "t" to the end of every timestamped event, which is not the
    # evidence the caller wrote. Order carries meaning in an event log.
    engine = build_engine()
    questions: dict[str, Question] = {
        "stuck": NoulQuestion(type="noul", instructions="Is the shopper stuck?")
    }
    request = DecisionRequest(
        state={"t": 1, "event": "promo_rejected", "cart_total": 42, "logged_in": True},
        questions=questions,
        options=DecisionOptions(canvas_length=32),
    )
    engine._prepare(request)
    conversation = engine.tokenizer.conversations[0]
    assert conversation[1]["content"] == (
        '{"t": 1, "event": "promo_rejected", "cart_total": 42, "logged_in": true}'
    )


def test_noul_criteria_reach_the_rendered_prompt() -> None:
    engine = build_engine()
    questions: dict[str, Question] = {
        "stuck": NoulQuestion(
            type="noul",
            instructions="Is the shopper stuck?",
            criteria=NoulCriteria(true="Repeated promo errors.", false="One clean apply."),
        )
    }
    engine._prepare(
        DecisionRequest(state={}, questions=questions, options=DecisionOptions(canvas_length=32))
    )
    system = engine.tokenizer.conversations[0][0]["content"]
    assert "yes: Repeated promo errors.\nno: One clean apply." in system


@pytest.mark.parametrize("profile", [1, 8])
def test_sampling_profile_fills_only_an_omitted_sample_count(profile: int) -> None:
    engine = object.__new__(LocalEngine)
    engine.default_samples = profile
    questions: dict[str, Question] = {
        "q": NoulQuestion(type="noul", instructions="The change is safe")
    }
    omitted = DecisionRequest(state=None, questions=questions)
    assert engine._apply_sampling_profile(omitted).options.samples == profile

    # An explicit request wins, including one that repeats the schema default.
    for requested in (1, 4):
        explicit = DecisionRequest(
            state=None, questions=questions, options=DecisionOptions(samples=requested)
        )
        assert engine._apply_sampling_profile(explicit).options.samples == requested


def test_sampling_profile_leaves_the_rest_of_the_request_alone() -> None:
    engine = object.__new__(LocalEngine)
    engine.default_samples = 8
    questions: dict[str, Question] = {
        "q": NoulQuestion(type="noul", instructions="The change is safe")
    }
    request = DecisionRequest(
        state={"note": "keep me"},
        questions=questions,
        options=DecisionOptions(mode="independent", seed=7, canvas_length=32),
    )
    applied = engine._apply_sampling_profile(request)
    assert applied.state == request.state
    assert applied.questions == request.questions
    assert applied.model == request.model
    assert (applied.options.mode, applied.options.seed, applied.options.canvas_length) == (
        "independent", 7, 32,
    )
