"""Exact reasoning continuation, independent conditioning, and work accounting."""

import threading
from collections.abc import Iterator
from typing import Literal

import mlx.core as mx
import pytest
from test_canvas import LexemeTokenizer
from test_engine_math import FakeModel

from diffusion_jev import reasoning
from diffusion_jev.canvas import THOUGHT_CLOSE, THOUGHT_OPEN
from diffusion_jev.engine import LocalEngine, PreparedRead
from diffusion_jev.reasoning import ReasoningMetrics, ReasoningResult
from diffusion_jev.runtime_types import Cache
from diffusion_jev.schemas import DecisionOptions, DecisionRequest, NoulQuestion


class ChatTokenizer(LexemeTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.conversations: list[list[dict[str, str]]] = []

    def apply_chat_template(
        self, conversation: list[dict[str, str]], *, tokenize: Literal[False],
        add_generation_prompt: bool, enable_thinking: bool,
    ) -> str:
        assert not tokenize and add_generation_prompt and enable_thinking
        self.conversations.append(conversation)
        return "THINK_ENABLED " + " ".join(turn["content"] for turn in conversation)


class RecordingEngine(LocalEngine):
    def _prefill(self, prompt: tuple[int, ...]) -> list[Cache]:
        self.prefilled.append(prompt)
        return []

    def _read(
        self, prepared: PreparedRead, cache: list[Cache], request: DecisionRequest,
    ) -> list[list[float]]:
        self.read_canvases.append(prepared)
        return [[0.8, 0.2] for _ in prepared.canvas.slots]

    def __init__(self) -> None:
        # No model construction or GPU buffers are needed to test continuation assembly.
        self.model = FakeModel()
        self.tokenizer = ChatTokenizer()
        self.max_prompt_tokens = 8192
        self.reasoning_tokens = 256
        self.trajectory_steps = 1
        self._lock = threading.Lock()
        self.last_diagnostics = None
        self.prefilled: list[tuple[int, ...]] = []
        self.read_canvases: list[PreparedRead] = []


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


@pytest.mark.parametrize("mode, groups", [("packed", 1), ("independent", 2)])
@pytest.mark.parametrize("budget", [256, 512])
def test_exact_continuation_and_one_thought_per_group_shared_across_samples(
    mode: Literal["packed", "independent"], groups: int, monkeypatch: pytest.MonkeyPatch,
    budget: int,
) -> None:
    engine = RecordingEngine()
    engine.reasoning_tokens = budget
    generated: list[tuple[int, ...]] = []
    seeds: list[int] = []

    def fake_prepass(
        model: reasoning.ReasoningModel, tokenizer: reasoning.ReasoningTokenizer,
        generation_prefix: tuple[int, ...], *, thought_close_id: int, max_tokens: int,
        seed: int, max_prompt_tokens: int,
    ) -> ReasoningResult:
        assert max_tokens == budget and max_prompt_tokens == 8192
        generated.append(generation_prefix)
        seeds.append(seed)
        return ReasoningResult(
            generation_prefix + (99991, 99992, thought_close_id),
            ReasoningMetrics(
                seed=seed, max_tokens=max_tokens, retained_tokens=2,
                denoising_steps=3, work_tokens=768, wall_ms=12.5, natural_close=True,
                continuation_prefill_tokens=256 if budget == 512 else 0,
            ),
        )

    monkeypatch.setattr(reasoning, "reasoning_prepass", fake_prepass)
    request = DecisionRequest(
        state={"evidence": "SAME_PUBLIC_EVIDENCE"},
        questions={
            "one": NoulQuestion(type="noul", instructions="TARGET_ONE"),
            "two": NoulQuestion(type="noul", instructions="TARGET_TWO"),
        }, options=DecisionOptions(mode=mode, samples=4, seed=2**32 - 1, canvas_length=32),
    )
    response = engine.decide(request)
    assert len(generated) == groups
    assert seeds == [2**32 - 1, 0][:groups]
    assert len(engine.prefilled) == groups
    close = engine.tokenizer.encode(THOUGHT_CLOSE, add_special_tokens=False)[0]
    opening = tuple(engine.tokenizer.encode(THOUGHT_OPEN, add_special_tokens=False))
    for prefix, final, read in zip(generated, engine.prefilled, engine.read_canvases, strict=True):
        assert prefix[-len(opening):] == opening
        assert final == prefix + (99991, 99992, close)
        assert close not in read.canvas.tokens
        assert 99991 not in read.canvas.tokens
    assert "99991" not in engine.tokenizer.vocabulary
    for conversation in engine.tokenizer.conversations:
        assert "SAME_PUBLIC_EVIDENCE" in conversation[1]["content"]
        if mode == "independent":
            system = conversation[0]["content"]
            assert ("TARGET_ONE" in system) != ("TARGET_TWO" in system)
    usage = response.usage
    assert usage.prompt_tokens == (
        sum(map(len, generated)) + sum(map(len, engine.prefilled))
        + groups * (256 if budget == 512 else 0)
    )
    assert usage.decoder_passes == groups * (4 + 3)
    assert usage.canvas_tokens == groups * (32 * 4 + 768)
    assert usage.reasoning_tokens == groups * 2
    assert usage.reasoning_ms == groups * 12.5
    assert usage.reasoning_passes == groups * 3
    assert usage.reasoning_forced_closures == 0
    assert "99991" not in response.model_dump_json()


def test_worst_case_reasoning_length_is_checked_before_generation() -> None:
    engine = RecordingEngine()
    engine.max_prompt_tokens = 256
    request = DecisionRequest(
        state="public", questions={"q": NoulQuestion(type="noul", instructions="true")},
    )
    with pytest.raises(ValueError, match="Prompt exceeds"):
        engine._prepare(request)
    assert not engine.prefilled


def test_trajectory_requires_full_projection_before_any_prefill() -> None:
    engine = RecordingEngine()
    engine.trajectory_steps = 2
    request = DecisionRequest(
        state="public", questions={"q": NoulQuestion(type="noul", instructions="true")},
    )
    with pytest.raises(ValueError, match="requires full projection"):
        engine._prepare(request)
    assert not engine.prefilled


def test_engine_reserves_513_tokens_for_512_reasoning_before_generation() -> None:
    engine = RecordingEngine()
    engine.reasoning_tokens = 512
    request = DecisionRequest(
        state="public", questions={"q": NoulQuestion(type="noul", instructions="true")},
    )
    prepared = engine._prepare(request)
    exact_limit = len(prepared[0].prompt) + 513
    engine.max_prompt_tokens = exact_limit - 1
    with pytest.raises(ValueError, match="Prompt exceeds"):
        engine._prepare(request)
    engine.max_prompt_tokens = exact_limit
    assert engine._prepare(request) == prepared
    assert not engine.prefilled
