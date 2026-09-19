"""Exercise the real decision read path with CPU arrays and no model weights."""

import math
from dataclasses import dataclass
from typing import Literal

import mlx.core as mx
import pytest

from diffusion_jev.canvas import Canvas, Slot
from diffusion_jev.engine import LocalEngine, PreparedRead
from diffusion_jev.runtime_types import Cache
from diffusion_jev.schemas import DecisionOptions, DecisionRequest, NoulQuestion, Question
from diffusion_jev.trajectory import TrajectoryMetrics, TrajectoryModel, TrajectoryResult


class FakeEmbedding:
    def __call__(self, tokens: mx.array) -> mx.array:
        return mx.eye(2, dtype=mx.bfloat16)[tokens]

    def as_linear(self, hidden: mx.array) -> mx.array:
        return hidden @ mx.eye(2, dtype=mx.bfloat16)


class FakeDecoder:
    embed_tokens = FakeEmbedding()

    def __call__(self, canvas_ids: mx.array, *, cache: list[Cache]) -> mx.array:
        hidden = mx.array([[[20.0, 20.125]]], dtype=mx.bfloat16)
        return mx.broadcast_to(hidden, (1, canvas_ids.shape[1], 2))


class UnusedEncoder:
    def __call__(
        self, input_ids: mx.array, *, cache: list[Cache]
    ) -> tuple[mx.array, list[Cache]]:
        raise AssertionError("A decision read must not prefill the encoder")


class FakeBackbone:
    decoder = FakeDecoder()
    encoder = UnusedEncoder()


@dataclass(frozen=True)
class FakeTextConfig:
    vocab_size: int = 2
    pad_token_id: int = 0
    final_logit_softcapping: float = 30.0


@dataclass(frozen=True)
class FakeConfig:
    text_config: FakeTextConfig = FakeTextConfig()


class FakeModel:
    model = FakeBackbone()
    config = FakeConfig()

    def __init__(self) -> None:
        self.softcap_calls = 0

    def make_cache(self) -> list[Cache]:
        raise AssertionError("This regression bypasses prefill")

    def _softcap(self, logits: mx.array) -> mx.array:
        assert logits.dtype == mx.bfloat16
        self.softcap_calls += 1
        cap = self.config.text_config.final_logit_softcapping
        return mx.tanh(logits.astype(mx.float32) / cap) * cap


@pytest.mark.parametrize("projection", ["labels", "full"])
def test_read_preserves_canonical_fp32_softcap(projection: Literal["labels", "full"]) -> None:
    # CPU stream scopes every array operation without changing the server's default device.
    with mx.stream(mx.cpu):
        model = FakeModel()
        engine = object.__new__(LocalEngine)
        engine.model = model
        engine.trajectory_steps = 1
        engine.diagnostics = False
        engine.last_diagnostics = None
        questions: dict[str, Question] = {
            "decision": NoulQuestion(type="noul", instructions="The proposition is true")
        }
        request = DecisionRequest(
            state=None, questions=questions, options=DecisionOptions(projection=projection)
        )
        prepared = PreparedRead(
            prompt=(),
            canvas=Canvas(tokens=(0,) * 16, slots=(Slot("decision", 3, (0, 1)),)),
            questions=questions,
        )
        probabilities = engine._read(prepared, [], request)

    expected_yes = 1 / (1 + math.exp(30 * (math.tanh(20.125 / 30) - math.tanh(20 / 30))))
    assert probabilities[0] == pytest.approx([expected_yes, 1 - expected_yes], abs=1e-6)
    assert model.softcap_calls == 1


class DiagnosticEmbedding:
    def __init__(self) -> None:
        self.projected_rows: list[int] = []

    def __call__(self, tokens: mx.array) -> mx.array:
        return mx.eye(3, dtype=mx.bfloat16)[tokens]

    def as_linear(self, hidden: mx.array) -> mx.array:
        self.projected_rows.append(hidden.shape[1])
        return hidden


class DiagnosticDecoder:
    def __init__(self) -> None:
        self.embed_tokens = DiagnosticEmbedding()

    def __call__(self, canvas_ids: mx.array, *, cache: list[Cache]) -> mx.array:
        hidden = mx.array([[[2.0, 1.0, 9.0]]], dtype=mx.bfloat16)
        return mx.broadcast_to(hidden, (1, canvas_ids.shape[1], 3))


class DiagnosticBackbone:
    encoder = UnusedEncoder()

    def __init__(self) -> None:
        self.decoder = DiagnosticDecoder()


class DiagnosticModel(FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = DiagnosticBackbone()
        self.config = FakeConfig(FakeTextConfig(vocab_size=3))


@pytest.mark.parametrize("projection", ["labels", "full"])
def test_diagnostics_detect_invalid_vocab_winner_without_changing_decision(
    projection: Literal["labels", "full"],
) -> None:
    with mx.stream(mx.cpu):
        model = DiagnosticModel()
        engine = object.__new__(LocalEngine)
        engine.model = model
        engine.trajectory_steps = 1
        engine.last_diagnostics = None
        questions: dict[str, Question] = {
            "decision": NoulQuestion(type="noul", instructions="Synthetic control")
        }
        request = DecisionRequest(
            state=None, questions=questions,
            options=DecisionOptions(projection=projection, samples=2),
        )
        prepared = PreparedRead(
            prompt=(), questions=questions,
            canvas=Canvas(tokens=(0,) * 16, slots=(Slot("decision", 3, (0, 1)),)),
        )
        engine.diagnostics = False
        baseline = engine._read(prepared, [], request)
        engine.diagnostics = True
        measured = engine._read(prepared, [], request)

    assert measured == baseline
    assert measured[0][0] > 0.7
    diagnostics = engine.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.slots == 2
    assert diagnostics.invalid_argmax_count == 2
    capped = [30 * math.tanh(value / 30) for value in (2, 1, 9)]
    expected = sum(math.exp(value) for value in capped[:2]) / sum(
        math.exp(value) for value in capped
    )
    assert diagnostics.allowed_label_mass_mean == pytest.approx(expected, rel=1e-5)
    assert diagnostics.allowed_label_mass_min == pytest.approx(expected, rel=1e-5)
    assert model.model.decoder.embed_tokens.projected_rows
    expected_rows = {16} if projection == "full" else {1}
    assert set(model.model.decoder.embed_tokens.projected_rows) == expected_rows


def test_trajectory_read_uses_final_logits_once_and_wraps_transition_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusion_jev import trajectory

    with mx.stream(mx.cpu):
        model = FakeModel()
        engine = object.__new__(LocalEngine)
        engine.model = model
        engine.trajectory_steps = 2
        engine.trajectory_accepted_slots = 0
        engine.diagnostics = False
        engine.last_diagnostics = None
        questions: dict[str, Question] = {
            "decision": NoulQuestion(type="noul", instructions="Synthetic test")
        }
        request = DecisionRequest(
            state=None, questions=questions,
            options=DecisionOptions(projection="full", samples=2, seed=2**32 - 1),
        )
        prepared = PreparedRead(
            prompt=(), questions=questions,
            canvas=Canvas(tokens=(0,) * 16, slots=(Slot("decision", 3, (0, 1)),)),
        )
        seeds: list[int] = []
        cache: list[Cache] = []

        def fake_trajectory(
            actual_model: TrajectoryModel, tokens: tuple[int, ...], positions: tuple[int, ...],
            actual_cache: list[Cache], *, seed: int,
        ) -> TrajectoryResult:
            assert actual_model is model and actual_cache is cache
            assert len(tokens) == 16 and positions == (3,)
            seeds.append(seed)
            return TrajectoryResult(
                mx.broadcast_to(mx.array([[[1.0, 2.0]]]), (1, 16, 2)),
                TrajectoryMetrics(
                    seed=seed, accepted_answer_positions=1, answer_positions=1,
                    canvas_tokens=16, work_tokens=32, wall_ms=1.0, peak_memory_gb=0.0,
                ),
            )

        monkeypatch.setattr(trajectory, "structured_trajectory", fake_trajectory)
        probabilities = engine._read(prepared, cache, request)
    assert probabilities[0] == pytest.approx([1 / (1 + math.e), 1 / (1 + math.exp(-1))])
    assert model.softcap_calls == 0
    assert engine.trajectory_accepted_slots == 2
    assert seeds == [2**32 - 1, 0]
