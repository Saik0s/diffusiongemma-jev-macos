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
