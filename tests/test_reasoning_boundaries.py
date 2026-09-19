"""CPU checks for evaluation order, exact forwarding, and failure propagation."""

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Unpack

import mlx.core as mx
import pytest

from diffusion_jev.reasoning_boundaries import ConditioningInputs, ReasoningBoundary
from diffusion_jev.runtime_types import Cache, Decoder, Encoder


@dataclass(frozen=True)
class Config:
    canvas_length: int = 256


class Backbone:
    @property
    def decoder(self) -> Decoder:
        raise AssertionError("The boundary must not inspect or replace the decoder")

    @property
    def encoder(self) -> Encoder:
        raise AssertionError("The boundary must not inspect or replace the encoder")


@dataclass(frozen=True)
class TinyCache:
    state: tuple[mx.array, mx.array]
    offset: int = 1


@dataclass(frozen=True)
class Output:
    logits: mx.array


class Source:
    def __init__(self, order: list[str]) -> None:
        self.model = Backbone()
        self.config = Config()
        self.order = order
        self.cache: list[Cache] = [TinyCache((mx.ones((1,)), mx.zeros((1,))))]
        self.output = Output(mx.ones((1,)))
        self.canvas: mx.array | None = None
        self.mask: dict[str, mx.array | None] | None = None
        self.inputs: ConditioningInputs = {}

    def make_cache(self) -> list[Cache]:
        return self.cache

    def __call__(
        self, *, cache: list[Cache], canvas_ids: mx.array,
        decoder_attention_mask: dict[str, mx.array | None],
        **inputs: Unpack[ConditioningInputs],
    ) -> Output:
        assert cache is self.cache
        self.order.append("delegate")
        self.canvas, self.mask, self.inputs = canvas_ids, decoder_attention_mask, inputs
        return self.output


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


@pytest.mark.parametrize("conditioned", [False, True])
def test_evaluate_inputs_before_delegate_preserves_identities(
    conditioned: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    order: list[str] = []
    source = Source(order)
    original_backbone, original_config = source.model, source.config
    evaluated: list[list[mx.array]] = []
    native_eval = mx.eval

    def evaluate(arrays: list[mx.array]) -> None:
        order.append("eval")
        evaluated.append(arrays)
        native_eval(arrays)

    monkeypatch.setattr(mx, "eval", evaluate)
    boundary = ReasoningBoundary(source)
    conditioning = mx.ones((1,)) + 2
    inputs: ConditioningInputs = (
        {"self_conditioning_embeddings": conditioning}
        if conditioned else {"self_conditioning_logits": None}
    )
    canvas = mx.array([[1]], dtype=mx.int32)
    mask: dict[str, mx.array | None] = {"full_attention": None}
    result = boundary(
        cache=source.cache, canvas_ids=canvas, decoder_attention_mask=mask, **inputs,
    )
    assert order == ["eval"] * (2 if conditioned else 1) + ["delegate"]
    assert all(a is b for a, b in zip(evaluated[0], source.cache[0].state, strict=True))
    if conditioned:
        assert evaluated[1][0] is conditioning
    assert result is source.output
    assert source.canvas is canvas and source.mask is mask
    assert source.inputs.keys() == inputs.keys()
    assert source.inputs.get("self_conditioning_logits") is inputs.get("self_conditioning_logits")
    assert source.inputs.get("self_conditioning_embeddings") is inputs.get(
        "self_conditioning_embeddings"
    )
    assert boundary.model is source.model is original_backbone
    assert boundary.config is source.config is original_config
    assert boundary.make_cache() is source.cache


@pytest.mark.parametrize("failure_call", [1, 2])
def test_evaluation_failure_propagates_before_delegation(
    failure_call: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = Source([])
    calls = 0
    failure = RuntimeError("Synthetic boundary failure")

    def fail(arrays: list[mx.array]) -> None:
        nonlocal calls
        calls += 1
        if calls == failure_call:
            raise failure

    monkeypatch.setattr(mx, "eval", fail)
    boundary = ReasoningBoundary(source)
    with pytest.raises(RuntimeError) as caught:
        boundary(cache=source.cache, canvas_ids=mx.array([[1]]), decoder_attention_mask={},
                 self_conditioning_embeddings=mx.ones((1,)))
    assert caught.value is failure
    assert calls == failure_call
    assert source.order == []
