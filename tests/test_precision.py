"""CPU-only checks for the opt-in router rounding boundary and module replacement."""

from collections.abc import Iterator
from dataclasses import dataclass

import mlx.core as mx
import pytest
from mlx.nn.layers.base import Module
from mlx.utils import tree_flatten

from diffusion_jev import precision
from diffusion_jev.precision import (
    Float32Router,
    Float32RouterExperts,
    enable_fp32_router,
)


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


@dataclass(frozen=True)
class TinyConfig:
    top_k_experts: int = 2


class FixedScores(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.array([0.25, 1.25, -0.5, 2.5], dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.broadcast_to(self.weight, (*x.shape[:-1], 4))


class TinyRouter(Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyConfig()
        self.eps = 1e-6
        self._root_size = 2**-0.5
        self.proj = FixedScores()
        self.scale = mx.ones((2,), dtype=mx.bfloat16)
        self.per_expert_scale = mx.array([1.0, 0.5, 1.5, 2.0], dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        raise AssertionError("The replacement must compute float32 weights itself")


class FixedExperts(Module):
    def __init__(self, outputs: list[list[float]]) -> None:
        super().__init__()
        self.weight = mx.array(outputs, dtype=mx.bfloat16)
        self.sort_flags: list[bool] = []

    def __call__(
        self, x: mx.array, indices: mx.array, *, sorted_indices: bool
    ) -> mx.array:
        self.sort_flags.append(sorted_indices)
        return mx.expand_dims(self.weight[indices], -2)


class TinyExperts(Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden_dims = 2
        self.gate_up_proj = FixedExperts([[1.0] * 4, [2.0] * 4])
        self.down_proj = FixedExperts([[1.0, 2.0], [-1.0, 3.0]])

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        raise AssertionError("The replacement must control contribution rounding")


class TinyLayer(Module):
    def __init__(self) -> None:
        super().__init__()
        self.router = TinyRouter()
        self.experts = TinyExperts()


class TinyDecoder(Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [TinyLayer()]


class TinyEncoder(Module):
    def __init__(self, decoder: TinyDecoder) -> None:
        super().__init__()
        # The production encoder reads decoder.layers through its shared decoder.
        self.decoder = decoder


class TinyBackbone(Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = TinyDecoder()
        self.encoder = TinyEncoder(self.decoder)


class TinyModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyBackbone()


def test_router_retains_float32_probabilities_and_original_topk() -> None:
    source = TinyRouter()
    router = Float32Router(source)
    indices, weights = router(mx.ones((3, 2), dtype=mx.bfloat16))
    scores = source.proj.weight.astype(mx.float32)
    probabilities = mx.softmax(scores)
    expected_indices = mx.argpartition(scores, kth=-2)[-2:]
    expected_weights = probabilities[expected_indices]
    expected_weights /= expected_weights.sum()
    expected_weights *= source.per_expert_scale[expected_indices]
    assert weights.dtype == mx.float32
    assert indices.tolist() == [expected_indices.tolist()] * 3
    assert mx.allclose(weights, expected_weights, atol=1e-7).item()
    assert not mx.array_equal(weights, weights.astype(mx.bfloat16).astype(mx.float32)).item()


@pytest.mark.parametrize("tokens", [3, 32])
def test_experts_round_each_contribution_and_preserve_sorting(tokens: int) -> None:
    source = TinyExperts()
    experts = Float32RouterExperts(source)
    indices = mx.array([[0, 1] if row % 2 == 0 else [1, 0] for row in range(tokens)])
    weights = mx.broadcast_to(mx.array([0.501, 0.5]), (tokens, 2))
    actual = experts(mx.ones((tokens, 2), dtype=mx.bfloat16), indices, weights)
    contributions = source.down_proj.weight[indices].astype(mx.float32) * weights[..., None]
    expected = contributions.astype(mx.bfloat16).sum(axis=-2)
    rounded_after_sum = contributions.sum(axis=-2).astype(mx.bfloat16)
    assert actual.dtype == mx.bfloat16
    assert mx.array_equal(actual, expected).item()
    assert not mx.array_equal(actual, rounded_after_sum).item()
    assert source.gate_up_proj.sort_flags == [tokens * 2 >= 64]
    assert source.down_proj.sort_flags == [tokens * 2 >= 64]


@pytest.mark.parametrize("training", [True, False])
def test_replacement_preserves_parameters_state_sharing_and_isolation(training: bool) -> None:
    model, untouched = TinyModel(), TinyModel()
    model.train(training)
    decoder = model.model.decoder
    parameters: dict[str, mx.array] = dict(tree_flatten(decoder.parameters()))
    original_router = decoder.layers[0].router
    original_experts = decoder.layers[0].experts
    enable_fp32_router(model)
    router = decoder.layers[0].router
    experts = decoder.layers[0].experts
    assert isinstance(router, Float32Router)
    assert isinstance(experts, Float32RouterExperts)
    assert router.training is training
    assert experts.training is training
    assert router.proj is original_router.proj
    assert experts.gate_up_proj is original_experts.gate_up_proj
    assert experts.down_proj is original_experts.down_proj
    replacement_parameters: dict[str, mx.array] = dict(tree_flatten(decoder.parameters()))
    assert parameters.keys() == replacement_parameters.keys()
    assert all(value is replacement_parameters[name] for name, value in parameters.items())
    assert model.model.encoder.decoder.layers[0].router is router
    assert isinstance(untouched.model.decoder.layers[0].router, TinyRouter)
    assert isinstance(untouched.model.decoder.layers[0].experts, TinyExperts)
    enable_fp32_router(model)
    assert decoder.layers[0].router is router
    assert decoder.layers[0].experts is experts


def test_incompatible_version_is_rejected_before_mutating_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = TinyModel()
    monkeypatch.setattr(precision, "version", lambda _: "0.5.13")
    with pytest.raises(RuntimeError, match="requires mlx-optiq 0.5.12"):
        enable_fp32_router(model)
    assert isinstance(model.model.decoder.layers[0].router, TinyRouter)
    assert isinstance(model.model.decoder.layers[0].experts, TinyExperts)
