"""Opt-in float32 router weights for the pinned OptiQ inference implementation."""

from collections.abc import Sequence
from importlib.metadata import version
from typing import Protocol

import mlx.core as mx
from mlx.nn import Module, gelu_approx
from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort


class Projection(Protocol):
    def __call__(self, x: mx.array) -> mx.array: ...


class ExpertProjection(Protocol):
    def __call__(
        self, x: mx.array, indices: mx.array, *, sorted_indices: bool
    ) -> mx.array: ...


class RouterConfig(Protocol):
    @property
    def top_k_experts(self) -> int: ...


class RouterSurface(Protocol):
    config: RouterConfig
    eps: float
    _root_size: float
    proj: Projection
    scale: mx.array
    per_expert_scale: mx.array

    @property
    def training(self) -> bool: ...

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]: ...


class ExpertsSurface(Protocol):
    hidden_dims: int
    gate_up_proj: ExpertProjection
    down_proj: ExpertProjection

    @property
    def training(self) -> bool: ...

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array: ...


class PrecisionLayer(Protocol):
    router: RouterSurface
    experts: ExpertsSurface


class PrecisionDecoder(Protocol):
    @property
    def layers(self) -> Sequence[PrecisionLayer]: ...


class PrecisionBackbone(Protocol):
    @property
    def decoder(self) -> PrecisionDecoder: ...


class PrecisionModel(Protocol):
    @property
    def model(self) -> PrecisionBackbone: ...


def _geglu(gate: mx.array, up: mx.array) -> mx.array:
    return gelu_approx(gate) * up


_compiled_geglu = mx.compile(_geglu, shapeless=True)


class Float32Router(Module):
    """Reuse loaded projections; retain router probabilities in float32."""

    def __init__(self, source: RouterSurface) -> None:
        super().__init__()
        self.config: RouterConfig = source.config
        self.eps: float = source.eps
        self._root_size: float = source._root_size
        self.proj: Projection = source.proj
        self.scale: mx.array = source.scale
        self.per_expert_scale: mx.array = source.per_expert_scale
        self._training = source.training

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        x = mx.fast.rms_norm(x, None, self.eps)
        scores = self.proj(x * self.scale * self._root_size)
        indices = mx.argpartition(scores, kth=-self.config.top_k_experts, axis=-1)[
            ..., -self.config.top_k_experts :
        ]
        selected = mx.take_along_axis(scores, indices, axis=-1).astype(mx.float32)
        weights = mx.softmax(selected, axis=-1, precise=True)
        return indices, weights * self.per_expert_scale[indices]


class Float32RouterExperts(Module):
    """Round each weighted expert contribution before the hidden-dtype reduction."""

    def __init__(self, source: ExpertsSurface) -> None:
        super().__init__()
        self.hidden_dims: int = source.hidden_dims
        self.gate_up_proj: ExpertProjection = source.gate_up_proj
        self.down_proj: ExpertProjection = source.down_proj
        self._training = source.training

    def __call__(
        self, x: mx.array, top_k_indices: mx.array, top_k_weights: mx.array
    ) -> mx.array:
        hidden_dtype = x.dtype
        x = mx.expand_dims(x, (-2, -3))
        do_sort = top_k_indices.size >= 64
        indices = top_k_indices
        inv_order: mx.array | None = None
        if do_sort:
            x, indices, inv_order = _gather_sort(x, top_k_indices)
        if self.training:
            indices = mx.stop_gradient(indices)
        gate_up = self.gate_up_proj(x, indices, sorted_indices=do_sort)
        gate = gate_up[..., : self.hidden_dims]
        up = gate_up[..., self.hidden_dims :]
        y = self.down_proj(_compiled_geglu(gate, up), indices, sorted_indices=do_sort)
        if inv_order is not None:
            y = _scatter_unsort(y, inv_order, top_k_indices.shape)
        y = y.squeeze(-2)
        # HF rounds the weighted contributions before adding them to hidden states.
        weighted = (y * top_k_weights[..., None]).astype(hidden_dtype)
        return weighted.sum(axis=-2)


def enable_fp32_router(model: PrecisionModel) -> None:
    """Replace this model's shared encoder/decoder routing modules once."""
    if version("mlx-optiq") != "0.5.12":
        raise RuntimeError("The float32 router policy requires mlx-optiq 0.5.12")
    for layer in model.model.decoder.layers:
        if not isinstance(layer.router, Float32Router):
            layer.router = Float32Router(layer.router)
        if not isinstance(layer.experts, Float32RouterExperts):
            layer.experts = Float32RouterExperts(layer.experts)
