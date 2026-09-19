"""Materialize native reasoning inputs before building the next decoder graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypedDict, Unpack, runtime_checkable

if TYPE_CHECKING:
    import mlx.core as mx

    from diffusion_jev.reasoning import ReasoningConfig, ReasoningModel
    from diffusion_jev.runtime_types import Backbone, Cache


class DecoderOutput(Protocol):
    @property
    def logits(self) -> mx.array: ...


class ConditioningInputs(TypedDict, total=False):
    self_conditioning_logits: mx.array | None
    self_conditioning_embeddings: mx.array | None


@runtime_checkable
class _CallableModel(Protocol):
    def __call__(
        self, *, cache: list[Cache], canvas_ids: mx.array,
        decoder_attention_mask: dict[str, mx.array | None],
        self_conditioning_logits: mx.array | None = None,
        self_conditioning_embeddings: mx.array | None = None,
    ) -> DecoderOutput: ...

    def make_cache(self) -> list[Cache]: ...


class ReasoningBoundary:
    """Preserve the loaded model while separating lazy input evaluation."""

    def __init__(self, source: ReasoningModel) -> None:
        # Check methods only; MLX keeps config/backbone in a dynamic module mapping.
        if not isinstance(source, _CallableModel):
            raise ValueError("Reasoning boundaries require a callable diffusion model")
        self._source = source
        self._callable: _CallableModel = source

    @property
    def config(self) -> ReasoningConfig:
        return self._source.config

    @property
    def model(self) -> Backbone:
        return self._source.model

    def make_cache(self) -> list[Cache]:
        return self._source.make_cache()

    def __call__(
        self, *, cache: list[Cache], canvas_ids: mx.array,
        decoder_attention_mask: dict[str, mx.array | None],
        **inputs: Unpack[ConditioningInputs],
    ) -> DecoderOutput:
        import mlx.core as mx

        # Native continuation prefill and soft embeddings otherwise remain lazy
        # until the following decoder step's progress evaluation.
        mx.eval([array for item in cache for array in item.state])
        conditioning = [array for array in (
            inputs.get("self_conditioning_logits"), inputs.get("self_conditioning_embeddings"),
        ) if array is not None]
        if conditioning:
            mx.eval(conditioning)
        return self._callable(
            cache=cache, canvas_ids=canvas_ids, decoder_attention_mask=decoder_attention_mask,
            **inputs,
        )
