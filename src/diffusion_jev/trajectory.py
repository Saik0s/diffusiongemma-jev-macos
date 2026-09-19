"""Experimental two-pass diffusion with fixed format and unrestricted answer proposals."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Literal, Protocol

from pydantic import Field

from diffusion_jev.reasoning import preserved_random_state
from diffusion_jev.schemas import NonnegativeFloat, StrictModel

if TYPE_CHECKING:
    import mlx.core as mx

    from diffusion_jev.runtime_types import Cache


class TrajectoryEmbedding(Protocol):
    @property
    def weight(self) -> mx.array: ...

    def as_linear(self, hidden: mx.array) -> mx.array: ...


class TrajectoryDecoder(Protocol):
    @property
    def embed_tokens(self) -> TrajectoryEmbedding: ...

    @property
    def embed_scale(self) -> float: ...

    def __call__(
        self, canvas_ids: mx.array, *, cache: list[Cache],
        self_conditioning_embeddings: mx.array | None = None,
    ) -> mx.array: ...


class TrajectoryBackbone(Protocol):
    @property
    def decoder(self) -> TrajectoryDecoder: ...


class TrajectoryModel(Protocol):
    @property
    def model(self) -> TrajectoryBackbone: ...

    def _softcap(self, logits: mx.array) -> mx.array: ...


class TrajectoryMetrics(StrictModel):
    seed: Annotated[int, Field(ge=0, le=2**32 - 1)]
    steps: Literal[2] = 2
    schedule_steps: Literal[48] = 48
    transition_temperature: float = 0.8
    entropy_bound: float = 0.1
    accepted_answer_positions: Annotated[int, Field(ge=0)]
    answer_positions: Annotated[int, Field(ge=1)]
    canvas_tokens: Annotated[int, Field(ge=1)]
    work_tokens: Annotated[int, Field(ge=1)]
    wall_ms: NonnegativeFloat
    peak_memory_gb: NonnegativeFloat


@dataclass(frozen=True)
class TrajectoryResult:
    logits: mx.array = field(repr=False)
    metrics: TrajectoryMetrics


def structured_trajectory(
    model: TrajectoryModel, canvas_tokens: tuple[int, ...], answer_positions: tuple[int, ...],
    cache: list[Cache], *, seed: int,
) -> TrajectoryResult:
    """Read the final temperature-1 logits after exactly two decoder calls.

    The caller supplies the already-seeded canvas and holds its single-model lock. Full-vocabulary
    self-conditioning remains native at every row; hard nonanswer IDs are restored between passes.
    This clamping is a structured adaptation, not exact parity with the upstream free-canvas loop.
    Only ``result.metrics`` may be serialized. The prompt cache is never committed or rebuilt here.
    """
    if not 0 <= seed <= 2**32 - 1 or not 1 <= len(canvas_tokens) <= 256:
        raise ValueError("Trajectory requires a uint32 seed and a 1..256-token canvas")
    if (not answer_positions or len(set(answer_positions)) != len(answer_positions)
            or any(not 0 <= position < len(canvas_tokens) for position in answer_positions)):
        raise ValueError("Trajectory answer positions must be unique valid canvas positions")
    if any(token < 0 for token in canvas_tokens):
        raise ValueError("Trajectory requires nonnegative canvas tokens")

    import mlx.core as mx
    from optiq.vlm._mlxvlm.generate.diffusion import (
        _diffusion_entropy_and_soft_embeddings,
        _diffusion_entropy_transfer_mask,
        _diffusion_initialize_canvas,
        _diffusion_linear_temperature,
        _diffusion_sample_canvas,
        _diffusion_soft_embedding_weight,
    )

    started = time.perf_counter()
    try:
        with preserved_random_state(seed):
            decoder = model.model.decoder
            embedding = decoder.embed_tokens
            if max(canvas_tokens) >= embedding.weight.shape[0]:
                raise ValueError("Trajectory canvas token exceeds vocabulary")
            initial = mx.array([canvas_tokens], dtype=mx.int32)
            hidden = decoder(initial, cache=cache)
            logits = model._softcap(embedding.as_linear(hidden))
            temperature = _diffusion_linear_temperature(48, 48, {"t_min": 0.4, "t_max": 0.8})
            if temperature is None or temperature <= 0:
                raise ValueError("Invalid trajectory schedule")
            scheduled = logits / temperature
            weight = _diffusion_soft_embedding_weight(embedding)
            entropy, conditioning = _diffusion_entropy_and_soft_embeddings(
                scheduled, weight, decoder.embed_scale
            )
            accepted = _diffusion_entropy_transfer_mask(entropy, 0.1)
            proposed = _diffusion_sample_canvas(scheduled, mx.int32, 1.0)
            noise = _diffusion_initialize_canvas(1, len(canvas_tokens), weight.shape[0], mx.int32)
            slots = mx.array(
                [[index in answer_positions for index in range(len(canvas_tokens))]], dtype=mx.bool_
            )
            next_canvas = mx.where(slots, mx.where(accepted, proposed, noise), initial)
            accepted_count = mx.sum(accepted & slots)
            mx.eval(next_canvas, conditioning, accepted_count)
            count = accepted_count.item()
            if not isinstance(count, int):
                raise ValueError("Invalid trajectory acceptance count")
            # Drop full-vocabulary first-pass graphs before evaluating the second projection.
            del hidden, logits, scheduled, weight, entropy, accepted, proposed, noise
            final_hidden = decoder(
                next_canvas, cache=cache, self_conditioning_embeddings=conditioning
            )
            final_logits = model._softcap(embedding.as_linear(final_hidden))
            mx.eval(final_logits)
        return TrajectoryResult(
            final_logits,
            TrajectoryMetrics(
                seed=seed, transition_temperature=temperature, accepted_answer_positions=count,
                answer_positions=len(answer_positions), canvas_tokens=len(canvas_tokens),
                work_tokens=2 * len(canvas_tokens),
                wall_ms=(time.perf_counter() - started) * 1000,
                peak_memory_gb=mx.get_peak_memory() / 1e9,
            ),
        )
    except Exception:
        raise RuntimeError("Structured trajectory failed") from None
