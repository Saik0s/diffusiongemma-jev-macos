"""Experimental token-preserving reasoning; generated content stays in memory."""

from __future__ import annotations

import math
import time
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Protocol, runtime_checkable

from pydantic import Field

from diffusion_jev.schemas import StrictModel

if TYPE_CHECKING:
    import mlx.core as mx

    from diffusion_jev.runtime_types import Backbone, Cache


class ReasoningConfig(Protocol):
    @property
    def canvas_length(self) -> int: ...


class ReasoningModel(Protocol):
    @property
    def config(self) -> ReasoningConfig: ...

    @property
    def model(self) -> Backbone: ...

    def make_cache(self) -> list[Cache]: ...


class ReasoningTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...


class ReasoningEvent(Protocol):
    @property
    def token(self) -> int | None: ...
    @property
    def is_draft(self) -> bool: ...
    @property
    def diffusion_block_complete(self) -> bool: ...
    @property
    def finish_reason(self) -> str | None: ...
    @property
    def prompt_tokens(self) -> int: ...
    @property
    def generation_tokens(self) -> int: ...
    @property
    def prompt_tps(self) -> float: ...
    @property
    def generation_tps(self) -> float: ...
    @property
    def peak_memory(self) -> float: ...
    @property
    def diffusion_canvas_tokens(self) -> int: ...
    @property
    def diffusion_denoising_steps(self) -> int: ...
    @property
    def diffusion_work_tokens(self) -> int: ...


Nonnegative = Annotated[float, Field(ge=0, allow_inf_nan=False)]
Count = Annotated[int, Field(ge=0)]


class ReasoningMetrics(StrictModel):
    seed: Count
    max_tokens: Count
    retained_tokens: Count = 0
    generation_tokens: Count = 0
    canvas_tokens: Count = 0
    continuation_prefill_tokens: Count = 0
    denoising_steps: Count = 0
    work_tokens: Count = 0
    prefill_ms: Nonnegative = 0.0
    generation_ms: Nonnegative = 0.0
    wall_ms: Nonnegative = 0.0
    peak_memory_gb: Nonnegative = 0.0
    natural_close: bool = False
    eos_stop: bool = False
    budget_exhausted: bool = False
    empty: bool = True


@dataclass(frozen=True)
class ReasoningResult:
    prefix: tuple[int, ...] = field(repr=False)
    metrics: ReasoningMetrics


@runtime_checkable
class RandomState(Protocol):
    def __getitem__(self, index: int) -> mx.array: ...


@contextmanager
def preserved_random_state(seed: int) -> Iterator[None]:
    """Restore the existing MLX key object; callers must hold the model lock."""
    import mlx.core as mx

    state = mx.random.state
    if not isinstance(state, RandomState) or not isinstance(state[0], mx.array):
        raise RuntimeError("Unsupported MLX random state interface")
    saved = mx.array(state[0])
    mx.eval(saved)
    try:
        mx.random.seed(seed)
        yield
    finally:
        # MLX 0.32.2 exposes a read-only container with a mutable current key array.
        current = state[0]
        current[...] = saved
        mx.eval(current)


def _native_stream(
    model: ReasoningModel, tokenizer: ReasoningTokenizer, prefix: tuple[int, ...], max_tokens: int
) -> Generator[ReasoningEvent, None, None]:
    import mlx.core as mx
    from optiq.vlm._mlxvlm.generate.diffusion import stream_diffusion_generate

    from diffusion_jev.reasoning_boundaries import ReasoningBoundary

    # Native progress events materialize each step instead of one 48-step Metal graph.
    # The consumer discards drafts; no progress text is printed or retained.
    return stream_diffusion_generate(
        ReasoningBoundary(model), tokenizer, tokenizer,
        mx.array([prefix], dtype=mx.int32), None, None,
        max_tokens=max_tokens, skip_special_token_ids=set(), temperature=1.0,
        max_denoising_steps=48, diffusion_full_canvas=True,
        diffusion_min_canvas_length=256, diffusion_max_canvas_length=256,
        diffusion_static_cache=False, diffusion_sampler="entropy-bound", diffusion_compile=False,
        diffusion_show_unmasking=True, diffusion_unmasking_interval=1,
        diffusion_unmasking_width=1, mm_token_type_ids=None, prefill_step_size=512,
    )


def _elapsed_ms(tokens: int, rate: float) -> float:
    if tokens < 0 or not math.isfinite(rate) or rate < 0:
        raise ValueError("Invalid reasoning work counters")
    return 1000 * tokens / rate if rate > 0 else 0.0


def consume_reasoning(
    events: Generator[ReasoningEvent, None, None], generation_prefix: tuple[int, ...],
    *, thought_close_id: int, max_tokens: int, seed: int,
) -> ReasoningResult:
    """Keep committed IDs only; draft, block and terminal events repeat tokens."""
    started = time.perf_counter()
    retained: list[int] = []
    metrics = ReasoningMetrics(seed=seed, max_tokens=max_tokens)
    try:
        for event in events:
            # These are cumulative canvas-work counters, including work after an early close.
            metrics.generation_tokens = max(metrics.generation_tokens, event.generation_tokens)
            metrics.canvas_tokens = max(metrics.canvas_tokens, event.diffusion_canvas_tokens)
            # Native counters include only completed 256-token canvases. Each later
            # completed canvas proves the preceding canvas was encoded into the cache.
            # Drafts lag that work; failed streams return no metrics, not zero work.
            metrics.continuation_prefill_tokens = max(0, metrics.canvas_tokens - 256)
            metrics.denoising_steps = max(metrics.denoising_steps, event.diffusion_denoising_steps)
            metrics.work_tokens = max(metrics.work_tokens, event.diffusion_work_tokens)
            metrics.peak_memory_gb = max(metrics.peak_memory_gb, event.peak_memory)
            metrics.prefill_ms = max(
                metrics.prefill_ms, _elapsed_ms(event.prompt_tokens, event.prompt_tps)
            )
            metrics.generation_ms = max(
                metrics.generation_ms, _elapsed_ms(event.generation_tokens, event.generation_tps)
            )
            if event.finish_reason is not None:
                metrics.eos_stop = event.finish_reason == "stop"
                metrics.budget_exhausted = event.finish_reason == "length"
                break
            if event.is_draft or event.diffusion_block_complete or event.token is None:
                continue
            if event.token == thought_close_id:
                metrics.natural_close = True
                break
            if event.token < 0:
                raise ValueError("Invalid committed token")
            retained.append(event.token)
            if len(retained) == max_tokens:
                metrics.budget_exhausted = True
                break
    finally:
        events.close()
    metrics.retained_tokens = len(retained)
    metrics.empty = not retained
    metrics.wall_ms = (time.perf_counter() - started) * 1000
    return ReasoningResult(generation_prefix + tuple(retained) + (thought_close_id,), metrics)


def reasoning_prepass(
    model: ReasoningModel, tokenizer: ReasoningTokenizer, generation_prefix: tuple[int, ...], *,
    thought_close_id: int, max_tokens: int = 256, seed: int = 0, max_prompt_tokens: int = 8192,
) -> ReasoningResult:
    """Generate one bounded thought under the caller's existing single-model lock.

    The supplied prefix already contains the thinking-enabled template and thought-open IDs.
    The returned exact prefix ends with one close marker and requires a fresh causal prefill.
    Only ``result.metrics`` may be serialized, never the result or prefix itself.
    """
    if (not (1 <= max_tokens <= 256 or max_tokens == 512)
            or not 0 <= seed <= 2**32 - 1):
        raise ValueError("Reasoning requires a 1..256 or 512 token budget and a uint32 seed")
    if model.config.canvas_length != 256:
        raise ValueError("Reasoning policy requires a native 256-token canvas")
    if (not generation_prefix or thought_close_id < 0
            or any(token < 0 for token in generation_prefix)):
        raise ValueError("Reasoning requires a valid generation prefix and closing token")
    if len(generation_prefix) + max_tokens + 1 > max_prompt_tokens:
        raise ValueError("Worst-case reasoning prefix exceeds the configured token limit")
    started = time.perf_counter()
    try:
        with preserved_random_state(seed):
            events = _native_stream(model, tokenizer, generation_prefix, max_tokens)
            result = consume_reasoning(
                events, generation_prefix, thought_close_id=thought_close_id,
                max_tokens=max_tokens, seed=seed,
            )
        result.metrics.wall_ms = (time.perf_counter() - started) * 1000
        return result
    except Exception:
        # Upstream failures can include tokenizer/model text; keep that out of reports and logs.
        raise RuntimeError("Reasoning prepass failed") from None
