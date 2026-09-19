"""CPU-only fake-stream checks; no real model, tokenizer or generation is invoked."""

import sys
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import mlx.core as mx
import pytest

from diffusion_jev.reasoning import (
    ReasoningEvent,
    ReasoningModel,
    ReasoningTokenizer,
    consume_reasoning,
    preserved_random_state,
    reasoning_prepass,
)
from diffusion_jev.reasoning_boundaries import DecoderOutput, ReasoningBoundary
from diffusion_jev.runtime_types import Backbone, Cache


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    with mx.stream(mx.cpu), preserved_random_state(731):
        yield


@dataclass(frozen=True)
class Event:
    token: int | None
    is_draft: bool = False
    diffusion_block_complete: bool = False
    finish_reason: str | None = None
    prompt_tokens: int = 10
    generation_tokens: int = 2
    prompt_tps: float = 2.0
    generation_tps: float = 4.0
    peak_memory: float = 3.0
    diffusion_canvas_tokens: int = 256
    diffusion_denoising_steps: int = 48
    diffusion_work_tokens: int = 12288

    @property
    def text(self) -> str:
        raise AssertionError("Generated text must never be accessed")

    @property
    def draft_text(self) -> str:
        raise AssertionError("Draft text must never be accessed")


def stream(
    events: list[Event], closed: list[bool], visited: list[int] | None = None
) -> Generator[ReasoningEvent, None, None]:
    try:
        for index, event in enumerate(events):
            if visited is not None:
                visited.append(index)
            yield event
    finally:
        closed.append(True)


def test_natural_close_preserves_exact_ids_and_full_canvas_work_then_closes_generator() -> None:
    closed: list[bool] = []
    visited: list[int] = []
    result = consume_reasoning(
        stream([Event(881111), Event(99), Event(882222)], closed, visited), (10, 20),
        thought_close_id=99, max_tokens=256, seed=0,
    )
    assert result.prefix == (10, 20, 881111, 99)
    assert visited == [0, 1]
    assert closed == [True]
    assert result.metrics.natural_close
    assert not result.metrics.budget_exhausted
    assert not result.metrics.empty
    assert result.metrics.retained_tokens == 1
    assert result.metrics.generation_tokens == 2
    assert result.metrics.denoising_steps == 48
    assert result.metrics.work_tokens == 12288
    assert result.metrics.canvas_tokens == 256
    assert result.metrics.prefill_ms == 5000.0
    assert result.metrics.generation_ms == 500.0
    assert result.metrics.peak_memory_gb == 3.0
    assert "881111" not in repr(result)
    assert "881111" not in result.metrics.model_dump_json()
    assert "prefix" not in result.metrics.model_dump_json()


def test_drafts_blocks_and_final_repeated_tokens_are_not_retained_or_double_counted() -> None:
    closed: list[bool] = []
    result = consume_reasoning(
        stream([
            Event(500, is_draft=True), Event(11),
            Event(11, diffusion_block_complete=True), Event(None), Event(12),
            Event(12, finish_reason="length"),
        ], closed), (10,), thought_close_id=99, max_tokens=256, seed=3,
    )
    assert result.prefix == (10, 11, 12, 99)
    assert result.metrics.retained_tokens == 2
    assert result.metrics.denoising_steps == 48
    assert result.metrics.work_tokens == 12288
    assert result.metrics.budget_exhausted
    assert closed == [True]


@pytest.mark.parametrize("events,natural,eos", [
    ([Event(99)], True, False),
    ([Event(77, diffusion_block_complete=True), Event(77, finish_reason="stop")], False, True),
])
def test_immediate_close_and_eos_produce_empty_closed_thoughts(
    events: list[Event], natural: bool, eos: bool
) -> None:
    closed: list[bool] = []
    result = consume_reasoning(stream(events, closed), (10, 20),
                               thought_close_id=99, max_tokens=256, seed=1)
    assert result.prefix == (10, 20, 99)
    assert result.metrics.empty
    assert result.metrics.retained_tokens == 0
    assert result.metrics.natural_close == natural
    assert result.metrics.eos_stop == eos
    assert closed == [True]


def test_eos_retains_prior_tokens_and_budget_never_reads_a_later_token() -> None:
    closed: list[bool] = []
    eos = consume_reasoning(
        stream([Event(11), Event(77, diffusion_block_complete=True),
                Event(77, finish_reason="stop")], closed), (10,),
        thought_close_id=99, max_tokens=256, seed=1,
    )
    assert eos.prefix == (10, 11, 99)
    assert eos.metrics.eos_stop
    visited: list[int] = []
    budget = consume_reasoning(
        stream([Event(11), Event(12), Event(13), Event(99)], closed, visited), (10,),
        thought_close_id=99, max_tokens=2, seed=1,
    )
    assert budget.prefix == (10, 11, 12, 99)
    assert visited == [0, 1]
    assert budget.metrics.budget_exhausted
    assert not budget.metrics.natural_close
    assert closed == [True, True]


@dataclass(frozen=True)
class Config:
    canvas_length: int = 256


@dataclass(frozen=True)
class Model:
    config: Config = Config()

    @property
    def model(self) -> Backbone:
        raise AssertionError("The fake native stream must not access the backbone")

    def make_cache(self) -> list[Cache]:
        raise AssertionError("The fake native stream must not construct a cache")

    def __call__(
        self, *, cache: list[Cache], canvas_ids: mx.array,
        decoder_attention_mask: dict[str, mx.array | None],
        self_conditioning_logits: mx.array | None = None,
        self_conditioning_embeddings: mx.array | None = None,
    ) -> DecoderOutput:
        raise AssertionError("The fake native stream must not run the model")


class Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        raise AssertionError("Exact input and output IDs must never be retokenized")


@pytest.mark.parametrize("budget", [256, 512])
def test_native_policy_is_explicit_and_input_ids_are_not_reencoded(
    monkeypatch: pytest.MonkeyPatch, budget: int,
) -> None:
    seen: list[bool] = []
    closed: list[bool] = []

    def generate(
        model: ReasoningModel, processor: ReasoningTokenizer, tokenizer: ReasoningTokenizer,
        input_ids: mx.array, pixel_values: None, attention_mask: None, *,
        max_tokens: int, skip_special_token_ids: set[int], temperature: float,
        max_denoising_steps: int, diffusion_full_canvas: bool, diffusion_min_canvas_length: int,
        diffusion_max_canvas_length: int, diffusion_static_cache: bool,
        diffusion_sampler: Literal["entropy-bound"], diffusion_compile: bool,
        diffusion_show_unmasking: bool, diffusion_unmasking_interval: int,
        diffusion_unmasking_width: int, mm_token_type_ids: None, prefill_step_size: int,
    ) -> Generator[ReasoningEvent, None, None]:
        assert isinstance(model, ReasoningBoundary)
        assert processor is tokenizer
        assert input_ids.tolist() == [[101, 102, 103]]
        assert input_ids.dtype == mx.int32
        assert pixel_values is attention_mask is mm_token_type_ids is None
        assert max_tokens == budget
        assert skip_special_token_ids == set()
        assert temperature == 1.0
        assert max_denoising_steps == 48
        assert diffusion_full_canvas
        assert diffusion_min_canvas_length == diffusion_max_canvas_length == 256
        assert not diffusion_static_cache
        assert diffusion_sampler == "entropy-bound"
        assert not diffusion_compile
        assert diffusion_show_unmasking
        assert diffusion_unmasking_interval == diffusion_unmasking_width == 1
        assert prefill_step_size == 512
        seen.append(True)
        yield from stream([Event(99, is_draft=True), Event(11), Event(99)], closed)

    module = ModuleType("optiq.vlm._mlxvlm.generate.diffusion")
    monkeypatch.setattr(module, "stream_diffusion_generate", generate, raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    result = reasoning_prepass(
        Model(), Tokenizer(), (101, 102, 103), thought_close_id=99, max_tokens=budget
    )
    assert result.prefix == (101, 102, 103, 11, 99)
    assert seen == closed == [True]


@pytest.mark.parametrize("budget", [256, 512])
def test_seed_is_repeatable_and_global_rng_restored_after_success_and_failure(
    monkeypatch: pytest.MonkeyPatch, budget: int,
) -> None:
    closed: list[bool] = []
    fail = False

    def random_events(
        model: ReasoningModel, tokenizer: ReasoningTokenizer,
        prefix: tuple[int, ...], max_tokens: int,
    ) -> Generator[ReasoningEvent, None, None]:
        try:
            token = mx.random.randint(1000, 2000, shape=()).item()
            assert isinstance(token, int)
            yield Event(token)
            if max_tokens == 512:
                for _ in range(255):
                    yield Event(token)
                yield Event(token, diffusion_block_complete=True)
                # Block two has encoded the first canvas, but draft counters still lag.
                yield Event(99, is_draft=True)
            if fail:
                raise ValueError("private generated thought sentinel")
            yield Event(
                99, diffusion_canvas_tokens=max_tokens,
                diffusion_denoising_steps=48 * (max_tokens // 256),
                diffusion_work_tokens=12288 * (max_tokens // 256),
            )
        finally:
            closed.append(True)

    monkeypatch.setattr("diffusion_jev.reasoning._native_stream", random_events)
    mx.random.seed(555)
    expected = mx.random.uniform(shape=(4,))
    mx.eval(expected)
    mx.random.seed(555)
    first = reasoning_prepass(
        Model(), Tokenizer(), (10,), thought_close_id=99, seed=17, max_tokens=budget
    )
    second = reasoning_prepass(
        Model(), Tokenizer(), (10,), thought_close_id=99, seed=17, max_tokens=budget
    )
    assert first.prefix == second.prefix
    assert first.metrics.continuation_prefill_tokens == budget - 256
    actual = mx.random.uniform(shape=(4,))
    mx.eval(actual)
    assert mx.array_equal(expected, actual).item()
    fail = True
    mx.random.seed(555)
    with pytest.raises(RuntimeError, match="Reasoning prepass failed") as caught:
        reasoning_prepass(
            Model(), Tokenizer(), (10,), thought_close_id=99, seed=17, max_tokens=budget
        )
    after_error = mx.random.uniform(shape=(4,))
    mx.eval(after_error)
    assert mx.array_equal(expected, after_error).item()
    assert "private generated thought" not in str(caught.value)
    assert caught.value.__suppress_context__
    assert closed == [True, True, True]


def test_invalid_budget_or_prefix_bound_fails_before_starting_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unused_stream(
        model: ReasoningModel, tokenizer: ReasoningTokenizer,
        prefix: tuple[int, ...], max_tokens: int,
    ) -> Generator[ReasoningEvent, None, None]:
        raise AssertionError("Invalid reasoning setup must not invoke generation")

    monkeypatch.setattr("diffusion_jev.reasoning._native_stream", unused_stream)
    for budget in (0, 257, 511, 513):
        with pytest.raises(ValueError, match="budget"):
            reasoning_prepass(Model(), Tokenizer(), (10,), thought_close_id=99, max_tokens=budget)
    with pytest.raises(ValueError, match="token limit"):
        reasoning_prepass(
            Model(), Tokenizer(), (10, 20), thought_close_id=99, max_prompt_tokens=258
        )
    with pytest.raises(ValueError, match="native 256"):
        reasoning_prepass(Model(Config(128)), Tokenizer(), (10,), thought_close_id=99)


def two_blocks(
    *, close_at: int | None = None, eos_at: int | None = None,
) -> list[Event]:
    """Mirror native counters: drafts report only previously completed canvases."""
    events: list[Event] = []
    for block in range(2):
        events.append(Event(
            99, is_draft=True, generation_tokens=block * 256,
            diffusion_canvas_tokens=block * 256, diffusion_denoising_steps=block * 48,
            diffusion_work_tokens=block * 12288,
        ))
        for offset in range(256):
            index = block * 256 + offset
            token = 99 if index == close_at else 1000 + index
            event = Event(
                token, generation_tokens=index + 1,
                diffusion_canvas_tokens=(block + 1) * 256,
                diffusion_denoising_steps=(block + 1) * 48,
                diffusion_work_tokens=(block + 1) * 12288,
            )
            if index == eos_at:
                events.append(Event(
                    77, diffusion_block_complete=True, generation_tokens=index + 1,
                    diffusion_canvas_tokens=(block + 1) * 256,
                    diffusion_denoising_steps=(block + 1) * 48,
                    diffusion_work_tokens=(block + 1) * 12288,
                ))
                events.append(Event(
                    77, finish_reason="stop", generation_tokens=index + 1,
                    diffusion_canvas_tokens=(block + 1) * 256,
                    diffusion_denoising_steps=(block + 1) * 48,
                    diffusion_work_tokens=(block + 1) * 12288,
                ))
                return events
            events.append(event)
        events.append(Event(
            1255 + block * 256, diffusion_block_complete=True,
            generation_tokens=(block + 1) * 256, diffusion_canvas_tokens=(block + 1) * 256,
            diffusion_denoising_steps=(block + 1) * 48,
            diffusion_work_tokens=(block + 1) * 12288,
        ))
    events.append(Event(
        1511, finish_reason="length", generation_tokens=512,
        diffusion_canvas_tokens=512, diffusion_denoising_steps=96, diffusion_work_tokens=24576,
    ))
    return events


@pytest.mark.parametrize("close_at", [0, 255, 256, 511])
def test_512_natural_close_at_exact_block_boundaries(close_at: int) -> None:
    closed: list[bool] = []
    visited: list[int] = []
    result = consume_reasoning(
        stream(two_blocks(close_at=close_at), closed, visited), (10, 20),
        thought_close_id=99, max_tokens=512, seed=0,
    )
    blocks = close_at // 256 + 1
    assert result.prefix == (10, 20, *range(1000, 1000 + close_at), 99)
    assert result.metrics.natural_close
    assert not result.metrics.budget_exhausted
    assert result.metrics.retained_tokens == close_at
    assert result.metrics.canvas_tokens == blocks * 256
    assert result.metrics.continuation_prefill_tokens == (blocks - 1) * 256
    assert result.metrics.denoising_steps == blocks * 48
    assert result.metrics.work_tokens == blocks * 12288
    assert len(visited) == close_at + 1 + blocks + (blocks - 1)
    assert closed == [True]


@pytest.mark.parametrize("eos_at", [0, 255, 256, 511])
def test_512_eos_in_either_block_does_not_retain_terminal_token(eos_at: int) -> None:
    closed: list[bool] = []
    result = consume_reasoning(
        stream(two_blocks(eos_at=eos_at), closed), (10,),
        thought_close_id=99, max_tokens=512, seed=0,
    )
    assert result.prefix == (10, *range(1000, 1000 + eos_at), 99)
    assert result.metrics.eos_stop
    assert not result.metrics.natural_close
    assert not result.metrics.budget_exhausted
    assert result.metrics.continuation_prefill_tokens == (eos_at // 256) * 256
    assert closed == [True]


def test_512_forced_close_counts_two_canvases_and_one_continuation_prefill() -> None:
    closed: list[bool] = []
    visited: list[int] = []
    events = two_blocks()
    result = consume_reasoning(
        stream(events, closed, visited), (10,), thought_close_id=99, max_tokens=512, seed=0,
    )
    assert result.prefix == (10, *range(1000, 1512), 99)
    assert result.metrics.retained_tokens == result.metrics.generation_tokens == 512
    assert result.metrics.canvas_tokens == 512
    assert result.metrics.continuation_prefill_tokens == 256
    assert result.metrics.denoising_steps == 96
    assert result.metrics.work_tokens == 24576
    assert result.metrics.budget_exhausted
    assert not result.metrics.natural_close
    assert len(visited) == len(events) - 2
    assert closed == [True]


def test_512_prefix_capacity_reserves_a_forced_close_before_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[bool] = []

    def generate(
        model: ReasoningModel, tokenizer: ReasoningTokenizer,
        prefix: tuple[int, ...], max_tokens: int,
    ) -> Generator[ReasoningEvent, None, None]:
        assert max_tokens == 512
        yield from stream([Event(99)], closed)

    monkeypatch.setattr("diffusion_jev.reasoning._native_stream", generate)
    with pytest.raises(ValueError, match="token limit"):
        reasoning_prepass(
            Model(), Tokenizer(), (10, 20), thought_close_id=99,
            max_tokens=512, max_prompt_tokens=514,
        )
    assert closed == []
    result = reasoning_prepass(
        Model(), Tokenizer(), (10, 20), thought_close_id=99,
        max_tokens=512, max_prompt_tokens=515,
    )
    assert result.metrics.natural_close
    assert closed == [True]


@pytest.mark.parametrize("budget", [0, 256, 512])
def test_benchmark_parsers_and_report_schemas_accept_supported_budgets(budget: int) -> None:
    from diffusion_jev.accuracy_benchmark import RunConfiguration
    from diffusion_jev.accuracy_benchmark import build_parser as accuracy_parser
    from diffusion_jev.coding_benchmark import CodingReport, CodingRunConfiguration
    from diffusion_jev.coding_benchmark import build_parser as coding_parser

    common = ["--model", "/unused", "--reasoning-tokens", str(budget)]
    accuracy = accuracy_parser().parse_args(
        common + ["--split", "development", "--label", "cpu", "--output", "/unused"]
    )
    coding = coding_parser().parse_args(common)
    assert accuracy.reasoning_tokens == coding.reasoning_tokens == budget
    for schema in (RunConfiguration, CodingRunConfiguration, CodingReport):
        assert schema.model_json_schema()["properties"]["reasoning_tokens"]["enum"] == [
            0, 256, 512,
        ]
