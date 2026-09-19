"""CPU-only synthetic trajectory checks; no model weights or inference workload."""

import math
from collections.abc import Iterator
from dataclasses import dataclass

import mlx.core as mx
import pytest

from diffusion_jev.reasoning import preserved_random_state
from diffusion_jev.runtime_types import Cache
from diffusion_jev.trajectory import structured_trajectory


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    # The installed SDK cannot link MLX CPU JIT kernels; exercise the same native ops eagerly.
    mx.disable_compile()
    try:
        with mx.stream(mx.cpu), preserved_random_state(417):
            yield
    finally:
        mx.enable_compile()


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_native_soft_embeddings_use_full_distribution_embedding_dtype_and_scale(
    dtype: mx.Dtype,
) -> None:
    from optiq.vlm._mlxvlm.generate.diffusion import _diffusion_entropy_and_soft_embeddings

    logits = mx.log(mx.array([[[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]]]))
    table = mx.array([[1, 2], [3, 4], [5, 6]], dtype=dtype)
    entropy, soft = _diffusion_entropy_and_soft_embeddings(logits, table, 2.0)
    mx.eval(entropy, soft)
    assert soft.dtype == dtype
    assert mx.allclose(soft, mx.array([[[5.0, 7.0], [6.0, 8.0]]]), atol=1e-5).item()
    expected_entropy = -0.5 * math.log(0.5) - 0.5 * math.log(0.25)
    assert mx.allclose(entropy, mx.array([[expected_entropy] * 2]), atol=1e-6).item()


def test_native_entropy_mask_is_global_cumulative_budget_not_per_position_threshold() -> None:
    from optiq.vlm._mlxvlm.generate.diffusion import _diffusion_entropy_transfer_mask

    mask = _diffusion_entropy_transfer_mask(mx.array([[0.7, 0.01, 0.2, 0.08]]), 0.1)
    assert mask.tolist() == [[False, True, True, True]]
    # The lowest-entropy position is selected even if its own entropy exceeds the bound.
    assert _diffusion_entropy_transfer_mask(mx.array([[8.0, 9.0]]), 0.1).tolist() == [[True, False]]


class Embedding:
    def __init__(self) -> None:
        self.weight = mx.eye(7, dtype=mx.float32)
        self.rows: list[int] = []

    def as_linear(self, hidden: mx.array) -> mx.array:
        self.rows.append(hidden.shape[1])
        return hidden @ self.weight.T


class PromptCache:
    def __init__(self) -> None:
        self.offset = 3
        self.state = (mx.arange(3, dtype=mx.float32), mx.ones((3,), dtype=mx.float32))


@dataclass(frozen=True)
class Call:
    tokens: mx.array
    conditioning: mx.array | None
    cache: list[Cache]


class Decoder:
    def __init__(self, *, uniform: bool = False, fail_second: bool = False) -> None:
        self.embed_tokens = Embedding()
        self.embed_scale = math.sqrt(7)
        self.calls: list[Call] = []
        self.first = mx.zeros((1, 4, 7), dtype=mx.float32)
        if not uniform:
            self.first[:, :, 6] = 50.0
        self.final = mx.arange(28, dtype=mx.float32).reshape(1, 4, 7) / 10
        self.fail_second = fail_second

    def __call__(
        self, canvas_ids: mx.array, *, cache: list[Cache],
        self_conditioning_embeddings: mx.array | None = None,
    ) -> mx.array:
        self.calls.append(Call(mx.array(canvas_ids), self_conditioning_embeddings, cache))
        if len(self.calls) == 1:
            assert self_conditioning_embeddings is None
            return self.first
        if self.fail_second:
            raise ValueError("private trajectory content sentinel")
        assert len(self.calls) == 2
        assert self_conditioning_embeddings is not None
        return self.final


class Backbone:
    def __init__(self, decoder: Decoder) -> None:
        self.decoder = decoder

    @property
    def encoder(self) -> None:
        raise AssertionError("A read trajectory must never call the encoder")


class Model:
    def __init__(self, *, uniform: bool = False, fail_second: bool = False) -> None:
        self.model = Backbone(Decoder(uniform=uniform, fail_second=fail_second))
        self.softcap_calls = 0

    def _softcap(self, logits: mx.array) -> mx.array:
        self.softcap_calls += 1
        return mx.tanh(logits / 30) * 30


def test_exactly_two_native_passes_preserve_cache_format_and_unrestricted_answer_tokens() -> None:
    model = Model()
    cached = PromptCache()
    original = tuple(mx.array(item) for item in cached.state)
    cache: list[Cache] = [cached]
    result = structured_trajectory(model, (0, 1, 2, 3), (1, 3), cache, seed=9)
    decoder = model.model.decoder
    assert len(decoder.calls) == model.softcap_calls == 2
    assert decoder.embed_tokens.rows == [4, 4]
    assert all(call.cache is cache for call in decoder.calls)
    assert cached.offset == 3
    assert all(mx.array_equal(a, b).item() for a, b in zip(original, cached.state, strict=True))
    assert decoder.calls[0].tokens.tolist() == [[0, 1, 2, 3]]
    assert decoder.calls[1].tokens.tolist() == [[0, 6, 2, 6]]
    # Proposals may be outside illustrative allowed labels {0, 1}; hard format remains fixed.
    assert result.metrics.accepted_answer_positions == 2
    assert result.metrics.steps == 2
    assert result.metrics.schedule_steps == 48
    assert result.metrics.transition_temperature == 0.8
    assert result.metrics.canvas_tokens == 4
    assert result.metrics.work_tokens == 8
    assert result.metrics.answer_positions == 2
    expected_logits = mx.tanh(decoder.final / 30) * 30
    assert mx.array_equal(result.logits, expected_logits).item()
    assert not mx.array_equal(result.logits, expected_logits / 0.8).item()
    expected_signal = mx.zeros((1, 4, 7), dtype=mx.float32)
    expected_signal[:, :, 6] = math.sqrt(7)
    assert decoder.calls[1].conditioning is not None
    assert mx.allclose(decoder.calls[1].conditioning, expected_signal, atol=1e-6).item()
    assert "logits" not in result.metrics.model_dump_json()
    assert "tokens" not in repr(result).split("metrics=")[0]
    assert "logits" not in repr(result)


def test_unaccepted_answers_are_renoised_but_fixed_positions_are_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from optiq.vlm._mlxvlm.generate import diffusion

    sampled_shapes: list[tuple[int, ...]] = []

    def proposals(logits: mx.array, dtype: mx.Dtype, temperature: float) -> mx.array:
        sampled_shapes.append(logits.shape)
        assert temperature == 1.0
        return mx.full((1, 4), 6, dtype=dtype)

    def renoise(batch_size: int, width: int, vocab_size: int, dtype: mx.Dtype) -> mx.array:
        assert (batch_size, width, vocab_size) == (1, 4, 7)
        return mx.full((1, 4), 5, dtype=dtype)

    monkeypatch.setattr(diffusion, "_diffusion_sample_canvas", proposals)
    monkeypatch.setattr(diffusion, "_diffusion_initialize_canvas", renoise)
    model = Model(uniform=True)
    structured_trajectory(model, (0, 1, 2, 3), (1, 3), [], seed=0)
    # Uniform entropy accepts only one canvas position. The chosen answer slots are otherwise noise.
    from optiq.vlm._mlxvlm.generate.diffusion import _diffusion_entropy_transfer_mask

    mask = _diffusion_entropy_transfer_mask(mx.full((1, 4), math.log(7)), 0.1)
    expected = mx.where(
        mx.array([[False, True, False, True]]),
        mx.where(mask, mx.full((1, 4), 6), mx.full((1, 4), 5)),
        mx.array([[0, 1, 2, 3]]),
    )
    assert mx.array_equal(model.model.decoder.calls[1].tokens, expected).item()
    assert sampled_shapes == [(1, 4, 7)]


def test_transition_seed_is_repeatable_and_rng_restores_on_success_and_error() -> None:
    mx.random.seed(881)
    expected = mx.random.uniform(shape=(4,))
    mx.eval(expected)
    mx.random.seed(881)
    first, second = Model(uniform=True), Model(uniform=True)
    structured_trajectory(first, (0, 1, 2, 3), (1, 3), [], seed=22)
    structured_trajectory(second, (0, 1, 2, 3), (1, 3), [], seed=22)
    assert mx.array_equal(
        first.model.decoder.calls[1].tokens, second.model.decoder.calls[1].tokens
    ).item()
    assert mx.array_equal(expected, mx.random.uniform(shape=(4,))).item()
    mx.random.seed(881)
    with pytest.raises(RuntimeError, match="Structured trajectory failed") as caught:
        structured_trajectory(Model(fail_second=True), (0, 1, 2, 3), (1, 3), [], seed=22)
    assert mx.array_equal(expected, mx.random.uniform(shape=(4,))).item()
    assert "private trajectory content" not in str(caught.value)
    assert caught.value.__suppress_context__


def test_invalid_positions_fail_before_decoder_work() -> None:
    model = Model()
    for positions in ((), (1, 1), (-1,), (4,)):
        with pytest.raises(ValueError, match="positions"):
            structured_trajectory(model, (0, 1, 2, 3), positions, [], seed=0)
    assert model.model.decoder.calls == []
