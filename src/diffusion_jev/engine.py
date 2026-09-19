"""One-step structured reads on the existing OptiQ model, without text decoding."""

import json
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import mlx.core as mx
from optiq.vlm.diffusion_gemma import load

from diffusion_jev.canvas import (
    THOUGHT_CLOSE,
    THOUGHT_OPEN,
    Canvas,
    compile_canvas,
    make_answer,
    system_prompt,
)
from diffusion_jev.component_source import RestorationInfo
from diffusion_jev.precision import enable_fp32_router
from diffusion_jev.projection import float32_label_embeddings
from diffusion_jev.runtime_types import Cache, DiffusionModel, Tokenizer
from diffusion_jev.schemas import (
    Answer,
    DecisionRequest,
    DecisionResponse,
    Question,
    Usage,
)


@dataclass(frozen=True)
class PreparedRead:
    prompt: tuple[int, ...]
    canvas: Canvas
    questions: dict[str, Question]
    thought_close_id: int | None = None


@dataclass(frozen=True)
class ReadDiagnostics:
    """Aggregate formatting evidence, never a claim that an answer is correct."""

    slots: int
    allowed_label_mass_min: float
    allowed_label_mass_mean: float
    invalid_argmax_count: int


class LocalEngine:
    """Construct and call on one thread; MLX lazy buffers keep thread-local streams."""

    model_id = "diffusiongemma-local"

    def __init__(
        self,
        model_path: Path,
        max_prompt_tokens: int = 8192,
        *,
        router_precision: Literal["native", "fp32"] = "native",
        diagnostics: bool = False,
        cache_limit_bytes: int | None = None,
        reasoning_tokens: int = 0,
        trajectory_steps: int = 1,
        router_bf16_source: Path | None = None,
        default_samples: int = 1,
    ) -> None:
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise FileNotFoundError("A local model directory containing config.json is required")
        if max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")
        if router_precision not in ("native", "fp32"):
            raise ValueError("Unsupported router precision policy")
        if reasoning_tokens not in (0, 256, 512):
            raise ValueError("Reasoning budget must be 0, 256 or 512 tokens")
        if trajectory_steps not in (1, 2):
            raise ValueError("Structured trajectory must use 1 or 2 steps")
        if not 1 <= default_samples <= 8:
            raise ValueError("default_samples must be between 1 and 8")
        if cache_limit_bytes is not None:
            if cache_limit_bytes < 0:
                raise ValueError("cache_limit_bytes must be nonnegative")
            mx.set_cache_limit(cache_limit_bytes)
        mx.reset_peak_memory()
        started = time.perf_counter()
        self.model: DiffusionModel
        self.tokenizer: Tokenizer
        self.model, self.tokenizer = load(str(model_path.resolve()))
        self.restoration_info: RestorationInfo | None = None
        if router_bf16_source is not None:
            from diffusion_jev.components import restore_bf16_router_projections

            self.restoration_info = restore_bf16_router_projections(self.model, router_bf16_source)
        if router_precision == "fp32":
            enable_fp32_router(self.model)
        self.load_seconds = time.perf_counter() - started
        self.load_peak_memory_gb = mx.get_peak_memory() / 1e9
        self.load_active_memory_gb = mx.get_active_memory() / 1e9
        self.load_cache_memory_gb = mx.get_cache_memory() / 1e9
        self.max_prompt_tokens = max_prompt_tokens
        self.router_precision = router_precision
        self.diagnostics = diagnostics
        self.cache_limit_bytes = cache_limit_bytes
        self.reasoning_tokens = reasoning_tokens
        self.trajectory_steps = trajectory_steps
        self.default_samples = default_samples
        self.trajectory_accepted_slots = 0
        self.last_diagnostics: ReadDiagnostics | None = None
        self._lock = threading.Lock()

    def _prepare(self, request: DecisionRequest) -> list[PreparedRead]:
        if self.trajectory_steps == 2 and request.options.projection != "full":
            raise ValueError("Two-step trajectory requires full projection")
        # Keep the caller's field order. Sorting moved "t" to the end of every
        # timestamped event, which is not the evidence the caller wrote.
        state = json.dumps(request.state, ensure_ascii=False, allow_nan=False)
        groups = (
            [request.questions]
            if request.options.mode == "packed"
            else [{key: question} for key, question in request.questions.items()]
        )
        prepared: list[PreparedRead] = []
        for questions in groups:
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt(questions)},
                    {"role": "user", "content": state},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.reasoning_tokens > 0,
            )
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            thought_close_id = None
            if self.reasoning_tokens:
                opening = self.tokenizer.encode(THOUGHT_OPEN, add_special_tokens=False)
                closing = self.tokenizer.encode(THOUGHT_CLOSE, add_special_tokens=False)
                scaffold = self.tokenizer.encode(
                    THOUGHT_OPEN + THOUGHT_CLOSE, add_special_tokens=False,
                )
                if len(closing) != 1 or opening + closing != scaffold:
                    raise ValueError("Unsupported tokenizer thought markers")
                thought_close_id = closing[0]
                tokens += opening
            reserve = self.reasoning_tokens + 1 if self.reasoning_tokens else 0
            if len(tokens) + reserve > self.max_prompt_tokens:
                raise ValueError("Prompt exceeds the configured token limit")
            canvas = compile_canvas(
                self.tokenizer,
                questions,
                width=request.options.canvas_length,
                pad_token_id=self.model.config.text_config.pad_token_id,
                include_empty_thought=not self.reasoning_tokens,
            )
            prepared.append(PreparedRead(tuple(tokens), canvas, questions, thought_close_id))
        return prepared

    def _prefill(self, prompt: tuple[int, ...]) -> list[Cache]:
        cache = self.model.make_cache()
        # Bound prefill activations while preserving the encoder's causal KV history.
        for offset in range(0, len(prompt), 512):
            _, cache = self.model.model.encoder(
                mx.array([prompt[offset : offset + 512]], dtype=mx.int32), cache=cache
            )
            mx.eval([item.state for item in cache])
        return cache

    def _read(
        self, prepared: PreparedRead, cache: list[Cache], request: DecisionRequest
    ) -> list[list[float]]:
        canvas = prepared.canvas
        embedding = self.model.model.decoder.embed_tokens
        alias_ids = sorted({token for slot in canvas.slots for token in slot.token_ids})
        alias_indices = {token: index for index, token in enumerate(alias_ids)}
        label_tokens = mx.array(alias_ids, dtype=mx.int32)
        alias_embeddings = (
            float32_label_embeddings(embedding, label_tokens)
            if request.options.projection == "labels-fp32"
            else embedding(label_tokens)
        )
        positions = mx.array([slot.position for slot in canvas.slots], dtype=mx.int32)
        accumulated = [[0.0] * len(slot.token_ids) for slot in canvas.slots]
        for sample in range(request.options.samples):
            tokens = canvas.seeded(
                request.options.seed + sample, self.model.config.text_config.vocab_size
            )
            full_logits = None
            if self.trajectory_steps == 2:
                from diffusion_jev.trajectory import structured_trajectory

                trajectory = structured_trajectory(
                    self.model, tuple(tokens), tuple(slot.position for slot in canvas.slots),
                    cache, seed=(request.options.seed + sample) % (2**32),
                )
                self.trajectory_accepted_slots += trajectory.metrics.accepted_answer_positions
                full_logits = trajectory.logits[:, positions, :]
            else:
                hidden = self.model.model.decoder(mx.array([tokens], dtype=mx.int32), cache=cache)
                answer_hidden = hidden[:, positions, :]
                if request.options.projection == "full":
                    # Preserve the native full-canvas projection's matrix shape and rounding.
                    full_logits = self.model._softcap(embedding.as_linear(hidden))[:, positions, :]
                elif self.diagnostics:
                    # Diagnostics need every vocabulary label, but only at answer positions.
                    full_logits = self.model._softcap(embedding.as_linear(answer_hidden))
            if request.options.projection == "full":
                assert full_logits is not None
                logits = full_logits
            else:
                # The tied embedding table is the output head. Gather before projecting.
                if request.options.projection == "labels-fp32":
                    answer_hidden = answer_hidden.astype(mx.float32)
                logits = self.model._softcap(answer_hidden @ alias_embeddings.T)
            distributions: list[mx.array] = []
            for index, slot in enumerate(canvas.slots):
                if request.options.projection == "full":
                    selected = logits[0, index, mx.array(slot.token_ids, dtype=mx.int32)]
                else:
                    selected = logits[
                        0,
                        index,
                        mx.array(
                            [alias_indices[token] for token in slot.token_ids], dtype=mx.int32
                        ),
                    ]
                distributions.append(mx.softmax(selected.astype(mx.float32)))
            # Synchronize GPU work before measuring and before dropping this sample's graph.
            mx.eval(distributions)
            for total, distribution in zip(accumulated, distributions, strict=True):
                for index in range(len(total)):
                    value = distribution[index].item()
                    if isinstance(value, complex):
                        raise ValueError("Unexpected complex probability")
                    total[index] += float(value) / request.options.samples
            if self.diagnostics:
                assert full_logits is not None
                self._record_diagnostics(full_logits, canvas)
        return accumulated

    def _record_diagnostics(self, logits: mx.array, canvas: Canvas) -> None:
        masses: list[mx.array] = []
        valid: list[mx.array] = []
        for index, slot in enumerate(canvas.slots):
            row = logits[0, index]
            labels = mx.array(slot.token_ids, dtype=mx.int32)
            masses.append(mx.exp(mx.logsumexp(row[labels]) - mx.logsumexp(row)))
            valid.append(mx.any(labels == mx.argmax(row)))
        mx.eval(masses, valid)
        values: list[float] = []
        for mass in masses:
            value = mass.item()
            if isinstance(value, complex):
                raise ValueError("Unexpected complex label mass")
            values.append(min(1.0, max(0.0, float(value))))
        count = len(values)
        total = sum(values)
        minimum = min(values)
        invalid = sum(not bool(item.item()) for item in valid)
        previous = self.last_diagnostics
        if previous is not None:
            total += previous.allowed_label_mass_mean * previous.slots
            count += previous.slots
            minimum = min(minimum, previous.allowed_label_mass_min)
            invalid += previous.invalid_argmax_count
        self.last_diagnostics = ReadDiagnostics(
            slots=count,
            allowed_label_mass_min=minimum,
            allowed_label_mass_mean=total / count,
            invalid_argmax_count=invalid,
        )

    def _apply_sampling_profile(self, request: DecisionRequest) -> DecisionRequest:
        """Fill in the server's sampling profile only when the caller omitted `samples`."""
        if "samples" in request.options.model_fields_set:
            return request
        options = request.options.model_copy(update={"samples": self.default_samples})
        return request.model_copy(update={"options": options})

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        if request.model != self.model_id:
            raise ValueError("Unknown model")
        request = self._apply_sampling_profile(request)
        with self._lock:
            self.last_diagnostics = None
            self.trajectory_accepted_slots = 0
            started = time.perf_counter()
            prepared = self._prepare(request)
            mx.reset_peak_memory()
            answers: dict[str, Answer] = {}
            prefill_ms = decode_ms = 0.0
            reasoning_ms = 0.0
            reasoning_tokens = reasoning_passes = reasoning_work = forced_closures = 0
            prompt_tokens = canvas_tokens = 0
            for group_index, read in enumerate(prepared):
                if read.thought_close_id is not None:
                    from diffusion_jev.reasoning import reasoning_prepass

                    reasoned = reasoning_prepass(
                        self.model, self.tokenizer, read.prompt,
                        thought_close_id=read.thought_close_id, max_tokens=self.reasoning_tokens,
                        seed=(request.options.seed + group_index) % (2**32),
                        max_prompt_tokens=self.max_prompt_tokens,
                    )
                    metrics = reasoned.metrics
                    reasoning_ms += metrics.wall_ms
                    reasoning_tokens += metrics.retained_tokens
                    reasoning_passes += metrics.denoising_steps
                    reasoning_work += metrics.work_tokens
                    forced_closures += int(not metrics.natural_close)
                    prompt_tokens += len(read.prompt) + metrics.continuation_prefill_tokens
                    read = replace(read, prompt=reasoned.prefix)
                    del reasoned
                prompt_tokens += len(read.prompt)
                canvas_tokens += (
                    len(read.canvas.tokens) * request.options.samples * self.trajectory_steps
                )
                before = time.perf_counter()
                cache = self._prefill(read.prompt)
                prefill_ms += (time.perf_counter() - before) * 1000
                before = time.perf_counter()
                distributions = self._read(read, cache, request)
                decode_ms += (time.perf_counter() - before) * 1000
                for slot, probabilities in zip(read.canvas.slots, distributions, strict=True):
                    answers[slot.question_id] = make_answer(
                        read.questions[slot.question_id], probabilities
                    )
                del cache
            return DecisionResponse(
                model=self.model_id,
                answers=answers,
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    questions=len(request.questions),
                    decoder_passes=(
                        len(prepared) * request.options.samples * self.trajectory_steps
                        + reasoning_passes
                    ),
                    canvas_tokens=canvas_tokens + reasoning_work,
                    prefill_ms=prefill_ms,
                    decode_ms=decode_ms,
                    total_ms=(time.perf_counter() - started) * 1000,
                    peak_memory_gb=mx.get_peak_memory() / 1e9,
                    reasoning_tokens=reasoning_tokens,
                    reasoning_ms=reasoning_ms,
                    reasoning_passes=reasoning_passes,
                    reasoning_forced_closures=forced_closures,
                    trajectory_accepted_slots=self.trajectory_accepted_slots,
                ),
            )
