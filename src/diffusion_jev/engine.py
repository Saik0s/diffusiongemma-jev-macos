"""One-step structured reads on the existing OptiQ model, without text decoding."""

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from optiq.vlm.diffusion_gemma import load

from diffusion_jev.canvas import Canvas, compile_canvas, make_answer, system_prompt
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


class LocalEngine:
    """Construct and call on one thread; MLX lazy buffers keep thread-local streams."""

    model_id = "diffusiongemma-local"

    def __init__(self, model_path: Path, max_prompt_tokens: int = 8192) -> None:
        if not model_path.is_dir() or not (model_path / "config.json").is_file():
            raise FileNotFoundError("A local model directory containing config.json is required")
        if max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")
        started = time.perf_counter()
        self.model: DiffusionModel
        self.tokenizer: Tokenizer
        self.model, self.tokenizer = load(str(model_path.resolve()))
        self.load_seconds = time.perf_counter() - started
        self.max_prompt_tokens = max_prompt_tokens
        self._lock = threading.Lock()

    def _prepare(self, request: DecisionRequest) -> list[PreparedRead]:
        state = json.dumps(request.state, ensure_ascii=False, sort_keys=True, allow_nan=False)
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
                enable_thinking=False,
            )
            tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) > self.max_prompt_tokens:
                raise ValueError("Prompt exceeds the configured token limit")
            canvas = compile_canvas(
                self.tokenizer,
                questions,
                width=request.options.canvas_length,
                pad_token_id=self.model.config.text_config.pad_token_id,
            )
            prepared.append(PreparedRead(tuple(tokens), canvas, questions))
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
        alias_embeddings = embedding(mx.array(alias_ids, dtype=mx.int32))
        positions = mx.array([slot.position for slot in canvas.slots], dtype=mx.int32)
        accumulated = [[0.0] * len(slot.token_ids) for slot in canvas.slots]
        for sample in range(request.options.samples):
            tokens = canvas.seeded(
                request.options.seed + sample, self.model.config.text_config.vocab_size
            )
            hidden = self.model.model.decoder(mx.array([tokens], dtype=mx.int32), cache=cache)
            if request.options.projection == "full":
                logits = embedding.as_linear(hidden)
            else:
                # The tied embedding table is the output head. Gather before projecting.
                logits = hidden[:, positions, :] @ alias_embeddings.T
            logits = self.model._softcap(logits)
            distributions: list[mx.array] = []
            for index, slot in enumerate(canvas.slots):
                if request.options.projection == "full":
                    selected = logits[0, slot.position, mx.array(slot.token_ids, dtype=mx.int32)]
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
        return accumulated

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        if request.model != self.model_id:
            raise ValueError("Unknown model")
        with self._lock:
            started = time.perf_counter()
            prepared = self._prepare(request)
            mx.reset_peak_memory()
            answers: dict[str, Answer] = {}
            prefill_ms = decode_ms = 0.0
            for read in prepared:
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
                    prompt_tokens=sum(len(read.prompt) for read in prepared),
                    questions=len(request.questions),
                    decoder_passes=len(prepared) * request.options.samples,
                    canvas_tokens=sum(len(read.canvas.tokens) for read in prepared)
                    * request.options.samples,
                    prefill_ms=prefill_ms,
                    decode_ms=decode_ms,
                    total_ms=(time.perf_counter() - started) * 1000,
                    peak_memory_gb=mx.get_peak_memory() / 1e9,
                ),
            )
