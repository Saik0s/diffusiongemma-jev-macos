"""The small, version-pinned OptiQ surface used by the local adapter."""

from collections.abc import Sequence
from typing import Literal, Protocol

import mlx.core as mx

from diffusion_jev.precision import PrecisionLayer


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: Literal[False],
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str: ...


class Cache(Protocol):
    @property
    def offset(self) -> int: ...

    @property
    def state(self) -> tuple[mx.array, mx.array]: ...


class Embedding(Protocol):
    @property
    def weight(self) -> mx.array: ...

    def __call__(self, tokens: mx.array) -> mx.array: ...

    def as_linear(self, hidden: mx.array) -> mx.array: ...


class Decoder(Protocol):
    @property
    def embed_scale(self) -> float: ...

    @property
    def layers(self) -> Sequence[PrecisionLayer]: ...

    @property
    def embed_tokens(self) -> Embedding: ...

    def __call__(
        self, canvas_ids: mx.array, *, cache: list[Cache],
        self_conditioning_embeddings: mx.array | None = None,
    ) -> mx.array: ...


class Encoder(Protocol):
    def __call__(
        self, input_ids: mx.array, *, cache: list[Cache]
    ) -> tuple[mx.array, list[Cache]]: ...


class Backbone(Protocol):
    @property
    def encoder(self) -> Encoder: ...

    @property
    def decoder(self) -> Decoder: ...


class TextConfig(Protocol):
    @property
    def vocab_size(self) -> int: ...

    @property
    def pad_token_id(self) -> int: ...

    @property
    def final_logit_softcapping(self) -> float: ...


class ModelConfig(Protocol):
    @property
    def canvas_length(self) -> int: ...

    @property
    def text_config(self) -> TextConfig: ...


class DiffusionModel(Protocol):
    @property
    def config(self) -> ModelConfig: ...

    @property
    def model(self) -> Backbone: ...

    def make_cache(self) -> list[Cache]: ...

    def _softcap(self, logits: mx.array) -> mx.array: ...
