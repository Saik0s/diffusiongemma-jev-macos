"""The small, version-pinned OptiQ surface used by the local adapter."""

from typing import Literal, Protocol

import mlx.core as mx


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
    def state(self) -> tuple[mx.array, mx.array]: ...


class Embedding(Protocol):
    def __call__(self, tokens: mx.array) -> mx.array: ...

    def as_linear(self, hidden: mx.array) -> mx.array: ...


class Decoder(Protocol):
    @property
    def embed_tokens(self) -> Embedding: ...

    def __call__(self, canvas_ids: mx.array, *, cache: list[Cache]) -> mx.array: ...


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
    def text_config(self) -> TextConfig: ...


class DiffusionModel(Protocol):
    @property
    def config(self) -> ModelConfig: ...

    @property
    def model(self) -> Backbone: ...

    def make_cache(self) -> list[Cache]: ...

    def _softcap(self, logits: mx.array) -> mx.array: ...
