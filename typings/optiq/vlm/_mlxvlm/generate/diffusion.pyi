from collections.abc import Generator
from typing import Literal

import mlx.core as mx

from diffusion_jev.reasoning import ReasoningEvent, ReasoningModel, ReasoningTokenizer
from diffusion_jev.trajectory import TrajectoryEmbedding

def stream_diffusion_generate(
    model: ReasoningModel,
    processor: ReasoningTokenizer,
    tokenizer: ReasoningTokenizer,
    input_ids: mx.array,
    pixel_values: None,
    attention_mask: None,
    *,
    max_tokens: int,
    skip_special_token_ids: set[int],
    temperature: float,
    max_denoising_steps: int,
    diffusion_full_canvas: bool,
    diffusion_min_canvas_length: int,
    diffusion_max_canvas_length: int,
    diffusion_static_cache: bool,
    diffusion_sampler: Literal["entropy-bound"],
    diffusion_compile: bool,
    diffusion_show_unmasking: bool,
    diffusion_unmasking_interval: int,
    diffusion_unmasking_width: int,
    mm_token_type_ids: None,
    prefill_step_size: int,
) -> Generator[ReasoningEvent]: ...

def _diffusion_linear_temperature(
    cur_step: int, max_denoising_steps: int, schedule_config: dict[str, float] | None,
) -> float | None: ...

def _diffusion_soft_embedding_weight(embed_tokens: TrajectoryEmbedding) -> mx.array: ...

def _diffusion_entropy_and_soft_embeddings(
    processed_logits: mx.array, embedding_weight: mx.array, embed_scale: float,
) -> tuple[mx.array, mx.array]: ...

def _diffusion_entropy_transfer_mask(entropy: mx.array, entropy_bound: float) -> mx.array: ...

def _diffusion_sample_canvas(
    processed_logits: mx.array, dtype: mx.Dtype, temperature: float,
) -> mx.array: ...

def _diffusion_initialize_canvas(
    batch_size: int, canvas_length: int, vocab_size: int, dtype: mx.Dtype,
) -> mx.array: ...
