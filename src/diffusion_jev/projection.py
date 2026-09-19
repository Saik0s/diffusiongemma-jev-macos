"""Selected output rows with explicit arithmetic precision."""

import mlx.core as mx
from mlx.nn import QuantizedEmbedding

from diffusion_jev.runtime_types import Embedding


def float32_label_embeddings(embedding: Embedding, labels: mx.array) -> mx.array:
    """Avoid rounding affine dequantized rows to the embedding's storage dtype."""
    # MLX stores parameter arrays dynamically, outside runtime Protocol inspection.
    if isinstance(embedding, QuantizedEmbedding):
        if embedding.mode != "affine":
            raise ValueError("Float32 label projection currently requires affine quantization")
        biases = embedding.biases
        return mx.dequantize(
            embedding.weight[labels],
            embedding.scales[labels].astype(mx.float32),
            biases[labels].astype(mx.float32) if biases is not None else None,
            group_size=embedding.group_size,
            bits=embedding.bits,
            mode=embedding.mode,
            dtype=mx.float32,
        )
    return embedding(labels).astype(mx.float32)
