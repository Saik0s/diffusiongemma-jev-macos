"""CPU checks for selected-row projection without BF16 dequantization rounding."""

from collections.abc import Iterator

import mlx.core as mx
import pytest
from mlx.nn import QuantizedEmbedding

from diffusion_jev.projection import float32_label_embeddings


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


class AffineEmbedding(QuantizedEmbedding):
    def __init__(self, bits: int = 4) -> None:
        super().__init__(4, 32, group_size=32, bits=bits)
        source = mx.sin(mx.arange(128, dtype=mx.float32)).reshape(4, 32)
        self.weight, self.scales, self.biases = mx.quantize(
            source.astype(mx.bfloat16), group_size=self.group_size, bits=self.bits
        )

    def __call__(self, tokens: mx.array) -> mx.array:
        assert self.biases is not None
        return mx.dequantize(
            self.weight[tokens], self.scales[tokens], self.biases[tokens],
            group_size=self.group_size, bits=self.bits,
        )

    def as_linear(self, hidden: mx.array) -> mx.array:
        raise AssertionError("Selected rows must not project the full vocabulary")


@pytest.mark.parametrize("bits", [4, 8])
def test_affine_rows_dequantize_before_rounding_and_keep_requested_order(bits: int) -> None:
    embedding = AffineEmbedding(bits)
    labels = mx.array([3, 1, 3])
    actual = float32_label_embeddings(embedding, labels)
    packed = embedding.weight[labels]
    shifts = mx.arange(32 // bits, dtype=mx.uint32) * bits
    codes = ((packed[..., None] >> shifts) & ((1 << bits) - 1)).reshape(3, 32)
    assert embedding.biases is not None
    expected = (
        codes.astype(mx.float32) * embedding.scales[labels].astype(mx.float32)
        + embedding.biases[labels].astype(mx.float32)
    )
    assert actual.dtype == mx.float32
    assert mx.array_equal(actual, expected).item()
    assert not mx.array_equal(actual, embedding(labels).astype(mx.float32)).item()


def test_non_affine_quantization_is_rejected() -> None:
    embedding = AffineEmbedding()
    embedding.mode = "mxfp4"
    with pytest.raises(ValueError, match="requires affine"):
        float32_label_embeddings(embedding, mx.array([0]))


class DenseEmbedding:
    def __init__(self) -> None:
        self.weight = mx.array([[1.0, 2.0], [3.0, 4.0]], dtype=mx.bfloat16)

    def __call__(self, tokens: mx.array) -> mx.array:
        return self.weight[tokens]

    def as_linear(self, hidden: mx.array) -> mx.array:
        raise AssertionError("Selected rows must not project the full vocabulary")


def test_dense_embedding_is_gathered_and_promoted() -> None:
    embedding = DenseEmbedding()
    actual = float32_label_embeddings(embedding, mx.array([1, 0]))
    assert actual.dtype == mx.float32
    assert actual.tolist() == [[3.0, 4.0], [1.0, 2.0]]
