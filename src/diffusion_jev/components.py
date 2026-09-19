"""Restore selected genuine BF16 checkpoint weights without loading whole shards."""

from importlib.metadata import version
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.nn import Linear

from diffusion_jev.component_source import (
    ROUTER_COUNT,
    ROUTER_SHAPE,
    read_router_source,
)
from diffusion_jev.component_source import RestorationInfo as RestorationInfo
from diffusion_jev.precision import PrecisionModel, Projection


def _array_field(projection: Projection, name: str) -> mx.array:
    # MLX stores arrays in a mapping; Python 3.13 runtime Protocol checks miss them.
    value: object = getattr(projection, name, None)
    if not isinstance(value, mx.array):
        raise ValueError("BF16 restoration requires previously unrestored quantized routers")
    return value


def restore_bf16_router_projections(model: PrecisionModel, source: Path) -> RestorationInfo:
    """Replace 30 quantized projections after validation and staging of every weight.

    The source must have passed full checkpoint verification during provisioning.
    The bounded reader rechecks metadata and fingerprints selected bytes only.
    Repeat application fails because the live projections are no longer quantized.
    """
    if version("mlx-optiq") != "0.5.12":
        raise RuntimeError("BF16 restoration requires mlx-optiq 0.5.12")
    layers = model.model.decoder.layers
    if len(layers) != ROUTER_COUNT:
        raise ValueError("BF16 restoration requires exactly 30 decoder layers")
    if len({id(layer.router) for layer in layers}) != ROUTER_COUNT:
        raise ValueError("BF16 restoration requires distinct decoder routers")
    training: list[bool] = []
    for layer in layers:
        projection = layer.router.proj
        weight = _array_field(projection, "weight")
        scales = _array_field(projection, "scales")
        biases = _array_field(projection, "biases")
        is_training: object = getattr(projection, "training", None)
        if not isinstance(is_training, bool):
            raise ValueError("BF16 restoration requires previously unrestored quantized routers")
        if (
            getattr(projection, "bits", None) != 8 or getattr(projection, "group_size", None) != 64
            or getattr(projection, "mode", None) != "affine"
            or getattr(projection, "bias", None) is not None
            or weight.shape != (128, 704) or weight.dtype != mx.uint32
            or scales.shape != (128, 44) or scales.dtype != mx.bfloat16
            or biases.shape != (128, 44) or biases.dtype != mx.bfloat16
            or layer.router.scale.shape != (2816,) or layer.router.scale.dtype != mx.bfloat16
            or layer.router.per_expert_scale.shape != (128,)
            or layer.router.per_expert_scale.dtype != mx.bfloat16
            or layer.router.config.top_k_experts != 8
        ):
            raise ValueError("BF16 restoration requires the pinned 8-bit OptiQ router layout")
        training.append(is_training)

    selected = read_router_source(source)
    replacements: list[Linear] = []
    for payload, is_training in zip(selected.payloads, training, strict=True):
        # Reinterpret the words; a numerical uint16-to-BF16 cast is incorrect.
        weight = mx.array(np.frombuffer(payload, dtype="<u2"), dtype=mx.uint16).view(mx.bfloat16)
        weight = weight.reshape(ROUTER_SHAPE)
        mx.eval(weight)
        projection = Linear(ROUTER_SHAPE[1], ROUTER_SHAPE[0], bias=False)
        projection.weight = weight
        projection.train(is_training)
        replacements.append(projection)
    for layer, projection in zip(layers, replacements, strict=True):
        layer.router.proj = projection
    return selected.info
