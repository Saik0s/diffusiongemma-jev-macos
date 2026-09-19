"""CPU evidence for selective restoration; no model or real checkpoint loads."""

import hashlib
import json
import struct
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.nn import Linear, Module
from mlx.nn.layers.quantized import QuantizedLinear

from diffusion_jev import component_source
from diffusion_jev.component_source import inspect_router_source, read_router_source
from diffusion_jev.components import restore_bf16_router_projections
from diffusion_jev.model_manifest import Artifact, CheckpointProfile
from diffusion_jev.precision import ExpertsSurface, PrecisionLayer, Projection, RouterSurface


@pytest.fixture(autouse=True)
def cpu_stream() -> Iterator[None]:
    with mx.stream(mx.cpu):
        yield


@dataclass(frozen=True)
class Config:
    top_k_experts: int = 8


class PackedProjection(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = mx.zeros((128, 704), dtype=mx.uint32)
        self.scales = mx.ones((128, 44), dtype=mx.bfloat16)
        self.biases = mx.zeros((128, 44), dtype=mx.bfloat16)
        self.bits, self.group_size, self.mode = 8, 64, "affine"

    def __call__(self, x: mx.array) -> mx.array:
        raise AssertionError("Restoration must not execute or dequantize the old projection")


class Router(Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = Config()
        self.eps, self._root_size = 1e-6, 2816**-0.5
        self.proj: Projection = PackedProjection()
        self.scale = mx.ones((2816,), dtype=mx.bfloat16)
        self.per_expert_scale = mx.ones((128,), dtype=mx.bfloat16)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        raise AssertionError("Restoration must not run router inference")


@dataclass
class Layer:
    router: RouterSurface

    @property
    def experts(self) -> ExpertsSurface:
        raise AssertionError("Restoration must not inspect or replace experts")

    @experts.setter
    def experts(self, value: ExpertsSurface) -> None:
        raise AssertionError("Restoration must not inspect or replace experts")


@dataclass
class Decoder:
    layers: list[PrecisionLayer]


@dataclass
class Encoder:
    decoder: Decoder


@dataclass
class Backbone:
    decoder: Decoder
    encoder: Encoder


class Model:
    def __init__(self, training: bool = False) -> None:
        layers: list[PrecisionLayer] = []
        for _ in range(30):
            router = Router()
            router.train(training)
            layers.append(Layer(router))
        decoder = Decoder(layers)
        self.model = Backbone(decoder, Encoder(decoder))


@dataclass(frozen=True)
class Fixture:
    root: Path
    shard: Path
    payload_offset: int
    tensor_hash: str


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Fixture:
    header: dict[str, dict[str, str | list[int]]] = {}
    weight_map: dict[str, str] = {}
    shard = tmp_path / "model-00001-of-00001.safetensors"
    size = 720_896
    for layer in range(30):
        name = f"model.decoder.layers.{layer}.router.proj.weight"
        header[name] = {"dtype": "BF16", "shape": [128, 2816],
                        "data_offsets": [layer * size, (layer + 1) * size]}
        weight_map[name] = shard.name
    encoded = json.dumps(header).encode()
    payload = struct.pack("<H", 0x3F80) * (size // 2)
    with shard.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for _ in range(30):
            handle.write(payload)
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    artifacts: list[Artifact] = []
    for path in (tmp_path / "config.json", tmp_path / "model.safetensors.index.json", shard):
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path != shard else "0" * 64
        artifacts.append(Artifact(path.name, path.stat().st_size, digest))
    profile = CheckpointProfile("bf16", "synthetic-test-source", "test-revision",
                                tuple(artifacts), "mlx", None)
    monkeypatch.setattr(component_source, "CHECKPOINTS", {"bf16": profile})
    monkeypatch.setattr(component_source, "ROUTER_SOURCE_REVISION", profile.revision)
    monkeypatch.setattr(component_source, "ROUTER_SHA256", {
        name: hashlib.sha256(payload).hexdigest() for name in weight_map
    })
    return Fixture(tmp_path, shard, len(encoded) + 8, hashlib.sha256(payload).hexdigest())


def test_all_bf16_bit_patterns_reinterpret_exactly() -> None:
    raw = struct.pack("<65536H", *range(65536))
    restored = mx.array(np.frombuffer(raw, dtype="<u2"), dtype=mx.uint16).view(mx.bfloat16)
    mx.eval(restored)
    del raw
    assert restored.dtype == mx.bfloat16
    assert restored.view(mx.uint16).tolist() == list(range(65536))


@pytest.mark.parametrize("training", [True, False])
def test_restore_preserves_sharing_state_and_scalars(source: Fixture, training: bool) -> None:
    model, untouched = Model(training), Model()
    layers = model.model.decoder.layers
    routers = [layer.router for layer in layers]
    scalars = [(router.scale, router.per_expert_scale) for router in routers]
    info = restore_bf16_router_projections(model, source.root)
    assert info.projection_count == 30
    assert info.payload_bytes == 21_626_880
    assert info.source_revision == "test-revision"
    assert all(t.sha256 == source.tensor_hash and t.size_bytes == 720_896 for t in info.tensors)
    for index, layer in enumerate(layers):
        projection = layer.router.proj
        assert isinstance(projection, Linear)
        assert projection.training is training
        assert projection.weight.dtype == mx.bfloat16
        assert projection.weight.shape == (128, 2816)
        assert not hasattr(projection, "scales") and not hasattr(projection, "biases")
        assert mx.all(projection.weight == 1).item()
        assert layer.router is routers[index]
        assert layer.router.scale is scalars[index][0]
        assert layer.router.per_expert_scale is scalars[index][1]
        assert model.model.encoder.decoder.layers[index].router.proj is projection
        assert isinstance(untouched.model.decoder.layers[index].router.proj, PackedProjection)
    with pytest.raises(ValueError, match="previously unrestored"):
        restore_bf16_router_projections(model, source.root)


@pytest.mark.parametrize("failure", ["dtype", "shape", "offset", "overlap", "nonfinite"])
def test_bad_final_tensor_leaves_every_projection_unchanged(source: Fixture, failure: str) -> None:
    model = Model()
    originals = [layer.router.proj for layer in model.model.decoder.layers]
    with source.shard.open("r+b") as handle:
        if failure == "nonfinite":
            handle.seek(source.payload_offset + 29 * 720_896)
            handle.write(struct.pack("<H", 0x7F80))
        else:
            header = handle.read(source.payload_offset)
            if failure == "dtype":
                position = header.rfind(b'"BF16"')
                header = header[:position] + header[position:].replace(b'"BF16"', b'"FP16"', 1)
            elif failure == "shape":
                position = header.rfind(b"2816")
                header = header[:position] + header[position:].replace(b"2816", b"2815", 1)
            elif failure == "offset":
                header = header.replace(b"21626880", b"21626881")
            else:
                header = header.replace(b"20905984, 21626880", b"20185088, 20905984")
            handle.seek(0)
            handle.write(header)
    with pytest.raises(ValueError):
        restore_bf16_router_projections(model, source.root)
    assert all(layer.router.proj is original for layer, original in
               zip(model.model.decoder.layers, originals, strict=True))


@pytest.mark.parametrize("filename", ["config.json", "model.safetensors.index.json"])
def test_identity_mismatch_rejects_before_mutation(source: Fixture, filename: str) -> None:
    model = Model()
    target = source.root / filename
    payload = target.read_bytes()
    target.write_bytes(b"x" + payload[1:])
    with pytest.raises(ValueError, match="identity mismatch"):
        restore_bf16_router_projections(model, source.root)
    assert all(isinstance(layer.router.proj, PackedProjection)
               for layer in model.model.decoder.layers)


def test_incompatible_final_live_projection_rejects_before_mutation(source: Fixture) -> None:
    model = Model()
    final = model.model.decoder.layers[-1].router.proj
    assert isinstance(final, PackedProjection)
    final.bits = 4
    with pytest.raises(ValueError, match="pinned 8-bit"):
        restore_bf16_router_projections(model, source.root)
    assert all(isinstance(layer.router.proj, PackedProjection)
               for layer in model.model.decoder.layers)


def test_truncated_payload_is_rejected(source: Fixture) -> None:
    selection = component_source._Selection("synthetic", source.shard.name, source.shard,
                                            source.shard.stat().st_size - 2)
    with pytest.raises(ValueError, match="Truncated"):
        component_source._read_payload(selection)


def test_native_quantized_module_is_replaced_with_dense_execution(source: Fixture) -> None:
    model = Model()
    old = QuantizedLinear(2816, 128, bias=False, group_size=64, bits=8)
    old.scales = old.scales.astype(mx.bfloat16)
    old.biases = old.biases.astype(mx.bfloat16)
    old.eval()
    model.model.decoder.layers[-1].router.proj = old
    restore_bf16_router_projections(model, source.root)
    dense = model.model.decoder.layers[-1].router.proj
    assert isinstance(dense, Linear)
    assert not dense.training
    result = dense(mx.ones((1, 2816), dtype=mx.bfloat16))
    assert result.dtype == mx.bfloat16
    assert result.tolist() == [[2816.0] * 128]


def test_source_inspection_import_does_not_import_mlx() -> None:
    subprocess.run(
        [sys.executable, "-c", "import sys; import diffusion_jev.component_source; "
         "assert not any(key == 'mlx' or key.startswith('mlx.') for key in sys.modules)"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
    )


def test_inspection_and_restoration_identity_match_without_exposing_payloads(
    source: Fixture,
) -> None:
    loaded = read_router_source(source.root)
    assert "payloads=" not in repr(loaded)
    inspected = inspect_router_source(source.root)
    restored = restore_bf16_router_projections(Model(), source.root)
    assert inspected == loaded.info == restored
    assert inspected.model_validate_json(inspected.model_dump_json()) == inspected


def test_finite_single_byte_corruption_is_rejected_without_model_mutation(source: Fixture) -> None:
    model = Model()
    originals = [layer.router.proj for layer in model.model.decoder.layers]
    with source.shard.open("r+b") as handle:
        handle.seek(source.payload_offset + 29 * 720_896)
        handle.write(b"\x01")  # 0x3f80 (1.0) becomes finite BF16 0x3f81.
    with pytest.raises(ValueError, match="payload checksum mismatch"):
        inspect_router_source(source.root)
    with pytest.raises(ValueError, match="payload checksum mismatch"):
        restore_bf16_router_projections(model, source.root)
    assert all(layer.router.proj is original for layer, original in
               zip(model.model.decoder.layers, originals, strict=True))


def test_payload_pins_are_bound_to_the_profile_revision(
    source: Fixture, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(component_source, "ROUTER_SOURCE_REVISION", "different-revision")
    with pytest.raises(ValueError, match="pins do not match the checkpoint revision"):
        inspect_router_source(source.root)
