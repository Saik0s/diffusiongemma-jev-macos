"""Bounded, GPU-free inspection of selected pinned BF16 checkpoint weights."""

import hashlib
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, RootModel

from diffusion_jev.model_manifest import CHECKPOINTS, Artifact

ROUTER_COUNT = 30
ROUTER_SHAPE = (128, 2816)
ROUTER_BYTES = 720_896
_HEADER_LIMIT = 1_048_576

# Selected bytes were hashed from the snapshot after all 18 pinned artifacts passed
# full SHA256 verification. Keep these payload identities tied to that revision.
ROUTER_SOURCE_REVISION = "2cd36f950eb065c96c80810fb6b859b114cd052d"
ROUTER_SHA256: Mapping[str, str] = MappingProxyType({
    f"model.decoder.layers.{layer}.router.proj.weight": digest
    for layer, digest in enumerate((
        "b36a39f6f28fbf52d7372ce549f1e62e21cfad9b029e1677b7cb67c984cd2b05",
        "ec4fa50203b88a6f7f54e2abe6a446c7d6e674cd569e5297105c37e26d365b03",
        "165296ef4231e45187a82e5fe988d599e735f576c185fadd8dd0a24da40b9d74",
        "6ff9d79c0891becea07cc45177a00647a2f75ce683b6a47cc36d0c9c25da4318",
        "fab463cc8556487c0b78c6f602e5aef5d1c07a29f3cb61d19f65e03f0931c0e6",
        "3eca5fe414cb0978bc44b4e7d9f9b2a847f26b1586b4039cdc073f5ff3a6d650",
        "1efbd3709c77bee48793ebe1b608b48d6f5700da90bc52cd79e540b7533bc39a",
        "5ba80af021ea4d66886fe53e24909951c7b31acad2dd180a762c580c05702d18",
        "1d42162a3877933fe4e1ff68e9408bdb6ae4230064c7379a7cba80d3b0a62de4",
        "3e6cffb7798e61e09d03e656e8a4c19b67d07d16309df28a2f86f93574c6e1ae",
        "152f0d09fe2d23c45896aed7c873073df50d0eddd1c8ff0e39f384b1c40dc92d",
        "3b300ea8ab44994f78dea980114b949af96c5dd1f02660369f7ca85c94862ad8",
        "a6aeaefaf1cee132c3a25237a039b2e1211b5569f90604ba2d425800ab9c5152",
        "2b7c1a20020250d33a868489270756fdd5150d0a7d7225c877f1128b258e1c5f",
        "1d245f91ce1b910fe094ef6275dd469e9156020109eccaee536c486417bba410",
        "758f54a76ca3b724a96f17d90a6f33925838e1d33762334a32aaf473badf7208",
        "aa12b9a2292c3e62e421bca9b730db88e3ed9c3904c72e4333fe689b023338ab",
        "0321151400487e0664650a0a66754f69dfafc887c1c8110515930fc0ffda7c0c",
        "fb6e5f614fadf3d34fd41f41c437c1a2d1e53a8e8bcd62163c1ff536fde09b54",
        "d236f9d39036f763b71e851332a4a21b358d9a4be89b8d9651b1b165cb19308a",
        "386b65fe62414b68a181f6985ec6d31b57fba0a8d62758766063cb0cb74ef98a",
        "8bf4d8c6654049e634f3799255c7c07f76096d43ffd156910983a39ae77bf38d",
        "2a74dca095d64ddf8251ebb2dd1c7b457a161949dd871fb93d10f3803719f163",
        "2f53470c3699dd607315cd1cf72bb46ec824b5e8482027e264655cb6e2547291",
        "b4731f412932e4d68c582bffcaa246e9694d1b685b1ff63f578ba8365d148627",
        "e1bd27cee871f8c6f71e8dad0d8d15fb9ca58152fb8144fd4bcf79d4f8075580",
        "50e675e26054f43e83a6d59f5efdd093ebdd141b19f7a8d0ba5ab9526dd62540",
        "d3952ab9052289bb9c4c0dace3c25c3ba0a6d51174238b26f882e9c0b5e5ed76",
        "3340fef6f4c918c0bf6df7ce3e74d8b3049e09c7b54b68974842b546938f1fe1",
        "4be11e0180cae3ce2404d585fc33217b95819986f3834aa5f6a20dc2d4a76745",
    ))
})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class RestoredTensor(_Strict):
    name: str
    shard: str
    sha256: str
    size_bytes: int


class RestorationInfo(_Strict):
    source_repo: str
    source_revision: str
    config_sha256: str
    index_sha256: str
    projection_count: int
    payload_bytes: int
    tensors: tuple[RestoredTensor, ...]


@dataclass(frozen=True)
class RouterSource:
    info: RestorationInfo
    payloads: tuple[bytes, ...] = field(repr=False)


class _Index(BaseModel):
    weight_map: dict[str, str]


class _Tensor(_Strict):
    dtype: str
    shape: list[int]
    data_offsets: tuple[int, int]


class _Header(RootModel[dict[str, _Tensor | dict[str, str]]]):
    pass


@dataclass(frozen=True)
class _Selection:
    name: str
    shard: str
    path: Path
    offset: int


def _name(layer: int) -> str:
    return f"model.decoder.layers.{layer}.router.proj.weight"


def _small_artifact(source: Path, artifact: Artifact) -> bytes:
    path = source / artifact.name
    if path.stat().st_size != artifact.size:
        raise ValueError(f"BF16 source size mismatch: {artifact.name}")
    with path.open("rb") as handle:
        payload = handle.read(artifact.size + 1)
    if hashlib.sha256(payload).hexdigest() != artifact.sha256:
        raise ValueError(f"BF16 source identity mismatch: {artifact.name}")
    return payload


def _selections(source: Path) -> tuple[list[_Selection], str, str]:
    profile = CHECKPOINTS["bf16"]
    if profile.revision != ROUTER_SOURCE_REVISION:
        raise ValueError("BF16 router payload pins do not match the checkpoint revision")
    artifacts = {item.name: item for item in profile.artifacts}
    config = _small_artifact(source, artifacts["config.json"])
    index_payload = _small_artifact(source, artifacts["model.safetensors.index.json"])
    index = _Index.model_validate_json(index_payload)
    expected = {_name(layer) for layer in range(ROUTER_COUNT)}
    actual = {key for key in index.weight_map if ".router.proj." in key}
    if actual != expected:
        raise ValueError("BF16 source must contain exactly 30 router projection weights")

    headers: dict[str, tuple[_Header, int]] = {}
    selections: list[_Selection] = []
    for name in sorted(expected, key=lambda key: int(key.split(".")[3])):
        shard = index.weight_map[name]
        if shard not in artifacts or not shard.endswith(".safetensors"):
            raise ValueError("BF16 index references an unpinned weight shard")
        path = source / shard
        if path.stat().st_size != artifacts[shard].size:
            raise ValueError(f"BF16 shard size mismatch: {shard}")
        if shard not in headers:
            with path.open("rb") as handle:
                prefix = handle.read(8)
                if len(prefix) != 8:
                    raise ValueError("Truncated BF16 shard header")
                length = struct.unpack("<Q", prefix)[0]
                if length <= 0 or length > _HEADER_LIMIT or 8 + length > artifacts[shard].size:
                    raise ValueError("Invalid BF16 shard header length")
                payload = handle.read(length)
                if len(payload) != length:
                    raise ValueError("Truncated BF16 shard header")
                headers[shard] = (_Header.model_validate_json(payload), 8 + length)
        header, start = headers[shard]
        tensor = header.root.get(name)
        if not isinstance(tensor, _Tensor) or tensor.dtype != "BF16":
            raise ValueError(f"Missing or non-BF16 router tensor: {name}")
        if tuple(tensor.shape) != ROUTER_SHAPE:
            raise ValueError(f"Incorrect BF16 router shape: {name}")
        first, last = tensor.data_offsets
        if first < 0 or last - first != ROUTER_BYTES or start + last > artifacts[shard].size:
            raise ValueError(f"Invalid BF16 router byte range: {name}")
        if any(
            entry.shard == shard
            and max(entry.offset, start + first) < min(entry.offset + ROUTER_BYTES, start + last)
            for entry in selections
        ):
            raise ValueError("Overlapping BF16 router byte ranges")
        selections.append(_Selection(name, shard, path, start + first))
    return selections, hashlib.sha256(config).hexdigest(), hashlib.sha256(index_payload).hexdigest()


def _read_payload(selection: _Selection) -> tuple[bytes, RestoredTensor]:
    with selection.path.open("rb") as handle:
        handle.seek(selection.offset)
        payload = handle.read(ROUTER_BYTES)
    if len(payload) != ROUTER_BYTES:
        raise ValueError(f"Truncated BF16 router payload: {selection.name}")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != ROUTER_SHA256.get(selection.name):
        raise ValueError(f"BF16 router payload checksum mismatch: {selection.name}")
    # BF16 exponent bits all set represent either infinity or NaN; no MLX is needed.
    if any((word & 0x7F80) == 0x7F80 for word in memoryview(payload).cast("H")):
        raise ValueError(f"Nonfinite BF16 router payload: {selection.name}")
    return payload, RestoredTensor(
        name=selection.name, shard=selection.shard,
        sha256=digest, size_bytes=len(payload),
    )


def read_router_source(source: Path) -> RouterSource:
    """Authenticate and read only selected pinned BF16 projection bytes.

    Checks config/index hashes, shard sizes, and each selected payload's pinned
    SHA256. Other shard contents are not rehashed. Payloads remain in memory
    only and are omitted from the returned object's representation.
    """
    if sys.byteorder != "little":
        raise RuntimeError("BF16 source inspection requires little-endian hardware")
    selections, config_hash, index_hash = _selections(source)
    payloads: list[bytes] = []
    fingerprints: list[RestoredTensor] = []
    for selection in selections:
        payload, fingerprint = _read_payload(selection)
        payloads.append(payload)
        fingerprints.append(fingerprint)
    profile = CHECKPOINTS["bf16"]
    return RouterSource(
        info=RestorationInfo(
            source_repo=profile.repo, source_revision=profile.revision,
            config_sha256=config_hash, index_sha256=index_hash,
            projection_count=len(payloads), payload_bytes=sum(t.size_bytes for t in fingerprints),
            tensors=tuple(fingerprints),
        ),
        payloads=tuple(payloads),
    )


def inspect_router_source(source: Path) -> RestorationInfo:
    """Fingerprint selected source bytes before model loading or benchmark resume."""
    return read_router_source(source).info
