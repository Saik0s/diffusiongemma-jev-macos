"""Resolve local checkpoints or provision the pinned model without importing MLX."""

import hashlib
import logging
import os
import shutil
import socket
import tempfile
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from huggingface_hub import constants, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import BaseModel, ConfigDict, ValidationError

from diffusion_jev import model_manifest
from diffusion_jev.model_manifest import MODEL_REPO, CheckpointProfile, Precision

LM_STUDIO_MODEL = Path("~/.cache/lm-studio/models") / MODEL_REPO
DISK_RESERVE = 1024**3
_DOWNLOAD_LOG_LOCK = threading.RLock()
_TRANSPORT_LOGGERS = ("huggingface_hub", "httpx", "httpcore")


class ProvisioningError(Exception):
    """An actionable error safe to show without raw transport exception details."""


class QuantizationConfig(BaseModel):
    model_config = ConfigDict(strict=True)
    bits: int


class ModelConfig(BaseModel):
    model_config = ConfigDict(strict=True)
    model_type: str
    quantization: QuantizationConfig | None = None


class WeightIndex(BaseModel):
    model_config = ConfigDict(strict=True)
    weight_map: dict[str, str]


def validate_local_model(path: Path, *, profile: CheckpointProfile | None = None) -> Path:
    """Check local structure; user-managed files are not checksum-certified."""
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ProvisioningError("Model directory does not exist. Check --model or JEV_MODEL_PATH.")
    required = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
    )
    for name in required:
        _require_file(path / name, name)
    try:
        config = ModelConfig.model_validate_json((path / "config.json").read_bytes())
        index = WeightIndex.model_validate_json(
            (path / "model.safetensors.index.json").read_bytes()
        )
    except (ValidationError, OSError) as exc:
        raise ProvisioningError("Model config or weight index is unreadable or invalid.") from exc
    if config.model_type != "diffusion_gemma" or not index.weight_map:
        raise ProvisioningError(
            "Expected a DiffusionGemma checkpoint with a nonempty weight index."
        )
    if profile is not None:
        bits = config.quantization.bits if config.quantization is not None else None
        if config.model_type != profile.model_type or bits != profile.quantization_bits:
            raise ProvisioningError("Model configuration does not match the selected precision.")
    for shard in set(index.weight_map.values()):
        if Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise ProvisioningError("The model weight index contains an unsupported shard path.")
        _require_file(path / shard, "a weight shard referenced by the index")
    indexed_vision = any(
        key.startswith("model.encoder.vision_tower.") for key in index.weight_map
    ) and any(key.startswith("model.encoder.embed_vision.") for key in index.weight_map)
    optiq_layout = (
        profile.layout == "optiq" if profile is not None
        else (path / "optiq_metadata.json").exists() or not indexed_vision
    )
    if optiq_layout:
        _require_file(path / "optiq_metadata.json", "optiq_metadata.json")
        sidecar = path / "optiq/optiq_vision.safetensors"
        if not sidecar.is_file():
            sidecar = path / "optiq_vision.safetensors"
        _require_file(sidecar, "optiq_vision.safetensors (in optiq/ or the model directory)")
    elif not indexed_vision:
        raise ProvisioningError("Standard MLX checkpoint is missing indexed vision weights.")
    return path


def _require_file(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise ProvisioningError(f"Model is incomplete: missing or empty {label}.")


def preflight_port(host: str, port: int) -> None:
    """Fail before provisioning if the requested listener cannot bind."""
    try:
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        for family, socktype, protocol, _, address in addresses:
            with socket.socket(family, socktype, protocol) as listener:
                listener.bind(address)
    except OSError as exc:
        raise ProvisioningError(
            "Cannot bind the requested host/port. Check --host or choose another --port."
        ) from exc


def verify_snapshot(
    path: Path, report: Callable[[str], None], *, profile: CheckpointProfile
) -> None:
    """Hash every pinned artifact, including warm caches and completed downloads."""
    for artifact in profile.artifacts:
        file = path / artifact.name
        _require_file(file, artifact.name)
        if file.stat().st_size != artifact.size:
            raise ProvisioningError(
                f"Checkpoint size mismatch for {artifact.name}. "
                "Retry with a fresh --cache-dir or repair this Hugging Face cache."
            )
        report(f"Verifying {artifact.name}...")
        with file.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != artifact.sha256:
            raise ProvisioningError(
                f"Checkpoint checksum mismatch for {artifact.name}. "
                "Retry with a fresh --cache-dir or repair this Hugging Face cache."
            )


@contextmanager
def _private_download_transport() -> Iterator[None]:
    # Retry warnings can contain signed URLs or credentials from raw exceptions.
    # Include child overrides; setting only a parent level does not silence them.
    with _DOWNLOAD_LOG_LOCK:
        names = set(_TRANSPORT_LOGGERS)
        names.update(
            name
            for name in tuple(logging.Logger.manager.loggerDict)
            if name.startswith(tuple(f"{root}." for root in _TRANSPORT_LOGGERS))
        )
        previous = [(logging.getLogger(name), logging.getLogger(name).level) for name in names]
        previous_xet_disabled = constants.HF_HUB_DISABLE_XET
        try:
            for logger, _ in previous:
                logger.setLevel(logging.CRITICAL + 1)
            # Xet's Rust tracing also logs raw request errors, outside Python logging.
            constants.HF_HUB_DISABLE_XET = True
            yield
        finally:
            constants.HF_HUB_DISABLE_XET = previous_xet_disabled
            for logger, level in previous:
                logger.setLevel(level)


def _snapshot(cache_dir: Path, *, offline: bool, profile: CheckpointProfile) -> Path:
    with _private_download_transport():
        result = snapshot_download(
            profile.repo,
            revision=profile.revision,
            cache_dir=cache_dir,
            allow_patterns=[artifact.name for artifact in profile.artifacts],
            local_files_only=offline,
        )
    if not isinstance(result, str):
        raise ProvisioningError("Hugging Face returned an unexpected snapshot result.")
    return Path(result)


def _complete(path: Path, profile: CheckpointProfile) -> bool:
    # HF local_files_only may return a snapshot containing just one downloaded file.
    return all((path / item.name).is_file() for item in profile.artifacts)


def _preflight_cache(cache_dir: Path, cached: Path | None, profile: CheckpointProfile) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    # A real write probe also catches ACLs and read-only volumes.
    with tempfile.TemporaryFile(dir=cache_dir):
        pass
    missing = sum(
        item.size
        for item in profile.artifacts
        if cached is None or not (cached / item.name).is_file()
    )
    required = missing + DISK_RESERVE
    if shutil.disk_usage(cache_dir).free < required:
        raise ProvisioningError(
            f"Insufficient disk space: need at least {required / 1024**3:.1f} GiB free "
            "for missing model files plus 1 GiB reserve. Free space or use --cache-dir."
        )


def resolve_model(
    model: Path | None = None,
    *,
    cache_dir: Path | None = None,
    offline: bool = False,
    precision: Precision = "optiq4",
    report: Callable[[str], None] = print,
) -> Path:
    """Resolve an explicit directory or the selected immutable checkpoint profile."""
    try:
        profile = model_manifest.CHECKPOINTS[precision]
        explicit = model or (
            Path(os.environ["JEV_MODEL_PATH"]) if os.environ.get("JEV_MODEL_PATH") else None
        )
        if explicit is not None:
            result = validate_local_model(explicit)
            report(
                "Using the selected local model (structure checked; checksums not verified). "
                "--precision applies only to managed checkpoints; the explicit path takes priority."
            )
            return result
        local = LM_STUDIO_MODEL.expanduser()
        if precision == "optiq4" and local.exists():
            try:
                result = validate_local_model(local)
            except ProvisioningError:
                report("LM Studio checkpoint is incomplete; checking the Hugging Face cache.")
            else:
                report("Using the LM Studio model (structure checked; checksums not verified).")
                return result
        cache_dir = (cache_dir or Path(constants.HF_HUB_CACHE)).expanduser().resolve()
        cached: Path | None = None
        try:
            cached = _snapshot(cache_dir, offline=True, profile=profile)
        except LocalEntryNotFoundError:
            pass
        if cached is not None and _complete(cached, profile):
            report("Found the pinned checkpoint in the Hugging Face cache; verifying integrity.")
            verify_snapshot(cached, report, profile=profile)
            return validate_local_model(cached, profile=profile)
        if offline or constants.HF_HUB_OFFLINE:
            raise ProvisioningError(
                "No complete cached checkpoint is available offline. "
                "Supply --model or rerun start online to download it."
            )
        _preflight_cache(cache_dir, cached, profile)
        total = sum(item.size for item in profile.artifacts) / 1024**3
        report(f"Downloading pinned {profile.repo} ({total:.1f} GiB total).")
        report("Completed files are reused; an interrupted file may restart.")
        downloaded = _snapshot(cache_dir, offline=False, profile=profile)
        verify_snapshot(downloaded, report, profile=profile)
        result = validate_local_model(downloaded, profile=profile)
        report("Checkpoint verified. Loading the model...")
        return result
    except ProvisioningError:
        raise
    except OSError as exc:
        raise ProvisioningError(
            "Cannot read or write the model cache. Check space and permissions, "
            "or choose another --cache-dir."
        ) from exc
    except Exception as exc:
        # Transport exceptions may include signed URLs, proxy credentials, or tokens.
        raise ProvisioningError(
            "Checkpoint provisioning failed. Check the connection, Hugging Face access, "
            "and HTTPS_PROXY settings, then rerun start. "
            "Completed files are reused; an interrupted file may restart."
        ) from exc
