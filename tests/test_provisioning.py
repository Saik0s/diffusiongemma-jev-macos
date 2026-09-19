import hashlib
import io
import json
import logging
import shutil
import socket
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import httpx
import pytest
from huggingface_hub import constants, file_download
from huggingface_hub.errors import LocalEntryNotFoundError
from huggingface_hub.utils._runtime import is_xet_available

from diffusion_jev import model_manifest, provisioning
from diffusion_jev.model_manifest import Artifact, Precision


@pytest.fixture
def checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.delenv("JEV_MODEL_PATH", raising=False)
    monkeypatch.setattr(provisioning, "LM_STUDIO_MODEL", tmp_path / "absent-lm-studio")
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    path = tmp_path / "snapshot"
    path.mkdir()
    contents = {
        "config.json": json.dumps({"model_type": "diffusion_gemma", "quantization": {"bits": 4}}),
        "model.safetensors.index.json": json.dumps({"weight_map": {"weight": "model.safetensors"}}),
        "model.safetensors": "weights",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
        "optiq_metadata.json": "{}",
        "optiq_vision.safetensors": "vision",
    }
    manifest = []
    for name, content in contents.items():
        data = content.encode()
        (path / name).write_bytes(data)
        manifest.append(Artifact(name, len(data), hashlib.sha256(data).hexdigest()))
    profile = replace(model_manifest.CHECKPOINTS["optiq4"], artifacts=tuple(manifest))
    monkeypatch.setattr(
        model_manifest, "CHECKPOINTS", {**model_manifest.CHECKPOINTS, "optiq4": profile}
    )
    return path


def test_explicit_model_does_not_access_hub(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = Mock(side_effect=AssertionError("unexpected hub access"))
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    assert provisioning.resolve_model(checkpoint) == checkpoint
    download.assert_not_called()


def test_missing_explicit_path_never_falls_back(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provisioning, "LM_STUDIO_MODEL", checkpoint)
    with pytest.raises(provisioning.ProvisioningError, match="does not exist"):
        provisioning.resolve_model(checkpoint / "missing")


def test_environment_path_and_flag_precedence(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JEV_MODEL_PATH", str(checkpoint))
    assert provisioning.resolve_model() == checkpoint
    with pytest.raises(provisioning.ProvisioningError, match="does not exist"):
        provisioning.resolve_model(checkpoint / "missing")


def test_lm_studio_reuse_is_offline(checkpoint: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(provisioning, "LM_STUDIO_MODEL", checkpoint)
    download = Mock(side_effect=AssertionError("unexpected hub access"))
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    assert provisioning.resolve_model() == checkpoint
    download.assert_not_called()


def test_warm_cache_verifies_without_online_lookup(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = Mock(return_value=str(checkpoint))
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    assert provisioning.resolve_model(offline=True) == checkpoint
    assert download.call_count == 1
    assert download.call_args.kwargs["local_files_only"] is True
    assert download.call_args.kwargs["revision"] == model_manifest.CHECKPOINTS["optiq4"].revision


def test_cold_cache_downloads_pinned_revision_then_verifies(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = Mock(side_effect=[LocalEntryNotFoundError("missing"), str(checkpoint)])
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    assert provisioning.resolve_model(cache_dir=tmp_path / "cache") == checkpoint
    assert download.call_count == 2
    assert download.call_args.kwargs["local_files_only"] is False
    assert download.call_args.kwargs["revision"] == model_manifest.CHECKPOINTS["optiq4"].revision
    assert download.call_args.kwargs["allow_patterns"] == [
        item.name for item in model_manifest.CHECKPOINTS["optiq4"].artifacts
    ]


def test_partial_cache_does_not_count_as_offline_success(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_text("{}")
    download = Mock(return_value=str(partial))
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    with pytest.raises(provisioning.ProvisioningError, match="No complete cached checkpoint"):
        provisioning.resolve_model(offline=True)
    assert download.call_count == 1


def test_partial_cache_resumes_online(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    download = Mock(side_effect=[str(partial), str(checkpoint)])
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    assert provisioning.resolve_model(cache_dir=tmp_path / "cache") == checkpoint
    assert download.call_count == 2


def test_native_hugging_face_complete_cache_is_reused_offline(
    checkpoint: Path,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "native-cache"
    snapshot = cache / ("models--" + model_manifest.MODEL_REPO.replace("/", "--"))
    snapshot = snapshot / "snapshots" / model_manifest.CHECKPOINTS["optiq4"].revision
    snapshot.mkdir(parents=True)
    for artifact in model_manifest.CHECKPOINTS["optiq4"].artifacts:
        (snapshot / artifact.name).symlink_to(checkpoint / artifact.name)
    assert provisioning.resolve_model(cache_dir=cache, offline=True) == snapshot


def test_native_hugging_face_partial_snapshot_is_rejected(
    checkpoint: Path,
    tmp_path: Path,
) -> None:
    cache = tmp_path / "native-cache"
    snapshot = cache / ("models--" + model_manifest.MODEL_REPO.replace("/", "--"))
    snapshot = snapshot / "snapshots" / model_manifest.CHECKPOINTS["optiq4"].revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").symlink_to(checkpoint / "config.json")
    with pytest.raises(provisioning.ProvisioningError, match="No complete cached checkpoint"):
        provisioning.resolve_model(cache_dir=cache, offline=True)


def test_hub_offline_environment_never_downloads(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", True)
    download = Mock(side_effect=LocalEntryNotFoundError("missing"))
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    with pytest.raises(provisioning.ProvisioningError, match="No complete cached checkpoint"):
        provisioning.resolve_model()
    assert download.call_count == 1


@pytest.mark.parametrize("replacement", [b"corrupt", b"truncated weights"])
def test_corrupt_cache_fails_integrity_check(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: bytes,
) -> None:
    (checkpoint / "model.safetensors").write_bytes(replacement)
    monkeypatch.setattr(provisioning, "snapshot_download", Mock(return_value=str(checkpoint)))
    with pytest.raises(provisioning.ProvisioningError, match="(checksum|size) mismatch"):
        provisioning.resolve_model()


def test_insufficient_space_prevents_download(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    download = Mock(side_effect=LocalEntryNotFoundError("missing"))
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    monkeypatch.setattr(shutil, "disk_usage", Mock(return_value=Mock(free=1)))
    with pytest.raises(provisioning.ProvisioningError, match="Insufficient disk space"):
        provisioning.resolve_model(cache_dir=tmp_path / "cache")
    assert download.call_count == 1


def test_cache_path_file_is_actionable(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "file"
    cache.write_text("not a directory")
    monkeypatch.setattr(
        provisioning, "snapshot_download", Mock(side_effect=LocalEntryNotFoundError("missing"))
    )
    with pytest.raises(provisioning.ProvisioningError, match="permissions"):
        provisioning.resolve_model(cache_dir=cache)


def test_download_errors_do_not_expose_credentials(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        provisioning,
        "snapshot_download",
        Mock(
            side_effect=[
                LocalEntryNotFoundError("missing"),
                RuntimeError("https://user:secret@proxy.invalid?token=secret"),
            ]
        ),
    )
    with pytest.raises(provisioning.ProvisioningError) as error:
        provisioning.resolve_model(cache_dir=tmp_path / "cache")
    assert "secret" not in str(error.value)
    assert "Completed files are reused; an interrupted file may restart" in str(error.value)


def test_native_timeout_logs_do_not_expose_download_url(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "TEST_ONLY_DOWNLOAD_SIGNATURE"
    url = f"https://download.invalid/model?signature={sentinel}"
    caplog.set_level(logging.DEBUG)
    caplog.set_level(logging.DEBUG, logger="huggingface_hub.file_download")
    original_level = file_download.logger.level
    monkeypatch.setattr(
        file_download, "http_stream_backoff", Mock(side_effect=httpx.ReadTimeout(sentinel))
    )

    def timeout() -> None:
        logging.getLogger("httpx").info("GET %s", url)
        logging.getLogger("httpcore.connection").debug("transport %s", url)
        logging.getLogger("diffusion_jev.test").warning("unrelated log remains visible")
        file_download.http_get(url, io.BytesIO(), _nb_retries=0)

    def download(*args: object, **kwargs: object) -> str:
        if kwargs["local_files_only"]:
            raise LocalEntryNotFoundError("missing")
        # Match snapshot_download's worker-thread logging behavior.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(timeout).result()
        raise AssertionError("Expected a timeout")

    monkeypatch.setattr(provisioning, "snapshot_download", download)
    with pytest.raises(provisioning.ProvisioningError) as error:
        provisioning.resolve_model(cache_dir=tmp_path / "cache")
    captured = capsys.readouterr()
    assert sentinel not in caplog.text + captured.out + captured.err + str(error.value)
    assert "unrelated log remains visible" in caplog.text
    assert "Downloading pinned" in captured.out
    assert file_download.logger.level == original_level
    file_download.logger.warning("restored logging")
    assert "restored logging" in caplog.text


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("originally_disabled", [False, True])
def test_snapshot_scopes_http_transport_and_restores_settings(
    checkpoint: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail: bool,
    originally_disabled: bool,
) -> None:
    monkeypatch.setattr(constants, "HF_HUB_DISABLE_XET", originally_disabled)
    logger = logging.getLogger("httpcore.proxy")
    original_level = logger.level

    def download(*args: object, **kwargs: object) -> str:
        assert not is_xet_available()
        assert not logger.isEnabledFor(logging.CRITICAL)
        if fail:
            raise RuntimeError("controlled transport failure")
        return str(checkpoint)

    monkeypatch.setattr(provisioning, "snapshot_download", download)
    if fail:
        with pytest.raises(RuntimeError, match="controlled transport failure"):
            provisioning._snapshot(
                checkpoint, offline=False, profile=model_manifest.CHECKPOINTS["optiq4"]
            )
    else:
        assert provisioning._snapshot(
            checkpoint, offline=False, profile=model_manifest.CHECKPOINTS["optiq4"]
        ) == checkpoint
    assert constants.HF_HUB_DISABLE_XET == originally_disabled
    assert logger.level == original_level


def test_weight_index_missing_shard_rejected(checkpoint: Path) -> None:
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "missing.safetensors"}})
    )
    with pytest.raises(provisioning.ProvisioningError, match="weight shard"):
        provisioning.validate_local_model(checkpoint)


def test_weight_index_path_traversal_rejected(checkpoint: Path) -> None:
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}})
    )
    with pytest.raises(provisioning.ProvisioningError, match="unsupported shard path"):
        provisioning.validate_local_model(checkpoint)


def test_busy_port_rejected_before_model_load() -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        with pytest.raises(provisioning.ProvisioningError, match="Cannot bind"):
            provisioning.preflight_port("127.0.0.1", listener.getsockname()[1])


@pytest.fixture
def standard_checkpoint(
    checkpoint: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    precision: Precision,
) -> Path:
    path = tmp_path / precision
    path.mkdir()
    config = provisioning.ModelConfig(
        model_type="diffusion_gemma",
        quantization=provisioning.QuantizationConfig(bits=8) if precision == "8bit" else None,
    )
    contents = {
        "config.json": config.model_dump_json(exclude_none=True),
        "model.safetensors.index.json": json.dumps({"weight_map": {
            "model.decoder.embed_tokens.weight": "model.safetensors",
            "model.encoder.vision_tower.patch_embedding.weight": "model.safetensors",
            "model.encoder.embed_vision.weight": "model.safetensors",
        }}),
        "model.safetensors": "weights",
        "tokenizer.json": "{}",
        "tokenizer_config.json": "{}",
    }
    manifest = []
    for name, content in contents.items():
        data = content.encode()
        (path / name).write_bytes(data)
        manifest.append(Artifact(name, len(data), hashlib.sha256(data).hexdigest()))
    profile = replace(model_manifest.CHECKPOINTS[precision], artifacts=tuple(manifest))
    monkeypatch.setattr(
        model_manifest, "CHECKPOINTS", {**model_manifest.CHECKPOINTS, precision: profile}
    )
    return path


@pytest.mark.parametrize("precision", ["8bit", "bf16"])
def test_standard_local_checkpoint_needs_no_optiq_sidecar(
    standard_checkpoint: Path, precision: Precision
) -> None:
    assert provisioning.validate_local_model(standard_checkpoint) == standard_checkpoint
    assert not (standard_checkpoint / "optiq_metadata.json").exists()


@pytest.mark.parametrize("precision", ["8bit", "bf16"])
def test_selected_precision_download_skips_lm_studio(
    standard_checkpoint: Path,
    checkpoint: Path,
    precision: Precision,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provisioning, "LM_STUDIO_MODEL", checkpoint)
    download = Mock(side_effect=[LocalEntryNotFoundError("missing"), str(standard_checkpoint)])
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    result = provisioning.resolve_model(precision=precision, cache_dir=tmp_path / "cache")
    assert result == standard_checkpoint
    profile = model_manifest.CHECKPOINTS[precision]
    assert download.call_count == 2
    assert all(call.args == (profile.repo,) for call in download.call_args_list)
    assert all(call.kwargs["revision"] == profile.revision for call in download.call_args_list)
    assert download.call_args.kwargs["allow_patterns"] == [item.name for item in profile.artifacts]


@pytest.mark.parametrize("precision", ["8bit", "bf16"])
def test_native_standard_cache_is_reused_and_verified_offline(
    standard_checkpoint: Path,
    precision: Precision,
    tmp_path: Path,
) -> None:
    profile = model_manifest.CHECKPOINTS[precision]
    cache = tmp_path / "cache"
    snapshot = cache / ("models--" + profile.repo.replace("/", "--"))
    snapshot = snapshot / "snapshots" / profile.revision
    snapshot.mkdir(parents=True)
    for artifact in profile.artifacts:
        (snapshot / artifact.name).symlink_to(standard_checkpoint / artifact.name)
    result = provisioning.resolve_model(cache_dir=cache, precision=precision, offline=True)
    assert result == snapshot
    (standard_checkpoint / "model.safetensors").write_text("corrupt")
    with pytest.raises(provisioning.ProvisioningError, match="checksum mismatch"):
        provisioning.resolve_model(cache_dir=cache, precision=precision, offline=True)


@pytest.mark.parametrize("precision", ["8bit", "bf16"])
def test_missing_selected_precision_never_falls_back_to_optiq(
    checkpoint: Path, precision: Precision, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provisioning, "LM_STUDIO_MODEL", checkpoint)
    download = Mock(side_effect=LocalEntryNotFoundError("missing"))
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    with pytest.raises(provisioning.ProvisioningError, match="No complete cached checkpoint"):
        provisioning.resolve_model(precision=precision, offline=True)
    assert download.call_count == 1
    assert download.call_args.args == (model_manifest.CHECKPOINTS[precision].repo,)


@pytest.mark.parametrize("precision", ["8bit", "bf16"])
def test_explicit_path_overrides_managed_precision_with_notice(
    checkpoint: Path, precision: Precision, monkeypatch: pytest.MonkeyPatch
) -> None:
    download = Mock(side_effect=AssertionError("Unexpected download"))
    monkeypatch.setattr(provisioning, "snapshot_download", download)
    messages: list[str] = []
    assert provisioning.resolve_model(checkpoint, precision=precision, report=messages.append)
    assert "explicit path takes priority" in messages[0]
    monkeypatch.setenv("JEV_MODEL_PATH", str(checkpoint))
    assert provisioning.resolve_model(precision=precision, report=messages.append) == checkpoint
    download.assert_not_called()


@pytest.mark.parametrize("precision", ["8bit", "bf16"])
def test_standard_profile_requires_indexed_vision(
    standard_checkpoint: Path, precision: Precision
) -> None:
    (standard_checkpoint / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.decoder.weight": "model.safetensors"}})
    )
    with pytest.raises(provisioning.ProvisioningError, match="missing indexed vision"):
        provisioning.validate_local_model(
            standard_checkpoint, profile=model_manifest.CHECKPOINTS[precision]
        )


@pytest.mark.parametrize("precision", ["8bit", "bf16"])
def test_managed_profile_rejects_wrong_quantization(
    standard_checkpoint: Path, precision: Precision
) -> None:
    (standard_checkpoint / "config.json").write_text(
        json.dumps({"model_type": "diffusion_gemma", "quantization": {"bits": 4}})
    )
    with pytest.raises(provisioning.ProvisioningError, match="selected precision"):
        provisioning.validate_local_model(
            standard_checkpoint, profile=model_manifest.CHECKPOINTS[precision]
        )


def test_optiq_sidecar_requirement_is_preserved(checkpoint: Path) -> None:
    (checkpoint / "optiq_vision.safetensors").write_bytes(b"")
    with pytest.raises(provisioning.ProvisioningError, match="optiq_vision"):
        provisioning.validate_local_model(checkpoint)


@pytest.mark.parametrize("precision", ["8bit", "bf16"])
def test_disk_preflight_uses_selected_artifact_sizes_and_cache_volume(
    checkpoint: Path,
    precision: Precision,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "external-volume" / "cache"
    profile = model_manifest.CHECKPOINTS[precision]
    needed = sum(artifact.size for artifact in profile.artifacts) + provisioning.DISK_RESERVE
    usage = Mock(return_value=Mock(free=needed - 1))
    monkeypatch.setattr(shutil, "disk_usage", usage)
    with pytest.raises(provisioning.ProvisioningError, match="Insufficient disk space"):
        provisioning._preflight_cache(cache, None, profile)
    usage.assert_called_once_with(cache)
    usage.return_value = Mock(free=needed)
    provisioning._preflight_cache(cache, None, profile)
