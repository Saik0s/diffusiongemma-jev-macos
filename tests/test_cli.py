import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from diffusion_jev import cli
from diffusion_jev.provisioning import ProvisioningError


@pytest.mark.parametrize("command", [[], ["start"], ["serve"], ["demo"], ["benchmark-search"]])
def test_help_never_imports_gpu(command: list[str]) -> None:
    code = (
        "import sys; from diffusion_jev.cli import main; "
        f"sys.argv = ['jev-local', *{command!r}, '--help']; "
        "\ntry: main()\nfinally: assert 'mlx.core' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_unsupported_platform_does_not_provision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    provision = Mock()
    monkeypatch.setattr(cli, "resolve_model", provision)
    with pytest.raises(SystemExit) as error:
        cli.main(["start"])
    assert error.value.code == 2
    provision.assert_not_called()


def test_busy_port_does_not_provision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli, "preflight_port", Mock(side_effect=ProvisioningError("busy")))
    provision = Mock()
    monkeypatch.setattr(cli, "resolve_model", provision)
    with pytest.raises(SystemExit) as error:
        cli.main(["start"])
    assert error.value.code == 2
    provision.assert_not_called()


def test_serve_missing_path_never_provisions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli, "preflight_port", Mock())
    provision = Mock()
    monkeypatch.setattr(cli, "resolve_model", provision)
    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--model", "/nonexistent/jev-test-model"])
    assert error.value.code == 2
    provision.assert_not_called()


def test_demo_dispatch_preserves_failure_exit_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from diffusion_jev import demo

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    child = Mock(return_value=1)
    monkeypatch.setattr(demo, "main", child)
    with pytest.raises(SystemExit) as error:
        cli.main(["demo", "search"])
    assert error.value.code == 1
    child.assert_called_once_with(["search"])


@pytest.mark.parametrize("precision", ["optiq4", "8bit", "bf16"])
def test_start_passes_selected_precision_to_provisioning(
    precision: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli, "preflight_port", Mock())
    provision = Mock(side_effect=ProvisioningError("Stop before model loading"))
    monkeypatch.setattr(cli, "resolve_model", provision)
    with pytest.raises(SystemExit) as error:
        cli.main(["start", "--precision", precision])
    assert error.value.code == 2
    assert provision.call_args.kwargs["precision"] == precision


def test_start_defaults_to_optiq4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli, "preflight_port", Mock())
    provision = Mock(side_effect=ProvisioningError("Stop before model loading"))
    monkeypatch.setattr(cli, "resolve_model", provision)
    with pytest.raises(SystemExit):
        cli.main(["start"])
    assert provision.call_args.kwargs["precision"] == "optiq4"


def test_invalid_precision_fails_before_provisioning(monkeypatch: pytest.MonkeyPatch) -> None:
    provision = Mock()
    monkeypatch.setattr(cli, "resolve_model", provision)
    with pytest.raises(SystemExit) as error:
        cli.main(["start", "--precision", "4bit"])
    assert error.value.code == 2
    provision.assert_not_called()


def test_negative_cache_limit_fails_before_provisioning(monkeypatch: pytest.MonkeyPatch) -> None:
    provision = Mock()
    monkeypatch.setattr(cli, "resolve_model", provision)
    with pytest.raises(SystemExit) as error:
        cli.main(["start", "--cache-limit-mib", "-1"])
    assert error.value.code == 2
    provision.assert_not_called()


@pytest.mark.parametrize("command", ["start", "serve"])
@pytest.mark.parametrize(
    "arguments, expected_mib, expected_reasoning",
    [
        ([], 512, 0),
        (["--cache-limit-mib", "0"], 0, 0),
        (["--reasoning-tokens", "256"], 512, 256),
        (["--reasoning-tokens", "512"], 512, 512),
    ],
)
def test_server_factory_passes_process_settings(
    command: str, arguments: list[str], expected_mib: int, expected_reasoning: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    from diffusion_jev import api, engine

    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli, "preflight_port", Mock())
    monkeypatch.setattr(cli, "resolve_model", Mock(return_value=Path("/verified-model")))
    monkeypatch.setattr(cli, "validate_local_model", Mock(return_value=Path("/verified-model")))
    factory = Mock(return_value=Mock(load_seconds=0.1))
    app = Mock()
    monkeypatch.setattr(engine, "LocalEngine", factory)
    monkeypatch.setattr(api, "create_app", app)
    monkeypatch.setattr(uvicorn, "run", Mock())
    cli.main([command, *arguments])
    factory.assert_not_called()
    app.call_args.args[0]()
    factory.assert_called_once_with(
        model_path=Path("/verified-model"), max_prompt_tokens=8192,
        cache_limit_bytes=expected_mib * 1024**2,
        reasoning_tokens=expected_reasoning,
    )


@pytest.mark.parametrize("command", ["start", "serve"])
@pytest.mark.parametrize("budget", ["-1", "1", "1024", "invalid"])
def test_invalid_reasoning_budget_fails_before_model_work(
    command: str, budget: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preflight = Mock()
    provision = Mock()
    validate = Mock()
    monkeypatch.setattr(cli, "preflight_port", preflight)
    monkeypatch.setattr(cli, "resolve_model", provision)
    monkeypatch.setattr(cli, "validate_local_model", validate)
    with pytest.raises(SystemExit) as error:
        cli.main([command, "--reasoning-tokens", budget])
    assert error.value.code == 2
    preflight.assert_not_called()
    provision.assert_not_called()
    validate.assert_not_called()
