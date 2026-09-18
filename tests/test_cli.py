import subprocess
import sys
from unittest.mock import Mock

import pytest

from diffusion_jev import cli


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
    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    provision = Mock()
    monkeypatch.setattr(cli, "resolve_model", provision)
    with pytest.raises(SystemExit) as error:
        cli.main(["start"])
    assert error.value.code == 2
    provision.assert_not_called()


def test_busy_port_does_not_provision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli, "preflight_port", Mock(side_effect=cli.ProvisioningError("busy")))
    provision = Mock()
    monkeypatch.setattr(cli, "resolve_model", provision)
    with pytest.raises(SystemExit) as error:
        cli.main(["start"])
    assert error.value.code == 2
    provision.assert_not_called()


def test_serve_missing_path_never_provisions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(cli.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(cli, "preflight_port", Mock())
    provision = Mock()
    monkeypatch.setattr(cli, "resolve_model", provision)
    with pytest.raises(SystemExit) as error:
        cli.main(["serve", "--model", "/nonexistent/jev-test-model"])
    assert error.value.code == 2
    provision.assert_not_called()


def test_demo_dispatch_preserves_failure_exit_status(monkeypatch: pytest.MonkeyPatch) -> None:
    from diffusion_jev import demo

    monkeypatch.setattr(cli.platform, "system", lambda: "Linux")
    child = Mock(return_value=1)
    monkeypatch.setattr(demo, "main", child)
    with pytest.raises(SystemExit) as error:
        cli.main(["demo", "search"])
    assert error.value.code == 1
    child.assert_called_once_with(["search"])
