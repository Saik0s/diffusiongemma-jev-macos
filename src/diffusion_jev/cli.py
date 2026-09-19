"""Single-process local server entry point."""

import argparse
import os
import platform
import sys
from pathlib import Path

from diffusion_jev.model_manifest import CHECKPOINTS, Precision
from diffusion_jev.provisioning import (
    LM_STUDIO_MODEL,
    ProvisioningError,
    preflight_port,
    resolve_model,
    validate_local_model,
)


class ServeArguments(argparse.Namespace):
    command: str
    model: Path | None
    host: str
    port: int
    max_prompt_tokens: int
    reasoning_tokens: int
    cache_limit_mib: int
    cache_dir: Path | None
    offline: bool
    precision: Precision


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "demo":
        from diffusion_jev.demo import main as demo_main

        raise SystemExit(demo_main(argv[1:]))
    if argv and argv[0] == "benchmark-search":
        from diffusion_jev.retrieval_benchmark import main as benchmark_main

        benchmark_main(argv[1:])
        return
    parser = argparse.ArgumentParser(description="Serve local DiffusionGemma structured decisions")
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("serve", "Load an existing local model; never download a checkpoint"),
        ("start", "Reuse a local model or download the pinned checkpoint, then serve"),
    ):
        serve = commands.add_parser(command, help=help_text)
        serve.add_argument("--model", type=Path, help="Local model directory (or JEV_MODEL_PATH)")
        serve.add_argument("--host", default="127.0.0.1")
        serve.add_argument("--port", type=int, default=8017)
        serve.add_argument("--max-prompt-tokens", type=int, default=8192)
        serve.add_argument(
            "--reasoning-tokens", type=int, choices=(0, 256, 512), default=0,
            help="Experimental reasoning budget before decisions (default: 0); may be slower",
        )
        serve.add_argument(
            "--cache-limit-mib", type=int, default=512,
            help="MLX allocator cache limit in MiB (default: 512); not a total memory limit",
        )
        if command == "start":
            serve.add_argument(
                "--precision", choices=tuple(CHECKPOINTS), default="optiq4",
                help="Managed precision (default: optiq4); explicit model paths override",
            )
            serve.add_argument("--cache-dir", type=Path, help="Hugging Face model cache directory")
            serve.add_argument("--offline", action="store_true", help="Never download model files")
    commands.add_parser("demo", help="Run structured-decision demos", add_help=False)
    commands.add_parser(
        "benchmark-search", help="Evaluate retrieval on CodeSearchNet Python", add_help=False
    )
    args = parser.parse_args(argv, namespace=ServeArguments())
    if args.max_prompt_tokens < 1:
        parser.error("--max-prompt-tokens must be positive")
    if args.cache_limit_mib < 0:
        parser.error("--cache-limit-mib must be nonnegative")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Local inference requires macOS on Apple Silicon")
    try:
        preflight_port(args.host, args.port)
        if args.command == "start":
            model_path = resolve_model(
                args.model,
                cache_dir=args.cache_dir,
                offline=args.offline,
                precision=args.precision,
                report=lambda message: print(message, flush=True),
            )
        else:
            model_path = validate_local_model(
                args.model or Path(os.environ.get("JEV_MODEL_PATH", str(LM_STUDIO_MODEL)))
            )
    except ProvisioningError as exc:
        parser.error(str(exc))
    except OSError:
        parser.error("Cannot read the local model. Check its files and permissions.")
    except KeyboardInterrupt:
        parser.exit(
            130,
            "\nStartup interrupted. Rerun the command; completed files are reused, "
            "but an interrupted file may restart.\n",
        )

    import uvicorn

    from diffusion_jev.api import create_app
    from diffusion_jev.engine import LocalEngine

    def load_engine() -> LocalEngine:
        engine = LocalEngine(
            model_path=model_path, max_prompt_tokens=args.max_prompt_tokens,
            cache_limit_bytes=args.cache_limit_mib * 1024**2,
            reasoning_tokens=args.reasoning_tokens,
        )
        print(f"Model loaded in {engine.load_seconds:.1f}s. Starting the local API.", flush=True)
        return engine

    uvicorn.run(create_app(load_engine), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
