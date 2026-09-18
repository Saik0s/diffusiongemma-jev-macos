"""Single-process local server entry point."""

import argparse
import os
import platform
from pathlib import Path


class ServeArguments(argparse.Namespace):
    command: str
    model: Path
    host: str
    port: int
    max_prompt_tokens: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve local DiffusionGemma structured decisions")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="Load a local model and start the decision API")
    serve.add_argument(
        "--model",
        type=Path,
        default=Path(
            os.environ.get(
                "JEV_MODEL_PATH",
                "~/.cache/lm-studio/models/mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit",
            )
        ),
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8017)
    serve.add_argument("--max-prompt-tokens", type=int, default=8192)
    args = parser.parse_args(namespace=ServeArguments())
    if args.max_prompt_tokens < 1:
        parser.error("--max-prompt-tokens must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        parser.error("Local inference requires macOS on Apple Silicon")

    import uvicorn

    from diffusion_jev.api import create_app
    from diffusion_jev.engine import LocalEngine

    def load_engine() -> LocalEngine:
        engine = LocalEngine(
            model_path=args.model.expanduser(), max_prompt_tokens=args.max_prompt_tokens
        )
        print(f"Model loaded in {engine.load_seconds:.1f}s. Starting the local API.", flush=True)
        return engine

    uvicorn.run(create_app(load_engine), host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
