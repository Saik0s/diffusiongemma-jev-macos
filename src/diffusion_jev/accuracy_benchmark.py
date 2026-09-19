"""Run one frozen accuracy cohort in-process, with content-free query checkpoints."""

import argparse
import hashlib
import os
import re
import time
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from diffusion_jev.benchmark import Environment, environment_metadata
from diffusion_jev.component_source import RestorationInfo, inspect_router_source
from diffusion_jev.retrieval_benchmark import (
    DiagnosticSummary,
    QueryMeasurement,
    Report,
    run_retrieval,
)
from diffusion_jev.retrieval_data import (
    DEFAULT_CACHE,
    SOURCES,
    EvaluationSplit,
    load_data,
    select_split,
    split_description,
)
from diffusion_jev.schemas import DecisionOptions, StrictModel

RUNTIME_SOURCES = (
    "engine.py", "canvas.py", "schemas.py", "precision.py", "projection.py",
    "runtime_types.py", "reasoning.py", "reasoning_boundaries.py", "trajectory.py",
    "components.py", "component_source.py",
    "retrieval_data.py", "retrieval_benchmark.py", "accuracy_benchmark.py", "benchmark.py",
    "model_manifest.py",
)


class RunConfiguration(StrictModel):
    label: str
    split: EvaluationSplit
    query_ids: list[str]
    source_sha256: dict[str, str]
    model_revision: str
    revision_evidence: Literal["user-supplied", "snapshot-directory"]
    model_file_sha256: dict[str, str]
    runtime_file_sha256: dict[str, str]
    runtime_dependency_sha256: dict[str, str] = Field(default_factory=dict)
    environment: Environment
    runtime_policy: Literal["baseline", "router-fp32"]
    options: DecisionOptions
    group_size: int
    max_prompt_tokens: int
    cache_limit_bytes: int | None
    diagnostics: bool = False
    reasoning_tokens: Literal[0, 256, 512] = 0
    trajectory_steps: Literal[1, 2] = 1
    restoration_info: RestorationInfo | None = None
    candidate_order: str = "Pinned source BM25 order; stable ranking ties"


class Checkpoint(StrictModel):
    schema_version: int = 1
    configuration: RunConfiguration
    measurements: list[QueryMeasurement]


class AccuracyReport(StrictModel):
    schema_version: int = 1
    configuration: RunConfiguration
    load_seconds: float
    load_peak_memory_gb: float
    load_active_memory_gb: float
    load_cache_memory_gb: float
    resumed_queries: int
    retrieval: Report


def atomic_write(path: Path, value: StrictModel) -> None:
    pending = path.with_name(path.name + ".tmp")
    with pending.open("w", encoding="utf-8") as stream:
        stream.write(value.model_dump_json(indent=2) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def revision_identity(
    model: Path, supplied: str | None
) -> tuple[str, Literal["user-supplied", "snapshot-directory"]]:
    if supplied is not None:
        if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", supplied) is None:
            raise ValueError("Model revision must be an immutable 40 or 64 hex digest")
        return supplied.lower(), "user-supplied"
    resolved = model.resolve()
    if resolved.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", resolved.name):
        return resolved.name, "snapshot-directory"
    raise ValueError("Supply --model-revision for a checkpoint outside an immutable snapshot")


def stream_sha256(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def file_fingerprints(root: Path, names: tuple[str, ...]) -> dict[str, str]:
    return {
        name: stream_sha256(root / name)
        for name in names if (root / name).is_file()
    }


class WeightIndex(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    weight_map: dict[str, str]


def model_fingerprints(root: Path) -> dict[str, str]:
    """Hash loaded weight files as well as metadata without buffering whole shards."""
    weights = {path.name for path in root.glob("model*.safetensors") if path.is_file()}
    index = root / "model.safetensors.index.json"
    if index.is_file():
        manifest = WeightIndex.model_validate_json(index.read_text(encoding="utf-8"))
        for name in manifest.weight_map.values():
            relative = Path(name)
            if (relative.is_absolute() or ".." in relative.parts
                    or relative.suffix != ".safetensors"):
                raise ValueError("Weight index must reference relative safetensor files")
            if not (root / relative).is_file():
                raise ValueError("Weight index references a missing shard")
            weights.add(relative.as_posix())
    if not weights:
        raise ValueError("Model fingerprint requires local weight shards")
    for sidecar in ("optiq/optiq_vision.safetensors", "optiq_vision.safetensors"):
        if (root / sidecar).is_file():
            weights.add(sidecar)
    metadata = (
        "config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
        "model.safetensors.index.json", "optiq_metadata.json", "generation_config.json",
        "processor_config.json",
    )
    return file_fingerprints(root, metadata + tuple(sorted(weights)))


def dependency_fingerprints() -> dict[str, str]:
    sources = {
        "mlx-optiq": (
            "optiq/vlm/diffusion_gemma/loader.py",
            "optiq/vlm/_mlxvlm/models/diffusion_gemma/language.py",
            "optiq/vlm/_mlxvlm/models/diffusion_gemma/diffusion_gemma.py",
            "optiq/vlm/_mlxvlm/models/diffusion_gemma/config.py",
            "optiq/vlm/_mlxvlm/models/cache.py",
            "optiq/vlm/_mlxvlm/generate/diffusion.py",
            "optiq/vlm/_mlxvlm/generate/common.py",
            "optiq/vlm/_mlxvlm/tokenizer_utils.py",
        ),
        "mlx-lm": ("mlx_lm/models/switch_layers.py", "mlx_lm/models/cache.py"),
    }
    fingerprints: dict[str, str] = {}
    for package, names in sources.items():
        try:
            installed = distribution(package)
        except PackageNotFoundError:
            continue
        for name in names:
            source = Path(str(installed.locate_file(name)))
            if not source.is_file():
                raise ValueError("Required inference dependency source is missing")
            fingerprints[f"{package}/{name}"] = stream_sha256(source)
    return fingerprints


def restore_checkpoint(path: Path, configuration: RunConfiguration) -> list[QueryMeasurement]:
    checkpoint = Checkpoint.model_validate_json(path.read_text(encoding="utf-8"))
    if checkpoint.configuration != configuration:
        raise ValueError("Checkpoint model, runtime, cohort, or options differ")
    if [item.qid for item in checkpoint.measurements] != configuration.query_ids[
        :len(checkpoint.measurements)
    ]:
        raise ValueError("Checkpoint query IDs are not an exact cohort prefix")
    return checkpoint.measurements


class Arguments(argparse.Namespace):
    model: Path
    model_revision: str | None
    split: EvaluationSplit
    label: str
    runtime_policy: Literal["baseline", "router-fp32"]
    mode: Literal["packed", "independent"]
    projection: Literal["labels", "labels-fp32", "full"]
    samples: int
    canvas_length: int | None
    seed: int
    group_size: int
    max_prompt_tokens: int
    cache_limit_mib: int | None
    cache: Path
    output: Path
    resume: bool
    diagnostics: bool
    reasoning_tokens: Literal[0, 256, 512]
    trajectory_steps: Literal[1, 2]
    router_bf16_source: Path | None


class EngineOverrides(TypedDict, total=False):
    reasoning_tokens: Literal[0, 256, 512]
    trajectory_steps: Literal[1, 2]
    router_bf16_source: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--split", choices=("historical", "development", "validation", "test"),
                        required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--runtime-policy", choices=("baseline", "router-fp32"), default="baseline")
    parser.add_argument("--mode", choices=("packed", "independent"), default="packed")
    parser.add_argument("--projection", choices=("labels", "labels-fp32", "full"), default="labels")
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--canvas-length", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=5)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--cache-limit-mib", type=int)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--reasoning-tokens", type=int, choices=(0, 256, 512), default=0)
    parser.add_argument("--trajectory-steps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--router-bf16-source", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv, namespace=Arguments())
    if args.trajectory_steps == 2 and args.projection != "full":
        parser.error("--trajectory-steps 2 requires --projection full")
    if not 1 <= args.group_size <= 30 or args.max_prompt_tokens < 1:
        parser.error("group-size must be 1..30 and max-prompt-tokens must be positive")
    if args.cache_limit_mib is not None and args.cache_limit_mib < 0:
        parser.error("cache-limit-mib must be nonnegative")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", args.label):
        parser.error("label must be a short identifier containing only letters, digits, . _ -")
    model = args.model.expanduser()
    output = args.output.expanduser()
    checkpoint_path = output.with_name(output.name + ".checkpoint.json")
    try:
        options = DecisionOptions(seed=args.seed, samples=args.samples, mode=args.mode,
                                  projection=args.projection, canvas_length=args.canvas_length)
        revision, evidence = revision_identity(model, args.model_revision)
        restoration_info = (
            inspect_router_source(args.router_bf16_source.expanduser())
            if args.router_bf16_source is not None else None
        )
        if not args.resume and (output.exists() or checkpoint_path.exists()):
            raise ValueError("Output exists; use a new output or --resume")
        print("Hashing local model artifacts for reproducible resume identity.", flush=True)
        model_hashes = model_fingerprints(model)
        data = select_split(load_data(300, args.cache.expanduser()), args.split)
        configuration = RunConfiguration(
            label=args.label, split=args.split, query_ids=[query.qid for query in data.queries],
            source_sha256={source.name: source.sha256 for source in SOURCES},
            model_revision=revision,
            revision_evidence=evidence,
            model_file_sha256=model_hashes,
            runtime_file_sha256=file_fingerprints(Path(__file__).parent, RUNTIME_SOURCES),
            runtime_dependency_sha256=dependency_fingerprints(),
            environment=environment_metadata(model), runtime_policy=args.runtime_policy,
            options=options, group_size=args.group_size, max_prompt_tokens=args.max_prompt_tokens,
            cache_limit_bytes=(args.cache_limit_mib * 1024**2
                               if args.cache_limit_mib is not None else None),
            diagnostics=args.diagnostics,
            reasoning_tokens=args.reasoning_tokens,
            trajectory_steps=args.trajectory_steps,
            restoration_info=restoration_info,
        )
        completed = restore_checkpoint(checkpoint_path, configuration) if args.resume else []
        checkpoint = Checkpoint(configuration=configuration, measurements=list(completed))
        atomic_write(checkpoint_path, checkpoint)

        from diffusion_jev.engine import LocalEngine

        started = time.perf_counter()
        engine_overrides: EngineOverrides = {}
        if args.reasoning_tokens:
            engine_overrides["reasoning_tokens"] = args.reasoning_tokens
        if args.trajectory_steps != 1:
            engine_overrides["trajectory_steps"] = args.trajectory_steps
        if args.router_bf16_source is not None:
            engine_overrides["router_bf16_source"] = args.router_bf16_source.expanduser()
        engine = LocalEngine(
            model, max_prompt_tokens=args.max_prompt_tokens,
            router_precision="fp32" if args.runtime_policy == "router-fp32" else "native",
            cache_limit_bytes=configuration.cache_limit_bytes,
            diagnostics=args.diagnostics,
            **engine_overrides,
        )
        load_seconds = time.perf_counter() - started
        if engine.restoration_info != configuration.restoration_info:
            raise ValueError("Restored router identity differs from pre-load inspection")
        print(f"Loaded model; {len(completed)}/{len(data.queries)} queries already recorded.",
              flush=True)

        def save_query(item: QueryMeasurement) -> None:
            checkpoint.measurements.append(item)
            atomic_write(checkpoint_path, checkpoint)

        def read_diagnostics() -> DiagnosticSummary | None:
            diagnostic = engine.last_diagnostics
            if diagnostic is None:
                return None
            return DiagnosticSummary(
                slots=diagnostic.slots, allowed_label_mass_min=diagnostic.allowed_label_mass_min,
                allowed_label_mass_mean=diagnostic.allowed_label_mass_mean,
                invalid_argmax_count=diagnostic.invalid_argmax_count,
            )

        report = run_retrieval(
            data, engine.decide, group_size=args.group_size, options=options,
            selection=split_description(args.split), completed=completed, on_measurement=save_query,
            read_diagnostics=read_diagnostics if args.diagnostics else None,
        )
        atomic_write(output, AccuracyReport(
            configuration=configuration, load_seconds=load_seconds,
            load_peak_memory_gb=engine.load_peak_memory_gb,
            load_active_memory_gb=engine.load_active_memory_gb,
            load_cache_memory_gb=engine.load_cache_memory_gb,
            resumed_queries=len(completed), retrieval=report,
        ))
    except (OSError, ValueError, RuntimeError):
        parser.exit(1, "Accuracy run failed; check model identity, options, files, or runtime. "
                    "Completed query checkpoints are retained; no input content logged.\n")
    print(f"Complete: {report.successful} successful, {report.failed} failed queries.", flush=True)
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
