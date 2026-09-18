"""Reproducible local benchmark; reports contain no request or response content."""

import argparse
import math
import os
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal

from diffusion_jev.api import DecisionEngine
from diffusion_jev.benchmark_cases import BenchmarkCase, quality_cases, scaling_case
from diffusion_jev.schemas import (
    Answer,
    DecisionOptions,
    DecisionResponse,
    NoulAnswer,
    StrictModel,
    Usage,
)


@dataclass(frozen=True)
class Configuration:
    name: str
    canvas_length: int | None = None
    projection: Literal["full", "labels"] = "labels"
    mode: Literal["packed", "independent"] = "packed"


CONFIGURATIONS = (
    Configuration("baseline_full_256", 256, "full"),
    Configuration("compact_full", projection="full"),
    Configuration("optimized"),
    Configuration("independent", mode="independent"),
)


class Measurement(StrictModel):
    case: str
    configuration: str
    seed: int
    wall_ms: float
    correct: int
    questions: int
    usage: Usage


class Aggregate(StrictModel):
    requests: int
    questions: int
    correct: int
    accuracy: float
    wall_p50_ms: float
    wall_p95_ms: float
    questions_per_second: float
    prefill_median_ms: float
    decode_median_ms: float
    prompt_tokens_median: float
    canvas_tokens_median: float
    peak_memory_gb: float


class ProbabilityComparison(StrictModel):
    compared_questions: int = 0
    max_probability_delta: float = 0.0
    argmax_disagreements: int = 0
    tolerance: float = 0.01
    passed: bool = True


class Environment(StrictModel):
    system: str
    architecture: str
    chip: str
    memory_bytes: int | None
    python: str
    packages: dict[str, str]
    model_basename: str
    reference_model_hf_id: str = "mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit"


class BenchmarkReport(StrictModel):
    schema_version: int = 1
    timestamp_utc: str
    environment: Environment
    repeats: int
    configurations: dict[str, DecisionOptions]
    seeds: list[int]
    percentile_method: str = "nearest-rank: sorted[ceil(p*n)-1]"
    correctness_method: str = "Noul >= 0.5; Choice exact; Score probability argmax level"
    warmup_policy: str = "One excluded warmup per case/configuration before measured repeats"
    order_policy: str = "Rotate configurations each repeat; reverse on odd repeats"
    load_seconds: float
    first_request: Measurement
    quality: dict[str, Aggregate]
    quality_by_case: dict[str, dict[str, Aggregate]]
    projection_equivalence: ProbabilityComparison
    scaling: dict[str, Aggregate]
    measurements: list[Measurement]


def nearest_rank(values: list[float], percentile: float) -> float:
    if not values or not 0 < percentile <= 1:
        raise ValueError("Nonempty values and percentile in (0, 1] required")
    return sorted(values)[math.ceil(percentile * len(values)) - 1]


def aggregate(measurements: list[Measurement]) -> Aggregate:
    if not measurements:
        raise ValueError("At least one measurement is required")
    questions = sum(item.questions for item in measurements)
    correct = sum(item.correct for item in measurements)
    elapsed = sum(item.wall_ms for item in measurements)
    if elapsed <= 0:
        raise ValueError("Positive wall latency is required")
    return Aggregate(
        requests=len(measurements), questions=questions, correct=correct,
        accuracy=correct / questions,
        wall_p50_ms=nearest_rank([item.wall_ms for item in measurements], 0.5),
        wall_p95_ms=nearest_rank([item.wall_ms for item in measurements], 0.95),
        questions_per_second=questions * 1000 / elapsed,
        prefill_median_ms=statistics.median(item.usage.prefill_ms for item in measurements),
        decode_median_ms=statistics.median(item.usage.decode_ms for item in measurements),
        prompt_tokens_median=statistics.median(item.usage.prompt_tokens for item in measurements),
        canvas_tokens_median=statistics.median(item.usage.canvas_tokens for item in measurements),
        peak_memory_gb=max(item.usage.peak_memory_gb for item in measurements),
    )


def probabilities(answer: Answer) -> dict[str, float]:
    if isinstance(answer, NoulAnswer):
        return {"yes": answer.noul, "no": 1 - answer.noul}
    return answer.probabilities


def compare_probabilities(
    reference: DecisionResponse, optimized: DecisionResponse, tolerance: float = 0.01
) -> ProbabilityComparison:
    if reference.answers.keys() != optimized.answers.keys():
        raise ValueError("Projection comparison requires matching questions")
    largest = 0.0
    disagreements = 0
    for key, answer in reference.answers.items():
        left, right = probabilities(answer), probabilities(optimized.answers[key])
        if answer.type != optimized.answers[key].type or left.keys() != right.keys():
            raise ValueError("Projection comparison requires matching labels and types")
        largest = max(largest, max(abs(value - right[label]) for label, value in left.items()))
        disagreements += max(left, key=left.__getitem__) != max(right, key=right.__getitem__)
    return ProbabilityComparison(
        compared_questions=len(reference.answers), max_probability_delta=largest,
        argmax_disagreements=disagreements, tolerance=tolerance,
        passed=largest <= tolerance and disagreements == 0,
    )


def measure(
    engine: DecisionEngine, case: BenchmarkCase, configuration: Configuration, seed: int
) -> tuple[Measurement, DecisionResponse]:
    options = DecisionOptions(
        seed=seed, canvas_length=configuration.canvas_length,
        projection=configuration.projection, mode=configuration.mode,
    )
    request = case.request.model_copy(update={"options": options})
    started = time.perf_counter()
    response = engine.decide(request)
    wall_ms = (time.perf_counter() - started) * 1000
    if set(response.answers) != set(request.questions):
        raise ValueError("Benchmark response has missing or unexpected answers")
    if response.usage.questions != len(request.questions):
        raise ValueError("Benchmark response has incorrect question accounting")
    return Measurement(
        case=case.name, configuration=configuration.name, seed=seed, wall_ms=wall_ms,
        correct=case.correct_count(response), questions=len(request.questions),
        usage=response.usage,
    ), response


def configuration_order(repeat: int) -> tuple[Configuration, ...]:
    offset = repeat % len(CONFIGURATIONS)
    order = CONFIGURATIONS[offset:] + CONFIGURATIONS[:offset]
    return order[::-1] if repeat % 2 else order


def run_benchmark(
    engine: DecisionEngine, *, repeats: int, load_seconds: float, environment: Environment
) -> BenchmarkReport:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    cases = quality_cases()
    first, _ = measure(engine, cases[0], CONFIGURATIONS[0], 0)
    for case in cases:
        for configuration in CONFIGURATIONS:
            measure(engine, case, configuration, 0)
    measurements: list[Measurement] = []
    comparison = ProbabilityComparison()
    for repeat in range(repeats):
        for case in cases:
            paired: dict[str, DecisionResponse] = {}
            for configuration in configuration_order(repeat):
                result, response = measure(engine, case, configuration, repeat)
                measurements.append(result)
                if configuration.name in {"compact_full", "optimized"}:
                    paired[configuration.name] = response
            delta = compare_probabilities(paired["compact_full"], paired["optimized"])
            comparison = ProbabilityComparison(
                compared_questions=comparison.compared_questions + delta.compared_questions,
                max_probability_delta=max(
                    comparison.max_probability_delta, delta.max_probability_delta
                ),
                argmax_disagreements=comparison.argmax_disagreements + delta.argmax_disagreements,
                passed=comparison.passed and delta.passed,
            )
    quality = {
        config.name: aggregate([m for m in measurements if m.configuration == config.name])
        for config in CONFIGURATIONS
    }
    by_case = {
        config.name: {
            case.name: aggregate([
                m for m in measurements if m.configuration == config.name and m.case == case.name
            ]) for case in cases
        } for config in CONFIGURATIONS
    }
    scaling: dict[str, Aggregate] = {}
    scale_cases = [scaling_case(count) for count in (1, 4, 8, 16, 32)]
    for case in scale_cases:
        measure(engine, case, CONFIGURATIONS[2], 0)
    scale_measurements: list[Measurement] = []
    for repeat in range(repeats):
        for case in scale_cases if repeat % 2 == 0 else scale_cases[::-1]:
            result, _ = measure(engine, case, CONFIGURATIONS[2], repeat)
            scale_measurements.append(result)
    for case in scale_cases:
        scaling[case.name] = aggregate([m for m in scale_measurements if m.case == case.name])
    return BenchmarkReport(
        timestamp_utc=datetime.now(UTC).isoformat(), environment=environment, repeats=repeats,
        configurations={
            config.name: DecisionOptions(
                canvas_length=config.canvas_length, projection=config.projection, mode=config.mode
            ) for config in CONFIGURATIONS
        },
        seeds=list(range(repeats)),
        load_seconds=load_seconds, first_request=first, quality=quality, quality_by_case=by_case,
        projection_equivalence=comparison, scaling=scaling,
        measurements=measurements + scale_measurements,
    )


def environment_metadata(model_path: Path) -> Environment:
    chip = platform.machine()
    memory: int | None = None
    if platform.system() == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.memsize"):
            result = subprocess.run(
                ["/usr/sbin/sysctl", "-n", key], capture_output=True, text=True, check=False
            )
            if result.returncode == 0:
                if key == "hw.memsize":
                    memory = int(result.stdout.strip())
                else:
                    chip = result.stdout.strip()
    packages: dict[str, str] = {}
    for name in ("diffusion-jev", "mlx", "mlx-optiq", "pydantic"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return Environment(
        system=f"{platform.system()} {platform.release()}", architecture=platform.machine(),
        chip=chip, memory_bytes=memory, python=platform.python_version(), packages=packages,
        model_basename=model_path.name,
    )


class BenchmarkArguments(argparse.Namespace):
    model: Path
    repeats: int
    output: Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path(os.environ.get(
        "JEV_MODEL_PATH",
        "~/.cache/lm-studio/models/mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit",
    )))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmark-results.local.json"))
    args = parser.parse_args(namespace=BenchmarkArguments())
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    from diffusion_jev.engine import LocalEngine

    model_path = args.model.expanduser()
    started = time.perf_counter()
    engine = LocalEngine(model_path)
    load_seconds = time.perf_counter() - started
    print("Model loaded; measuring cold request, warmed quality, and scaling.", flush=True)
    report = run_benchmark(
        engine, repeats=args.repeats, load_seconds=load_seconds,
        environment=environment_metadata(model_path),
    )
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    print("Benchmark complete. Aggregate report written.", flush=True)


if __name__ == "__main__":
    main()
