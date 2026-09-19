"""Synthetic coding evaluation with grouped counts and payload-free reports.

Select configuration on development only; run locked once the winner is fixed.
This measures this synthetic suite, not native JEV or general coding ability.
"""

import argparse
import hashlib
import math
import statistics
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Literal, TypedDict

from diffusion_jev.accuracy_benchmark import (
    RUNTIME_SOURCES,
    dependency_fingerprints,
    file_fingerprints,
    model_fingerprints,
    revision_identity,
)
from diffusion_jev.benchmark import Environment, environment_metadata
from diffusion_jev.coding_cases import CodingCase, Family, Primitive, Split, coding_cases
from diffusion_jev.coding_metrics import (
    AcceptanceMetrics,
    ScoreMetrics,
    acceptance_metrics,
    acceptance_signature,
    false_acceptance,
    ordinal_losses,
    summarize_scores,
)
from diffusion_jev.component_source import RestorationInfo, inspect_router_source
from diffusion_jev.schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionOptions,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    ScoreQuestion,
    StrictModel,
    Usage,
)

LOG_EPSILON = 1e-12
Decide = Callable[[DecisionRequest], DecisionResponse]
CODING_RUNTIME_SOURCES = RUNTIME_SOURCES + (
    "coding_benchmark.py", "coding_cases.py", "coding_cases_development.py",
    "coding_cases_locked.py", "coding_metrics.py",
)


class CodingRunConfiguration(StrictModel):
    model_revision: str
    revision_evidence: Literal["user-supplied", "snapshot-directory"]
    model_file_sha256: dict[str, str]
    runtime_file_sha256: dict[str, str]
    runtime_dependency_sha256: dict[str, str]
    environment: Environment
    router_precision: Literal["native", "fp32"]
    max_prompt_tokens: int
    cache_limit_bytes: int | None
    reasoning_tokens: Literal[0, 256, 512]
    trajectory_steps: Literal[1, 2] = 1
    restoration_info: RestorationInfo | None = None
    options: DecisionOptions


class CaseResult(StrictModel):
    id: str
    family: Family
    group_id: str
    primitive: Primitive
    correct: bool
    failed: bool = False
    false_acceptance: bool | None = None
    brier: float | None = None
    log_loss: float | None = None
    normalized_absolute_error: float | None = None
    normalized_rps: float | None = None
    wall_ms: float
    usage_observed: bool = False
    decoder_passes: int = 0
    canvas_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_ms: float = 0.0
    reasoning_passes: int = 0
    reasoning_forced_closures: int = 0
    trajectory_accepted_slots: int = 0


class NoulMetrics(StrictModel):
    count: int
    positives: int
    negatives: int
    balanced_accuracy: float
    brier: float
    log_loss: float


class ChoiceMetrics(StrictModel):
    count: int
    accuracy: float
    log_loss: float


class CodingReport(StrictModel):
    schema_version: int = 1
    suite: str = "public-synthetic-coding-v1"
    suite_sha256: str
    split: Split
    options: DecisionOptions
    reasoning_tokens: Literal[0, 256, 512] = 0
    trajectory_steps: Literal[1, 2] = 1
    identity_status: Literal["not-supplied", "fingerprinted"] = "not-supplied"
    configuration: CodingRunConfiguration | None = None
    interpretation: str = (
        "Three correlated judgments per scenario group; synthetic evidence tasks, "
        "not independent real-world tasks or a native JEV comparison. "
        "Usage totals include returned responses only; a request failing before a response "
        "has unknown work, marked usage_observed=false, rather than measured zero work."
    )
    metric_policy: str = (
        "Noul threshold >=0.5, binary Brier, class-balanced accuracy; "
        "natural-log losses clip true-label probability to 1e-12; "
        "Score MAE uses expected zero-based score divided by maximum level; "
        "normalized RPS averages squared cumulative-probability errors over K-1 boundaries; "
        "argmax ties use criterion order. Failed requests or invalid responses remain "
        "in denominators, count incorrect, and receive Brier=1, log-loss=-ln(1e-12), "
        "or normalized ordinal MAE=1 and RPS=1 as applicable. "
        "Failure penalties are conventions, not inferred probability distributions."
    )
    judgments: int
    failures: int
    scenario_groups: int
    family_counts: dict[str, int]
    family_group_counts: dict[str, int]
    primitive_counts: dict[str, int]
    choice_position_counts: dict[str, int]
    score_level_counts: dict[str, int]
    noul: NoulMetrics
    choice: ChoiceMetrics
    score: ScoreMetrics
    total_wall_ms: float
    total_decoder_passes: int = 0
    total_canvas_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_reasoning_ms: float = 0.0
    total_reasoning_passes: int = 0
    total_reasoning_forced_closures: int = 0
    total_trajectory_accepted_slots: int = 0
    cases: list[CaseResult]
    acceptance: AcceptanceMetrics | None = None


def _distribution(probabilities: dict[str, float], labels: list[str]) -> list[float]:
    if set(probabilities) != set(labels):
        raise ValueError("Coding response has invalid probability labels")
    values = [probabilities[label] for label in labels]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError("Coding response has invalid probabilities")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-6):
        raise ValueError("Coding probabilities do not sum to one")
    return values


def evaluate_case(case: CodingCase, response: DecisionResponse, wall_ms: float = 0.0) -> CaseResult:
    if set(response.answers) != {"judgment"}:
        raise ValueError("Coding response has missing or unexpected answers")
    if response.usage.questions != len(case.request.questions):
        raise ValueError("Coding response has incorrect question accounting")
    answer = response.answers["judgment"]
    question = case.request.questions["judgment"]
    correct: bool
    brier = log_loss = error = rps = None
    if case.primitive == "noul":
        if not isinstance(answer, NoulAnswer) or type(case.expected) is not bool:
            raise ValueError("Coding response type does not match Noul ground truth")
        probability = answer.noul
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Coding Noul probability is invalid")
        correct = (probability >= 0.5) == case.expected
        brier = (probability - int(case.expected)) ** 2
        truth_probability = probability if case.expected else 1 - probability
        log_loss = -math.log(max(LOG_EPSILON, truth_probability))
    elif case.primitive == "choice":
        if (
            not isinstance(answer, ChoiceAnswer) or not isinstance(question, ChoiceQuestion)
            or not isinstance(case.expected, str)
        ):
            raise ValueError("Coding response type does not match Choice ground truth")
        labels = list(question.criteria)
        values = _distribution(answer.probabilities, labels)
        predicted = labels[max(range(len(values)), key=values.__getitem__)]
        if answer.choice != predicted:
            raise ValueError("Coding Choice disagrees with its probability distribution")
        correct = predicted == case.expected
        log_loss = -math.log(max(LOG_EPSILON, answer.probabilities[case.expected]))
    else:
        if (
            not isinstance(answer, ScoreAnswer) or not isinstance(question, ScoreQuestion)
            or type(case.expected) is not int
        ):
            raise ValueError("Coding response type does not match Score ground truth")
        values = _distribution(
            answer.probabilities, [str(index) for index in range(len(question.criteria))]
        )
        losses = ordinal_losses(values, case.expected, answer.score)
        correct, error = losses.correct, losses.normalized_absolute_error
        rps = losses.normalized_rps
    return CaseResult(
        id=case.id, family=case.family, group_id=case.group_id, primitive=case.primitive,
        correct=correct, brier=brier, log_loss=log_loss, normalized_absolute_error=error,
        normalized_rps=rps,
        false_acceptance=false_acceptance(case, answer),
        wall_ms=wall_ms,
        usage_observed=True,
        decoder_passes=response.usage.decoder_passes, canvas_tokens=response.usage.canvas_tokens,
        reasoning_tokens=response.usage.reasoning_tokens, reasoning_ms=response.usage.reasoning_ms,
        reasoning_passes=response.usage.reasoning_passes,
        reasoning_forced_closures=response.usage.reasoning_forced_closures,
        trajectory_accepted_slots=response.usage.trajectory_accepted_slots,
    )


def failed_result(case: CodingCase, wall_ms: float, usage: Usage | None = None) -> CaseResult:
    """Keep unusable responses in denominators with explicit worst-scale penalties."""
    return CaseResult(
        id=case.id, family=case.family, group_id=case.group_id, primitive=case.primitive,
        correct=False, failed=True, wall_ms=wall_ms,
        usage_observed=usage is not None,
        brier=1.0 if case.primitive == "noul" else None,
        log_loss=-math.log(LOG_EPSILON) if case.primitive != "score" else None,
        normalized_absolute_error=1.0 if case.primitive == "score" else None,
        normalized_rps=1.0 if case.primitive == "score" else None,
        decoder_passes=usage.decoder_passes if usage is not None else 0,
        canvas_tokens=usage.canvas_tokens if usage is not None else 0,
        reasoning_tokens=usage.reasoning_tokens if usage is not None else 0,
        reasoning_ms=usage.reasoning_ms if usage is not None else 0.0,
        reasoning_passes=usage.reasoning_passes if usage is not None else 0,
        reasoning_forced_closures=usage.reasoning_forced_closures if usage is not None else 0,
        trajectory_accepted_slots=usage.trajectory_accepted_slots if usage is not None else 0,
    )


def summarize(
    cases: tuple[CodingCase, ...], results: list[CaseResult], options: DecisionOptions,
    reasoning_tokens: Literal[0, 256, 512] = 0,
    configuration: CodingRunConfiguration | None = None,
    trajectory_steps: Literal[1, 2] = 1,
) -> CodingReport:
    if configuration is not None and (
        configuration.options != options or configuration.reasoning_tokens != reasoning_tokens
        or configuration.trajectory_steps != trajectory_steps
    ):
        raise ValueError("Coding configuration does not match the evaluated settings")
    if not cases or [case.id for case in cases] != [result.id for result in results]:
        raise ValueError("Coding results must match the ordered case set")
    noul_pairs = [
        (case, result) for case, result in zip(cases, results, strict=True)
        if case.primitive == "noul"
    ]
    positives = [result for case, result in noul_pairs if case.expected is True]
    negatives = [result for case, result in noul_pairs if case.expected is False]
    choices = [result for result in results if result.primitive == "choice"]
    scores = [result for result in results if result.primitive == "score"]
    if not positives or not negatives or not choices or not scores:
        raise ValueError("Coding summary requires both Noul classes and all primitives")
    digest = hashlib.sha256()
    for case in cases:
        for component in (
            case.id, case.request.model_dump_json(), repr(case.expected), case.rationale,
            acceptance_signature(case.acceptance),
        ):
            digest.update(component.encode("utf-8"))
            digest.update(b"\x00")
    return CodingReport(
        suite_sha256=digest.hexdigest(), split=cases[0].split, options=options,
        reasoning_tokens=reasoning_tokens,
        trajectory_steps=trajectory_steps,
        identity_status="not-supplied" if configuration is None else "fingerprinted",
        configuration=configuration,
        judgments=len(cases), scenario_groups=len({case.group_id for case in cases}),
        failures=sum(result.failed for result in results),
        family_counts=dict(Counter(case.family for case in cases)),
        family_group_counts=dict(Counter({case.group_id: case.family for case in cases}.values())),
        primitive_counts=dict(Counter(case.primitive for case in cases)),
        choice_position_counts=dict(Counter(
            str(case.expected) for case in cases if case.primitive == "choice"
        )),
        score_level_counts=dict(Counter(
            str(case.expected) for case in cases if case.primitive == "score"
        )),
        noul=NoulMetrics(
            count=len(noul_pairs), positives=len(positives), negatives=len(negatives),
            balanced_accuracy=(
                statistics.mean(result.correct for result in positives)
                + statistics.mean(result.correct for result in negatives)
            ) / 2,
            brier=statistics.mean(
                result.brier for _, result in noul_pairs if result.brier is not None
            ),
            log_loss=statistics.mean(
                result.log_loss for _, result in noul_pairs if result.log_loss is not None
            ),
        ),
        choice=ChoiceMetrics(
            count=len(choices), accuracy=statistics.mean(result.correct for result in choices),
            log_loss=statistics.mean(
                result.log_loss for result in choices if result.log_loss is not None
            ),
        ),
        score=summarize_scores(scores),
        total_wall_ms=sum(result.wall_ms for result in results), cases=results,
        total_decoder_passes=sum(result.decoder_passes for result in results),
        total_canvas_tokens=sum(result.canvas_tokens for result in results),
        total_reasoning_tokens=sum(result.reasoning_tokens for result in results),
        total_reasoning_ms=sum(result.reasoning_ms for result in results),
        total_reasoning_passes=sum(result.reasoning_passes for result in results),
        total_reasoning_forced_closures=sum(result.reasoning_forced_closures for result in results),
        total_trajectory_accepted_slots=sum(result.trajectory_accepted_slots for result in results),
        acceptance=acceptance_metrics(cases, results),
    )


def run_coding_benchmark(
    decide: Decide, *, split: Split = "development", options: DecisionOptions | None = None,
    reasoning_tokens: Literal[0, 256, 512] = 0,
    configuration: CodingRunConfiguration | None = None,
    trajectory_steps: Literal[1, 2] = 1,
) -> CodingReport:
    selected_options = options if options is not None else DecisionOptions()
    if trajectory_steps == 2 and selected_options.projection != "full":
        raise ValueError("Two trajectory steps require full projection")
    if configuration is not None and (
        configuration.options != selected_options
        or configuration.reasoning_tokens != reasoning_tokens
        or configuration.trajectory_steps != trajectory_steps
    ):
        raise ValueError("Coding configuration does not match the evaluated settings")
    cases = coding_cases(split)
    results: list[CaseResult] = []
    for case in cases:
        request = case.request.model_copy(update={"options": selected_options})
        started = time.perf_counter()
        response: DecisionResponse | None = None
        try:
            response = decide(request)
            result = evaluate_case(case, response, (time.perf_counter() - started) * 1000)
        except (OSError, RuntimeError, ValueError):
            result = failed_result(
                case, (time.perf_counter() - started) * 1000,
                response.usage if response is not None else None,
            )
        results.append(result)
    return summarize(
        cases, results, selected_options, reasoning_tokens, configuration, trajectory_steps
    )


class Arguments(argparse.Namespace):
    model: Path
    model_revision: str | None
    split: Split
    output: Path
    seed: int
    samples: int
    mode: Literal["packed", "independent"]
    projection: Literal["labels", "labels-fp32", "full"]
    canvas_length: int | None
    router_precision: Literal["native", "fp32"]
    max_prompt_tokens: int
    cache_limit_mib: int | None
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
    parser.add_argument(
        "--model-revision",
        help="Immutable 40/64-character hex revision; required outside snapshots",
    )
    parser.add_argument("--split", choices=("development", "locked"), default="development")
    parser.add_argument("--output", type=Path, default=Path("coding-report.local.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--mode", choices=("packed", "independent"), default="packed")
    parser.add_argument("--projection", choices=("labels", "labels-fp32", "full"), default="labels")
    parser.add_argument("--canvas-length", type=int)
    parser.add_argument("--router-precision", choices=("native", "fp32"), default="native")
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--cache-limit-mib", type=int)
    parser.add_argument("--reasoning-tokens", type=int, choices=(0, 256, 512), default=0)
    parser.add_argument("--trajectory-steps", type=int, choices=(1, 2), default=1)
    parser.add_argument("--router-bf16-source", type=Path)
    return parser


def run_configuration(args: Arguments, options: DecisionOptions) -> CodingRunConfiguration:
    if args.trajectory_steps == 2 and options.projection != "full":
        raise ValueError("Two trajectory steps require full projection")
    if args.max_prompt_tokens < 1 or (
        args.cache_limit_mib is not None and args.cache_limit_mib < 0
    ):
        raise ValueError("Invalid prompt or cache limit")
    model = args.model.expanduser()
    revision, evidence = revision_identity(model, args.model_revision)
    restoration_info = (
        inspect_router_source(args.router_bf16_source.expanduser())
        if args.router_bf16_source is not None else None
    )
    return CodingRunConfiguration(
        model_revision=revision, revision_evidence=evidence,
        model_file_sha256=model_fingerprints(model),
        runtime_file_sha256=file_fingerprints(Path(__file__).parent, CODING_RUNTIME_SOURCES),
        runtime_dependency_sha256=dependency_fingerprints(),
        environment=environment_metadata(model), router_precision=args.router_precision,
        max_prompt_tokens=args.max_prompt_tokens,
        cache_limit_bytes=(
            None if args.cache_limit_mib is None else args.cache_limit_mib * 1024 * 1024
        ),
        reasoning_tokens=args.reasoning_tokens, options=options,
        trajectory_steps=args.trajectory_steps,
        restoration_info=restoration_info,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv, namespace=Arguments())
    if args.trajectory_steps == 2 and args.projection != "full":
        parser.error("--trajectory-steps 2 requires --projection full")
    try:
        options = DecisionOptions(
            seed=args.seed, samples=args.samples, mode=args.mode,
            projection=args.projection, canvas_length=args.canvas_length,
        )
        configuration = run_configuration(args, options)
        from diffusion_jev.engine import LocalEngine

        engine_overrides: EngineOverrides = {}
        if args.reasoning_tokens:
            engine_overrides["reasoning_tokens"] = args.reasoning_tokens
        if args.trajectory_steps != 1:
            engine_overrides["trajectory_steps"] = args.trajectory_steps
        if args.router_bf16_source is not None:
            engine_overrides["router_bf16_source"] = args.router_bf16_source.expanduser()
        engine = LocalEngine(
            args.model.expanduser(), args.max_prompt_tokens, router_precision=args.router_precision,
            cache_limit_bytes=configuration.cache_limit_bytes,
            **engine_overrides,
        )
        if engine.restoration_info != configuration.restoration_info:
            raise ValueError("Restored router identity differs from pre-load inspection")
        report = run_coding_benchmark(
            engine.decide, split=args.split, options=options,
            reasoning_tokens=args.reasoning_tokens,
            configuration=configuration,
            trajectory_steps=args.trajectory_steps,
        )
        args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError):
        raise SystemExit(
            "Coding benchmark failed; no request or response payload is reported."
        ) from None
    print(
        f"Coding benchmark complete: {report.judgments - report.failures} successful, "
        f"{report.failures} failed judgments; {report.scenario_groups} groups."
    )
    if report.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
