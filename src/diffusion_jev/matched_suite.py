"""Replay the frozen matched suite locally and compare it with recorded hosted answers.

The suite in `benchmarks/` carries one request per case plus the hosted answer measured
through OpenRouter. This module sends the same state, instructions, criteria, and option
keys to a running local server, then reports per-case agreement, the labelled grid
accuracy, and both latency distributions. Hosted answers are a reference point, not
ground truth, so only cases carrying a derived label count toward accuracy.
"""

import argparse
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import httpx
from pydantic import Field, JsonValue, ValidationError

from diffusion_jev.accuracy_benchmark import atomic_write
from diffusion_jev.benchmark import nearest_rank
from diffusion_jev.client import DecisionClient, DecisionClientError
from diffusion_jev.compat import COMPAT_PATH
from diffusion_jev.schemas import (
    Answer,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    Identifier,
    NoulAnswer,
    Question,
    ScoreAnswer,
    StrictModel,
)


class SuiteRequest(StrictModel):
    state: JsonValue
    questions: Annotated[dict[Identifier, Question], Field(min_length=1, max_length=32)]


class HostedRecord(StrictModel):
    answers: dict[Identifier, Answer]
    usage: dict[str, float]
    latency_s: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class SuiteCase(StrictModel):
    tag: str
    request: SuiteRequest
    expected: dict[Identifier, list[Identifier]] | None
    hosted: HostedRecord


class ValidationCase(StrictModel):
    """A request the local schema rejects, with what the hosted API did with it.

    `hosted_accepted` marks the cases where hosted Jev answered the same payload, so a
    rejection here is an API-parity gap rather than shared strictness.
    """

    tag: str
    request: dict[str, JsonValue]
    expected_status: Annotated[int, Field(ge=400, le=599)]
    hosted_accepted: bool
    hosted: HostedRecord | None = None


class MatchedSuite(StrictModel):
    schema_version: int
    frozen_at: str
    reclassified_at: str | None = None
    reclassification_note: str | None = None
    purpose: str
    content_note: str
    source: dict[str, str]
    hosted: dict[str, str]
    labels: dict[str, str]
    cases: Annotated[list[SuiteCase], Field(min_length=1)]
    validation_cases: list[ValidationCase]


class CaseOutcome(StrictModel):
    """One replayed case. Only its tag, timings, and counts are recorded, never payloads."""

    tag: str
    succeeded: bool
    local_round_trip_s: float | None = None
    local_inference_ms: float | None = None
    peak_memory_gb: float | None = None
    hosted_latency_s: float
    # Largest absolute distance between the local and hosted number for the same
    # question: the noul, the winning option's probability, or the score.
    max_answer_gap: float | None = None
    top_choice_agreements: int = 0
    top_choice_comparisons: int = 0
    labelled_correct: int | None = None
    labelled_questions: int | None = None


class LatencySummary(StrictModel):
    n: int
    median: float
    p90: float


class MatchedReport(StrictModel):
    frozen_at: str
    local_base_url: str
    local_model: str
    hosted_model: str
    cases: int
    failures: int
    local_round_trip_s: LatencySummary | None
    local_inference_ms: LatencySummary | None
    hosted_round_trip_s: LatencySummary
    peak_memory_gb: float | None
    top_choice_agreement: str
    labelled_accuracy: str
    hosted_labelled_accuracy: str
    max_answer_gap: float | None
    validation_parity: str
    # Requests the hosted API answered and the local schema rejects.
    hosted_only_inputs: int
    outcomes: list[CaseOutcome]
    method: str = (
        "Same states, instructions, criteria, and option keys on both sides. Each side "
        "uses its own documented defaults because the suite sends no options block. "
        "Hosted latency was measured separately over a residential network and includes "
        "transport, so it sits beside local round trip and local inference time rather "
        "than being subtracted from them. Hosted answers are an agreement reference, not "
        "ground truth; only the mechanically labelled grid cases count as accuracy. "
        "Retrieval metrics such as nDCG and MRR are undefined here: no case ranks a list. "
        "`hosted_only_inputs` counts payloads the hosted API answered and the local strict "
        "schema rejects; those cannot be compared answer to answer and are excluded."
    )


def leading_key(answer: ChoiceAnswer | ScoreAnswer) -> str:
    """The winning label, using the response's own winner for Choice."""
    if isinstance(answer, ChoiceAnswer):
        return answer.choice
    ranked = sorted(answer.probabilities.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def answer_gap(local: Answer, hosted: Answer) -> float:
    """Absolute distance between two answers of one type, on that type's own scale."""
    if isinstance(local, NoulAnswer) and isinstance(hosted, NoulAnswer):
        return abs(local.noul - hosted.noul)
    if isinstance(local, ChoiceAnswer) and isinstance(hosted, ChoiceAnswer):
        return abs(local.probabilities[local.choice] - hosted.probabilities.get(local.choice, 0.0))
    if isinstance(local, ScoreAnswer) and isinstance(hosted, ScoreAnswer):
        return abs(local.score - hosted.score)
    raise ValueError("Answer types differ between the local and hosted response")


class CaseScore(StrictModel):
    max_gap: float
    agreements: int
    comparisons: int
    correct: int
    labelled: int


def score_case(case: SuiteCase, answers: dict[str, Answer]) -> CaseScore:
    if answers.keys() != case.hosted.answers.keys():
        raise ValueError("Local and hosted responses answer different questions")
    agreements = comparisons = correct = labelled = 0
    for name, local in answers.items():
        hosted = case.hosted.answers[name]
        if isinstance(local, NoulAnswer) or isinstance(hosted, NoulAnswer):
            continue
        comparisons += 1
        agreements += leading_key(local) == leading_key(hosted)
    for name, accepted in (case.expected or {}).items():
        answer = answers[name]
        if not isinstance(answer, ChoiceAnswer | ScoreAnswer):
            raise ValueError("Only Choice and Score answers carry a label in this suite")
        labelled += 1
        correct += leading_key(answer) in accepted
    return CaseScore(
        max_gap=max(answer_gap(answers[name], case.hosted.answers[name]) for name in answers),
        agreements=agreements,
        comparisons=comparisons,
        correct=correct,
        labelled=labelled,
    )


def hosted_labelled_accuracy(suite: MatchedSuite) -> str:
    """Score the recorded hosted answers against the same labels, for a like-for-like row."""
    correct = total = 0
    for case in suite.cases:
        for name, accepted in (case.expected or {}).items():
            answer = case.hosted.answers[name]
            if not isinstance(answer, ChoiceAnswer | ScoreAnswer):
                raise ValueError("Only Choice and Score answers carry a label in this suite")
            total += 1
            correct += leading_key(answer) in accepted
    return f"{correct}/{total}"


def summarize(values: list[float]) -> LatencySummary | None:
    if not values:
        return None
    return LatencySummary(
        n=len(values),
        median=round(statistics.median(values), 4),
        p90=round(nearest_rank(values, 0.9), 4),
    )


def check_validation(cases: list[ValidationCase], base_url: str) -> str:
    """Send the suite's rejected requests to the compat route and count matching statuses."""
    if not cases:
        return "0/0"
    matched = 0
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=30.0) as client:
        for case in cases:
            body = dict(case.request) | {"model": "diffusiongemma-local"}
            try:
                status = client.post(COMPAT_PATH, json=body).status_code
            except httpx.HTTPError:
                continue
            matched += status == case.expected_status
    return f"{matched}/{len(cases)}"


def replay(
    suite: MatchedSuite,
    decide: Callable[[DecisionRequest], DecisionResponse],
    *,
    base_url: str,
    validation_parity: str,
) -> MatchedReport:
    outcomes: list[CaseOutcome] = []
    round_trips: list[float] = []
    inference: list[float] = []
    memories: list[float] = []
    gaps: list[float] = []
    agreements = comparisons = correct = labelled = 0
    model = "unavailable"
    for case in suite.cases:
        request = DecisionRequest(state=case.request.state, questions=case.request.questions)
        started = time.perf_counter()
        try:
            response = decide(request)
        except DecisionClientError:
            outcomes.append(
                CaseOutcome(tag=case.tag, succeeded=False, hosted_latency_s=case.hosted.latency_s)
            )
            continue
        elapsed = time.perf_counter() - started
        model = response.model
        scored = score_case(case, response.answers)
        round_trips.append(elapsed)
        inference.append(response.usage.total_ms)
        memories.append(response.usage.peak_memory_gb)
        gaps.append(scored.max_gap)
        agreements += scored.agreements
        comparisons += scored.comparisons
        correct += scored.correct
        labelled += scored.labelled
        outcomes.append(
            CaseOutcome(
                tag=case.tag,
                succeeded=True,
                local_round_trip_s=round(elapsed, 4),
                local_inference_ms=round(response.usage.total_ms, 1),
                peak_memory_gb=round(response.usage.peak_memory_gb, 2),
                hosted_latency_s=case.hosted.latency_s,
                max_answer_gap=round(scored.max_gap, 4),
                top_choice_agreements=scored.agreements,
                top_choice_comparisons=scored.comparisons,
                labelled_correct=scored.correct if case.expected else None,
                labelled_questions=scored.labelled if case.expected else None,
            )
        )
    hosted_latencies = summarize([case.hosted.latency_s for case in suite.cases])
    if hosted_latencies is None:
        raise ValueError("The suite carries no hosted latencies")
    return MatchedReport(
        frozen_at=suite.frozen_at,
        local_base_url=base_url,
        local_model=model,
        hosted_model=suite.hosted["resolved_model"],
        cases=len(suite.cases),
        failures=sum(not item.succeeded for item in outcomes),
        local_round_trip_s=summarize(round_trips),
        local_inference_ms=summarize(inference),
        hosted_round_trip_s=hosted_latencies,
        peak_memory_gb=round(max(memories), 2) if memories else None,
        top_choice_agreement=f"{agreements}/{comparisons}",
        labelled_accuracy=f"{correct}/{labelled}",
        hosted_labelled_accuracy=hosted_labelled_accuracy(suite),
        max_answer_gap=round(max(gaps), 4) if gaps else None,
        validation_parity=validation_parity,
        hosted_only_inputs=sum(case.hosted_accepted for case in suite.validation_cases),
        outcomes=outcomes,
    )


class Arguments(argparse.Namespace):
    suite: Path
    output: Path
    base_url: str


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Replay the frozen matched suite locally.")
    parser.add_argument(
        "--suite",
        type=Path,
        default=Path("benchmarks/matched-suite-2026-09-19.json"),
        help="Frozen matched suite (default: the committed 2026-09-19 freeze)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Report destination")
    parser.add_argument("--base-url", default="http://127.0.0.1:8017")
    args = parser.parse_args(argv, namespace=Arguments())
    try:
        suite = MatchedSuite.model_validate_json(args.suite.read_text(encoding="utf-8"))
        parity = check_validation(suite.validation_cases, args.base_url)
        with DecisionClient(base_url=args.base_url, timeout=600.0) as client:
            report = replay(
                suite, client.decide, base_url=args.base_url, validation_parity=parity
            )
        atomic_write(args.output, report)
    except (OSError, ValidationError, ValueError):
        parser.exit(
            1, "Matched replay failed; check the suite file, the server, and the output path.\n"
        )
        return
    print(
        f"Replayed {report.cases} cases with {report.failures} failures. Labelled accuracy "
        f"local {report.labelled_accuracy} versus hosted {report.hosted_labelled_accuracy}; "
        f"top-choice agreement {report.top_choice_agreement}; validation parity "
        f"{report.validation_parity}; {report.hosted_only_inputs} hosted-only inputs."
    )


if __name__ == "__main__":
    main()
