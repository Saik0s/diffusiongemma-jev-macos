"""Run public teaching workflows through the real local decision endpoint."""

import argparse
from collections.abc import Sequence
from typing import Literal

from diffusion_jev.client import DecisionClient, DecisionClientError
from diffusion_jev.schemas import ChoiceQuestion, ScoreQuestion
from diffusion_jev.workflows import (
    WORKFLOW_NAMES,
    SuggestedAction,
    WorkflowCase,
    WorkflowName,
    apply_policy,
    workflow_cases,
)


def _show_questions(case: WorkflowCase) -> None:
    print("Questions sent to the model:")
    rubrics: dict[tuple[tuple[str, str], ...], str] = {}
    for name, question in case.request.questions.items():
        print(f"  {name} ({question.type}): {question.instructions}")
        if isinstance(question, ChoiceQuestion):
            levels = tuple(question.criteria.items())
        elif isinstance(question, ScoreQuestion):
            levels = tuple((str(index), text) for index, text in enumerate(question.criteria))
        else:
            continue
        if levels in rubrics:
            print(f"    Same answer meanings as {rubrics[levels]} above.")
        else:
            rubrics[levels] = name
            for key, meaning in levels:
                print(f"    {key}: {meaning}")
        if isinstance(question, ScoreQuestion):
            print(f"    Score is a probability-weighted position from 0 to {len(levels) - 1}.")


def _show_outcome(case: WorkflowCase, actions: tuple[SuggestedAction, ...]) -> None:
    expected = {
        label: action
        for (_, label), action in zip(case.items, case.expected_actions, strict=True)
    }
    uncertain = {"review_trajectory", "review_if_needed", "review_evidence"}
    outcomes = []
    for result in actions:
        if result.action == expected[result.label]:
            outcome = "agrees"
        elif result.action in uncertain:
            outcome = "uncertain"
        else:
            outcome = "disagrees"
        outcomes.append(outcome)
        print(f"  {result.label}: {outcome} with expected {expected[result.label]}")
    overall = "disagrees" if "disagrees" in outcomes else (
        "uncertain" if "uncertain" in outcomes else "agrees"
    )
    print(f"Teaching comparison: {overall}. This comparison is not an accuracy benchmark.")


def run(name: WorkflowName | Literal["all"], url: str = "http://127.0.0.1:8017") -> int:
    """Print curated labels and judgments, never arbitrary request/response bodies."""
    names: Sequence[WorkflowName] = WORKFLOW_NAMES if name == "all" else (name,)
    failed = False
    print("Public synthetic teaching fixtures. Actual local model judgments, not Jev benchmarks.")
    print("Thresholds are demonstration policy, not calibrated guarantees. No actions execute.")
    with DecisionClient(url) as client:
        for workflow in names:
            for case in workflow_cases(workflow):
                print(f"\n{case.title}\n{case.purpose}\nInspired by: {case.source}")
                print("Fixture evidence:")
                for evidence in case.evidence:
                    print(f"  {evidence}")
                _show_questions(case)
                print("Expected teaching behavior (not sent to the model):")
                for (_, label), expected in zip(case.items, case.expected_actions, strict=True):
                    print(f"  {label}: {expected}")
                try:
                    response = client.decide(case.request)
                except DecisionClientError:
                    response = None
                    failed = True
                    print("Endpoint unavailable or malformed response; evidence stays available.")
                actions = apply_policy(case, response)
                print("Model judgments and suggested next actions:")
                for result in actions:
                    print(f"  {result.label}: {result.action}")
                    for signal in result.signals:
                        print(f"    {signal}")
                    print(f"    Policy: {result.reason}")
                _show_outcome(case, actions)
                if any(result.action == "review_evidence" for result in actions):
                    failed = True
                if response is not None:
                    print(f"  Model time: {response.usage.total_ms:.0f} ms")
    print("\nCompare judgments with the fixture source; no accuracy claim is made.")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=(*WORKFLOW_NAMES, "all"))
    parser.add_argument("--url", default="http://127.0.0.1:8017")
    args = parser.parse_args(argv)
    if args.name == "all":
        return run("all", args.url)
    for name in WORKFLOW_NAMES:
        if args.name == name:
            return run(name, args.url)
    raise ValueError("Unknown workflow.")


if __name__ == "__main__":
    raise SystemExit(main())
