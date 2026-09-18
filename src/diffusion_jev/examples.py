"""Public synthetic fixtures; never read a real repository or agent history."""

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from time import perf_counter

from diffusion_jev.client import DecisionClient, DecisionClientError
from diffusion_jev.schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
)


@dataclass(frozen=True)
class DecisionFixture:
    name: str
    request: DecisionRequest
    expected_choices: Mapping[str, str] = field(default_factory=dict)
    expected_noul: Mapping[str, bool] = field(default_factory=dict)
    expected_score_levels: Mapping[str, int] = field(default_factory=dict)

    def passed(self, response: DecisionResponse) -> bool:
        for key, expected in self.expected_choices.items():
            answer = response.answers.get(key)
            if not isinstance(answer, ChoiceAnswer) or answer.choice != expected:
                return False
        for key, expected_truth in self.expected_noul.items():
            answer = response.answers.get(key)
            if not isinstance(answer, NoulAnswer) or not math.isfinite(answer.noul):
                return False
            if (answer.noul >= 0.5) != expected_truth:
                return False
        for key, expected_level in self.expected_score_levels.items():
            answer = response.answers.get(key)
            question = self.request.questions.get(key)
            if not isinstance(answer, ScoreAnswer) or not isinstance(question, ScoreQuestion):
                return False
            levels = {str(index) for index in range(len(question.criteria))}
            if set(answer.probabilities) != levels:
                return False
            predicted_level = max(answer.probabilities, key=answer.probabilities.__getitem__)
            if predicted_level != str(expected_level):
                return False
        return True


def build_failure_triage() -> DecisionFixture:
    return DecisionFixture(
        name="failure_triage",
        request=DecisionRequest(
            state={
                "fixture": "synthetic public example",
                "command": "pytest tests/test_settings.py",
                "failure": "ModuleNotFoundError: No module named 'yaml'",
                "context": "PyYAML is declared in project dependencies but not installed.",
            },
            questions={
                "cause": ChoiceQuestion(
                    type="choice",
                    instructions="Choose the direct cause supported by the failure.",
                    criteria={
                        "dependency": "A declared Python dependency is missing.",
                        "assertion": "An assertion compares two different values.",
                        "network": "A remote HTTP request timed out.",
                    },
                ),
                "missing_dependency": NoulQuestion(
                    type="noul",
                    instructions="Does the failure indicate an unavailable Python dependency?",
                ),
                "network_timeout": NoulQuestion(
                    type="noul",
                    instructions="Does the failure indicate a remote network request timed out?",
                ),
            },
        ),
        expected_choices={"cause": "dependency"},
        expected_noul={"missing_dependency": True, "network_timeout": False},
    )


def build_file_selection() -> DecisionFixture:
    return DecisionFixture(
        name="file_selection",
        request=DecisionRequest(
            state={
                "fixture": "synthetic public example",
                "task": "Fix email validation accepting addresses without an @ sign.",
                "files": {
                    "validators": "src/email_validation.py defines validate_email.",
                    "styles": "assets/colors.css defines page colors.",
                    "readme": "README.md explains installation.",
                },
            },
            questions={
                "file": ChoiceQuestion(
                    type="choice",
                    instructions="Select the file most directly responsible for the bug.",
                    criteria={
                        "validators": "src/email_validation.py",
                        "styles": "assets/colors.css",
                        "readme": "README.md",
                    },
                )
            },
        ),
        expected_choices={"file": "validators"},
    )


def build_patch_review() -> DecisionFixture:
    return DecisionFixture(
        name="patch_review",
        request=DecisionRequest(
            state={
                "fixture": "synthetic public example",
                "requirement": "Only authenticated users may access private documents.",
                "before": "if not user.authenticated: return forbidden()",
                "after": "if False: return forbidden()",
                "effect": "Every request now reaches the private document handler.",
            },
            questions={
                "security_regression": NoulQuestion(
                    type="noul",
                    instructions="Does this patch remove the authentication check?",
                ),
                "review_priority": ScoreQuestion(
                    type="score",
                    instructions="Rate the patch's review priority using its observed effect.",
                    criteria=[
                        "Cosmetic change without behavioral impact.",
                        "Nonblocking issue without exposing protected resources.",
                        "Critical security issue exposing protected resources.",
                    ],
                ),
            },
        ),
        expected_noul={"security_regression": True},
        expected_score_levels={"review_priority": 2},
    )


@dataclass(frozen=True)
class ContextUnit:
    """An indivisible message or complete tool call/result pair."""

    id: str
    messages: tuple[str, ...]
    pinned: bool = False


def synthetic_context() -> tuple[ContextUnit, ...]:
    return (
        ContextUnit("goal", ("Fix email validation rejecting valid plus addresses.",), True),
        ContextUnit(
            "validation_evidence",
            (
                '{"role":"tool_call","id":"c1","name":"read_validator"}',
                '{"role":"tool_result","id":"c1","text":"Regex omits +."}',
            ),
        ),
        ContextUnit(
            "unrelated_weather",
            (
                '{"role":"tool_call","id":"c2","name":"weather_example"}',
                '{"role":"tool_result","id":"c2","text":"Sunny, unrelated to coding."}',
            ),
        ),
        ContextUnit("recent_user", ("Keep ordinary email addresses working too.",), True),
        ContextUnit("recent_assistant", ("Next I will inspect validation tests.",), True),
    )


def build_compaction_fixture() -> DecisionFixture:
    units = synthetic_context()
    return DecisionFixture(
        name="context_compaction",
        request=DecisionRequest(
            state={
                "fixture": "synthetic public example",
                "goal": units[0].messages[0],
                "units": [
                    {"id": unit.id, "messages": list(unit.messages)} for unit in units
                ],
            },
            questions={
                unit.id: NoulQuestion(
                    type="noul",
                    instructions=(
                        f"Is context unit {unit.id} useful for the email validation goal? "
                        "Judge whether its contents help complete that goal."
                    ),
                )
                for unit in units
                if not unit.pinned
            },
        ),
        expected_noul={"validation_evidence": True, "unrelated_weather": False},
    )


def compact_context(
    original: tuple[ContextUnit, ...], response: DecisionResponse | None
) -> tuple[ContextUnit, ...]:
    """Fail closed on incomplete decisions; retain exact original unit objects."""
    ids = [unit.id for unit in original]
    eligible = {unit.id for unit in original if not unit.pinned}
    if len(ids) != len(set(ids)) or response is None:
        return original
    if set(response.answers) != eligible:
        return original
    keep: set[str] = set()
    for unit_id, answer in response.answers.items():
        if (
            not isinstance(answer, NoulAnswer)
            or not math.isfinite(answer.noul)
            or not 0.0 <= answer.noul <= 1.0
        ):
            return original
        if answer.noul >= 0.5:
            keep.add(unit_id)
    return tuple(unit for unit in original if unit.pinned or unit.id in keep)


def coding_fixtures() -> tuple[DecisionFixture, ...]:
    return (
        build_failure_triage(),
        build_file_selection(),
        build_patch_review(),
        build_compaction_fixture(),
    )


def show_decisions(fixture: DecisionFixture, response: DecisionResponse) -> None:
    """Explain the fixed public fixture; never use this to log arbitrary requests."""
    for key, question in fixture.request.questions.items():
        print(f"\nQuestion: {question.instructions}")
        answer = response.answers.get(key)
        if isinstance(answer, NoulAnswer):
            print(f"Probability of yes: {answer.noul:.3f}")
        elif isinstance(answer, ChoiceAnswer):
            print(f"Selected: {answer.choice}")
            for option, probability in answer.probabilities.items():
                print(f"  {option}: {probability:.3f}")
        elif isinstance(answer, ScoreAnswer):
            print(f"Weighted score: {answer.score:.3f}")
            for level, probability in answer.probabilities.items():
                meaning = answer.legend.get(level, "missing level description")
                print(f"  {level} ({meaning}): {probability:.3f}")
        else:
            print("No valid answer; inspect the evidence manually.")


def run_example(name: str) -> None:
    parser = argparse.ArgumentParser(description="Run one public synthetic decision example.")
    parser.add_argument("--url", default="http://127.0.0.1:8017")
    args = parser.parse_args()
    fixture = next(fixture for fixture in coding_fixtures() if fixture.name == name)
    original = synthetic_context()
    scenarios = {
        "failure_triage": "Test collection stops because Python cannot import yaml.",
        "file_selection": "An email validator accepts addresses without an @ sign.",
        "patch_review": "A patch replaces the authentication guard with 'if False'.",
        "context_compaction": "An email-validation task has code evidence and a weather result.",
    }
    print(f"Scenario: {scenarios[name]}")
    print("Synthetic preview. Probabilities and thresholds are uncalibrated.")
    started = perf_counter()
    try:
        with DecisionClient(args.url) as client:
            response = client.decide(fixture.request)
    except DecisionClientError:
        if name == "context_compaction":
            retained = compact_context(original, None)
            print(f"Request failed; retained all {len(retained)} original units.")
        else:
            print("Local decision request failed.")
        raise SystemExit(1) from None
    elapsed = perf_counter() - started
    passed = fixture.passed(response)
    show_decisions(fixture, response)
    print(f"\nExpected fixture judgments: {'PASS' if passed else 'FAIL'}; {elapsed:.3f}s")
    if name == "context_compaction":
        retained = compact_context(original, response)
        original_chars = sum(len(message) for unit in original for message in unit.messages)
        retained_chars = sum(len(message) for unit in retained for message in unit.messages)
        print("Policy: keep pinned units; keep complete tool pairs when P(useful) >= 0.5.")
        retained_ids = {unit.id for unit in retained}
        for unit in original:
            action = "KEEP" if unit.id in retained_ids else "OMIT FROM PREVIEW"
            reason = "pinned by code" if unit.pinned else "model relevance judgment"
            print(f"  {action}: {unit.id} ({reason})")
        print(
            f"Preview: {len(retained)}/{len(original)} units; "
            f"{retained_chars}/{original_chars} chars"
        )
        print("The original transcript is unchanged. Retained messages keep their exact text.")
    elif name == "failure_triage":
        answer = response.answers.get("cause")
        if isinstance(answer, ChoiceAnswer):
            actions = {
                "dependency": "Inspect dependency installation and the active Python environment.",
                "assertion": "Inspect the failing assertion and its implementation.",
                "network": "Inspect the failed remote request and connection settings.",
            }
            print(
                "Suggested next step: "
                + actions.get(answer.choice, "Inspect the failure manually.")
            )
    elif name == "file_selection":
        answer = response.answers.get("file")
        question = fixture.request.questions["file"]
        if isinstance(answer, ChoiceAnswer) and isinstance(question, ChoiceQuestion):
            print(
                "Suggested file to read: "
                + question.criteria.get(answer.choice, "manual search")
            )
    elif name == "patch_review":
        answer = response.answers.get("security_regression")
        if isinstance(answer, NoulAnswer) and answer.noul >= 0.5:
            print("Suggested next step: inspect authentication before accepting the patch.")
        else:
            print("No authentication flag from this check; tests and review are still needed.")
    if not passed:
        raise SystemExit(1)
