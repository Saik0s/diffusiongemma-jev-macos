"""Grouped public synthetic coding judgments, not independent real-world tasks.

Each scenario supplies one Noul, Choice, and Score judgment. Development and
locked splits share task rubrics but no scenario groups. Rationales are evaluation
metadata and never included in model state. Locked cases must not guide tuning.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Literal

from diffusion_jev.schemas import (
    ChoiceQuestion,
    DecisionRequest,
    NoulQuestion,
    Question,
    ScoreQuestion,
)

Split = Literal["development", "locked"]
Family = Literal["triage", "completion", "patch", "file"]
Primitive = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class Scenario:
    group_id: str
    family: Family
    facts: str
    claim: str
    truth: bool
    choices: tuple[str, str, str]
    choice: int
    level: int
    rationale: tuple[str, str, str]
    accepting_choices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ChoiceAcceptance:
    expected_accept: bool
    accepting_labels: frozenset[str]


@dataclass(frozen=True)
class ScoreAcceptance:
    expected_accept: bool
    accepting_levels: frozenset[int]


AcceptanceSemantics = ChoiceAcceptance | ScoreAcceptance


@dataclass(frozen=True)
class CodingCase:
    id: str
    split: Split
    family: Family
    group_id: str
    primitive: Primitive
    request: DecisionRequest
    expected: bool | str | int
    rationale: str
    acceptance: AcceptanceSemantics | None = None


RUBRICS: dict[Family, tuple[str, str, str]] = {
    "triage": (
        "No current evidence of a failure in the target execution.",
        "Current failure is shown, but its cause remains unresolved.",
        "Current failure and direct evidence identifying its cause are provided.",
    ),
    "completion": (
        "A stated acceptance requirement is demonstrably unmet.",
        "No demonstrated unmet requirement, but required current verification is missing.",
        "Every stated acceptance requirement has current passing verification.",
    ),
    "patch": (
        "No violation of the stated behavior or security requirements.",
        "Functional requirement violated without security exposure or data loss.",
        "Security boundary violated or persistent user data can be lost.",
    ),
    "file": (
        "No listed file has evidence of owning the target behavior.",
        "Several files remain plausible owners; available evidence cannot choose one.",
        "Exactly one listed file is supported as owner of the target behavior.",
    ),
}

CHOICE_INSTRUCTIONS: dict[Family, str] = {
    "triage": "Select the diagnosis justified by current evidence; do not invent a cause.",
    "completion": "Select the completion assessment justified by the acceptance evidence.",
    "patch": "Select the effect of the patch under the stated requirements.",
    "file": "Select the file or uncertainty assessment supported by ownership evidence.",
}


def scenario_cases(
    scenario: Scenario, split: Split, target_choice_position: int = 0
) -> tuple[CodingCase, ...]:
    labels = ("a", "b", "c")
    # Rotate labels across scenarios so a fixed-position policy cannot solve the suite.
    offset = (scenario.choice - target_choice_position) % len(labels)
    ordered_choices = scenario.choices[offset:] + scenario.choices[:offset]
    acceptance: dict[Primitive, AcceptanceSemantics] = {}
    if scenario.family in ("completion", "patch"):
        if not scenario.accepting_choices or any(
            index not in range(3) for index in scenario.accepting_choices
        ):
            raise ValueError("Acceptance scenarios require explicit accepting choice indices")
        accepting_level = 2 if scenario.family == "completion" else 0
        expected_accept = scenario.level == accepting_level
        if (scenario.choice in scenario.accepting_choices) != expected_accept:
            raise ValueError("Choice and ordinal acceptance annotations disagree")
        acceptance = {
            "choice": ChoiceAcceptance(
                expected_accept,
                frozenset(labels[(index - offset) % 3] for index in scenario.accepting_choices),
            ),
            "score": ScoreAcceptance(expected_accept, frozenset({accepting_level})),
        }
    questions: tuple[tuple[Primitive, Question, bool | str | int, str], ...] = (
        (
            "noul", NoulQuestion(type="noul", instructions=scenario.claim),
            scenario.truth, scenario.rationale[0],
        ),
        (
            "choice", ChoiceQuestion(
                type="choice", instructions=CHOICE_INSTRUCTIONS[scenario.family],
                criteria={
                    label: description for label, description in zip(
                        labels, ordered_choices, strict=True
                    )
                },
            ), labels[target_choice_position], scenario.rationale[1],
        ),
        (
            "score", ScoreQuestion(
                type="score", instructions="Assign the level justified by the provided facts.",
                criteria=list(RUBRICS[scenario.family]),
            ), scenario.level, scenario.rationale[2],
        ),
    )
    return tuple(
        CodingCase(
            id=f"{split}.{scenario.group_id}.{primitive}", split=split,
            family=scenario.family, group_id=f"{split}.{scenario.group_id}",
            primitive=primitive,
            request=DecisionRequest(
                state={"source": "Public synthetic fixture", "facts": scenario.facts},
                questions={"judgment": question},
            ),
            expected=expected, rationale=rationale,
            acceptance=acceptance.get(primitive),
        )
        for primitive, question, expected, rationale in questions
    )


def coding_cases(split: Split) -> tuple[CodingCase, ...]:
    if split == "development":
        from diffusion_jev.coding_cases_development import SCENARIOS
    elif split == "locked":
        from diffusion_jev.coding_cases_locked import SCENARIOS
    else:
        raise ValueError("Unknown coding split")
    cases = tuple(
        case for index, scenario in enumerate(SCENARIOS)
        for case in scenario_cases(scenario, split, index % 3)
    )
    validate_cases(cases, split)
    return cases


def validate_cases(cases: tuple[CodingCase, ...], split: Split) -> None:
    expected_count = 30 if split == "development" else 120
    if len(cases) != expected_count or len({case.id for case in cases}) != len(cases):
        raise ValueError("Coding suite size or IDs are invalid")
    counts = Counter(case.primitive for case in cases)
    if counts != {primitive: expected_count // 3 for primitive in ("noul", "choice", "score")}:
        raise ValueError("Coding suite primitives must be balanced")
    groups: dict[str, set[Primitive]] = {}
    for case in cases:
        groups.setdefault(case.group_id, set()).add(case.primitive)
        if case.split != split or not case.rationale or set(case.request.questions) != {"judgment"}:
            raise ValueError("Coding case metadata is invalid")
        question = case.request.questions["judgment"]
        if case.primitive != question.type:
            raise ValueError("Coding primitive does not match its question")
        if isinstance(question, NoulQuestion):
            if type(case.expected) is not bool:
                raise ValueError("Noul ground truth must be Boolean")
        elif isinstance(question, ChoiceQuestion):
            if not isinstance(case.expected, str) or case.expected not in question.criteria:
                raise ValueError("Choice ground truth must be an available label")
        elif type(case.expected) is not int or not 0 <= case.expected < len(question.criteria):
            raise ValueError("Score ground truth must be an ordinal index")
        rule = case.acceptance
        applicable = case.family in ("completion", "patch") and case.primitive != "noul"
        if applicable != (rule is not None):
            raise ValueError("Acceptance applicability must be explicit")
        if isinstance(rule, ChoiceAcceptance):
            if not isinstance(question, ChoiceQuestion) or not isinstance(case.expected, str):
                raise ValueError("Choice acceptance requires a Choice judgment")
            if not rule.accepting_labels or not rule.accepting_labels <= question.criteria.keys():
                raise ValueError("Acceptance labels must exist in the Choice criteria")
            if (case.expected in rule.accepting_labels) != rule.expected_accept:
                raise ValueError("Acceptance ground truth contradicts the Choice label")
        elif isinstance(rule, ScoreAcceptance):
            if not isinstance(question, ScoreQuestion) or type(case.expected) is not int:
                raise ValueError("Ordinal acceptance requires a Score judgment")
            if not rule.accepting_levels or not rule.accepting_levels <= set(
                range(len(question.criteria))
            ):
                raise ValueError("Acceptance levels must exist in the Score criteria")
            if (case.expected in rule.accepting_levels) != rule.expected_accept:
                raise ValueError("Acceptance ground truth contradicts the Score level")
    if len(groups) != expected_count // 3 or any(len(types) != 3 for types in groups.values()):
        raise ValueError("Each scenario group must contain all three primitives")
    truth_counts = Counter(case.expected for case in cases if case.primitive == "noul")
    if truth_counts != {True: expected_count // 6, False: expected_count // 6}:
        raise ValueError("Noul ground truth must be class balanced")
    if {case.expected for case in cases if case.primitive == "score"} != {0, 1, 2}:
        raise ValueError("Score fixtures must cover all ordinal levels")
