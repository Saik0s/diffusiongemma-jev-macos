"""Public, deterministic synthetic truth for quality and scaling measurements."""

from dataclasses import dataclass

from diffusion_jev.examples import coding_fixtures
from diffusion_jev.schemas import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)


@dataclass(frozen=True)
class ExpectedAnswer:
    question_id: str
    value: bool | str | int


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    request: DecisionRequest
    expected: tuple[ExpectedAnswer, ...]

    def correct_count(self, response: DecisionResponse) -> int:
        correct = 0
        for truth in self.expected:
            answer = response.answers.get(truth.question_id)
            if isinstance(truth.value, bool):
                correct += isinstance(answer, NoulAnswer) and (answer.noul >= 0.5) == truth.value
            elif isinstance(truth.value, str):
                correct += isinstance(answer, ChoiceAnswer) and answer.choice == truth.value
            elif isinstance(answer, ScoreAnswer) and answer.probabilities:
                # Score is an expectation, so assess the most probable rubric level.
                selected = max(answer.probabilities, key=answer.probabilities.__getitem__)
                correct += selected == str(truth.value)
        return correct


def quality_cases() -> tuple[BenchmarkCase, ...]:
    """Combine basic label mechanics with synthetic coding decisions."""
    cases: list[BenchmarkCase] = []
    for index, (enabled, color, level) in enumerate(
        ((True, "green", 0), (False, "blue", 1), (True, "red", 2), (False, "green", 1))
    ):
        cases.append(
            BenchmarkCase(
                name=f"mixed_{index + 1}",
                request=DecisionRequest(
                    state={"enabled": enabled, "color": color, "completed_steps": level},
                    questions={
                        "enabled": NoulQuestion(
                            type="noul", instructions="Is enabled true in the supplied state?"
                        ),
                        "color": ChoiceQuestion(
                            type="choice",
                            instructions="Select the exact color in the supplied state.",
                            criteria={"red": "Color is red.", "green": "Color is green.",
                                      "blue": "Color is blue."},
                        ),
                        "progress": ScoreQuestion(
                            type="score",
                            instructions="Rate completed_steps using the exact numeric rubric.",
                            criteria=["Zero steps completed.", "One step completed.",
                                      "Two steps completed."],
                        ),
                    },
                ),
                expected=(ExpectedAnswer("enabled", enabled), ExpectedAnswer("color", color),
                          ExpectedAnswer("progress", level)),
            )
        )
    for fixture in coding_fixtures():
        expected = tuple(
            ExpectedAnswer(key, value) for key, value in fixture.expected_choices.items()
        ) + tuple(
            ExpectedAnswer(key, value) for key, value in fixture.expected_noul.items()
        ) + tuple(
            ExpectedAnswer(key, value) for key, value in fixture.expected_score_levels.items()
        )
        cases.append(BenchmarkCase(fixture.name, fixture.request, expected))
    return tuple(cases)


def scaling_case(count: int) -> BenchmarkCase:
    if not 1 <= count <= 32:
        raise ValueError("Scaling question count must be between 1 and 32")
    questions: dict[str, Question] = {}
    expected: list[ExpectedAnswer] = []
    for index in range(count):
        key = f"predicate_{index}"
        truth = index % 2 == 0
        questions[key] = NoulQuestion(
            type="noul",
            instructions=f"Is the color {'green' if truth else 'red'} in the supplied state?",
        )
        expected.append(ExpectedAnswer(key, truth))
    return BenchmarkCase(
        name=f"scaling_{count}",
        request=DecisionRequest(state={"color": "green"}, questions=questions),
        expected=tuple(expected),
    )
