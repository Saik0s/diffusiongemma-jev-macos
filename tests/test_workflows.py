"""Policy regression tests; these do not measure model accuracy."""

import pytest

from diffusion_jev.schemas import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)
from diffusion_jev.workflows import (
    WORKFLOW_NAMES,
    WorkflowCase,
    apply_policy,
    valid_answer,
    workflow_cases,
)


def response(answers: dict[str, Answer]) -> DecisionResponse:
    return DecisionResponse(
        model="policy-test",
        answers=answers,
        usage=Usage(
            prompt_tokens=0, questions=max(1, len(answers)), decoder_passes=0,
            canvas_tokens=0, prefill_ms=0.0, decode_ms=0.0, total_ms=0.0, peak_memory_gb=0.0,
        ),
    )


def nouls(values: dict[str, float]) -> DecisionResponse:
    return response({name: NoulAnswer(noul=value) for name, value in values.items()})


def low_value_logs(case: WorkflowCase) -> dict[str, Answer]:
    answers: dict[str, Answer] = {}
    for name, question in case.request.questions.items():
        if isinstance(question, NoulQuestion):
            answers[name] = NoulAnswer(noul=0.1)
        elif isinstance(question, ChoiceQuestion):
            answers[name] = ChoiceAnswer(
                choice="routine", confidence=0.9,
                probabilities={"routine": 0.9, "investigate": 0.05, "unclear": 0.05},
            )
        else:
            answers[name] = ScoreAnswer(
                score=0.2, confidence=0.9, probabilities={"0": 0.9, "1": 0.0, "2": 0.1},
                legend={str(index): text for index, text in enumerate(question.criteria)},
            )
    return answers


@pytest.mark.parametrize("name", WORKFLOW_NAMES)
def test_missing_answers_preserve_every_item_and_original_state(name: str) -> None:
    selected = next(value for value in WORKFLOW_NAMES if value == name)
    for case in workflow_cases(selected):
        before = case.request.model_dump_json()
        for result in (None, response({})):
            decisions = apply_policy(case, result)
            assert len(decisions) == len(case.items)
            assert all(decision.action == "review_evidence" for decision in decisions)
            assert case.request.model_dump_json() == before


def test_search_ranks_without_deleting_low_matches() -> None:
    case = workflow_cases("search")[0]
    values = dict.fromkeys(case.request.questions, 0.1)
    values["refresh_cache"] = 0.9
    values["load_settings"] = 0.7
    decisions = apply_policy(case, nouls(values))
    assert len(decisions) == 6
    assert decisions[0].label.endswith("refresh_cache")
    assert decisions[0].action == "inspect_first"
    assert decisions[-1].action == "rank_lower"


def test_completion_does_not_accept_patch_without_regression_evidence() -> None:
    case = workflow_cases("completion")[0]
    missing = nouls({"implemented": 0.99, "regression": 0.1, "verified": 0.1})
    assert apply_policy(case, missing)[0].action == "request_regression_evidence"
    ready = nouls({"implemented": 0.8, "regression": 0.8, "verified": 0.8})
    assert apply_policy(case, ready)[0].action == "ready_for_review"


def test_progress_requires_agreement_and_never_halts() -> None:
    case = workflow_cases("progress")[0]
    for repeated, progress, expected in (
        (0.9, 0.1, "suggest_replan"),
        (0.1, 0.9, "continue_investigation"),
        (0.9, 0.9, "review_trajectory"),
        (0.5, 0.5, "review_trajectory"),
    ):
        result = nouls({"same_assumption": repeated, "new_evidence": progress})
        assert apply_policy(case, result)[0].action == expected


def test_logs_keep_uncertain_or_actionable_records_in_investigation() -> None:
    case = workflow_cases("logs")[0]
    answers = low_value_logs(case)
    assert all(item.action == "archive_only" for item in apply_policy(case, response(answers)))
    answers["retry_actionable"] = NoulAnswer(noul=0.3)
    answers["checkout_actionable"] = NoulAnswer(noul=0.9)
    actions = {item.label: item.action for item in apply_policy(case, response(answers))}
    assert actions == {
        "Health check": "archive_only",
        "Redis retry": "investigate",
        "Checkout error": "investigate",
    }


def test_invalid_distribution_retains_all_log_evidence() -> None:
    case = workflow_cases("logs")[0]
    answers = low_value_logs(case)
    answers["health_priority"] = ChoiceAnswer(
        choice="routine", confidence=1.0,
        probabilities={"routine": 1.0, "investigate": 1.0, "unclear": 1.0},
    )
    assert all(item.action == "review_evidence" for item in apply_policy(case, response(answers)))


def test_score_must_match_its_distribution_and_rubric() -> None:
    question = ScoreQuestion(
        type="score", instructions="Diagnostic value", criteria=["none", "high"]
    )
    wrong_mean = ScoreAnswer(
        score=0.1, confidence=0.9, probabilities={"0": 0.1, "1": 0.9},
        legend={"0": "none", "1": "high"},
    )
    assert not valid_answer(question, wrong_mean)
    wrong_rubric = ScoreAnswer(
        score=0.9, confidence=0.9, probabilities={"0": 0.1, "1": 0.9},
        legend={"0": "high", "1": "none"},
    )
    assert not valid_answer(question, wrong_rubric)
