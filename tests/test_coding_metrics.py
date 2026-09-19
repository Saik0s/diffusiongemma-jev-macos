"""Acceptance semantics must not confuse positive predicates with safe approval."""

from dataclasses import replace

import pytest

from diffusion_jev.coding_benchmark import evaluate_case, failed_result
from diffusion_jev.coding_cases import (
    ChoiceAcceptance,
    CodingCase,
    ScoreAcceptance,
    coding_cases,
    scenario_cases,
    validate_cases,
)
from diffusion_jev.coding_cases_locked import SCENARIOS
from diffusion_jev.coding_metrics import (
    acceptance_metrics,
    acceptance_signature,
    ordinal_losses,
    summarize_scores,
)
from diffusion_jev.schemas import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)


def case_named(name: str, primitive: str) -> CodingCase:
    return next(
        case for case in coding_cases("development") + coding_cases("locked")
        if case.group_id.endswith(f".{name}") and case.primitive == primitive
    )


def response(answer: Answer) -> DecisionResponse:
    return DecisionResponse(
        model="synthetic", answers={"judgment": answer},
        usage=Usage(
            prompt_tokens=0, questions=1, decoder_passes=0, canvas_tokens=0,
            prefill_ms=0.0, decode_ms=0.0, total_ms=0.0, peak_memory_gb=0.0,
        ),
    )


def ordinal(probabilities: tuple[float, float, float]) -> DecisionResponse:
    return response(ScoreAnswer(
        score=sum(index * value for index, value in enumerate(probabilities)),
        legend={"0": "low", "1": "middle", "2": "high"}, confidence=0.0,
        probabilities={str(index): value for index, value in enumerate(probabilities)},
    ))


def choice(case: CodingCase, selected: str) -> DecisionResponse:
    question = case.request.questions["judgment"]
    assert isinstance(question, ChoiceQuestion)
    return response(ChoiceAnswer(
        choice=selected, confidence=1.0,
        probabilities={label: float(label == selected) for label in question.criteria},
    ))


@pytest.mark.parametrize("name", ["partial_platform", "acceptance_timeout"])
def test_true_incomplete_work_predicates_are_not_acceptance_decisions(name: str) -> None:
    case = case_named(name, "noul")
    assert case.expected is True and case.acceptance is None
    result = evaluate_case(case, response(NoulAnswer(noul=1.0)))
    assert result.correct and result.false_acceptance is None
    metrics = acceptance_metrics((case,), [result])
    assert metrics.overall.unmatched == 1
    assert metrics.overall.decisions == metrics.overall.false_acceptances == 0
    assert metrics.overall.false_acceptance_rate is None


def test_every_benign_choice_is_counterbalanced_with_its_acceptance_annotation() -> None:
    scenario = next(scenario for scenario in SCENARIOS if scenario.group_id == "unsafe_cleanup")
    for position in range(3):
        case = next(case for case in scenario_cases(scenario, "locked", position)
                    if case.primitive == "choice")
        assert isinstance(case.acceptance, ChoiceAcceptance)
        question = case.request.questions["judgment"]
        assert isinstance(question, ChoiceQuestion)
        assert {question.criteria[label] for label in case.acceptance.accepting_labels} == {
            "Only temporary rows deleted", "Read-only change",
        }
        for label in case.acceptance.accepting_labels:
            result = evaluate_case(case, choice(case, label))
            assert result.false_acceptance is True and not result.correct


def test_score_acceptance_uses_argmax_not_expected_score_and_obeys_tie_order() -> None:
    completion = case_named("untested_upgrade", "score")
    result = evaluate_case(completion, ordinal((0.4, 0.1, 0.5)))
    assert result.false_acceptance is True  # Expected score 1.1, but level 2 wins.
    patch = case_named("cross_tenant_query", "score")
    result = evaluate_case(patch, ordinal((0.45, 0.1, 0.45)))
    assert result.false_acceptance is True  # First tied level 0 accepts the unsafe patch.
    result = evaluate_case(patch, ordinal((0.2, 0.45, 0.35)))
    assert not result.correct and result.false_acceptance is False


def test_false_acceptance_denominators_separate_failures_and_unmatched_judgments() -> None:
    cases = (
        case_named("untested_upgrade", "choice"), case_named("untested_upgrade", "score"),
        case_named("silent_export", "score"), case_named("verified_limits", "score"),
        case_named("partial_platform", "noul"),
    )
    rule = cases[0].acceptance
    assert isinstance(rule, ChoiceAcceptance)
    results = [
        evaluate_case(cases[0], choice(cases[0], next(iter(rule.accepting_labels)))),
        evaluate_case(cases[1], ordinal((0.0, 1.0, 0.0))), failed_result(cases[2], 1.0),
        evaluate_case(cases[3], ordinal((0.0, 0.0, 1.0))),
        evaluate_case(cases[4], response(NoulAnswer(noul=1.0))),
    ]
    metrics = acceptance_metrics(cases, results)
    assert metrics.overall.decisions == 4 and metrics.overall.unmatched == 1
    assert metrics.overall.reject_required == 3
    assert metrics.overall.valid_reject_required == 2
    assert metrics.overall.false_acceptances == metrics.overall.failed_decisions == 1
    assert metrics.overall.false_acceptance_rate == 0.5
    assert metrics.by_primitive["noul"].decisions == 0
    failed = acceptance_metrics((cases[2],), [results[2]])
    assert failed.overall.reject_required == failed.overall.failed_decisions == 1
    assert failed.overall.false_acceptance_rate is None


def test_suite_has_explicit_acceptance_coverage_and_rejects_contradictory_annotations() -> None:
    assert sum(case.acceptance is not None for case in coding_cases("development")) == 10
    assert sum(case.acceptance is not None for case in coding_cases("locked")) == 40
    cases = coding_cases("development")
    index = next(index for index, case in enumerate(cases) if case.acceptance is not None)
    rule = cases[index].acceptance
    assert rule is not None
    changed = replace(
        cases[index], acceptance=replace(rule, expected_accept=not rule.expected_accept)
    )
    with pytest.raises(ValueError, match="contradicts"):
        validate_cases(cases[:index] + (changed,) + cases[index + 1:], "development")
    assert acceptance_signature(ScoreAcceptance(False, frozenset({0, 2}))) == acceptance_signature(
        ScoreAcceptance(False, frozenset({2, 0}))
    )


def test_migration_evidence_verifies_records_instead_of_only_row_count() -> None:
    case = case_named("migration_roundtrip", "score")
    assert isinstance(case.request.state, dict)
    facts = case.request.state["facts"]
    assert isinstance(facts, str)
    assert "primary-key sets and every column value" in facts
    assert "after upgrade and after the round trip" in facts
    assert "exact original schema" in facts


@pytest.mark.parametrize(("probabilities", "truth", "expected_rps"), [
    ([0.0, 1.0, 0.0], 1, 0.0),
    ([1.0, 0.0, 0.0], 1, 0.5),
    ([0.0, 0.0, 1.0], 0, 1.0),
    ([0.4, 0.2, 0.4], 1, 0.16),
    ([0.8, 0.2], 1, 0.64),
    ([1.0, 0.0, 0.0, 0.0], 3, 1.0),
])
def test_normalized_rps_matches_cumulative_probability_anchors(
    probabilities: list[float], truth: int, expected_rps: float,
) -> None:
    reported_score = sum(index * value for index, value in enumerate(probabilities))
    losses = ordinal_losses(probabilities, truth, reported_score)
    assert losses.normalized_rps == pytest.approx(expected_rps)
    if probabilities == [0.4, 0.2, 0.4]:
        assert losses.normalized_absolute_error == 0.0
        assert not losses.correct


def test_rps_summary_retains_failed_cases_with_explicit_penalty_and_failure_count() -> None:
    case = case_named("untested_upgrade", "score")
    results = [
        evaluate_case(case, ordinal((0.0, 1.0, 0.0))),
        evaluate_case(case, ordinal((1.0, 0.0, 0.0))),
        failed_result(case, 1.0),
    ]
    assert results[2].normalized_rps == 1.0
    metrics = summarize_scores(results)
    assert metrics.count == 3 and metrics.failures == 1
    assert metrics.normalized_rps == 0.5
    assert metrics.argmax_accuracy == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="both MAE and RPS"):
        summarize_scores([results[0].model_copy(update={"normalized_rps": None})])
