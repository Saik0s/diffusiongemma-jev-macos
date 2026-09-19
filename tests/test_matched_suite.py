import json
from pathlib import Path

import pytest

from diffusion_jev.client import DecisionClientError
from diffusion_jev.matched_suite import (
    MatchedSuite,
    answer_gap,
    hosted_labelled_accuracy,
    leading_key,
    replay,
    score_case,
    summarize,
)
from diffusion_jev.schemas import (
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)

SUITE = Path(__file__).resolve().parents[1] / "benchmarks" / "matched-suite-2026-09-19.json"


def usage() -> Usage:
    return Usage(
        prompt_tokens=100, questions=1, decoder_passes=1, canvas_tokens=32,
        prefill_ms=1.0, decode_ms=1.0, total_ms=2.0, peak_memory_gb=3.0,
    )


def load() -> MatchedSuite:
    return MatchedSuite.model_validate_json(SUITE.read_text(encoding="utf-8"))


def test_frozen_suite_matches_the_loader_and_keeps_its_labels_answerable() -> None:
    suite = load()
    assert suite.schema_version == 2
    assert suite.frozen_at == "2026-09-19"
    assert len(suite.cases) == 38
    assert [case.expected_status for case in suite.validation_cases] == [400, 400, 400, 400]
    # Two probes carry a hosted answer for a payload the local schema rejects. They are
    # a parity gap, so they stay out of the comparable cases and keep their evidence.
    hosted_only = [case for case in suite.validation_cases if case.hosted_accepted]
    assert len(hosted_only) == 2
    assert all(case.hosted is not None and case.hosted.answers for case in hosted_only)
    for case in suite.cases:
        # Every case must be replayable against the same questions the hosted side saw.
        assert case.request.questions.keys() == case.hosted.answers.keys()
        for name in case.expected or {}:
            assert name in case.request.questions
    labelled = [case for case in suite.cases if case.expected]
    assert len(labelled) == 20
    # The hosted answers are scored against the same labels, so the table has both sides.
    correct, total = hosted_labelled_accuracy(suite).split("/")
    assert int(total) == len(labelled) and 0 <= int(correct) <= int(total)


def test_frozen_suite_carries_no_account_identifier_or_credential() -> None:
    text = SUITE.read_text(encoding="utf-8")
    for marker in ("user_id", "OPENROUTER", "Bearer ", "sk-or-", "@gmail"):
        assert marker not in text


def test_leading_key_uses_the_response_winner_and_breaks_score_ties_by_label() -> None:
    choice = ChoiceAnswer(choice="down", probabilities={"down": 0.5, "up": 0.5}, confidence=0.0)
    assert leading_key(choice) == "down"
    tied = ScoreAnswer(
        score=0.5, legend={"0": "low", "1": "high"},
        probabilities={"1": 0.5, "0": 0.5}, confidence=0.0,
    )
    assert leading_key(tied) == "0"


def test_answer_gap_compares_like_with_like_and_rejects_mixed_types() -> None:
    assert answer_gap(NoulAnswer(noul=0.9), NoulAnswer(noul=0.75)) == pytest.approx(0.15)
    local = ChoiceAnswer(choice="a", probabilities={"a": 0.7, "b": 0.3}, confidence=0.4)
    hosted = ChoiceAnswer(choice="b", probabilities={"a": 0.4, "b": 0.6}, confidence=0.2)
    # The gap follows the local winner's own probability, not the hosted winner's.
    assert answer_gap(local, hosted) == pytest.approx(0.3)
    with pytest.raises(ValueError):
        answer_gap(local, NoulAnswer(noul=0.5))


def test_score_case_counts_labels_and_agreement_separately() -> None:
    suite = load()
    case = next(case for case in suite.cases if case.expected)
    name, accepted = next(iter((case.expected or {}).items()))
    hosted = case.hosted.answers[name]
    assert isinstance(hosted, ChoiceAnswer)
    wrong = next(
        key for key in hosted.probabilities if key not in accepted and key != hosted.choice
    )
    answers = dict(case.hosted.answers)
    answers[name] = ChoiceAnswer(
        choice=wrong, probabilities=hosted.probabilities, confidence=hosted.confidence
    )
    scored = score_case(case, answers)
    assert scored.labelled == len(case.expected or {})
    assert scored.correct == scored.labelled - 1
    assert scored.agreements < scored.comparisons
    with pytest.raises(ValueError):
        score_case(case, {"unasked": NoulAnswer(noul=0.5)})


def test_replay_records_failures_without_stopping_and_separates_the_two_clocks() -> None:
    suite = load()
    calls: list[DecisionRequest] = []
    # Fail a labelled case, so the labelled denominator drops below the hosted one.
    failing = next(index for index, case in enumerate(suite.cases) if case.expected)

    def decide(request: DecisionRequest) -> DecisionResponse:
        calls.append(request)
        index = len(calls) - 1
        if index == failing:
            raise DecisionClientError("failed")
        return DecisionResponse(
            model="test-local", answers=dict(suite.cases[index].hosted.answers), usage=usage()
        )

    report = replay(suite, decide, base_url="http://test", validation_parity="2/2")
    assert len(calls) == 38
    assert report.failures == 1
    assert report.outcomes[failing].succeeded is False
    assert report.outcomes[failing].local_round_trip_s is None
    # Hosted latency is recorded for the failed case too, so the columns stay comparable.
    assert report.outcomes[failing].hosted_latency_s > 0
    assert report.local_inference_ms is not None and report.local_inference_ms.n == 37
    assert report.local_inference_ms.median == 2.0
    assert report.local_round_trip_s is not None
    assert report.local_round_trip_s.median != report.local_inference_ms.median
    assert report.hosted_round_trip_s.n == 38
    assert report.peak_memory_gb == 3.0
    # Replaying the hosted answers agrees with them everywhere and matches their accuracy.
    assert report.max_answer_gap == 0.0
    agreements, comparisons = report.top_choice_agreement.split("/")
    assert agreements == comparisons and int(comparisons) > 0
    assert report.labelled_accuracy != report.hosted_labelled_accuracy
    assert report.hosted_model == "typesafe/jev-1.13-20260917"
    assert report.validation_parity == "2/2"
    # Both hosted-only inputs are reported, so the parity gap is never silent.
    assert report.hosted_only_inputs == 2


def test_replay_sends_the_frozen_state_and_questions_unchanged() -> None:
    suite = load()
    sent: list[DecisionRequest] = []

    def decide(request: DecisionRequest) -> DecisionResponse:
        sent.append(request)
        index = len(sent) - 1
        return DecisionResponse(
            model="test-local", answers=dict(suite.cases[index].hosted.answers), usage=usage()
        )

    replay(suite, decide, base_url="http://test", validation_parity="0/0")
    for case, request in zip(suite.cases, sent, strict=True):
        assert request.state == case.request.state
        assert request.questions == case.request.questions
        # The suite sends no options, so the server's startup profile picks the read
        # count and each side keeps its own documented defaults.
        assert "samples" not in request.options.model_fields_set
        assert json.loads(request.model_dump_json())["options"] == {
            "seed": 0, "samples": 1, "mode": "packed",
            "projection": "labels", "canvas_length": None,
        }


def test_summarize_reports_the_nearest_rank_and_nothing_for_no_data() -> None:
    assert summarize([]) is None
    assert summarize([0.4, 0.1, 0.2]) == summarize([0.1, 0.2, 0.4])
    computed = summarize([float(i) for i in range(10)])
    assert computed is not None
    assert (computed.n, computed.median, computed.p90) == (10, 4.5, 8.0)
