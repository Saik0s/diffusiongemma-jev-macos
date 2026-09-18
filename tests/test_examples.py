"""Behavioral guarantees for the synthetic compaction preview."""

import pytest

from diffusion_jev.examples import (
    ContextUnit,
    build_patch_review,
    compact_context,
    synthetic_context,
)
from diffusion_jev.schemas import (
    Answer,
    ChoiceAnswer,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)


def decision(answers: dict[str, Answer]) -> DecisionResponse:
    return DecisionResponse(
        model="diffusiongemma-local",
        answers=answers,
        usage=Usage(
            prompt_tokens=0,
            questions=max(1, len(answers)),
            decoder_passes=0,
            canvas_tokens=0,
            prefill_ms=0.0,
            decode_ms=0.0,
            total_ms=0.0,
            peak_memory_gb=0.0,
        ),
    )


def test_compaction_preserves_exact_units_and_call_result_pairs() -> None:
    original = synthetic_context()
    response = decision(
        {
            "validation_evidence": NoulAnswer(noul=0.5),
            "unrelated_weather": NoulAnswer(noul=0.499),
        }
    )
    retained = compact_context(original, response)
    assert [unit.id for unit in retained] == [
        "goal", "validation_evidence", "recent_user", "recent_assistant"
    ]
    assert retained[1] is original[1]
    assert retained[1].messages == original[1].messages
    assert len(retained[1].messages) == 2
    assert all(
        unit is original[index] for unit, index in zip(retained, (0, 1, 3, 4), strict=True)
    )


def test_all_low_scores_still_preserve_goal_and_latest_messages() -> None:
    original = synthetic_context()
    response = decision(
        {
            "validation_evidence": NoulAnswer(noul=0.0),
            "unrelated_weather": NoulAnswer(noul=0.0),
        }
    )
    assert compact_context(original, response) == (original[0], original[3], original[4])


@pytest.mark.parametrize(
    "response",
    [
        None,
        decision({}),
        decision({"validation_evidence": NoulAnswer(noul=0.0)}),
        decision({"unknown": NoulAnswer(noul=0.0)}),
        decision(
            {
                "validation_evidence": NoulAnswer(noul=0.0),
                "unrelated_weather": NoulAnswer(noul=0.0),
                "unknown": NoulAnswer(noul=0.0),
            }
        ),
        decision(
            {
                "validation_evidence": ChoiceAnswer(
                    choice="drop", probabilities={"drop": 1.0}, confidence=1.0
                ),
                "unrelated_weather": NoulAnswer(noul=0.0),
            }
        ),
        decision(
            {
                "validation_evidence": NoulAnswer(noul=0.0),
                "unrelated_weather": NoulAnswer(noul=0.0),
                "goal": NoulAnswer(noul=0.0),
            }
        ),
    ],
)
def test_bad_or_failed_decisions_return_original_unchanged(
    response: DecisionResponse | None,
) -> None:
    original = synthetic_context()
    assert compact_context(original, response) is original


def test_duplicate_unit_ids_fail_closed() -> None:
    original = (ContextUnit("same", ("one",)), ContextUnit("same", ("two",)))
    assert compact_context(original, decision({"same": NoulAnswer(noul=0.0)})) is original


@pytest.mark.parametrize(
    ("probabilities", "expected"),
    [
        ({"0": 0.4, "1": 0.15, "2": 0.45}, True),
        ({"0": 0.1, "1": 0.8, "2": 0.1}, False),
        ({}, False),
        ({"2": 1.0}, False),
        ({"0": 0.0, "1": 0.0, "2": 0.9, "unknown": 0.1}, False),
    ],
)
def test_score_quality_uses_complete_distribution_argmax(
    probabilities: dict[str, float], expected: bool
) -> None:
    response = decision(
        {
            "security_regression": NoulAnswer(noul=1.0),
            "review_priority": ScoreAnswer(
                score=1.05,
                legend={"0": "cosmetic", "1": "nonblocking", "2": "critical"},
                probabilities=probabilities,
                confidence=0.0,
            ),
        }
    )
    assert build_patch_review().passed(response) is expected


def test_missing_score_answer_fails_quality_check() -> None:
    response = decision({"security_regression": NoulAnswer(noul=1.0)})
    assert not build_patch_review().passed(response)
