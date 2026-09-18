import pytest

from diffusion_jev.benchmark import (
    Environment,
    Measurement,
    aggregate,
    compare_probabilities,
    configuration_order,
    nearest_rank,
    run_benchmark,
)
from diffusion_jev.benchmark_cases import quality_cases, scaling_case
from diffusion_jev.schemas import (
    Answer,
    ChoiceAnswer,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)


def usage(questions: int) -> Usage:
    return Usage(
        prompt_tokens=100, questions=questions, decoder_passes=1, canvas_tokens=32,
        prefill_ms=20.0, decode_ms=30.0, total_ms=50.0, peak_memory_gb=14.0,
    )


def response(answers: dict[str, Answer]) -> DecisionResponse:
    return DecisionResponse(
        model="diffusiongemma-local", answers=answers, usage=usage(len(answers))
    )


def test_nearest_rank_and_throughput_use_actual_counts() -> None:
    assert nearest_rank([40.0, 10.0, 30.0, 20.0], 0.5) == 20.0
    assert nearest_rank([40.0, 10.0, 30.0, 20.0], 0.95) == 40.0
    measurements = [
        Measurement(case="small", configuration="optimized", seed=0, wall_ms=100.0,
                    correct=1, questions=1, usage=usage(1)),
        Measurement(case="large", configuration="optimized", seed=0, wall_ms=300.0,
                    correct=3, questions=4, usage=usage(4)),
    ]
    result = aggregate(measurements)
    assert result.questions_per_second == 12.5
    assert result.accuracy == 0.8
    assert result.wall_p50_ms == 100.0
    assert result.wall_p95_ms == 300.0
    with pytest.raises(ValueError):
        nearest_rank([], 0.5)


def test_probability_comparison_detects_delta_and_argmax_changes() -> None:
    left = response({"truth": NoulAnswer(noul=0.501)})
    near = response({"truth": NoulAnswer(noul=0.499)})
    compared = compare_probabilities(left, near)
    assert compared.max_probability_delta == pytest.approx(0.002)
    assert compared.argmax_disagreements == 1
    assert not compared.passed
    assert compare_probabilities(left, response({"truth": NoulAnswer(noul=0.502)})).passed
    assert not compare_probabilities(left, response({"truth": NoulAnswer(noul=0.8)})).passed


def test_synthetic_truth_covers_false_middle_and_score_extremes() -> None:
    cases = quality_cases()[:4]
    assert sum(len(case.expected) for case in cases) == 12
    assert [case.expected[0].value for case in cases] == [True, False, True, False]
    assert [case.expected[1].value for case in cases] == ["green", "blue", "red", "green"]
    assert [case.expected[2].value for case in cases] == [0, 1, 2, 1]
    known = response({
        "enabled": NoulAnswer(noul=0.9),
        "color": ChoiceAnswer(choice="green", probabilities={"green": 0.9, "red": 0.1},
                              confidence=0.5),
        "progress": ScoreAnswer(score=0.2, legend={"0": "low", "1": "mid", "2": "high"},
                                probabilities={"0": 0.85, "1": 0.1, "2": 0.05}, confidence=0.5),
    })
    assert cases[0].correct_count(known) == 3
    assert cases[1].correct_count(known) == 0
    scale = scaling_case(32)
    assert len(scale.request.questions) == 32
    assert sum(truth.value is True for truth in scale.expected) == 16
    assert scale.request.state == {"color": "green"}


def test_coding_truth_includes_triage_review_selection_and_compaction() -> None:
    cases = quality_cases()[4:]
    assert [case.name for case in cases] == [
        "failure_triage", "file_selection", "patch_review", "context_compaction"
    ]
    assert sum(len(case.expected) for case in cases) == 8
    assert {
        case.name: {truth.question_id: truth.value for truth in case.expected}
        for case in cases
    } == {
        "failure_triage": {
            "cause": "dependency", "missing_dependency": True, "network_timeout": False,
        },
        "file_selection": {"file": "validators"},
        "patch_review": {"security_regression": True, "review_priority": 2},
        "context_compaction": {"validation_evidence": True, "unrelated_weather": False},
    }
    for case in quality_cases():
        assert {truth.question_id for truth in case.expected} == set(case.request.questions)


class RecordingEngine:
    def __init__(self) -> None:
        self.requests: list[DecisionRequest] = []

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        answers: dict[str, Answer] = {}
        for key, question in request.questions.items():
            if question.type == "choice":
                labels = list(question.criteria)
                answers[key] = ChoiceAnswer(
                    choice=labels[0], probabilities={label: 1 / len(labels) for label in labels},
                    confidence=0.0,
                )
            elif question.type == "score":
                answers[key] = ScoreAnswer(
                    score=1.0, probabilities={"0": 0.0, "1": 1.0, "2": 0.0},
                    legend={"0": "low", "1": "middle", "2": "high"}, confidence=1.0,
                )
            else:
                answers[key] = NoulAnswer(noul=0.9)
        return response(answers)


def test_harness_excludes_warmups_and_serializes_no_content() -> None:
    engine = RecordingEngine()
    report = run_benchmark(
        engine, repeats=2, load_seconds=2.0,
        environment=Environment(system="test", architecture="test", chip="test",
                                memory_bytes=None, python="3.12", packages={},
                                model_basename="synthetic-model"),
    )
    assert len(engine.requests) == 1 + 32 + 64 + 5 + 10
    assert len(report.measurements) == 74
    assert all(result.questions == 40 for result in report.quality.values())
    assert report.scaling["scaling_32"].questions == 64
    assert report.projection_equivalence.compared_questions == 40
    assert report.projection_equivalence.passed
    serialized = report.model_dump_json()
    for private_field in ('"state"', '"answers"', '"probabilities"', '"instructions"'):
        assert private_field not in serialized
    assert report.first_request.configuration == "baseline_full_256"
    assert configuration_order(0) != configuration_order(1)
    measured_quality = engine.requests[33:97]
    assert [request.options.seed for request in measured_quality] == [0] * 32 + [1] * 32
