"""Suite integrity, metric math, and payload-free reporting without inference."""

import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Literal, Unpack

import pytest

from diffusion_jev.coding_benchmark import (
    Arguments,
    CodingReport,
    EngineOverrides,
    build_parser,
    evaluate_case,
    main,
    run_coding_benchmark,
    run_configuration,
    summarize,
)
from diffusion_jev.coding_cases import CodingCase, Split, coding_cases, validate_cases
from diffusion_jev.component_source import RestorationInfo, RestoredTensor
from diffusion_jev.schemas import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionOptions,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    ScoreAnswer,
    Usage,
)


def response(answer: Answer) -> DecisionResponse:
    return DecisionResponse(
        model="diffusiongemma-local", answers={"judgment": answer},
        usage=Usage(
            prompt_tokens=0, questions=1, decoder_passes=0, canvas_tokens=0,
            prefill_ms=0.0, decode_ms=0.0, total_ms=0.0, peak_memory_gb=0.0,
        ),
    )


def score_answer(probabilities: tuple[float, float, float]) -> ScoreAnswer:
    return ScoreAnswer(
        score=sum(index * value for index, value in enumerate(probabilities)),
        legend={"0": "low", "1": "middle", "2": "high"},
        probabilities={str(index): value for index, value in enumerate(probabilities)},
        confidence=0.0,
    )


@pytest.mark.parametrize(("split", "count", "groups"), [
    ("development", 30, 10), ("locked", 120, 40),
])
def test_suite_counts_balance_and_rationale_separation(
    split: Split, count: int, groups: int
) -> None:
    cases = coding_cases(split)
    assert len(cases) == len({case.id for case in cases}) == count
    assert len({case.group_id for case in cases}) == groups
    assert Counter(case.primitive for case in cases) == dict.fromkeys(
        ("noul", "choice", "score"), count // 3
    )
    assert Counter(case.expected for case in cases if case.primitive == "noul") == {
        True: count // 6, False: count // 6,
    }
    assert {case.expected for case in cases if case.primitive == "score"} == {0, 1, 2}
    positions: Counter[int] = Counter()
    for case in cases:
        question = case.request.questions["judgment"]
        if isinstance(question, ChoiceQuestion):
            assert isinstance(case.expected, str)
            positions[list(question.criteria).index(case.expected)] += 1
    assert max(positions.values()) - min(positions.values()) <= 1
    assert len({case.request.model_dump_json(include={"state"}) for case in cases}) == groups
    for case in cases:
        assert case.rationale
        assert isinstance(case.request.state, dict)
        assert set(case.request.state) == {"source", "facts"}
    if split == "locked":
        assert Counter(case.family for case in cases) == {
            "triage": 30, "completion": 30, "patch": 30, "file": 30,
        }


def test_development_and_locked_scenarios_are_disjoint() -> None:
    development, locked = coding_cases("development"), coding_cases("locked")
    assert {case.group_id for case in development}.isdisjoint(case.group_id for case in locked)
    assert {case.request.model_dump_json(include={"state"}) for case in development}.isdisjoint(
        case.request.model_dump_json(include={"state"}) for case in locked
    )


def test_validation_rejects_duplicate_ids_wrong_truth_types_and_class_imbalance() -> None:
    cases = coding_cases("development")
    with pytest.raises(ValueError, match="IDs"):
        validate_cases((cases[0],) + cases[1:-1] + (cases[0],), "development")
    with pytest.raises(ValueError, match="Boolean"):
        validate_cases((replace(cases[0], expected=1),) + cases[1:], "development")
    with pytest.raises(ValueError, match="class balanced"):
        validate_cases((replace(cases[0], expected=False),) + cases[1:], "development")


@pytest.mark.parametrize(("truth", "probability", "correct"), [
    (True, 0.8, True), (False, 0.2, True), (True, 0.2, False), (False, 0.8, False),
])
def test_binary_losses_follow_probability_of_actual_class(
    truth: bool, probability: float, correct: bool
) -> None:
    case = replace(coding_cases("development")[0], expected=truth)
    result = evaluate_case(case, response(NoulAnswer(noul=probability)))
    assert result.correct is correct
    assert result.brier == pytest.approx((probability - int(truth)) ** 2)
    assert result.log_loss == pytest.approx(-math.log(0.8 if correct else 0.2))


def test_zero_true_probability_has_finite_clipped_log_loss() -> None:
    case = replace(coding_cases("development")[0], expected=True)
    result = evaluate_case(case, response(NoulAnswer(noul=0.0)))
    assert result.brier == 1.0
    assert result.log_loss == pytest.approx(-math.log(1e-12))


def test_choice_loss_uses_ground_truth_even_when_prediction_is_wrong() -> None:
    case = next(case for case in coding_cases("development") if case.primitive == "choice")
    question = case.request.questions["judgment"]
    assert isinstance(question, ChoiceQuestion)
    labels = list(question.criteria)
    case = replace(case, expected=labels[1])
    result = evaluate_case(case, response(ChoiceAnswer(
        choice=labels[0], probabilities=dict(zip(labels, (0.7, 0.2, 0.1), strict=True)),
        confidence=0.0,
    )))
    assert not result.correct
    assert result.log_loss == pytest.approx(-math.log(0.2))


def test_ordinal_error_uses_expected_score_not_argmax_or_rounding() -> None:
    case = next(case for case in coding_cases("development") if case.primitive == "score")
    middle = evaluate_case(replace(case, expected=1), response(score_answer((0.4, 0.2, 0.4))))
    assert middle.normalized_absolute_error == 0.0
    assert middle.normalized_rps == pytest.approx(0.16)
    assert not middle.correct
    extreme = evaluate_case(replace(case, expected=2), response(score_answer((0.5, 0.5, 0.0))))
    assert extreme.normalized_absolute_error == 0.75


def test_wrong_answer_type_missing_answers_and_invalid_distribution_are_rejected() -> None:
    case = next(case for case in coding_cases("development") if case.primitive == "score")
    with pytest.raises(ValueError, match="type"):
        evaluate_case(case, response(NoulAnswer(noul=0.5)))
    missing = response(NoulAnswer(noul=0.5)).model_copy(update={"answers": {}})
    with pytest.raises(ValueError, match="missing"):
        evaluate_case(case, missing)
    invalid = score_answer((0.1, 0.1, 0.1))
    with pytest.raises(ValueError, match="sum"):
        evaluate_case(case, response(invalid))
    incoherent = score_answer((0.0, 0.0, 1.0)).model_copy(update={"score": 0.0})
    with pytest.raises(ValueError, match="disagrees"):
        evaluate_case(case, response(incoherent))


def test_balanced_accuracy_weights_classes_equally_in_an_unequal_subset() -> None:
    cases = coding_cases("development")
    noul = cases[0]
    choice = next(case for case in cases if case.primitive == "choice")
    score = next(case for case in cases if case.primitive == "score")
    subset = (
        replace(noul, id="positive-1", expected=True),
        replace(noul, id="positive-2", expected=True),
        replace(noul, id="negative-1", expected=False), choice, score,
    )
    results = [
        evaluate_case(case, response(NoulAnswer(noul=value)))
        for case, value in zip(subset[:3], (0.9, 0.1, 0.1), strict=True)
    ]
    results.extend(evaluate_case(case, perfect_response(case)) for case in subset[3:])
    report = summarize(subset, results, DecisionOptions())
    assert report.noul.balanced_accuracy == 0.75
    assert report.noul.positives == 2
    assert report.noul.negatives == 1


def perfect_response(case: CodingCase) -> DecisionResponse:
    if type(case.expected) is bool:
        return response(NoulAnswer(noul=float(case.expected)))
    if isinstance(case.expected, str):
        question = case.request.questions["judgment"]
        assert isinstance(question, ChoiceQuestion)
        return response(ChoiceAnswer(
            choice=case.expected,
            probabilities={label: float(label == case.expected) for label in question.criteria},
            confidence=1.0,
        ))
    return response(score_answer((
        float(case.expected == 0), float(case.expected == 1), float(case.expected == 2)
    )))


def test_callback_runner_applies_options_and_reports_no_payloads() -> None:
    cases = coding_cases("development")
    by_request = {case.request.model_dump_json(exclude={"options"}): case for case in cases}
    options = DecisionOptions(seed=3, samples=2, projection="full", mode="independent")

    def decide(request: DecisionRequest) -> DecisionResponse:
        assert request.options == options
        return perfect_response(by_request[request.model_dump_json(exclude={"options"})])

    report = run_coding_benchmark(decide, options=options)
    assert report.judgments == 30 and report.scenario_groups == 10
    assert report.noul.balanced_accuracy == report.choice.accuracy == 1.0
    assert report.noul.brier == report.noul.log_loss == report.score.normalized_mae == 0.0
    assert report.score.normalized_rps == 0.0
    assert report.score.failures == 0
    assert len(report.suite_sha256) == 64
    assert report.identity_status == "not-supplied"
    assert report.configuration is None
    assert report.acceptance is not None
    assert report.acceptance.overall.decisions == 10
    assert report.acceptance.overall.false_acceptances == 0
    assert report.acceptance.overall.unmatched == 20
    serialized = report.model_dump_json()
    for forbidden_key in ("state", "answers", "rationale", "probabilities", "facts"):
        assert f'"{forbidden_key}":' not in serialized


def test_failures_remain_in_all_metric_denominators_without_exception_payloads() -> None:
    def failing_decide(request: DecisionRequest) -> DecisionResponse:
        raise RuntimeError("Payload that must never enter the report")

    report = run_coding_benchmark(failing_decide)
    assert report.failures == report.judgments == len(report.cases) == 30
    assert report.noul.count == report.choice.count == report.score.count == 10
    assert report.noul.balanced_accuracy == report.choice.accuracy == 0.0
    assert report.noul.brier == report.score.normalized_mae == 1.0
    assert report.score.normalized_rps == 1.0
    assert report.score.failures == 10
    assert report.noul.log_loss == report.choice.log_loss == pytest.approx(-math.log(1e-12))
    assert all(result.failed for result in report.cases)
    assert all(not result.usage_observed for result in report.cases)
    assert report.acceptance is not None
    assert report.acceptance.overall.failed_decisions == 10
    assert report.acceptance.overall.valid_reject_required == 0
    assert report.acceptance.overall.false_acceptance_rate is None
    assert "Payload that must" not in report.model_dump_json()


def test_incorrect_question_accounting_fails_without_discarding_observed_work() -> None:
    cases = coding_cases("development")
    by_request = {case.request.model_dump_json(exclude={"options"}): case for case in cases}

    def decide(request: DecisionRequest) -> DecisionResponse:
        result = perfect_response(by_request[request.model_dump_json(exclude={"options"})])
        result.usage.questions = 2
        result.usage.decoder_passes = 3
        return result

    with pytest.raises(ValueError, match="question accounting"):
        evaluate_case(cases[0], decide(cases[0].request))
    report = run_coding_benchmark(decide)
    assert report.failures == report.judgments == 30
    assert report.noul.count == report.choice.count == report.score.count == 10
    assert report.noul.balanced_accuracy == report.choice.accuracy == 0.0
    assert report.noul.brier == report.score.normalized_mae == report.score.normalized_rps == 1.0
    assert report.total_decoder_passes == 90
    assert all(result.failed and result.usage_observed for result in report.cases)
    assert report.acceptance is not None
    assert report.acceptance.overall.valid_reject_required == 0
    assert report.acceptance.overall.false_acceptance_rate is None


@pytest.mark.parametrize("invalid_response", [False, True])
def test_coding_reasoning_totals_include_invalid_response_work(invalid_response: bool) -> None:
    cases = coding_cases("development")
    by_request = {case.request.model_dump_json(exclude={"options"}): case for case in cases}

    def decide(request: DecisionRequest) -> DecisionResponse:
        result = perfect_response(by_request[request.model_dump_json(exclude={"options"})])
        result.usage.decoder_passes = 5
        result.usage.canvas_tokens = 1024
        result.usage.reasoning_tokens = 200
        result.usage.reasoning_ms = 40.0
        result.usage.reasoning_passes = 4
        result.usage.reasoning_forced_closures = 1
        result.usage.trajectory_accepted_slots = 3
        if invalid_response:
            result.answers.clear()
        return result

    report = run_coding_benchmark(
        decide, reasoning_tokens=256, trajectory_steps=2, options=DecisionOptions(projection="full")
    )
    assert report.reasoning_tokens == 256
    assert report.total_decoder_passes == 150
    assert report.total_canvas_tokens == 30720
    assert report.total_reasoning_tokens == 6000
    assert report.total_reasoning_ms == 1200.0
    assert report.total_reasoning_passes == 120
    assert report.total_reasoning_forced_closures == 30
    assert report.trajectory_steps == 2
    assert report.total_trajectory_accepted_slots == 90
    assert report.failures == (30 if invalid_response else 0)
    assert all(item.reasoning_tokens == 200 for item in report.cases)
    assert all(item.usage_observed for item in report.cases)


@pytest.mark.parametrize("budget", [0, 256])
@pytest.mark.parametrize("steps", [1, 2])
@pytest.mark.parametrize("fail_requests", [False, True])
@pytest.mark.parametrize("restoration_mode", ["none", "match", "mismatch"])
def test_cli_passes_opt_in_budget_to_constructor_and_report_without_loading_model(
    budget: Literal[0, 256], steps: Literal[1, 2], fail_requests: bool,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str], restoration_mode: str,
) -> None:
    seen: list[EngineOverrides] = []
    cases = coding_cases("development")
    by_request = {case.request.model_dump_json(exclude={"options"}): case for case in cases}
    source_path = tmp_path / "router-source"
    source_info = synthetic_restoration_info()
    inspected: list[Path] = []

    def inspect(source: Path) -> RestorationInfo:
        inspected.append(source)
        assert not seen
        return source_info

    monkeypatch.setattr("diffusion_jev.coding_benchmark.inspect_router_source", inspect)

    class FakeEngine:
        def __init__(
            self, model: Path, max_prompt_tokens: int, *,
            router_precision: Literal["native", "fp32"], cache_limit_bytes: int | None,
            **overrides: Unpack[EngineOverrides],
        ) -> None:
            assert inspected == ([] if restoration_mode == "none" else [source_path])
            seen.append(overrides)
            self.restoration_info = source_info if restoration_mode != "none" else None
            if restoration_mode == "mismatch":
                self.restoration_info = source_info.model_copy(update={"config_sha256": "changed"})

        def decide(self, request: DecisionRequest) -> DecisionResponse:
            assert restoration_mode != "mismatch", "Mismatched restoration must stop before queries"
            if fail_requests:
                raise RuntimeError("Synthetic failure payload")
            return perfect_response(by_request[request.model_dump_json(exclude={"options"})])

    engine_module = ModuleType("diffusion_jev.engine")
    monkeypatch.setattr(engine_module, "LocalEngine", FakeEngine, raising=False)
    monkeypatch.setitem(sys.modules, "diffusion_jev.engine", engine_module)
    model = synthetic_model(tmp_path)
    monkeypatch.setattr(
        "diffusion_jev.coding_benchmark.dependency_fingerprints", lambda: {"dependency": "hash"}
    )
    output = tmp_path / "coding.json"
    argv = ["--model", str(model), "--model-revision", "a" * 40, "--output", str(output),
            "--reasoning-tokens", str(budget), "--trajectory-steps", str(steps),
            "--projection", "full"]
    if restoration_mode != "none":
        argv.extend(["--router-bf16-source", str(source_path)])
    if restoration_mode == "mismatch":
        with pytest.raises(SystemExit):
            main(argv)
        assert not output.exists()
        assert seen[0]["router_bf16_source"] == source_path
        return
    if fail_requests:
        with pytest.raises(SystemExit) as exit_info:
            main(argv)
        assert exit_info.value.code == 1
        assert "0 successful, 30 failed judgments" in capsys.readouterr().out
    else:
        main(argv)
        assert "30 successful, 0 failed judgments" in capsys.readouterr().out
    expected_overrides: EngineOverrides = {}
    if budget:
        expected_overrides["reasoning_tokens"] = budget
    if steps != 1:
        expected_overrides["trajectory_steps"] = steps
    if restoration_mode != "none":
        expected_overrides["router_bf16_source"] = source_path
    assert seen == [expected_overrides]
    report = CodingReport.model_validate_json(output.read_text())
    assert report.reasoning_tokens == budget
    assert report.trajectory_steps == steps
    assert report.identity_status == "fingerprinted"
    assert report.configuration is not None
    assert report.configuration.model_revision == "a" * 40
    assert report.configuration.router_precision == "native"
    assert report.configuration.trajectory_steps == steps
    assert report.configuration.restoration_info == (
        source_info if restoration_mode == "match" else None
    )
    assert "model.safetensors" in report.configuration.model_file_sha256
    assert "coding_benchmark.py" in report.configuration.runtime_file_sha256
    assert "trajectory.py" in report.configuration.runtime_file_sha256
    assert "components.py" in report.configuration.runtime_file_sha256
    assert "component_source.py" in report.configuration.runtime_file_sha256
    assert report.configuration.runtime_dependency_sha256 == {"dependency": "hash"}
    assert report.failures == (30 if fail_requests else 0)
    assert "Synthetic failure payload" not in output.read_text()


def test_coding_cli_reasoning_defaults_to_off_and_rejects_other_budgets() -> None:
    parser = build_parser()
    assert parser.parse_args(["--model", "unused"], namespace=Arguments()).reasoning_tokens == 0
    assert parser.parse_args(["--model", "unused"], namespace=Arguments()).trajectory_steps == 1
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "unused", "--reasoning-tokens", "255"], namespace=Arguments())


def test_two_step_projection_constraint_is_rejected_before_identity_or_callback() -> None:
    with pytest.raises(SystemExit) as invalid_cli:
        main(["--model", "missing-model", "--trajectory-steps", "2"])
    assert invalid_cli.value.code == 2

    def unused_decide(request: DecisionRequest) -> DecisionResponse:
        raise AssertionError("Invalid trajectory options must fail before inference")

    with pytest.raises(ValueError, match="full projection"):
        run_coding_benchmark(unused_decide, trajectory_steps=2)


def synthetic_model(root: Path) -> Path:
    model = root / "synthetic-model"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"synthetic"}')
    (model / "model.safetensors").write_bytes(b"synthetic weights, never loaded")
    return model


def synthetic_restoration_info() -> RestorationInfo:
    return RestorationInfo(
        source_repo="public/synthetic", source_revision="a" * 40,
        config_sha256="b" * 64, index_sha256="c" * 64,
        projection_count=1, payload_bytes=4,
        tensors=(RestoredTensor(
            name="synthetic.router", shard="model.safetensors", sha256="d" * 64, size_bytes=4,
        ),),
    )


def test_coding_identity_distinguishes_weights_revision_router_and_runtime_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = synthetic_model(tmp_path)
    monkeypatch.setattr(
        "diffusion_jev.coding_benchmark.dependency_fingerprints", lambda: {"dependency": "hash"}
    )
    parser = build_parser()
    argv = ["--model", str(model), "--model-revision", "b" * 40,
            "--max-prompt-tokens", "2048", "--cache-limit-mib", "64"]
    args = parser.parse_args(argv, namespace=Arguments())
    options = DecisionOptions(samples=2, projection="labels-fp32")
    original = run_configuration(args, options)
    assert original.max_prompt_tokens == 2048
    assert original.cache_limit_bytes == 64 * 1024 * 1024
    assert original.options == options
    args.router_precision = "fp32"
    router_changed = run_configuration(args, options)
    assert router_changed != original
    assert router_changed.model_file_sha256 == original.model_file_sha256
    assert router_changed.router_precision == "fp32"
    args.model_revision = "c" * 40
    assert run_configuration(args, options).model_revision != original.model_revision
    (model / "model.safetensors").write_bytes(b"different weights")
    assert run_configuration(args, options).model_file_sha256 != original.model_file_sha256
    args.model_revision = None
    with pytest.raises(ValueError, match="model-revision"):
        run_configuration(args, options)


def test_configuration_mismatch_is_rejected_before_callback_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = synthetic_model(tmp_path)
    monkeypatch.setattr("diffusion_jev.coding_benchmark.dependency_fingerprints", lambda: {})
    args = build_parser().parse_args(
        ["--model", str(model), "--model-revision", "d" * 40], namespace=Arguments()
    )
    config = run_configuration(args, DecisionOptions(samples=2))

    def unused_decide(request: DecisionRequest) -> DecisionResponse:
        raise AssertionError("Configuration mismatch must be rejected before inference")

    with pytest.raises(ValueError, match="configuration"):
        run_coding_benchmark(unused_decide, configuration=config)
