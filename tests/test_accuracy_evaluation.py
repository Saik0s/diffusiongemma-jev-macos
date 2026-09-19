import hashlib
from importlib.metadata import PathDistribution
from pathlib import Path

import pytest

from diffusion_jev.accuracy_benchmark import (
    RUNTIME_SOURCES,
    Arguments,
    Checkpoint,
    RunConfiguration,
    atomic_write,
    build_parser,
    dependency_fingerprints,
    file_fingerprints,
    main,
    model_fingerprints,
    restore_checkpoint,
    revision_identity,
)
from diffusion_jev.accuracy_evaluation import compare_reports, exact_mcnemar
from diffusion_jev.benchmark import Environment, environment_metadata
from diffusion_jev.component_source import RestorationInfo, RestoredTensor
from diffusion_jev.model_manifest import CHECKPOINTS
from diffusion_jev.retrieval_benchmark import (
    DiagnosticSummary,
    Ranking,
    Report,
    merge_diagnostics,
    run_retrieval,
)
from diffusion_jev.retrieval_data import (
    ArchivedPrediction,
    Candidate,
    EvaluationSplit,
    Query,
    RetrievalData,
    select_split,
)
from diffusion_jev.schemas import (
    DecisionOptions,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    Usage,
)


def synthetic_data(count: int) -> RetrievalData:
    queries = [
        Query(qid=f"q{index}", query="synthetic public query", relevant={"d0": 1},
              n_rel_top30=1, present=[Candidate(did=f"d{i}", bm25=float(30 - i))
                                    for i in range(30)])
        for index in range(count)
    ]
    return RetrievalData(
        queries, {f"d{i}": "synthetic public function" for i in range(30)},
        {query.qid: ArchivedPrediction(
            qid=query.qid, variant="present", model="jev-noul-batch", ok=True,
            dids=[candidate.did for candidate in query.present], scores=[1.0] + [0.0] * 29,
        ) for query in queries},
    )


def decide(request: DecisionRequest) -> DecisionResponse:
    return DecisionResponse(
        model="test", answers={key: NoulAnswer(noul=0.5) for key in request.questions},
        usage=Usage(prompt_tokens=10, questions=len(request.questions), decoder_passes=1,
                    canvas_tokens=32, prefill_ms=1.0, decode_ms=1.0, total_ms=2.0,
                    peak_memory_gb=1.0),
    )


def scored_report(scores: list[float | None]) -> Report:
    report = run_retrieval(synthetic_data(len(scores)), decide)
    for item, score in zip(report.measurements, scores, strict=True):
        item.success = score is not None
        item.local = (Ranking(top1=score, ndcg10=0.5 + score / 2, mrr10=0.5 + score / 2)
                      if score is not None else None)
    return report


def test_frozen_splits_are_disjoint_with_exact_hash_order_and_historical_exclusion() -> None:
    data = synthetic_data(300)
    cohorts: dict[EvaluationSplit, list[str]] = {}
    sizes: dict[EvaluationSplit, int] = {
        "historical": 50, "development": 40, "validation": 60, "test": 150,
    }
    for name, count in sizes.items():
        selected = select_split(data, name)
        cohorts[name] = [query.qid for query in selected.queries]
        assert len(cohorts[name]) == count
    assert len({qid for ids in cohorts.values() for qid in ids}) == 300
    assert cohorts["historical"] == [f"q{i}" for i in range(50)]
    assert cohorts["development"][:6] == ["q90", "q165", "q182", "q131", "q136", "q66"]
    assert cohorts["validation"][:6] == ["q148", "q118", "q189", "q224", "q79", "q193"]
    assert cohorts["test"][:6] == ["q212", "q58", "q65", "q167", "q258", "q270"]
    shuffled = RetrievalData(data.queries[:50] + data.queries[:49:-1],
                             data.documents, data.archived)
    assert (
        select_split(shuffled, "development").queries == select_split(data, "development").queries
    )
    with pytest.raises(ValueError, match="300 unique"):
        select_split(synthetic_data(299), "test")
    duplicate = RetrievalData(data.queries[:299] + [data.queries[0]], data.documents, data.archived)
    with pytest.raises(ValueError, match="300 unique"):
        select_split(duplicate, "development")


def test_exact_mcnemar_matches_known_binomial_tails() -> None:
    assert exact_mcnemar(0, 0) == 1.0
    assert exact_mcnemar(2, 2) == 1.0
    assert exact_mcnemar(4, 0) == 0.125
    assert exact_mcnemar(0, 10) == 0.001953125
    assert exact_mcnemar(1, 5) == pytest.approx(0.21875)
    with pytest.raises(ValueError):
        exact_mcnemar(-1, 2)


def test_paired_bootstrap_known_improvement_and_determinism() -> None:
    baseline, candidate = scored_report([0.0] * 8), scored_report([1.0] * 8)
    result = compare_reports(baseline, candidate, bootstrap_repeats=200, seed=11)
    assert result.candidate_wins == 8
    assert result.candidate_losses == 0
    assert result.top1.mean == 1.0
    assert result.top1.bootstrap95 == [1.0, 1.0]
    assert result.ndcg10.bootstrap95 == [0.5, 0.5]
    assert result.exact_mcnemar_two_sided_p == 0.0078125
    mixed = scored_report([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    a = compare_reports(mixed, candidate, bootstrap_repeats=200, seed=37)
    b = compare_reports(mixed, candidate, bootstrap_repeats=200, seed=37)
    assert a == b
    assert a.top1.bootstrap95[0] < a.top1.mean < a.top1.bootstrap95[1]


def test_paired_failures_count_as_wrong_and_query_identity_is_checked() -> None:
    baseline, candidate = scored_report([1.0, None]), scored_report([None, 1.0])
    result = compare_reports(baseline, candidate, bootstrap_repeats=50)
    assert result.queries == 2
    assert result.candidate_wins == result.candidate_losses == 1
    assert result.baseline_failures == result.candidate_failures == result.newly_failed == 1
    assert result.top1.mean == 0.0
    candidate.measurements.reverse()
    assert compare_reports(baseline, candidate, bootstrap_repeats=50) == result
    changed_source = candidate.model_copy(update={"source_sha256": {"other": "different"}})
    with pytest.raises(ValueError, match="source identities"):
        compare_reports(baseline, changed_source)
    candidate.measurements[0].qid = "other"
    with pytest.raises(ValueError, match="query ID sets"):
        compare_reports(baseline, candidate)


def configuration() -> RunConfiguration:
    return RunConfiguration(
        label="unit-test", split="development", query_ids=["q0", "q1"],
        source_sha256={"candidates.jsonl": "abc"}, model_revision="1" * 40,
        revision_evidence="user-supplied", model_file_sha256={"config.json": "config-hash"},
        runtime_file_sha256={"engine.py": "engine-hash"},
        environment=Environment(system="test", architecture="test", chip="test",
                                memory_bytes=None, python="3.12", packages={},
                                model_basename="synthetic-model"),
        runtime_policy="baseline", options=DecisionOptions(), group_size=5,
        max_prompt_tokens=8192, cache_limit_bytes=None,
    )


def test_checkpoint_rejects_changed_configuration_and_bad_query_prefix(tmp_path: Path) -> None:
    path = tmp_path / "run.checkpoint.json"
    config = configuration()
    measurements = scored_report([None, 1.0]).measurements
    checkpoint = Checkpoint(configuration=config, measurements=measurements[:1])
    atomic_write(path, checkpoint)
    restored = restore_checkpoint(path, config)
    assert restored == measurements[:1]
    assert not restored[0].success
    assert not path.with_name(path.name + ".tmp").exists()
    changed = config.model_copy(update={"options": DecisionOptions(samples=4)})
    with pytest.raises(ValueError, match="differ"):
        restore_checkpoint(path, changed)
    checkpoint.measurements = [measurements[1]]
    atomic_write(path, checkpoint)
    with pytest.raises(ValueError, match="exact cohort prefix"):
        restore_checkpoint(path, config)
    content = path.read_text()
    assert "synthetic public query" not in content
    assert "synthetic public function" not in content
    assert '"answers"' not in content
    assert '"probabilities"' not in content


def test_revision_identity_requires_an_immutable_digest(tmp_path: Path) -> None:
    assert revision_identity(tmp_path, "a" * 40) == ("a" * 40, "user-supplied")
    snapshot = tmp_path / "snapshots" / ("b" * 40)
    assert revision_identity(snapshot, None) == ("b" * 40, "snapshot-directory")
    with pytest.raises(ValueError, match="immutable"):
        revision_identity(tmp_path, "main")
    with pytest.raises(ValueError, match="Supply"):
        revision_identity(tmp_path, None)


def test_diagnostic_mean_is_weighted_by_slots_not_request_count() -> None:
    a = DiagnosticSummary(slots=5, allowed_label_mass_min=0.1,
                          allowed_label_mass_mean=0.2, invalid_argmax_count=3)
    b = DiagnosticSummary(slots=1, allowed_label_mass_min=0.8,
                          allowed_label_mass_mean=0.8, invalid_argmax_count=0)
    merged = merge_diagnostics(a, b)
    assert merged.allowed_label_mass_mean == pytest.approx(0.3)
    assert merged.allowed_label_mass_min == 0.1
    assert merged.slots == 6
    assert merged.invalid_argmax_count == 3


@pytest.mark.parametrize("changed_name", [
    "model-00001-of-00002.safetensors", "extra-weights.safetensors",
    "optiq/optiq_vision.safetensors", "optiq_vision.safetensors",
])
def test_resume_rejects_changed_weight_shards_with_unchanged_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_name: str
) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"synthetic"}')
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":24},"weight_map":'
        '{"a":"model-00001-of-00002.safetensors","b":"extra-weights.safetensors"}}'
    )
    (tmp_path / "optiq").mkdir()
    names = ("model-00001-of-00002.safetensors", "extra-weights.safetensors",
             "optiq/optiq_vision.safetensors", "optiq_vision.safetensors")
    for name in names:
        (tmp_path / name).write_bytes(b"weights1")

    def reject_whole_file_read(path: Path) -> bytes:
        raise AssertionError("Weight fingerprinting must not read entire shards into memory")

    monkeypatch.setattr(Path, "read_bytes", reject_whole_file_read)
    original = model_fingerprints(tmp_path)
    for name in names:
        assert original[name] == hashlib.sha256(b"weights1").hexdigest()
    config = configuration().model_copy(update={"model_file_sha256": original})
    checkpoint = tmp_path / "checkpoint.json"
    atomic_write(checkpoint, Checkpoint(configuration=config, measurements=[]))
    assert restore_checkpoint(checkpoint, config) == []
    (tmp_path / changed_name).write_bytes(b"weights2")
    changed = model_fingerprints(tmp_path)
    assert changed["config.json"] == original["config.json"]
    assert changed["model.safetensors.index.json"] == original["model.safetensors.index.json"]
    assert changed[changed_name] != original[changed_name]
    assert all(not Path(name).is_absolute() for name in changed)
    with pytest.raises(ValueError, match="differ"):
        restore_checkpoint(checkpoint, config.model_copy(update={"model_file_sha256": changed}))


def test_weight_index_rejects_missing_or_nonrelative_shards(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text('{"weight_map":{"a":"missing.safetensors"}}')
    with pytest.raises(ValueError, match="missing shard"):
        model_fingerprints(tmp_path)
    index.write_text('{"weight_map":{"a":"../outside.safetensors"}}')
    with pytest.raises(ValueError, match="relative"):
        model_fingerprints(tmp_path)


@pytest.mark.parametrize("package", ["mlx-lm", "transformers", "tokenizers"])
def test_changed_dependency_version_rejects_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, package: str,
) -> None:
    dependency_version = "1.0"

    def installed_version(name: str) -> str:
        return dependency_version if name == package else "unchanged"

    monkeypatch.setattr("diffusion_jev.benchmark.version", installed_version)
    monkeypatch.setattr("diffusion_jev.benchmark.platform.system", lambda: "test")
    original = environment_metadata(tmp_path / "model")
    assert original.packages[package] == "1.0"
    config = configuration().model_copy(update={"environment": original})
    checkpoint = tmp_path / "checkpoint.json"
    atomic_write(checkpoint, Checkpoint(configuration=config, measurements=[]))
    dependency_version = "2.0"
    changed = environment_metadata(tmp_path / "model")
    assert changed.packages[package] == "2.0"
    with pytest.raises(ValueError, match="differ"):
        restore_checkpoint(checkpoint, config.model_copy(update={"environment": changed}))


def test_environment_does_not_mislabel_higher_precision_as_optiq(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("diffusion_jev.benchmark.platform.system", lambda: "test")
    for profile in CHECKPOINTS.values():
        named = tmp_path / profile.repo.split("/")[-1]
        snapshot = (tmp_path / ("models--" + profile.repo.replace("/", "--"))
                    / "snapshots" / profile.revision)
        assert environment_metadata(named).reference_model_hf_id == profile.repo
        assert environment_metadata(snapshot).reference_model_hf_id == profile.repo
    assert environment_metadata(tmp_path / "custom-model").reference_model_hf_id is None


@pytest.mark.parametrize("changed_name", [
    "mlx_lm/models/switch_layers.py",
    "optiq/vlm/_mlxvlm/generate/diffusion.py",
    "optiq/vlm/_mlxvlm/generate/common.py",
    "optiq/vlm/_mlxvlm/tokenizer_utils.py",
])
def test_changed_dependency_source_rejects_resume_without_version_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed_name: str
) -> None:
    dependency_files = (
        "optiq/vlm/diffusion_gemma/loader.py",
        "optiq/vlm/_mlxvlm/models/diffusion_gemma/language.py",
        "optiq/vlm/_mlxvlm/models/diffusion_gemma/diffusion_gemma.py",
        "optiq/vlm/_mlxvlm/models/diffusion_gemma/config.py",
        "optiq/vlm/_mlxvlm/models/cache.py",
        "optiq/vlm/_mlxvlm/generate/diffusion.py",
        "optiq/vlm/_mlxvlm/generate/common.py",
        "optiq/vlm/_mlxvlm/tokenizer_utils.py",
        "mlx_lm/models/switch_layers.py", "mlx_lm/models/cache.py",
    )
    for name in dependency_files:
        source = tmp_path / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("original_source = 1\n")

    def installed_distribution(name: str) -> PathDistribution:
        return PathDistribution(tmp_path / f"{name}.dist-info")

    monkeypatch.setattr("diffusion_jev.accuracy_benchmark.distribution", installed_distribution)
    original = dependency_fingerprints()
    assert len(original) == 10
    config = configuration().model_copy(update={"runtime_dependency_sha256": original})
    checkpoint = tmp_path / "checkpoint.json"
    atomic_write(checkpoint, Checkpoint(configuration=config, measurements=[]))
    (tmp_path / changed_name).write_text("patched_source = 2\n")
    changed = dependency_fingerprints()
    package = "mlx-lm" if changed_name.startswith("mlx_lm/") else "mlx-optiq"
    key = f"{package}/{changed_name}"
    assert changed[key] != original[key]
    with pytest.raises(ValueError, match="differ"):
        restore_checkpoint(
            checkpoint, config.model_copy(update={"runtime_dependency_sha256": changed})
        )


def test_reasoning_budget_cli_is_explicit_and_changes_resume_identity(tmp_path: Path) -> None:
    argv = ["--model", "unused", "--split", "development", "--label", "unit",
            "--output", "unused.json"]
    parser = build_parser()
    assert parser.parse_args(argv, namespace=Arguments()).reasoning_tokens == 0
    enabled = parser.parse_args(argv + ["--reasoning-tokens", "256"], namespace=Arguments())
    assert enabled.reasoning_tokens == 256
    with pytest.raises(SystemExit):
        parser.parse_args(argv + ["--reasoning-tokens", "255"], namespace=Arguments())
    original = configuration()
    path = tmp_path / "checkpoint.json"
    atomic_write(path, Checkpoint(configuration=original, measurements=[]))
    changed = original.model_copy(update={"reasoning_tokens": enabled.reasoning_tokens})
    with pytest.raises(ValueError, match="differ"):
        restore_checkpoint(path, changed)


@pytest.mark.parametrize("source_name", [
    "reasoning.py", "trajectory.py", "components.py", "component_source.py",
])
def test_changed_experimental_source_invalidates_runtime_identity(
    tmp_path: Path, source_name: str,
) -> None:
    assert source_name in RUNTIME_SOURCES
    source = tmp_path / source_name
    source.write_text("configuration = 0\n")
    original = configuration().model_copy(update={
        "runtime_file_sha256": file_fingerprints(tmp_path, RUNTIME_SOURCES)
    })
    checkpoint = tmp_path / "checkpoint.json"
    atomic_write(checkpoint, Checkpoint(configuration=original, measurements=[]))
    source.write_text("configuration = 256\n")
    changed = original.model_copy(update={
        "runtime_file_sha256": file_fingerprints(tmp_path, RUNTIME_SOURCES)
    })
    with pytest.raises(ValueError, match="differ"):
        restore_checkpoint(checkpoint, changed)


def test_trajectory_cli_constraints_and_resume_identity(tmp_path: Path) -> None:
    argv = ["--model", "missing-model", "--split", "development", "--label", "unit",
            "--output", "unused.json"]
    parser = build_parser()
    assert parser.parse_args(argv, namespace=Arguments()).trajectory_steps == 1
    enabled = parser.parse_args(
        argv + ["--trajectory-steps", "2", "--projection", "full"], namespace=Arguments()
    )
    assert enabled.trajectory_steps == 2
    with pytest.raises(SystemExit) as invalid_cli:
        main(argv + ["--trajectory-steps", "2"])
    assert invalid_cli.value.code == 2
    with pytest.raises(SystemExit):
        parser.parse_args(argv + ["--trajectory-steps", "3"], namespace=Arguments())
    original = configuration().model_copy(update={"options": DecisionOptions(projection="full")})
    checkpoint = tmp_path / "checkpoint.json"
    atomic_write(checkpoint, Checkpoint(configuration=original, measurements=[]))
    changed = original.model_copy(update={"trajectory_steps": 2})
    with pytest.raises(ValueError, match="differ"):
        restore_checkpoint(checkpoint, changed)


def test_changed_selected_router_bytes_reject_resume(tmp_path: Path) -> None:
    tensor = RestoredTensor(
        name="synthetic.router", shard="model.safetensors", sha256="a" * 64, size_bytes=4,
    )
    info = RestorationInfo(
        source_repo="public/synthetic", source_revision="b" * 40,
        config_sha256="c" * 64, index_sha256="d" * 64,
        projection_count=1, payload_bytes=4, tensors=(tensor,),
    )
    original = configuration().model_copy(update={"restoration_info": info})
    checkpoint = tmp_path / "checkpoint.json"
    atomic_write(checkpoint, Checkpoint(configuration=original, measurements=[]))
    assert restore_checkpoint(checkpoint, original) == []
    changed_info = info.model_copy(update={
        "tensors": (tensor.model_copy(update={"sha256": "e" * 64}),)
    })
    changed = original.model_copy(update={"restoration_info": changed_info})
    with pytest.raises(ValueError, match="differ"):
        restore_checkpoint(checkpoint, changed)
    assert build_parser().parse_args(
        ["--model", "unused", "--split", "development", "--label", "test", "--output", "out",
         "--router-bf16-source", str(tmp_path)], namespace=Arguments()
    ).router_bf16_source == tmp_path
