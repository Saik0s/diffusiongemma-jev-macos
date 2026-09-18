import hashlib
import math
from pathlib import Path

import pytest

from diffusion_jev.client import DecisionClientError
from diffusion_jev.retrieval_benchmark import (
    build_requests,
    rank_metrics,
    read_scores,
    run_retrieval,
    summarize,
)
from diffusion_jev.retrieval_data import (
    ArchivedPrediction,
    Candidate,
    Query,
    RetrievalData,
    Source,
    fetch_source,
    validate_join,
)
from diffusion_jev.schemas import DecisionRequest, DecisionResponse, NoulAnswer, Usage


def fixture_data() -> RetrievalData:
    query = Query(qid="q", query="Find matching function", relevant={"d1": 1}, n_rel_top30=1,
                  present=[Candidate(did=f"d{i}", bm25=float(30 - i)) for i in range(30)])
    archive = ArchivedPrediction(qid="q", variant="present", model="jev-noul-batch", ok=True,
                                 dids=[c.did for c in query.present],
                                 scores=[float(i == 1) for i in range(30)])
    return RetrievalData([query], {f"d{i}": "code" * 600 for i in range(30)}, {"q": archive})


def answer(request: DecisionRequest) -> DecisionResponse:
    return DecisionResponse(
        model="test-local",
        answers={key: NoulAnswer(noul=0.9 if key == "p02" else 0.1)
                 for key in request.questions},
        usage=Usage(prompt_tokens=100, questions=len(request.questions), decoder_passes=1,
                    canvas_tokens=32, prefill_ms=1.0, decode_ms=1.0, total_ms=2.0,
                    peak_memory_gb=1.0),
    )


def test_ranking_matches_upstream_discount_and_stable_ties() -> None:
    query = fixture_data().queries[0]
    metric = rank_metrics(query, [1.0] * 30)
    assert metric.ndcg10 == pytest.approx(1 / math.log2(3))
    assert metric.top1 == 0
    assert metric.mrr10 == 0.5
    query.relevant = {"d0": 2, "outside": 1}
    result = rank_metrics(query, [1.0] * 30)
    assert result.ndcg10 == pytest.approx(2 / (2 + 1 / math.log2(3)))
    with pytest.raises(ValueError):
        rank_metrics(query, [1.0])
    with pytest.raises(ValueError):
        rank_metrics(query, [math.nan] * 30)


def test_grouping_preserves_all_candidates_and_source_criteria() -> None:
    data = fixture_data()
    requests = build_requests(data.queries[0], data.documents, group_size=5, seed=17)
    assert len(requests) == 6
    assert [key for request in requests for key in request.questions] == [
        f"p{i:02d}" for i in range(1, 31)
    ]
    assert requests[0].state == {
        "query": data.queries[0].query,
        "passages": {f"p{i:02d}": "code" * 500 for i in range(1, 6)},
    }
    assert requests[0].options.seed == 17
    instruction = requests[0].questions["p01"].instructions
    assert instruction.startswith("Does passage p01 contain the information needed")
    assert "True: The passage states or directly implies" in instruction
    assert "False: The passage is only on a related topic" in instruction
    assert len(build_requests(data.queries[0], data.documents, group_size=1, seed=0)) == 30


def test_join_rejects_candidate_order_and_missing_corpus() -> None:
    data = fixture_data()
    validate_join(data)
    data.archived["q"].dids.reverse()
    with pytest.raises(ValueError, match="identity or order"):
        validate_join(data)
    data = fixture_data()
    del data.documents["d0"]
    with pytest.raises(ValueError, match="missing from corpus"):
        validate_join(data)


def test_invalid_accounting_is_not_a_valid_prediction() -> None:
    data = fixture_data()
    request = build_requests(data.queries[0], data.documents, group_size=5, seed=0)[0]
    response = answer(request)
    response.usage.questions = 4
    with pytest.raises(ValueError, match="accounting"):
        read_scores(request, response)
    response = answer(request)
    del response.answers["p01"]
    with pytest.raises(ValueError, match="identity"):
        read_scores(request, response)


def test_failure_stays_in_fixed_quality_denominator_and_report_has_no_content() -> None:
    calls = 0

    def fail_after_warmup(request: DecisionRequest) -> DecisionResponse:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise DecisionClientError("request failed")
        return answer(request)

    data = fixture_data()
    report = run_retrieval(data, fail_after_warmup)
    assert report.failed == 1
    assert report.quality["local"].n == 1
    assert report.quality["local"].top1 == 0
    assert report.quality["archived_jev_noul_batch"].top1 == 1
    assert report.wall_p50_ms is None
    serialized = report.model_dump_json()
    assert "Find matching function" not in serialized
    assert "codecode" not in serialized
    assert '"scores"' not in serialized


def test_warmup_excluded_and_all_thirty_scores_ranked() -> None:
    calls = 0

    def decide(request: DecisionRequest) -> DecisionResponse:
        nonlocal calls
        calls += 1
        return answer(request)

    report = run_retrieval(fixture_data(), decide)
    assert calls == 7
    assert report.measurements[0].request_count == 6
    assert report.measurements[0].prompt_tokens == 600
    assert report.quality["local"].top1 == 1
    assert report.successful == 1
    assert report.wall_p95_ms is not None
    assert summarize([]).n == 0


def test_checksum_is_checked_even_for_cached_file(tmp_path: Path) -> None:
    content = b"safe public data"
    source = Source("input", "https://unused.invalid", hashlib.sha256(content).hexdigest())
    (tmp_path / "input").write_bytes(content)
    assert fetch_source(source, tmp_path) == content
    (tmp_path / "input").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        fetch_source(source, tmp_path)


def test_retrieval_miss_is_in_all_query_metrics_but_not_conditional_metrics() -> None:
    data = fixture_data()
    data.queries[0].relevant = {"not-in-candidates": 1}
    data.queries[0].n_rel_top30 = 0
    validate_join(data)
    report = run_retrieval(data, answer)
    assert report.coverage == 0
    assert report.successful == 1
    assert report.quality["local"].n == 0
    assert report.quality["local"].ndcg10 is None
    assert report.all_query_quality["local"].n == 1
    assert report.all_query_quality["local"].ndcg10 == 0
