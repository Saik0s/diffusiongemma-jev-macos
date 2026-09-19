"""Bounded real-code retrieval pilot with archived Jev quality references."""

import argparse
import math
import platform
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from pydantic import JsonValue

from diffusion_jev.benchmark import nearest_rank
from diffusion_jev.client import DecisionClient, DecisionClientError
from diffusion_jev.retrieval_data import (
    DATASET_REVISION,
    DEFAULT_CACHE,
    REPO_REVISION,
    SOURCES,
    MissingBenchmarkDependency,
    Query,
    RetrievalData,
    load_data,
)
from diffusion_jev.schemas import (
    DecisionOptions,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    StrictModel,
)

# Source question wording: jev-rerank-bench (MIT), see the bundled NOTICE.
RELEVANCE_QUESTION = (
    "Does the passage contain the information needed to answer or verify the query?"
)
RELEVANCE_TRUE = (
    "The passage states or directly implies the answer to the query, "
    "or the evidence that verifies it."
)
RELEVANCE_FALSE = (
    "The passage is only on a related topic; it does not supply what the query asks for."
)


class Ranking(StrictModel):
    ndcg10: float
    top1: float
    mrr10: float


class DiagnosticSummary(StrictModel):
    slots: int
    allowed_label_mass_min: float
    allowed_label_mass_mean: float
    invalid_argmax_count: int


def merge_diagnostics(
    previous: DiagnosticSummary | None, current: DiagnosticSummary
) -> DiagnosticSummary:
    if current.slots < 1:
        raise ValueError("Diagnostics require at least one slot")
    if previous is None:
        return current
    return DiagnosticSummary(
        slots=previous.slots + current.slots,
        allowed_label_mass_min=min(previous.allowed_label_mass_min, current.allowed_label_mass_min),
        allowed_label_mass_mean=(
            previous.allowed_label_mass_mean * previous.slots
            + current.allowed_label_mass_mean * current.slots
        ) / (previous.slots + current.slots),
        invalid_argmax_count=previous.invalid_argmax_count + current.invalid_argmax_count,
    )


class QueryMeasurement(StrictModel):
    qid: str
    eligible: bool
    success: bool
    error: str | None = None
    local: Ranking | None = None
    bm25: Ranking
    archived_jev: Ranking
    request_count: int = 0
    response_count: int = 0
    decoder_passes: int = 0
    canvas_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_ms: float = 0.0
    reasoning_passes: int = 0
    reasoning_forced_closures: int = 0
    trajectory_accepted_slots: int = 0
    top_score_ties: int | None = None
    diagnostics: DiagnosticSummary | None = None
    wall_ms: float = 0.0
    prompt_tokens: int = 0
    peak_memory_gb: float = 0.0


class Quality(StrictModel):
    n: int
    ndcg10: float | None
    top1: float | None
    mrr10: float | None
    top1_wilson95: list[float] | None


class ClientEnvironment(StrictModel):
    system: str
    architecture: str
    python: str
    packages: dict[str, str]
    note: str = "Client runtime only; server hardware and package versions are not inferred"


def client_environment() -> ClientEnvironment:
    packages: dict[str, str] = {}
    for name in ("diffusion-jev", "httpx", "pydantic", "pyarrow"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not installed"
    return ClientEnvironment(system=f"{platform.system()} {platform.release()}",
                             architecture=platform.machine(), python=platform.python_version(),
                             packages=packages)


class Report(StrictModel):
    schema_version: int = 1
    timestamp_utc: str
    dataset: str = "mteb/CodeSearchNetRetrieval:python:test"
    dataset_revision: str = DATASET_REVISION
    reference_revision: str = REPO_REVISION
    source_sha256: dict[str, str]
    dataset_license: str = "MIT (dataset card); original code retains source repository terms"
    reference_license: str = "MIT"
    selection: str = (
        "First limit rows in pinned candidates/csn-python.jsonl; no model-based selection"
    )
    limit: int
    seed: int
    group_size: int
    decision_options: DecisionOptions
    client_environment: ClientEnvironment
    passage_characters: int = 2000
    candidate_count: int = 30
    model: str
    protocol: str
    caveats: list[str]
    warmup_ms: float
    eligible: int
    coverage: float
    successful: int
    failed: int
    quality: dict[str, Quality]
    all_query_quality: dict[str, Quality]
    wall_p50_ms: float | None
    wall_p95_ms: float | None
    measurements: list[QueryMeasurement]


def rank_metrics(query: Query, scores: list[float]) -> Ranking:
    if len(scores) != len(query.present) or any(not math.isfinite(s) for s in scores):
        raise ValueError("Ranking requires one finite score per candidate")
    # Stable sort intentionally matches the upstream optimistic BM25-order tie break.
    ranked = sorted(range(len(scores)), key=lambda index: -scores[index])
    grades = [query.relevant.get(query.present[index].did, 0) for index in ranked]
    available = sorted(query.relevant.values(), reverse=True)
    dcg = sum(grade / math.log2(i + 2) for i, grade in enumerate(grades[:10]))
    ideal = sum(grade / math.log2(i + 2) for i, grade in enumerate(available[:10]))
    first = next((i + 1 for i, grade in enumerate(grades[:10]) if grade > 0), None)
    return Ranking(ndcg10=dcg / ideal if ideal else 0.0, top1=float(grades[0] > 0),
                   mrr10=1 / first if first else 0.0)


def summarize(values: list[Ranking]) -> Quality:
    if not values:
        return Quality(n=0, ndcg10=None, top1=None, mrr10=None, top1_wilson95=None)
    n = len(values)
    accuracy = statistics.mean(item.top1 for item in values)
    z = 1.959963984540054
    denominator = 1 + z * z / n
    center = (accuracy + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(accuracy * (1 - accuracy) / n + z * z / (4 * n * n)) / denominator
    return Quality(n=n, ndcg10=statistics.mean(item.ndcg10 for item in values),
                   top1=accuracy, mrr10=statistics.mean(item.mrr10 for item in values),
                   top1_wilson95=[max(0.0, center - radius), min(1.0, center + radius)])


def build_requests(
    query: Query, documents: dict[str, str], *, group_size: int, seed: int = 0,
    options: DecisionOptions | None = None,
) -> list[DecisionRequest]:
    if not 1 <= group_size <= 30:
        raise ValueError("Group size must be between 1 and 30")
    requests: list[DecisionRequest] = []
    for start in range(0, 30, group_size):
        passages: dict[str, JsonValue] = {}
        questions: dict[str, Question] = {}
        for index in range(start, min(start + group_size, 30)):
            pid = f"p{index + 1:02d}"
            passages[pid] = documents[query.present[index].did][:2000]
            instruction = RELEVANCE_QUESTION.replace("the passage", f"passage {pid}")
            # Local Noul has no criteria field; retain the source criteria in its instructions.
            questions[pid] = NoulQuestion(
                type="noul", instructions=f"{instruction}\nTrue: {RELEVANCE_TRUE}\n"
                f"False: {RELEVANCE_FALSE}",
            )
        requests.append(DecisionRequest(
            state={"query": query.query, "passages": passages}, questions=questions,
            options=options or DecisionOptions(seed=seed, mode="packed", projection="labels"),
        ))
    return requests


def read_scores(request: DecisionRequest, response: DecisionResponse) -> list[float]:
    if response.answers.keys() != request.questions.keys():
        raise ValueError("Response question identity mismatch")
    if response.usage.questions != len(request.questions):
        raise ValueError("Response question accounting mismatch")
    result: list[float] = []
    for key in request.questions:
        answer = response.answers[key]
        if not isinstance(answer, NoulAnswer):
            raise ValueError("Expected Noul response")
        result.append(answer.noul)
    return result


def run_retrieval(
    data: RetrievalData,
    decide: Callable[[DecisionRequest], DecisionResponse],
    *, group_size: int = 5, seed: int = 0, options: DecisionOptions | None = None,
    selection: str | None = None, completed: list[QueryMeasurement] | None = None,
    on_measurement: Callable[[QueryMeasurement], None] | None = None,
    read_diagnostics: Callable[[], DiagnosticSummary | None] | None = None,
) -> Report:
    if not data.queries:
        raise ValueError("At least one public query is required")
    effective = options or DecisionOptions(seed=seed)
    measurements = list(completed or [])
    if [item.qid for item in measurements] != [q.qid for q in data.queries[:len(measurements)]]:
        raise ValueError("Completed measurements must be an exact query prefix")
    first = build_requests(
        data.queries[0], data.documents, group_size=group_size, options=effective
    )[0]
    started = time.perf_counter()
    warmup = decide(first)
    read_scores(first, warmup)
    warmup_ms = (time.perf_counter() - started) * 1000
    for query in data.queries[len(measurements):]:
        item = QueryMeasurement(
            qid=query.qid, eligible=query.n_rel_top30 > 0, success=False,
            bm25=rank_metrics(query, [candidate.bm25 for candidate in query.present]),
            archived_jev=rank_metrics(query, data.archived[query.qid].scores),
        )
        started = time.perf_counter()
        scores: list[float] = []
        try:
            for request in build_requests(
                query, data.documents, group_size=group_size, options=effective
            ):
                item.request_count += 1
                response = decide(request)
                item.response_count += 1
                # Work happened even if the returned answer cannot be used for ranking.
                item.prompt_tokens += response.usage.prompt_tokens
                item.decoder_passes += response.usage.decoder_passes
                item.canvas_tokens += response.usage.canvas_tokens
                item.reasoning_tokens += response.usage.reasoning_tokens
                item.reasoning_ms += response.usage.reasoning_ms
                item.reasoning_passes += response.usage.reasoning_passes
                item.reasoning_forced_closures += response.usage.reasoning_forced_closures
                item.trajectory_accepted_slots += response.usage.trajectory_accepted_slots
                item.peak_memory_gb = max(item.peak_memory_gb, response.usage.peak_memory_gb)
                if response.model != warmup.model:
                    raise ValueError("Model identity changed during benchmark")
                scores.extend(read_scores(request, response))
                if read_diagnostics is not None:
                    diagnostic = read_diagnostics()
                    if diagnostic is not None:
                        item.diagnostics = merge_diagnostics(item.diagnostics, diagnostic)
            item.local = rank_metrics(query, scores)
            item.top_score_ties = scores.count(max(scores))
            item.success = True
        except (DecisionClientError, ValueError, RuntimeError):
            item.error = "Request or response validation failed; no partial ranking used"
        item.wall_ms = (time.perf_counter() - started) * 1000
        measurements.append(item)
        if on_measurement is not None:
            on_measurement(item)
        print(f"Completed query {len(measurements)}/{len(data.queries)}; "
              f"success={item.success}", flush=True)

    def quality(eligible_only: bool) -> dict[str, Quality]:
        # Keep the cohort fixed: a failed local query contributes zero, never disappears.
        selected = [m for m in measurements if m.eligible or not eligible_only]
        zero = Ranking(ndcg10=0.0, top1=0.0, mrr10=0.0)
        return {
            "local": summarize([m.local if m.local is not None else zero for m in selected]),
            "bm25": summarize([m.bm25 for m in selected]),
            "archived_jev_noul_batch": summarize([m.archived_jev for m in selected]),
        }

    eligible = sum(item.eligible for item in measurements)
    latency = [item.wall_ms for item in measurements if item.success]
    report = Report(
        timestamp_utc=datetime.now(UTC).isoformat(),
        source_sha256={source.name: source.sha256 for source in SOURCES},
        limit=len(data.queries), seed=effective.seed, group_size=group_size, model=warmup.model,
        decision_options=first.options, client_environment=client_environment(),
        protocol=(f"30 BM25 candidates, sequential groups of {group_size}; "
                  f"{effective.mode} Noul with the full group state. "
                  "Source Noul criteria appended to local instructions."),
        caveats=[
            "Docstring-to-code retrieval pilot; not code generation, bug detection or SWE-bench.",
            "Archived Jev used all 30 passages in one call; smaller groups change conditioning.",
            "Different local prompt wrapper and decoder; archived quality is contextual, not a "
            "controlled architecture or latency comparison.",
            "Conditional metrics include only queries with labeled relevant candidates; "
            "all-query metrics include retrieval misses. Local failures count as zero.",
            "Stable ties retain BM25 order, as upstream; sparse labels may miss valid matches.",
            "Wilson intervals are descriptive for a small fixed cohort, not population or "
            "equivalence guarantees. Public training contamination is unknown.",
            "One excluded first-group warmup; wall latency includes sequential groups and any "
            "client transport overhead. "
            "No archived cloud latency comparison is made.",
            "Usage includes returned responses, including invalid ones; requests failing before "
            "a response may have unreported work. Compare request_count with response_count.",
        ],
        warmup_ms=warmup_ms, eligible=eligible, coverage=eligible / len(measurements),
        successful=len(latency), failed=len(measurements) - len(latency),
        quality=quality(True), all_query_quality=quality(False),
        wall_p50_ms=nearest_rank(latency, 0.5) if latency else None,
        wall_p95_ms=nearest_rank(latency, 0.95) if latency else None,
        measurements=measurements,
    )
    if selection is not None:
        report.selection = selection
    return report


class Arguments(argparse.Namespace):
    limit: int
    group_size: int
    seed: int
    base_url: str
    cache: Path
    output: Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--batch-size", "--group-size", dest="group_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default="http://127.0.0.1:8017")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=Path("retrieval-results.local.json"))
    args = parser.parse_args(argv, namespace=Arguments())
    if not 1 <= args.limit <= 300 or not 1 <= args.group_size <= 30:
        parser.error("limit must be 1..300 and group-size must be 1..30")
    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("seed must be 0..4294967295")
    print("Loading verified public data; raw reference responses stay in memory.", flush=True)
    try:
        data = load_data(args.limit, args.cache.expanduser())
        with DecisionClient(args.base_url) as client:
            report = run_retrieval(data, client.decide, group_size=args.group_size, seed=args.seed)
        args.output.expanduser().write_text(
            report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except MissingBenchmarkDependency:
        parser.exit(1, "Install the optional data reader with: uv sync --extra benchmark\n")
    except OSError:
        parser.exit(1, "Could not read or write benchmark files; check cache/output access.\n")
    except (ValueError, DecisionClientError):
        parser.exit(1, "Benchmark failed during data validation or warmup; no content logged.\n")
    print(f"Report written; {report.successful} successful, {report.failed} failed queries.")
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
