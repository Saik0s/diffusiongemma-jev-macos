"""Paired query-level accuracy comparisons using only IDs and aggregate metrics."""

import argparse
import math
import random
import statistics
from pathlib import Path

from pydantic import TypeAdapter

from diffusion_jev.accuracy_benchmark import AccuracyReport, atomic_write
from diffusion_jev.benchmark import nearest_rank
from diffusion_jev.retrieval_benchmark import QueryMeasurement, Ranking, Report
from diffusion_jev.schemas import StrictModel


class MetricDifference(StrictModel):
    mean: float
    bootstrap95: list[float]


class PairedComparison(StrictModel):
    queries: int
    eligible_queries: int
    candidate_wins: int
    candidate_losses: int
    both_correct: int
    both_wrong: int
    baseline_failures: int
    candidate_failures: int
    newly_failed: int
    exact_mcnemar_two_sided_p: float
    top1: MetricDifference
    ndcg10: MetricDifference
    mrr10: MetricDifference
    bootstrap_repeats: int
    bootstrap_seed: int
    source_sha256: dict[str, str]
    query_ids: list[str]
    method: str = (
        "Paired query percentile bootstrap, nearest-rank 2.5% and 97.5%; "
        "failed queries score zero. Exact two-sided McNemar conditional binomial test. "
        "No multiple-comparison correction; queries sharing targets are not clustered. "
        "Different seeds do not create additional independent query observations."
    )


def exact_mcnemar(wins: int, losses: int) -> float:
    if wins < 0 or losses < 0:
        raise ValueError("Discordant counts must be nonnegative")
    discordant = wins + losses
    if not discordant:
        return 1.0
    tail = sum(math.comb(discordant, count) for count in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / (1 << discordant))


def validated_measurements(report: Report) -> dict[str, QueryMeasurement]:
    indexed = {item.qid: item for item in report.measurements}
    if len(indexed) != len(report.measurements) or len(indexed) != report.limit:
        raise ValueError("Comparison requires a complete report with unique query IDs")
    for item in indexed.values():
        if item.success != (item.local is not None):
            raise ValueError("Query success and metric availability disagree")
        if item.local is not None:
            if item.local.top1 not in (0.0, 1.0):
                raise ValueError("Query top1 must be binary")
            if not all(math.isfinite(value) and 0 <= value <= 1
                       for value in (item.local.ndcg10, item.local.mrr10)):
                raise ValueError("Query metrics must be finite proportions")
    return indexed


def compare_reports(
    baseline: Report, candidate: Report, *, bootstrap_repeats: int = 10000, seed: int = 0
) -> PairedComparison:
    if bootstrap_repeats < 1:
        raise ValueError("bootstrap_repeats must be positive")
    if (baseline.source_sha256 != candidate.source_sha256
            or baseline.dataset_revision != candidate.dataset_revision
            or baseline.reference_revision != candidate.reference_revision
            or baseline.dataset != candidate.dataset):
        raise ValueError("Paired comparison requires identical source identities")
    left, right = validated_measurements(baseline), validated_measurements(candidate)
    if not left or left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical nonempty query ID sets")
    ids = sorted(left)
    zero = Ranking(top1=0.0, ndcg10=0.0, mrr10=0.0)
    top1: list[float] = []
    ndcg: list[float] = []
    mrr: list[float] = []
    both_correct = both_wrong = 0
    for qid in ids:
        old, new = left[qid], right[qid]
        if (old.eligible != new.eligible or old.bm25 != new.bm25
                or old.archived_jev != new.archived_jev):
            raise ValueError("Query coverage or fixed reference metrics differ")
        a, b = old.local or zero, new.local or zero
        top1.append(b.top1 - a.top1)
        ndcg.append(b.ndcg10 - a.ndcg10)
        mrr.append(b.mrr10 - a.mrr10)
        both_correct += a.top1 == 1.0 and b.top1 == 1.0
        both_wrong += a.top1 == 0.0 and b.top1 == 0.0
    rng = random.Random(seed)
    top_samples: list[float] = []
    ndcg_samples: list[float] = []
    mrr_samples: list[float] = []
    for _ in range(bootstrap_repeats):
        indices = [rng.randrange(len(ids)) for _ in ids]
        top_samples.append(sum(top1[index] for index in indices) / len(ids))
        ndcg_samples.append(sum(ndcg[index] for index in indices) / len(ids))
        mrr_samples.append(sum(mrr[index] for index in indices) / len(ids))

    def difference(values: list[float], samples: list[float]) -> MetricDifference:
        return MetricDifference(
            mean=statistics.mean(values),
            bootstrap95=[nearest_rank(samples, 0.025), nearest_rank(samples, 0.975)],
        )

    wins, losses = top1.count(1.0), top1.count(-1.0)
    return PairedComparison(
        queries=len(ids), eligible_queries=sum(item.eligible for item in left.values()),
        candidate_wins=wins, candidate_losses=losses,
        both_correct=both_correct, both_wrong=both_wrong,
        baseline_failures=sum(not item.success for item in left.values()),
        candidate_failures=sum(not item.success for item in right.values()),
        newly_failed=sum(left[qid].success and not right[qid].success for qid in ids),
        exact_mcnemar_two_sided_p=exact_mcnemar(wins, losses),
        top1=difference(top1, top_samples), ndcg10=difference(ndcg, ndcg_samples),
        mrr10=difference(mrr, mrr_samples), bootstrap_repeats=bootstrap_repeats,
        bootstrap_seed=seed, source_sha256=baseline.source_sha256, query_ids=ids,
    )


def read_report(path: Path) -> Report:
    adapter: TypeAdapter[AccuracyReport | Report] = TypeAdapter(AccuracyReport | Report)
    loaded = adapter.validate_json(path.read_text(encoding="utf-8"))
    return loaded.retrieval if isinstance(loaded, AccuracyReport) else loaded


class Arguments(argparse.Namespace):
    baseline: Path
    candidate: Path
    output: Path
    bootstrap_repeats: int
    seed: int


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv, namespace=Arguments())
    try:
        report = compare_reports(
            read_report(args.baseline), read_report(args.candidate),
            bootstrap_repeats=args.bootstrap_repeats, seed=args.seed,
        )
        atomic_write(args.output, report)
    except (OSError, ValueError):
        parser.exit(1, "Comparison failed; report identities, metrics, or paths are invalid.\n")
    print(f"Compared {report.queries} paired queries; "
          f"wins={report.candidate_wins}, losses={report.candidate_losses}.")


if __name__ == "__main__":
    main()
