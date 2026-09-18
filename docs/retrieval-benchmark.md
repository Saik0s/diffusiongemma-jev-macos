# Can the local model find the right Python function?

This evaluation asks the model to rank real Python functions for a description of the behavior you want.
It tests a useful coding-agent step: **which function should I read first?**
It does not test writing a patch, running code, or completing a software issue.

**On this 50-query pilot, the local model put the labeled function first in 31/50 queries (62%), compared with 24/50 (48%) for keyword search.**
Archived Jev results achieved **42/50 (84%)** on the same queries, with different batching and model conditioning.
A complete local rerank took **19.1 seconds median** for 30 functions, so the improvement has a substantial latency cost.

The source is [Aness Belbati's public Jev reranking study](https://github.com/anessbelbati/jev-rerank-bench),
using [CodeSearchNet Python through MTEB](https://huggingface.co/datasets/mteb/CodeSearchNetRetrieval).
We reuse its frozen candidates and recompute its archived Jev reference on exactly our selected queries.
No paid Jev request is made.

## Results

The initial candidate lists contained a labeled relevant function for **44 of the 50 queries**.
The other six queries cannot be repaired by changing candidate order.
The table below measures ranking quality on those **44 eligible queries**, matching the source study's convention.

| Method | Correct first choice | nDCG@10 | MRR@10 |
| --- | ---: | ---: | ---: |
| BM25 keyword order | **24/44 (54.5%)** | 0.744 | 0.677 |
| Local DiffusionGemma, groups of five | **31/44 (70.5%)** | **0.853** | **0.805** |
| Archived Jev Noul, all 30 together | **42/44 (95.5%)** | 0.983 | 0.977 |

Across **all 50 queries**, including retrieval misses, nDCG@10 is **0.750 local**, **0.655 BM25**, and **0.865 archived Jev**.
All **50 local queries completed**, with **zero failed requests** across the **300 measured calls**.

Local time per 30-function query was **19.13 seconds median** and **40.96 seconds p95**.
The 50 measured queries took **1,074.82 seconds** in total, about **17.9 minutes**.
The separate first-group warmup took **8.96 seconds**. Peak MLX active allocation was **18.91 GB**, including loaded weights.

These are descriptive pilot results. The local top-1 rate on eligible queries has a **95% Wilson interval of 55.8–81.8%**.
The corresponding intervals are **40.1–68.3%** for BM25 and **84.9–98.7%** for archived Jev.
These intervals do not establish a statistically reliable improvement, architectural equivalence, or general coding ability.

The useful result is narrower: this local interface improved the observed ordering of this fixed set of real functions, while remaining well behind archived Jev.
It is a candidate for experimentation, not evidence of fast or dependable repository-wide search.
See [the full measurement data](../benchmarks/codesearchnet-python-2026-09-18.json).

## How the task works

Each query is a function description drawn from documentation.
A keyword search method called **BM25** supplies 30 candidate functions from a corpus of 1,000.
A *reranker* examines those candidates and puts the most relevant ones first.

We ask a Noul question about each candidate: does this passage provide the information requested by the query?
The local model's probability of yes becomes that function's ranking score.
We compare the resulting order with the dataset's recorded matching function.

This is an existing task from a Jev evaluation, not a new collection of hand-picked toy examples.
However, it remains a small public retrieval dataset whose descriptions can overlap with code vocabulary.
It is not a comprehensive measure of code understanding.

## Fixed protocol

The selection was made before local inference:

- Take the **first 50 rows** in the frozen source candidate file. Do not select cases based on model success.
- Keep **all 30 candidates per query**, in their original BM25 order.
- Keep the source's **first 2,000 characters per passage**.
- Use **seed 0**, **one sample**, **packed** questions, selected-label projection, and the automatic canvas width.
- Evaluate **five candidates per request**, making **six sequential requests per query**.
- Run one excluded warmup request containing the first query's first group.
- Keep failed local queries in the quality denominator with zero scores. Report retrieval misses separately from conditional ranking quality.

The benchmark sends **1,500 candidate judgments** in **300 measured requests**, plus the warmup.
It neither executes retrieved code nor sends it to a hosted inference API.

## Why use groups of five?

The default local server accepts at most **8,192 input tokens** after formatting.
CPU tokenization found that **47 of the 50** full candidate sets exceed that limit; the largest needs **15,753 tokens**.
Five-candidate groups fit, with a maximum of **3,434 tokens** in this cohort.

**This changes the context seen by the model.** The archived Jev run considered all 30 candidates in one call.
The local wrapper and model are different too. The comparison provides useful reference points, but does not isolate one architectural difference.
The Noul true/false definitions from the source study are included in local instructions because the local API lacks separate Noul criteria fields.

## What the measurements mean

**Candidate coverage** asks whether keyword search found any labeled correct candidate in its first 30 results.
Here, that is **44/50 queries, or 88%**. A reranker cannot recover a function that is absent from its candidate list.

**Top-1** is the fraction of queries whose first-ranked function is labeled relevant.
For example, 24/44 means the first suggestion matches the label in 24 eligible queries.

**nDCG@10** rewards relevant functions near the top of the first ten results, normalized against an ideal order.
A score of 1 means ideal ranking under these labels; 0.8 does **not** mean 80% of tasks were solved.
The calculation uses the source study's linear relevance gains and all supplied labels for the ideal ranking.

**MRR@10** averages the reciprocal position of the first relevant function: rank 1 earns 1, rank 2 earns 0.5, and so on.
No relevant result in the first ten earns zero.
Tied model scores retain the original BM25 order, matching the source protocol. That tie rule can benefit results.

The main quality table uses the **44 eligible queries**.
The JSON report also includes all-query metrics over **50**, so the six retrieval misses are visible rather than disappearing.

## Timing and environment

The client and server run on the same **Apple M2 Ultra with 64 GiB memory**.
The server uses **Python 3.13.9**, **MLX 0.32.2**, **mlx-optiq 0.5.12**, and the pinned 4-bit OptiQ DiffusionGemma checkpoint.
The service was already loaded. Model loading and the excluded warmup are not part of measured query latency.

Query timing includes all six HTTP requests and their sequential local inference.
It therefore describes one complete 30-function rerank, not one Noul question.
We do not compare these timings with cloud Jev latency from another machine and network.
This run was not thermally controlled, and other applications could affect timings.

## Reproduce

Start the server in one terminal:

```sh
uv run jev-local start
```

From the checkout, run the benchmark in another:

```sh
uv run --extra benchmark jev-local benchmark-search \
  --limit 50 \
  --batch-size 5 \
  --seed 0 \
  --base-url http://127.0.0.1:8017 \
  --output retrieval-results.local.json
```

This uses the running server, so **do not stop it** for this benchmark.
The separate [runtime benchmark](benchmarks.md) loads its own model and has different startup instructions.

`--limit` accepts **1–300** frozen queries. `--batch-size` accepts **1–30**.
A larger group may exceed the server's input limit. Changing the group size changes the evaluation context and can change quality.
Use a new output filename when comparing configurations.

The benchmark needs the optional `benchmark` dependency group to read the Parquet data format.
Public corpus/candidate files are cached under `~/.cache/diffusion-jev/retrieval-v1`, or the directory given by `--cache`.
Their checksums are verified. The archived reference response file is fetched and decoded in memory on each run.

The result contains source revisions, hashes, settings, per-query metrics, timing, and query identifiers.
It omits query text, function bodies, raw API responses, and per-candidate model scores.

## Sources and limits

Frozen source revisions:

- Jev study: [`cd9a35b22aeb4187334f7018a0ee1960a7470586`](https://github.com/anessbelbati/jev-rerank-bench/tree/cd9a35b22aeb4187334f7018a0ee1960a7470586).
- Dataset: [`68e8f0731a656fa4bd5b7c81936d95ad48a39bfe`](https://huggingface.co/datasets/mteb/CodeSearchNetRetrieval/tree/68e8f0731a656fa4bd5b7c81936d95ad48a39bfe).
- Candidate list: `candidates/csn-python.jsonl`.
- Corpus: `python-corpus/test-00000-of-00001.parquet`.
- Archived Jev results: `cache/jev-noul-batch/csn-python.present.jsonl.gz`.

The harness independently reproduces the source's full **256-eligible-query** archived Jev nDCG, top-1, and MRR values to **1e-12**.
This checks the join and scoring implementation; it does not independently rerun Jev.

The dataset card and source harness declare MIT licenses. Original functions retain their source repositories' terms.
We download them for evaluation and do not vendor their bodies here.
The source's question wording is credited in [the third-party notice](../src/diffusion_jev/NOTICE).

The labels are sparse and can miss other useful functions. Public training-data overlap is unknown.
A fixed 50-query prefix is a pilot, not a representative sample of all software engineering.
The report's 95% Wilson intervals describe top-1 uncertainty in this small cohort; they are not a controlled significance or equivalence test.

For another software-focused Jev evaluation, the source study includes BRIGHT StackOverflow retrieval.
For rule checking, [jev-code-review-benchmark](https://github.com/gemanor/jev-code-review-benchmark) uses constructed program variants.
We did not run those suites here. Neither this evaluation nor the simple demos produces a SWE-bench score.
