# Local runtime speed on an M2 Ultra

This report asks **how quickly the server evaluates small requests** and which
implementation changes save time. For quality on real Python functions, use the
separate [CodeSearchNet evaluation](retrieval-benchmark.md).

In this report, **median (p50)** is the middle measured time; **p95** is the time
below which 95% of measurements fall. A **canvas** is the model's short answer
workspace. **Prefill** means reading the input. **Projection** is the final
calculation that turns model values into preferences for allowed answers.

The default configuration reduced median warm latency from **472 ms to 291 ms
(1.62× faster)** on the measured mixed workload. Most of the gain came from
shorter canvases. All four configurations answered the small synthetic suite
correctly; this is a mechanics and example check, not evidence of general coding
ability or calibrated probabilities.

[Measurement data](../benchmarks/m2-ultra-2026-09-18.json) contains configuration
options, individual timing measurements, aggregate correctness, and environment
metadata. It contains no request or answer payloads.

## Environment and method

- **Apple M2 Ultra**, **64 GiB** unified memory, macOS **27.0**.
- Python **3.13.9**, MLX **0.32.2**, mlx-optiq **0.5.12**; complete versions in
  `uv.lock`.
- Existing `diffusiongemma-26B-A4B-it-OptiQ-4bit` checkpoint. No model conversion,
  training, or weight download in the measured run.
- One process, one model, sequential inference. The HTTP server was stopped
  while benchmarking. Timing surrounds the direct engine call and excludes
  network transport and queueing.
- **8 cases, 20 distinct judgments**: four simple state-reading cases and four
  coding examples. Each runs with seeds **0, 1, 2** under every configuration.
- One excluded warmup per case and configuration. Configuration order rotates
  and reverses across repeats. No cross-request result or prompt cache.
- **149 total requests**: 1 first request, 32 quality warmups, 96 measured quality
  requests, 5 scaling warmups, and 15 measured scaling requests.
- p50 and p95 use nearest rank. Throughput is the total number of questions
  divided by total measured wall time. GPU arrays are evaluated before timing
  boundaries. Three repeats are too few to estimate production tail latency.

Loading took **12.68 seconds**. The first 256-token reference request took
**4.55 seconds**, including initial kernel/setup work. Both are excluded from warm
latency. These are cold-process measurements; the operating system's disk cache
was not flushed.

## Warm mixed workload

Each row contains **24 requests and 60 judgments**, with one noise draw per
question group. All are structured reads, not freeform text generation.

| Configuration | p50 / p95 | Questions/s | Correct |
| --- | ---: | ---: | ---: |
| Full vocabulary, 256-token canvas | 472 / 562 ms | 5.14 | 60/60 |
| Full vocabulary, smallest fitting canvas | 295 / 362 ms | 8.27 | 60/60 |
| **Selected labels, smallest canvas, default** | **291 / 359 ms** | **8.47** | **60/60** |
| Selected labels, questions isolated | 659 / 729 ms | 4.18 | 60/60 |

The compact full-vocabulary and selected-label paths used **identical prompts,
canvas widths, and noise**. Their maximum probability difference was **0.0** over
**60 paired judgments**, with **zero argmax disagreements**. The harness records
a tolerance of 0.01; the observed difference was zero. A separate CPU regression
checks the pinned runtime's float32 softcap transformation.

Canvas width and question isolation change conditioning. Their quality is
therefore measured separately; they are not mathematical equivalence claims.

## What each optimization bought

1. **Short canvases:** 472 → 295 ms median request time. The decoder median fell
   from **261 to 84 ms**. Most small requests need only 16 or 32 canvas positions,
   rather than 256. The largest observed MLX active allocation across the mixed
   workload fell from **18.65 to 18.30 GB**.
2. **Project only allowed labels:** 295 → 291 ms median request time, about **1.6%**
   on top of compact canvases. Decoder median fell from **83.9 to 79.5 ms**.
   This is a modest gain on short canvases, not the main speedup.
3. **Pack related questions:** 659 → 291 ms median on this workload versus
   isolated questions. This avoids repeated prefill, but allows questions to
   influence one another. Use isolated mode when that distinction matters.
4. **Keep the model loaded and reuse prefill within noise sampling:** startup is
   excluded from subsequent requests. A separate live API check with four noise
   draws measured **234 ms prefill + 328 ms decode**, with exactly four decoder
   calls and one encoded prompt. This is structural reuse, not a separately
   measured speedup against a no-reuse implementation.

Prompt prefill still dominates typical warm requests: approximately **211 ms**
of the default **291 ms** median. The service deliberately keeps no persistent
cross-request prompt cache.

## Coding examples

These use the same public fixtures as the runnable examples, three seeds each.
The table shows the default direct-engine median, without HTTP overhead.

| Example | Questions/request | p50 | Correct judgments |
| --- | ---: | ---: | ---: |
| Failure triage | 3 | 314 ms | 9/9 |
| File selection | 1 | 255 ms | 3/3 |
| Patch review | 2 | 269 ms | 6/6 |
| Context compaction | 2 | 359 ms | 6/6 |

Correctness means the supplied expected choice, Boolean threshold at **0.5**, or
most-probable Score level matched. It does not validate the reliability of the
numeric confidence or probability. The examples are intentionally simple and
their expected answers are visible in source. They are not a held-out coding
benchmark, realistic transcript evaluation, or security-review certification.

All four examples also passed through the actual HTTP service. During a real
32-question request, `/health` returned in **2.03 ms**. Body and prompt limits,
transport-ID invariance, and four-sample accounting were checked on that service.

## Scaling question count

This workload repeats simple balanced true/false color predicates over one tiny
state. It measures packing overhead, not the difficulty of many distinct coding
questions. There are three repetitions per size.

| Questions/request | p50 | Questions/s | Canvas width |
| --- | ---: | ---: | ---: |
| 1 | 226 ms | 4.48 | 16 |
| 8 | 405 ms | 19.80 | 48 |
| 16 | 634 ms | 25.21 | 96 |
| 32 | 1,079 ms | 29.60 | 192 |

The additional four-question measurement was **286 ms / 13.89 questions/s**.
All **183 scaling judgments** matched their simple expected predicates. Peak MLX
active allocation reached **18.68 GB** at 32 questions, including loaded weights.
This metric excludes allocator cache and is not process resident memory.

## Reproduce and interpret

Stop the server to avoid loading a second model, then run:

```sh
uv sync --frozen
uv run python -m diffusion_jev.benchmark --repeats 3 --output benchmark-results.local.json
```

Use `--model /path/to/model` for a different local directory. Report metadata
names the intended reference model separately from the selected directory; it
does not attest the provenance of an arbitrary checkpoint.

These figures describe one machine and one run. They do not measure long
contexts near the 8,192-token limit, simultaneous clients under load, adversarial
inputs, or power/thermal-controlled performance. The maximum measured scaling
prompt was **1,053 tokens**. More noise samples cost decoder time and do not prove
calibration. Recheck both quality and latency after changing prompts, candidate
order, canvas width, quantization, hardware, or the pinned runtime.
