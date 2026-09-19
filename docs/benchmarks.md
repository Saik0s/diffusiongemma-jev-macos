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

## M4 Max memory-constrained run

The same harness and current source tree were also run on an **Apple M4 Max with
36 GiB** of unified memory and macOS **26.5.2**. Python and all seven recorded
package versions matched the M2 environment, and all **13 model files** were
verified before the run. The process completed only after enough application
memory had been freed; an earlier attempt was stopped by the normal-pressure
guard during model loading.

The completed run answered all **60/60** mixed-workload judgments and all scaling
judgments correctly. It measured **296 / 410 ms p50 / p95** and **7.98
questions/second** for the optimized path. Loading took **9.21 seconds**, and the
first 256-token request took **2.36 seconds**. At 32 questions, the median was
**1,220 ms** and throughput was **26.24 questions/second**. Selected-label and
full-vocabulary projection again had **0.0** maximum probability difference and
zero argmax disagreements over 60 paired judgments.

The guard sampled a **20.35 GB** peak process footprint and no increase in swap
use or swap-out bytes. MLX reported a **19.98 GB** load peak. These use different
memory measurements and should not be compared as the same quantity.

This is evidence that the existing MLX runtime works on the tested M4 Max with
adequate free memory. It is not a controlled M2-versus-M4 benchmark: the public
M2 data predates the current source fingerprint, and neither machine was held to
the same background load, thermal state, or operating-system version.

## Splash M4 versus oMLX M2 setup comparison

Splash 1.0 cannot load DiffusionGemma, so this separate test used its supported
**Qwen3.8-27B 4-bit** package. Both setups used target revision
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`, greedy sampling, and the same
messages. Splash ran on the **M4 Max, 32-core GPU, 36 GB**; oMLX 0.6.1 ran on the
**M2 Ultra, 60-core GPU, 64 GB** with its default BatchedEngine and DFlash
disabled. These are two complete deployments, not a same-hardware engine test.

The decode workload used three synthetic coding prompts, medium reasoning, and
a 1,024-token limit. Each prompt had one excluded warmup followed by three
measured exact replays. Every measured request reached 1,024 tokens.

| Measure, median of 9 | Splash / M4 Max | oMLX / M2 Ultra | Ratio |
| --- | ---: | ---: | ---: |
| Client decode throughput | **39.80 tok/s** | 33.23 tok/s | **1.20×** |
| First generated token | **0.375 s** | 0.674 s | **1.79× faster** |

Decode ranges were **26.52–41.37 tok/s** for Splash and **21.87–34.31 tok/s**
for oMLX. Per-prompt median Splash/oMLX ratios were **1.18×, 1.20×, and 1.04×**,
so the gain varied materially with the prompt and draft acceptance.

The long-prompt probe used three synthetic source prompts and one output token.
The engine-specific template paths produced 3,407 Splash tokens versus 3,435
oMLX tokens. The first uncached request favored oMLX: **19.03 seconds** median
to first token versus **21.45 seconds** for Splash. Exact replay strongly favored
Splash: **0.293 seconds** versus **7.93 seconds**, a **27.11×** latency ratio.
Splash reported 3,392 cached tokens; oMLX reported 2,048, so the replay result
measures their cache policies as well as cache-hit execution.

Both guarded runs stayed at normal memory pressure with zero swap growth. The M2
needed an unrelated background indexing service paused during model loading; it
was restored immediately afterward.
No prompt or generated text was persisted. The machine-readable aggregate is
[published with the benchmark artifacts](../benchmarks/splash-m4-vs-omlx-m2-2026-09-19.json).
Three repeats support medians and ranges, not reliable p95 claims. These Qwen
results do not establish a DiffusionGemma speedup because Splash has no
DiffusionGemma backend.

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
