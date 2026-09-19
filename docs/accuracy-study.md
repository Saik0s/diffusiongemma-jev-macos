# Accuracy-first evaluation

This study compares changes to the local adapter before changing its defaults.
Higher precision, more samples, and extra reasoning are hypotheses, not promised
improvements. The [Jev comparison](jev-differences.md) separates known differences
from native implementation details that remain undisclosed.

## What the experiments hold fixed

The retrieval data comes from the same pinned public sources as the
[historical retrieval pilot](retrieval-benchmark.md): 300 queries, a corpus of
1,000 functions, and 30 supplied BM25 candidates per query. Each passage is
limited to 2,000 characters. Baseline requests contain five passages at a time;
the questions ask whether each passage meets the query. A separate experiment
places all 30 candidates in one request, keeping passage limits and order fixed.

The first 50 source rows were already inspected in that pilot. They are excluded
from configuration selection and final testing. Sort the remaining query IDs by
`(SHA256("local-jev-accuracy-v1:" + qid), qid)` and take:

| Cohort | Queries | Queries with a labeled relevant candidate | Use |
| --- | ---: | ---: | --- |
| Development | 40 | 34 | Compare experimental settings |
| Validation | 60 | 50 | Select at most two finalists |
| Final test | 150 | 128 | Compare the frozen winner with baseline |

The split rule was frozen before running new configurations. Final-test results
must not select prompts, settings, or another winner. These are disjoint local
evaluation queries, not proof of absence from model training. Shared distractors
and sparse relevance labels also limit interpretation.

All-query top-1 accuracy is primary. Queries without a relevant candidate and
failed requests remain in its denominator. Eligible-only accuracy, candidate
coverage, nDCG@10, MRR@10, latency, and failures are reported separately. Ranking
ties retain the source BM25 order. Archived Jev measurements are a contextual
reference under a different protocol and undisclosed checkpoint.

Top-1 asks whether the first result is labeled relevant. nDCG@10 measures the
ordering of the first ten results; MRR@10 rewards an early first relevant result.
Higher values are better for all three metrics.

### Which build produced these numbers

Every number in this study was measured on prompt template
**`jev-local-prompt-v1`**, with state objects serialized using sorted JSON keys.
Two later changes affect the rendered prompt: Noul `criteria` are now rendered
when supplied, and state keys keep the caller's field order instead of being
sorted. Reordering keys changes the token sequence the model reads, so these
results describe the v1 template and are not automatically reproducible on
`jev-local-prompt-v2`. Re-measurement on v2 is outstanding work, listed with the
other unfinished items in the README. The `confidence` statistic also changed from entropy concentration to the
top-two margin; it never entered any accuracy figure here, which uses top-1,
nDCG@10, and MRR@10 only.

As the experiment runner gained fingerprint coverage, reports added the OptiQ
vision-sidecar hash and the `transformers` and `tokenizers` package versions.
For the original and current OptiQ baseline, all previously recorded model and
dependency hashes match. The omitted historical fields cannot be independently
verified from those earlier reports. A fresh
baseline control matched all 40 stored per-query metrics and tie counts exactly,
with zero failures. Reports omit full predictions and rankings, so this does not
prove those unrecorded outputs are identical. New runs require complete matching
fingerprints; inference-source changes require another current baseline control.

## Selection rules

A development candidate advances with at least **two net extra correct queries**,
or an **nDCG@10 gain of 0.02 without losing top-1 accuracy**, and no new failures.
This is a screening rule, not a significance claim. Plausible finalists are
rechecked with seeds 1 and 2; repeated seeds do not create more independent
queries.

These retrieval screens use an explicit width-32 answer canvas. Normal API
requests choose the width automatically from the actual answer template, rounded
up to a multiple of 16. Tokenizer-only checks confirm that these packed groups of
five Noul questions also resolve to width 32, with or without the empty thought
scaffold. That does not cover other request shapes: singleton coding judgments
resolve to width 16. Any general-default recommendation needs separate development
checks of the settings it will actually use, including coding false accepts.

Validation compares at most two finalists with the baseline. Choose by top-1,
then nDCG, then lower cost. Freeze that choice before final testing. The final
comparison reports paired query-bootstrap 95% intervals and an exact two-sided
McNemar test. A supported meaningful gain requires at least **five percentage
points of top-1 improvement**, a paired interval whose lower endpoint is above
zero, no decrease in mean nDCG, and no newly failed query. On 150 queries, the
top-1 threshold requires **at least eight net additional correct queries**.
Arithmetic comparisons allow 1e-12 rounding tolerance; the interval's lower
endpoint must be strictly positive. The full nDCG interval is reported, so an
unchanged or improved observed mean is not a claim of statistical noninferiority.
A smaller or uncertain result does not establish parity with Jev.

Before any validation, final, or coding outcomes, we clarified the original
qualitative nDCG rule conservatively and added a locked-coding deployment veto.
A general-default change requires **no newly observed false-accept case or newly
failed case** on both development and locked coding under the proposed deployment
settings. A false accept after a baseline failure also counts as new; improvements
on other cases cannot cancel it. A veto preserves the baseline default without
retuning from locked results. Passing these synthetic checks does not establish
production safety. Automatic-width deployment requires automatic-width coding
evidence; fixed-width results alone do not establish that behavior.

## Candidate families

The initial comparison crosses the original OptiQ checkpoint and genuine 8-bit
weights with packed and independent question modes. Every arm retains the same
five-passage state, width-32 canvas, seed, and question wording. Independent mode
changes the prompt and scaffold and reuses the first noise draw for each separate
question; packed mode draws noise in slot order. This compares complete protocols
and does not isolate the causal effect of sibling questions alone.
The checkpoints also have different conversion histories. Matching their tensor
names and tokenizer does not prove identical original source weights, so this
comparison cannot attribute a quality change uniquely to bit width.

Additional controlled candidates test float32 routing arithmetic, float32
selected-label projection, a width-256 canvas, four independent samples, genuine
BF16 router projections, a 256-token reasoning prepass, and a two-step diffusion
trajectory. A two-independent-sample control matches the trajectory's decoder
call count. Only measured promising changes justify combinations or extensions.

The all-30 experiment tests a common evidence context against six groups of five.
It does not add retrieval candidates. Both controls use packed mode, a width-256
canvas, and a 32,768-token prompt limit. Development prompts range from 8,959 to
21,341 tokens. Before evaluating accuracy, the longest prompt must pass a guarded
runtime check: correct mapping of all 30 questions, no input truncation, unchanged
prefill cache, repeat probabilities within 0.000001, and a maximum probability
difference of 0.03 between 512- and 256-token prefill chunks. These thresholds
were fixed before the pilot. Changed positions, aliases, and noise assignment
mean this also compares complete protocols rather than isolating context alone.

The router arithmetic policy preserves top-k selection and the loaded projection,
computes selected routing probabilities in float32, then rounds each weighted
expert contribution to the hidden-state dtype before reduction. It is a local
adapter tied to `mlx-optiq==0.5.12`; it does not edit installed dependency files.
Restoring BF16 router weights is a separate experiment that can recover values
lost during quantization. Casting existing quantized weights cannot do that.

The reasoning prepass uses the native entropy-bound sampler and 48 denoising
passes per 256-token canvas. Its token budget is 256 or 512; the larger budget
allows a second native block, with the completed first block encoded into the
existing causal cache. It stops retaining tokens at the thought-channel close
and appends one close marker when needed.
The structured read gets a fresh prefill of those exact tokens. No generated
thought text is logged or saved. Work counters include the extra generation and
prefill rather than calling this a single-pass decision. Prompt-token counts
include the 256 continuation encoder tokens when a second block completes;
initial native prefill timing excludes that continuation, whose time remains
inside the native generation and total reasoning timing.

The native stream's draft-progress option evaluates each denoising step's output.
Draft events are discarded without printing or retaining their text. The adapter
also evaluates cached attention state and self-conditioning inputs before each
decoder call. These boundaries keep the weights and sampler settings fixed while
splitting deferred GPU work into smaller evaluations.

The earlier progress-only path completed the 256-token screen but still failed
intermittently with Metal command-buffer error `0000000e` during a second block.
The additional input boundaries passed a 25-call replay, including both observed
failure points. Three paired original/repaired requests then had identical
retained token prefixes, work counts, and scores, with maximum score difference
**0.0**. Both probes completed under unchanged resource limits with normal OS
pressure and no measured swap-outs. Fresh full-cohort runs are required to judge
the repaired path's accuracy and longer-run reliability.

The same repair in the public adapter then passed all seven continuation-pilot
calls, including five with a second block. Repeated retained tokens and scores
matched exactly; sampled peak process footprint was 21.56 GB, with no measured
swap-outs and normal process release. This supports starting the full cohorts;
it does not replace their longer reliability and accuracy checks.

The two-step trajectory uses full-vocabulary self-conditioning, the native
48-step schedule's first transition, and final temperature-1 decision logits.
Fixed formatting tokens are restored between steps. That restoration makes it
a structured adaptation of native generation, not exact free-generation parity.

## Checkpoint and question-mode results

All four arms completed the same 40 development queries with zero failures.
None of the alternatives passed the development improvement gate, so subsequent
experiments use the original OptiQ checkpoint in packed mode.

| Checkpoint and mode | Top-1 correct | nDCG@10 | Median query time |
| --- | ---: | ---: | ---: |
| OptiQ, packed | 25/40 | 0.750793 | 11.930 s |
| Genuine 8-bit, packed | 22/40 | 0.717746 | 10.963 s |
| OptiQ, independent | 26/40 | 0.715580 | 46.060 s |
| Genuine 8-bit, independent | 23/40 | 0.696335 | 45.301 s |

The OptiQ independent arm gained seven correct queries and lost six relative to
baseline. Its one-query net gain came with lower ranking quality; the paired
top-1 interval was −15 to +20 percentage points. The 8-bit packed arm had two
wins and five losses; 8-bit independent had five wins and seven losses.
These small-cohort results do not establish that higher precision is inherently
worse or that independent questions cannot help another task.

Peak active MLX allocation was approximately 19.0 GB for OptiQ and 29.2 GB for
8-bit. Every arm used a 512 MiB allocator cache limit and diagnostics. Timing is
descriptive: a model download overlapped part of the 8-bit packed run. A separate
8-bit arithmetic diagnostic later stopped on an OS-pressure warning before
producing a result. Full BF16 loading also failed its separate pressure gate
before inference. These are observations under the recorded workload and guards,
not universal hardware capacity limits.

## Numerical precision results

Neither arithmetic change passed the development gate. Float32 routing changed
seven top-1 outcomes, gaining three and losing four. Float32 label projection
preserved every top-1 outcome and gained only 0.000447 nDCG, below the 0.02 gate.
Both completed all 40 queries without failures.
Restoring genuine BF16 router projections also failed the gate: it lost two
correct queries and gained none, while completing all 40 without failures.

| OptiQ packed policy | Top-1 correct | nDCG@10 | Median query time |
| --- | ---: | ---: | ---: |
| Native arithmetic | 25/40 | 0.750793 | 11.930 s |
| Float32 routing | 24/40 | 0.727578 | 10.870 s |
| Float32 label projection | 25/40 | 0.751240 | 10.790 s |
| Restored BF16 router projections | 23/40 | 0.731691 | 10.818 s |

These timing observations come from separate runs and do not establish a speed
advantage. Quality is the selection priority; none of these policies is promoted.

## Independent-read averaging

Four independent reads passed the development continuation gate: **27/40 correct**
against 25/40, with nDCG **0.772966** against 0.750793. It gained three correct
queries and lost one, with zero failures. Median query time was **13.671 seconds**,
compared with the baseline's 11.930 seconds.

This is a promising screen, not an established gain. The paired 95% interval for
top-1 change was **−5 to +15 percentage points**.

Eight reads, the API's existing maximum, then scored **28/40 correct** with
nDCG **0.786360** and zero failures. It gained four correct queries and lost one
against baseline. Its paired top-1 interval was **−2.5 to +17.5 percentage points**;
median query time was **14.996 seconds**. Relative to four reads, it gained one
correct query and lost none. These are development results, not an established
improvement or a selected default.

Plausible finalists still need the seed-2 recheck, the coding development screen,
and held-out evaluation. Multi-read seed windows overlap; they do not create
independent query observations. No sample setting beyond the API maximum is
introduced in this study.

## Reasoning development result

The first 256-token run with per-step evaluation completed all 40 queries without failures.
It scored **29/40 correct**, compared with baseline's 25/40, and nDCG **0.793854**
compared with 0.750793. It gained eight correct queries and lost four. The paired
95% interval for the top-1 change was **−7.5 to +27.5 percentage points**, so this
passes the continuation gate without establishing a reliable gain.

Median query time rose to **125.230 seconds**, compared with 11.930 seconds.
The run used 11,760 decoder passes across 240 requests and peaked at approximately
20.93 GB of active MLX allocation. OS pressure stayed normal under the watchdog.
Reasoning remains an experiment pending seed, coding, and held-out checks.
The 256-token limit forced closure in 191 of 240 requests. Together with the
development improvement, this motivates testing the 512-token extension after
a separate two-block runtime pilot. That pilot passed seven calls, including five
that executed a second block. The repeated input had identical retained token IDs
and scores, and all seven thoughts closed naturally. Sampled OS pressure stayed
normal, with no measured swap-outs. Full-cohort testing remains necessary to
determine whether longer reasoning improves accuracy.

The first 512-token cohort stopped after a runtime failure on its fourth query.
The failing group subsequently passed alone and in a seven-call replay of that
query. Replaying the preceding queries reproduced the same Metal error at an
earlier call, motivating the input-boundary repair above. The original failure
remains recorded. Both reasoning budgets require complete screens with the
repaired runtime; no partial accuracy result is used to select the policy.

The first repaired 256-token cohort was interrupted by the unchanged system
memory-pressure guard after nine successful queries. It recorded no model
exception, peaked at 21.67 GB sampled process footprint, and had no measured
swap-outs. The process exited, and pressure returned to normal. Its checkpoint
is retained. A separate fresh attempt, after three normal-pressure checks over
20 seconds, hit the same guard during loading at 19.38 GB sampled footprint.
An unrelated background indexing service was then paused to recover headroom;
the inference settings and resource limits remain unchanged. Recovery uses the
runner's fingerprint-checked resume path and preserves the original checkpoint
and process records. No accuracy decision uses the nine-query partial result.
Resumed-cohort latency describes completed query attempts and excludes aborted
work; interruptions are reported separately from completed-query failures.

The resumed repaired run completed all **40 queries with zero failures** and
scored **29/40**, with nDCG **0.793854**, matching the earlier reasoning result.
The original nine measurements were preserved exactly. Against the fresh
baseline it gained eight correct queries and lost four; the paired top-1
interval remains **−7.5 to +27.5 percentage points**. Median completed-query
time was **125.405 seconds**. The resumed process peaked at **21.59 GB** sampled
footprint, recorded no swap-outs, and released normally. It passes the
development continuation gate, not the final improvement criterion.

The repaired **512-token** screen also completed **40/40 without failures**, but
scored **28/40**, with nDCG **0.769491**, and took **210.449 seconds** per query
at the median. Against 256 tokens, it gained one correct query and lost two;
its paired top-1 difference was −2.5 percentage points, with a **−12.5 to +5.0**
interval. This does not establish a population-level disadvantage, but it fails
the predefined requirement to improve the previous best reasoning budget.
Although forced closure fell from **191 to 46 of 240 requests**, budget growth
stops here. The 256-token policy remains the reasoning candidate.

Reasoning and eight-read averaging each passed their standalone development
gate. Their predefined combination completed **40/40 without failures**, scoring
**28/40** with nDCG **0.786360** and median query time **129.196 seconds**. It
gained eight correct queries and lost five against baseline; the paired top-1
interval was **−10 to +25 percentage points**. It lost one correct query against
reasoning alone. Compared with eight reads alone, it gained four queries and
lost four, despite matching aggregate top-1 and nDCG.

The combination clears the baseline screening gate, but has not shown added
value over either component on seed zero. As declared before its results, it
joins both standalone candidates in paired seeds 1 and 2 and coding-development
checks. Query-level means across seeds 0, 1, and 2 will rank candidates for at
most two validation slots. These are development checks, not held-out results.

## Seed-1 rechecks and the interrupted seed-2 baseline

All four seed-1 development runs completed **40/40 queries with zero failures**.
The seed-1 baseline and eight-read runs each finished in one attempt. The
reasoning run needed four attempts: the unchanged OS-pressure guard stopped it
after 15, 18, and 19 successful queries, first at a sampled peak process
footprint of 21.65 GB with 438.9 MB of swap-out growth. Each resume verified the
exact earlier measurements before advancing the same checkpoint, and no partial
result was inspected to decide whether to continue. The completed attempt peaked
at 21.57 GB with 374.8 MB of swap-out growth; the other three seed-1 runs
recorded no swap-out growth. Resumed-cohort latency describes completed query
attempts only.

| Seed-1 policy | Top-1 correct | nDCG@10 | Median query time | Gained / lost vs seed-1 baseline |
| --- | ---: | ---: | ---: | ---: |
| Baseline, one read | 23/40 | 0.736059 | 10.946 s | – |
| Eight reads | 28/40 | 0.784627 | 14.964 s | 6 / 1 |
| Reasoning 256 | 24/40 | 0.718598 | 127.247 s | 6 / 5 |
| Reasoning 256 + eight reads | 27/40 | 0.780406 | 131.634 s | 6 / 2 |

Seed 1 changed the ordering seen on seed 0. Eight reads repeated its seed-0
aggregate exactly, 28/40, and its paired top-1 interval was **0.0 to +25.0
percentage points**, with an nDCG@10 interval of **+0.004867 to +0.094701**.
A lower endpoint of zero does not satisfy the final criterion, which requires a
strictly positive lower endpoint, and this is a development cohort. Reasoning
alone fell to 24/40 with nDCG below the seed-1 baseline; its top-1 interval was
**−15.0 to +17.5 percentage points**, and its exact McNemar p-value was 1.0.
The combination scored within one query of eight reads alone on both seeds,
with a **−2.5 to +25.0** interval, at roughly nine times the latency.

Across seeds 0 and 1, mean top-1 is 24.0/40 for baseline, 28.0/40 for eight
reads, 26.5/40 for reasoning 256, and 27.5/40 for the combination; mean nDCG@10
is 0.743426, 0.785493, 0.756226, and 0.783383. The frozen rule ranks candidates
by query-level means across seeds 0, 1, and 2, so this two-seed ordering is
provisional. It selects no finalist and changes no default.

The seed-2 baseline completed **10/40 queries with zero failures** before the
unchanged pressure guard stopped it at a 20.74 GB sampled peak footprint with
no swap-out growth. The guard responded to memory pressure from unrelated
applications, not to a model exception. Its ten-query checkpoint is retained,
and a fingerprint-checked continuation exists. The remaining seed-2 candidates,
the coding development screen, validation 60, final 150, locked coding 120, and
the HTTP deployment proof are unexecuted. Local benchmarks and model runs are
paused at the operator's request, and the unrelated background indexing service
paused earlier is now disabled at the operator's request. These interruptions do
not establish a model accuracy gain or loss.

## Two-step refinement result

The structured two-step trajectory reduced accuracy to **15/40**, compared with
25/40 for baseline. A matched two-independent-read control scored **24/40**.
All three runs completed without failures. The trajectory gained two correct
queries and lost twelve against baseline; its paired top-1 interval was
**−42.5 to −7.5 percentage points**. It failed the gate, so this family is closed.

| Policy | Top-1 correct | nDCG@10 | Median query time |
| --- | ---: | ---: | ---: |
| One structured read | 25/40 | 0.750793 | 11.930 s |
| Two independent full-projection reads | 24/40 | 0.749453 | 11.357 s |
| Two-step full-projection trajectory | 15/40 | 0.564701 | 11.695 s |

Both two-call policies used 480 decoder passes across 240 requests. These results
reject this particular structured adaptation; they do not establish that native
free-text diffusion or every possible refinement policy has the same weakness.

## Long-context feasibility

The width-256 control scored 25/40 with nDCG 0.731798, gaining five correct
queries and losing five. It failed the standalone improvement gate.

The longest-input pilot returned 30 valid answers at 21,341 prompt tokens.
Cache values and offsets remained unchanged; repeated reads agreed within
0.000001, and OS pressure stayed normal. However, changing prefill chunks from
512 to 256 tokens changed one probability by **0.080014944**, exceeding the
predeclared **0.03** limit. The all-30 accuracy experiment was therefore closed
without evaluating its 40-query quality. This identifies numerical sensitivity;
it does not establish that all-30 retrieval would be better or worse.

The first pilot attempt stopped on a harness error: native global and rotating
caches expose different metadata types. After repairing that check and adding
native-cache regressions, the second attempt used identical inputs and unchanged
thresholds. Both attempt records are retained.

## M4 hardware reproducibility check

After the M2 finalist recheck stopped on an operating-system pressure warning,
the current source, all 13 model files, package versions, inputs, and seed-0
baseline settings were copied to a 36 GiB M4 Max. A prospective bridge required
all 40 per-query ranking metrics and tie counts to match before any M4 candidate
could run. The bridge failed, so the queue stopped without evaluating a candidate.

The M4 baseline scored **22/40 top-1** with nDCG@10 **0.705085**, compared with
**25/40** and **0.750793** on the M2 control. Eleven queries changed ranking
metrics; two became top-1 correct and five became incorrect. Tie counts remained
unchanged. An excluded same-host diagnostic repeated the M4 baseline and matched
all 40 per-query metrics and tie counts exactly. The observed difference is
therefore stable across the two M4 runs, but its cause was not isolated beyond
the recorded chip and operating-system change.

Both M4 runs completed with normal memory pressure and zero measured swap growth.
Their median query times were **14.789** and **14.749 seconds**. These results
must not be mixed with the M2 seed results or interpreted as a controlled
hardware-speed comparison. A separate M4 study would need complete baseline and
candidate reruns under a protocol declared before inspecting candidate outcomes.

## Reproduce a retrieval cohort

Use a checkout with benchmark dependencies installed, and stop the local server
first so the experiment owns the only loaded model:

```sh
uv sync --frozen --extra benchmark
uv run python -m diffusion_jev.accuracy_benchmark \
  --model /path/to/diffusiongemma-26B-A4B-it-OptiQ-4bit \
  --model-revision 30f3c7c7746bf41cfd1a290155cc3b777ab588b9 \
  --split development --label optiq-packed32 \
  --canvas-length 32 --cache-limit-mib 512 --diagnostics \
  --output development.local.json
```

An immutable Hugging Face snapshot directory supplies its revision automatically.
For another directory, `--model-revision` records the claimed revision; loaded
file hashes provide the actual identity. A revision argument alone is not
verification that those files match a release.

The runner checkpoints after each query. Use `--resume` with the same command
after an interruption. It rejects changed model files, dependency source,
adapter source, options, or cohort IDs. Do not edit inference code mid-run and
expect its old checkpoint to resume under a new fingerprint. A failed request
remains in the report, and the command exits unsuccessfully.

For a paired comparison of complete matching cohorts:

```sh
uv run python -m diffusion_jev.accuracy_evaluation \
  --baseline baseline.local.json --candidate candidate.local.json \
  --output comparison.local.json
```

Inspect each module's `--help` for experimental flags. Two trajectory steps
require `--projection full`. Server reasoning is an explicit experimental startup
option and is disabled by default; trajectories remain benchmark-only.
`--cache-limit-mib` limits reusable allocator cache,
not total memory; it is not a process watchdog.

## Coding decisions and diagnostic limits

The separate synthetic suite contains **30 development judgments in 10 scenario
groups**, and **120 locked judgments in 40 groups**. Each scenario supplies a
Noul, Choice, and Score judgment about the same evidence. Locked cases span
triage, completion, patch review, and file selection. Expected answers and
rationales stay outside model prompts. These are correlated, short evidence
interpretation tasks, not 120 independent repository investigations.

Noul reports balanced accuracy, Brier score, and log loss; Choice reports accuracy
and log loss. Score reports argmax accuracy, normalized error of the expected
level, and normalized ranked probability score (RPS), the mean squared error of
the cumulative distribution at each boundary. Lower losses are better. Failed
cases remain in denominators with explicit worst-scale penalties; they are
separately counted, and those penalties are conventions rather than predictions.

Completion and patch acceptance are explicitly annotated. Choice uses its winning
label; Score uses its most probable level, with level 2 accepting completion and
level 0 accepting a patch. Mixed-orientation Noul questions have no implicit
acceptance decision. False-accept rates divide observed false accepts by valid
responses that required rejection. Reports also show failures and all cases
requiring rejection. Failures are not counted as correct rejections; the rate
describes valid responses only and must be read alongside the failure count.
Development coding decisions screen finalists for new false accepts before
freezing the winner; locked cases are evaluated only afterward.

Reports retain identifiers, hashes, numerical measurements, and configuration.
They do not contain code passages, prompts, model distributions by answer label,
or generated thoughts. Noul Brier/log loss and Choice loss describe these
explicitly labeled synthetic cases; sparse retrieval labels do not establish
probability calibration.

Run the development coding suite with the same model and experimental settings:

```sh
uv run python -m diffusion_jev.coding_benchmark \
  --model /path/to/diffusiongemma-26B-A4B-it-OptiQ-4bit \
  --model-revision 30f3c7c7746bf41cfd1a290155cc3b777ab588b9 \
  --split development --canvas-length 32 --cache-limit-mib 512 \
  --output coding-development.local.json
```

Use `--split locked` only after selecting the configuration. These commands do
not call a hosted model or paid Jev API.

Optional retrieval diagnostics report the total full-vocabulary probability mass
assigned to allowed labels and how often the unrestricted winner lies outside
those labels. Conditional label probabilities always sum to one, even when
allowed-label mass is small. Neither large mass nor concentrated probabilities
prove the decision correct.

For selected-label decisions, diagnostics use a separate native full-vocabulary
projection at answer rows. Different matrix shapes and precision paths can round
differently, so that diagnostic is not an exact reconstruction of the selected
projection's full-vocabulary distribution. Diagnostic work is included in the
reported latency and memory; compare timings with that setting held fixed.
