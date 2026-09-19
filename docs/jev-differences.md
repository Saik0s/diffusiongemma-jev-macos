# Why this local adapter differs from Jev

The largest unresolved difference is the model: this project uses a general
DiffusionGemma checkpoint, while TypeSafe trains Jev for decisions. Runtime
improvements cannot reproduce undisclosed training or establish equal accuracy.

This inventory covers the public contracts, the local implementation, a live
probe of the hosted model through OpenRouter on **September 19, 2026**, and a
source review of Featherless's open Simple-JEV reimplementation at the same
date. The hosted model resolved to **`typesafe/jev-1.13-20260917`**. Native
implementation details that TypeSafe has not published remain unknown.

Each statement below is labeled. **Observed** means we measured it in the
hosted probe or a local run. **Documented** means a public source describes a
contract or implementation. **Inferred** means consistent with observations but
not confirmed by a source. **Unknown** means no reviewed source settles it.
None of these labels is a measured causal effect on accuracy.

## What the hosted probe measured

The probe sent **42 requests** to `POST https://openrouter.ai/api/alpha/decisions`
with `model: typesafe/jev-1.13`, using synthetic content written for the probe.
The sanitized requests, answers, usage, and latencies are
[published with the benchmark artifacts](../benchmarks/openrouter-jev-1.13-probe-2026-09-19.json).
Request and account identifiers were removed. Three scenarios anchored it:

- **Checkout telemetry.** State is an ordered list of timestamped form events.
  Eight Noul questions with explicit true/false criteria ask about promo
  trouble, imminent quit, card testing, readiness to pay, shipping uncertainty,
  whether live chat would help, comparison shopping, and returning customers.
- **Grid chase.** One Choice question picks `down` or `right` on a described
  6×6 board with walls and distances, sent at a cadence target of two to three
  decisions per second.
- **Feed moderation.** One Noul per post against the rule “Hide posts written
  to make me angry or afraid so I react and share,” hidden at 0.5.

Because local model runs are paused, no local numbers were produced for these
scenarios. Local behavior below comes from source and existing recorded
artifacts, not from a paired local run.

| Scenario result, observed | Hosted Jev 1.13 |
| --- | --- |
| Struggling shopper: promo 0.98, help 0.79, quit 0.74; both product thresholds at 0.7 fire | Consistent with the scripted signals |
| Ready shopper: ready 0.96, returning 0.92, all other Nouls at most 0.07 | Consistent |
| Card-testing pattern: fraud 0.90, help 0.85, ready 0.28 | Fraud detected; the high help value is arguable |
| Grid: 8/8 boards chose the legal closer move; blocked or off-board moves received 0.00 to 0.19 | Correct on every synthetic board |
| Feed: bait posts 0.96 to 0.98, benign posts 0.02 to 0.03, in single and batched requests | Wide separation around the 0.5 threshold |

These are eight, eight, and six hand-written cases. They demonstrate API
behavior and decisiveness; they are not an accuracy benchmark.

## Wire format

**Endpoint and envelope, observed and documented.** All 40 successful hosted
responses in the probe carried `model`, `provider`, `answers`, `usage`, and an
opaque `id`. The stored artifact strips `id`, the account identifier, and the
response headers, so the shape is recorded in its `endpoint` block rather than in
the sanitized records. `usage` has `input_tokens`, `output_tokens`,
and `cost`; cost equaled input tokens at OpenRouter's listed $0.042 per million,
and output tokens were nonzero but free. The native `POST /v1/systemone` response
has `model`, `answers`, and a local `usage` with token, pass, timing, and memory
counters instead. Both key answers by the caller's question IDs with the same
`type`, `noul`, `choice`, `probabilities`, `confidence`, `score`, and `legend`
fields. See the [native API contract](https://docs.typesafe.ai/api).

**Compatibility route, local implementation.** The server also answers
`POST /api/alpha/decisions` with the hosted envelope: `id`, `model`, `provider`,
`answers`, `usage`. The local `id` is a fresh random identifier per response, kept
only in that response, because the server writes no request log to correlate it
with. That route reports `provider: "local"`, `cost: 0.0` because
nothing is billed, and `output_tokens` equal to the number of questions, since
one verified label position is read per question rather than a hosted structure
around it. It accepts only `model: "diffusiongemma-local"`. A client written for
hosted Jev therefore needs exactly one changed field, the model name, plus a base
URL; this is a minimal client adapter, not drop-in SDK compatibility, and the
answers come from a different model.

**Validation, observed.** OpenRouter rejected an unknown question type and a
numeric state root with **HTTP 400** and a discriminated-union message listing
`noul`, `choice`, `score`, or `string`, `record`, `array`. TypeSafe's own
documentation lists 422 for validation. The native local route returns 422 and
accepts any JSON root for `state`, so a numeric or boolean root that works there
fails on the hosted API. The compatibility route matches the hosted behavior
instead: it rejects a scalar `state` root and reports failures as HTTP 400 with
`{"error_code", "error_summary"}`. Neither local shape echoes the request, because
a rejected payload can contain private source code.

**Rubrics and schemas, observed and documented.** The hosted API accepted
object-valued `instructions` and Noul `criteria`, `null` Choice descriptions,
and a Noul with no criteria at all. Local instructions and descriptions are
required strings. Local Noul now accepts the same optional `criteria` rubric and
renders both sides in the prompt; unlike the hosted API it requires `true` and
`false` together, because a one-sided rubric has no defined meaning here. Local
Choice accepts
**2–26 alternatives**, native Choice up to **255**; local requests accept at
most **32 questions**. In the one case tested, removing the Noul criteria did
not change the hosted answer (0.98 either way); that is one case, not a
general finding. See the [advanced input guide](https://docs.typesafe.ai/primitives/advanced)
and [Choice contract](https://docs.typesafe.ai/primitives/choice).

**Context limits, documented.** OpenRouter lists a **32,000-token** context for
this endpoint. TypeSafe documents **64k tokens for state plus all questions**
and **32k for state plus the longest question**. The local default is **8,192
tokens per rendered prompt**, checked per prompt in independent mode. See
[current native models](https://docs.typesafe.ai/models).

## Probability meaning

**Decision training, documented difference.** TypeSafe describes reinforcement
learning for calibrated decisions, or RLCD. This adapter reads allowed-token
preferences from a general model without reproducing that training. The reward,
loss, training data, backbone, output heads, and calibration procedure are not
publicly specified. This is a plausible major source of the quality gap, but its
contribution cannot be separated from the others here.
See the [official training primer](https://docs.typesafe.ai/introduction/machine-learning-primer).

**Precision and extremes, observed.** Every hosted value carried at most two
decimal places, and confident answers were exactly `0` and `1`. Local answers are
unrounded float32 softmax outputs that never reach exactly 0 or 1. Application
code that compares against thresholds is unaffected; code that takes logs or
ratios of hosted probabilities must handle exact zeros.

**Repeat variation, observed.** Three identical hosted checkout requests differed
by at most 0.02 in any Noul, and eight identical grid requests returned
down-probabilities between 0.86 and 0.89. Hosted answers are therefore not
bit-deterministic. Locally, a fixed `options.seed` reproduces the answer-slot
noise on one machine, and the M2 and M4 runs in the [accuracy study](accuracy-study.md#m4-hardware-reproducibility-check)
show that identical inputs can rank differently across chips.

**Conditional probabilities, documented local behavior.** Local answers normalize
only the permitted token scores, so every answer can look decisive even when the
model prefers a token outside the permitted labels. Diagnostic label mass
measures probability retained by those labels. Whether the hosted model
normalizes over a fixed label set the same way is unknown.
The local output probabilities have not been shown to be calibrated.

**Absolute versus relative judgments, documented native distinction.** Jev's
Noul assesses a proposition; Choice compares the alternatives offered. Native
Noul and a yes/no Choice can disagree, and separately worded positive and
negative Nouls need not add to one. Our Noul uses a yes/no token comparison,
while Choice uses letter labels. See [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

**Confidence, inferred native formula and known local formula.** On all 21
hosted Choice and Score answers in the probe, `confidence` equaled the gap
between the two largest probabilities within 0.01 rounding, for example
probabilities 0.81/0.19 with confidence 0.63 and 0.02/0.98 with 0.96. TypeSafe
documents confidence as derived from the probabilities without publishing the
formula, so a top-two margin is an inference, not a confirmed contract. Local
Choice and Score now return that same top-two margin, replacing the earlier
entropy concentration `1 - H(p)/log(N)`, which gave 0.30 where the hosted API
gave 0.63 for 0.81/0.19. Thresholds tuned against the entropy value do not carry
over. Simple-JEV uses the largest probability instead, so its confidence still
differs from both. Noul has no confidence field in any of them. The local
probabilities feeding the formula come from a different model, so matching the
statistic does not make the numbers interchangeable.
See the [native confidence guide](https://docs.typesafe.ai/confidence).

## Question and answer conditioning

**Question independence, documented and observed.** Native Jev evaluates
questions independently against shared state. In the probe, `promo` scored 0.99
alone and 0.98 inside the batch of eight, `help` scored 0.79 both ways, and
reversing the order of all eight questions changed no answer by more than
0.01, within the repeat variation above. Local packed mode puts all questions in
one prompt and answer canvas, so questions and their positions can affect one
another. Local independent mode creates separate prompts and reads, retaining
the same state but processing it again for every question.
See the [native primitives contract](https://docs.typesafe.ai/primitives).

**Score levels, documented difference.** Native Score evaluates each description
without its index or neighboring descriptions. The local adapter presents all
levels together as letter alternatives, in either question mode. Both return a
normalized distribution and its zero-based weighted mean; the hosted three-level
probe returned probabilities 0/0.02/0.98 with score 1.98. Native conversion
from independently assessed levels to that distribution is undisclosed.
See the [Score contract](https://docs.typesafe.ai/primitives/score).

**Labels, order, noise, and canvas, known local dependencies.** The adapter fixes
answer formatting, initializes answer slots with seeded random tokens, then
reads one decoder pass per sample. Answer labels must occupy one token. Their
wording, position, neighboring slots, and canvas width can affect the result.
Additional samples average independent initializations. Native label encoding,
sampler, and forward-pass count are unknown.

**Reasoning, local experiment.** The default one-step path does not generate
intermediate reasoning. An experimental prepass, enabled explicitly in benchmarks
or at server startup, retains the exact generated token prefix before the
structured read. On the development cohort it scored 29/40 on seed 0 and 24/40
on seed 1, against 25/40 and 23/40 for baseline, at roughly ten times the
latency; see the [local accuracy study](accuracy-study.md). Whether the hosted
model performs any internal deliberation is unknown; its 0.26 to 1.5 second
responses leave little room for long generation.

## Precision and runtime arithmetic

**Weights, known local precision and unknown native precision.** The original
local checkpoint is mixed-precision OptiQ quantization. Genuine 8-bit and BF16
checkpoints allow a local precision comparison; the 8-bit development run did
not improve accuracy. OpenRouter lists the hosted quantization as unknown.
Jev's weight precision, parameter count, and accelerator hardware are undisclosed.

**Output projection, measured numerical sensitivity.** Selected-label lookup and
full-vocabulary projection are algebraically related, but may use different
dequantization and matrix kernels. The float32 softcap must be preserved in
every path. Equality on the earlier small synthetic suite does not imply
equality for every prompt or checkpoint.

**Routing and prefill, source-backed numerical sensitivity.** The pinned runtime
rounds router probabilities and expert contributions at specific dtype
boundaries, and attention and quantized matrix implementations depend on input
shape. Changing the prefill chunk from 512 to 256 tokens moved one long-context
probability by 0.08. These are controlled precision experiments, not evidence of
a general cache bug.
See the pinned [attention implementation](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/scaled_dot_product_attention.cpp)
and [quantized dispatch](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/quantized.cpp).

## Speed, batching, and cost

**Latency, observed with a scope.** Over 40 successful hosted requests from an
Apple Silicon Mac on a residential connection, the median round trip was
**0.334 seconds**, p90 **0.469 seconds**, minimum 0.261, maximum 1.500. Eight
sequential single-question grid requests had a median of 0.347 seconds, about
**2.9 decisions per second** sequentially; four concurrent requests each
completed in 0.30 to 0.36 seconds. These figures include network transport and
OpenRouter's proxy, and the request bodies were small: 296 to 1,399 input
tokens. They are not a matched local hardware benchmark. For scale, the local
adapter's small synthetic suite measured **291 ms median** on an M2 Ultra
without network, and real 30-function retrieval reranks took **10.8 to 11.9
seconds** median per query across the accuracy study's baseline runs.

**Shared-state batching, observed and documented.** Eight checkout questions cost
1,399 input tokens against 938 for one, so each added question cost about 66
tokens rather than another copy of the state. Six posts in one state with one
Noul each cost 1,337 input tokens instead of 2,549 across six single requests,
and reproduced the single-request answers within 0.02. Local packed mode shares
a prefill in the same way but allows question interactions; local independent
mode repeats prefill. Native attention masks, cache reuse across requests, and
scheduling are unknown. The official cookbook's **0.27 versus 2.71 second**
batched-versus-sequential figure covers 13 questions over about 54k characters.
See the [parallel questions cookbook](https://docs.typesafe.ai/cookbooks/parallel_questions).

**Concurrency, known local limit.** The server has one inference worker and
eight waiting slots, and model construction and inference share one thread
because MLX lazy buffers can depend on thread-local streams. Hosted concurrency
and hardware are not disclosed, so hosted latency does not isolate an
architecture advantage against this Mac.

**Retrieval comparisons have a scope too.** The archived Jev results use a
different question batch size and an undisclosed model revision. Our frozen
local comparisons can identify improvements over the local baseline; the
archive provides context, not a controlled claim of closing a particular
percentage of Jev's advantage. See the [retrieval evaluation](retrieval-benchmark.md).

## Matched suite, replayed against both sides

The probe's own requests are frozen as a replayable suite,
[`benchmarks/matched-suite-2026-09-19.json`](../benchmarks/matched-suite-2026-09-19.json):
38 comparable cases, each carrying the exact state and questions sent to hosted Jev, the
hosted answer, its token usage, and its measured round trip, plus 4 validation cases the
local schema rejects. Every state is synthetic. Neither side receives an `options` block,
so each uses its own documented defaults.

**Two of those validation cases are a parity gap, not shared strictness.** Hosted Jev
answered a structured Noul `criteria` object and `null` Choice descriptions; the local
strict schema rejects both. They carry `hosted_accepted: true` and their hosted answer,
and the report counts them as `hosted_only_inputs`. They are excluded from the answer
comparison because there is no local answer to compare against. Schema 2 of the suite
moved them out of `cases` for that reason; no request, label, or recorded answer changed.

Labels exist for the 20 grid cases only, and they are derived mechanically from each
state and the question's own rule: a legal move with the smallest remaining Manhattan
distance. Ties accept either key. **Only 6 of those 20 states have a single correct
key**; the rest are genuine ties that any answer passes, so grid accuracy is a weak
signal rather than a quality verdict. The checkout and feed cases have no ground truth
and are compared for agreement and stability only, because hosted output is not a label.
No case ranks a list, so nDCG and MRR are undefined here and are left unreported rather
than filled in with a placeholder.

Replay it against a running local server with:

```bash
uv run jev-local benchmark-matched --output /tmp/matched-local.json
```

That writes per-case agreement, the labelled score for both sides, and three separate
timings: local round trip, local `usage.total_ms` inference time, and the hosted round
trip, which includes network transport and is never subtracted from the local numbers.

### Measured result, 2026-09-20

One paired run of each local profile plus a fresh hosted run, all issued from the same
Apple M4 Max with 36 GiB of memory on one residential network, so transport and workload
origin are common to every column. Full record:
[`benchmarks/matched-comparison-2026-09-20.json`](../benchmarks/matched-comparison-2026-09-20.json).

| | local, `fast` | local, `accuracy` | hosted `jev-1.13` |
| --- | --- | --- | --- |
| cases answered | 38, 0 failures | 38, 0 failures | 38, 0 failures |
| labelled accuracy | 20/20 | 20/20 | 20/20 |
| top-choice agreement with hosted | 20/21 | 20/21 | — |
| round trip, median | 0.336 s | 0.384 s | 0.326 s |
| round trip, p90 | 0.901 s | 0.991 s | 0.511 s |
| server inference, median | 335 ms | 383 ms | not reported by the API |
| peak MLX memory | 18.68 GB | 18.68 GB | — |
| cost for the 38 calls | 0 | 0 | 0.0011 USD |

**The accuracy profile bought nothing measurable here.** Eight reads produced the same
20/20, the same 20/21 agreement, and the same largest answer gap as one read, while
adding about 14 percent to median round trip. That is a result about this suite, not a
general finding, and it is the reason the shipped default stays `fast`.

**Hosted is tighter at the tail.** The medians are within 60 ms of each other, but hosted
p90 of 0.511 s beats local p90 of 0.901 s. Local p90 is dominated by the larger batched
requests and by first-request warm-up: `checkout/struggling` is the first case of each
run and took 4.005 s on the fast run against a 0.336 s median.

**The one disagreement is `probe/score_three_levels`.** Hosted scored 1.98 of 2 with 0.96
confidence; local scored about 1.02, one level lower, on a state whose text reads as a
third escalation. Hosted's reading is the more plausible one. It is a single case, so it
is evidence of a difference in calibration, not a measured quality gap.

**Hosted answers were stable across the two days.** Replaying the same 38 requests on
2026-09-20 reproduced all 21 comparable winning labels from 2026-09-19, with a largest
numeric drift of 0.03 on probabilities the API reports rounded to two decimals.

This is one run per profile on one machine on one day. No trial was repeated, so no
confidence interval is claimed on any latency number, and **no accuracy winner is claimed
at all**: at 20/20 against 20 labels where 14 accept either key, this suite cannot
separate the two systems.

## Injection and ambiguous evidence, shared risks

Local instructions tell the model to treat state as data; that instruction is
not a security boundary. TypeSafe documents susceptibility to adversarial state,
irrelevant context, indirection, numerical tasks, and conflicting criteria.
Neither interface makes these problems disappear. Score interpolation does not
recover an exact numeric quantity. See [native limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13).

## Simple-JEV, an open reimplementation on autoregressive models

[Simple-JEV](https://github.com/featherless-ai/simple-jev), reviewed at commit
`0dd5396ffce671ab7c4bfc031506d8e558cf8d23`, serves the same request shape from
open Hugging Face models. It is a third, independent design point, and its
README states that it does not reproduce TypeSafe's training or establish
equivalent accuracy. Everything below is source-backed observation of that
repository, not a measurement of its quality.

**License, observed.** The repository contains no license file, no license
field in `hf-server/pyproject.toml` or `demos/jevpilot/package.json`, and no
copyright headers. Its own JevPilot notes say no root code license was present
upstream either. **No code, prompt text, or test text from Simple-JEV was copied
into this project.** The points below are ideas and contracts described in its
`common/PROMPT_STRUCTURE_V1.md`, which frames itself as a language-independent
specification for reimplementation.

**How it reads a decision.** One prompt per question, ending in an incomplete
assistant JSON prefill; the next-token logits for the permitted single-token
labels are softmaxed without temperature (`common/response_scoring.py`). All
questions are briefed before the context so that the rendered token prefix is
shared, the prefix is prefilled once, and per-question suffixes run as a batch
from copies of that KV cache (`hf-server/hf_server.py`). This adapter instead
denoises one canvas holding all packed answer slots in a single diffusion pass.
Both approaches avoid generating and parsing text.

**Primitive semantics differ from both Jev and this adapter.** Simple-JEV's Noul
is a nine-way rating over the digit tokens 1–9 whose expected value is mapped
to **0.01–0.99**; it is explicitly not a yes/no softmax. Its Choice and Score
confidence is the **largest probability**. Score is the weighted mean of
zero-based indices, as here. It caps Choice and Score at **50** alternatives and
allows **256** questions per request, with structured or null instructions and
criteria and an optional `messages` chat context instead of `state`.

**Contract practices worth adopting locally.** Each label is checked for
single-token stability at the fully rendered answer boundary, not in
isolation, and duplicate token IDs are rejected. The prompt template carries a
version pinned by tests, chosen at the integration boundary rather than by a
request field. Usage counts unique prompt-token prefixes. Diagnostics are gated
behind an environment flag, with a test proving public answers do not change.
This adapter already verifies single-token labels and pins its prompt through
tests, and now carries a prompt template version chosen at the integration
boundary (`jev-local-prompt-v2`, reported by `/health`) so recorded accuracy
numbers can be tied to the wording that produced them. Unique-prefix usage
accounting remains an open idea. Sharing one prefill across per-question reads in
independent mode is also open: packed mode already shares a prefill, and the
change would move the numerically sensitive prefill path described above.

**Decision decomposition, inspiration.** Its driving demo splits a control
decision into a binary `drive`/`stop` Choice and a conditional path Choice, and
multiplies the probabilities, so that many similar paths do not dilute the
binary preference. It also resolves single-candidate questions in client code
and sends only legal moves as candidates. These are client-side policies this
project's examples could adopt without any server change.

**Training path.** Its RFDT scripts fine-tune the same allowed-label logits
with a soft cross-entropy loss, optionally from teacher-labeled probabilities.
That is a route toward decision-specific training for an open model; it has no
published accuracy results, and this project has not attempted it.

## What remains unknown

- Jev's architecture, parameter count, precision, sampler, forward-pass count,
  and serving hardware.
- Whether hosted probabilities are normalized over a fixed label set, and how
  Score levels assessed independently become one distribution.
- The exact hosted confidence formula; the top-two margin fits all 21 probe
  answers but is unconfirmed.
- How hosted accuracy compares with this adapter on the local development
  cohort, because the local seed-2 recheck and held-out evaluations are
  unfinished and no paired hosted run of that cohort exists.
