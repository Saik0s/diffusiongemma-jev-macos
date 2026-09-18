# Research and semantic boundaries

This project explores typed decisions from DiffusionGemma on Apple Silicon. It
borrows Jev's API shape, not Jev's model, training method, calibration, or quality
claims. Source review date: 2026-09-18.

## The useful idea

A coding workflow often needs a bounded judgment rather than generated prose:
which file to inspect, whether a failure concerns authentication, or how strongly
a patch deserves review. The caller supplies the candidates; model probabilities
become inputs to ordinary application logic.

[TypeSafe's primitive documentation](https://docs.typesafe.ai/primitives) defines
three question types sharing a state. Question IDs identify responses but are not
semantic model inputs. Questions are independent; a judgment requiring an earlier
answer needs another request with that answer in its state.

## API shape and limits

The local endpoint is `POST /v1/systemone`. Requests contain `model`, `state`, and
a `questions` mapping. Each question supplies `type`, `instructions`, and, where
needed, `criteria`. Responses map the original IDs to typed `answers`.

- [Noul](https://docs.typesafe.ai/primitives/noul): a yes/no proposition. The
  answer's `noul` is the probability of yes, with no separate confidence field.
  Native JEV additionally accepts `criteria.true` and `criteria.false`; this
  local subset takes the proposition and any clarification in `instructions`.
- [Choice](https://docs.typesafe.ai/primitives/choice): named alternatives and
  descriptions in a criteria mapping. Return `choice`, complete `probabilities`,
  and `confidence`. This project supports **2–26** alternatives; TypeSafe
  documents a maximum of **255**.
- [Score](https://docs.typesafe.ai/primitives/score): **2–10** ordered descriptions.
  Return `score`, `legend`, complete `probabilities`, and `confidence`. Indices
  start at zero and `score = sum(index * probability)`.

## Direct diffusion reads

[vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250), open when reviewed,
demonstrates seeded diffusion canvases with bounded answer slots. Its mechanics
include a maximum denoising-step count, read-only output without a commit forward,
and requested label-token log probabilities. It deliberately leaves the public
question API to an external adapter. Each answer label must occupy one tokenizer
token so the canvas positions remain fixed.

The local design pins `mlx-optiq==0.5.12` and reads the seeded canvas directly.
It does not generate answer text. One decoder pass evaluates each noise sample;
the prompt encoder result is reused across samples. Original question IDs are
replaced with positional labels such as `q0` before prompt construction.

At an answer slot, the selected label logits are normalized over the permitted
aliases only:

```text
p_i = exp(logit_i - max(logits)) / sum_j exp(logit_j - max(logits))
```

This is a distribution conditional on the offered labels. It excludes probability
mass assigned to other vocabulary tokens. Repeated noise samples can characterize
variation in this readout; agreement does not establish correctness.

## Conditioning is not native Jev parity

Default **packed** mode places questions together in a prompt and canvas. Their
representations can interact, so adding a question may change another answer.
**Independent** mode isolates questions into separate model inputs.

Even independent mode treats Score as a joint choice over ordered descriptions.
Native TypeSafe Score instead evaluates each level without its index or neighboring
levels. Isolating questions therefore does not reproduce native Score conditioning.
See the [Score contract](https://docs.typesafe.ai/primitives/score).

The local confidence convention is `1 - H(p) / log(N)`, where
`H(p) = -sum(p_i * log(p_i))` and zero terms contribute zero. A uniform
distribution has confidence zero; a point mass has confidence one. This is a
concentration statistic, **not measured calibration**. TypeSafe's
[confidence documentation](https://docs.typesafe.ai/confidence) does not specify
its exact formula, so numeric parity is not claimed.

## A small optimization worth measuring

Only answer positions and permitted alias logits are needed. Select those decoder
hidden states and project onto the corresponding embedding rows instead of
materializing every canvas position's full vocabulary logits. Apply the same
normalization and any output transformations as the full projection.

Compare the optimized result with the full-vocabulary baseline using identical
inputs and noise: selected logits, probabilities, decisions, elapsed time, and
peak memory. This can reduce projection work and output allocation; it does not
remove decoder attention or expert computation. A benchmark must establish the
actual benefit on the target hardware.

## Useful examples and comparison projects

Failure triage combines focused Nouls with a Choice of failure causes.
File selection builds candidates from supplied file summaries. Patch review
combines an authentication finding with a review-priority Score. These are decision
helpers, not code generators or substitutes for tests.

[fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) demonstrates
a fourth use: separate keep-call and keep-result judgments while preserving
retained content verbatim and pairing results with their calls. Its documentation
also makes clear that probability is not proof a result can safely be deleted.
An educational adaptation should preview its decisions and retain the original
transcript on validation failure.

[NanoJev's contract audit](https://github.com/TianyuCodings/NanoJev/blob/main/docs/TYPESAFE_CONTRACT.md)
is useful for distinguishing a compatible response shape from compatible
conditioning. Its local schema and encoder have documented gaps, including
`boolean` versus `noul`, structured entries, confidence, and Score legends. Its
game-trained decision heads do not establish coding-task performance here.

## Runtime and licenses

The [OptiQ model card](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit)
explicitly warns that stock `mlx-lm` and `mlx-vlm` cannot load this diffusion
checkpoint; its documented route uses OptiQ's diffusion decoder. Generic hosting
snippets on the model page should not override that model-specific warning.

Model weights remain an external dependency with their own license and upstream
terms; this repository's license does not relicense them. The model card currently
labels the checkpoint Apache-2.0. Verify the selected revision's license before
redistributing weights.

[NanoJev](https://github.com/TianyuCodings/NanoJev/blob/main/LICENSE) and
[fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction/blob/main/LICENSE)
publish MIT licenses. Retain required notices if copying substantial portions.
This project does not train or fine-tune a model, reproduce Jev's training, or
claim native Jev decision quality.
