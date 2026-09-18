# How the local decision engine works

This page explains the implementation and its limits. You can use the server without knowing these details.
For the user-facing concepts, start with [Jev concepts for developers](concepts.md).
Source review date: **September 18, 2026**.

## The interface and the model are separate choices

A decision API defines how software supplies evidence and receives answers.
Training determines how well a model makes those decisions.
This project implements a local interface inspired by [TypeSafe's primitives](https://docs.typesafe.ai/primitives), using an existing DiffusionGemma checkpoint.
It does not reproduce Jev's training, calibration, or quality.

That distinction matters when reading a benchmark.
A result from hosted Jev, NanoJev, or another local adapter is not a result for this model.
The [public retrieval evaluation](retrieval-benchmark.md) therefore labels archived Jev measurements separately.

## From JSON to a short answer workspace

The implementation has four stages:

1. **Read the input.** Serialize the state and question descriptions, then convert the text into tokens, the model's numbered text pieces.
2. **Prepare answer positions.** Create a short sequence called a *canvas*, with fixed labels and a position for each answer.
3. **Evaluate the canvas.** Run one diffusion decoder pass for each requested noise sample.
4. **Return numbers.** Read the model's preference for allowed answer labels and construct the JSON response in Python.

The model does not generate response JSON or an explanation.
Your original question IDs identify answers in the response, while positional aliases such as `q0` identify slots internally.
Choice candidate names and descriptions are still part of the model's input.

The design draws on the [vLLM structured-read proposal, PR #57250](https://github.com/vllm-project/vllm/pull/57250).
That proposal separates low-level diffusion reads from a public question API.
Our adapter provides that question API using the OptiQ runtime on Apple Silicon.

## Why each allowed answer must fit one token

The canvas needs a stable answer position.
The adapter maps permitted answers to short internal labels and checks their tokenization.
Alternatives must differ at exactly one token position, or the request cannot use this readout safely.

Only the answer positions receive seeded random starting values.
Surrounding formatting stays fixed. The default canvas is the smallest multiple of 16 that fits the answer scaffold.
Explicit canvas lengths range from **16 to 256**, in steps of 16.
A different canvas width can change answers, so it is part of the benchmark configuration.

## How preferences become probabilities

At each answer position, the model produces numerical preferences called *logits*.
The adapter selects only the permitted labels and normalizes them:

```text
probability(label i) = exp(logit i) / sum(exp(logit j) for each allowed label j)
```

The implementation subtracts the largest logit before exponentiation for numerical stability.
The resulting probabilities add to one **among the labels you offered**.
They exclude every other word or token the model might prefer.

This guarantees a valid distribution over your choices. It does not establish that the choices are correct or well calibrated.
For example, two poor alternatives still divide the full probability between them.

## What extra samples do

`options.samples` accepts **1–8** independent starting-noise draws.
The server reuses the already-read input, evaluates each draw once, and averages the resulting probabilities.
These are not successive steps refining one answer.

`options.seed` makes the starting noise repeatable.
Agreement across samples measures stability under this readout, not truth.
Exact floating-point results may still vary with hardware or runtime versions.

## Packed versus independent questions

Default **packed** mode puts several questions in the same input and canvas.
This saves repeated input processing, but the questions can influence each other.
Adding, removing, or reordering a question can therefore change another answer.

**Independent** mode builds a separate model input for every question.
It costs more time because the state is read again for each question.
It still evaluates the alternatives within a Choice or Score together.

Native Jev documents independent question evaluation and different Score conditioning.
Local independent mode therefore narrows one difference; it does not provide native Jev parity.
See [the official state contract](https://docs.typesafe.ai/concepts/state) and [Score contract](https://docs.typesafe.ai/primitives/score).

## Exact response differences from native Jev

- **Noul:** the local schema takes the proposition in `instructions`. Native Jev also accepts separate `criteria.true` and `criteria.false` descriptions.
- **Choice:** this implementation accepts **2–26 alternatives**. TypeSafe documents up to **255**. The returned winner has the highest supplied-label probability.
- **Score:** this implementation accepts **2–10 ordered descriptions** and evaluates them together. It returns the probability-weighted zero-based index and a legend.
- **Confidence:** local Choice and Score use the formula below. TypeSafe's public documentation does not specify that exact formula, so numeric equivalence is not claimed.

For `N` alternatives and probabilities `p`, the local concentration statistic is:

```text
H(p) = -sum(p[i] * log(p[i]))
confidence = 1 - H(p) / log(N)
```

Zero-probability terms contribute zero.
An even distribution produces confidence zero; all weight on one alternative produces confidence one.
This quantity measures how concentrated the answer is, not its empirically measured reliability.
See the [official confidence guide](https://docs.typesafe.ai/confidence) for the native product's explanation.

## Where the optimization saves work

A general output calculation can score the entire vocabulary at every canvas position.
This API only needs the answer positions and their allowed labels.
The default path selects those positions and the corresponding output-weight rows before calculating the scores.

Both paths apply the pinned model's exact output transformation, including its float32 softcap.
The `projection: "full"` reference path remains available to compare numerical results.
On the earlier measured suite, the compact full path and selected-label path returned identical selected probabilities.
See [the measured comparison](benchmarks.md).

Shortening the answer canvas produced most of that suite's speedup.
Selecting fewer output scores produced a smaller additional gain.
Neither change avoids reading the input or doing the model's internal computation, so longer real-code requests can take much longer.

The server keeps the model loaded and reuses input processing across samples within one request.
It does not keep a cross-request prompt cache.
One worker thread loads and calls the model because MLX's lazy GPU values can depend on thread-local streams.

## Why the runtime is pinned

The [OptiQ model card](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit) warns that stock `mlx-lm` and `mlx-vlm` cannot load this checkpoint.
We pin **mlx-optiq 0.5.12**, **MLX 0.32.2**, and the **Hugging Face downloader 1.32.0**.
The checkout's lockfile fixes the remaining package environment.
The engine uses a small internal decoder interface, so runtime upgrades require a real-model recheck.

`start` downloads an immutable model revision and checks a bundled size/hash manifest.
Existing user-managed model directories receive structural checks rather than a claim of verified provenance.
The [setup guide](setup.md) explains selection and verification behavior.

## What would establish stronger claims?

For a useful application claim, evaluate held-out cases from that workflow and record what happens after the decision.
Retrieval quality alone cannot show that a coding agent fixes more issues or safely removes conversation history.

For calibration, compare predicted probabilities against observed labels across enough examples, including hard and ambiguous cases.
For performance, report input size, hardware, queueing, warmup, and the actual end-to-end operation being timed.

The [community guide](community.md) lists practical designs and external benchmarks worth studying.
Model weights and datasets retain their separate upstream licenses; this repository's MIT license applies to its own code.
