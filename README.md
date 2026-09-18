# DiffusionGemma local decisions

A small JEV-style decision API for Apple Silicon. Give it a state and questions;
receive Boolean probabilities, choices, or ordered scores. It runs the existing
[OptiQ DiffusionGemma checkpoint](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit)
locally through MLX, with no hosted inference service.

This is a research and education project. Probabilities are **uncalibrated** and
the API is a documented subset of JEV, not a replacement for its trained model.
Use the examples to explore decisions inside coding workflows; keep tests and
human review as the authority for consequential actions.

## Run

Requires **macOS on Apple Silicon**, [uv](https://docs.astral.sh/uv/), and the model
already downloaded. Tested on an **M2 Ultra with 64 GiB memory**. Model weights
are external and never committed to this repository.

```sh
uv sync --frozen
uv run jev-local serve
```

The default model directory is
`~/.cache/lm-studio/models/mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit`.
Override it with `--model /path/to/model` or `JEV_MODEL_PATH`.

The service listens at **http://127.0.0.1:8017**. Wait for Uvicorn's startup
message after the model loads. Open **http://127.0.0.1:8017/docs** for interactive
API documentation, or call it directly:

```sh
curl http://127.0.0.1:8017/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {"test_result": "3 passed, 1 failed"},
    "questions": {
      "needs_investigation": {
        "type": "noul",
        "instructions": "Did any test fail?"
      }
    }
  }'
```

Read `answers.needs_investigation.noul` as the conditional probability of yes.
`usage` includes measured prefill, decoder, and total inference time. The model
stays loaded until the service stops. Stop with **Ctrl-C**.

## Try the coding examples

With the server running in another terminal:

```sh
uv run python examples/failure_triage.py
uv run python examples/file_selection.py
uv run python examples/patch_review.py
uv run python examples/context_compaction.py
```

The examples use public synthetic fixtures and report pass/fail and latency.
They cover a missing dependency, choosing a relevant file, spotting an
authentication regression, and previewing context pruning. They never inspect
your repository, execute model-selected commands, or modify agent history.
Compaction keeps pinned messages and tool call/result pairs intact, and retains
the original on invalid or failed responses. See [the examples guide](docs/examples.md).

## How a decision works

1. Encode the state and question descriptions once.
2. Build a short answer canvas with fixed labels such as `q0: A`. Only the answer
   positions receive seeded noise. Every possible alias is checked to occupy
   exactly one tokenizer position.
3. Run one diffusion decoder pass, then normalize logits over the allowed labels.
4. Map labels back to your IDs. Return distributions directly, with no generated
   JSON to parse.

The default path projects only answer positions onto their allowed vocabulary
rows. It avoids computing an entire vocabulary distribution for every canvas
position. Short canvases and reuse of the encoded prompt across noise samples
reduce additional work. A full-vocabulary path remains available for comparison.

## API and limits

`POST /v1/systemone` supports **1–32 questions** per request:

- `noul`: `instructions` describing a yes/no proposition; returns a probability
  from **0 to 1**.
- `choice`: `instructions` and **2–26** named `criteria`; returns the selected
  key, every candidate's probability, and distribution confidence.
- `score`: `instructions` and **2–10** ordered criterion descriptions; returns
  an expected zero-based index, the level distribution, legend, and confidence.

Question instructions and criterion descriptions are strings. State can be any
JSON value. Unknown fields are rejected. Requests are limited to **1 MiB** and
each encoded prompt to **8,192 tokens** by default; oversized prompts are rejected,
never silently truncated. Use `--max-prompt-tokens` to change the token limit with care.

Optional `options`:

```json
{
  "seed": 0,
  "samples": 1,
  "mode": "packed",
  "projection": "labels",
  "canvas_length": null
}
```

`samples` accepts **1–8** noise draws, averaged after normalization. `seed` makes
the input noise repeatable. `canvas_length: null` chooses the smallest multiple
of 16 that fits; explicit widths are **16–256**, in steps of 16.
`projection: "full"` enables the reference computation.

**Packed mode lets questions interact.** Use `mode: "independent"` to encode and
evaluate each question separately, at additional cost. Score levels still share
a prompt in both modes. This differs from native JEV's conditioning.
`confidence = 1 - entropy(probabilities) / log(number_of_choices)` measures how
concentrated a distribution is; it does not estimate reliability. See
[the research notes](docs/research.md) for source links and compatibility details.

`GET /health` and `GET /v1/models` are available. This is a decision API, not an
OpenAI chat endpoint. Inference is serialized on one worker, with **8 waiting
requests**; excess requests return **503**. Health remains responsive during
inference. Run one server process to avoid loading duplicate models.
See the [API reference](docs/api.md) for a mixed request, timing definitions,
limits, and error codes.

## Benchmark and develop

Stop the server before benchmarking so there is only one loaded model:

```sh
uv run python -m diffusion_jev.benchmark --repeats 3 --output benchmark-results.local.json
uv run pytest -q
uv run ruff check src tests examples
uv run mypy
```

The benchmark compares a 256-token full-vocabulary baseline, compact canvases,
selected-label projection, and independent questions. It separates loading and
first-request setup from warm timings and checks synthetic decision quality.
Reports contain measurements and aggregate checks, not prompt or answer payloads.
On the measured M2 Ultra workload, the default reduced warm median latency from
**472 to 291 ms (1.62×)**, with **60/60 synthetic judgments** correct. A batch of
32 simple predicates took **1.08 seconds**. This small suite is not a general
coding-quality evaluation; see [the benchmark report](docs/benchmarks.md) for
methodology, individual examples, memory use, and limitations.

The runtime pins **mlx-optiq 0.5.12** and the lockfile pins dependencies. Its small
internal decoder interface is covered by the real-model benchmark; recheck that
benchmark when upgrading OptiQ. Stock `mlx-lm` and `mlx-vlm` do not load this quant.

## Local data and license

Inference runs on this machine. The service does not write prompts, answers, or
access logs, and keeps no cross-request prompt cache. Validation and inference
errors omit submitted content. The default loopback listener has no
authentication; keep it local. Explicitly binding another interface exposes it
to that network. Swagger's documentation page loads its UI assets from a CDN;
the inference API itself needs no internet after dependencies and weights exist.

Code is [MIT licensed](LICENSE). Model weights have separate upstream terms.
Inspired by [JEV](https://docs.typesafe.ai/primitives),
[vLLM's structured-read proposal](https://github.com/vllm-project/vllm/pull/57250),
[NanoJev](https://github.com/TianyuCodings/NanoJev), and
[fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction).
