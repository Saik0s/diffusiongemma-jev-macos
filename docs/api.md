# API reference

Send evidence and bounded questions to **`POST /v1/systemone`**. The response contains
typed answers your code can use directly, without parsing generated prose.

Start `uv run jev-local start`, then use `http://127.0.0.1:8017`. The interactive
schema is at `/docs`; the machine-readable schema is at `/openapi.json`.
For the meaning of state, Noul, Choice, Score, and confidence, start with
[the concepts guide](concepts.md). This page documents the local implementation.

## Call it from Python

Run this from an environment containing the package, with the server running:

```python
from diffusion_jev.client import DecisionClient, DecisionClientError
from diffusion_jev.schemas import DecisionRequest, NoulAnswer, NoulQuestion

request = DecisionRequest(
    state={
        "function": "def parse_count(text):\n"
        "    try: return int(text)\n"
        "    except ValueError: return 0",
    },
    questions={
        "hides_error": NoulQuestion(
            type="noul",
            instructions="Does this function catch an exception without reporting it?",
        ),
    },
)

try:
    with DecisionClient() as client:
        answer = client.decide(request).answers.get("hides_error")
    if isinstance(answer, NoulAnswer):
        print(f"Probability of yes: {answer.noul:.3f}")
        print("Inspect this function before deciding whether its fallback is appropriate.")
    else:
        print("No usable answer. Keep the function available for manual review.")
except DecisionClientError:
    print("The request failed. Keep the function available for manual review.")
```

The printed number is the model's judgment. Returning zero after a parse failure may
be intentional, so detecting that behavior alone does not establish a bug.

`DecisionClient` accepts `base_url` and an optional `timeout` in seconds. It defaults
to the local address above and a **120-second timeout**. It validates the response
shape and raises `DecisionClientError` without exposing the request or raw response.
Check that the expected question IDs
and answer types are present before acting. The workflow examples also validate
Choice distributions and Score legends against their requests.

## One request, three decision types

Send this synthetic example to `POST /v1/systemone`:

```json
{
  "model": "diffusiongemma-local",
  "state": {
    "test_result": "Login rejects every user with HTTP 500.",
    "workaround": "None known."
  },
  "questions": {
    "investigate": {
      "type": "noul",
      "instructions": "Does login fail for users?"
    },
    "next_file": {
      "type": "choice",
      "instructions": "Which supplied file should be inspected first?",
      "criteria": {
        "auth": "src/auth.py handles login requests.",
        "styles": "static/theme.css controls colors.",
        "other": "Neither file matches the problem."
      }
    },
    "severity": {
      "type": "score",
      "instructions": "How disruptive is the reported issue?",
      "criteria": [
        "Cosmetic issue; functionality works.",
        "A feature fails, with a working alternative.",
        "A feature is blocked without a known workaround."
      ]
    }
  }
}
```

Responses have `model`, an `answers` map using your original IDs, and `usage`.
No natural-language output is returned. Optional reasoning stays in memory and
is not included in the response.

- `answers.investigate`: `type` and `noul`, a probability of yes. A threshold such
  as 0.5 is an application choice, not a reliability guarantee.
- `answers.next_file`: `type`, `choice`, `probabilities`, and `confidence`.
  `choice` is the highest-probability supplied key; ties use insertion order.
- `answers.severity`: `type`, `score`, `probabilities`, `legend`, and `confidence`.
  Levels are zero-based. `score` is the weighted mean, which may differ from the
  most probable level. Probability and legend keys are strings in JSON.

For Choice and Score, all allowed labels have a probability and their sum is one.
The distribution is conditional on those labels, not on the entire vocabulary.
An `other` candidate can be useful when your list is not exhaustive.

## Options

The optional `options` object accepts:

- `seed`: integer **0–4,294,967,295**, default **0**.
- `samples`: integer **1–8**, default **1**. Each sample starts from fresh seeded
  answer-slot noise; the encoded prompt is reused and probabilities are averaged.
  These are independent single-step reads, not successive denoising steps.
- `mode`: **`packed`** by default, or `independent`. Packed questions share a
  prompt and canvas. Independent mode runs each question separately. Score levels
  remain a joint choice in either mode.
- `projection`: **`labels`** by default, `full` for the reference computation,
  or experimental `labels-fp32` for float32 selected-label arithmetic. The latter
  did not improve development top-1 accuracy; it is not a recommended default.
- `canvas_length`: **`null`** by default, selecting the smallest fitting multiple
  of 16. Explicit values are **16–256**, in multiples of 16. A canvas too small
  for the answer scaffold is rejected. Changing width can change probabilities.

IDs are transport labels. The model sees positional aliases instead. Candidate
names and descriptions are semantic input; changing their order can affect the
answer. Identical seeds reproduce input noise, not a cross-hardware guarantee of
bit-identical floating-point execution.

## Timing and limits

`usage.prompt_tokens` counts encoded prompt tokens actually evaluated. Independent
mode counts each separate prompt. `decoder_passes` counts model decoder calls.
`canvas_tokens` counts the sum of canvas widths across those calls; it is **not**
an output-token count.

Usage also reports `reasoning_tokens`, `reasoning_ms`, `reasoning_passes`,
`reasoning_forced_closures`, and `trajectory_accepted_slots`. Reasoning is disabled
by default. Start the server with experimental `--reasoning-tokens 256` or
`--reasoning-tokens 512` to enable a bounded reasoning pass before decisions;
both `start` and `serve` accept the flag. This setting applies to every request
and is not a per-request option. Responses can be substantially slower, with no
guaranteed accuracy improvement. `options.samples` still controls **1–8**
independent decision reads, default **1**, after the reasoning pass.

Reasoning metrics are zero with `--reasoning-tokens 0`. Structured trajectories
remain benchmark-only, and `trajectory_accepted_slots` stays zero on the server.
See the [accuracy study](accuracy-study.md) for experimental policies and results.

`prefill_ms`, `decode_ms`, and `total_ms` use a monotonic clock with evaluated MLX
arrays. `total_ms` includes preparation and answer assembly, but excludes time
waiting in the HTTP queue and transport time. `peak_memory_gb` is peak MLX active
allocation during the request, including loaded weights, in decimal GB; it is
not system RAM usage or the process's resident size.

Requests allow **1–32 questions**, Choice **2–26 candidates**, Score **2–10
levels**, IDs up to **128 characters**, and descriptions up to **4,096
characters**. Instructions and descriptions must be nonblank strings; `state`
may contain any JSON value. Nonfinite JSON numbers cannot be processed.
The body limit is **1 MiB**. Each encoded prompt is limited to **8,192 tokens** by
default. Independent mode repeats the state for each question, so long states
and many questions can take substantially longer.
With reasoning enabled, each formatted prompt, including its thought-opening
marker, must also leave **257 tokens** for the 256-token budget or **513 tokens**
for the 512-token budget within `--max-prompt-tokens`. These reserves include one
closing token; oversized requests are rejected rather than shortened.

There is **one inference worker and eight waiting slots**. The client's default
network timeout is 120 seconds, which longer reasoning requests may exceed. Set
`DecisionClient(timeout=600.0)` for a longer timeout. Disconnecting a client does not interrupt
active GPU work. This is a local single-process research service, not a multi-user
scheduler.

## Errors and operational endpoints

- **400**: unknown model ID.
- **413**: request body exceeds 1 MiB.
- **422**: invalid schema, excessive prompt length, or canvas that cannot fit.
- **503**: model unavailable or inference queue full.
- **500**: inference failed.

Errors omit request contents. `GET /health` reports service availability;
`GET /v1/models` lists the one configured model. There is no chat-completions
endpoint, authentication, or persistent conversation storage.

The [research notes](research.md) distinguish this API subset from native Jev.
Compared with the hosted Jev endpoint, the local API returns unrounded
probabilities, a different `confidence` statistic, a local `usage` object, and
422 rather than 400 for validation errors; it also accepts numeric and boolean
`state` roots that the hosted API rejects. See [the observed differences](jev-differences.md#wire-format).
