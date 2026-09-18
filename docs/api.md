# API reference

Start `uv run jev-local serve`, then use `http://127.0.0.1:8017`. The interactive
schema is at `/docs`; the machine-readable schema is at `/openapi.json`.

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
No natural-language output is generated.

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
- `projection`: **`labels`** by default, or `full` for the reference computation.
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

There is **one inference worker and eight waiting slots**. The client timeout is
120 seconds. Disconnecting a client does not interrupt active GPU work. This is
a local single-process research service, not a multi-user scheduler.

## Errors and operational endpoints

- **400**: unknown model ID.
- **413**: request body exceeds 1 MiB.
- **422**: invalid schema, excessive prompt length, or canvas that cannot fit.
- **503**: model unavailable or inference queue full.
- **500**: inference failed.

Errors omit request contents. `GET /health` reports service availability;
`GET /v1/models` lists the one configured model. There is no chat-completions
endpoint, authentication, or persistent conversation storage.

The [research notes](research.md) distinguish this API subset from native JEV.
