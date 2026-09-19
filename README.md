# Local decisions for coding agents

Give your coding agent a local helper for decisions such as **which function to read, which log deserves attention, and whether a fix has enough test evidence**.

You supply the evidence and the allowed answers. The server returns numbers your code can rank or branch on. Everything runs on your Apple Silicon Mac.

This project is inspired by **Jev**, TypeSafe's decision model. It runs **DiffusionGemma**, a different model, through a similar interface. **Its probabilities are experimental; matching Jev's API shape does not reproduce Jev's accuracy or training.**

## Start with one command

You need **an Apple Silicon Mac**, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Git. The matched comparison above ran on a **36 GiB** M4 Max with a measured **18.68 GB** peak allocation and memory pressure never below 29 percent free; earlier work used a 64 GiB machine. Reserve about **20 GB of free disk** for the model.

```sh
uvx --python 3.13 --from git+https://github.com/Saik0s/diffusiongemma-jev-macos jev-local start
```

This installs the Python package into uv's tool cache, downloads the model if needed, then starts the server at **http://127.0.0.1:8017**. The first download is approximately **18 GB**. Later starts reuse the files. An existing compatible LM Studio download is reused too.

Wait for `Application startup complete`. Stop with **Ctrl-C**. No API key or hosted inference account is required.

Already cloned this repository? The equivalent command is:

```sh
uv run jev-local start
```

See [setup and troubleshooting](docs/setup.md) for cache locations, offline use, another port, and existing model paths.

## Make your first decision

In a second terminal, send a log and ask which investigation fits it:

```sh
curl -s http://127.0.0.1:8017/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "pytest cannot collect tests: ModuleNotFoundError: No module named yaml",
    "questions": {
      "next_step": {
        "type": "choice",
        "instructions": "Which investigation best fits this failure?",
        "criteria": {
          "environment": "Inspect missing dependencies and the Python environment.",
          "assertion": "Inspect an assertion that compared the wrong values.",
          "network": "Inspect a remote request that timed out."
        }
      }
    }
  }'
```

The response contains `answers.next_step.choice`, the selected key, and `answers.next_step.probabilities`, a number for each alternative. The intended choice for this example is `environment`; the actual numbers come from your run.

Your code can use that choice to suggest the next step. The server itself does not run commands, edit files, or start another agent.

The other two question types work the same way. A **Noul** returns one probability, and you can spell out what each side means:

```sh
curl -s http://127.0.0.1:8017/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": {"patch": "if not email: raise ValueError(\"email required\")", "tests": ["test_valid_email passes"]},
    "questions": {
      "regression_test": {
        "type": "noul",
        "instructions": "Does a supplied test exercise the empty-email case?",
        "criteria": {
          "true": "A supplied test calls the code with an empty email.",
          "false": "No supplied test covers that input."
        }
      }
    }
  }'
```

A **Score** places the evidence on a scale you describe, and returns the weighted level plus a legend:

```sh
curl -s http://127.0.0.1:8017/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Connection reset by peer while fetching the package index.",
    "questions": {
      "usefulness": {
        "type": "score",
        "instructions": "How useful is this log line for explaining the failure?",
        "criteria": [
          "Routine information with no evidence about the failure.",
          "A related symptom that narrows the investigation.",
          "Specific failure evidence that directly guides the next step."
        ]
      }
    }
  }'
```

Both `criteria` sides are required together on a Noul, or leave `criteria` out for the default yes/no rubric.

### Speed or extra reads

The server reads each answer once per request by default. Start it with `--profile accuracy` to average **eight** reads instead:

```sh
uv run jev-local start --profile accuracy
```

On the 40-query development cohort, eight reads scored **28/40 on both seed 0 and seed 1** against **25/40 and 23/40** for one read. That is repeatable on those two seeds and still development data, so `fast` stays the default. Expect proportionally more time per request: extra reads add decoder passes, while the prompt is prefilled once either way. A request that sets `options.samples` itself always overrides the profile.

`GET /health` reports the model and `prompt_template`, the version of the prompt wording, so you can tell which template produced a recorded number.

### Already calling hosted Jev?

The server also answers `POST /api/alpha/decisions` with the hosted envelope (`id`, `model`, `provider`, `answers`, `usage`), so an existing hosted client needs one changed field, the model name, plus the local base URL. It reports `provider: "local"` and `cost: 0.0`. This is a minimal client adapter, not drop-in SDK compatibility, and the answers still come from a different model. See [the API reference](docs/api.md#hosted-shaped-route).

## What is Jev, in plain language?

Jev answers bounded questions for software. Instead of asking for a paragraph explaining a failure, you ask a question whose answers your program already understands.

There are three building blocks:

| Type | Question you might ask | Answer your code gets |
| --- | --- | --- |
| **Noul** | Does this function silently ignore an exception? | Probability of yes, from **0 to 1** |
| **Choice** | Which of these files should I inspect first? | Selected option plus every option's probability |
| **Score** | How useful is this log for diagnosing the failure? | Position on a scale whose levels you describe |

**State** means the evidence you provide: code, logs, requirements, or test results. **Questions** say what to judge. **Your code** decides what happens next.

```text
Evidence + questions → local model → probabilities → your ranking or routing code
```

A number such as `0.9` is a strong model preference. **We have not shown that it means “correct 90% of the time.”** A high confidence value also cannot prove an answer right.

Read [Jev concepts for developers](docs/concepts.md) for complete examples, probabilities versus confidence, Score arithmetic, and when to ask several questions together. No machine-learning background is assumed.

## Where this is useful

Start with narrow judgments where all necessary evidence fits in the request, and a wrong suggestion is easy to inspect:

- **Search code by behavior.** Rank functions that may swallow errors, even when their names do not mention errors.
- **Triage logs.** Build a smaller investigation bundle while keeping the original records.
- **Check completion evidence.** Distinguish “a patch exists” from “the reported failure has a regression test.”
- **Spot a stalled agent.** Recognize repeated retries that bring no new evidence.
- **Preview context pruning.** Keep relevant tool exchanges and pinned instructions together.

These uses come from [public Jev projects](docs/community.md), including Every, Jev Logs, Foreman, ProgressGate, and fast-jev-compaction. Our examples adapt their ideas into inspectable teaching scenarios.

Use ordinary code for exact facts such as exit codes. Use a coding model to write or explain a patch. Use tests and review to establish whether the patch works. This server supplies judgments that can help decide where those tools spend effort.

## Try the examples

Keep the server running. To get the source and run the examples:

```sh
git clone https://github.com/Saik0s/diffusiongemma-jev-macos.git
cd diffusiongemma-jev-macos
uv run jev-local demo search
```

Then explore the other workflows:

```sh
uv run jev-local demo logs
uv run jev-local demo completion
uv run jev-local demo progress
uv run python examples/context_compaction.py
```

Each demo shows the question, the model's actual judgments, and the resulting suggestion. They use small public synthetic fixtures. They do not scan your repository or control a real agent.
Use `uv run jev-local demo all` to run all four decision workflows together.

The [examples walkthrough](docs/examples.md) explains the input, policy, expected behavior, and failure handling for each workflow. It also links to the implementation you can adapt.

## What have we measured?

**Real-code retrieval:** on 50 CodeSearchNet queries, the local model put the labeled function first **31/50 times (62%)**, compared with **24/50 (48%)** for keyword search. Archived Jev results reached **42/50 (84%)**, with different batching. A local rerank of 30 functions took **19.1 seconds median**. This is a small pilot, not proof of a general advantage. See [the retrieval benchmark](docs/retrieval-benchmark.md) for the public source, method, and limits.

**Local runtime speed:** on an M2 Ultra, the earlier small synthetic suite took **291 ms median** per request with the default configuration, versus **472 ms** for the original baseline. Those tiny prompts are easier than real retrieval requests. They do not establish coding competence. See [the runtime benchmark](docs/benchmarks.md).

**Accuracy experiments in progress:** on a frozen 40-query development cohort,
the one-read baseline scored **25/40** on seed 0 and **23/40** on seed 1. Eight
averaged reads scored **28/40** on both seeds; a 256-token reasoning prepass
scored **29/40** and then **24/40**. These are development screens, not held-out
proof. The seed-2 recheck, coding screens, and held-out validation are unfinished,
and the general default has not changed. See [the accuracy study](docs/accuracy-study.md).

**Matched against hosted Jev, same requests, same machine, same network:** on
September 20, 2026 the frozen 38-case suite ran through this server in both
sampling profiles and through real `typesafe/jev-1.13`, all issued from one
Apple M4 Max with 36 GiB of memory on one residential connection.

| | local, `fast` | local, `accuracy` | hosted `jev-1.13` |
| --- | --- | --- | --- |
| cases answered | 38, 0 failures | 38, 0 failures | 38, 0 failures |
| labelled accuracy | 20/20 | 20/20 | 20/20 |
| top-choice agreement with hosted | 20/21 | 20/21 | — |
| round trip, median | 0.336 s | 0.384 s | 0.326 s |
| round trip, p90 | 0.901 s | 0.991 s | 0.511 s |
| server inference, median | 335 ms | 383 ms | not reported by the API |
| peak MLX memory | 18.68 GB | 18.68 GB | — |
| cost for all 38 calls | 0 | 0 | 0.0011 USD |

Scope: one run per profile, one day, one machine; `diffusiongemma-26B-A4B-it`
OptiQ-4bit on prompt template `jev-local-prompt-v2`, one inference worker on
port 8017, no `options` block, so each side used its own defaults. Median round
trip is within 60 ms across all three columns, and hosted is clearly tighter at
the tail. **No accuracy winner is claimed:** all three scored 20/20 against 20
mechanically derived labels of which only 6 have a single correct key, so this
suite cannot separate the systems on quality. The eight-read `accuracy` profile
changed no answer here while costing about 14 percent more median latency, which
is why the default stays `fast`. Full record:
[`benchmarks/matched-comparison-2026-09-20.json`](benchmarks/matched-comparison-2026-09-20.json).
See [differences from Jev](docs/jev-differences.md#measured-result-2026-09-20)
for the single disagreement, the two payloads hosted accepts and this server
rejects, and the rest of the limits.

To reproduce the real-code evaluation against the running server:

```sh
uv run --extra benchmark jev-local benchmark-search --limit 50 --output retrieval-results.local.json
```

The benchmark downloads public evaluation data. It does not call a paid Jev API. Its report records measurements and identifiers, without copying the source functions into the report.

### What is not finished

- **The matched comparison cannot separate quality.** It ran, and the table above
  reports it, but labels cover the 20 grid cases only and just 6 of those have a
  single correct key. Both sides reached 20/20, which is a ceiling effect rather
  than a quality verdict. Replay it yourself with
  `uv run jev-local benchmark-matched --output report.json`. See
  [the suite's method and limits](docs/jev-differences.md#matched-suite-replayed-against-both-sides).
- **No held-out accuracy.** Only the 40-query development cohort has been
  screened. The validation and 150-query final-test cohorts have not run, and the
  seed-2 baseline stopped at 10/40.
- **The retrieval and accuracy numbers predate the current prompt.** Those were
  measured on prompt template `jev-local-prompt-v1` with sorted state keys. The
  template is now `jev-local-prompt-v2`, which keeps the caller's key order and
  renders Noul criteria, so they need re-measuring. Only the matched comparison
  above was measured on `v2`.
- **No calibration claim.** Nothing here shows a probability of 0.9 corresponds
  to being right 90% of the time.

## Limits to understand before integrating

- **A decision interface, not a chat server.** It returns Noul, Choice, and Score answers. It does not generate code or expose an OpenAI chat-completions endpoint.
- **Evidence must be supplied.** It does not inspect your project or retain a conversation between requests.
- **Small bounded requests.** Up to **32 questions**, **26 Choice options**, and **10 Score levels**. Default prompt limit: **8,192 tokens**, roughly chunks of text. Request body limit: **1 MiB**.
- **Question interactions.** Default `packed` mode evaluates questions together; adding or reordering questions can affect answers. `independent` mode separates questions at additional cost.
- **Experimental probabilities.** Thresholds in examples are demonstration policy. They are not a security boundary, deletion guarantee, or substitute for measured error rates.
- **One local inference worker.** Requests are serialized, with **eight waiting slots**. Keep one server process running to avoid loading duplicate models.

## Learn more or develop

- [Concepts](docs/concepts.md): start here if Jev is new to you.
- [Setup](docs/setup.md): installation, downloads, offline use, and troubleshooting.
- [Examples](docs/examples.md): decisions inside coding workflows.
- [API reference](docs/api.md): complete fields, options, limits, and errors.
- [Community sources](docs/community.md): the projects behind these examples.
- [Implementation notes](docs/research.md): how the local model produces answers and where it differs from Jev.
- [Differences from Jev](docs/jev-differences.md): observed hosted behavior, contracts, probabilities, batching, latency, and the open Simple-JEV design.
- [Accuracy evaluation](docs/accuracy-study.md): frozen cohorts, experiment controls, and reproducible comparisons.

Run the local checks from a checkout:

```sh
uv sync --frozen --extra benchmark
uv run pytest -q
uv run ruff check src tests examples
uv run mypy
```

Inference stays on your machine. The server does not store prompts or answers or emit request access logs. Downloads need internet; inference can run offline once dependencies and model files exist. The interactive `/docs` page loads its interface assets from a CDN.

The default listener is local-only and has no authentication. Binding another interface exposes it to that network.

Code is [MIT licensed](LICENSE). Model weights and evaluation datasets retain their own upstream terms. The [OptiQ model card](https://huggingface.co/mlx-community/diffusiongemma-26B-A4B-it-OptiQ-4bit) documents the required runtime; stock `mlx-lm` and `mlx-vlm` cannot load this checkpoint.
