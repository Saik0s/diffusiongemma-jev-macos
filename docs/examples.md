# Decisions inside a coding workflow

These examples show how a model judgment becomes an ordinary branch or ranking in your code.
They use small, public synthetic fixtures so you can inspect every input.
They do not read a real repository, change an agent's history, or execute a suggested action.

Start the server with `uv run jev-local start`. In another terminal, run:

```sh
uv run jev-local demo all
```

Use `--url http://127.0.0.1:8018` if your server uses another port.
All probabilities shown in the terminal come from the running model.
The examples label suggested actions separately from the expected fixture behavior.

If the three question types are unfamiliar, read [the concepts guide](concepts.md) first.

## Find functions worth reading

```sh
uv run jev-local demo search
```

**Problem:** you suspect that a codebase hides exceptions, but a search for `except` also finds correct error handling.

**Evidence:** six short Python functions with different behaviors, including logging, reraising, and returning after a failure.
The question asks whether each function discards an exception without reporting it.
A Noul answer supplies the probability of yes for each candidate.

**What your code does:** rank candidates for inspection. This helps choose the next file to read.
It does not change code or establish a confirmed defect.
The demo labels probabilities **≥ 0.65** as `inspect_first`, **> 0.2 but < 0.65** as `review_if_needed`, and **≤ 0.2** as `rank_lower`.
Every candidate remains in the output.

**What to watch:** a function that logs an exception should rank differently from one that silently returns a fallback.
If the model misses that distinction, you can see the error directly.
Behavior outside the supplied function remains unknown.

Inspired by [Every](https://github.com/sufianetaouil/every).
Run the same example through [the standalone wrapper](../examples/semantic_search.py).

## Turn a log stream into an investigation bundle

```sh
uv run jev-local demo logs
```

**Problem:** sending every routine log line to a coding agent makes the investigation harder to follow.

**Evidence:** a short stream containing routine activity, a recovered retry, and an unresolved failure.
The concrete records are a successful health check, a Redis timeout followed by success, and a checkout error caused by a missing database column.
For each record, the model judges actionability, category, and diagnostic value.
This gives a practical use for Noul, Choice, and Score in one workflow.

**What your code does:** select records for the suggested investigation bundle using an explicit policy.
The original log stream remains intact.
A failed or invalid decision falls back to keeping the evidence.

**What to watch:** a frightening word such as “retry” does not necessarily indicate an unresolved problem.
The surrounding result matters. The suggested bundle should help explain the current failure rather than merely collect alarming words.

The policy leaves a record in the archive only when **P(actionable) ≤ 0.2**, **P(routine) ≥ 0.8**, and **Score < 1.0** all hold.
Any other valid combination keeps it in the investigation bundle.

Inspired by [Jev Logs](https://github.com/reachjalil/jevlogs).
See [the standalone wrapper](../examples/log_triage.py).

## Check whether a fix has enough evidence

```sh
uv run jev-local demo completion
```

**Problem:** an agent says “done” after editing code, but the evidence may not show that the reported failure is covered.

**Evidence:** two snapshots of a small fix.
The requirement is to reject retry counts **≤ 0** with `ValueError`, while accepting positive counts.
The first snapshot adds the guard but only tests a count of **3**.
The second adds tests for **0** and **-1**, plus output showing all **three tests passed**.

**What your code does:** combine separate yes/no judgments about the requirement, regression coverage, and verification evidence.
The result suggests continuing work or preparing for review.
A missing answer cannot promote the task to ready.

All three checks must reach **0.8** before the example suggests `ready_for_review`.
That is permission to begin review within the demo's policy, not proof that the code is correct.

**What to watch:** “tests passed” is useful only alongside evidence of what was tested.
The two snapshots should produce different recommendations when the missing regression evidence is material.
The model does not execute the tests or certify the implementation.

Inspired by [Foreman](https://github.com/thruwire/foreman).
See [the standalone wrapper](../examples/completion_check.py).

## Notice a stalled agent loop

```sh
uv run jev-local demo progress
```

**Problem:** an agent keeps trying new commands while relying on the same assumption that the evidence already contradicted.

**Evidence:** two short action histories.
Both begin with a deployment API returning **403** because a token lacks the required scope.
One retries with `sudo` and a different working directory; the other inspects which token alias the deployment job uses.
The questions distinguish repetition from new evidence.

**What your code does:** suggest continuing or replanning.
It does not stop an agent or execute a replacement command.
If the model response is invalid, the demo asks for manual inspection rather than asserting progress.

Replanning requires **P(repeated assumption) ≥ 0.8** and **P(new evidence) ≤ 0.2**.
Continuing requires the reverse: **P(new evidence) ≥ 0.8** and **P(repeated assumption) ≤ 0.2**.
Other combinations suggest reviewing the trajectory.

**What to watch:** different command text does not automatically mean a different strategy.
A useful signal concerns the underlying hypothesis and what each action learned.

**Observed limitation:** in our live run, the stalled sequence correctly suggested replanning.
The productive sequence returned probabilities **0.688** and **0.686**, so the policy requested review instead of continuing automatically.
The demo preserves that uncertainty; it does not replace the model's answer with the expected teaching outcome.

Inspired by [ProgressGate](https://github.com/AshutoshVJTI/progressgate).
See [the standalone wrapper](../examples/stagnation_check.py).

## Preview context compaction

```sh
uv run python examples/context_compaction.py
```

**Problem:** an agent's conversation contains useful code evidence alongside unrelated old tool output.
You want to inspect a shorter version without rewriting retained evidence.

**Evidence:** a deliberately tiny email-validation task with five context units.
One old tool exchange contains validator evidence; another contains unrelated weather information.
The initial goal and two most recent messages are pinned by code.

**What your code does:** keep pinned units and retain an eligible unit when its usefulness probability is **at least 0.5**.
Every tool call and its result form one indivisible unit.
Retained text remains unchanged and in its original order.

The terminal shows each keep/omit decision and the before/after character count.
Our measured run produced:

```text
KEEP: goal (pinned by code)
KEEP: validation_evidence (model relevance judgment)
OMIT FROM PREVIEW: unrelated_weather (model relevance judgment)
KEEP: recent_user (pinned by code)
KEEP: recent_assistant (pinned by code)
Preview: 4/5 units; 241/365 chars
```

The threshold is a demonstration setting, not a proven safe deletion rule.
Missing answers, unexpected IDs, invalid values, or request failure preserve the complete original context.

Inspired by [fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction).
Its full design can judge calls and results separately. This demo deliberately uses whole pairs.
See [the wrapper](../examples/context_compaction.py) and [the policy](../src/diffusion_jev/examples.py).

## Three smaller API smoke checks

These are intentionally easy examples of the response contract:

```sh
uv run python examples/failure_triage.py
uv run python examples/file_selection.py
uv run python examples/patch_review.py
```

- **Failure triage:** choose the likely cause of a missing Python import and suggest an investigation.
- **File selection:** choose a validator from a short list of file descriptions.
- **Patch review:** flag an obvious removed authentication guard and score review priority.

Their `PASS` means the model matched the fixture's expected answer.
It does not establish general debugging, security-review, or coding ability.
The older [runtime benchmark](benchmarks.md) uses these unchanged requests to compare execution strategies.

## Adapt an example to your agent

The shared implementations are [workflows.py](../src/diffusion_jev/workflows.py),
[demo.py](../src/diffusion_jev/demo.py), and [examples.py](../src/diffusion_jev/examples.py).
They use [DecisionClient](../src/diffusion_jev/client.py) to send typed requests to the server.

1. Pick a decision with an observable outcome, such as whether the selected function was useful.
2. Supply only the evidence needed for that judgment, including relevant requirements and constraints.
3. Define the permitted answers and an explicit fallback before interpreting the probabilities.
4. Try labeled cases from your workflow, including ambiguous cases and failures.
5. Measure errors, time, and downstream effects before automating a consequential action.

Keep the original evidence available. Do not turn a tutorial threshold into an authorization rule.
The [API reference](api.md) documents the exact request shape and limits.
