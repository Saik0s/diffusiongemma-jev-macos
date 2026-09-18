# Synthetic coding examples

Start the local server as described in the README, then run:

```sh
uv run python examples/failure_triage.py
uv run python examples/file_selection.py
uv run python examples/patch_review.py
uv run python examples/context_compaction.py
```

Each command accepts `--url http://127.0.0.1:8017`. Fixtures are explicitly
synthetic and public. They do not inspect your repository, execute model-selected
commands, or connect to an agent's actual history. Output contains pass/fail,
elapsed time, and, for compaction, counts and retained synthetic unit IDs.

- Failure triage combines a three-way cause choice with two probability questions:
  whether a dependency is missing and whether a network request timed out.
- File selection identifies the validator responsible for a described email bug.
- Patch review checks an obvious removal of authentication and scores review
  priority across cosmetic, nonblocking, and critical security levels.
- Context compaction judges which old units support an email validation task.

Together these exercise all three answer types: `choice`, `noul` (a yes
probability), and `score` (an expected zero-based level). Triage and patch review
also demonstrate multiple questions in one request. Quality checks compare the
most probable score level with the expected category, rather than rounding the
weighted score. These fixtures are available through `coding_fixtures()` for
repeatable smoke checks.

Compaction pins the initial goal and both latest messages. Each tool call and its
result form one indivisible unit. A useful probability of at least **0.5** retains
the whole unit. Retained strings remain byte-for-byte unchanged and in order.
Missing answers, extra or unknown IDs, invalid values, or request failure preserve
the original context. This is a preview, not an integration with agent memory.

Probabilities are experimental and uncalibrated. These deliberately easy fixtures
are smoke checks, not evidence of coding competence, security assurance, or safe
automatic deletion of real context. A passing example does not authorize an action.
