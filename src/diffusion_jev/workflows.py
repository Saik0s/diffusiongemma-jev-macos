"""Small public teaching fixtures and conservative, code-owned workflow policies."""

from dataclasses import dataclass
from math import isclose
from typing import Literal

from pydantic import JsonValue

from diffusion_jev.schemas import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
)

WorkflowName = Literal["search", "logs", "completion", "progress"]
WORKFLOW_NAMES: tuple[WorkflowName, ...] = ("search", "logs", "completion", "progress")


@dataclass(frozen=True)
class WorkflowCase:
    name: WorkflowName
    title: str
    purpose: str
    source: str
    evidence: tuple[str, ...]
    request: DecisionRequest
    items: tuple[tuple[str, str], ...]
    expected_actions: tuple[str, ...]


@dataclass(frozen=True)
class SuggestedAction:
    label: str
    action: str
    reason: str
    signals: tuple[str, ...]


def _noul(instructions: str) -> NoulQuestion:
    return NoulQuestion(type="noul", instructions=instructions)


def _search_case() -> WorkflowCase:
    functions: dict[str, JsonValue] = {
        "load_settings": "def load_settings(path):\n    try:\n        return read_json(path)\n"
        "    except ValueError:\n        pass\n    return {}",
        "send_event": "def send_event(event):\n    try:\n        transport.send(event)\n"
        "    except ConnectionError:\n        logger.exception('Event delivery failed')\n"
        "        raise",
        "parse_count": "def parse_count(text):\n    try:\n        return int(text)\n"
        "    except ValueError as exc:\n        raise InvalidCount(text) from exc",
        "refresh_cache": "def refresh_cache():\n    try:\n        cache.refresh()\n"
        "    except RuntimeError:\n        return None",
        "close_session": "def close_session(session):\n    try:\n        session.flush()\n"
        "    finally:\n        session.close()",
        "sync_profile": "def sync_profile(profile):\n    try:\n        remote.save(profile)\n"
        "    except TimeoutError:\n        metrics.increment('profile_sync_timeout')\n"
        "        queue.enqueue(profile)\n        return 'queued'",
    }
    questions: dict[str, Question] = {
        name: _noul(
            f"In the function named {name}, is an exception caught and discarded without "
            "logging it, recording a failure metric, retrying, or raising an error? "
            "Judge only the function body provided."
        )
        for name in functions
    }
    return WorkflowCase(
        name="search",
        title="Find swallowed exceptions across six functions",
        purpose="Rank code to inspect when a keyword search would also match correct handlers.",
        source="https://github.com/sufianetaouil/every",
        evidence=(
            "load_settings: catches ValueError, passes, then returns an empty dictionary.",
            "send_event: catches ConnectionError, logs it, and raises again.",
            "parse_count: translates ValueError into InvalidCount.",
            "refresh_cache: catches RuntimeError and returns None.",
            "close_session: uses finally to close the session.",
            "sync_profile: catches TimeoutError, records a metric, and queues a retry.",
        ),
        request=DecisionRequest(state={"functions": functions}, questions=questions),
        items=tuple((name, f"sample.py::{name}") for name in functions),
        expected_actions=(
            "inspect_first", "rank_lower", "rank_lower",
            "inspect_first", "rank_lower", "rank_lower",
        ),
    )


def _logs_case() -> WorkflowCase:
    records: dict[str, JsonValue] = {
        "health": "INFO GET /health 200, duration=2ms; readiness checks passed",
        "retry": "WARN Redis connect timed out on attempt 1; attempt 2 connected after 40ms; "
        "request completed with HTTP 200",
        "checkout": "ERROR POST /checkout 500: database insert failed: "
        "column orders.currency_code does not exist; deployment version=2026.09.18.2",
    }
    questions: dict[str, Question] = {}
    for name in records:
        questions[f"{name}_actionable"] = _noul(
            f"Does log record {name} show an unresolved problem that merits engineer investigation?"
        )
        questions[f"{name}_priority"] = ChoiceQuestion(
            type="choice",
            instructions=f"Which investigation priority fits log record {name}?",
            criteria={
                "routine": "Normal operation or a transient issue explicitly recovered",
                "investigate": "An unresolved failure affecting an application operation",
                "unclear": "Not enough context to determine whether the problem recovered",
            },
        )
        questions[f"{name}_value"] = ScoreQuestion(
            type="score",
            instructions=f"How much diagnostic evidence does log record {name} provide?",
            criteria=[
                "Routine status without a failure to diagnose",
                "Reports a symptom but no specific failing component or cause",
                "Identifies a specific failing component, operation, or cause",
            ],
        )
    return WorkflowCase(
        name="logs",
        title="Build a smaller investigation bundle from application logs",
        purpose="Spend deeper analysis on useful evidence while retaining every original record.",
        source="https://github.com/reachjalil/jevlogs",
        evidence=(
            "Health: HTTP 200; readiness checks passed.",
            "Redis: first attempt timed out; second connected; request finished HTTP 200.",
            "Checkout: HTTP 500; database insert cannot find orders.currency_code.",
        ),
        request=DecisionRequest(state={"records": records}, questions=questions),
        items=(
            ("health", "Health check"),
            ("retry", "Redis retry"),
            ("checkout", "Checkout error"),
        ),
        expected_actions=("archive_only", "archive_only", "investigate"),
    )


def _completion_cases() -> tuple[WorkflowCase, ...]:
    questions: dict[str, Question] = {
        "implemented": _noul(
            "Does the provided patch reject zero and negative retry counts while accepting "
            "positive counts?"
        ),
        "regression": _noul(
            "Do the provided tests exercise both zero and negative retry counts and assert "
            "that ValueError is raised?"
        ),
        "verified": _noul(
            "Does the transcript provide a passing test-run result for tests covering rejection "
            "of both zero and negative retry counts?"
        ),
    }
    patch = (
        "def configure_retries(count):\n    if count <= 0:\n        raise ValueError("
        "'count must be positive')\n    return RetryPolicy(count=count)"
    )
    test = "def test_positive():\n    assert configure_retries(3).count == 3"
    snapshots = (
        ("After the first patch", test, "pytest test_retry.py: 1 passed"),
        (
            "After adding boundary tests",
            test + "\n@pytest.mark.parametrize('count', [0, -1])\n"
            "def test_rejects_nonpositive(count):\n    with pytest.raises(ValueError):\n"
            "        configure_retries(count)",
            "pytest test_retry.py: 3 passed; collected test_positive, "
            "test_rejects_nonpositive[0], test_rejects_nonpositive[-1]",
        ),
    )
    return tuple(
        WorkflowCase(
            name="completion",
            title=title,
            purpose="Separate a plausible implementation from evidence that the bug is covered.",
            source="https://github.com/thruwire/foreman",
            evidence=(
                "Requirement: reject zero/negative retry counts; accept positive counts.",
                "Patch adds a count <= 0 check that raises ValueError.",
                transcript,
            ),
            request=DecisionRequest(
                state={
                    "requirement": "Reject counts <= 0 with ValueError; accept positive counts.",
                    "patch": patch,
                    "tests": tests,
                    "test_run": transcript,
                },
                questions=questions,
            ),
            items=(("completion", "Retry-count fix"),),
            expected_actions=(
                "request_regression_evidence" if index == 0 else "ready_for_review",
            ),
        )
        for index, (title, tests, transcript) in enumerate(snapshots)
    )


def _progress_cases() -> tuple[WorkflowCase, ...]:
    snapshots: tuple[tuple[str, list[JsonValue]], ...] = (
        (
            "Deployment investigation A",
            [
                "Run deploy: remote API returns 403, token lacks deployment:write scope.",
                "Run sudo deploy: remote API returns 403, token lacks deployment:write scope.",
                "Run deploy from /tmp: remote API returns 403, token lacks deployment:write scope.",
            ],
        ),
        (
            "Deployment investigation B",
            [
                "Run deploy: remote API returns 403, token lacks deployment:write scope.",
                "Inspect deployment credential configuration: CI references read-only token alias.",
                "Read credential setup docs: deployment job requires deploy-service token alias; "
                "prepare a configuration patch for review without modifying credentials.",
            ],
        ),
    )
    return tuple(
        WorkflowCase(
            name="progress",
            title=title,
            purpose="Detect repeated rejected assumptions, rather than merely counting steps.",
            source="https://github.com/AshutoshVJTI/progressgate",
            evidence=tuple(str(step) for step in steps),
            request=DecisionRequest(
                state={"goal": "Diagnose the failed deployment", "recent_steps": steps},
                questions={
                    "same_assumption": _noul(
                        "Do later actions continue the same approach contradicted by the error, "
                        "without addressing the missing remote token permission?"
                    ),
                    "new_evidence": _noul(
                        "Do later steps establish new relevant evidence about the cause or a "
                        "concrete correction that addresses the error?"
                    ),
                },
            ),
            items=(("progress", "Recent agent trajectory"),),
            expected_actions=(
                "suggest_replan" if index == 0 else "continue_investigation",
            ),
        )
        for index, (title, steps) in enumerate(snapshots)
    )


def workflow_cases(name: WorkflowName) -> tuple[WorkflowCase, ...]:
    if name == "search":
        return (_search_case(),)
    if name == "logs":
        return (_logs_case(),)
    if name == "completion":
        return _completion_cases()
    return _progress_cases()


def valid_answer(question: Question, answer: Answer | None) -> bool:
    """Validate relationships that the general response schema cannot express."""
    if isinstance(question, NoulQuestion):
        return isinstance(answer, NoulAnswer)
    if isinstance(question, ChoiceQuestion) and isinstance(answer, ChoiceAnswer):
        probabilities = answer.probabilities
        return (
            set(probabilities) == set(question.criteria)
            and isclose(sum(probabilities.values()), 1.0, abs_tol=1e-5)
            and answer.choice in probabilities
            and probabilities[answer.choice] == max(probabilities.values())
        )
    if isinstance(question, ScoreQuestion) and isinstance(answer, ScoreAnswer):
        expected = {str(index): text for index, text in enumerate(question.criteria)}
        return (
            set(answer.probabilities) == set(expected)
            and answer.legend == expected
            and isclose(sum(answer.probabilities.values()), 1.0, abs_tol=1e-5)
            and isclose(
                answer.score,
                sum(int(key) * value for key, value in answer.probabilities.items()),
                abs_tol=1e-5,
            )
        )
    return False


def _signals(answers: dict[str, Answer]) -> tuple[str, ...]:
    result = []
    for name, answer in answers.items():
        if isinstance(answer, NoulAnswer):
            result.append(f"{name}: yes={answer.noul:.3f}")
        elif isinstance(answer, ChoiceAnswer):
            distribution = ", ".join(
                f"{key}={value:.3f}" for key, value in answer.probabilities.items()
            )
            result.append(f"{name}: {distribution}")
        else:
            distribution = ", ".join(
                f"{key}={value:.3f}" for key, value in answer.probabilities.items()
            )
            result.append(f"{name}: score={answer.score:.3f}; levels {distribution}")
    return tuple(result)


def apply_policy(
    case: WorkflowCase, response: DecisionResponse | None
) -> tuple[SuggestedAction, ...]:
    """Suggest actions only; fixtures and original evidence are never changed."""
    answers = response.answers if response is not None else {}
    if set(answers) != set(case.request.questions) or any(
        not valid_answer(question, answers.get(name))
        for name, question in case.request.questions.items()
    ):
        return tuple(
            SuggestedAction(
                label, "review_evidence", "Missing or invalid answers; retain all evidence.", ()
            )
            for _, label in case.items
        )
    results = []
    for key, label in case.items:
        selected = (
            answers if case.name in ("completion", "progress") else {
                name: answer for name, answer in answers.items()
                if name == key or name.startswith(f"{key}_")
            }
        )
        action, reason = _action(case.name, key, selected)
        results.append(SuggestedAction(label, action, reason, _signals(selected)))
    if case.name == "search":
        by_label = {label: key for key, label in case.items}
        results.sort(key=lambda result: _probability(answers, by_label[result.label]), reverse=True)
    return tuple(results)


def _probability(answers: dict[str, Answer], key: str) -> float:
    answer = answers[key]
    if not isinstance(answer, NoulAnswer):
        raise ValueError("Expected a validated Noul answer.")
    return answer.noul


def _action(name: WorkflowName, key: str, answers: dict[str, Answer]) -> tuple[str, str]:
    if name == "search":
        probability = _probability(answers, key)
        if probability >= 0.65:
            return "inspect_first", "Yes >= 0.65; inspect the function before claiming a bug."
        if probability > 0.2:
            return "review_if_needed", "Ambiguous match; keep it available for inspection."
        return "rank_lower", "Yes probability <= 0.20; still retained in the search results."
    if name == "logs":
        priority = answers[f"{key}_priority"]
        value = answers[f"{key}_value"]
        if not isinstance(priority, ChoiceAnswer) or not isinstance(value, ScoreAnswer):
            raise ValueError("Expected validated log answers.")
        if (
            _probability(answers, f"{key}_actionable") <= 0.2
            and priority.probabilities["routine"] >= 0.8
            and value.score < 1.0
        ):
            return "archive_only", "Low-value checks agree; retain original, skip deeper analysis."
        return "investigate", "Failure, useful detail, or uncertainty; retain and analyze."
    if name == "completion":
        if all(
            _probability(answers, key) >= 0.8 for key in ("implemented", "regression", "verified")
        ):
            return "ready_for_review", "All evidence checks >= 0.80; next step is human review."
        if _probability(answers, "regression") < 0.8 or _probability(answers, "verified") < 0.8:
            return "request_regression_evidence", "Coverage or a passing run is not established."
        return "continue_implementation", "The implementation requirement is not established."
    repeated = _probability(answers, "same_assumption")
    progress = _probability(answers, "new_evidence")
    if repeated >= 0.8 and progress <= 0.2:
        return "suggest_replan", "Repeated assumption >= 0.80 and new evidence <= 0.20."
    if progress >= 0.8 and repeated <= 0.2:
        return "continue_investigation", "New evidence >= 0.80 and repeated assumption <= 0.20."
    return "review_trajectory", "Signals are uncertain or disagree; retain and review."
