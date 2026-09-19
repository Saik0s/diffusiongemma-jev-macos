# Where the examples come from

These are public projects that use Jev for concrete software decisions.
We reviewed their source and documentation on **September 18, 2026**.
This verifies the shared designs, not their authors' performance claims.

Our examples are original, small teaching fixtures inspired by these workflows.
They run the local DiffusionGemma adapter, which has different training and behavior from Jev.

## Search for behavior in code

[Every](https://github.com/sufianetaouil/every) asks a yes/no question about each function and ranks the answers.
Its examples include ignored exceptions, unbounded retries, and tests without assertions.
This is useful when relevant behavior uses different words than your search query.

Our [semantic-search demo](examples.md#find-functions-worth-reading) compares six short functions and produces an inspection order.
It treats matches as candidates to review. A function alone may omit important behavior in its callers.

## Spend investigation effort on useful logs

[Jev Logs](https://github.com/reachjalil/jevlogs) evaluates diagnostic value, priority, and actionability before a larger model investigates.
Its design preserves the original log archive and has explicit fallback behavior.

Our [log-triage demo](examples.md#turn-a-log-stream-into-an-investigation-bundle) mixes routine messages, recovered failures, and unresolved errors.
It demonstrates all three answer types and a code-owned selection policy.

## Check an agent's evidence before accepting “done”

[Foreman](https://github.com/thruwire/foreman) supervises coding agents using requirements, test evidence, progress signals, and deterministic interventions.
Its author describes it as an architectural experiment.

Our [completion demo](examples.md#check-whether-a-fix-has-enough-evidence) compares a patch with only a happy-path test against a better-verified snapshot.
The output suggests the next verification step. It does not certify the patch or merge anything.

## Recognize activity that is not progress

[ProgressGate](https://github.com/AshutoshVJTI/progressgate) checks whether an agent keeps acting on a rejected assumption.
Different commands can still repeat the same unsuccessful approach.
Its policy distinguishes continuing, warning, replanning, and halting; automatic halt is disabled by default.

Our [progress demo](examples.md#notice-a-stalled-agent-loop) contrasts repeated retries with new diagnostic evidence.
It suggests a replan without controlling a real agent.

## Keep useful context without rewriting it

[fast-jev-compaction](https://github.com/tamaratran/fast-jev-compaction) judges whether to retain a tool call and its full result.
Its policy can preserve, shorten, or remove a pair while protecting pinned messages.
Its animated showcase is explicitly scripted, so we do not use that animation as performance evidence.

Our [compaction example](examples.md#preview-context-compaction) uses a simpler whole-pair policy.
It preserves retained text exactly and keeps the original transcript if decisions are invalid or unavailable.

## Further designs worth studying

[Jev Review](https://github.com/devagrawal09/jev-review) stages risk screening, evidence selection, severity scoring, and reviewer routing.
Its findings are prompts for review, not proof of defects.
The existing patch-review smoke example here illustrates a much smaller part of that design.

[LangChain's Jev harness guide](https://www.langchain.com/blog/building-a-harness-with-jev) demonstrates model routing and proposed-tool-call classification.
A classifier can inform a route; it should not become the only security check around tool execution.

[NanoJev](https://github.com/TianyuCodings/NanoJev) uses small trained decision heads in maze and Snake controllers.
It demonstrates how code can own rules and memory while a model supplies judgments.
Its game results do not measure coding ability or transfer directly to this project.

[Simple-JEV](https://github.com/featherless-ai/simple-jev), from Featherless, serves the same request shape from open Hugging Face models by reading next-token logits after an assistant JSON prefill.
It is the closest open design to this project, with a different model family and different Noul and confidence semantics.
Its repository carries no license file, so we describe its ideas in [the Jev comparison](jev-differences.md#simple-jev-an-open-reimplementation-on-autoregressive-models) and copied nothing from it.

## Benchmarks people have published

[jev-rerank-bench](https://github.com/anessbelbati/jev-rerank-bench) compares Jev with other rerankers using shared candidate lists and saved responses.
We adapt its **CodeSearchNet Python** task: given a description, rank real Python functions by relevance.
The [local evaluation](retrieval-benchmark.md) reports our method and recomputes references on the same selected queries.

[jev-code-review-benchmark](https://github.com/gemanor/jev-code-review-benchmark) uses supplied rules and constructed program variants.
It is relevant to rule checking, but we chose real-code retrieval to complement our existing synthetic smoke checks.

The reranking study also uses **BRIGHT StackOverflow**, a harder software-question retrieval task.
We document it as a possible next evaluation; we do not claim to have run it here.
