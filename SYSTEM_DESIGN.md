# Orion system design

## 1. Purpose

Orion is a small research prototype for studying context management in coding
agents that operate over multiple model/tool steps.

The design answers one focused question:

> Can an agent observe its current context state, select an explicit context
> action, execute it through a stable interface, and measure the result?

The project does not assume that retrieval, compression, isolation, or storage
is automatically useful. It records what happened so those hypotheses can be
tested with controlled experiments.

## 2. Design goals

### Provider independence

The agent depends on the small `LLMClient` contract rather than a provider
SDK. Ollama, Groq, and Gemini adapters normalize model responses to the same
`LLMResponse` model containing content, tool calls, finish reason, and usage.

### Policy independence

Policies implement:

```python
def decide(state: State) -> Action:
    ...
```

The agent does not know why an action was selected. This keeps baselines and
adaptive policies comparable and makes future policy implementations
replaceable.

### Storage independence

`ActionExecutor` receives a `ContextStore`. The local prototype uses
`InMemoryContextStore`, while the protocol leaves room for a persistent or
vector-backed implementation later.

### Reproducibility

Experiments use fixed task fixtures, fresh temporary workspaces, the same
provider/model/tool set/horizon, and unique run IDs. Raw per-step JSONL is the
primary record; CSV and plots are derived artifacts.

### Explicit observability

The system separates:

- model input/output token usage,
- context growth,
- retrieval-specific tokens,
- provider latency,
- retrieval latency,
- objective test success,
- agent termination status.

This prevents a passing test or a model request from being mistaken for proof
that a context action was useful.

## 3. Runtime data flow

For each task, `CodingAgent.run` performs the following loop:

1. Build the initial system and user messages.
2. Ingest the task constraint into the context store.
3. Observe the current messages with `StateObserver`.
4. Ask the configured `Policy` for one `Action`.
5. Pass the action, messages, and state to `ActionExecutor`.
6. If retrieval is selected, query the store and append returned items as
   system context.
7. Call the provider with the managed context and tool definitions.
8. Record usage and latency.
9. Execute returned tool calls in the task workspace.
10. Ingest legitimate tool observations such as file contents and test output.
11. Continue until the provider returns no tool calls or the horizon is reached.

The agent returns `completed` only when a model response contains no tool calls.
If every allowed step still contains tool calls, it returns
`horizon_exceeded`.

## 4. Core components

### `CodingAgent`

`CodingAgent` owns the provider/tool loop. It is responsible for orchestration,
not policy decisions or storage algorithms.

It records one step record per model request, including:

- run, task, provider, model, and policy identifiers,
- action,
- context token measurements,
- input/output/total tokens,
- model latency,
- observed state values,
- tool calls and tool latency,
- errors and success fields,
- optional retrieval trace.

### `StateObserver`

`HeuristicStateObserver` computes a deterministic `State` from the current
message sequence. The state includes:

- `context_utilization`,
- `semantic_relevance`,
- `information_freshness`,
- `retrieval_confidence`,
- `remaining_token_budget`,
- `task_phase`.

These are transparent proxies for policy decisions, not claims of semantic
understanding. The observer uses available token metadata and workflow/tool
signals rather than treating words in a prompt as proof of task phase.

### `Policy`

The included policies are:

`NoManagementPolicy`

Always returns `IGNORE`. This is the pass-through baseline.

`StaticPolicy`

Returns `COMPRESS` only when context utilization reaches its configured fixed
threshold; otherwise it returns `IGNORE`.

`RuleBasedOrion`

Uses a fixed decision order:

1. Low semantic relevance -> `RETRIEVE`
2. High context utilization -> `COMPRESS`
3. Low freshness -> `STORE`
4. Testing with high utilization -> `ISOLATE`
5. Otherwise -> `IGNORE`

The thresholds are explicit constructor parameters. The current policy is not
learned and does not implement online adaptation or a contextual bandit.

### `ActionExecutor`

`OrionActionExecutor` translates an action into a context transformation:

- `RETRIEVE`: query the store and append returned content.
- `COMPRESS`: create a bounded condensed context representation.
- `ISOLATE`: switch the active scope.
- `STORE`: persist the latest context content as a context item.
- `IGNORE`: return the context unchanged.

The executor also exposes ingestion for legitimate durable information.

### `ContextStore`

`ContextStore` defines add, retrieve, update, delete, and scoped-context
operations. `InMemoryContextStore` assigns stable per-run item IDs and ranks
matching items using normalized token overlap weighted by item importance.

Stored `ContextItem` metadata includes:

- generated `_id` and `item_id`,
- source type,
- task ID,
- run ID,
- scope,
- importance,
- creation timestamp,
- optional similarity/relevance data.

The store is intentionally deterministic for local validation. It is not
presented as a production vector database.

### Tools

The default workspace tools expose file inspection, file writing, and test
execution. Tool output is returned to the model as observations. The executor
ingests selected observations into the context store so future retrieval can
recover information the agent legitimately encountered.

### Experiment runner

`experiments.py` creates a fresh workspace for every task/policy condition,
constructs the same provider and tool configuration, runs the agent, then
executes the objective test command independently.

The classification is:

```text
objective_test_success = test_result.returncode == 0
task_success = objective_test_success
agent_status = result.status
agent_completed = result.status == "completed"
```

This distinction is necessary because a correct workspace can be reached
before the model stops requesting tools.

## 5. Retrieval instrumentation

Each retrieval event stores a `RetrievalTrace` with:

```text
query
item_ids
content_lengths
relevance_scores
context_before
context_after
context_tokens_before
context_tokens_after
retrieval_tokens_added
retrieval_latency_ms
executed
```

`retrieval_tokens_added` is computed only from newly returned item content. It
is not inferred from the provider's input-token delta. Provider context growth
and retrieval contribution therefore remain separate measurements.

The trace distinguishes:

- selected but not executed,
- executed with no returned context,
- executed with returned context inserted,
- instrumentation failure.

The current compact context snapshots record message counts and content
character counts. They provide an auditable before/after signal without
duplicating the complete conversation into every trace record.

## 6. Evaluation model

The raw JSONL file is the authoritative per-step record. A final
`retrieval_evaluation` event records objective test success, agent status, and
retrieval steps. This event is metadata, not an agent step, and analysis code
must exclude it from step/action statistics.

Recommended analysis order:

1. Verify all expected raw logs exist.
2. Check objective test output and status separately.
3. Analyze action sequences from records with a real step number.
4. Analyze retrieval traces only where the action is `retrieve`.
5. Compare token and latency distributions only across controlled conditions.
6. Label interpretations as observed, possible, or unsupported.

One task-policy run is useful for validating wiring and instrumentation. It is
not enough to establish general efficiency, usefulness, or statistical
superiority.

## 7. Why `refactor_pipeline` is separate

`refactor_pipeline` intentionally creates a more demanding workload:

- multiple modules,
- cross-file imports,
- ordered stages,
- audit behavior,
- several tests,
- revisit cycles.

It is a stress task for context pressure and longer-horizon behavior. Mixing it
with small bounded fixtures in the first comparison would make policy effects
harder to interpret because workload complexity would vary within the same
small sample. It should be evaluated as a separate longer-horizon experiment,
with its own horizon and task-focused analysis.

## 8. Extension points

The architecture supports future work without changing the agent/provider
boundary:

- persistent MongoDB-backed context storage,
- vector or hybrid retrieval behind `ContextStore`,
- richer provenance and deduplication,
- learned policies behind `Policy`,
- repeated runs and confidence intervals,
- task-specific stress suites.

Those extensions should preserve the current separation between observed
execution facts and research interpretation.
