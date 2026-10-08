# Orion

Orion is a provider-agnostic research prototype for **state-aware context
management in long-horizon LLM coding agents**.

The project tests whether an agent can observe the state of its working
context, choose an explicit context-management action, and execute that action
before the next model request. Orion is intentionally interpretable: the
current adaptive policy is a deterministic rule-based policy, not a learned
controller.

> Orion is an experiment platform and reference implementation. It does not
> claim that context management improves coding performance in general.

## What Orion does

At each agent step Orion:

1. Maintains the task conversation and available coding tools.
2. Observes context utilization, semantic relevance, freshness, retrieval
   confidence, token budget, and task phase.
3. Lets a policy select one action:
   `IGNORE`, `RETRIEVE`, `COMPRESS`, `ISOLATE`, or `STORE`.
4. Executes that action through an injectable executor and context store.
5. Sends the resulting context to an interchangeable LLM provider.
6. Records token usage, latency, tool calls, action selection, and retrieval
   traces as structured JSONL.
7. Evaluates the task objectively with its test suite after the agent run.

The objective test result and agent-loop termination are deliberately separate:
an agent can pass the task tests while still reaching the configured horizon.

## System architecture

```text
CodingAgent
    |
    v
StateObserver -----> Policy
    |                   |
    |                   v
    +-------------> ActionExecutor -----> ContextStore
                          |
                          v
                    Provider / LLM
```

The main contracts are:

- `CodingAgent`: provider-neutral tool-calling loop with a bounded horizon.
- `StateObserver`: converts the current messages into a deterministic
  `State`.
- `Policy`: maps a `State` to exactly one `Action`.
- `ActionExecutor`: applies context-management actions without coupling the
  policy to storage.
- `ContextStore`: stores and retrieves durable `ContextItem` records.
- Provider adapters: normalize Ollama, Groq, and Gemini responses to the same
  `LLMResponse` shape.

See [SYSTEM_DESIGN.md](SYSTEM_DESIGN.md) for the detailed data flow,
responsibilities, persistence boundary, and design rationale.

## Policies

Three policies are included for controlled comparisons:

| Policy | Purpose |
|---|---|
| `no_management` | Baseline that always selects `IGNORE`. |
| `static` | Fixed compression-threshold baseline. |
| `rule_based_orion` | Interpretable adaptive policy using relevance, utilization, freshness, and task phase. |

The policy interface is intentionally small so another policy can be added
without changing the agent loop or provider boundary.

## Context retrieval

The local prototype uses `InMemoryContextStore`. It is not a replacement for
the `ContextStore` abstraction; it is the deterministic implementation used by
tests and local experiments.

Legitimate context enters the store from:

- task constraints,
- file contents returned by `read_file`,
- test results returned by `run_tests`,
- other explicit agent facts or decisions routed through the executor.

Each stored item preserves content, source type, task and run identifiers,
scope, importance, timestamp, and item identifiers. Retrieval uses normalized
token overlap with deterministic relevance ranking. A future persistent or
vector-backed store can implement the same protocol.

Every retrieval step records:

- query and item IDs,
- content lengths and relevance scores,
- context snapshots before and after retrieval,
- provider context tokens before and after retrieval,
- tokens contributed by retrieved content only,
- retrieval latency and execution status.

## Tasks and experiments

Controlled fixtures are defined in `src/orion/tasks.py`:

- `add_numbers`
- `fix_greeting`
- `complete_config`
- `refactor_pipeline`

The first three are bounded comparison tasks. `refactor_pipeline` is a
cross-file, revisit-oriented stress task and should be analyzed separately as
a longer-horizon workload rather than mixed into the first small comparison.

The experiment runner keeps the task prompt, fixture, provider, model, tool
set, and horizon constant while varying the policy. For each run it creates a
fresh workspace, preserves raw JSONL records, exports CSV data, and generates
plots.

### Local setup

Requirements:

- Python 3.11+
- Ollama for local LLM experiments
- An Ollama model with tool-calling support, such as `qwen3:8b`

```sh
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

The interpreter path above is the documented macOS setup. Any Python 3.11+
interpreter may be used if it is available on the system.

### CLI

Show available commands:

```sh
python -m orion --help
```

Run one coding task with one policy:

```sh
python -m orion run \
  --policy rule_based_orion \
  --provider ollama \
  --model qwen3:8b \
  --task-id add_numbers \
  --horizon 5 \
  --prompt "Implement the requested task using the available tools."
```

Run the controlled comparison:

```sh
python -m orion experiment \
  --provider ollama \
  --model qwen3:8b \
  --horizon 5 \
  --output-dir results/local-comparison \
  --experiment-id local-comparison
```

Run one fixture under all three policies:

```sh
python -m orion experiment \
  --provider ollama \
  --model qwen3:8b \
  --horizon 5 \
  --task-id add_numbers \
  --output-dir results/add-numbers-h5 \
  --experiment-id add-numbers-h5
```

Analyze existing JSONL without running a provider:

```sh
python -m orion analyze \
  --input results/add-numbers-h5/raw \
  --plots results/add-numbers-h5/plots
```

The `analyze` command reads existing logs only. It does not invoke Ollama.

## Providers and configuration

Provider adapters are implemented for:

- `ollama` (local HTTP service; recommended for reproducible development),
- `groq`,
- `gemini`.

Optional environment variables are documented in [.env.example](.env.example)
and [SETUP_REQUIRED.md](SETUP_REQUIRED.md):

- `OLLAMA_BASE_URL`
- `GROQ_API_KEY`
- `GEMINI_API_KEY`
- `ORION_WORKSPACE_ROOT`

Never commit real credentials. `.env` is ignored by git.

MongoDB is an optional future persistence boundary; local tests and prototype
experiments use the in-memory store.

## Validation and interpretation

Run the test suite with:

```sh
python -m pytest -q
```

The repository's experiment records are designed to support measurement, not
automatic research claims. In particular:

- `task_success` means the objective task test passed.
- `agent_status` describes the agent loop, such as `completed` or
  `horizon_exceeded`.
- `agent_completed` is true only when the agent terminates before the horizon.
- A single task-policy run is not evidence of statistical significance or
  general superiority.

Generated experiment results are ignored under `results/` so raw logs and
derived plots are not accidentally committed with source changes.

## Completed experiment record

The following experiments were run locally with Ollama and `qwen3:8b`. The
results are documented here as implementation evidence, not as claims of
general model or policy superiority.

### Initial four-task comparison

Configuration:

- Tasks: `add_numbers`, `fix_greeting`, `complete_config`, `refactor_pipeline`
- Policies: `no_management`, `static`, `rule_based_orion`
- Provider/model: Ollama / `qwen3:8b`
- Horizon: 5
- Matrix: 12 task-policy runs

This run established that the agent loop, policy variants, raw JSONL logging,
CSV export, and objective test evaluation worked across all four fixtures.
The original summary incorrectly treated a passing objective test combined
with `horizon_exceeded` as task failure. That classification was corrected so
`task_success` now means only that the objective test passed, while
`agent_status` and `agent_completed` remain separate fields.

The initial logs selected `RETRIEVE` three times for RuleBasedOrion:

- `add_numbers`: steps 2 and 3
- `refactor_pipeline`: step 2

Those pre-ingestion retrievals returned no context items. This exposed a real
prototype limitation: the in-memory store had not yet received enough
legitimate task information to make retrieval useful.

### Retrieval instrumentation validation

Two targeted runs (`add_numbers` and `refactor_pipeline`, both under
RuleBasedOrion) validated the retrieval trace and reproduced the empty-store
case. Five retrieval events were selected and executed, but all returned zero
items. The traces correctly showed no active-context change and zero
retrieval-contributed tokens.

The retrieval path was then corrected without changing policy thresholds or
decision rules. The local `InMemoryContextStore` now receives task
constraints, file contents, and test observations through explicit ingestion.
Retrieval accounting was separated from provider input-token growth.

A focused local validation reached 22 passing tests and proved that a stored
context item could be retrieved, identified, and inserted into active context.

### End-to-end retrieval validation

One real `add_numbers` / RuleBasedOrion / Ollama run validated the corrected
path:

1. legitimate context was ingested,
2. `RETRIEVE` was selected,
3. stored items were returned,
4. retrieved content was inserted into active context,
5. item IDs, relevance scores, token contribution, snapshots, and retrieval
   latency were persisted.

This validated the wiring and observability of retrieval. It was not a
performance comparison.

### Post-fix three-policy comparison

Configuration:

- Task: `add_numbers`
- Policies: `no_management`, `static`, `rule_based_orion`
- Provider/model: Ollama / `qwen3:8b`
- Horizon: 5
- One fresh workspace per policy

All three objective test suites passed. All three agents reached the
five-step horizon and therefore had `agent_completed=false`.

| Policy | Objective test | Status | Steps | Input tokens | Output tokens | Total tokens | Latency |
|---|---:|---|---:|---:|---:|---:|---:|
| No Management | Pass | `horizon_exceeded` | 5 | 1,762 | 3,030 | 4,792 | 158.3 s |
| Static | Pass | `horizon_exceeded` | 5 | 1,978 | 4,860 | 6,878 | 372.5 s |
| RuleBasedOrion | Pass | `horizon_exceeded` | 5 | 1,745 | 2,243 | 3,988 | 188.4 s |

Observed action sequences:

```text
No Management:    IGNORE -> IGNORE -> IGNORE -> IGNORE -> IGNORE
Static:           IGNORE -> IGNORE -> IGNORE -> IGNORE -> IGNORE
RuleBasedOrion:   IGNORE -> RETRIEVE -> RETRIEVE -> RETRIEVE -> RETRIEVE
```

RuleBasedOrion produced four successful retrieval events:

| Step | Item IDs | Relevance scores | Retrieval tokens added | Context tokens before -> after |
|---:|---|---|---:|---|
| 2 | `2`, `1` | `0.80`, `0.18` | 47 | 256 -> 340 |
| 3 | `1` | `0.45` | 32 | 340 -> 353 |
| 4 | `1` | `0.45` | 32 | 353 -> 383 |
| 5 | `1` | `0.45` | 32 | 383 -> 413 |

The final recorded context sizes were 505 tokens for No Management, 544 for
Static, and 413 for RuleBasedOrion. In this single task and single run per
policy, RuleBasedOrion used fewer total tokens than the two baselines, while
No Management had lower latency than RuleBasedOrion. All policies passed, so
these records do not establish that retrieval caused the token difference or
improved task performance.

### What the experiments establish

**Observed**

- The five-action policy/executor architecture runs against a real coding
  agent.
- Objective test success and agent termination are recorded separately.
- RuleBasedOrion selected real retrieval actions.
- The corrected in-memory retrieval path returned stored items and inserted
  them into active context.
- Retrieval-specific token contribution and latency were persisted.
- Raw JSONL, CSV, and plot generation were validated.

**Possible interpretation**

- Explicit context management can change the active context and model token
  measurements during execution.
- A populated context store may allow a policy to recover previously observed
  task information.

**Not supported**

- Orion is not shown to be generally better than either baseline.
- No statistical significance or general efficiency claim can be made.
- Retrieval is not shown to be necessary for task correctness.
- The single-task results do not establish behavior on other tasks, models,
  horizons, or providers.

## Repository layout

```text
src/orion/
  agent.py         Agent loop and provider-neutral tool contracts
  analysis.py      JSONL loading and plot generation
  context.py       ContextStore protocol and in-memory implementation
  executor.py      Context actions, ingestion, and retrieval traces
  experiments.py   Controlled experiment runner and evaluation semantics
  models.py        Actions, states, and context-item models
  observer.py      Deterministic state observation
  policy.py        Baseline and RuleBasedOrion policies
  providers.py     Shared provider setup/error contracts
  ollama.py        Ollama adapter
  groq.py          Groq adapter
  gemini.py        Gemini adapter
  tasks.py         Reproducible coding fixtures
  tools.py         Workspace coding tools
tests/             Unit and integration-oriented component tests
SYSTEM_DESIGN.md  Detailed design and rationale
```

## Status

The implementation and retrieval instrumentation have been validated locally
and in a real end-to-end Ollama run. The next planned comparison is:

- Tasks: `add_numbers`, `complete_config`, `fix_greeting`
- Policies: `no_management`, `static`, `rule_based_orion`
- Provider/model: Ollama / `qwen3:8b`
- Horizon: 5

That comparison has not been included in this repository as a new result and
should be run as a fresh experiment with preserved raw logs. The
`refactor_pipeline` fixture should remain a separate longer-horizon stress
experiment because it exercises multi-file dependencies and revisit cycles.
No broad performance conclusion should be drawn from the existing
single-task validation.
