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

The current implementation validates the local retrieval path end to end and
persists retrieval observability in agent step records. The next research step
is a fresh multi-task comparison across the three policies. That experiment
must be run separately from the longer-horizon `refactor_pipeline` stress
task, and no broad performance conclusion should be drawn from the existing
single-task validation.
