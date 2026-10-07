# Orion setup

## Required

Install the package and development dependencies:

```sh
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

For local experiments install Ollama, run `ollama serve`, and pull the model
selected in `ExperimentConfig` (the default is `llama3.2`). Results are written
under `results/` as raw JSONL; use `ExperimentLogger.export_csv` for derived CSV.

## Optional cloud providers

Set `GROQ_API_KEY` for `GroqProvider` or `GEMINI_API_KEY` for
`GeminiProvider`. Set `OLLAMA_BASE_URL` when Ollama is not on its default
`http://localhost:11434`. Never commit real credentials; `.env` is ignored.

MongoDB is optional for this sprint because tests and local development use
`InMemoryContextStore`.

Run the first real comparison:

```sh
python -m orion experiment --provider ollama --model llama3.2 --horizon 10
```

The command executes the three policies over the same three fixtures. It
creates unique run IDs, writes raw JSONL to `results/raw/`, CSV to
`results/csv/`, and plots to `results/plots/`. If Ollama is unavailable, each
run is marked failed with a setup error rather than fabricating metrics.
For the first one-task validation, use
`--task-id add_numbers`; this produces exactly three policy runs.

## Future

The next phase adds `BanditOrion` behind the existing `Policy` interface and
uses the per-step JSONL records to calculate task-progress minus token-cost
reward. No bandit or experimental result is included yet.
