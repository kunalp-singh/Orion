"""Small command-line entry point for local Orion experiments."""

import argparse
import json
from pathlib import Path

from .analysis import generate_plots
from .agent import AgentTask, CodingAgent
from .experiments import ExperimentConfig, build_components, run_comparison
from .gemini import GeminiProvider
from .groq import GroqProvider
from .ollama import OllamaProvider
from .providers import ProviderSetupError
from .settings import get_settings
from .tasks import TASKS, task_by_id
from .tools import default_tools


def _provider(name: str, model: str):
    settings = get_settings()
    if name == "ollama":
        return OllamaProvider(model, settings.ollama_base_url)
    if name == "groq":
        return GroqProvider(settings.groq_api_key or "", model)
    if name == "gemini":
        return GeminiProvider(settings.gemini_api_key or "", model)
    raise ValueError(f"Unknown provider: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m orion")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--policy", default="rule_based_orion")
    run.add_argument("--provider", default="ollama")
    run.add_argument("--model", default="llama3.2")
    run.add_argument("--prompt", required=True)
    run.add_argument("--task-id", default="cli-task")
    run.add_argument("--horizon", type=int, default=10)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--csv")
    analyze.add_argument("--plots", default="results/plots")
    experiment = subparsers.add_parser("experiment")
    experiment.add_argument("--provider", default="ollama")
    experiment.add_argument("--model", default="llama3.2")
    experiment.add_argument("--horizon", type=int, default=10)
    experiment.add_argument("--output-dir", default="results")
    experiment.add_argument("--experiment-id", default="local-orion")
    experiment.add_argument("--task-id", default=None)
    args = parser.parse_args()
    if args.command == "analyze":
        from .logging import ExperimentLogger
        input_path = Path(args.input)
        sources = sorted(input_path.glob("*.jsonl")) if input_path.is_dir() else [input_path]
        if not sources:
            raise SystemExit(f"No JSONL experiment logs found in {input_path}")
        output = args.csv or str(input_path.with_suffix(".csv"))
        try:
            print(ExperimentLogger(sources[0]).export_csv(output))
            print("\n".join(str(path) for path in generate_plots(sources, args.plots)))
        except ValueError as error:
            raise SystemExit(f"Analysis unavailable: {error}") from error
        return
    if args.command == "experiment":
        config = ExperimentConfig(
            provider=args.provider,
            model=args.model,
            horizon=args.horizon,
            output_dir=Path(args.output_dir),
            experiment_id=args.experiment_id,
        )
        tasks = (task_by_id(args.task_id),) if args.task_id else TASKS
        summaries = run_comparison(config, tasks, _provider)
        print(json.dumps(summaries, indent=2))
        return
    config = ExperimentConfig(policy=args.policy, provider=args.provider, model=args.model, horizon=args.horizon)
    policy, observer, executor, logger = build_components(config)
    provider = _provider(args.provider, args.model)
    try:
        if hasattr(provider, "check_setup"):
            try:
                provider.check_setup()
            except ProviderSetupError as error:
                raise SystemExit(str(error)) from error
        result = CodingAgent(
            provider,
            default_tools(Path(get_settings().workspace_root)),
            observer=observer,
            policy=policy,
            action_executor=executor,
            logger=logger,
        ).run(AgentTask(task_id=args.task_id, prompt=args.prompt, horizon_limit=args.horizon))
        print(json.dumps(result.model_dump(), default=str))
    finally:
        provider.close()


if __name__ == "__main__":
    main()
