"""Small reproducible experiment configuration and policy factory."""

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import uuid4

from .agent import AgentResult, AgentTask, CodingAgent
from .context import InMemoryContextStore
from .executor import OrionActionExecutor
from .logging import ExperimentLogger
from .observer import HeuristicStateObserver
from .policy import NoManagementPolicy, RuleBasedOrion, StaticPolicy, Policy
from .tasks import CodingTask, create_workspace, remove_workspace
from .tools import default_tools


def classify_task_result(agent_status: str, objective_test_success: bool) -> dict[str, Any]:
    """Separate objective correctness from agent-loop termination."""

    return {
        "task_success": objective_test_success,
        "agent_status": agent_status,
        "agent_completed": agent_status == "completed",
    }


@dataclass
class ExperimentConfig:
    policy: str = "rule_based_orion"
    provider: str = "ollama"
    model: str = "llama3.2"
    horizon: int = 10
    repetitions: int = 1
    context_budget: int = 8_000
    output_dir: Path = Path("results")
    tasks: list[dict[str, Any]] = field(default_factory=list)
    experiment_id: str = "local-orion"


def policy_from_name(name: str) -> Policy:
    policies: dict[str, Policy] = {
        "no_management": NoManagementPolicy(),
        "static": StaticPolicy(),
        "rule_based": RuleBasedOrion(),
        "rule_based_orion": RuleBasedOrion(),
    }
    try:
        return policies[name]
    except KeyError as error:
        raise ValueError(f"Unknown policy: {name}") from error


def build_components(config: ExperimentConfig) -> tuple[Policy, HeuristicStateObserver, OrionActionExecutor, ExperimentLogger]:
    output = config.output_dir / "raw" / f"{config.experiment_id}.jsonl"
    return (
        policy_from_name(config.policy),
        HeuristicStateObserver(config.context_budget),
        OrionActionExecutor(InMemoryContextStore()),
        ExperimentLogger(output),
    )


def run_comparison(
    config: ExperimentConfig,
    tasks: tuple[CodingTask, ...],
    provider_factory,
    workspace_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Run identical task fixtures under each policy and preserve raw records."""

    config.output_dir.mkdir(parents=True, exist_ok=True)
    policies = ("no_management", "static", "rule_based_orion")
    summaries: list[dict[str, Any]] = []
    raw_paths: list[Path] = []
    for task in tasks:
        for policy_name in policies:
            run_id = f"{config.experiment_id}-{task.task_id}-{policy_name}-{uuid4().hex[:8]}"
            workspace = create_workspace(task, workspace_root)
            provider = provider_factory(config.provider, config.model)
            policy = policy_from_name(policy_name)
            logger = ExperimentLogger(config.output_dir / "raw" / f"{run_id}.jsonl")
            raw_paths.append(logger.path)
            try:
                if hasattr(provider, "check_setup"):
                    provider.check_setup()
                agent = CodingAgent(
                    provider,
                    default_tools(workspace),
                    observer=HeuristicStateObserver(config.context_budget),
                    policy=policy,
                    action_executor=OrionActionExecutor(InMemoryContextStore()),
                    logger=logger,
                    run_id=run_id,
                    policy_name=policy_name,
                    provider_name=getattr(provider, "provider_name", config.provider),
                    model_name=getattr(provider, "model_name", config.model),
                )
                result = agent.run(
                    AgentTask(task_id=task.task_id, prompt=task.prompt, horizon_limit=config.horizon)
                )
                test_result = subprocess.run(
                    [sys.executable, *task.test_command[1:]],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                objective_test_success = test_result.returncode == 0
                retrieval_evaluation = {
                    "run_id": run_id,
                    "task_id": task.task_id,
                    "policy": policy_name,
                    "event": "retrieval_evaluation",
                    "objective_test_success": objective_test_success,
                    "agent_status": result.status,
                    "retrieval_steps": [
                        record["step"]
                        for record in result.records
                        if record.get("action") == "retrieve"
                    ],
                }
                logger.log(retrieval_evaluation)
                summary = {
                    "run_id": run_id,
                    "task_id": task.task_id,
                    "policy": policy_name,
                    "provider": config.provider,
                    "model": config.model,
                    "horizon": config.horizon,
                    "status": result.status,
                    "steps": result.steps,
                    **classify_task_result(result.status, objective_test_success),
                    "test_output": (test_result.stdout + test_result.stderr).strip(),
                    "raw_log": str(logger.path),
                    "error": None,
                }
                logger.export_csv(config.output_dir / "csv" / f"{run_id}.csv")
            except Exception as error:
                logger.log({
                    "run_id": run_id,
                    "task_id": task.task_id,
                    "step": None,
                    "policy": policy_name,
                    "provider": config.provider,
                    "model": config.model,
                    "action": None,
                    "task_phase": None,
                    "context_tokens_before": None,
                    "context_tokens_after": None,
                    "context_utilization": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "latency_ms": None,
                    "semantic_relevance": None,
                    "information_freshness": None,
                    "retrieval_confidence": None,
                    "remaining_token_budget": None,
                    "tool_calls": [],
                    "tool_latency_ms": None,
                    "success": False,
                    "error": str(error),
                    "error_type": type(error).__name__,
                })
                summary = {
                    "run_id": run_id,
                    "task_id": task.task_id,
                    "policy": policy_name,
                    "provider": config.provider,
                    "model": config.model,
                    "horizon": config.horizon,
                    "status": "failed",
                    "steps": 0,
                    "raw_log": str(logger.path),
                    "error": str(error),
                }
            finally:
                provider.close()
                remove_workspace(workspace)
            summaries.append(summary)
    summary_path = config.output_dir / f"{config.experiment_id}_runs.json"
    summary_path.write_text(__import__("json").dumps(summaries, indent=2), encoding="utf-8")
    from .analysis import generate_plots

    import json

    usable_paths = [
        path
        for path in raw_paths
        if path.exists()
        and any(
            json.loads(line).get("step") is not None
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        )
    ]
    if usable_paths:
        generate_plots(usable_paths, config.output_dir / "plots")
    return summaries
