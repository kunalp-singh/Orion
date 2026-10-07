"""Small deterministic coding-task fixtures for local Orion experiments."""

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile


@dataclass(frozen=True)
class CodingTask:
    task_id: str
    description: str
    expected_outcome: str
    test_command: list[str]
    files: dict[str, str]
    prompt: str
    setup: str = "No additional setup."


TASKS = (
    CodingTask(
        task_id="add_numbers",
        description="Read a small arithmetic module, implement addition, and run its tests.",
        expected_outcome="add_numbers returns the sum of two integers.",
        test_command=["python", "-m", "pytest", "-q"],
        files={
            "calculator.py": "def add(a: int, b: int) -> int:\n    raise NotImplementedError\n",
            "test_calculator.py": "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        },
        prompt="Inspect calculator.py and test_calculator.py. Implement add correctly and run the test command. Use the provided run_tests tool.",
    ),
    CodingTask(
        task_id="fix_greeting",
        description="Find and fix a formatting bug in a greeting helper.",
        expected_outcome="greet('Orion') returns 'Hello, Orion!'.",
        test_command=["python", "-m", "pytest", "-q"],
        files={
            "greeting.py": "def greet(name: str) -> str:\n    return f'Hello {name}'\n",
            "test_greeting.py": "from greeting import greet\n\n\ndef test_greet():\n    assert greet('Orion') == 'Hello, Orion!'\n",
        },
        prompt="Read the greeting implementation and its test. Fix the formatting bug, then use the run_tests tool.",
    ),
    CodingTask(
        task_id="complete_config",
        description="Read a configuration helper, add a default timeout, and verify behavior.",
        expected_outcome="load_config returns timeout=30 when no timeout is supplied.",
        test_command=["python", "-m", "pytest", "-q"],
        files={
            "config.py": "def load_config(values: dict[str, object]) -> dict[str, object]:\n    return dict(values)\n",
            "test_config.py": "from config import load_config\n\n\ndef test_default_timeout():\n    assert load_config({})['timeout'] == 30\n",
        },
        prompt="Inspect config.py and its test. Add the required default timeout without removing existing values, then use the run_tests tool.",
    ),
    CodingTask(
        task_id="refactor_pipeline",
        description="Refactor a small multi-module pipeline while preserving cross-file behavior and tests.",
        expected_outcome="The pipeline applies configured stages in order and records each stage in the audit log.",
        test_command=["python", "-m", "pytest", "-q"],
        files={
            "pipeline/models.py": (
                "from dataclasses import dataclass\n\n"
                "@dataclass\nclass PipelineConfig:\n"
                "    stages: list[str]\n"
                "    audit_prefix: str = 'stage'\n"
            ),
            "pipeline/stages.py": (
                "from collections.abc import Callable\n\n"
                "def normalize(value: str) -> str:\n    return value.strip().lower()\n\n"
                "def decorate(value: str) -> str:\n"
                "    return f'decorated:{value}'\n\n"
                "STAGES: dict[str, Callable[[str], str]] = {'normalize': normalize}\n"
            ),
            "pipeline/runner.py": (
                "from .models import PipelineConfig\n"
                "from .stages import STAGES\n\n"
                "def run(value: str, config: PipelineConfig) -> tuple[str, list[str]]:\n"
                "    audit: list[str] = []\n"
                "    for stage in config.stages:\n"
                "        value = STAGES[stage](value)\n"
                "        audit.append(f'{config.audit_prefix}:{stage}')\n"
                "    return value, audit\n"
            ),
            "pipeline/report.py": (
                "def render(value: str, audit: list[str]) -> str:\n"
                "    return value + ' [' + ','.join(audit) + ']'\n"
            ),
            "tests/test_runner.py": (
                "from pipeline.models import PipelineConfig\n"
                "from pipeline.runner import run\n\n"
                "def test_pipeline_order_and_audit():\n"
                "    value, audit = run('  Orion  ', PipelineConfig(['normalize', 'decorate']))\n"
                "    assert value == 'decorated:orion'\n"
                "    assert audit == ['stage:normalize', 'stage:decorate']\n"
            ),
            "tests/test_report.py": (
                "from pipeline.report import render\n\n"
                "def test_report_uses_audit():\n"
                "    assert render('orion', ['stage:normalize']) == 'orion [stage:normalize]'\n"
            ),
            "tests/test_contract.py": (
                "from pipeline.models import PipelineConfig\n\n"
                "def test_config_keeps_stage_contract():\n"
                "    config = PipelineConfig(['normalize'])\n"
                "    assert config.stages[0] == 'normalize'\n"
            ),
        },
        prompt=(
            "Inspect every pipeline module and all three tests before editing. "
            "Add a decorate stage, make the runner support it using the shared "
            "configuration and audit contract, update tests, run tests, then "
            "revisit the runner to ensure normalize behavior remains unchanged. "
            "Use the provided tools and preserve cross-file interfaces."
        ),
    ),
)


def task_by_id(task_id: str) -> CodingTask:
    for task in TASKS:
        if task.task_id == task_id:
            return task
    raise ValueError(f"Unknown task: {task_id}")


def create_workspace(task: CodingTask, root: Path | None = None) -> Path:
    workspace = Path(tempfile.mkdtemp(prefix=f"orion-{task.task_id}-", dir=root))
    for relative, content in task.files.items():
        path = workspace / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return workspace


def remove_workspace(workspace: Path) -> None:
    shutil.rmtree(workspace, ignore_errors=False)
