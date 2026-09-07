"""Safe local tools exposed to the coding agent."""

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path


class WorkspaceTools:
    """Read, write, and test files inside a constrained workspace."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path(self, relative_path: str) -> Path:
        candidate = (self.root / relative_path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError("path must stay inside the workspace")
        return candidate

    def read_file(self, arguments: Mapping[str, object]) -> str:
        path = self._path(self._required_string(arguments, "path"))
        return path.read_text(encoding="utf-8")

    def write_file(self, arguments: Mapping[str, object]) -> str:
        path = self._path(self._required_string(arguments, "path"))
        content = self._required_string(arguments, "content")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return f"Wrote {path.relative_to(self.root)}"

    def run_tests(self, arguments: Mapping[str, object]) -> str:
        command_value = arguments.get("command", ["python", "-m", "pytest"])
        if not isinstance(command_value, list) or not all(
            isinstance(part, str) for part in command_value
        ):
            raise ValueError("command must be a list of strings")
        result = subprocess.run(
            command_value,
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )
        output = (result.stdout + result.stderr).strip()
        return f"exit_code={result.returncode}\n{output}"

    @staticmethod
    def _required_string(arguments: Mapping[str, object], name: str) -> str:
        value = arguments.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string")
        return value


class ReadFileTool:
    name = "read_file"

    def __init__(self, workspace: WorkspaceTools) -> None:
        self.workspace = workspace

    def definition(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": {"type": "object", "required": ["path"]},
        }

    def invoke(self, arguments: Mapping[str, object]) -> str:
        return self.workspace.read_file(arguments)


class WriteFileTool:
    name = "write_file"

    def __init__(self, workspace: WorkspaceTools) -> None:
        self.workspace = workspace

    def definition(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "description": "Write UTF-8 text to a file in the workspace.",
            "parameters": {"type": "object", "required": ["path", "content"]},
        }

    def invoke(self, arguments: Mapping[str, object]) -> str:
        return self.workspace.write_file(arguments)


class RunTestsTool:
    name = "run_tests"

    def __init__(self, workspace: WorkspaceTools) -> None:
        self.workspace = workspace

    def definition(self) -> Mapping[str, object]:
        return {
            "name": self.name,
            "description": "Run a test command in the workspace.",
            "parameters": {"type": "object", "properties": {"command": {"type": "array"}}},
        }

    def invoke(self, arguments: Mapping[str, object]) -> str:
        return self.workspace.run_tests(arguments)


def default_tools(root: Path) -> Sequence[object]:
    """Build the standard coding-tool set for a workspace."""

    workspace = WorkspaceTools(root)
    return [ReadFileTool(workspace), WriteFileTool(workspace), RunTestsTool(workspace)]
