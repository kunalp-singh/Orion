from pathlib import Path

from orion.agent import AgentTask, CodingAgent, LLMResponse, ToolCall
from orion.tools import ReadFileTool, WorkspaceTools, WriteFileTool


class FakeClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, messages, tools) -> LLMResponse:
        self.calls += 1
        assert tools
        return next(self.responses)


def test_agent_executes_tool_then_completes(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("hello", encoding="utf-8")
    client = FakeClient(
        [
            LLMResponse(
                content="",
                tool_calls=[ToolCall(name="read_file", arguments={"path": "input.txt"})],
                finish_reason="tool_call",
            ),
            LLMResponse(content="The file says hello."),
        ]
    )
    agent = CodingAgent(client, [ReadFileTool(WorkspaceTools(tmp_path))])

    result = agent.run(AgentTask(task_id="read", prompt="Read input.txt"))

    assert result.status == "completed"
    assert result.steps == 2
    assert result.response == "The file says hello."
    assert {"role": "tool", "name": "read_file", "content": "hello"} in result.messages


def test_agent_stops_at_horizon(tmp_path: Path) -> None:
    client = FakeClient(
        [LLMResponse(tool_calls=[ToolCall(name="write_file", arguments={"path": "x", "content": "x"})])]
    )
    agent = CodingAgent(client, [WriteFileTool(WorkspaceTools(tmp_path))])

    result = agent.run(AgentTask(task_id="write", prompt="Keep writing", horizon_limit=1))

    assert result.status == "horizon_exceeded"
    assert result.steps == 1
    assert (tmp_path / "x").read_text(encoding="utf-8") == "x"
