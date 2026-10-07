from pathlib import Path
from types import SimpleNamespace

from orion.agent import AgentTask, CodingAgent, LLMResponse, ToolCall
from orion.models import Action, State, TaskPhase
from orion.tools import ReadFileTool, WorkspaceTools, WriteFileTool


class FakeClient:
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def complete(self, messages, tools) -> LLMResponse:
        self.calls += 1
        assert tools
        return next(self.responses)


class RetrievalExecutor:
    last_retrieval = SimpleNamespace(
        query="important constraint",
        item_ids=["1"],
        content_lengths=[21],
        relevance_scores=[None],
        context_before={"message_count": 2, "content_chars": 10},
        context_after={"message_count": 3, "content_chars": 31},
        context_tokens_before=None,
        context_tokens_after=None,
        retrieval_tokens_added=0,
        retrieval_latency_ms=0.2,
        executed=True,
    )

    def execute(self, action, context, state):
        assert action is Action.RETRIEVE
        return context


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


def test_agent_persists_retrieval_trace_in_step_record(tmp_path: Path) -> None:
    state = State(
        context_utilization=0.1,
        semantic_relevance=0.1,
        information_freshness=0.9,
        retrieval_confidence=0.5,
        remaining_token_budget=0.9,
        task_phase=TaskPhase.PLANNING,
    )

    class Observer:
        def observe(self, context):
            return state

    class Policy:
        def decide(self, observed):
            return Action.RETRIEVE

    result = CodingAgent(
        FakeClient([LLMResponse(content="done", usage={"input_tokens": 42})]),
        [ReadFileTool(WorkspaceTools(tmp_path))],
        observer=Observer(),
        policy=Policy(),
        action_executor=RetrievalExecutor(),
    ).run(AgentTask(task_id="retrieve", prompt="Use context", horizon_limit=1))

    assert result.records[0]["retrieval"]["query"] == "important constraint"
    assert result.records[0]["retrieval"]["context_tokens_after"] == 42
