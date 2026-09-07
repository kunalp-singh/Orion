"""Provider-agnostic coding-agent loop and function-calling contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class ToolCall(BaseModel):
    """A function call requested by an LLM."""

    model_config = ConfigDict(extra="forbid")

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Normalized response returned by an LLM adapter."""

    model_config = ConfigDict(extra="forbid")

    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"


class LLMClient(Protocol):
    """Minimal interface required by the agent loop."""

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMResponse:
        """Generate the next response from the current conversation."""


class Tool(Protocol):
    """Interface implemented by coding tools."""

    name: str

    def definition(self) -> Mapping[str, Any]:
        """Return provider-neutral function metadata."""

    def invoke(self, arguments: Mapping[str, Any]) -> str:
        """Execute the tool and return a serializable observation."""


class AgentTask(BaseModel):
    """A bounded coding task given to the agent."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    prompt: str
    horizon_limit: int = Field(default=10, ge=1)


class AgentResult(BaseModel):
    """Terminal result of one agent run."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: str
    steps: int
    response: str
    messages: list[dict[str, Any]]


@dataclass
class CodingAgent:
    """Run a coding task using an injectable LLM and tool set."""

    client: LLMClient
    tools: Sequence[Tool]
    system_prompt: str = (
        "You are a coding agent. Use the available tools to inspect and modify "
        "the workspace, then report when the task is complete."
    )

    def run(self, task: AgentTask) -> AgentResult:
        """Execute tool calls until completion or the horizon is reached."""

        tool_map = {tool.name: tool for tool in self.tools}
        definitions = [tool.definition() for tool in self.tools]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task.prompt},
        ]

        for step in range(1, task.horizon_limit + 1):
            response = self.client.complete(messages, definitions)
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [call.model_dump() for call in response.tool_calls],
                }
            )

            if not response.tool_calls:
                return AgentResult(
                    task_id=task.task_id,
                    status="completed",
                    steps=step,
                    response=response.content,
                    messages=messages,
                )

            for call in response.tool_calls:
                tool = tool_map.get(call.name)
                if tool is None:
                    observation = f"Unknown tool: {call.name}"
                else:
                    try:
                        observation = tool.invoke(call.arguments)
                    except (OSError, ValueError) as error:
                        observation = f"Tool error: {error}"
                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": observation,
                    }
                )

        return AgentResult(
            task_id=task.task_id,
            status="horizon_exceeded",
            steps=task.horizon_limit,
            response="",
            messages=messages,
        )
