"""Provider-agnostic coding-agent loop and function-calling contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from .models import Action
from .observer import StateObserver
from .policy import Policy
from .executor import ActionExecutor, OrionActionExecutor


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
    usage: dict[str, Any] = Field(default_factory=dict)


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
    records: list[dict[str, Any]] = Field(default_factory=list)


@dataclass
class CodingAgent:
    """Run a coding task using an injectable LLM and tool set."""

    client: LLMClient
    tools: Sequence[Tool]
    system_prompt: str = (
        "You are a coding agent. Use the available tools to inspect and modify "
        "the workspace, then report when the task is complete."
    )
    observer: StateObserver | None = None
    policy: Policy | None = None
    action_executor: ActionExecutor | None = None
    logger: Any = None
    run_id: str | None = None
    policy_name: str | None = None
    provider_name: str | None = None
    model_name: str | None = None

    def run(self, task: AgentTask) -> AgentResult:
        """Execute tool calls until completion or the horizon is reached."""

        tool_map = {tool.name: tool for tool in self.tools}
        definitions = [tool.definition() for tool in self.tools]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": task.prompt},
        ]
        if isinstance(self.action_executor, OrionActionExecutor):
            self.action_executor.ingest(
                task.prompt,
                source_type="task_constraint",
                task_id=task.task_id,
                run_id=self.run_id,
                importance=0.9,
            )
        records: list[dict[str, Any]] = []
        previous_input_tokens: int | None = None

        for step in range(1, task.horizon_limit + 1):
            action = Action.IGNORE
            state = None
            managed_messages = messages
            retrieval_trace = None
            if self.observer and self.policy and self.action_executor:
                state = self.observer.observe(messages)
                action = self.policy.decide(state)
                managed_messages = self.action_executor.execute(action, messages, state)
                retrieval_trace = getattr(self.action_executor, "last_retrieval", None)
            response = self.client.complete(managed_messages, definitions)
            input_tokens = response.usage.get("input_tokens")
            output_tokens = response.usage.get("output_tokens")
            total_tokens = response.usage.get("total_tokens")
            tool_calls: list[dict[str, Any]] = []
            tool_errors: list[str] = []
            tool_latency_ms = 0.0
            record = {
                "run_id": self.run_id,
                "task_id": task.task_id,
                "step": step,
                "policy": self.policy_name,
                "provider": self.provider_name,
                "model": self.model_name,
                "action": action.value,
                "context_tokens_before": previous_input_tokens,
                "context_tokens_after": input_tokens,
                "context_utilization": state.context_utilization if state else None,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "latency_ms": response.usage.get("latency_ms"),
                "task_phase": state.task_phase.value if state else None,
                "semantic_relevance": state.semantic_relevance if state else None,
                "information_freshness": state.information_freshness if state else None,
                "retrieval_confidence": state.retrieval_confidence if state else None,
                "remaining_token_budget": state.remaining_token_budget if state else None,
                "tool_calls": tool_calls,
                "tool_latency_ms": tool_latency_ms,
                "success": None,
                "error": None,
                "error_type": None,
            }
            if retrieval_trace is not None:
                retrieval_trace.context_tokens_before = record["context_tokens_before"]
                retrieval_trace.context_tokens_after = input_tokens
                if (
                    retrieval_trace.context_tokens_before is not None
                    and input_tokens is not None
                ):
                    retrieval_trace.tokens_added = (
                        input_tokens - retrieval_trace.context_tokens_before
                    )
                record["retrieval"] = {
                    "query": retrieval_trace.query,
                    "item_ids": retrieval_trace.item_ids,
                    "content_lengths": retrieval_trace.content_lengths,
                    "relevance_scores": retrieval_trace.relevance_scores,
                    "context_before": retrieval_trace.context_before,
                    "context_after": retrieval_trace.context_after,
                    "context_tokens_before": retrieval_trace.context_tokens_before,
                    "context_tokens_after": retrieval_trace.context_tokens_after,
                    "retrieval_tokens_added": retrieval_trace.retrieval_tokens_added,
                    "retrieval_latency_ms": retrieval_trace.retrieval_latency_ms,
                    "executed": retrieval_trace.executed,
                }
            previous_input_tokens = input_tokens
            records.append(record)
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [call.model_dump() for call in response.tool_calls],
                }
            )

            if not response.tool_calls:
                record["success"] = True
                if self.logger:
                    self.logger.log(record)
                return AgentResult(
                    task_id=task.task_id,
                    status="completed",
                    steps=step,
                    response=response.content,
                    messages=messages,
                    records=records,
                )

            for call in response.tool_calls:
                tool_calls.append(call.model_dump())
                tool = tool_map.get(call.name)
                if tool is None:
                    observation = f"Unknown tool: {call.name}"
                    tool_errors.append(observation)
                else:
                    try:
                        import time

                        started = time.perf_counter()
                        observation = tool.invoke(call.arguments)
                        tool_latency_ms += (time.perf_counter() - started) * 1000
                    except (OSError, ValueError) as error:
                        observation = f"Tool error: {error}"
                        tool_errors.append(observation)
                if isinstance(self.action_executor, OrionActionExecutor):
                    source_type = {
                        "read_file": "file_content",
                        "run_tests": "test_result",
                    }.get(call.name)
                    if source_type and observation.strip():
                        self.action_executor.ingest(
                            observation,
                            source_type=source_type,
                            task_id=task.task_id,
                            run_id=self.run_id,
                            importance=0.8 if call.name == "read_file" else 0.9,
                        )
                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "content": observation,
                    }
                )
            record["tool_calls"] = tool_calls
            record["tool_latency_ms"] = tool_latency_ms
            record["error"] = "\n".join(tool_errors) or None
            record["error_type"] = "tool_error" if tool_errors else None
            record["success"] = not tool_errors
            if self.logger:
                self.logger.log(record)

        return AgentResult(
            task_id=task.task_id,
            status="horizon_exceeded",
            steps=task.horizon_limit,
            response="",
            messages=messages,
            records=records,
        )
