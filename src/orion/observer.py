"""State-observer contract for the agent loop."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from .models import State, TaskPhase


class StateObserver(ABC):
    """Build a state vector from data already available to the agent loop."""

    @abstractmethod
    def observe(self, context: Any) -> State:
        """Observe the candidate context immediately before an LLM call."""


class HeuristicStateObserver(StateObserver):
    """Deterministic prototype features from active agent context.

    The observer accepts the agent's message list, or a mapping containing
    ``messages`` plus optional provider/context metadata. Token counts supplied
    by the provider are preferred; character counts are only a fallback for
    pre-call states where no provider count exists.
    """

    def __init__(self, context_budget: int = 8_000) -> None:
        if context_budget < 1:
            raise ValueError("context_budget must be positive")
        self.context_budget = context_budget

    def observe(self, context: Any) -> State:
        metadata: Mapping[str, Any] = context if isinstance(context, Mapping) else {}
        raw_messages = metadata.get("messages", context)
        messages = (
            raw_messages
            if isinstance(raw_messages, Sequence) and not isinstance(raw_messages, (str, bytes))
            else [raw_messages]
        )
        text = " ".join(
            str(item.get("content", "")) if isinstance(item, Mapping) else str(item)
            for item in messages
        )
        tokens = self._context_tokens(metadata, messages, text)
        utilization = min(1.0, tokens / self.context_budget)
        freshness = self._information_freshness(metadata, messages)
        relevance = self._semantic_relevance(metadata, messages)
        phase = self._task_phase(messages)
        return State(
            context_utilization=utilization,
            semantic_relevance=relevance,
            information_freshness=freshness,
            retrieval_confidence=self._retrieval_confidence(metadata),
            remaining_token_budget=max(0.0, 1.0 - utilization),
            task_phase=phase,
        )

    @staticmethod
    def _context_tokens(
        metadata: Mapping[str, Any],
        messages: Sequence[Any],
        text: str,
    ) -> int:
        supplied = metadata.get("context_tokens")
        if isinstance(supplied, int) and supplied >= 0:
            return supplied
        message_tokens: list[int] = []
        for item in messages:
            if isinstance(item, Mapping) and isinstance(item.get("token_count"), int):
                message_tokens.append(item["token_count"])
        if message_tokens and len(message_tokens) == len(messages):
            return sum(message_tokens)
        return max(0, len(text) // 4)

    @staticmethod
    def _task_phase(messages: Sequence[Any]) -> TaskPhase:
        roles = [item.get("role") for item in messages if isinstance(item, Mapping)]
        tool_names = {
            call.get("name")
            for item in messages
            if isinstance(item, Mapping)
            for call in item.get("tool_calls", [])
            if isinstance(call, Mapping)
        }
        if "run_tests" in tool_names or "test" in {
            str(item.get("name", "")) for item in messages if isinstance(item, Mapping)
        }:
            return TaskPhase.TESTING
        if "write_file" in tool_names:
            return TaskPhase.EDITING
        if "read_file" in tool_names:
            return TaskPhase.PLANNING
        if "assistant" not in roles:
            return TaskPhase.PLANNING
        return TaskPhase.EXPLAINING

    @staticmethod
    def _semantic_relevance(
        metadata: Mapping[str, Any],
        messages: Sequence[Any],
    ) -> float:
        """Proxy for future embedding similarity.

        Explicit relevance from a retrieval subsystem is preferred. Otherwise
        a failed tool result is treated as low relevance, while successful tool
        evidence receives a bounded lexical-overlap score.
        """

        supplied = metadata.get("semantic_relevance")
        if isinstance(supplied, (int, float)):
            return max(0.0, min(1.0, float(supplied)))
        user_text = " ".join(
            str(item.get("content", ""))
            for item in messages
            if isinstance(item, Mapping) and item.get("role") == "user"
        ).lower()
        tool_text = " ".join(
            str(item.get("content", ""))
            for item in messages
            if isinstance(item, Mapping) and item.get("role") == "tool"
        ).lower()
        if "error" in tool_text or "failed" in tool_text:
            return 0.2
        query_terms = {term for term in user_text.split() if len(term) > 3}
        overlap = sum(term in tool_text for term in query_terms)
        return min(1.0, 0.4 + overlap / max(1, len(query_terms)))

    @staticmethod
    def _information_freshness(
        metadata: Mapping[str, Any],
        messages: Sequence[Any],
    ) -> float:
        supplied = metadata.get("information_freshness")
        if isinstance(supplied, (int, float)):
            return max(0.0, min(1.0, float(supplied)))
        ages: list[float] = []
        now = datetime.now(timezone.utc)
        for item in messages:
            if not isinstance(item, Mapping):
                continue
            timestamp = item.get("timestamp") or item.get("created_at")
            if isinstance(timestamp, datetime):
                age_seconds = max(0.0, (now - timestamp.astimezone(timezone.utc)).total_seconds())
                ages.append(max(0.0, 1.0 - age_seconds / 3600))
        if ages:
            return sum(ages) / len(ages)
        return max(0.0, 1.0 - max(0, len(messages) - 2) / 30)

    @staticmethod
    def _retrieval_confidence(metadata: Mapping[str, Any]) -> float:
        supplied = metadata.get("retrieval_confidence")
        if isinstance(supplied, (int, float)):
            return max(0.0, min(1.0, float(supplied)))
        retrieval = metadata.get("retrieval")
        if isinstance(retrieval, Mapping):
            confidence = retrieval.get("confidence")
            if isinstance(confidence, (int, float)):
                return max(0.0, min(1.0, float(confidence)))
        return 0.5
