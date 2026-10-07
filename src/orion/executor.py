"""Action-executor contract for context transformations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import time
from typing import Any

from .models import Action, ContextItem
from .context import ContextStore


@dataclass
class RetrievalTrace:
    query: str
    item_ids: list[str]
    content_lengths: list[int]
    relevance_scores: list[float | None]
    context_before: dict[str, Any]
    context_after: dict[str, Any]
    context_tokens_before: int | None
    context_tokens_after: int | None
    retrieval_tokens_added: int
    retrieval_latency_ms: float
    executed: bool = True


def _context_snapshot(context: Any) -> dict[str, int]:
    messages = context if isinstance(context, list) else []
    return {
        "message_count": len(messages),
        "content_chars": sum(
            len(str(item.get("content", "")))
            for item in messages
            if isinstance(item, dict)
        ),
    }


class ActionExecutor(ABC):
    """Execute the action selected by a policy against a context store."""

    @abstractmethod
    def execute(self, action: Action, context: Any, state: Any = None) -> Any:
        """Return the context payload that should be sent to the LLM."""


class OrionActionExecutor(ActionExecutor):
    """Execute all five actions against an injectable context store."""

    def __init__(self, store: ContextStore) -> None:
        self.store = store
        self.scope = "default"
        self.last_retrieval: RetrievalTrace | None = None

    def ingest(
        self,
        content: str,
        *,
        source_type: str,
        task_id: str | None = None,
        run_id: str | None = None,
        importance: float = 0.7,
    ) -> ContextItem:
        return self.store.add(
            ContextItem(
                content=content,
                source=source_type,
                task_id=task_id,
                scope=self.scope,
                importance=importance,
                metadata={
                    "source_type": source_type,
                    "task_id": task_id,
                    "run_id": run_id,
                },
            )
        )

    def execute(self, action: Action, context: Any, state: Any = None) -> Any:
        self.last_retrieval = None
        if action is Action.RETRIEVE:
            query = str(context[-1].get("content", "")) if context else ""
            before = _context_snapshot(context)
            started = time.perf_counter()
            items = self.store.retrieve(query, self.scope)
            managed_context = context + [
                {"role": "system", "content": item.content} for item in items
            ]
            after = _context_snapshot(managed_context)
            self.last_retrieval = RetrievalTrace(
                query=query,
                item_ids=[str(item.metadata.get("_id", "")) for item in items],
                content_lengths=[len(item.content) for item in items],
                relevance_scores=[
                    self._relevance_score(item) for item in items
                ],
                context_before=before,
                context_after=after,
                context_tokens_before=None,
                context_tokens_after=None,
                retrieval_tokens_added=sum(max(1, len(item.content) // 4) for item in items),
                retrieval_latency_ms=(time.perf_counter() - started) * 1000,
            )
            return managed_context
        if action is Action.COMPRESS:
            text = "\n".join(str(item.get("content", "")) for item in context if isinstance(item, dict))
            return [{"role": "system", "content": f"Compressed context:\n{text[:max(1, len(text) // 2)]}"}]
        if action is Action.ISOLATE:
            self.scope = f"scope-{len(self.store.get_scoped_context(self.scope)) + 1}"
        elif action is Action.STORE:
            from .models import ContextItem
            text = str(context[-1].get("content", "")) if context else ""
            self.store.add(ContextItem(content=text, scope=self.scope, importance=0.8))
        return context

    @staticmethod
    def _relevance_score(item: ContextItem) -> float | None:
        score = item.metadata.get("similarity", item.metadata.get("relevance"))
        return float(score) if isinstance(score, (int, float)) else None
