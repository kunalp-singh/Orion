"""Provider-independent context storage for Orion experiments."""

from collections.abc import Sequence
import re
from typing import Protocol

from .models import ContextItem


class ContextStore(Protocol):
    def add(self, item: ContextItem) -> ContextItem: ...
    def retrieve(self, query: str, scope: str = "default", limit: int = 5) -> list[ContextItem]: ...
    def update(self, item: ContextItem) -> ContextItem: ...
    def delete(self, item_id: str) -> None: ...
    def get_active_context(self, scope: str = "default") -> list[ContextItem]: ...
    def get_scoped_context(self, scope: str) -> list[ContextItem]: ...


class InMemoryContextStore:
    """Deterministic store used by tests and local experiments."""

    def __init__(self) -> None:
        self._items: dict[str, ContextItem] = {}
        self._counter = 0

    def add(self, item: ContextItem) -> ContextItem:
        self._counter += 1
        item_id = str(self._counter)
        stored = item.model_copy(
            update={
                "metadata": {
                    **item.metadata,
                    "_id": item_id,
                    "item_id": item_id,
                }
            }
        )
        self._items[str(self._counter)] = stored
        return stored

    def retrieve(self, query: str, scope: str = "default", limit: int = 5) -> list[ContextItem]:
        terms = self._terms(query)
        scored: list[tuple[float, ContextItem]] = []
        for item in self._items.values():
            if item.scope != scope:
                continue
            item_terms = self._terms(item.content)
            overlap = terms.intersection(item_terms)
            if not overlap:
                continue
            score = len(overlap) / max(1, len(terms)) * item.importance
            supplied_similarity = item.metadata.get("similarity")
            stored = item.model_copy(
                update={
                    "metadata": {
                        **item.metadata,
                        "similarity": (
                            float(supplied_similarity)
                            if isinstance(supplied_similarity, (int, float))
                            else score
                        ),
                    }
                }
            )
            scored.append((score, stored))
        scored.sort(key=lambda result: result[0], reverse=True)
        return [item for _, item in scored[:limit]]

    @staticmethod
    def _terms(text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-z0-9_]+", text.lower())
            if len(term) > 2
        }

    def update(self, item: ContextItem) -> ContextItem:
        item_id = str(item.metadata.get("_id", ""))
        if item_id not in self._items:
            raise KeyError(f"Unknown context item: {item_id}")
        self._items[item_id] = item
        return item

    def delete(self, item_id: str) -> None:
        self._items.pop(item_id, None)

    def get_active_context(self, scope: str = "default") -> list[ContextItem]:
        return self.get_scoped_context(scope)

    def get_scoped_context(self, scope: str) -> list[ContextItem]:
        return [item for item in self._items.values() if item.scope == scope]
