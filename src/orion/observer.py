"""State-observer contract for the agent loop."""

from abc import ABC, abstractmethod
from typing import Any

from .models import State


class StateObserver(ABC):
    """Build a state vector from data already available to the agent loop."""

    @abstractmethod
    def observe(self, context: Any) -> State:
        """Observe the candidate context immediately before an LLM call."""
