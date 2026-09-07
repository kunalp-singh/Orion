"""Action-executor contract for context transformations."""

from abc import ABC, abstractmethod
from typing import Any

from .models import Action


class ActionExecutor(ABC):
    """Execute the action selected by a policy against a context store."""

    @abstractmethod
    def execute(self, action: Action, context: Any) -> Any:
        """Return the context payload that should be sent to the LLM."""
