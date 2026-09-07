"""Policy contracts and baseline implementations."""

from abc import ABC, abstractmethod

from .models import Action, State


class Policy(ABC):
    """Map one observed state to exactly one context-management action."""

    @abstractmethod
    def decide(self, state: State) -> Action:
        """Return the action to apply before the next LLM call."""


class NoManagementPolicy(Policy):
    """Baseline that passes the candidate context through unchanged."""

    def decide(self, state: State) -> Action:
        return Action.IGNORE
