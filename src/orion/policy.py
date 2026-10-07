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


class StaticPolicy(Policy):
    """Fixed compression-threshold baseline."""

    def __init__(self, compression_threshold: float = 0.75) -> None:
        if not 0 < compression_threshold <= 1:
            raise ValueError("compression_threshold must be in (0, 1]")
        self.compression_threshold = compression_threshold

    def decide(self, state: State) -> Action:
        return (
            Action.COMPRESS
            if state.context_utilization >= self.compression_threshold
            else Action.IGNORE
        )


class RuleBasedOrion(Policy):
    """Interpretable adaptive policy; a future bandit can replace this class."""

    def __init__(
        self,
        compression_threshold: float = 0.8,
        retrieval_relevance_threshold: float = 0.35,
        freshness_threshold: float = 0.35,
    ) -> None:
        self.compression_threshold = compression_threshold
        self.retrieval_relevance_threshold = retrieval_relevance_threshold
        self.freshness_threshold = freshness_threshold

    def decide(self, state: State) -> Action:
        if state.semantic_relevance < self.retrieval_relevance_threshold:
            return Action.RETRIEVE
        if state.context_utilization >= self.compression_threshold:
            return Action.COMPRESS
        if state.information_freshness < self.freshness_threshold:
            return Action.STORE
        if state.task_phase.value == "testing" and state.context_utilization > 0.5:
            return Action.ISOLATE
        return Action.IGNORE
