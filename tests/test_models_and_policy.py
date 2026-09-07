import pytest
from pydantic import ValidationError

from orion.models import Action, State, TaskPhase
from orion.policy import NoManagementPolicy


def make_state() -> State:
    return State(
        context_utilization=0.2,
        semantic_relevance=0.8,
        information_freshness=0.9,
        retrieval_confidence=1.0,
        remaining_token_budget=0.7,
        task_phase=TaskPhase.PLANNING,
    )


def test_state_rejects_out_of_range_values() -> None:
    with pytest.raises(ValidationError):
        State(**{**make_state().model_dump(), "context_utilization": 1.1})


def test_no_management_policy_passes_context_through() -> None:
    assert NoManagementPolicy().decide(make_state()) is Action.IGNORE
