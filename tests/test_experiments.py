import pytest

from orion.experiments import classify_task_result


@pytest.mark.parametrize(
    ("objective_test_success", "agent_status", "task_success", "agent_completed"),
    [
        (True, "completed", True, True),
        (True, "horizon_exceeded", True, False),
        (False, "completed", False, True),
        (False, "horizon_exceeded", False, False),
    ],
)
def test_task_success_uses_objective_test_only(
    objective_test_success: bool,
    agent_status: str,
    task_success: bool,
    agent_completed: bool,
) -> None:
    assert classify_task_result(agent_status, objective_test_success) == {
        "task_success": task_success,
        "agent_status": agent_status,
        "agent_completed": agent_completed,
    }
