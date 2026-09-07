from fastapi.testclient import TestClient

from orion.agent import LLMResponse
from orion.main import create_app


class FakeClient:
    def complete(self, messages, tools) -> LLMResponse:
        return LLMResponse(content="done")


def test_health_endpoint() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_agent_run_endpoint_uses_injected_client(tmp_path) -> None:
    with TestClient(create_app(llm_client=FakeClient())) as client:
        response = client.post(
            "/agent/run",
            json={"task_id": "smoke", "prompt": "Say done", "horizon_limit": 1},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["response"] == "done"


def test_agent_run_requires_api_key() -> None:
    with TestClient(create_app()) as client:
        response = client.post("/agent/run", json={"task_id": "x", "prompt": "x"})

    assert response.status_code == 503
