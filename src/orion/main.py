"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import AgentResult, AgentTask, CodingAgent, LLMClient
from .database import MongoConnection
from .gemini import GeminiClient
from .settings import Settings, get_settings
from .tools import default_tools


class RunRequest(BaseModel):
    task_id: str
    prompt: str
    horizon_limit: int = Field(default=10, ge=1)


def create_app(
    mongo_connection: MongoConnection | None = None,
    llm_client: LLMClient | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Create the Orion API application with injectable infrastructure."""

    connection = mongo_connection or MongoConnection()
    config = settings or get_settings()
    client = llm_client
    owned_client: GeminiClient | None = None

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        connection.close()
        if owned_client is not None:
            owned_client.close()

    app = FastAPI(title=config.app_name, version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/agent/run", response_model=AgentResult)
    def run_agent(request: RunRequest) -> AgentResult:
        nonlocal client, owned_client
        if client is None:
            if not config.gemini_api_key:
                raise HTTPException(status_code=503, detail="Gemini API key is not configured")
            owned_client = GeminiClient(config.gemini_api_key, config.gemini_model)
            client = owned_client
        task = AgentTask(
            task_id=request.task_id,
            prompt=request.prompt,
            horizon_limit=request.horizon_limit,
        )
        agent = CodingAgent(client, default_tools(Path(config.workspace_root)))
        return agent.run(task)

    return app


app = create_app()
