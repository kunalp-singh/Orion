"""Provider-agnostic models shared by Orion's control loop."""

from enum import StrEnum
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Action(StrEnum):
    """The single context-management action selected for an agent step."""

    RETRIEVE = "retrieve"
    COMPRESS = "compress"
    ISOLATE = "isolate"
    STORE = "store"
    IGNORE = "ignore"


class TaskPhase(StrEnum):
    """The phase of the coding task represented by the current context."""

    PLANNING = "planning"
    EDITING = "editing"
    TESTING = "testing"
    EXPLAINING = "explaining"


class State(BaseModel):
    """The deterministic state vector observed before an LLM call."""

    model_config = ConfigDict(extra="forbid")

    context_utilization: float = Field(ge=0, le=1)
    semantic_relevance: float = Field(ge=0, le=1)
    information_freshness: float = Field(ge=0, le=1)
    retrieval_confidence: float = Field(ge=0, le=1)
    remaining_token_budget: float = Field(ge=0, le=1)
    task_phase: TaskPhase


class ContextItem(BaseModel):
    """A durable fact that can be retained independently of active context."""

    content: str
    source: str = "agent"
    task_id: str | None = None
    scope: str = "default"
    importance: float = Field(default=0.5, ge=0, le=1)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)
