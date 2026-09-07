"""Provider-agnostic models shared by Orion's control loop."""

from enum import StrEnum

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
