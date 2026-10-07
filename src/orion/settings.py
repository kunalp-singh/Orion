"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings with safe local defaults."""

    model_config = SettingsConfigDict(env_prefix="ORION_", env_file=".env")

    app_name: str = "Orion"
    environment: str = "development"
    mongo_uri: str = Field(default="mongodb://localhost:27017")
    mongo_database: str = "orion"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.0-flash"
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    llm_provider: str = "gemini"
    temperature: float = 0.0
    max_tokens: int | None = None
    workspace_root: str = "."


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""

    return Settings()
