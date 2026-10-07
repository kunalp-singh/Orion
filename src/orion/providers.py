"""HTTP LLM providers with a shared normalized response contract."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any
import time

import httpx

from .agent import LLMResponse, ToolCall


class ProviderSetupError(RuntimeError):
    """Raised when a local provider cannot satisfy an experiment request."""


class LLMProvider(ABC):
    provider_name: str
    model_name: str

    @abstractmethod
    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> LLMResponse:
        """Generate one normalized response."""

    def close(self) -> None:
        """Close provider resources when applicable."""


class OllamaProvider(LLMProvider):
    provider_name = "ollama"

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        http_client: httpx.Client | None = None,
        timeout: float = 300,
    ) -> None:
        self.model_name = model
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None
        self.base_url = base_url.rstrip("/")

    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> LLMResponse:
        started = time.perf_counter()
        payload: dict[str, Any] = {"model": self.model_name, "messages": list(messages), "stream": False}
        if tools:
            payload["tools"] = [
                tool if tool.get("type") == "function" else {"type": "function", "function": dict(tool)}
                for tool in tools
            ]
        response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        calls = []
        for call in message.get("tool_calls", []):
            function = call.get("function", call)
            calls.append(
                ToolCall(
                    name=function.get("name", ""),
                    arguments=function.get("arguments", {}),
                )
            )
        usage = {
            "input_tokens": data.get("prompt_eval_count"),
            "output_tokens": data.get("eval_count"),
            "total_tokens": (
                data["prompt_eval_count"] + data["eval_count"]
                if data.get("prompt_eval_count") is not None and data.get("eval_count") is not None
                else None
            ),
            "provider_total_duration_ns": data.get("total_duration"),
            "provider_prompt_eval_duration_ns": data.get("prompt_eval_duration"),
            "provider_eval_duration_ns": data.get("eval_duration"),
            "latency_ms": (
                data["total_duration"] / 1_000_000
                if data.get("total_duration") is not None
                else (time.perf_counter() - started) * 1000
            ),
        }
        return LLMResponse(content=message.get("content", ""), tool_calls=calls, usage=usage)

    def check_setup(self) -> None:
        """Verify Ollama is reachable and the configured model is installed."""

        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise ProviderSetupError(
                f"Ollama is unavailable at {self.base_url}. Start it with "
                "`ollama serve`, then retry the experiment."
            ) from error
        models = {item.get("name") for item in response.json().get("models", [])}
        if self.model_name not in models and not any(
            name and name.split(":")[0] == self.model_name.split(":")[0] for name in models
        ):
            raise ProviderSetupError(
                f"Ollama model {self.model_name!r} is not installed. "
                f"Run `ollama pull {self.model_name}` and retry."
            )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class GroqProvider(LLMProvider):
    provider_name = "groq"

    def __init__(self, api_key: str, model: str, http_client: httpx.Client | None = None) -> None:
        if not api_key:
            raise ValueError("Groq API key is required")
        self.model_name = model
        self._client = http_client or httpx.Client(timeout=120, headers={"Authorization": f"Bearer {api_key}"})
        self._owns_client = http_client is None

    def complete(self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]) -> LLMResponse:
        payload: dict[str, Any] = {"model": self.model_name, "messages": list(messages)}
        if tools:
            payload["tools"] = [
                tool if tool.get("type") == "function" else {"type": "function", "function": dict(tool)}
                for tool in tools
            ]
        response = self._client.post("https://api.groq.com/openai/v1/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        calls = [ToolCall(name=c["function"]["name"], arguments=c["function"].get("arguments", {})) for c in message.get("tool_calls", [])]
        return LLMResponse(content=message.get("content") or "", tool_calls=calls, finish_reason=choice.get("finish_reason", "stop"), usage=data.get("usage", {}))

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
