"""Gemini REST adapter implementing Orion's provider-neutral LLM contract."""

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from .agent import LLMResponse, ToolCall


class GeminiClient:
    """Call Gemini generateContent and normalize its function-call response."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        http_client: httpx.Client | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required")
        self.model = model
        self._client = http_client or httpx.Client(timeout=120)
        self._owns_client = http_client is None
        self._api_key = api_key

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMResponse:
        system_instruction, contents = self._format_messages(messages)
        payload: dict[str, Any] = {"contents": contents}
        if system_instruction:
            payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        if tools:
            payload["tools"] = [{"functionDeclarations": list(tools)}]

        response = self._client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
            params={"key": self._api_key},
            json=payload,
        )
        response.raise_for_status()
        return self._parse_response(response.json())

    def close(self) -> None:
        """Close the internally owned HTTP client."""

        if self._owns_client:
            self._client.close()

    @staticmethod
    def _format_messages(
        messages: Sequence[Mapping[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        system = ""
        contents: list[dict[str, Any]] = []
        for message in messages:
            role = message["role"]
            if role == "system":
                system = str(message.get("content", ""))
            elif role == "tool":
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "functionResponse": {
                                    "name": message["name"],
                                    "response": {"content": message["content"]},
                                }
                            }
                        ],
                    }
                )
            else:
                parts: list[dict[str, Any]] = []
                if message.get("content"):
                    parts.append({"text": message["content"]})
                for call in message.get("tool_calls", []):
                    parts.append({"functionCall": call})
                contents.append({"role": "model" if role == "assistant" else "user", "parts": parts})
        return system, contents

    @staticmethod
    def _parse_response(data: Mapping[str, Any]) -> LLMResponse:
        candidate = data.get("candidates", [{}])[0]
        parts = candidate.get("content", {}).get("parts", [])
        text = "".join(part.get("text", "") for part in parts)
        calls = [
            ToolCall(name=part["functionCall"]["name"], arguments=part["functionCall"].get("args", {}))
            for part in parts
            if "functionCall" in part
        ]
        return LLMResponse(
            content=text,
            tool_calls=calls,
            finish_reason=candidate.get("finishReason", "stop"),
        )
