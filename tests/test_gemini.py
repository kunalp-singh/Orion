import httpx

from orion.gemini import GeminiClient


def test_gemini_client_normalizes_function_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["key"] == "test-key"
        payload = request.read().decode()
        assert "functionDeclarations" in payload
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {"functionCall": {"name": "read_file", "args": {"path": "a.txt"}}}
                            ]
                        },
                    }
                ]
            },
            request=request,
        )

    client = GeminiClient("test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    result = client.complete(
        [{"role": "user", "content": "Read a.txt"}],
        [{"name": "read_file", "description": "read", "parameters": {"type": "object"}}],
    )

    assert result.tool_calls[0].name == "read_file"
    assert result.tool_calls[0].arguments == {"path": "a.txt"}
