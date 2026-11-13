"""BYOK (bring your own key), Phase 5: per-request keys through the whole request path. No network.

SECRET is a user key; it must reach only the provider it belongs to, and never a log, a cache key, an error
body, a background task or the shared circuit breaker.
"""
import json

import httpx
import httpx2
import pytest

from app.providers import anthropic_provider, gemini_provider, openai_provider
from app.providers.base import send_request
from app.providers.common import ProviderError
from app.registry import load_registry

SECRET = "sk-test-SECRET123"
REGISTRY = load_registry()
MESSAGES = [{"role": "user", "content": "Hi"}]


class Wire:
    def __init__(self, body, lib=httpx, ctype="application/json"):
        self.body, self.lib, self.ctype, self.requests = body, lib, ctype, []

    def __call__(self, request):
        self.requests.append(request)
        return self.lib.Response(200, content=json.dumps(self.body), headers={"content-type": self.ctype})

    def client(self):
        return self.lib.Client(transport=self.lib.MockTransport(self))


OPENAI_REPLY = {"id": "c", "object": "chat.completion", "created": 0, "model": "m",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
ANTHROPIC_REPLY = {"id": "m", "type": "message", "role": "assistant", "model": "claude-haiku-5-5",
                   "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
                   "usage": {"input_tokens": 1, "output_tokens": 1}}
GEMINI_REPLY = {"candidates": [{"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2}}


@pytest.fixture
def server_keys(monkeypatch):
    """Server env keys that must NOT be used when a user key is given."""
    for env in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(env, "server-env-key")


# ---------- a: each adapter sends the user's key ----------

@pytest.mark.parametrize("model, module, reply, lib, header, prefix", [
    ("openai-luna", openai_provider, OPENAI_REPLY, httpx, "authorization", "Bearer "),
    ("groq-small", openai_provider, OPENAI_REPLY, httpx, "authorization", "Bearer "),
    ("claude-haiku", anthropic_provider, ANTHROPIC_REPLY, httpx2, "x-api-key", ""),
    ("gemini-flash", gemini_provider, GEMINI_REPLY, httpx, "x-goog-api-key", ""),
])
def test_adapter_sends_the_given_key_not_the_env_key(server_keys, monkeypatch, model, module, reply, lib, header,
                                                     prefix):
    wire = Wire(reply, lib)
    monkeypatch.setattr(module, "HTTP_CLIENT", wire.client())
    r = send_request(MESSAGES, REGISTRY[model], max_tokens=100, api_key=SECRET)
    assert r.text == "ok"
    (request,) = wire.requests
    assert request.headers[header] == prefix + SECRET
    assert "server-env-key" not in str(request.headers)


@pytest.mark.parametrize("model, module, reply, lib, header, prefix", [
    ("groq-small", openai_provider, OPENAI_REPLY, httpx, "authorization", "Bearer "),
    ("claude-haiku", anthropic_provider, ANTHROPIC_REPLY, httpx2, "x-api-key", ""),
    ("gemini-flash", gemini_provider, GEMINI_REPLY, httpx, "x-goog-api-key", ""),
])
def test_without_a_key_the_adapter_uses_the_env_key_as_before(server_keys, monkeypatch, model, module, reply, lib,
                                                              header, prefix):
    wire = Wire(reply, lib)
    monkeypatch.setattr(module, "HTTP_CLIENT", wire.client())
    send_request(MESSAGES, REGISTRY[model], max_tokens=100)
    assert wire.requests[0].headers[header] == prefix + "server-env-key"


def test_an_empty_given_key_is_an_error_not_the_env_key(server_keys):
    with pytest.raises(ProviderError) as error:
        send_request(MESSAGES, REGISTRY["groq-small"], max_tokens=100, api_key="")
    assert error.value.kind == "config"


def test_ollama_ignores_a_key(monkeypatch):
    from app.providers import ollama_provider
    seen = {}
    monkeypatch.setattr(ollama_provider, "send_openai_format", lambda *a, **kw: seen.update(kw) or "ok")
    ollama_provider.send(MESSAGES, "llama3.2:3b", api_key=SECRET, stream=False, max_tokens=1, timeout=1)
    assert seen["api_key"] == "ollama"
