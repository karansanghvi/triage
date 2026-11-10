"""A fake model backend (TRIAGE_FAKE_MODELS=1): no network, no API keys, deterministic answers.

FakeModels has the same signature as providers.base.send_request, so it plugs into the REAL ReliableClient
(and CachedSender): retries, fallback chains, the circuit breaker and the first-token limit all run for real.
Streams use the real StreamingResponse, so time-to-first-token and usage are measured the normal way.

Script a model's behaviour per name:
    fake.script["groq-small"] = FakeBehavior(fail_before="server")           # fails before any text
    fake.script["groq-small"] = FakeBehavior(first_token_delay=2.0)          # slow to start
    fake.script["groq-small"] = FakeBehavior(fail_after=2)                   # breaks after 2 pieces
"""
import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.providers.base import StreamingResponse, _build_response
from app.providers.common import ProviderError, Usage


@dataclass
class FakeBehavior:
    text: Optional[str] = None             # default: "Fake answer from <model>."
    first_token_delay: float = 0.0         # seconds before the first piece (or before a non-stream answer)
    piece_delay: float = 0.0               # seconds between pieces
    fail_before: Optional[str] = None      # a ProviderError kind raised before any text
    fail_after: Optional[int] = None       # streams: raise after this many pieces
    fail_kind: str = "server"              # the kind raised by fail_after


def fake_text(model_name: str) -> str:
    return f"Fake answer from {model_name}. Triage is running without real models."


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class FakeModels:
    def __init__(self, script: Optional[dict] = None, sleep=time.sleep):
        self.script = dict(script or {})
        self.calls = []                    # one dict per call, for tests
        self._sleep = sleep
        self._lock = threading.Lock()

    def __call__(self, messages, model_config, stream=False, temperature=0.0, max_tokens=4000, timeout=30.0,
                 reasoning_effort=None, response_format=None):
        b = self.script.get(model_config.name, FakeBehavior())
        with self._lock:
            self.calls.append({"model": model_config.name, "stream": stream, "temperature": temperature,
                               "max_tokens": max_tokens, "timeout": timeout, "reasoning_effort": reasoning_effort})
        text = b.text if b.text is not None else fake_text(model_config.name)
        usage = Usage(_tokens(" ".join(m["content"] for m in messages)), _tokens(text))
        provider = model_config.provider
        if not stream:
            if b.first_token_delay > timeout:
                raise ProviderError(provider, "timeout", "request timed out (fake)")
            self._sleep(b.first_token_delay)
            if b.fail_before:
                raise ProviderError(provider, b.fail_before, "scripted failure (fake)")
            return _build_response(model_config, text, usage, messages, b.first_token_delay * 1000, "stop")
        pieces = [w + " " for w in text.split(" ")[:-1]] + [text.split(" ")[-1]]

        def events():
            self._sleep(b.first_token_delay)
            if b.fail_before:
                raise ProviderError(provider, b.fail_before, "scripted failure before the first token (fake)")
            for i, piece in enumerate(pieces):
                if b.fail_after is not None and i == b.fail_after:
                    raise ProviderError(provider, b.fail_kind, "scripted failure mid-stream (fake)")
                if i:
                    self._sleep(b.piece_delay)
                yield piece
            yield usage

        return StreamingResponse(events(), model_config, messages)
