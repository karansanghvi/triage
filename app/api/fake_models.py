"""A fake model backend (TRIAGE_FAKE_MODELS=1): no network, no API keys, deterministic answers.

FakeModels has the same signature as providers.base.send_request, so it plugs into the REAL ReliableClient
(and CachedSender): retries, fallback chains, the circuit breaker and the first-token limit all run for real.
Streams use the real StreamingResponse, so time-to-first-token and usage are measured the normal way.

Script a model's behaviour per name:
    fake.script["groq-small"] = FakeBehavior(fail_before="server")           # fails before any text
    fake.script["groq-small"] = FakeBehavior(first_token_delay=2.0)          # slow to start
    fake.script["groq-small"] = FakeBehavior(fail_after=2)                   # breaks after 2 pieces
    fake.script["groq-small"] = FakeBehavior(check_reply='{"good": false, "reason": "wrong"}')

JSON-mode calls (response_format set) are the AI checker's or the judge's, told apart by their system prompt.
By default the checker says good and the judge gives 5; script check_reply / judge_reply (raw reply text) or
check_fail / judge_fail (a ProviderError kind) to change that. Answer settings never affect these calls.
"""
import threading
import time
from dataclasses import dataclass
from typing import Optional

from app.providers.base import StreamingResponse, TruncatedAnswer, _build_response
from app.providers.common import ProviderError, Usage


@dataclass
class FakeBehavior:
    text: Optional[str] = None             # default: "Fake answer from <model>."
    first_token_delay: float = 0.0         # seconds before the first piece (or before a non-stream answer)
    piece_delay: float = 0.0               # seconds between pieces
    fail_before: Optional[str] = None      # a ProviderError kind raised before any text
    fail_after: Optional[int] = None       # streams: raise after this many pieces
    fail_kind: str = "server"              # the kind raised by fail_after
    check_reply: Optional[str] = None      # the AI checker's raw reply (default: good)
    check_fail: Optional[str] = None       # a ProviderError kind for AI-checker calls only
    judge_reply: Optional[str] = None      # the judge's raw reply (default: score 5)
    judge_fail: Optional[str] = None       # a ProviderError kind for judge calls only
    truncated: bool = False                # answers: used every output token thinking, no text (TruncatedAnswer)


DEFAULT_CHECK_REPLY = '{"good": true, "reason": "fake check: looks right"}'
DEFAULT_JUDGE_REPLY = '{"constraints": [], "score": 5, "reason": "fake judge: as good as the reference"}'


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
                 reasoning_effort=None, response_format=None, thinking=None):
        b = self.script.get(model_config.name, FakeBehavior())
        kind = self._kind(messages, response_format)
        with self._lock:
            self.calls.append({"model": model_config.name, "stream": stream, "temperature": temperature,
                               "max_tokens": max_tokens, "timeout": timeout, "reasoning_effort": reasoning_effort,
                               "thinking": thinking, "kind": kind})
        if kind != "answer":
            fail = b.check_fail if kind == "check" else b.judge_fail
            if fail:
                raise ProviderError(model_config.provider, fail, f"scripted {kind} failure (fake)")
            reply = b.check_reply if kind == "check" else b.judge_reply
            reply = reply if reply is not None else (DEFAULT_CHECK_REPLY if kind == "check" else DEFAULT_JUDGE_REPLY)
            usage = Usage(_tokens(" ".join(m["content"] for m in messages)), _tokens(reply))
            return _build_response(model_config, reply, usage, messages, 1.0, "stop")
        if b.truncated:                    # billed for max_tokens of thinking, nothing written
            billed = _build_response(model_config, "", Usage(_tokens(" ".join(m["content"] for m in messages)),
                                                             max_tokens), messages, 1.0, "length")
            raise TruncatedAnswer(billed, max_tokens)
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

    @staticmethod
    def _kind(messages, response_format) -> str:
        """answer, check (the AI checker) or judge."""
        if response_format is None or not messages or messages[0].get("role") != "system":
            return "answer"
        from app.verifier.judge import JUDGE_SYSTEM_PROMPT
        return "judge" if messages[0]["content"] == JUDGE_SYSTEM_PROMPT else "check"
