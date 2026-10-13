"""The one function the rest of Triage uses to talk to any model."""
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from dotenv import load_dotenv

from app.providers import (
    anthropic_provider, gemini_provider, groq_provider,
    ollama_provider, openai_provider, openrouter_provider,
)
from app.providers.common import ProviderError, Usage
from app.registry import ModelConfig

load_dotenv()

VALID_ROLES = {"system", "user", "assistant"}
DEFAULT_TEMPERATURE = 0.0      # 0 = same question gives the same answer (needed for caching)
DEFAULT_MAX_TOKENS = 1024      # leaves room for reasoning models to think
DEFAULT_TIMEOUT_SECONDS = 30.0

# Which adapter handles which provider
ADAPTERS = {
    "openai": openai_provider.send,
    "groq": groq_provider.send,
    "ollama": ollama_provider.send,
    "openrouter": openrouter_provider.send,
    "gemini": gemini_provider.send,
    "anthropic": anthropic_provider.send,
}


@dataclass(frozen=True)
class Response:
    """The same answer object, whichever provider answered."""
    text: str
    model_name: str            # your nickname, e.g. "groq-small"
    model_id: str              # the provider's name, e.g. "openai/gpt-oss-20b"
    provider: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost: float                # what you actually paid
    reported_cost: float       # realistic cost (uses price_as), for savings math
    tokens_estimated: bool = False
    finish_reason: Optional[str] = None
    time_to_first_token_ms: Optional[float] = None   # streaming only


def _estimate_tokens(text: str) -> int:
    """Rough guess (about 4 characters per token) if a provider skips usage."""
    return max(1, len(text) // 4) if text else 0


def _build_response(model_config, text, usage, messages, latency_ms, finish_reason=None, ttft_ms=None):
    estimated = usage is None
    if estimated:
        prompt_text = " ".join(m["content"] for m in messages)
        usage = Usage(_estimate_tokens(prompt_text), _estimate_tokens(text))

    return Response(
        text=text,
        model_name=model_config.name,
        model_id=model_config.model_id,
        provider=model_config.provider,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        latency_ms=round(latency_ms, 1),
        cost=model_config.actual_cost(usage.input_tokens, usage.output_tokens),
        reported_cost=model_config.reported_cost(usage.input_tokens, usage.output_tokens),
        tokens_estimated=estimated,
        finish_reason=finish_reason,
        time_to_first_token_ms=None if ttft_ms is None else round(ttft_ms, 1),
    )


class StreamingResponse:
    """Loop over it to get text pieces as they arrive.

    After the loop finishes, .response holds the full Response.
    """

    def __init__(self, events, model_config: ModelConfig, messages: list[dict]):
        self._events = events
        self._model_config = model_config
        self._messages = messages
        self._started = False
        self.response: Optional[Response] = None

    def __iter__(self) -> Iterator[str]:
        if self._started:
            raise RuntimeError("A stream can only be read once")
        self._started = True

        started_at = time.perf_counter()  # the request really starts here
        pieces, usage, first_piece_at = [], None, None

        for event in self._events:
            if isinstance(event, Usage):
                usage = event
                continue
            if first_piece_at is None:
                first_piece_at = time.perf_counter()
            pieces.append(event)
            yield event

        finished_at = time.perf_counter()
        ttft_ms = None if first_piece_at is None else (first_piece_at - started_at) * 1000
        self.response = _build_response(
            self._model_config, "".join(pieces), usage, self._messages,
            (finished_at - started_at) * 1000, ttft_ms=ttft_ms,
        )


def _validate_messages(messages) -> None:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    for index, message in enumerate(messages):
        if (
            not isinstance(message, dict)
            or message.get("role") not in VALID_ROLES
            or not isinstance(message.get("content"), str)
        ):
            raise ValueError(
                f"message #{index + 1} must look like "
                "{'role': 'user', 'content': '...'} with role system, user, or assistant"
            )
    if all(message["role"] == "system" for message in messages):
        raise ValueError("messages need at least one user message")


def send_request(
    messages: list[dict],
    model_config: ModelConfig,
    stream: bool = False,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
):
    """Send messages to any model.

    stream=False returns a Response.
    stream=True returns a StreamingResponse to loop over.
    Raises ProviderError when the provider fails.
    """
    _validate_messages(messages)

    adapter = ADAPTERS.get(model_config.provider)
    if adapter is None:
        raise ProviderError(model_config.provider, "config", "no adapter for this provider")

    # Some models have a low output limit (e.g. on a free tier), so never ask for more
    if model_config.max_output_tokens is not None:
        max_tokens = min(max_tokens, model_config.max_output_tokens)

    settings = {
        "stream": stream,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }

    if stream:
        events = adapter(messages, model_config.model_id, **settings)
        return StreamingResponse(events, model_config, messages)

    started_at = time.perf_counter()
    raw = adapter(messages, model_config.model_id, **settings)
    latency_ms = (time.perf_counter() - started_at) * 1000
    return _build_response(model_config, raw.text, raw.usage, messages, latency_ms, raw.finish_reason)