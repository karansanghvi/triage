"""The one function the rest of Triage uses to talk to any model."""
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from dotenv import load_dotenv

from app.providers import (
    anthropic_provider, gemini_provider, groq_provider,
    ollama_provider, openai_provider, openrouter_provider,
)
from app.providers.common import TRUNCATED_FINISH_REASONS, Finish, ProviderError, Usage
from app.registry import ModelConfig

load_dotenv()

VALID_ROLES = {"system", "user", "assistant"}
DEFAULT_TEMPERATURE = 0.0      # 0 = same question gives the same answer (needed for caching)
DEFAULT_MAX_TOKENS = 1024      # leaves room for reasoning models to think
DEFAULT_TIMEOUT_SECONDS = 30.0

# Providers whose adapter speaks OpenAI's format (and so accepts reasoning_effort)
OPENAI_FORMAT_PROVIDERS = {"openai", "groq", "ollama", "openrouter"}
# Providers that accept reasoning_effort: the OpenAI-format ones, plus Anthropic (sent as output_config.effort)
REASONING_EFFORT_PROVIDERS = OPENAI_FORMAT_PROVIDERS | {"anthropic"}
# Anthropic only: thinking modes we may send (claude-haiku: "disabled")
THINKING_MODES = {"disabled", "adaptive"}

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
    cached: bool = False             # True if this answer came from the cache
    cache_saved_cost: float = 0.0    # on a cache hit, the reported cost we didn't pay
    reasoning: Optional[str] = None          # reasoning returned separately from text, if any
    reasoning_tokens: Optional[int] = None   # reasoning share of output_tokens, if reported


class TruncatedAnswer(ProviderError):
    """A reasoning model spent every output token (thinking counts toward max_tokens) and wrote no answer.

    Not retried and not counted by the circuit breaker: the chain falls back to the next model. .response is
    the full Response (its tokens were billed), so the API can add its cost to the request's total."""

    def __init__(self, response: "Response", max_tokens: int):
        super().__init__(response.provider, "truncated",
                         f"{response.model_name} used its {max_tokens} output tokens without writing an answer "
                         f"(finish_reason {response.finish_reason!r})")
        self.response = response
        self.max_tokens = max_tokens


def _is_truncated_empty(model_config, response: "Response") -> bool:
    """Only for models marked reasoning: an empty answer cut off by the token limit."""
    return (model_config.reasoning and not response.text.strip()
            and (response.finish_reason or "").lower() in TRUNCATED_FINISH_REASONS)


def _estimate_tokens(text: str) -> int:
    """Rough guess (about 4 characters per token) if a provider skips usage."""
    return max(1, len(text) // 4) if text else 0


def _build_response(model_config, text, usage, messages, latency_ms, finish_reason=None, ttft_ms=None,
                    reasoning=None):
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
        reasoning=reasoning,
        reasoning_tokens=usage.reasoning_tokens,
    )


class StreamingResponse:
    """Loop over it to get text pieces as they arrive.

    After the loop finishes, .response holds the full Response.
    """

    def __init__(self, events, model_config: ModelConfig, messages: list[dict], max_tokens: Optional[int] = None):
        self._events = events
        self._model_config = model_config
        self._messages = messages
        self._max_tokens = max_tokens
        self._started = False
        self.response: Optional[Response] = None

    def __iter__(self) -> Iterator[str]:
        if self._started:
            raise RuntimeError("A stream can only be read once")
        self._started = True

        started_at = time.perf_counter()  # the request really starts here
        pieces, usage, first_piece_at, finish_reason = [], None, None, None

        for event in self._events:
            if isinstance(event, Usage):
                usage = event
                continue
            if isinstance(event, Finish):
                finish_reason = event.reason
                continue
            if first_piece_at is None:
                first_piece_at = time.perf_counter()
            pieces.append(event)
            yield event

        finished_at = time.perf_counter()
        ttft_ms = None if first_piece_at is None else (first_piece_at - started_at) * 1000
        self.response = _build_response(
            self._model_config, "".join(pieces), usage, self._messages,
            (finished_at - started_at) * 1000, finish_reason=finish_reason, ttft_ms=ttft_ms,
        )
        # No piece was yielded, so this is raised from the reliability layer's first-token wait: it can fall back
        if _is_truncated_empty(self._model_config, self.response):
            raise TruncatedAnswer(self.response, self._max_tokens)


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
    reasoning_effort: Optional[str] = None,
    response_format: Optional[dict] = None,
    thinking: Optional[str] = None,
):
    """Send messages to any model.

    stream=False returns a Response.
    stream=True returns a StreamingResponse to loop over.
    reasoning_effort ("low", "medium", "high") is passed to reasoning models on OpenAI-format
    providers such as Groq's gpt-oss; leave it None for every other model.
    reasoning_effort is also accepted for Anthropic (sent as output_config.effort).
    response_format (e.g. {"type": "json_object"}) asks OpenAI-format providers for JSON-only output.
    thinking ("disabled" or "adaptive") is Anthropic only.
    Models marked `reasoning` get no temperature, and an empty answer cut off by max_tokens raises TruncatedAnswer.
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
        # Reasoning models get no temperature (None = not sent). Anthropic: sending it is a 400 on API version
        # 2026-10-21 (and the SDK no longer accepts it). OpenAI: not documented either way for gpt-6.x, and
        # leaving it out is always valid.
        "temperature": None if model_config.reasoning else temperature,
        "max_tokens": max_tokens,
        "timeout": timeout,
    }
    if reasoning_effort is not None:
        if model_config.provider not in REASONING_EFFORT_PROVIDERS:
            raise ProviderError(model_config.provider, "config", "reasoning_effort is only supported on OpenAI-format providers and Anthropic")
        settings["reasoning_effort"] = reasoning_effort
    if thinking is not None:
        if model_config.provider != "anthropic" or thinking not in THINKING_MODES:
            raise ProviderError(model_config.provider, "config", f"thinking={thinking!r} is only supported on Anthropic "
                                f"({', '.join(sorted(THINKING_MODES))})")
        settings["thinking"] = thinking
    if response_format is not None:
        if model_config.provider not in OPENAI_FORMAT_PROVIDERS:
            raise ProviderError(model_config.provider, "config", "response_format is only supported on OpenAI-format providers")
        settings["response_format"] = response_format

    if stream:
        events = adapter(messages, model_config.model_id, **settings)
        return StreamingResponse(events, model_config, messages, max_tokens)

    started_at = time.perf_counter()
    raw = adapter(messages, model_config.model_id, **settings)
    latency_ms = (time.perf_counter() - started_at) * 1000
    response = _build_response(
        model_config, raw.text, raw.usage, messages, latency_ms, raw.finish_reason, reasoning=raw.reasoning,
    )
    if _is_truncated_empty(model_config, response):
        raise TruncatedAnswer(response, max_tokens)
    return response