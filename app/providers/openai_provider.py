"""Adapter for every provider that uses OpenAI's request format."""
import openai
from openai import OpenAI

from app.providers.common import ProviderError, RawResult, Usage, kind_from_status, require_key


def _convert_error(error: openai.APIError, provider: str) -> ProviderError:
    # Order matters: a timeout is a special kind of connection error
    if isinstance(error, openai.APITimeoutError):
        return ProviderError(provider, "timeout", "request timed out")
    if isinstance(error, openai.APIConnectionError):
        return ProviderError(provider, "connection", "could not reach the provider")
    if isinstance(error, openai.APIStatusError):
        kind = kind_from_status(error.status_code)
        # Groq sends a 429 when one request is bigger than the per-minute limit.
        # Waiting never fixes that, so treat it as a client error: no retries.
        if kind == "rate_limit" and "request too large" in str(error.message).lower():
            kind = "client"
        return ProviderError(provider, kind, error.message, error.status_code)
    return ProviderError(provider, "client", str(error))


def _usage_from(usage) -> Usage | None:
    if usage is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = getattr(details, "reasoning_tokens", None) if details else None
    return Usage(usage.prompt_tokens or 0, usage.completion_tokens or 0, reasoning)


def _reasoning_from(message) -> str | None:
    """Reasoning some providers return beside the answer (Groq's gpt-oss uses a 'reasoning' field)."""
    reasoning = getattr(message, "reasoning", None) or (message.model_extra or {}).get("reasoning")
    return reasoning or None


def send_openai_format(
    messages, model_id, *, provider, base_url, api_key, stream, temperature, max_tokens, timeout,
    reasoning_effort=None, response_format=None,
):
    # max_retries=0: the library retries on its own by default.
    # Triage's reliability layer decides when to retry instead.
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=0)

    extra = {"reasoning_effort": reasoning_effort} if reasoning_effort is not None else {}
    if response_format is not None:
        extra["response_format"] = response_format
    if stream:
        return _stream(client, messages, model_id, provider, temperature, max_tokens, extra)

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        )
    except openai.APIError as error:
        raise _convert_error(error, provider) from error

    choice = completion.choices[0]
    return RawResult(
        text=choice.message.content or "",
        usage=_usage_from(completion.usage),
        finish_reason=choice.finish_reason,
        reasoning=_reasoning_from(choice.message),
    )


def _stream(client, messages, model_id, provider, temperature, max_tokens, extra=None):
    """Yield text pieces as they arrive, then a final Usage."""
    try:
        chunks = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            stream_options={"include_usage": True},  # ask for token counts at the end
            **(extra or {}),
        )
        for chunk in chunks:
            if chunk.choices:
                piece = chunk.choices[0].delta.content
                if piece:
                    yield piece
            if chunk.usage:
                yield _usage_from(chunk.usage)
    except openai.APIError as error:
        raise _convert_error(error, provider) from error


def send(messages, model_id, **settings):
    """Real OpenAI (only used for the optional final benchmark)."""
    return send_openai_format(
        messages,
        model_id,
        provider="openai",
        base_url=None,  # None means OpenAI's own servers
        api_key=require_key("OPENAI_API_KEY", "openai"),
        **settings,
    )