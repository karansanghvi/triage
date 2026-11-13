"""Adapter for Anthropic's Claude models (BYOK only: see routing.yaml server_providers).

Claude 5 models are reasoning models (adaptive thinking is on by default and counts toward max_tokens), and on
API version 2026-10-21 any temperature / top_p / top_k is a 400, so none is ever sent. Thinking blocks are left out
of the answer text; their tokens are billed as output and already counted in usage.output_tokens.
Docs: platform.claude.com/docs/en/build-with-claude/thinking and /effort, /api/versioning (checked 2026-11-12).
"""
import anthropic

from app.providers.common import (
    Finish, ProviderError, RawResult, Usage, kind_from_status, server_or_given_key, split_system,
)

PROVIDER = "anthropic"
# Pinned so a new SDK default can't change request rules under us (2026-10-21: sampling parameters are a 400)
API_VERSION = "2026-10-21"
# Tests set this to an httpx.Client with a MockTransport to see the exact request; None = the SDK's own client
HTTP_CLIENT = None


def _convert_error(error) -> ProviderError:
    if isinstance(error, ProviderError):
        return error
    if isinstance(error, anthropic.APITimeoutError):
        return ProviderError(PROVIDER, "timeout", "request timed out")
    if isinstance(error, anthropic.APIConnectionError):
        return ProviderError(PROVIDER, "connection", "could not reach the provider")
    if isinstance(error, anthropic.APIStatusError):
        return ProviderError(PROVIDER, kind_from_status(error.status_code), error.message, error.status_code)
    # Anything else (a TypeError from an SDK argument change, the SDK refusing a request locally, ...): a client
    # error so the chain moves on instead of a 500. Only the type name: str(error) could echo request details,
    # including a key, until key redaction exists (BYOK wiring step).
    return ProviderError(PROVIDER, "client", type(error).__name__)


def _request(messages, model_id, max_tokens, reasoning_effort, thinking) -> dict:
    system_text, turns = split_system(messages)
    request = {"model": model_id, "messages": turns, "max_tokens": max_tokens}
    if system_text:
        request["system"] = system_text
    if reasoning_effort is not None:
        request["output_config"] = {"effort": reasoning_effort}
    if thinking is not None:
        request["thinking"] = {"type": thinking}
    return request


def send(messages, model_id, *, stream, max_tokens, timeout, temperature=None, reasoning_effort=None,
         thinking=None, api_key=None):
    """temperature is accepted for a uniform adapter signature and never sent (see the module docstring).
    api_key: the caller's own key (BYOK); None = ANTHROPIC_API_KEY from the environment."""
    api_key = server_or_given_key(api_key, "ANTHROPIC_API_KEY", PROVIDER)
    try:                                   # building the client can fail too (e.g. a bad http_client)
        client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout,
            max_retries=0,
            default_headers={"anthropic-version": API_VERSION},
            http_client=HTTP_CLIENT,
        )
    except Exception as error:
        raise _convert_error(error) from None
    request = _request(messages, model_id, max_tokens, reasoning_effort, thinking)

    if stream:
        return _stream(client, request)

    try:
        message = client.messages.create(**request)
    except Exception as error:
        raise _convert_error(error) from None

    text = "".join(block.text for block in message.content if block.type == "text")
    return RawResult(
        text=text,
        usage=Usage(message.usage.input_tokens, message.usage.output_tokens),
        finish_reason=message.stop_reason,
    )


def _stream(client, request):
    """Yield text pieces as they arrive, then a Finish (stop_reason) and the final Usage."""
    try:
        with client.messages.stream(**request) as stream:
            for text in stream.text_stream:          # an iterator attribute, not a method
                yield text
            final = stream.get_final_message()
    except Exception as error:
        raise _convert_error(error) from None
    yield Finish(final.stop_reason)
    yield Usage(final.usage.input_tokens, final.usage.output_tokens)
