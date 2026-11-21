"""Adapter for Google Gemini, using Googl'es own library"""
import httpx
from google import genai
from google.genai import errors, types

from app.providers.common import (
    ProviderError, RawResult, Usage, kind_from_status, server_or_given_key, split_system,
)

PROVIDER = "gemini"
# Tests set this to an httpx client with a MockTransport to see the exact request; None = the SDK's own client
HTTP_CLIENT = None
ROLE_MAP = {"user": "user", "assistant": "model"} # Gemini calls the AI role "model"

def _is_deadline(error: errors.APIError) -> bool:
    """Google's server stopped the call at its deadline (504 DEADLINE_EXCEEDED, "Deadline expired before operation
    could complete"). The SDK sends our timeout as X-Server-Timeout, so this is our own time limit running out
    on Google's side: a timeout, not a server fault."""
    return (error.code == 504 or (error.status or "").upper() == "DEADLINE_EXCEEDED"
            or "deadline expired" in (error.message or "").lower())


def _is_overloaded(error: errors.APIError) -> bool:
    """503 UNAVAILABLE: "This model is currently experiencing high demand" (Google's overload answer)."""
    return error.code == 503 or (error.status or "").upper() == "UNAVAILABLE"


def _convert_error(error) -> ProviderError:
    if isinstance(error, errors.APIError):
        code = error.code or 0
        if _is_deadline(error):
            kind = "timeout"
        elif _is_overloaded(error):
            kind = "overloaded"
        else:
            kind = kind_from_status(code)
        return ProviderError(PROVIDER, kind, error.message or str(error), code or None)
    if isinstance(error, httpx.TimeoutException):
        return ProviderError(PROVIDER, "timeout", "request timed out")
    return ProviderError(PROVIDER, "connection", "could not reach the provider")

def _usage_from(meta) -> Usage | None:
    if meta is None:
        return None
    # Thinking tokens are billed as output, so count them as output
    thoughts = meta.thoughts_token_count or 0
    output = (meta.candidates_token_count or 0) + thoughts
    return Usage(meta.prompt_token_count or 0, output, thoughts if meta.thoughts_token_count is not None else None)

def _to_contents(turns):
    return [
        types.Content(role=ROLE_MAP[m["role"]], parts=[types.Part(text=m["content"])])
        for m in turns
    ]

def send(messages, model_id, *, stream, temperature, max_tokens, timeout, api_key=None, thinking_level=None):
    """api_key: the caller's own key (BYOK); None = GEMINI_API_KEY from the environment.
    thinking_level: "low" etc., sent as ThinkingConfig(thinking_level="LOW"); None = no thinking config (the
    model's default level)."""
    # HttpOptions.timeout is in MILLISECONDS (our timeouts are seconds). The SDK turns it back into seconds for
    # httpx and also sends it to Google as X-Server-Timeout, so the server stops the call at the same limit.
    # retry_options attempts=1: no SDK retries (google-genai retries with tenacity when retry_options is set; 2.29's
    # default is none, pinned here so a new SDK default can't add hidden retries). Triage's reliability layer
    # decides when to retry or fall back, as max_retries=0 does for the openai and anthropic SDKs.
    http_options = {"timeout": int(timeout * 1000), "retry_options": types.HttpRetryOptions(attempts=1)}
    if HTTP_CLIENT is not None:
        http_options["httpx_client"] = HTTP_CLIENT
    client = genai.Client(
        api_key=server_or_given_key(api_key, "GEMINI_API_KEY", PROVIDER),
        http_options=types.HttpOptions(**http_options),
    )
    system_text, turns = split_system(messages)
    # Gemini 3.6+ ignores temperature ("deprecated and ignored"), and Google says future models will return HTTP
    # 400 for it (aistudio.google.com/docs/whats-new-gemini-3.6, checked 2026-11-12). Still sent so requests stay as
    # they are today; drop it (as base.send_request does for `reasoning` models) when a Gemini model starts rejecting it.
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_text or None,
        thinking_config=types.ThinkingConfig(thinking_level=thinking_level.upper()) if thinking_level else None,
    )
    model = model_id.removeprefix("models/") #accept IDs copied from the model list
    contents = _to_contents(turns)

    if stream:
        return _stream(client, model, contents, config)

    try:
        response = client.models.generate_content(model=model, contents=contents, config=config)
    except (errors.APIError, httpx.HTTPError) as error:
        raise _convert_error(error) from None

    finish_reason = None
    if response.candidates and response.candidates[0].finish_reason:
        reason = response.candidates[0].finish_reason
        finish_reason = getattr(reason, "name", str(reason))

    return RawResult(
        text=response.text or "",
        usage=_usage_from(response.usage_metadata),
        finish_reason=finish_reason,
    )

def _stream(client, model, contents, config):
    """Yield text pieces as they arrive, then a final Usage"""
    usage = None
    chunks = client.models.generate_content_stream(model=model, contents=contents, config=config)
    try:
        for chunk in chunks:
            if chunk.text:
                yield chunk.text
            if chunk.usage_metadata:
                usage = _usage_from(chunk.usage_metadata) # the last chunk has the totals
    except (errors.APIError, httpx.HTTPError) as error:
        raise _convert_error(error) from None
    finally:
        close = getattr(chunks, "close", None)        # stopped early: end the stream, so the model stops
        if close is not None:
            close()
    if usage:
        yield usage