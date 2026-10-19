"""Adapter for Google Gemini, using Googl'es own library"""
import httpx 
from google import genai 
from google.genai import errors, types 

from app.providers.common import (
    ProviderError, RawResult, Usage, kind_from_status, require_key, split_system,
)

PROVIDER = "gemini"
ROLE_MAP = {"user": "user", "assistant": "model"} # Gemini calls the AI role "model"

def _convert_error(error) -> ProviderError:
    if isinstance(error, errors.APIError):
        code = error.code or 0 
        return ProviderError(PROVIDER, kind_from_status(code), error.message or str(error), code or None)
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
    
def send(messages, model_id, *, stream, temperature, max_tokens, timeout):
    client = genai.Client(
        api_key=require_key("GEMINI_API_KEY", PROVIDER),
        http_options=types.HttpOptions(timeout=int(timeout * 1000)), #milliseconds
    )
    system_text, turns = split_system(messages)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system_text or None,
    )
    model = model_id.removeprefix("models/") #accept IDs copied from the model list 
    contents = _to_contents(turns)
    
    if stream:
        return _stream(client, model, contents, config)
    
    try:
        response = client.models.generate_content(model=model, contents=contents, config=config)
    except (errors.APIError, httpx.HTTPError) as error:
        raise _convert_error(error) from error 
    
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
    try:
        for chunk in client.models.generate_content_stream(
            model=model, contents=contents, config=config
        ):
            if chunk.text:
                yield chunk.text
            if chunk.usage_metadata:
                usage = _usage_from(chunk.usage_metadata) # the last chunk has the totals 
    except (errors.APIError, httpx.HTTPError) as error:
        raise _convert_error(error) from error
    if usage:
        yield usage