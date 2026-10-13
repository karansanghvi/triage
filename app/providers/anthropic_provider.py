"""Adapter for Anthropic's Claude models"""
import anthropic

from app.providers.common import(
    ProviderError, RawResult, Usage, kind_from_status, require_key, split_system,
)

PROVIDER = "anthropic"

def _convert_error(error) -> ProviderError:
    if isinstance(error, anthropic.APITimeoutError):
        return ProviderError(PROVIDER, "timeout", "request timed out")
    if isinstance(error, anthropic.APIConnectionError):
        return ProviderError(PROVIDER, "connection", "could not reach the provider")
    if isinstance(error, anthropic.APIError):
        return ProviderError(PROVIDER, kind_from_status(error.status_code), error.message, error.status_code)
    return ProviderError(PROVIDER, "client", str(error))

def send(messages, model_id, *, stream, temperature, max_tokens, timeout):
    client = anthropic.Anthropic(
        api_key=require_key("ANTHROPIC_API_KEY", PROVIDER),
        timeout=timeout,
        max_retries=0,
    )
    system_text, turns = split_system(messages)
    request = {
        "model": model_id,
        "messages": turns,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_text: 
        request["system"] = system_text
    
    if stream:
        return _stream(client, request)
    
    try:
        message = client.messages.create(**request)
    except anthropic.APIError as error:
        raise _convert_error(error) from error
    
    text = "".join(block.text for block in message.content if block.type == "text")
    return RawResult(
        text=text,
        usage=Usage(message.usage.input_tokens, message.usage.output_tokens),
        finish_reason=message.stop_reason,
    )

def _stream(client, request):
    """Yield text pieces as they arrive, then a final Usage"""
    try:
        with client.messages.stream(**request) as stream:
            for text in stream.text_stream():
                yield text
            final = stream.get_final_message()
    except anthropic.APIError as error:
        raise _convert_error(error) from error
    yield Usage(final.usage.input_tokens, final.usage.output_tokens)