# Shared building blocks used by every provider adapter 
import os 
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class Usage:
    # token counts reported by a provider 
    input_tokens: int
    output_tokens: int              # includes reasoning tokens, since providers bill them as output
    reasoning_tokens: Optional[int] = None   # the reasoning share of output_tokens, if reported
    
@dataclass(frozen=True)
class RawResult:
    """What an adapter reeturns for a normal (non-streaming) call"""
    text: str
    usage: Optional[Usage]
    finish_reason: Optional[str] = None
    reasoning: Optional[str] = None   # reasoning returned separately from the answer (e.g. gpt-oss on Groq)

class ProviderError(Exception):
    """One error type for every provider.
 
    kind is one of:
      rate_limit  429, too many requests
      server      5xx, a problem on the provider's side
      timeout     no answer in time
      connection  couldn't reach the provider
      client      4xx, e.g. bad key or bad model name
      config      a problem on our side, e.g. a missing API key
    """
    RETRYABLE_KINDS = {"rate_limit", "server"}
    def __init__(self, provider: str, kind: str, message: str, status_code: Optional[int] = None):
        super().__init__(f"[{provider}] {kind}: {message}")
        self.provider = provider 
        self.kind = kind
        self.status_code = status_code
    
    @property
    def retryable(self) -> bool:
        """Only rate limits and server errors are worth retrying"""
        return self.kind in self.RETRYABLE_KINDS

def require_key(env_name: str, provider: str)-> str:
    """Read an API key from the environment, or fall with a clear error"""
    key = os.getenv(env_name)
    if not key: 
        raise ProviderError(provider, "config", f"{env_name} is not set in the .env")
    return key

def kind_from_status(status_code: int) -> str:
    """Turn an HTTP status code into a ProviderError kind"""
    if status_code == 429:
        return "rate_limit"
    if status_code >= 500:
        return "server"
    return "client"

def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Seperate system messages from the conversation turns. Gemini and Anthropic want the system text passed seperately """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] != "system"]
    return "\n\n".join(system_parts), turns