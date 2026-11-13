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
class Finish:
    """Streams only: the provider's finish reason, yielded once near the end (adapters that report it)."""
    reason: Optional[str]


# Finish reasons meaning the answer was cut off by the token limit (lowercased; Gemini says "MAX_TOKENS")
TRUNCATED_FINISH_REASONS = {"length", "max_tokens"}

    
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
      auth        401 / 403: the key is missing, wrong or not allowed (provider-wide)
      client      other 4xx, e.g. a bad request or model name
      config      a problem on our side, e.g. a missing API key
      truncated   a reasoning model used every output token (thinking) and wrote no answer (base.TruncatedAnswer)
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

# The .env variable holding each provider's server-side key (None: needs no key, e.g. local Ollama)
PROVIDER_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
}


def missing_key_env(provider: str) -> Optional[str]:
    """The env variable name if this provider needs a key and none is set, else None (scripts use this to
    report a model as "skipped: no key" instead of a failure)."""
    env_name = PROVIDER_KEY_ENV.get(provider)
    return env_name if env_name and not os.getenv(env_name) else None


def server_or_given_key(api_key: Optional[str], env_name: str, provider: str) -> str:
    """The caller's own key when one is given (BYOK: never replaced by the env key, even if empty), else the
    server's key from the environment."""
    if api_key is not None:
        if not api_key:
            raise ProviderError(provider, "config", "an empty API key was given")
        return api_key
    return require_key(env_name, provider)


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
    if status_code in (401, 403):
        return "auth"
    return "client"

def split_system(messages: list[dict]) -> tuple[str, list[dict]]:
    """Seperate system messages from the conversation turns. Gemini and Anthropic want the system text passed seperately """
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    turns = [m for m in messages if m["role"] != "system"]
    return "\n\n".join(system_parts), turns