"""Reliability layer: timeouts, retries, fallback chains, and a circuit breaker."""
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from app.providers.base import send_request
from app.providers.common import ProviderError
from app.registry import PROJECT_ROOT, VALID_TIERS, ModelConfig, load_registry

logger = logging.getLogger("triage.reliability")

DEFAULT_ROUTING_PATH = PROJECT_ROOT / "config" / "routing.yaml"

# Errors that mean "this whole provider is unreachable or refusing us": skip all its models.
# A connection error means the provider can't be reached at all; an auth error (401/403) means its key is
# wrong or revoked, which no other model on that provider will get past either.
PROVIDER_BREAKER_KINDS = {"connection", "auth"}
# Errors that are about ONE model: skip only that model. Server errors (5xx) and timeouts usually come from
# one overloaded or slow model (a long groq-large answer timing out says nothing about groq-small), and rate
# limits are per model on Groq. Before Phase 4 Step 7 these tripped the whole provider, so one bad model could
# take every Groq tier down for the cooldown.
MODEL_BREAKER_KINDS = {"server", "timeout", "rate_limit"}
# Client errors (like a bad request) are usually our fault, so they don't count. Neither do config errors.


# ---------- Settings (loaded from routing.yaml) ----------

@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3            # tries per model, including the first
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 8.0


@dataclass(frozen=True)
class BreakerSettings:
    failure_threshold: int = 3
    cooldown_seconds: float = 60.0


# Generation settings that apply to every model, and request parameters that are set per MODEL
# (never per call or per tier), so a fallback model never receives a parameter meant for another model.
GENERATION_KEYS = {"temperature", "max_tokens"}
MODEL_PARAM_KEYS = {"reasoning_effort"}
# Providers whose adapter speaks OpenAI's format (the only ones that accept reasoning_effort)
OPENAI_FORMAT_PROVIDERS = {"openai", "groq", "ollama", "openrouter"}


@dataclass(frozen=True)
class RoutingConfig:
    chains: dict                     # tier -> list of model names, primary first
    retry: RetryPolicy
    breaker: BreakerSettings
    timeout_seconds: float = 30.0
    generation: dict = field(default_factory=dict)     # e.g. {"temperature": 0, "max_tokens": 4000}
    model_params: dict = field(default_factory=dict)   # model name -> e.g. {"reasoning_effort": "low"}
    model_timeouts: dict = field(default_factory=dict) # model name -> seconds, overrides timeout_seconds
    # Streaming only: how long to wait for the FIRST piece before giving up on a model (then fall back).
    # None = no first-token limit (the per-call timeout still applies to every read). There is never a
    # total-time limit on a stream: a long answer that is arriving steadily is not cut off.
    first_token_timeout_seconds: Optional[float] = None
    model_first_token_timeouts: dict = field(default_factory=dict)

    def timeout_for(self, model_name: str) -> float:
        return self.model_timeouts.get(model_name, self.timeout_seconds)

    def first_token_timeout_for(self, model_name: str) -> Optional[float]:
        return self.model_first_token_timeouts.get(model_name, self.first_token_timeout_seconds)


def _positive_number(value, field: str, *, whole: bool = False):
    allowed = (int,) if whole else (int, float)
    if isinstance(value, bool) or not isinstance(value, allowed) or value <= 0:
        kind = "a whole number" if whole else "a number"
        raise ValueError(f"routing.yaml: '{field}' must be {kind} above 0, got {value!r}")
    return value


def load_routing(registry: dict, path=DEFAULT_ROUTING_PATH) -> RoutingConfig:
    """Read routing.yaml and check every model name exists in the registry."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Routing config not found at {path}")

    with path.open() as file:
        data = yaml.safe_load(file) or {}
    return routing_from_dict(data, registry)


def routing_from_dict(data: dict, registry: dict) -> RoutingConfig:
    """Validate an already-parsed routing.yaml (the API's live config update uses this too)."""
    if not isinstance(data, dict):
        raise ValueError("routing.yaml must be a mapping")
    raw_chains = data.get("fallback_chains")
    if not isinstance(raw_chains, dict):
        raise ValueError("routing.yaml needs a 'fallback_chains' section")

    unknown_tiers = set(raw_chains) - VALID_TIERS
    if unknown_tiers:
        raise ValueError(f"routing.yaml: unknown tier(s): {', '.join(sorted(unknown_tiers))}")

    chains = {}
    for tier in ("low", "medium", "high"):
        names = raw_chains.get(tier)
        if not isinstance(names, list) or not names:
            raise ValueError(f"routing.yaml: tier '{tier}' needs a non-empty list of models")
        for name in names:
            if name not in registry:
                raise ValueError(
                    f"routing.yaml: tier '{tier}' lists unknown model '{name}'. "
                    "Check the names in models.yaml"
                )
        if len(set(names)) != len(names):
            raise ValueError(f"routing.yaml: tier '{tier}' lists the same model twice")
        chains[tier] = list(names)

    raw_retry = data.get("retry") or {}
    retry = RetryPolicy(
        max_attempts=_positive_number(raw_retry.get("max_attempts", 3), "retry.max_attempts", whole=True),
        base_delay_seconds=float(_positive_number(raw_retry.get("base_delay_seconds", 0.5), "retry.base_delay_seconds")),
        max_delay_seconds=float(_positive_number(raw_retry.get("max_delay_seconds", 8), "retry.max_delay_seconds")),
    )

    raw_breaker = data.get("circuit_breaker") or {}
    breaker = BreakerSettings(
        failure_threshold=_positive_number(raw_breaker.get("failure_threshold", 3), "circuit_breaker.failure_threshold", whole=True),
        cooldown_seconds=float(_positive_number(raw_breaker.get("cooldown_seconds", 60), "circuit_breaker.cooldown_seconds")),
    )

    timeout = float(_positive_number(data.get("timeout_seconds", 30), "timeout_seconds"))
    first_token, model_first_token = _load_streaming(data.get("streaming"), registry)
    return RoutingConfig(chains=chains, retry=retry, breaker=breaker, timeout_seconds=timeout,
                         generation=_load_generation(data.get("generation")),
                         model_params=_load_model_params(data.get("model_params"), registry),
                         model_timeouts=_load_model_timeouts(data.get("model_timeouts"), registry),
                         first_token_timeout_seconds=first_token, model_first_token_timeouts=model_first_token)


def _load_streaming(raw, registry: dict) -> tuple:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("routing.yaml: 'streaming' must be a mapping")
    unknown = set(raw) - {"first_token_timeout_seconds", "model_first_token_timeouts"}
    if unknown:
        raise ValueError(f"routing.yaml: unsupported streaming setting(s): {', '.join(sorted(unknown))}")
    default = raw.get("first_token_timeout_seconds")
    default = None if default is None else float(_positive_number(default, "streaming.first_token_timeout_seconds"))
    per_model = raw.get("model_first_token_timeouts") or {}
    if not isinstance(per_model, dict):
        raise ValueError("routing.yaml: 'streaming.model_first_token_timeouts' must be a mapping")
    out = {}
    for name, seconds in per_model.items():
        if name not in registry:
            raise ValueError(f"routing.yaml: streaming.model_first_token_timeouts lists unknown model '{name}'")
        out[name] = float(_positive_number(seconds, f"streaming.model_first_token_timeouts.{name}"))
    return default, out


def _load_model_timeouts(raw, registry: dict) -> dict:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("routing.yaml: 'model_timeouts' must be a mapping of model name -> seconds")
    out = {}
    for name, seconds in raw.items():
        if name not in registry:
            raise ValueError(f"routing.yaml: model_timeouts lists unknown model '{name}'")
        out[name] = float(_positive_number(seconds, f"model_timeouts.{name}"))
    return out


def _load_generation(raw) -> dict:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("routing.yaml: 'generation' must be a mapping")
    unknown = set(raw) - GENERATION_KEYS
    if unknown:
        raise ValueError(f"routing.yaml: unsupported generation setting(s): {', '.join(sorted(unknown))}")
    out = {}
    if "temperature" in raw:
        t = raw["temperature"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or t < 0:
            raise ValueError(f"routing.yaml: 'generation.temperature' must be a number >= 0, got {t!r}")
        out["temperature"] = float(t)
    if "max_tokens" in raw:
        out["max_tokens"] = _positive_number(raw["max_tokens"], "generation.max_tokens", whole=True)
    return out


def _load_model_params(raw, registry: dict) -> dict:
    raw = raw or {}
    if not isinstance(raw, dict):
        raise ValueError("routing.yaml: 'model_params' must be a mapping of model name -> parameters")
    out = {}
    for name, params in raw.items():
        if name not in registry:
            raise ValueError(f"routing.yaml: model_params lists unknown model '{name}'")
        if not isinstance(params, dict):
            raise ValueError(f"routing.yaml: model_params for '{name}' must be a mapping")
        unknown = set(params) - MODEL_PARAM_KEYS
        if unknown:
            raise ValueError(f"routing.yaml: unsupported model_params for '{name}': {', '.join(sorted(unknown))}")
        if "reasoning_effort" in params and registry[name].provider not in OPENAI_FORMAT_PROVIDERS:
            raise ValueError(f"routing.yaml: '{name}' ({registry[name].provider}) does not accept reasoning_effort")
        out[name] = dict(params)
    return out


# ---------- Backoff with jitter ----------

def backoff_delay(attempt: int, policy: RetryPolicy, rng: random.Random) -> float:
    """How long to wait before the next try.

    The limit doubles each time (0.5s, 1s, 2s, ...) up to max_delay_seconds.
    The actual wait is a random amount between 0 and that limit ("full jitter"),
    so many failed requests don't all retry at the exact same moment.
    """
    limit = min(policy.max_delay_seconds, policy.base_delay_seconds * 2 ** (attempt - 1))
    return rng.uniform(0, limit)


# ---------- Circuit breaker ----------

def _model_key(model: ModelConfig) -> str:
    return f"model:{model.name}"


class CircuitBreaker:
    """Skips something (a provider, or one model) after too many failures in a row.

    It tracks failures by a "key": a provider name like "groq", or a model
    key like "model:groq-qwen". The client decides which key to use.

    After the cooldown, the next request is allowed through. If it works,
    the counter resets. If it fails, the breaker trips again right away,
    because the failure count is still at or above the threshold.
    """

    def __init__(self, settings: BreakerSettings, clock=time.monotonic):
        self._settings = settings
        self._clock = clock                  # monotonic: only moves forward, safe for durations
        self._failures: dict[str, int] = {}
        self._open_until: dict[str, float] = {}
        self._lock = threading.Lock()        # the API shares one breaker across worker threads

    def is_open(self, key: str) -> bool:
        with self._lock:
            return self._clock() < self._open_until.get(key, 0.0)

    def seconds_left(self, key: str) -> float:
        with self._lock:
            return max(0.0, self._open_until.get(key, 0.0) - self._clock())

    def record_success(self, key: str) -> None:
        with self._lock:
            self._failures[key] = 0
            self._open_until.pop(key, None)

    def record_failure(self, key: str) -> None:
        with self._lock:
            count = self._failures.get(key, 0) + 1
            self._failures[key] = count
            tripped = count >= self._settings.failure_threshold
            if tripped:
                self._open_until[key] = self._clock() + self._settings.cooldown_seconds
        if tripped:
            logger.warning(
                "circuit open: skipping %s for %.0fs after %d failures in a row",
                key, self._settings.cooldown_seconds, count,
            )


# ---------- Results and logs ----------

@dataclass(frozen=True)
class Attempt:
    """One try at one model."""
    model_name: str
    provider: str
    outcome: str                     # "success", "error", or "skipped"
    error_kind: Optional[str] = None
    message: Optional[str] = None


@dataclass(frozen=True)
class FallbackEvent:
    """Recorded every time a request leaves one model for the next."""
    timestamp: float                 # real time (time.time()), for the dashboard
    tier: str
    from_model: str
    to_model: Optional[str]          # None means there was nothing left to try
    reason: str                      # an error kind, or "circuit_open"


@dataclass
class ReliableResult:
    response: object                 # a Response, or a stream to loop over
    model: ModelConfig               # the model that actually answered
    tier: str
    primary: str                     # the model the tier wanted first
    attempts: list

    @property
    def fell_back(self) -> bool:
        return self.model.name != self.primary


class AllModelsFailed(Exception):
    """Every model in the tier's chain failed or was skipped."""

    def __init__(self, tier: str, attempts: list):
        summary = ", ".join(f"{a.model_name}: {a.error_kind}" for a in attempts)
        super().__init__(f"every model in the '{tier}' tier failed ({summary})")
        self.tier = tier
        self.attempts = attempts


class _PrimedStream:
    """A stream whose first piece was already fetched.

    Fetching the first piece early means errors that happen before any text
    arrives can still fall back to the next model.
    """

    def __init__(self, stream, iterator, first_piece):
        self._stream = stream
        self._iterator = iterator
        self._first_piece = first_piece

    def __iter__(self):
        if self._first_piece is not None:
            yield self._first_piece
        yield from self._iterator

    @property
    def response(self):
        return self._stream.response


# ---------- The client ----------

class ReliableClient:
    """Sends a request to a tier, surviving provider failures.

    Create ONE client and reuse it, so the circuit breaker remembers
    failures across requests.
    """

    def __init__(
        self,
        registry: dict,
        routing: RoutingConfig,
        *,
        send_fn=send_request,
        sleep=time.sleep,
        clock=time.monotonic,
        rng: Optional[random.Random] = None,
    ):
        # send_fn, sleep, clock and rng can be swapped for fakes in tests
        self.registry = registry
        self.routing = routing
        self.breaker = CircuitBreaker(routing.breaker, clock)
        self.fallback_events: list[FallbackEvent] = []
        self._send = send_fn
        self._sleep = sleep
        self._rng = rng or random.Random()

    def send(self, messages: list, tier: str, stream: bool = False, **settings) -> ReliableResult:
        """Try each model in the tier's chain until one answers.

        Raises AllModelsFailed if none of them do.
        """
        if tier not in self.routing.chains:
            raise ValueError(f"unknown tier '{tier}'. Use one of: low, medium, high")
        per_model = MODEL_PARAM_KEYS & set(settings)
        if per_model:
            # A per-call value would also reach the fallback models (e.g. gemini-flash), which reject it
            raise ValueError(f"{', '.join(sorted(per_model))} is set per model: put it under model_params "
                             "in the routing config, not in send()")

        chain = [self.registry[name] for name in self.routing.chains[tier]]
        return self._send_chain(chain, tier, messages, stream, settings)

    def send_model(self, messages: list, model_name: str, stream: bool = False, **settings) -> ReliableResult:
        """Send to ONE named model (a pass-through request): its timeout, retries and circuit breaker apply,
        but there is no fallback to another model. Raises AllModelsFailed if it fails."""
        if model_name not in self.registry:
            raise ValueError(f"unknown model '{model_name}'")
        per_model = MODEL_PARAM_KEYS & set(settings)
        if per_model:
            raise ValueError(f"{', '.join(sorted(per_model))} is set per model: put it under model_params "
                             "in the routing config, not in send_model()")
        model = self.registry[model_name]
        return self._send_chain([model], model.tier, messages, stream, settings)

    def _send_chain(self, chain: list, tier: str, messages: list, stream: bool, settings: dict) -> ReliableResult:
        attempts: list[Attempt] = []

        for index, model in enumerate(chain):
            next_name = chain[index + 1].name if index + 1 < len(chain) else None

            blocked_key = self._blocked_key(model)
            if blocked_key:
                left = self.breaker.seconds_left(blocked_key)
                attempts.append(Attempt(
                    model.name, model.provider, "skipped", "circuit_open",
                    f"{blocked_key} is skipped for {left:.0f}s more",
                ))
                self._record_fallback(tier, model.name, next_name, "circuit_open")
                continue

            result, error = self._try_model(model, messages, stream, settings, attempts)
            if result is not None:
                return ReliableResult(result, model, tier, chain[0].name, attempts)
            self._record_fallback(tier, model.name, next_name, error.kind)

        raise AllModelsFailed(tier, attempts)

    def _try_model(self, model, messages, stream, settings, attempts):
        """Try one model, retrying rate limits and server errors."""
        policy = self.routing.retry
        last_error = None

        for attempt_number in range(1, policy.max_attempts + 1):
            try:
                result = self._call(model, messages, stream, settings)
            except ProviderError as error:
                last_error = error
                attempts.append(Attempt(model.name, model.provider, "error", error.kind, str(error)))
                if error.kind in PROVIDER_BREAKER_KINDS:
                    self.breaker.record_failure(model.provider)
                elif error.kind in MODEL_BREAKER_KINDS:
                    self.breaker.record_failure(_model_key(model))

                can_retry = (
                    error.retryable
                    and attempt_number < policy.max_attempts
                    and self._blocked_key(model) is None
                )
                if not can_retry:
                    break

                delay = backoff_delay(attempt_number, policy, self._rng)
                logger.info(
                    "retrying %s in %.2fs (try %d of %d) after %s",
                    model.name, delay, attempt_number + 1, policy.max_attempts, error.kind,
                )
                self._sleep(delay)
                continue

            # A cache hit says nothing about whether the provider is healthy right now
            if not getattr(result, "cached", False):
                self.breaker.record_success(model.provider)
                self.breaker.record_success(_model_key(model))
            attempts.append(Attempt(model.name, model.provider, "success"))
            return result, None

        return None, last_error

    def _blocked_key(self, model):
        """Which breaker (provider-wide or this model only) is blocking it, if any."""
        for key in (model.provider, _model_key(model)):
            if self.breaker.is_open(key):
                return key
        return None

    def _call(self, model, messages, stream, settings):
        # Every call gets the configured timeout (per model if set) and generation settings, unless the caller
        # set their own, plus the parameters configured for THIS model only (so a fallback never inherits them)
        settings = {"timeout": self.routing.timeout_for(model.name), **self.routing.generation, **settings,
                    **self.routing.model_params.get(model.name, {})}

        if not stream:
            return self._send(messages, model, **settings)

        stream_response = self._send(messages, model, stream=True, **settings)
        iterator = iter(stream_response)
        # Errors before the first piece are raised here, so we can still fall back.
        # Errors after text has started arriving can't be undone, so they go to the caller.
        first_piece = _first_piece(iterator, self.routing.first_token_timeout_for(model.name), model)
        return _PrimedStream(stream_response, iterator, first_piece)

    def _record_fallback(self, tier, from_model, to_model, reason):
        self.fallback_events.append(FallbackEvent(time.time(), tier, from_model, to_model, reason))
        if to_model:
            logger.warning("fallback (%s tier): %s -> %s because of %s", tier, from_model, to_model, reason)
        else:
            logger.error("fallback (%s tier): %s failed (%s) and no models are left", tier, from_model, reason)


def _first_piece(iterator, limit: Optional[float], model: ModelConfig):
    """next(iterator, None), but give up after `limit` seconds with a timeout ProviderError (so the model
    falls back). The waiting is done in a helper thread; a thread that times out can't be killed, so it is
    abandoned and ends when the provider's own read timeout fires."""
    if limit is None:
        return next(iterator, None)
    box, done = {}, threading.Event()

    def pull():
        try:
            box["piece"] = next(iterator, None)
        except BaseException as error:          # handed to the caller below
            box["error"] = error
        finally:
            done.set()

    threading.Thread(target=pull, name=f"first-token-{model.name}", daemon=True).start()
    if not done.wait(limit):
        raise ProviderError(model.provider, "timeout", f"no first token within {limit:g}s")
    if "error" in box:
        raise box["error"]
    return box["piece"]


def create_client(**kwargs) -> ReliableClient:
    """Build a client from models.yaml and routing.yaml."""
    registry = load_registry()
    return ReliableClient(registry, load_routing(registry), **kwargs)