"""Reliability layer: timeouts, retries, fallback chains, and a circuit breaker."""
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from app.providers.base import send_request
from app.providers.common import ProviderError
from app.registry import PROJECT_ROOT, VALID_TIERS, ModelConfig, load_registry

logger = logging.getLogger("triage.reliability")

DEFAULT_ROUTING_PATH = PROJECT_ROOT / "config" / "routing.yaml"

# Errors that mean "this whole provider is unhealthy right now": skip all its models.
PROVIDER_BREAKER_KINDS = {"server", "timeout", "connection"}
# Rate limits are usually per model (Groq limits each model separately),
# so a rate limit only skips the one model that hit it.
MODEL_BREAKER_KINDS = {"rate_limit"}
# Client errors (like a bad request) are usually our fault, so they don't count.


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


@dataclass(frozen=True)
class RoutingConfig:
    chains: dict                     # tier -> list of model names, primary first
    retry: RetryPolicy
    breaker: BreakerSettings
    timeout_seconds: float = 30.0


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
    return RoutingConfig(chains=chains, retry=retry, breaker=breaker, timeout_seconds=timeout)


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

    def is_open(self, key: str) -> bool:
        return self._clock() < self._open_until.get(key, 0.0)

    def seconds_left(self, key: str) -> float:
        return max(0.0, self._open_until.get(key, 0.0) - self._clock())

    def record_success(self, key: str) -> None:
        self._failures[key] = 0
        self._open_until.pop(key, None)

    def record_failure(self, key: str) -> None:
        count = self._failures.get(key, 0) + 1
        self._failures[key] = count
        if count >= self._settings.failure_threshold:
            self._open_until[key] = self._clock() + self._settings.cooldown_seconds
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

        chain = [self.registry[name] for name in self.routing.chains[tier]]
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
        # Every call gets a timeout, unless the caller set their own
        settings = {"timeout": self.routing.timeout_seconds, **settings}

        if not stream:
            return self._send(messages, model, **settings)

        stream_response = self._send(messages, model, stream=True, **settings)
        iterator = iter(stream_response)
        # Errors before the first piece are raised here, so we can still fall back.
        # Errors after text has started arriving can't be undone, so they go to the caller.
        first_piece = next(iterator, None)
        return _PrimedStream(stream_response, iterator, first_piece)

    def _record_fallback(self, tier, from_model, to_model, reason):
        self.fallback_events.append(FallbackEvent(time.time(), tier, from_model, to_model, reason))
        if to_model:
            logger.warning("fallback (%s tier): %s -> %s because of %s", tier, from_model, to_model, reason)
        else:
            logger.error("fallback (%s tier): %s failed (%s) and no models are left", tier, from_model, reason)


def create_client(**kwargs) -> ReliableClient:
    """Build a client from models.yaml and routing.yaml."""
    registry = load_registry()
    return ReliableClient(registry, load_routing(registry), **kwargs)