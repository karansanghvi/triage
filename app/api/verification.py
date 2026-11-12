"""Answer verification in the live API (Phase 4 Step 7): the policy from routing.yaml and sync mode.

Sync mode (X-Triage-Verify: sync, model="auto", non-streaming only) runs AFTER route_and_stream has produced
an answer, so routing itself is untouched. For the tier that answered, the policy lists the checks to run,
cheapest first, and the tier to escalate to when one says "bad":

    low     format                    bad -> medium, which then gets medium's checks
    medium  format, then ai_checker   bad -> high
    high    (nothing)                 never checked

The AI checker is app/verifier/ai_checker.py with config/checker.yaml (v3, groq-small). Its calls go through
the API's ReliableClient (retries, circuit breaker, cache, per-model timeouts from routing.yaml). When it
errors or gives no verdict, on_error decides: "escalate" (treat as bad) or "accept" (treat as good).

Selective checking (off by default): a tier with ai_checker may set
    skip_check_above   skip the AI check when the classifier's pass probability for that tier is at least this
                       (null = never skip; 1.0 also never skips)
    min_answer_tokens  skip it when the answer is shorter than this many visible tokens (0 = never)
A skipped check is recorded as decided_by / X-Triage-Check "skipped_confident" or "skipped_short". The free
format check still runs first, so a badly formatted answer is escalated either way.

agreement_on_comparable is reserved for "agreement on comparable prompts" (Step 3's strategy b). It is not
built: setting it to true is a config error, so nobody thinks it is running.
"""
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import anyio
import yaml

from app.providers.common import ProviderError
from app.reliability import AllModelsFailed
from app.verifier.ai_checker import TRUNCATED_FINISH_REASONS, check_answer_ai
from app.verifier.checker import check_answer

TIERS = ("low", "medium", "high")
CHECKS = ("format", "ai_checker")
ON_ERROR = ("escalate", "accept")

DEFAULT_SECTION = {
    "sync": {
        "tiers": {"low": {"checks": ["format"], "escalate_to": "medium"},
                  # skip_check_above 0.90: chosen on validation in Phase 4 Step 3, the best skip_above at every
                  # high/medium price ratio tried (reports/price_sensitivity_val.json); min_answer_tokens stays off
                  "medium": {"checks": ["format", "ai_checker"], "escalate_to": "high",
                             "skip_check_above": 0.90, "min_answer_tokens": 0},
                  "high": {"checks": []}},
        "on_error": "escalate",
        "agreement_on_comparable": False,
    },
    "background": {"sample_rate": 0.1, "daily_max_calls": 50, "daily_max_tokens": 50_000},
}


class VerificationConfigError(ValueError):
    pass


# ---------- policy ----------

@dataclass(frozen=True)
class TierPolicy:
    checks: tuple = ()                    # run in this order; the first "bad" stops the rest
    escalate_to: Optional[str] = None     # None: this tier's answer is final
    skip_check_above: Optional[float] = None   # skip the AI check when p(tier passes) >= this (None = off)
    min_answer_tokens: int = 0                  # skip it when the answer is shorter than this (0 = off)


@dataclass(frozen=True)
class SyncPolicy:
    tiers: dict                           # tier -> TierPolicy
    on_error: str = "escalate"
    agreement_on_comparable: bool = False


@dataclass(frozen=True)
class BackgroundPolicy:
    sample_rate: float = 0.1              # share of eligible auto requests that get a background label
    daily_max_calls: int = 50             # strong-model answers + judge calls per UTC day
    daily_max_tokens: int = 50_000


@dataclass(frozen=True)
class VerificationPolicy:
    sync: SyncPolicy
    background: BackgroundPolicy


def _number(value, where, *, whole=False, low=0.0, high=None):
    kinds = (int,) if whole else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds) or value < low or (high is not None and value > high):
        span = f"from {low} to {high}" if high is not None else f"of at least {low}"
        raise VerificationConfigError(f"routing.yaml: {where} must be a {'whole ' if whole else ''}number {span}, "
                                      f"got {value!r}")
    return value


def verification_from_dict(raw: dict) -> VerificationPolicy:
    """Validate routing.yaml's `verification:` section (missing: the defaults above)."""
    section = (raw or {}).get("verification")
    section = DEFAULT_SECTION if section is None else section
    if not isinstance(section, dict) or set(section) - {"sync", "background"}:
        raise VerificationConfigError("routing.yaml: verification must have only `sync` and `background`")
    sync = section.get("sync", DEFAULT_SECTION["sync"])
    if not isinstance(sync, dict) or set(sync) - {"tiers", "on_error", "agreement_on_comparable"}:
        raise VerificationConfigError("routing.yaml: verification.sync takes tiers, on_error, agreement_on_comparable")
    raw_tiers = sync.get("tiers", DEFAULT_SECTION["sync"]["tiers"])
    if not isinstance(raw_tiers, dict) or set(raw_tiers) != set(TIERS):
        raise VerificationConfigError("routing.yaml: verification.sync.tiers must list low, medium and high")
    tiers = {}
    for tier in TIERS:
        t = raw_tiers[tier] or {}
        if not isinstance(t, dict) or set(t) - {"checks", "escalate_to", "skip_check_above", "min_answer_tokens"}:
            raise VerificationConfigError(f"routing.yaml: verification.sync.tiers.{tier} takes checks, escalate_to, "
                                          "skip_check_above, min_answer_tokens")
        checks = t.get("checks") or []
        if not isinstance(checks, list) or set(checks) - set(CHECKS) or len(set(checks)) != len(checks):
            raise VerificationConfigError(f"routing.yaml: verification.sync.tiers.{tier}.checks must be a list "
                                          f"from {', '.join(CHECKS)}, each at most once")
        to = t.get("escalate_to")
        if to is not None and (to not in TIERS or TIERS.index(to) <= TIERS.index(tier)):
            raise VerificationConfigError(f"routing.yaml: verification.sync.tiers.{tier}.escalate_to must be a "
                                          f"higher tier, got {to!r}")
        if checks and to is None:
            raise VerificationConfigError(f"routing.yaml: verification.sync.tiers.{tier} runs checks but has no "
                                          "escalate_to: a bad verdict would have nowhere to go")
        skip_above, min_tokens = t.get("skip_check_above"), t.get("min_answer_tokens", 0)
        if (skip_above is not None or min_tokens) and "ai_checker" not in checks:
            raise VerificationConfigError(f"routing.yaml: verification.sync.tiers.{tier} sets skip_check_above or "
                                          "min_answer_tokens but runs no ai_checker to skip")
        if skip_above is not None:
            skip_above = float(_number(skip_above, f"verification.sync.tiers.{tier}.skip_check_above", high=1.0))
            if skip_above <= 0:
                raise VerificationConfigError(f"routing.yaml: verification.sync.tiers.{tier}.skip_check_above must "
                                              "be above 0 (0 would skip every check); use null to turn it off")
        min_tokens = _number(min_tokens if min_tokens is not None else 0,
                             f"verification.sync.tiers.{tier}.min_answer_tokens", whole=True)
        tiers[tier] = TierPolicy(tuple(checks), to, skip_above, min_tokens)
    on_error = sync.get("on_error", "escalate")
    if on_error not in ON_ERROR:
        raise VerificationConfigError(f"routing.yaml: verification.sync.on_error must be escalate or accept, "
                                      f"got {on_error!r}")
    agreement = sync.get("agreement_on_comparable", False)
    if agreement is not False:
        raise VerificationConfigError("routing.yaml: verification.sync.agreement_on_comparable is reserved and not "
                                      "built yet; it must be false")
    bg = section.get("background", DEFAULT_SECTION["background"])
    if not isinstance(bg, dict) or set(bg) - {"sample_rate", "daily_max_calls", "daily_max_tokens"}:
        raise VerificationConfigError("routing.yaml: verification.background takes sample_rate, daily_max_calls, "
                                      "daily_max_tokens")
    background = BackgroundPolicy(
        sample_rate=float(_number(bg.get("sample_rate", 0.1), "verification.background.sample_rate", high=1.0)),
        daily_max_calls=_number(bg.get("daily_max_calls", 50), "verification.background.daily_max_calls", whole=True),
        daily_max_tokens=_number(bg.get("daily_max_tokens", 50_000), "verification.background.daily_max_tokens",
                                 whole=True))
    return VerificationPolicy(SyncPolicy(tiers, on_error, False), background)


COLLECTION_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "collection.yaml"
MAX_BUDGET_SHARE = 0.5


def daily_token_budget(model: str, path: Path = COLLECTION_CONFIG_PATH) -> Optional[int]:
    """The model's tokens-per-day limit from collection.yaml's `limits:`, or None if none is listed."""
    limits = (yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}).get("limits") or {}
    tpd = (limits.get(model) or {}).get("tpd")
    return int(tpd) if tpd is not None else None


def check_background_budget(policy: BackgroundPolicy, models, path: Path = COLLECTION_CONFIG_PATH) -> None:
    """Startup check: the background token cap may use at most half of the daily budget of every model the
    background job calls (the strong model and the judge model). Those budgets are shared with live traffic."""
    for model in sorted(set(models)):
        tpd = daily_token_budget(model, path)
        if tpd is not None and policy.daily_max_tokens > tpd * MAX_BUDGET_SHARE:
            raise VerificationConfigError(
                f"routing.yaml: verification.background.daily_max_tokens is {policy.daily_max_tokens:,}, more than "
                f"half of {model}'s daily budget ({tpd:,} tokens/day in config/collection.yaml). That budget is "
                f"shared with live traffic; use at most {int(tpd * MAX_BUDGET_SHARE):,}.")


# ---------- calling models through the API's ReliableClient ----------

class ClientCheckSend:
    """The AI checker's send(messages, model_name, settings), through the reliable client: one model, its
    retries, breaker, cache and routing.yaml timeout. Failing models become a ProviderError, which
    check_answer_ai turns into an error result. Also adds up what was actually paid."""

    def __init__(self, client):
        self.client = client
        self.cost = 0.0

    def __call__(self, messages, model_name, settings):
        response_format = {"type": settings.response_format} if settings.response_format else None
        try:
            result = self.client.send_model(messages, model_name, stream=False, temperature=settings.temperature,
                                            max_tokens=settings.max_tokens, response_format=response_format)
        except AllModelsFailed as error:
            last = next((a for a in reversed(error.attempts) if a.error_kind), None)
            provider = self.client.registry[model_name].provider
            raise ProviderError(provider, last.error_kind if last else "server", str(error)) from None
        self.cost += result.response.cost
        return result.response


# ---------- sync verification ----------

@dataclass
class Served:
    tier: str
    model: str
    response: object                      # providers.base.Response
    text: str


@dataclass
class SyncOutcome:
    final: Served
    escalated: bool = False
    escalated_from: Optional[str] = None
    escalated_to: Optional[str] = None
    decided_by: str = "none"              # format | ai_checker | ai_checker_error | skipped_confident | skipped_short | none
    checker_verdict: Optional[str] = None # the deciding check's verdict: good | bad | error
    checker_version: Optional[int] = None # set when the AI checker ran
    check_cost: float = 0.0               # AI checks, price_as
    check_actual_cost: float = 0.0
    check_latency_ms: float = 0.0
    escalation_cost: float = 0.0          # every extra answer, price_as
    escalation_actual_cost: float = 0.0
    escalation_latency_ms: float = 0.0
    escalation_error: Optional[str] = None  # an escalation that found no model: the last answer is served
    steps: list = field(default_factory=list)   # [(tier, check, verdict)], for the reason header


def _truncated(response) -> bool:
    return (getattr(response, "finish_reason", None) or "").lower() in TRUNCATED_FINISH_REASONS


def visible_tokens(response) -> Optional[int]:
    """The answer's own tokens: output tokens minus hidden reasoning tokens (None if unknown)."""
    out = getattr(response, "output_tokens", None)
    if out is None:
        return None
    return max(0, int(out) - int(getattr(response, "reasoning_tokens", None) or 0))


def check_skip(tier_policy: TierPolicy, p_pass: Optional[float], response) -> Optional[str]:
    """"skipped_confident", "skipped_short" or None (run the AI check). Same rule as the Step 3 simulation."""
    above = tier_policy.skip_check_above
    if above is not None and above < 1.0 and p_pass is not None and p_pass >= above:
        return "skipped_confident"
    tokens = visible_tokens(response)
    if tier_policy.min_answer_tokens and tokens is not None and tokens < tier_policy.min_answer_tokens:
        return "skipped_short"
    return None


async def verify_sync(messages: list, first: Served, policy: SyncPolicy, deps, settings,
                      probabilities: Optional[dict] = None) -> SyncOutcome:
    """Check `first` (the routed answer) by its tier's policy; escalate while a check says bad.
    probabilities {tier: the classifier's pass probability}, for skip_check_above."""
    probabilities = probabilities or {}
    out = SyncOutcome(final=first)
    current = first
    while True:
        tier_policy = policy.tiers[current.tier]
        bad = False
        for check in tier_policy.checks:
            if check == "format":
                ok = check_answer(messages, current.text, truncated=_truncated(current.response)).ok
                out.decided_by, out.checker_verdict, bad = "format", "good" if ok else "bad", not ok
            else:
                skipped = check_skip(tier_policy, probabilities.get(current.tier), current.response)
                if skipped:                       # accept without the paid check
                    out.decided_by, out.checker_verdict = skipped, None
                    out.steps.append((current.tier, check, skipped))
                    continue
                send = ClientCheckSend(deps.client)
                started = time.perf_counter()
                result = await anyio.to_thread.run_sync(partial(
                    check_answer_ai, messages, current.text, model=deps.checker_settings.model_name,
                    send=send, settings=deps.checker_settings))
                out.check_latency_ms += (time.perf_counter() - started) * 1000
                out.check_cost += result.reported_cost
                out.check_actual_cost += send.cost
                out.checker_version = result.checker_version
                if result.status == "ok":
                    out.decided_by, out.checker_verdict, bad = "ai_checker", "good" if result.good else "bad", not result.good
                else:
                    out.decided_by, out.checker_verdict = "ai_checker_error", "error"
                    bad = policy.on_error == "escalate"
            out.steps.append((current.tier, check, out.checker_verdict))
            if bad:
                break
        if not bad or tier_policy.escalate_to is None:
            break
        target = tier_policy.escalate_to
        started = time.perf_counter()
        try:
            result = await anyio.to_thread.run_sync(partial(deps.client.send, messages, target, stream=False,
                                                            **settings.as_kwargs()))
        except AllModelsFailed:
            out.escalation_latency_ms += (time.perf_counter() - started) * 1000
            out.escalation_error = f"all_models_failed:{target}"
            break
        out.escalation_latency_ms += (time.perf_counter() - started) * 1000
        out.escalation_cost += result.response.reported_cost
        out.escalation_actual_cost += result.response.cost
        out.escalated, out.escalated_to = True, target
        out.escalated_from = out.escalated_from or first.tier
        current = Served(target, result.model.name, result.response, result.response.text)
        out.final = current
    return out
