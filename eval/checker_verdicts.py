"""AI-checker verdicts as an input to the routing simulators (Phase 4 Step 3). No API calls.

load_verdicts() reads a CheckLog file (eval/check_log.py) and returns one Verdict for EVERY requested
(prompt_id, tier), so no prompt silently drops out of a strategy:

    status "good" / "bad"   a finished ok record from the right checker model, version and prompt hash
    status "unknown"        anything else, with the reason in `why`:
                              "error:<kind>"  the check ran and failed (truncated, invalid_json, provider:...)
                              "missing"       no usable record (never checked, another model/version/prompt
                                              hash, or made from a different answer: input_sha256 mismatch)

What a router does with "unknown" is the strategy's on_error choice, not decided here.

Cost and latency: whenever a strategy runs a check, scoring charges Verdict.cost (dollars at the checker
model's price_as) and adds Verdict.latency_ms. Logged records carry their real figures (errors included:
a failed check is still paid for). A missing record gets an ESTIMATE (estimated=True): input tokens from
the checker messages' length, output tokens and latency from the median of this log's ok records for the
model (defaults below if there are none), priced with config/models.yaml.
"""
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.registry import load_registry
from app.verifier.ai_checker import build_checker_messages
from eval.check_log import input_sha256, is_finished

# Fitted on the Step 1 smoke calls: input tokens ~= 0.2502 x characters of the checker messages + 82.
TOKENS_PER_CHAR, TOKEN_OVERHEAD = 0.2502, 82
DEFAULT_OUTPUT_TOKENS = 200        # smoke calls used at most 220
DEFAULT_LATENCY_MS = 500.0         # smoke calls took about 0.4-0.6 s
STATUSES = ("good", "bad", "unknown")


@dataclass(frozen=True)
class Verdict:
    status: str                    # "good" | "bad" | "unknown"
    why: str = "ok"                # "ok", "error:<kind>" or "missing"
    cost: float = 0.0              # dollars for running this check (price_as)
    latency_ms: float = 0.0
    estimated: bool = False        # cost / latency are estimates (no logged call)


def read_records(path: Path) -> list:
    """The log's records, oldest first; [] if there is no file yet. Never creates anything."""
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


@dataclass(frozen=True)
class CheckEstimate:
    """Cost and latency of one check that hasn't been logged."""
    input_price: float             # $ per 1M tokens (price_as)
    output_price: float
    output_tokens: float = DEFAULT_OUTPUT_TOKENS
    latency_ms: float = DEFAULT_LATENCY_MS
    input_tokens: float = 600.0    # used only when the checker input is not known

    def cost(self, input_tokens: Optional[float] = None) -> float:
        tokens = self.input_tokens if input_tokens is None else input_tokens
        return (tokens * self.input_price + self.output_tokens * self.output_price) / 1_000_000

    @staticmethod
    def input_tokens_for(messages: list, answer: str) -> float:
        chars = sum(len(m["content"]) for m in build_checker_messages(messages, answer))
        return TOKENS_PER_CHAR * chars + TOKEN_OVERHEAD


def make_estimate(model: str, records: list = (), registry: Optional[dict] = None) -> CheckEstimate:
    """Price from models.yaml; output tokens and latency from the median of the model's ok records."""
    registry = registry or load_registry()
    price = registry[model].price_as
    if price is None:
        raise ValueError(f"{model} has no price_as in config/models.yaml; can't price its checks")
    ok = [r for r in records if r.get("checker_model") == model and r.get("status") == "ok" and r.get("api_calls")]
    out_tokens = [r["output_tokens"] for r in ok if r.get("output_tokens")]
    latencies = [r["latency_ms"] for r in ok if r.get("latency_ms")]
    in_tokens = [r["input_tokens"] for r in ok if r.get("input_tokens")]
    return CheckEstimate(price.input_price, price.output_price,
                         output_tokens=statistics.median(out_tokens) if out_tokens else DEFAULT_OUTPUT_TOKENS,
                         latency_ms=float(statistics.median(latencies)) if latencies else DEFAULT_LATENCY_MS,
                         input_tokens=statistics.median(in_tokens) if in_tokens else 600.0)


def load_verdicts(path: Path, keys, *, model: str, version: int, prompt_hash: str,
                  inputs: Optional[dict] = None, estimate: Optional[CheckEstimate] = None) -> dict:
    """{(prompt_id, tier): Verdict} for every key in `keys`, from the CheckLog file at `path`.

    Only records with this checker_model, checker_version and prompt_sha256 count; the newest one per
    (prompt_id, tier) wins. inputs = {(prompt_id, tier): (messages, answer)}: when given, a record made from
    a different checker input (input_sha256) is treated as missing, and missing checks are estimated from
    the input's length.
    """
    records = read_records(path)
    estimate = estimate or make_estimate(model, records)
    latest = {}
    for r in records:
        if (r.get("checker_model"), r.get("checker_version"), r.get("prompt_sha256")) == (model, version, prompt_hash):
            latest[(r["prompt_id"], r["tier"])] = r
    out = {}
    for key in keys:
        key = tuple(key)
        r = latest.get(key)
        if r is not None and inputs is not None and key in inputs \
                and r.get("input_sha256") != input_sha256(*inputs[key]):
            r = None                                  # made from another answer: stale
        if r is None:
            tokens = CheckEstimate.input_tokens_for(*inputs[key]) if inputs and key in inputs else None
            out[key] = Verdict("unknown", "missing", estimate.cost(tokens), estimate.latency_ms, estimated=True)
            continue
        cost, latency = float(r.get("reported_cost") or 0.0), float(r.get("latency_ms") or 0.0)
        if r["status"] == "ok" and r.get("good") is not None:
            out[key] = Verdict("good" if r["good"] else "bad", "ok", cost, latency)
        else:
            kind = r.get("error_kind") or "unknown"
            # A provider error is not finished: a rerun would retry it. Either way the router sees "unknown".
            out[key] = Verdict("unknown", f"error:{kind}" + ("" if is_finished(r) else " (unfinished)"), cost, latency)
    return out


def summarize_verdicts(verdicts: dict) -> dict:
    """Counts by status and reason, and how many costs are estimates."""
    by_status = {s: 0 for s in STATUSES}
    by_why = {}
    for v in verdicts.values():
        by_status[v.status] += 1
        by_why[v.why] = by_why.get(v.why, 0) + 1
    return {"n": len(verdicts), **by_status, "by_reason": dict(sorted(by_why.items())),
            "estimated_cost": sum(v.estimated for v in verdicts.values())}
