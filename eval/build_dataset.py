"""Phase 2 data collection: send every prompt to every tier and save the answers.

Usage:
    python -m eval.build_dataset --pilot     # 30-prompt pilot -> data/pilot/
    python -m eval.build_dataset --pilot2    # pilot 1's prompts + 30 auto-gradable own prompts -> data/pilot2/
    python -m eval.build_dataset             # every prompt in data/prompts.jsonl -> data/collection/

One fixed model per tier (config/collection.yaml), with fallback off, so every label
really describes that tier. Requests go through the Phase 1 cache and retry logic and are
paced to stay under each model's free-tier limits.

Answers are appended to answers.jsonl, keyed by (prompt_id, tier); the newest record for a
key wins. Rerunning skips pairs that already have a successful, non-truncated answer, so a
crash never repeats finished work, and failed pairs are retried.
"""
import argparse
import json
import os
import random
import re
import statistics
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import yaml

from app.cache import TRUNCATED_FINISH_REASONS, CachedSender, ResponseCache
from app.registry import PROJECT_ROOT, load_registry
from app.reliability import AllModelsFailed, BreakerSettings, ReliableClient, RetryPolicy, RoutingConfig
from app.verifier.scorers import score_record

COLLECTION_CONFIG_PATH = PROJECT_ROOT / "config" / "collection.yaml"
PROMPTS_PATH = PROJECT_ROOT / "data" / "prompts.jsonl"
FULL_DIR = PROJECT_ROOT / "data" / "collection"
PILOT_DIR = PROJECT_ROOT / "data" / "pilot"
PILOT2_DIR = PROJECT_ROOT / "data" / "pilot2"
API_KEY_ENV_NAMES = ["GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
PACIFIC = ZoneInfo("America/Los_Angeles")

# Pilot: 30 prompts, graded ones first so a daily cap never starves the auto-graded set
PILOT_SEED = 42
PILOT_GSM8K = 10
PILOT_MMLU = 10
PILOT_ARENA = 4
OWN_TASK_TYPES = ["extraction", "classification", "summarization", "qa", "reasoning", "creative"]

# Pilot 2 adds own prompts the scorers can grade: {task_type: (easy, medium, hard)}
PILOT2_OWN = {"extraction": (2, 3, 3), "classification": (2, 3, 3), "qa": (2, 2, 3), "reasoning": (2, 2, 3)}

# A 429 whose message mentions a daily quota: waiting a minute won't help, so stop that model for the run
DAILY_QUOTA_HINT = re.compile(r"per ?day|perday|daily|requests_per_day|RPD", re.IGNORECASE)


# ---------- Config ----------

@dataclass(frozen=True)
class Limits:
    rpm: Optional[int] = None
    rpd: Optional[int] = None
    tpm: Optional[int] = None
    tpd: Optional[int] = None
    daily_reset: str = "rolling_24h"     # or "midnight_pacific"


@dataclass(frozen=True)
class CollectionConfig:
    tiers: dict                # tier -> model name
    label_tiers: list
    model_params: dict         # model name -> extra request parameters (e.g. reasoning_effort)
    temperature: float
    max_tokens: int
    timeout_seconds: float
    retry: RetryPolicy
    limits: dict               # model name -> Limits


def load_collection_config(registry: dict, path=COLLECTION_CONFIG_PATH) -> CollectionConfig:
    data = yaml.safe_load(Path(path).read_text()) or {}
    tiers = data.get("tiers") or {}
    if not tiers:
        raise ValueError("collection.yaml needs a 'tiers' section")
    blocked = set(data.get("blocked_models") or [])
    for tier, name in tiers.items():
        if name not in registry:
            raise ValueError(f"collection.yaml: tier '{tier}' uses unknown model '{name}'")
        if name in blocked:
            raise ValueError(f"collection.yaml: model '{name}' is blocked from data collection")
    if len(set(tiers.values())) != len(tiers):
        raise ValueError("collection.yaml: each tier needs its own model")
    label_tiers = data.get("label_tiers") or []
    if not set(label_tiers) <= set(tiers):
        raise ValueError("collection.yaml: label_tiers must be listed under tiers")

    limits = {}
    for name in tiers.values():
        raw = (data.get("limits") or {}).get(name) or {}
        reset = raw.get("daily_reset", "rolling_24h")
        if reset not in ("rolling_24h", "midnight_pacific"):
            raise ValueError(f"collection.yaml: {name} daily_reset must be rolling_24h or midnight_pacific")
        limits[name] = Limits(raw.get("rpm"), raw.get("rpd"), raw.get("tpm"), raw.get("tpd"), reset)

    model_params = data.get("model_params") or {}
    for name, params in model_params.items():
        if name not in tiers.values():
            raise ValueError(f"collection.yaml: model_params lists '{name}', which no tier uses")
        if set(params) - {"reasoning_effort"}:
            raise ValueError(f"collection.yaml: unsupported model_params for '{name}': {sorted(set(params) - {'reasoning_effort'})}")

    generation = data.get("generation") or {}
    raw_retry = data.get("retry") or {}
    return CollectionConfig(
        tiers=dict(tiers),
        label_tiers=list(label_tiers),
        model_params={name: dict(params) for name, params in model_params.items()},
        temperature=float(generation.get("temperature", 0)),
        max_tokens=int(generation.get("max_tokens", 2000)),
        timeout_seconds=float(data.get("timeout_seconds", 180)),
        retry=RetryPolicy(
            max_attempts=int(raw_retry.get("max_attempts", 3)),
            base_delay_seconds=float(raw_retry.get("base_delay_seconds", 2)),
            max_delay_seconds=float(raw_retry.get("max_delay_seconds", 30)),
        ),
        limits=limits,
    )


# ---------- Pacing ----------

def day_start(now: datetime, reset: str) -> datetime:
    """Start of the current daily-quota window."""
    if reset == "midnight_pacific":
        local = now.astimezone(PACIFIC)
        return local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(timezone.utc)
    return now - timedelta(hours=24)


class Pacer:
    """Keeps one model under its per-minute and per-day request and token limits."""

    def __init__(self, limits: Limits, history: list):
        self.limits = limits
        self._minute = deque()          # (monotonic time, tokens) for calls in the last 60s
        self._day = list(history)       # (UTC datetime, api calls, tokens) already used, from earlier runs
        self._lock = threading.Lock()

    def _day_usage(self):
        start = day_start(datetime.now(timezone.utc), self.limits.daily_reset)
        recent = [(calls, tokens) for when, calls, tokens in self._day if when >= start]
        return sum(c for c, _ in recent), sum(t for _, t in recent)

    def daily_budget_left(self, estimate: int) -> bool:
        calls, tokens = self._day_usage()
        if self.limits.rpd is not None and calls + 1 > self.limits.rpd:
            return False
        if self.limits.tpd is not None and tokens + estimate > self.limits.tpd:
            return False
        return True

    def wait_for_minute_slot(self, estimate: int) -> float:
        """Sleep until one more call of about `estimate` tokens fits in the last minute. Returns seconds waited."""
        waited = 0.0
        while True:
            with self._lock:
                now = time.monotonic()
                while self._minute and now - self._minute[0][0] >= 60:
                    self._minute.popleft()
                calls = len(self._minute)
                tokens = sum(t for _, t in self._minute)
                over_rpm = self.limits.rpm is not None and calls + 1 > self.limits.rpm
                over_tpm = self.limits.tpm is not None and self._minute and tokens + estimate > self.limits.tpm
                if not (over_rpm or over_tpm):
                    return waited
                pause = max(0.5, 60 - (now - self._minute[0][0]))
            time.sleep(pause)
            waited += pause

    def record(self, api_calls: int, tokens: int) -> None:
        with self._lock:
            if api_calls:
                self._minute.append((time.monotonic(), tokens))
            self._day.append((datetime.now(timezone.utc), api_calls, tokens))


# ---------- Output ----------

class AnswerLog:
    """Append-only answers.jsonl. The newest record for a (prompt_id, tier) key wins."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.records = []
        if path.exists():
            self.records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def latest(self) -> dict:
        latest = {}
        for record in self.records:
            latest[(record["prompt_id"], record["tier"])] = record
        return latest

    def done(self) -> set:
        return {key for key, r in self.latest().items() if r["error"] is None and not r["truncated"]}

    def usage_history(self, model_name: str) -> list:
        """(time, api calls, tokens) for every earlier record of this model, for daily limits."""
        history = []
        for r in self.records:
            if r["model_name"] == model_name and r.get("api_calls"):
                tokens = (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)
                history.append((datetime.fromisoformat(r["timestamp"]), r["api_calls"], tokens))
        return history

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self.records.append(record)


def redact(text: str) -> str:
    """Remove any API key value that might appear in a provider's error message."""
    for name in API_KEY_ENV_NAMES:
        value = os.getenv(name)
        if value and len(value) >= 8:
            text = text.replace(value, f"<{name}>")
    return text


# ---------- Collection ----------

def estimate_tokens(messages: list, max_tokens: int) -> int:
    """Worst-case tokens for pacing: rough input size plus the full output allowance."""
    return sum(len(m["content"]) for m in messages) // 4 + max_tokens


def real_calls(attempts: list) -> int:
    """Provider calls that probably counted against quota (429s and skips don't)."""
    return sum(1 for a in attempts if a.outcome == "success" or a.error_kind not in ("rate_limit", "circuit_open"))


class Collector:
    def __init__(self, config: CollectionConfig, registry: dict, log: AnswerLog, run: str):
        self.config = config
        self.registry = registry
        self.log = log
        self.run = run
        self.sender = CachedSender(ResponseCache())
        routing = RoutingConfig(
            chains={tier: [name] for tier, name in config.tiers.items()},   # one model per tier: no fallback
            retry=config.retry,
            breaker=BreakerSettings(failure_threshold=3, cooldown_seconds=60),
            timeout_seconds=config.timeout_seconds,
        )
        self.client = ReliableClient(registry, routing, send_fn=self.sender)
        self.pacers = {name: Pacer(config.limits[name], log.usage_history(name)) for name in config.tiers.values()}
        self.counts = Counter()
        self._print_lock = threading.Lock()

    def say(self, message: str) -> None:
        with self._print_lock:
            print(message, flush=True)

    def run_tier(self, tier: str, prompts: list) -> None:
        model = self.registry[self.config.tiers[tier]]
        pacer = self.pacers[model.name]
        done = self.log.done()
        todo = [p for p in prompts if (p["id"], tier) not in done]
        self.counts[f"{tier}: skipped (already done)"] += len(prompts) - len(todo)
        for position, prompt in enumerate(todo, start=1):
            estimate = estimate_tokens(prompt["messages"], self.config.max_tokens)
            if not pacer.daily_budget_left(estimate):
                self.say(f"[{tier}] daily limit for {model.name} reached; {len(todo) - position + 1} pairs left for later")
                self.counts[f"{tier}: deferred (daily limit)"] += len(todo) - position + 1
                return
            pacer.wait_for_minute_slot(estimate)
            record, stop = self.call(tier, model, prompt)
            self.log.append(record)
            pacer.record(record["api_calls"], (record["input_tokens"] or 0) + (record["output_tokens"] or 0))
            status = "error " + record["error"]["kind"] if record["error"] else ("TRUNCATED" if record["truncated"] else "ok")
            self.counts[f"{tier}: {status.split()[0]}"] += 1
            self.say(f"[{tier}] {position}/{len(todo)} {prompt['id']}: {status} "
                     f"({record['output_tokens'] or 0} out tok, {record['latency_ms'] or 0:.0f} ms"
                     f"{', cache' if record['cache_hit'] else ''})")
            if stop:
                self.say(f"[{tier}] {model.name} hit its daily quota; stopping this tier for the run")
                self.counts[f"{tier}: deferred (daily limit)"] += len(todo) - position
                return

    def call(self, tier: str, model, prompt: dict) -> tuple:
        """Send one prompt to one tier. Returns (record, stop_tier)."""
        params = self.config.model_params.get(model.name, {})
        base = {
            "prompt_id": prompt["id"], "tier": tier, "model_name": model.name, "model_id": model.model_id,
            "provider": model.provider, "run": self.run,
            "temperature": self.config.temperature, "max_tokens": self.config.max_tokens,
            "reasoning_effort": params.get("reasoning_effort"),
        }
        stop = False
        try:
            result = self.client.send(prompt["messages"], tier, temperature=self.config.temperature,
                                      max_tokens=self.config.max_tokens, **params)
        except AllModelsFailed as failure:
            last = next((a for a in reversed(failure.attempts) if a.outcome != "success"), None)
            kind = last.error_kind if last else "unknown"
            message = redact(last.message or "") if last else ""
            stop = kind == "rate_limit" and bool(DAILY_QUOTA_HINT.search(message))
            if kind == "rate_limit" and not stop:
                time.sleep(60)   # still limited after retries: give the minute window time to clear
            if kind == "circuit_open":
                time.sleep(self.client.breaker.seconds_left(f"model:{model.name}") or 60)
            return {**base, "answer": None, "reasoning": None, "finish_reason": None, "truncated": False,
                    "input_tokens": None, "output_tokens": None, "reasoning_tokens": None,
                    "tokens_estimated": None, "latency_ms": None, "cost": 0.0, "price_as_cost": 0.0,
                    "cache_hit": False, "timestamp": datetime.now(timezone.utc).isoformat(),
                    "attempts": len(failure.attempts), "api_calls": real_calls(failure.attempts),
                    "error": {"kind": kind, "message": message[:500]}}, stop

        response = result.response
        truncated = (response.finish_reason or "").lower() in TRUNCATED_FINISH_REASONS
        return {**base, "model_id": response.model_id, "answer": response.text, "reasoning": response.reasoning,
                "finish_reason": response.finish_reason, "truncated": truncated,
                "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
                "reasoning_tokens": response.reasoning_tokens, "tokens_estimated": response.tokens_estimated,
                "latency_ms": response.latency_ms, "cost": response.cost,
                # on a cache hit, reported_cost is 0 and the real price_as cost is in cache_saved_cost
                "price_as_cost": response.cache_saved_cost if response.cached else response.reported_cost,
                "cache_hit": response.cached, "timestamp": datetime.now(timezone.utc).isoformat(),
                "attempts": len(result.attempts), "api_calls": 0 if response.cached else real_calls(result.attempts),
                "error": None}, stop

    def run_all(self, prompts: list) -> None:
        threads = [threading.Thread(target=self.run_tier, args=(tier, prompts), name=tier)
                   for tier in self.config.tiers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


# ---------- Prompts ----------

def load_prompts(path=PROMPTS_PATH) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def pilot_prompts(prompts: list, seed: int = PILOT_SEED) -> list:
    rng = random.Random(seed)
    by_source = lambda source: [p for p in prompts if p["source"] == source]
    own = by_source("own")
    picked = rng.sample(by_source("gsm8k"), PILOT_GSM8K) + rng.sample(by_source("mmlu"), PILOT_MMLU)
    picked += [rng.choice([p for p in own if p["task_type"] == t]) for t in OWN_TASK_TYPES]
    picked += rng.sample(by_source("arena"), PILOT_ARENA)
    return picked


def pilot2_prompts(prompts: list, seed: int = PILOT_SEED) -> list:
    """Pilot 1's prompts, then own prompts with automatic scorers, spread over difficulty."""
    first = pilot_prompts(prompts, seed)
    used = {p["id"] for p in first}
    rng = random.Random(seed)
    extra = []
    for task_type, counts in PILOT2_OWN.items():
        for difficulty, count in zip(["easy", "medium", "hard"], counts):
            pool = [p for p in prompts if p["source"] == "own" and p["task_type"] == task_type
                    and p["difficulty_guess"] == difficulty and p["scoring"] != "judge" and p["id"] not in used]
            extra += rng.sample(pool, count)
    return first + extra


# ---------- Review ----------

def grade(prompt: dict, record: Optional[dict]) -> Optional[bool]:
    """Auto-grade with the prompt's scorer; None for judge prompts or when there's no answer."""
    if prompt["scoring"] == "judge" or not record or record["error"] or record["answer"] is None:
        return None
    return score_record(prompt, record["answer"]).passed


def write_review(prompts: list, log: AnswerLog, tiers: list, path: Path) -> None:
    latest = log.latest()
    out = [f"Pilot review: {len(prompts)} prompts x {len(tiers)} tiers ({', '.join(tiers)})\n"]
    for number, prompt in enumerate(prompts, start=1):
        user = next(m["content"] for m in reversed(prompt["messages"]) if m["role"] == "user")
        out += ["=" * 110,
                f"#{number:02d} {prompt['id']}  [{prompt['source']} / {prompt['task_type']} / {prompt['scoring']}]"
                + (f"  reference: {prompt['reference']}" if isinstance(prompt["reference"], str) else ""),
                "-" * 110, user.strip(), ""]
        for tier in tiers:
            record = latest.get((prompt["id"], tier))
            passed = grade(prompt, record)
            mark = {True: "PASS", False: "FAIL", None: "-"}[passed]
            if record is None:
                out += [f"--- {tier}: (not run yet)", ""]
                continue
            if record["error"]:
                out += [f"--- {tier} ({record['model_id']}): ERROR {record['error']['kind']}: {record['error']['message'][:200]}", ""]
                continue
            flags = " TRUNCATED" if record["truncated"] else ""
            if passed is False:
                flags += f"  ({score_record(prompt, record['answer']).reason[:150]})"
            out += [f"--- {tier} ({record['model_id']})  grade: {mark}{flags}  "
                    f"[{record['output_tokens']} out tok, {record['latency_ms']:.0f} ms, finish={record['finish_reason']}]",
                    (record["answer"] or "(empty answer)").strip(), ""]
    path.write_text("\n".join(out), encoding="utf-8")


def summarize(prompts: list, log: AnswerLog, tiers: list) -> None:
    latest = log.latest()
    graded = [p for p in prompts if p["scoring"] != "judge"]
    print(f"\n{'tier':<8}{'graded pass':>14}{'ok':>5}{'trunc':>7}{'errors':>8}{'rate-lim':>10}"
          f"{'med ms':>9}{'max ms':>9}{'avg out':>9}")
    for tier in tiers:
        records = [latest.get((p["id"], tier)) for p in prompts]
        records = [r for r in records if r]
        grades = [grade(p, latest.get((p["id"], tier))) for p in graded]
        passed = sum(g is True for g in grades)
        answered = [r for r in records if not r["error"]]
        latencies = [r["latency_ms"] for r in answered if not r["cache_hit"]]
        print(f"{tier:<8}{f'{passed}/{len(graded)}':>14}{len(answered):>5}"
              f"{sum(r['truncated'] for r in answered):>7}{sum(bool(r['error']) for r in records):>8}"
              f"{sum(1 for r in records if r['error'] and r['error']['kind'] == 'rate_limit'):>10}"
              f"{(statistics.median(latencies) if latencies else 0):>9.0f}{(max(latencies) if latencies else 0):>9.0f}"
              f"{(statistics.mean(r['output_tokens'] for r in answered) if answered else 0):>9.0f}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Collect answers from every tier.")
    parser.add_argument("--pilot", action="store_true", help="run the 30-prompt pilot into data/pilot/")
    parser.add_argument("--pilot2", action="store_true", help="pilot 1 + 30 auto-gradable own prompts into data/pilot2/")
    parser.add_argument("--report-only", action="store_true", help="only rebuild the review and summary")
    args = parser.parse_args()

    registry = load_registry()
    config = load_collection_config(registry)
    prompts = load_prompts()
    out_dir = PILOT2_DIR if args.pilot2 else PILOT_DIR if args.pilot else FULL_DIR
    if args.pilot2:
        prompts = pilot2_prompts(prompts)
    elif args.pilot:
        prompts = pilot_prompts(prompts)
    log = AnswerLog(out_dir / "answers.jsonl")
    tiers = list(config.tiers)

    if not args.report_only:
        collector = Collector(config, registry, log, run="pilot2" if args.pilot2 else "pilot" if args.pilot else "full")
        started = time.monotonic()
        print(f"Collecting {len(prompts)} prompts x {len(tiers)} tiers -> {log.path.relative_to(PROJECT_ROOT)}")
        collector.run_all(prompts)
        print(f"\nFinished in {time.monotonic() - started:.0f}s")
        for key, n in sorted(collector.counts.items()):
            print(f"  {key}: {n}")
        print(f"  cache: {collector.sender.hits} hits, {collector.sender.misses} misses")

    if args.pilot or args.pilot2:
        write_review(prompts, log, tiers, out_dir / "review.txt")
        print(f"Wrote {(out_dir / 'review.txt').relative_to(PROJECT_ROOT)}")
    summarize(prompts, log, tiers)


if __name__ == "__main__":
    main()
