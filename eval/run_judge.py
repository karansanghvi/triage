"""Phase 2 Step 3: LLM-judge the judge-scored prompts, option C.

Usage:
    python -m eval.run_judge             # judge everything not judged yet
    python -m eval.run_judge status      # progress per tier

For each judge-scored prompt (data/prompts.jsonl, read-only) with answers in data/answers/answers.jsonl
(read-only):
    high    passes automatically when its answer is complete, since it is the reference. If the high answer
            is truncated or empty, the prompt is no_reference for every tier (no labels, left out of training).
    low     failed without a call if its answer is truncated, empty or too large; otherwise judged.
    medium  judged only if low failed. If low passed, medium is recorded as assumed_pass (option C), so it is
            always clear which results were judged and which were assumed.

Judgments are appended to data/judgments/judgments.jsonl, keyed by (prompt_id, tier); the newest record
wins. Rerunning skips finished judgments, retries pending and error ones, paces within gpt-oss-120b's
free-tier limits (shared with the collector's high tier, so it won't start while the collector runs),
and waits out daily limits using wall-clock time. Long runs:
    nohup caffeinate -i ./venv/bin/python -m eval.run_judge >> data/judgments/run.log 2>&1 &
"""
import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from app.providers.common import ProviderError
from app.registry import PROJECT_ROOT, load_registry
from app.verifier.judge import JudgeResult, judge, load_judge_settings
from eval.build_dataset import (
    DAILY_QUOTA_BACKOFF_SECONDS, DAILY_QUOTA_HINT, FULL_DIR, Pacer, load_collection_config, load_prompts,
    running_pid, simulate_finish, wait_until,
)
from eval.judge_cost import complete, prefailed
from eval.score_answers import is_input_too_large, load_answers

JUDGMENTS_DIR = PROJECT_ROOT / "data" / "judgments"
JUDGE_TIERS = ("low", "medium", "high")
FINISHED = {"pass", "fail", "assumed_pass", "no_reference"}
MAX_PASSES = 5
WAITS_FILE = "waits.json"


# ---------- Records ----------

def make_record(prompt_id: str, tier: str, status: str, method: str, reason: str = "",
                result: Optional[JudgeResult] = None, usage: Optional[dict] = None, error: Optional[dict] = None,
                judge_model: Optional[str] = None) -> dict:
    """One judgment. status: pass, fail, assumed_pass, no_reference (finished) or pending, error (retried).
    method says how it was decided: judged, auto_high, assumed (option C), pre_failed, no_reference, waiting."""
    usage = usage or {}
    return {
        "prompt_id": prompt_id, "tier": tier, "status": status, "method": method, "reason": reason,
        "score": result.score if result else None,
        "judge_score": result.judge_score if result else None,
        "broken_constraints": list(result.broken_constraints) if result else [],
        "attempts": result.attempts if result else 0,
        "judge_model": judge_model if method == "judged" else None,
        "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
        "api_calls": usage.get("api_calls", 0),
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def prefail_reason(record: dict) -> str:
    if record["error"]:
        return "input too large"
    return "truncated" if record["truncated"] else "empty"


# ---------- Option C for one prompt ----------

def judge_prompt(prompt: dict, answers: dict, finished: dict, judge_call: Callable, judge_model: str = "") -> list:
    """New records for this prompt's unfinished tiers, following option C.

    finished maps (prompt_id, tier) -> the newest finished record. judge_call(prompt, candidate, reference)
    returns (JudgeResult, usage dict) and may raise ProviderError, which becomes a pending record.
    """
    pid = prompt["id"]
    records = []

    def done(tier):
        return finished.get((pid, tier))

    def emit(tier, status, method, reason, **kw):
        record = make_record(pid, tier, status, method, reason, judge_model=judge_model, **kw)
        records.append(record)
        return record

    high = answers.get((pid, "high"))
    if high is None or (high["error"] and not is_input_too_large(high["error"])):
        for tier in JUDGE_TIERS:
            if not done(tier):
                emit(tier, "pending", "waiting", "no usable high answer yet")
        return records
    if not complete(high):
        for tier in JUDGE_TIERS:
            if not done(tier):
                emit(tier, "no_reference", "no_reference", f"high answer is {prefail_reason(high)}")
        return records
    if not done("high"):
        emit("high", "pass", "auto_high", "high answer is complete; it is the reference")

    def judge_tier(tier):
        answer = answers.get((pid, tier))
        if answer is None or (answer["error"] and not is_input_too_large(answer["error"])):
            return emit(tier, "pending", "waiting", "no answer collected yet")
        if prefailed(answer):
            return emit(tier, "fail", "pre_failed", prefail_reason(answer))
        try:
            result, usage = judge_call(prompt, answer["answer"], high["answer"])
        except ProviderError as error:
            return emit(tier, "pending", "judged", "provider error; will be retried",
                        error={"kind": error.kind, "message": str(error)[:300]})
        if result.status == "scored":
            return emit(tier, "pass" if result.passed else "fail", "judged", result.reason, result=result, usage=usage)
        return emit(tier, "error", "judged", result.reason, result=result, usage=usage)

    low = done("low") or judge_tier("low")
    if done("medium"):
        return records
    if low["status"] == "pass":
        emit("medium", "assumed_pass", "assumed (option C)", "low passed, so medium is assumed to pass")
    elif low["status"] == "fail":
        judge_tier("medium")
    elif not done("medium"):
        emit("medium", "pending", "waiting", "waiting for the low judgment")
    return records


# ---------- Log ----------

class JudgmentLog:
    """Append-only judgments.jsonl; the newest record per (prompt_id, tier) wins."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] \
            if path.exists() else []

    def latest(self) -> dict:
        out = {}
        for r in self.records:
            out[(r["prompt_id"], r["tier"])] = r
        return out

    def finished(self) -> dict:
        return {k: r for k, r in self.latest().items() if r["status"] in FINISHED}

    def append(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records.append(record)


# ---------- Runner ----------

class JudgeRunner:
    def __init__(self, prompts: list, answers: dict, log: JudgmentLog, judge_call: Optional[Callable] = None,
                 pacer: Optional[Pacer] = None, out_dir: Path = JUDGMENTS_DIR, settings=None):
        self.prompts = [p for p in prompts if p["scoring"] == "judge"]
        self.answers = answers
        self.log = log
        self.pacer = pacer                       # None: no pacing (tests)
        self.out_dir = out_dir
        self.settings = settings
        self.judge_call = judge_call or self.real_judge_call
        self.judge_model = settings.model_name if settings else "fake"

    def say(self, message: str) -> None:
        print(f"{datetime.now():%m-%d %H:%M:%S} {message}", flush=True)

    def set_waiting(self, until: Optional[datetime], reason: str = "") -> None:
        path = self.out_dir / WAITS_FILE
        state = {"until": until.isoformat(), "reason": reason} if until else {}
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps(state))
        os.replace(temp, path)

    def wait_logged(self, seconds: float, why: str) -> None:
        until = datetime.now().astimezone() + timedelta(seconds=seconds)
        self.say(f"{why}; waiting {seconds / 3600:.1f}h, until about {until:%Y-%m-%d %H:%M %Z}")
        self.set_waiting(until, why)
        try:
            wait_until(until.astimezone(timezone.utc))
        finally:
            self.set_waiting(None)

    def wait_for_budget(self, estimate: int) -> None:
        if self.pacer is None:
            return
        while not self.pacer.daily_budget_left(estimate):
            self.wait_logged(self.pacer.seconds_until_daily_room(estimate) + 30,
                             f"daily limit for {self.judge_model} reached")
        self.pacer.wait_for_minute_slot(estimate)

    def real_judge_call(self, prompt: dict, candidate: str, reference: str) -> tuple:
        """judge() with a paced, token-counting send to the configured judge model."""
        from app.providers.base import send_request
        model = load_registry()[self.settings.model_name]
        response_format = {"type": self.settings.response_format} if self.settings.response_format else None
        usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}

        def send(messages, settings):
            estimate = sum(len(m["content"]) for m in messages) // 4 + settings.max_tokens
            self.wait_for_budget(estimate)
            response = send_request(messages, model, temperature=settings.temperature, max_tokens=settings.max_tokens,
                                    timeout=settings.timeout_seconds, reasoning_effort=settings.reasoning_effort,
                                    response_format=response_format)
            usage["input_tokens"] += response.input_tokens
            usage["output_tokens"] += response.output_tokens
            usage["api_calls"] += 1
            if self.pacer is not None:
                self.pacer.record(1, response.input_tokens + response.output_tokens)
            return response.text

        result = judge(prompt["messages"], prompt["constraints"], reference, candidate, self.settings, send=send)
        return result, usage

    def run_pass(self) -> int:
        """One pass over every prompt with unfinished tiers. Returns how many tiers are still unfinished."""
        unfinished = 0
        for prompt in self.prompts:
            finished = self.log.finished()
            if all((prompt["id"], t) in finished for t in JUDGE_TIERS):
                continue
            for record in judge_prompt(prompt, self.answers, finished, self.judge_call, self.judge_model):
                self.log.append(record)
                if record["status"] not in FINISHED:
                    unfinished += 1
                if record["method"] in ("judged", "pre_failed") or record["status"] not in FINISHED:
                    score = f" score={record['score']}" if record["score"] is not None else ""
                    self.say(f"{record['prompt_id']} {record['tier']}: {record['status']} ({record['method']}){score}")
                error = record["error"] or {}
                if error.get("kind") == "rate_limit":
                    if DAILY_QUOTA_HINT.search(error.get("message", "")):
                        self.wait_logged(DAILY_QUOTA_BACKOFF_SECONDS, f"{self.judge_model} returned a daily-quota error")
                    else:
                        time.sleep(60)
        return unfinished

    def run(self, passes: int = MAX_PASSES, pause: float = 60) -> None:
        for attempt in range(1, passes + 1):
            left = self.run_pass()
            if not left:
                self.say("finished: every judge-scored prompt has a result")
                return
            if attempt < passes:
                self.say(f"pass {attempt} done; {left} judgments pending, retrying in {pause:.0f}s")
                time.sleep(pause)
        self.say(f"{left} judgments still pending after {passes} passes; rerun later")


# ---------- Status ----------

def show_status(prompts: list, out_dir: Path = JUDGMENTS_DIR) -> None:
    judged_prompts = [p for p in prompts if p["scoring"] == "judge"]
    log = JudgmentLog(out_dir / "judgments.jsonl")
    latest = log.latest()
    pid = running_pid(out_dir / "run.pid")
    print(f"{log.path.relative_to(PROJECT_ROOT)}: {len(log.records)} records | "
          + (f"judge running (pid {pid})" if pid else "judge NOT running")
          + (" | collector RUNNING (judging can't start yet)" if running_pid(FULL_DIR / "run.pid") else ""))
    try:
        wait = json.loads((out_dir / WAITS_FILE).read_text())
    except (FileNotFoundError, ValueError):
        wait = {}
    if pid and wait.get("until") and datetime.fromisoformat(wait["until"]) > datetime.now(timezone.utc):
        print(f"waiting until {datetime.fromisoformat(wait['until']).astimezone():%a %H:%M} ({wait['reason']})")
    statuses = ["pass", "fail", "assumed_pass", "no_reference", "pending", "error", "not started"]
    print(f"\n{'tier':<8}" + "".join(f"{s:>14}" for s in statuses))
    for tier in JUDGE_TIERS:
        counts = Counter(latest[(p["id"], tier)]["status"] if (p["id"], tier) in latest else "not started"
                         for p in judged_prompts)
        print(f"{tier:<8}" + "".join(f"{counts[s]:>14}" for s in statuses))
    calls = [r for r in log.records if r["api_calls"]]
    tokens = sum(r["input_tokens"] + r["output_tokens"] for r in calls)
    print(f"\njudge calls so far: {sum(r['api_calls'] for r in calls)} | tokens: {tokens:,}")

    # Remaining calls: every unfinished low, plus unfinished mediums weighted by the low fail rate so far
    low_done = [latest[(p["id"], "low")] for p in judged_prompts if latest.get((p["id"], "low"), {}).get("status") in ("pass", "fail")]
    fail_rate = sum(r["status"] == "fail" for r in low_done) / len(low_done) if low_done else 0.6
    unfinished = lambda tier: [p for p in judged_prompts if latest.get((p["id"], tier), {}).get("status") not in FINISHED]
    remaining = len(unfinished("low")) + len(unfinished("medium")) * fail_rate
    if remaining <= 0:
        print("estimated finish: done")
        return
    per_call = tokens / sum(r["api_calls"] for r in calls) if calls else 2_900
    config = load_collection_config(load_registry())
    settings = load_judge_settings()
    limits = config.limits[settings.model_name]
    answers = [json.loads(l) for l in (FULL_DIR / "answers.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][:-1]
    window = [(datetime.fromisoformat(r["timestamp"]), (r["input_tokens"] or 0) + (r["output_tokens"] or 0))
              for r in answers if r["model_name"] == settings.model_name and r.get("api_calls")]
    window += [(datetime.fromisoformat(r["timestamp"]), r["input_tokens"] + r["output_tokens"]) for r in calls]
    start = datetime.now(timezone.utc)
    if pid and wait.get("until"):
        start = max(start, datetime.fromisoformat(wait["until"]).astimezone(timezone.utc))
    fake_prompt = {"messages": [{"role": "user", "content": "x" * int(per_call * 3)}]}
    end = simulate_finish([fake_prompt] * round(remaining), lambda p: per_call, window, limits, start, 3.0,
                          settings.max_tokens)
    print(f"~{remaining:.0f} judge calls left (~{per_call:,.0f} tokens each; medium judged for {fail_rate:.0%} of "
          f"low results) | estimated finish {end.astimezone():%a %m-%d %H:%M}")


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="LLM-judge the judge-scored prompts (option C).")
    parser.add_argument("command", nargs="?", default="run", choices=["run", "status"])
    args = parser.parse_args()
    prompts = load_prompts()
    if args.command == "status":
        show_status(prompts)
        return

    collector = running_pid(FULL_DIR / "run.pid")
    if collector:
        raise SystemExit(f"the collector is still running (pid {collector}); judging shares gpt-oss-120b's daily "
                         "token limit with it, so wait until collection has finished")
    JUDGMENTS_DIR.mkdir(parents=True, exist_ok=True)
    pid_path = JUDGMENTS_DIR / "run.pid"
    other = running_pid(pid_path)
    if other and other != os.getpid():
        raise SystemExit(f"another judging run is already running (pid {other})")
    pid_path.write_text(str(os.getpid()))

    settings = load_judge_settings()
    config = load_collection_config(load_registry())
    answers_path = FULL_DIR / "answers.jsonl"
    history = [(datetime.fromisoformat(r["timestamp"]), r["api_calls"], (r["input_tokens"] or 0) + (r["output_tokens"] or 0))
               for r in map(json.loads, filter(str.strip, answers_path.read_text(encoding="utf-8").splitlines()))
               if r["model_name"] == settings.model_name and r.get("api_calls")]
    log = JudgmentLog(JUDGMENTS_DIR / "judgments.jsonl")
    history += [(datetime.fromisoformat(r["timestamp"]), r["api_calls"], r["input_tokens"] + r["output_tokens"])
                for r in log.records if r["api_calls"]]
    runner = JudgeRunner(prompts, load_answers(answers_path), log, pacer=Pacer(config.limits[settings.model_name], history),
                         settings=settings)
    runner.say(f"pid {os.getpid()}: judging {len(runner.prompts)} judge-scored prompts with {settings.model_name} (option C)")
    try:
        runner.run()
    finally:
        if running_pid(pid_path) == os.getpid():
            pid_path.unlink()


if __name__ == "__main__":
    main()
