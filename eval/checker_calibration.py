"""Phase 4 Step 2, part 2: run the AI checker on the calibration sample.

Usage:
    python -m eval.checker_calibration run --model groq-small            # check the whole sample (resumable)
    python -m eval.checker_calibration run --model groq-large --limit 10 # try a few first
    python -m eval.checker_calibration run --model groq-small --repeat   # repeat-run check (after the main run)
    python -m eval.checker_calibration status --model groq-small
Long runs:
    nohup caffeinate -i ./venv/bin/python -m eval.checker_calibration run --model groq-small \
        >> data/checker_calibration/run_groq-small.log 2>&1 &

One log per model and mode (data/checker_calibration/checks_<model>.jsonl, repeats_<model>.jsonl), written
through CheckLog, so a rerun reuses finished results and costs nothing. Different models use different files,
pid locks (one per model) and daily budgets, so groq-small and groq-large can run at the same time.

Every result is recorded, errors included (CheckLog appends every attempt's outcome). Deterministic errors
(truncated, empty, invalid JSON, bad shape) are finished and reported as errors. Provider errors (timeouts,
rate limits) are retried on the next pass, up to MAX_PASSES per run, and stay in the log either way.

Calls are paced with the Phase 2 Pacer against the model's limits in config/collection.yaml, counting earlier
usage of that model (Phase 2 answers, judge runs, every check log). When the daily budget is used up it waits,
using wall-clock time, unless --no-wait is given.

The repeat mode re-runs each repeat item once, in a separate log, so the main result is never reused for it.
It only runs items whose main result is finished.
"""
import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from app.providers.base import OPENAI_FORMAT_PROVIDERS
from app.registry import PROJECT_ROOT, load_registry
from app.verifier.ai_checker import default_send, load_checker_settings
from eval.build_dataset import (DAILY_QUOTA_BACKOFF_SECONDS, DAILY_QUOTA_HINT, FULL_DIR, Pacer,
                                load_collection_config, load_prompts, running_pid, wait_until)
from eval.check_log import CHECKS_DIR, CheckLog, is_finished, logged_check
from eval.checker_calibration_sample import CALIB_DIR, SAMPLE_PATH, stratum, stratum_key
from eval.run_judge import JUDGMENTS_DIR, judge_usage
from eval.score_answers import load_answers
from eval.splits import assert_no_test_ids

MAX_PASSES = 3
JUDGE_CALIB_DIR = PROJECT_ROOT / "data" / "judge_calibration"


def log_path(model: str, repeat: bool = False, calib_dir: Path = CALIB_DIR) -> Path:
    return calib_dir / f"{'repeats' if repeat else 'checks'}_{model}.jsonl"


def load_sample(path: Path = SAMPLE_PATH) -> dict:
    if not path.exists():
        raise SystemExit("no sample yet; run: python -m eval.checker_calibration_sample sample")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert_no_test_ids({c["prompt_id"] for c in data["items"]})
    return data


# ---------- Pacing ----------

def check_log_usage(path: Path, model: str) -> list:
    """(UTC time, api calls, tokens) for this model's real calls in one check log."""
    if not path.exists():
        return []
    usage = []
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("checker_model") == model and r.get("api_calls"):
            usage.append((datetime.fromisoformat(r["timestamp"]), r["api_calls"],
                          (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0)))
    return usage


def usage_history(model: str, calib_dir: Path = CALIB_DIR) -> list:
    """Everything that has used this model's quota: Phase 2 answers, judge runs, every check log."""
    history = []
    answers_path = FULL_DIR / "answers.jsonl"
    if answers_path.exists():
        for line in answers_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line) if line.strip() else {}
            if r.get("model_name") == model and r.get("api_calls"):
                history.append((datetime.fromisoformat(r["timestamp"]), r["api_calls"],
                                (r["input_tokens"] or 0) + (r["output_tokens"] or 0)))
    if model == judge_model_name():
        history += judge_usage(JUDGMENTS_DIR) + judge_usage(JUDGE_CALIB_DIR)
    for path in [CHECKS_DIR / "checks.jsonl", *sorted(Path(calib_dir).glob("*.jsonl"))]:
        history += check_log_usage(path, model)
    return history


def judge_model_name() -> str:
    from app.verifier.judge import load_judge_settings
    return load_judge_settings().model_name


class PacedSend:
    """default_send behind the Pacer: waits for daily room (wall clock) and for a per-minute slot."""

    def __init__(self, pacer: Optional[Pacer], send_fn: Callable = None, wait: bool = True, say=print,
                 waiter: Callable = wait_until):
        self.pacer, self.send_fn, self.wait, self.say, self.waiter = pacer, send_fn or default_send, wait, say, waiter
        self.calls = 0

    def __call__(self, messages, model_name, settings):
        estimate = sum(len(m["content"]) for m in messages) // 4 + settings.max_tokens
        if self.pacer is not None:
            while not self.pacer.daily_budget_left(estimate):
                if not self.wait:
                    raise BudgetExhausted(f"{model_name}'s daily budget is used up")
                seconds = self.pacer.seconds_until_daily_room(estimate) + 30
                until = datetime.now(timezone.utc) + timedelta(seconds=seconds)
                self.say(f"daily limit for {model_name} reached; waiting {seconds / 3600:.1f}h")
                self.waiter(until)
            self.pacer.wait_for_minute_slot(estimate)
        response = self.send_fn(messages, model_name, settings)
        self.calls += 1
        if self.pacer is not None:
            self.pacer.record(1, (response.input_tokens or 0) + (response.output_tokens or 0))
        return response


class BudgetExhausted(Exception):
    pass


# ---------- Running ----------

def todo_items(sample: dict, repeat: bool, main_log: CheckLog, model: str, version: int) -> list:
    items = sample["items"]
    if not repeat:
        return items
    wanted = {tuple(x) for x in sample["repeat_items"]}
    finished = main_log.finished(checker_model=model, checker_version=version)
    return [c for c in items if (c["prompt_id"], c["tier"]) in wanted
            and (c["prompt_id"], c["tier"], model, version) in finished]


def run(model: str, repeat: bool = False, limit: int = 0, wait: bool = True, send_fn: Optional[Callable] = None,
        pacer: Optional[Pacer] = None, use_pacer: bool = True, sample_path: Path = SAMPLE_PATH,
        calib_dir: Path = CALIB_DIR, prompts: Optional[dict] = None, answers: Optional[dict] = None,
        settings=None, sleep: Callable = time.sleep, say=print, passes: int = MAX_PASSES) -> dict:
    """Check the sample (or the repeat items) with `model`. Returns a summary of this run."""
    registry = load_registry()
    if model not in registry or registry[model].provider not in OPENAI_FORMAT_PROVIDERS:
        raise SystemExit(f"{model}: not a models.yaml name on an OpenAI-format provider")
    settings = settings or load_checker_settings()
    sample = load_sample(sample_path)
    prompts = prompts if prompts is not None else {p["id"]: p for p in load_prompts()}
    answers = answers if answers is not None else load_answers(FULL_DIR / "answers.jsonl")
    main_log = CheckLog(log_path(model, False, calib_dir))
    log = CheckLog(log_path(model, True, calib_dir)) if repeat else main_log
    if use_pacer and pacer is None:
        pacer = Pacer(load_collection_config(registry).limits[model], usage_history(model, calib_dir))
    send = PacedSend(pacer if use_pacer else None, send_fn, wait=wait, say=say)

    items = todo_items(sample, repeat, main_log, model, settings.checker_version)
    summary = Counter()
    for attempt in range(1, passes + 1):
        finished = log.finished(checker_model=model, checker_version=settings.checker_version)
        todo = [c for c in items if (c["prompt_id"], c["tier"], model, settings.checker_version) not in finished]
        if limit:
            todo = todo[:max(0, limit - summary["checked"])]
        if not todo:
            break
        say(f"{model} {'repeat' if repeat else 'main'} pass {attempt}: {len(finished)} finished, checking {len(todo)}")
        for c in todo:
            pid, tier = c["prompt_id"], c["tier"]
            try:
                result, reused = logged_check(log, pid, tier, prompts[pid]["messages"], answers[(pid, tier)]["answer"],
                                              model=model, send=send, settings=settings)
            except BudgetExhausted as stop:
                say(f"stopping: {stop}; rerun later to continue")
                summary["stopped_for_budget"] += 1
                return {**summary, "real_calls": send.calls}
            summary["checked"] += 0 if reused else 1
            summary[result.status if result.status == "ok" else f"error:{result.error_kind}"] += 1
            verdict = "error" if result.status != "ok" else ("good" if result.good else "not good")
            say(f"{pid} {tier}: truly {'bad' if c['truth_bad'] else 'good'}, checker {verdict}"
                + (f" ({result.error_kind})" if result.error_kind else ""))
            kind = result.error_kind or ""
            if kind == "provider:rate_limit":
                sleep(DAILY_QUOTA_BACKOFF_SECONDS if DAILY_QUOTA_HINT.search(result.reason) else 60)
        if limit and summary["checked"] >= limit:
            break
        if attempt < passes and any(k.startswith("error:provider") for k in summary):
            sleep(60)                                  # give a provider problem a minute before the next pass
    return {**summary, "real_calls": send.calls}


# ---------- Status ----------

def status(model: str, sample_path: Path = SAMPLE_PATH, calib_dir: Path = CALIB_DIR, settings=None) -> dict:
    settings = settings or load_checker_settings()
    sample = load_sample(sample_path)
    out = {}
    for repeat in (False, True):
        latest = CheckLog(log_path(model, repeat, calib_dir)).latest()
        items = sample["items"] if not repeat else [c for c in sample["items"]
                                                    if [c["prompt_id"], c["tier"]] in sample["repeat_items"]]
        by = Counter()
        for c in items:
            r = latest.get((c["prompt_id"], c["tier"], model, settings.checker_version))
            state = "not started" if r is None else (
                ("ok" if r["status"] == "ok" else f"error:{r['error_kind']}") if is_finished(r) else "pending")
            by[(stratum_key(stratum(c)), state)] += 1
        out["repeat" if repeat else "main"] = {f"{k} {s}": n for (k, s), n in sorted(by.items())}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the AI checker on the calibration sample.")
    parser.add_argument("command", choices=["run", "status"])
    parser.add_argument("--model", required=True, help="checker model (a models.yaml name)")
    parser.add_argument("--repeat", action="store_true", help="re-run the repeat items once (separate log)")
    parser.add_argument("--limit", type=int, default=0, help="at most this many real checks in this run")
    parser.add_argument("--no-wait", action="store_true", help="stop instead of waiting when the daily budget is used")
    args = parser.parse_args(argv)
    if args.command == "status":
        for mode, rows in status(args.model).items():
            print(f"{mode}:")
            for k, n in rows.items():
                print(f"  {k:<40}{n:>5}")
        return
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    pid_path = CALIB_DIR / f"run_{args.model}.pid"     # one lock per model: main and repeat share its budget
    other = running_pid(pid_path)
    if other and other != os.getpid():
        raise SystemExit(f"another run for {args.model} is already running (pid {other})")
    pid_path.write_text(str(os.getpid()))
    try:
        print(run(args.model, repeat=args.repeat, limit=args.limit, wait=not args.no_wait))
    finally:
        if running_pid(pid_path) == os.getpid():
            pid_path.unlink()


if __name__ == "__main__":
    main()
