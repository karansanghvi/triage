"""Phase 4 Step 3: run the AI checker on the VALIDATION answers, for the routing simulation (BUILT, NOT RUN YET).

Usage (the model is chosen after the Step 2 calibration):
    python -m eval.checker_val_run --model groq-small --dry-run            # count the calls; no API calls
    python -m eval.checker_val_run --model groq-small --limit 10           # try a few first
    python -m eval.checker_val_run --model groq-small                      # everything (resumable)
    python -m eval.checker_val_run --model groq-small --tiers medium       # medium answers only
Long runs:
    nohup caffeinate -i ./venv/bin/python -m eval.checker_val_run --model groq-small \
        >> data/checker_val/run_groq-small.log 2>&1 &

What gets checked: the Phase 2 low and medium answers (--tiers) on the 117 labeled validation prompts,
skipping answers the free format checker already fails (a router never sends those to the AI checker) and
missing answers. Results go to data/checker_val/checks_<model>.jsonl through CheckLog, so a rerun reuses
finished results and costs nothing; provider errors are retried on the next pass (up to MAX_PASSES).

Pacing: the Phase 2 Pacer, counting every earlier use of the model (Phase 2 answers, judge runs, the
calibration logs and this log). It refuses to start while a calibration run for the same model is running,
because the two would share one daily budget without seeing each other's calls.

Reads validation prompts and answers only; never the test split.
"""
import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Optional

from app.registry import PROJECT_ROOT, load_registry
from app.verifier.ai_checker import load_checker_settings, prompt_sha256
from app.verifier.checker import check_answer
from eval.build_dataset import (DAILY_QUOTA_BACKOFF_SECONDS, DAILY_QUOTA_HINT, FULL_DIR, Pacer,
                                load_collection_config, load_prompts, running_pid)
from eval.check_log import CheckLog, is_finished, logged_check
from eval.checker_calibration import CALIB_DIR, PacedSend, BudgetExhausted, check_log_usage, usage_history
from eval.checker_verdicts import CheckEstimate, make_estimate, read_records
from eval.routing_sim import load_eval_rows
from eval.score_answers import load_answers
from eval.splits import assert_no_test_ids, load_validation_split

VAL_DIR = PROJECT_ROOT / "data" / "checker_val"
DEFAULT_TIERS = ("low", "medium")
MAX_PASSES = 3


def log_path(model: str, val_dir: Path = VAL_DIR) -> Path:
    return Path(val_dir) / f"checks_{model}.jsonl"


def val_items(rows: list, prompts: dict, answers: dict, tiers=DEFAULT_TIERS) -> tuple:
    """([{prompt_id, tier}], skipped Counter) for the answers that pass the free format checker."""
    ids = {r.id for r in rows}
    assert_no_test_ids(ids)
    if not ids <= load_validation_split():
        raise ValueError("val_items is for validation prompts only")
    items, skipped = [], Counter()
    for r in sorted(rows, key=lambda r: r.id):
        messages = prompts[r.id]["messages"]
        for tier in tiers:
            a = answers.get((r.id, tier))
            if not a or not a.get("answer"):
                skipped[f"{tier}: no answer"] += 1
                continue
            fmt = check_answer(messages, a["answer"], truncated=bool(a.get("truncated")), error=a.get("error"))
            if not fmt.ok:
                skipped[f"{tier}: format checker fails it"] += 1
                continue
            items.append({"prompt_id": r.id, "tier": tier})
    return items, skipped


def load_inputs(tiers=DEFAULT_TIERS) -> tuple:
    """(validation rows, prompts, answers, items, skipped) from the real data files."""
    rows = load_eval_rows("val", allow_val=True)
    val = {r.id for r in rows}
    prompts = {p["id"]: p for p in load_prompts() if p["id"] in val}
    answers = {k: v for k, v in load_answers(FULL_DIR / "answers.jsonl").items() if k[0] in val}
    items, skipped = val_items(rows, prompts, answers, tiers)
    return rows, prompts, answers, items, skipped


def todo(items: list, records: list, model: str, version: int) -> list:
    """Items with no finished record from this model, checker version and prompt hash (newest record wins)."""
    latest = {}
    for r in records:
        if (r.get("checker_model"), r.get("checker_version")) == (model, version):
            latest[(r["prompt_id"], r["tier"])] = r
    current = prompt_sha256()
    done = {k for k, r in latest.items() if is_finished(r) and r.get("prompt_sha256") == current}
    return [c for c in items if (c["prompt_id"], c["tier"]) not in done]


def dry_run(model: str, items: list, skipped: Counter, prompts: dict, answers: dict, n_prompts: int,
            val_dir: Path = VAL_DIR, settings=None, registry=None) -> dict:
    """How many calls a run would make now, with a token / cost / time estimate. No API calls."""
    settings = settings or load_checker_settings()
    registry = registry or load_registry()
    left = todo(items, read_records(log_path(model, val_dir)), model, settings.checker_version)   # read-only
    calib = read_records(Path(CALIB_DIR) / f"checks_{model}.jsonl")
    estimate = make_estimate(model, calib, registry)        # output tokens / latency from calibration, if any
    in_tokens = [CheckEstimate.input_tokens_for(prompts[c["prompt_id"]]["messages"],
                                                answers[(c["prompt_id"], c["tier"])]["answer"]) for c in left]
    tokens = sum(in_tokens) + estimate.output_tokens * len(left)
    tpd = load_collection_config(registry).limits[model].tpd
    by_tier = Counter(c["tier"] for c in items)
    left_by_tier = Counter(c["tier"] for c in left)
    return {
        "model": model, "checker_version": settings.checker_version, "validation_prompts": n_prompts,
        "answers_to_check": dict(sorted(by_tier.items())), "skipped": dict(sorted(skipped.items())),
        "already_finished": len(items) - len(left), "calls_to_make": len(left),
        "calls_to_make_by_tier": dict(sorted(left_by_tier.items())),
        "estimated_tokens": round(tokens), "estimated_cost_usd": round(sum(estimate.cost(t) for t in in_tokens), 4),
        "estimated_days_at_tpd": round(tokens / tpd, 2) if tpd else None,
        "estimate_basis": f"output {estimate.output_tokens:.0f} tokens per call "
                          f"({'median of calibration' if calib else 'default'}); retries not included",
    }


def run(model: str, items: list, prompts: dict, answers: dict, *, limit: int = 0, wait: bool = True,
        send_fn: Optional[Callable] = None, pacer: Optional[Pacer] = None, use_pacer: bool = True,
        val_dir: Path = VAL_DIR, settings=None, sleep: Callable = time.sleep, say=print,
        passes: int = MAX_PASSES) -> dict:
    registry = load_registry()
    settings = settings or load_checker_settings()
    log = CheckLog(log_path(model, val_dir))
    if use_pacer and pacer is None:
        history = usage_history(model) + check_log_usage(log_path(model, val_dir), model)
        pacer = Pacer(load_collection_config(registry).limits[model], history)
    send = PacedSend(pacer if use_pacer else None, send_fn, wait=wait, say=say)
    summary = Counter()
    for attempt in range(1, passes + 1):
        left = todo(items, log.records, model, settings.checker_version)
        if limit:
            left = left[:max(0, limit - summary["checked"])]
        if not left:
            break
        say(f"{model} validation pass {attempt}: checking {len(left)}")
        for c in left:
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
            if result.error_kind == "provider:rate_limit":
                sleep(DAILY_QUOTA_BACKOFF_SECONDS if DAILY_QUOTA_HINT.search(result.reason) else 60)
        if limit and summary["checked"] >= limit:
            break
        if attempt < passes and any(k.startswith("error:provider") for k in summary):
            sleep(60)
    return {**summary, "real_calls": send.calls}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the AI checker on the validation answers.")
    parser.add_argument("--model", required=True, help="checker model (a models.yaml name), chosen after Step 2")
    parser.add_argument("--tiers", nargs="+", default=list(DEFAULT_TIERS), choices=["low", "medium"])
    parser.add_argument("--dry-run", action="store_true", help="only count the calls; no API calls")
    parser.add_argument("--limit", type=int, default=0, help="at most this many real checks in this run")
    parser.add_argument("--no-wait", action="store_true", help="stop instead of waiting when the daily budget is used")
    args = parser.parse_args(argv)
    if args.model not in load_registry():
        raise SystemExit(f"{args.model}: not a models.yaml name")
    rows, prompts, answers, items, skipped = load_inputs(tuple(args.tiers))
    if args.dry_run:
        print(json.dumps(dry_run(args.model, items, skipped, prompts, answers, len(rows)), indent=2))
        return
    calib_pid = running_pid(CALIB_DIR / f"run_{args.model}.pid")
    if calib_pid:
        raise SystemExit(f"a calibration run for {args.model} is running (pid {calib_pid}); they would share one "
                         "daily budget. Wait for it to finish.")
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    pid_path = VAL_DIR / f"run_{args.model}.pid"
    other = running_pid(pid_path)
    if other and other != os.getpid():
        raise SystemExit(f"another validation run for {args.model} is already running (pid {other})")
    pid_path.write_text(str(os.getpid()))
    try:
        print(run(args.model, items, prompts, answers, limit=args.limit, wait=not args.no_wait))
    finally:
        if running_pid(pid_path) == os.getpid():
            pid_path.unlink()


if __name__ == "__main__":
    main()
