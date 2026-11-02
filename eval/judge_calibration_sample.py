"""Phase 3 Step 4, judge calibration part 1: judge a sample of exact-match answers.

GSM8K and MMLU answers were graded by exact match, so we know whether each one is really right. Here we
also run the LLM judge on a sample of them, exactly the way it judged open-ended prompts (same judge,
same rubric, reference = high tier's answer). eval/judge_calibration.py then compares the two.

Usage:
    python -m eval.judge_calibration_sample sample          # pick and save the sample (no API calls)
    python -m eval.judge_calibration_sample run             # judge the sample (resumable)
    python -m eval.judge_calibration_sample run --limit 10  # try a few first
    python -m eval.judge_calibration_sample status

Rules this follows:
    * training-split prompts only (checked with assert_no_test_ids), because calibration may lead to
      tuning the judge or its threshold
    * low and medium answers only: judging the high answer against itself tells us nothing
    * balanced across source x tier x truly-right/truly-wrong, at most one answer per prompt
    * the sample is saved once with a fixed seed and never silently redrawn
    * results go to data/judge_calibration/, never to outcomes.jsonl or data/judgments/
    * shares gpt-oss-120b's limits with the collector and the main judge, so it refuses to run while
      either is running, and counts their past usage when pacing
"""
import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from app.providers.common import ProviderError
from app.registry import PROJECT_ROOT, load_registry
from app.verifier.judge import load_judge_settings
from eval.build_dataset import (
    DAILY_QUOTA_BACKOFF_SECONDS, DAILY_QUOTA_HINT, FULL_DIR, Pacer, load_collection_config, load_prompts,
    running_pid,
)
from eval.judge_cost import complete, prefailed
from eval.run_judge import JUDGMENTS_DIR, JudgeRunner, JudgmentLog, judge_usage
from eval.score_answers import load_answers
from eval.splits import assert_no_test_ids, load_split

OUTCOMES_PATH = PROJECT_ROOT / "data" / "outcomes.jsonl"
CALIB_DIR = PROJECT_ROOT / "data" / "judge_calibration"
SAMPLE_PATH = CALIB_DIR / "sample.json"
RESULTS_PATH = CALIB_DIR / "judgments.jsonl"   # this name lets run_judge.judge_usage() count its API usage
SOURCES = ("gsm8k", "mmlu")
TIERS = ("low", "medium")
SAMPLE_SIZE = 150
SEED = 42
FINISHED = {"scored", "error"}   # "error" = unparseable verdict after retries; "pending" = provider error, retried


# ---------- Sampling (pure functions, easy to test) ----------

def source_of(outcome: dict):
    """'gsm8k' or 'mmlu' if the outcome comes from one of them, else None."""
    source = str(outcome.get("source", "")).lower()
    return next((s for s in SOURCES if source.startswith(s)), None)


def find_candidates(outcomes: list, answers: dict, train_ids: set) -> list:
    """Every low/medium answer that can be calibrated: train split, GSM8K/MMLU, exact-match pass/fail,
    a complete answer, and a complete high answer to use as the judge's reference."""
    candidates = []
    for outcome in outcomes:
        source = source_of(outcome)
        pid = outcome["id"]
        if source is None or pid not in train_ids or outcome.get("scoring") == "judge":
            continue
        tiers = outcome.get("tiers") or {}
        high = answers.get((pid, "high"))
        if high is None or high.get("error") or not complete(high):
            continue
        for tier in TIERS:
            truth = tiers.get(tier) or {}
            if truth.get("status") not in ("pass", "fail"):
                continue
            answer = answers.get((pid, tier))
            # Truncated/empty answers would be failed without a judge call, so they can't test the judge
            if answer is None or answer.get("error") or prefailed(answer):
                continue
            candidates.append({
                "prompt_id": pid, "source": source, "tier": tier,
                "truth_pass": truth["status"] == "pass", "truth_reason": truth.get("reason", ""),
                # False = the strong model itself was wrong; the judge may then fail a correct answer
                "reference_correct": (tiers.get("high") or {}).get("status") == "pass",
            })
    return candidates


def select_sample(candidates: list, size: int = SAMPLE_SIZE, seed: int = SEED) -> list:
    """Balanced sample: round-robin over (source, tier, truth) groups, at most one answer per prompt.

    A group that runs out simply stops contributing and the others fill the gap, so the sample is as
    balanced as the data allows. Same candidates + same seed = same sample.
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for c in sorted(candidates, key=lambda c: (c["prompt_id"], c["tier"])):
        groups[(c["source"], c["tier"], c["truth_pass"])].append(c)
    for items in groups.values():
        rng.shuffle(items)

    keys = sorted(groups)
    picked, used = [], set()
    while len(picked) < size and any(groups[k] for k in keys):
        for key in keys:
            if len(picked) >= size:
                break
            while groups[key]:
                c = groups[key].pop()
                if c["prompt_id"] not in used:
                    picked.append(c)
                    used.add(c["prompt_id"])
                    break
    return picked


def describe(items: list) -> str:
    counts = Counter((c["source"], c["tier"], "right" if c["truth_pass"] else "wrong") for c in items)
    lines = [f"  {s:<6} {t:<7} {r:<6} {n:>4}" for (s, t, r), n in sorted(counts.items())]
    wrong_ref = sum(not c["reference_correct"] for c in items)
    return "\n".join(lines) + f"\n  (high answer itself wrong for {wrong_ref} of them)"


# ---------- Commands ----------

def load_jsonl(path: Path) -> list:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def make_sample(size: int, seed: int, force: bool) -> None:
    if SAMPLE_PATH.exists() and not force:
        raise SystemExit(f"{SAMPLE_PATH.relative_to(PROJECT_ROOT)} already exists. Redrawing the sample after "
                         "seeing results is cherry-picking; pass --force only if you have a real reason.")
    train_ids, _ = load_split()
    outcomes = load_jsonl(OUTCOMES_PATH)
    if not any(source_of(o) for o in outcomes):
        found = sorted({str(o.get("source")) for o in outcomes})
        raise SystemExit(f"no GSM8K/MMLU outcomes found; sources in outcomes.jsonl are: {found}. Edit SOURCES.")
    candidates = find_candidates(outcomes, load_answers(FULL_DIR / "answers.jsonl"), train_ids)
    sample = select_sample(candidates, size, seed)
    assert_no_test_ids({c["prompt_id"] for c in sample})

    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(), "seed": seed, "size": size,
        "candidates": len(candidates), "items": sample,
    }, indent=2), encoding="utf-8")
    print(f"{len(candidates)} candidate answers; saved a sample of {len(sample)} to "
          f"{SAMPLE_PATH.relative_to(PROJECT_ROOT)}\n{describe(sample)}")
    if len(sample) < size:
        print(f"warning: only {len(sample)} usable answers (one per prompt), fewer than {size}")


def make_record(item: dict, status: str, settings, result=None, usage=None, error=None) -> dict:
    usage = usage or {}
    return {
        **item, "status": status,
        "judge_pass": result.passed if result else None,
        "score": result.score if result else None,
        "judge_score": result.judge_score if result else None,
        "reason": result.reason if result else "",
        "broken_constraints": list(result.broken_constraints) if result else [],
        "attempts": result.attempts if result else 0,
        "judge_model": settings.model_name, "judge_version": settings.judge_version,
        "pass_threshold": settings.pass_threshold,
        "input_tokens": usage.get("input_tokens", 0), "output_tokens": usage.get("output_tokens", 0),
        "api_calls": usage.get("api_calls", 0),
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run(limit: int = 0) -> None:
    if not SAMPLE_PATH.exists():
        raise SystemExit("no sample yet; run: python -m eval.judge_calibration_sample sample")
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))["items"]
    assert_no_test_ids({c["prompt_id"] for c in sample})

    for name, directory in (("collector", FULL_DIR), ("main judge", JUDGMENTS_DIR)):
        pid = running_pid(directory / "run.pid")
        if pid:
            raise SystemExit(f"the {name} is running (pid {pid}) and shares the judge model's limits; "
                             "run calibration after it finishes")
    pid_path = CALIB_DIR / "run.pid"
    other = running_pid(pid_path)
    if other and other != os.getpid():
        raise SystemExit(f"another calibration run is already running (pid {other})")
    pid_path.write_text(str(os.getpid()))

    try:
        settings = load_judge_settings()
        config = load_collection_config(load_registry())
        prompts = {p["id"]: p for p in load_prompts()}
        answers_path = FULL_DIR / "answers.jsonl"
        answers = load_answers(answers_path)

        # Everything that has used this model's quota: collector, main judge, earlier calibration runs
        history = [(datetime.fromisoformat(r["timestamp"]), r["api_calls"],
                    (r["input_tokens"] or 0) + (r["output_tokens"] or 0))
                   for r in load_jsonl(answers_path)
                   if r["model_name"] == settings.model_name and r.get("api_calls")]
        history += judge_usage(JUDGMENTS_DIR) + judge_usage(CALIB_DIR)

        log = JudgmentLog(RESULTS_PATH)
        runner = JudgeRunner([], answers, log, pacer=Pacer(config.limits[settings.model_name], history),
                             out_dir=CALIB_DIR, settings=settings)

        done = {k for k, r in log.latest().items()
                if r["status"] in FINISHED and r.get("judge_version") == settings.judge_version}
        todo = [c for c in sample if (c["prompt_id"], c["tier"]) not in done]
        if limit:
            todo = todo[:limit]
        runner.say(f"pid {os.getpid()}: {len(done)} done, judging {len(todo)} with {settings.model_name}")

        for item in todo:
            pid, tier = item["prompt_id"], item["tier"]
            prompt = dict(prompts[pid])
            prompt.setdefault("constraints", [])
            candidate, reference = answers[(pid, tier)]["answer"], answers[(pid, "high")]["answer"]
            try:
                result, usage = runner.real_judge_call(prompt, candidate, reference)
            except ProviderError as error:
                log.append(make_record(item, "pending", settings,
                                       error={"kind": error.kind, "message": str(error)[:300]}))
                runner.say(f"{pid} {tier}: provider error ({error.kind}); will retry on the next run")
                if error.kind == "rate_limit":
                    if DAILY_QUOTA_HINT.search(str(error)):
                        runner.wait_logged(DAILY_QUOTA_BACKOFF_SECONDS, "daily-quota error")
                    else:
                        time.sleep(60)
                continue
            status = "scored" if result.status == "scored" else "error"
            log.append(make_record(item, status, settings, result=result, usage=usage))
            verdict = "pass" if result.passed else "fail"
            truth = "right" if item["truth_pass"] else "wrong"
            runner.say(f"{pid} {tier}: truly {truth}, judge {verdict if status == 'scored' else 'ERROR'}"
                       + (f" (score {result.score})" if result.score is not None else ""))
        runner.say("done for now; rerun to retry anything pending")
    finally:
        if running_pid(pid_path) == os.getpid():
            pid_path.unlink()


def show_status() -> None:
    if not SAMPLE_PATH.exists():
        print("no sample yet")
        return
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))["items"]
    print(f"sample: {len(sample)} answers\n{describe(sample)}")
    version = load_judge_settings().judge_version
    latest = {k: r for k, r in JudgmentLog(RESULTS_PATH).latest().items() if r.get("judge_version") == version}
    statuses = Counter(latest.get((c["prompt_id"], c["tier"]), {}).get("status", "not started") for c in sample)
    print("\nprogress: " + ", ".join(f"{s} {n}" for s, n in sorted(statuses.items())))
    scored = [r for r in latest.values() if r["status"] == "scored"]
    if scored:   # a quick look only; eval/judge_calibration.py does the real analysis
        agree = sum(r["judge_pass"] == r["truth_pass"] for r in scored)
        lenient = sum(r["judge_pass"] and not r["truth_pass"] for r in scored)
        print(f"so far: judge agrees on {agree}/{len(scored)}, passed {lenient} truly wrong answers")


def main():
    parser = argparse.ArgumentParser(description="Judge a sample of exact-match answers for calibration.")
    parser.add_argument("command", choices=["sample", "run", "status"])
    parser.add_argument("--size", type=int, default=SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="redraw an existing sample")
    parser.add_argument("--limit", type=int, default=0, help="judge at most this many in this run")
    args = parser.parse_args()
    if args.command == "sample":
        make_sample(args.size, args.seed, args.force)
    elif args.command == "run":
        run(args.limit)
    else:
        show_status()


if __name__ == "__main__":
    main()