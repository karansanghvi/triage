"""Phase 4 Step 4: run the AI checker on the TEST answers for the one final Phase 4 evaluation.

This is the test set's second use (the first was Phase 3's final check, reports/FINAL_TEST_DONE.json). The
strategy was chosen on validation (reports/pareto_val.json, reports/price_sensitivity_val.json) before any
test-set check was made, and is fixed here as STRATEGY:

    medium_then_check_selective, skip_above 0.90, min_answer_tokens 0, on_error escalate, checker groq-small v3

Usage:
    python -m eval.checker_test_run --final --dry-run        # count the calls and show the budget; no API calls
    python -m eval.checker_test_run --final                  # the real run (resumable: rerun to continue)
Long runs:
    nohup caffeinate -i ./venv/bin/python -m eval.checker_test_run --final \\
        >> data/checker_test/run_groq-small.log 2>&1 &

What gets checked: only what the strategy can ever ask for. That means a medium answer on a labeled test
prompt that exists, passes the free format checker, and has classifier v2's p_medium < skip_above (0.90).
Format failures escalate for free, and confident answers are accepted without a check.

Guards:
  * --final is required, even for --dry-run.
  * The first real run writes reports/PHASE4_PREREGISTRATION.json: the strategy, the checker's model,
    version and prompt hash, the classifier, and the exact list of items to check. Every later run (a
    resume) must match it, or it refuses.
  * Once the scoring script (eval/final_test_phase4.py) has written reports/FINAL_TEST_PHASE4_DONE.json,
    this refuses to run at all. Checking more after seeing the result would turn the test set into tuning
    data.
  * This never reads test outcomes (pass/fail). It reads only which test prompts are labeled, the prompts,
    medium's answers and the classifier's probabilities.
  * Results go to data/checker_test/, never to data/checker_val/.

Pacing: the same Pacer as the validation run. It counts every earlier use of the model: Phase 2 answers, the
judge, the calibration logs, the validation log and this log. It refuses to start while a calibration or
validation run for the same model is running.
"""
import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.registry import PROJECT_ROOT, load_registry
from app.verifier.ai_checker import load_checker_settings, prompt_sha256
from app.verifier.checker import check_answer
from eval.build_dataset import FULL_DIR, Pacer, load_collection_config, load_prompts, running_pid
from eval.checker_calibration import CALIB_DIR, check_log_usage, usage_history
from eval import checker_val_run as val_run
from eval.routing_sim import OUTCOMES_PATH
from eval.score_answers import load_answers
from eval.splits import load_split

TEST_DIR = PROJECT_ROOT / "data" / "checker_test"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREREG_PATH = REPORTS_DIR / "PHASE4_PREREGISTRATION.json"
FINAL_MARKER = REPORTS_DIR / "FINAL_TEST_PHASE4_DONE.json"
CLASSIFIER = ("classifier_c_logreg_cv", 2)
STRATEGY = {"name": "medium_then_check_selective", "skip_above": 0.90, "min_answer_tokens": 0,
            "on_error": "escalate", "checker_model": "groq-small", "checker_version": 3}


class FinalRunRefused(SystemExit):
    pass


def log_path(model: str, test_dir: Path = TEST_DIR) -> Path:
    return Path(test_dir) / f"checks_{model}.jsonl"


def labeled_test_ids(outcomes_path: Path = OUTCOMES_PATH) -> list:
    """The labeled test prompts (the same set routing_sim.load_eval_rows("test") scores), read WITHOUT their
    pass/fail: only id and label_status are looked at."""
    _, test_ids = load_split()
    ids = set()
    for line in Path(outcomes_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            if o.get("label_status") == "labeled" and o["id"] in test_ids:
                ids.add(o["id"])
    return sorted(ids)


def test_probabilities(ids: list) -> dict:
    """{id: (p_low, p_medium)} from classifier v2 (trained on the fit set: no test prompt), hand task types."""
    from types import SimpleNamespace
    from eval.pareto import predictor_probabilities
    return predictor_probabilities(*CLASSIFIER, [SimpleNamespace(id=i) for i in ids])


def strategy_items(ids: list, prompts: dict, answers: dict, probs: dict, skip_above: float = STRATEGY["skip_above"]):
    """([{prompt_id, tier: "medium"}], skipped Counter): the medium answers the strategy would send to the checker.
    The order matches the strategy: format first (a failure escalates for free), then the confidence skip."""
    items, skipped = [], Counter()
    for pid in sorted(ids):
        a = answers.get((pid, "medium"))
        if not a or not a.get("answer"):
            skipped["no medium answer (escalates)"] += 1
            continue
        fmt = check_answer(prompts[pid]["messages"], a["answer"], truncated=bool(a.get("truncated")),
                           error=a.get("error"))
        if not fmt.ok:
            skipped["format checker fails it (escalates, no check)"] += 1
            continue
        if probs[pid][1] >= skip_above:
            skipped[f"p_medium >= {skip_above:.2f} (accepted, no check)"] += 1
            continue
        items.append({"prompt_id": pid, "tier": "medium"})
    return items, skipped


def preregistration(items: list, settings) -> dict:
    return {"strategy": STRATEGY, "checker_prompt_sha256": prompt_sha256(),
            "checker_model": settings.model_name, "checker_version": settings.checker_version,
            "classifier": f"{CLASSIFIER[0]} v{CLASSIFIER[1]}", "items": sorted(c["prompt_id"] for c in items),
            "n_items": len(items)}


def check_guards(items: list, settings, *, prereg_path: Path = PREREG_PATH, marker: Path = FINAL_MARKER,
                 real: bool) -> dict:
    """Refuse if the final evaluation already ran, if the checker isn't the pre-registered one, or (on a resume)
    if anything differs from the pre-registration. Returns the pre-registration for this run."""
    if Path(marker).exists():
        raise FinalRunRefused(f"the Phase 4 final test evaluation already ran ({Path(marker).read_text().strip()}); "
                              "checking more test answers after seeing the result turns the test set into tuning data")
    if (settings.model_name, settings.checker_version) != (STRATEGY["checker_model"], STRATEGY["checker_version"]):
        raise FinalRunRefused(f"the checker is {settings.model_name} v{settings.checker_version}, but the "
                              f"pre-registered strategy uses {STRATEGY['checker_model']} v{STRATEGY['checker_version']}")
    prereg = preregistration(items, settings)
    if Path(prereg_path).exists():
        saved = json.loads(Path(prereg_path).read_text(encoding="utf-8"))
        saved_core = {k: v for k, v in saved.items() if k != "registered_at"}
        if saved_core != prereg:
            diff = sorted(k for k in prereg if saved_core.get(k) != prereg[k])
            raise FinalRunRefused(f"this run differs from the pre-registration in {prereg_path.name}: {diff}")
        return saved
    if real:
        prereg = {**prereg, "registered_at": datetime.now(timezone.utc).isoformat()}
        Path(prereg_path).parent.mkdir(parents=True, exist_ok=True)
        with open(prereg_path, "x", encoding="utf-8") as f:              # exclusive create: never overwritten
            f.write(json.dumps(prereg, indent=2) + "\n")
    return prereg


def model_history(model: str, test_dir: Path = TEST_DIR) -> list:
    """Every earlier use of the model's quota, including the validation run and this run's own log."""
    return (usage_history(model) + check_log_usage(val_run.log_path(model), model)
            + check_log_usage(log_path(model, test_dir), model))


def load_inputs() -> tuple:
    ids = labeled_test_ids()
    wanted = set(ids)
    prompts = {p["id"]: p for p in load_prompts() if p["id"] in wanted}
    answers = {k: v for k, v in load_answers(FULL_DIR / "answers.jsonl").items() if k[0] in wanted and k[1] == "medium"}
    probs = test_probabilities(ids)
    items, skipped = strategy_items(ids, prompts, answers, probs)
    return ids, prompts, answers, probs, items, skipped


def dry_run(model: str, ids, prompts, answers, items, skipped, settings, test_dir: Path = TEST_DIR) -> dict:
    out = val_run.dry_run(model, items, skipped, prompts, answers, len(ids), val_dir=test_dir, settings=settings)
    out.pop("validation_prompts", None)
    limits = load_collection_config(load_registry()).limits[model]
    calls, tokens = Pacer(limits, model_history(model, test_dir))._day_usage()
    return {"split": "test (Phase 4 final; second use of the test set)", "strategy": STRATEGY,
            "labeled_test_prompts": len(ids), "medium_answers_the_strategy_checks": len(items),
            **out,
            "budget_used_last_24h": {"tokens": tokens, "calls": calls, "tpd": limits.tpd,
                                     "note": "every Triage run on this model counts (Phase 2, judge, calibration, "
                                             "validation and test checks)"},
            "preregistration": {"path": str(PREREG_PATH.relative_to(PROJECT_ROOT)), "exists": PREREG_PATH.exists(),
                                "note": "written by the first real run; resumes must match it"}}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 4 final: run the AI checker on the test answers.")
    parser.add_argument("--final", action="store_true", help="required: this is the one Phase 4 test-set run")
    parser.add_argument("--dry-run", action="store_true", help="only count the calls and show the budget")
    parser.add_argument("--no-wait", action="store_true", help="stop instead of waiting when the daily budget is used")
    args = parser.parse_args(argv)
    if not args.final:
        raise SystemExit("this runs the checker on the TEST set; pass --final (with --dry-run to only count)")
    settings = load_checker_settings()
    model = settings.model_name
    ids, prompts, answers, probs, items, skipped = load_inputs()
    check_guards(items, settings, real=not args.dry_run)
    if args.dry_run:
        print(json.dumps(dry_run(model, ids, prompts, answers, items, skipped, settings), indent=2))
        return
    for other in (CALIB_DIR / f"run_{model}.pid", val_run.VAL_DIR / f"run_{model}.pid"):
        pid = running_pid(other)
        if pid:
            raise SystemExit(f"another run for {model} is running (pid {pid}, {other.name}); they would share one "
                             "daily budget. Wait for it to finish.")
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    pid_path = TEST_DIR / f"run_{model}.pid"
    other = running_pid(pid_path)
    if other and other != os.getpid():
        raise SystemExit(f"another test run for {model} is already running (pid {other})")
    pid_path.write_text(str(os.getpid()))
    try:
        pacer = Pacer(load_collection_config(load_registry()).limits[model], model_history(model))
        print(val_run.run(model, items, prompts, answers, wait=not args.no_wait, pacer=pacer, val_dir=TEST_DIR,
                          settings=settings, split_name="test"))
    except KeyboardInterrupt:
        raise SystemExit("\nstopped. Finished checks are saved; rerun the same command to continue.") from None
    finally:
        if running_pid(pid_path) == os.getpid():
            pid_path.unlink()


if __name__ == "__main__":
    main()
