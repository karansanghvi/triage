"""Phase 4 Step 2, part 1: draw the AI-checker calibration sample (no API calls).

Usage:
    python -m eval.checker_calibration_sample sample          # draw and save the sample
    python -m eval.checker_calibration_sample status          # counts per stratum

The pool is every fit-set answer the AI checker would actually see, with a real label:
    * fit set only (train minus validation; test never touched), checked with assert_no_test_ids
    * low and medium answers only
    * real labels only: exact match (final_answer, exact_match, label_match, field_check) or judged;
      not assumed_pass (option C), not auto_high, not pre-failed (truncated, empty, input too large)
    * only answers that PASS the free format checker (checker.check_answer): in the pipeline the AI checker
      runs only after it
    * minus EXCLUDED_ANSWERS (the 7 Step 1 smoke / prompt-tuning answers) and EXCLUDED_PROMPTS (mmlu_0079,
      an ambiguous MMLU question from data/review_decisions.md)

Stratified by (tier, truth source, good/bad). PLAN gives how many to draw from each stratum ("all" = every
item). Within a stratum items are shuffled with a fixed seed. REPEAT_SIZE items (half truly bad, half truly good)
are also picked for the repeat-run check. The sample is saved once; redrawing needs --force.
Results go to data/checker_calibration/, never to outcomes.jsonl or data/checks/.
"""
import argparse
import json
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app.registry import PROJECT_ROOT
from app.verifier.agreement import comparable
from app.verifier.checker import check_answer
from eval.build_dataset import FULL_DIR, load_prompts
from eval.score_answers import load_answers
from eval.splits import assert_no_test_ids, fit_ids, load_split

OUTCOMES_PATH = PROJECT_ROOT / "data" / "outcomes.jsonl"
CALIB_DIR = PROJECT_ROOT / "data" / "checker_calibration"
SAMPLE_PATH = CALIB_DIR / "sample.json"
SEED = 42
TIERS = ("low", "medium")
EXACT_METHODS = {"final_answer", "exact_match", "label_match", "field_check"}
EXCLUDED_ANSWERS = {("gsm8k_0142", "medium"), ("gsm8k_0076", "medium"), ("mmlu_0111", "medium"),
                    ("arena_0031", "medium"), ("gsm8k_0014", "low"), ("gsm8k_0177", "low"),
                    ("gsm8k_0065", "low")}
EXCLUDED_PROMPTS = {"mmlu_0079"}
# (tier, truth_source, "bad"/"good") -> how many to draw; "all" = the whole stratum
PLAN = {
    ("medium", "exact", "bad"): "all", ("medium", "judged", "bad"): "all",
    ("medium", "exact", "good"): 100, ("medium", "judged", "good"): "all",
    ("low", "exact", "bad"): 25, ("low", "judged", "bad"): 25,
    ("low", "exact", "good"): 25, ("low", "judged", "good"): 25,
}
REPEAT_SIZE = 20


def stratum(item: dict) -> tuple:
    return item["tier"], item["truth_source"], "bad" if item["truth_bad"] else "good"


def stratum_key(key: tuple) -> str:
    return "/".join(key)


# ---------- Pool (pure functions, easy to test) ----------

def find_candidates(outcomes: list, prompts: dict, answers: dict, fit: set) -> tuple:
    """(candidates, dropped counts). outcomes must already be restricted to fit-set ids."""
    candidates, dropped = [], Counter()
    for o in outcomes:
        pid = o["id"]
        if pid not in fit or o.get("label_status") != "labeled":
            continue
        if pid in EXCLUDED_PROMPTS:
            dropped["excluded_prompt"] += 2
            continue
        for tier in TIERS:
            t = o["tiers"][tier]
            if (pid, tier) in EXCLUDED_ANSWERS:
                dropped["excluded_smoke"] += 1
                continue
            if t["status"] == "assumed_pass":
                dropped["assumed_pass"] += 1
                continue
            if t["method"] == "pre_failed":
                dropped["pre_failed"] += 1
                continue
            if t["status"] not in ("pass", "fail"):
                dropped[f"status_{t['status']}"] += 1
                continue
            if t["method"] in EXACT_METHODS:
                source = "exact"
            elif t["method"] == "judged":
                source = "judged"
            else:
                dropped[f"method_{t['method']}"] += 1
                continue
            answer = answers.get((pid, tier)) or {}
            messages = prompts[pid]["messages"]
            if not check_answer(messages, answer.get("answer"), truncated=bool(answer.get("truncated")),
                                error=answer.get("error")).ok:
                dropped["format_flagged"] += 1
                continue
            candidates.append({
                "prompt_id": pid, "tier": tier, "truth_source": source, "truth_bad": t["status"] == "fail",
                "truth_reason": t.get("reason", ""), "task_type": o["task_type"], "dataset": o["source"],
                "comparable": comparable(messages), "answer_model": answer.get("model_name"),
            })
    return candidates, dropped


def select_sample(candidates: list, plan: dict = PLAN, seed: int = SEED) -> list:
    """Stratified sample. Same candidates + same seed = same sample. Raises if a stratum is too small."""
    rng = random.Random(seed)
    groups = {}
    for c in sorted(candidates, key=lambda c: (c["prompt_id"], c["tier"])):
        groups.setdefault(stratum(c), []).append(c)
    picked = []
    for key in sorted(plan):
        items = list(groups.get(key, []))
        want = len(items) if plan[key] == "all" else plan[key]
        if want > len(items):
            raise ValueError(f"stratum {stratum_key(key)} has {len(items)} items, plan asks for {want}")
        rng.shuffle(items)
        picked += sorted(items[:want], key=lambda c: (c["prompt_id"], c["tier"]))
    return picked


def select_repeats(sample: list, size: int = REPEAT_SIZE, seed: int = SEED) -> list:
    """[(prompt_id, tier)] for the repeat-run check: size // 2 truly bad and the rest truly good, seeded."""
    rng = random.Random(seed + 1)
    bad = sorted((c["prompt_id"], c["tier"]) for c in sample if c["truth_bad"])
    good = sorted((c["prompt_id"], c["tier"]) for c in sample if not c["truth_bad"])
    return sorted(rng.sample(bad, size // 2) + rng.sample(good, size - size // 2))


def counts(items: list) -> dict:
    return {stratum_key(k): n for k, n in sorted(Counter(stratum(c) for c in items).items())}


# ---------- Commands ----------

def load_fit_outcomes(path: Path = OUTCOMES_PATH) -> tuple:
    """(fit-set outcomes, fit ids). Rows outside the fit set are skipped before their outcomes are read."""
    train, _ = load_split()
    fit = fit_ids(train)
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            if o["id"] in fit:
                rows.append(o)
    return rows, fit


def build_sample(seed: int = SEED) -> dict:
    outcomes, fit = load_fit_outcomes()
    prompts = {p["id"]: p for p in load_prompts()}
    candidates, dropped = find_candidates(outcomes, prompts, load_answers(FULL_DIR / "answers.jsonl"), fit)
    sample = select_sample(candidates, PLAN, seed)
    assert_no_test_ids({c["prompt_id"] for c in sample})
    assert {c["prompt_id"] for c in sample} <= fit
    return {
        "created": datetime.now(timezone.utc).isoformat(), "seed": seed,
        "plan": {stratum_key(k): v for k, v in sorted(PLAN.items())},
        "pool_counts": counts(candidates), "sample_counts": counts(sample), "dropped": dict(sorted(dropped.items())),
        "excluded_answers": sorted(map(list, EXCLUDED_ANSWERS)), "excluded_prompts": sorted(EXCLUDED_PROMPTS),
        "repeat_items": [list(x) for x in select_repeats(sample, REPEAT_SIZE, seed)],
        "items": sample,
    }


def make_sample(seed: int, force: bool, path: Path = SAMPLE_PATH) -> dict:
    if path.exists() and not force:
        raise SystemExit(f"{path} already exists. Redrawing the sample after seeing results is cherry-picking; "
                         "pass --force only if you have a real reason.")
    data = build_sample(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def show(data: dict) -> None:
    print(f"{'stratum':<22}{'pool':>6}{'sample':>8}")
    for key, n in data["pool_counts"].items():
        print(f"{key:<22}{n:>6}{data['sample_counts'].get(key, 0):>8}")
    print(f"{'total':<22}{sum(data['pool_counts'].values()):>6}{len(data['items']):>8}")
    print(f"repeat-run items: {len(data['repeat_items'])}; dropped: {data['dropped']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Draw the AI-checker calibration sample.")
    parser.add_argument("command", choices=["sample", "status"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--force", action="store_true", help="redraw an existing sample")
    args = parser.parse_args(argv)
    if args.command == "sample":
        data = make_sample(args.seed, args.force)
        print(f"saved {SAMPLE_PATH.relative_to(PROJECT_ROOT)}")
    else:
        if not SAMPLE_PATH.exists():
            raise SystemExit("no sample yet; run: python -m eval.checker_calibration_sample sample")
        data = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    show(data)


if __name__ == "__main__":
    main()
