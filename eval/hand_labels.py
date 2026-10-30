"""Phase 2 Step 6: blind hand labels for a sample of our own prompts.

Usage:
    python -m eval.hand_labels setup     # pick the 50 prompts and write the two files (won't overwrite labels)
    python -m eval.hand_labels check     # confirm labels.txt has exactly those 50 IDs, each low/medium/high
    python -m eval.hand_labels compare   # compare the hand labels with the outcome labels (data/outcomes.jsonl)

The labeler sees only each prompt's ID and messages: no reference, difficulty guess, notes,
constraints, scores or answers. Prompts whose outcomes were already seen (pilot 1, pilot 2, and the
ones reviewed by hand) are left out.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

from app.registry import PROJECT_ROOT
from eval.build_dataset import OWN_TASK_TYPES, PILOT2_DIR, PILOT_DIR, load_prompts

LABELS_DIR = PROJECT_ROOT / "data" / "hand_labels"
TO_LABEL_PATH = LABELS_DIR / "to_label.txt"
LABELS_PATH = LABELS_DIR / "labels.txt"
SEED = 42
SAMPLE_SIZE = 50
VALID_LABELS = {"low", "medium", "high"}
TIER_ORDER = ("low", "medium", "high")
OUTCOMES_PATH = PROJECT_ROOT / "data" / "outcomes.jsonl"
# Prompts whose outcomes were already reviewed by hand (besides everything in the pilots)
ALREADY_SEEN = {"own_extraction_014", "own_extraction_016", "own_extraction_027", "own_reasoning_025"}
ID_HEADER = re.compile(r"^=+ \d+/\d+ +(own_\w+) =+$")

LABELS_INSTRUCTIONS = """\
# Hand labels (blind). Read each prompt in to_label.txt, then fill in one label per line:
#
#   low     the cheapest tier (allam-2-7b, a small non-reasoning model) would answer it correctly
#   medium  it needs gpt-oss-20b, but low would get it wrong
#   high    it needs gpt-oss-120b; medium would get it wrong too
#
# i.e. the CHEAPEST tier you think would answer correctly. Format: the ID, a space, the label.
# A note can follow after "#", for example:
#   own_extraction_005 medium  # nested JSON with a null field
#
# Check your file with:  python -m eval.hand_labels check
"""


def seen_ids() -> set:
    """Every prompt that appeared in pilot 1 or pilot 2, plus the hand-reviewed ones."""
    seen = set(ALREADY_SEEN)
    for directory in (PILOT_DIR, PILOT2_DIR):
        path = directory / "answers.jsonl"
        if path.exists():
            seen |= {json.loads(line)["prompt_id"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    return seen


def select_prompts(prompts: list, seen: set, size: int = SAMPLE_SIZE, seed: int = SEED) -> list:
    """size own prompts, spread as evenly as possible over the six task types, in a shuffled order."""
    rng = random.Random(seed)
    types = list(OWN_TASK_TYPES)
    rng.shuffle(types)                               # which types get the leftover extra prompts
    base, extra = divmod(size, len(types))
    picked = []
    for position, task_type in enumerate(types):
        pool = [p for p in prompts if p["source"] == "own" and p["task_type"] == task_type and p["id"] not in seen]
        count = base + (1 if position < extra else 0)
        if len(pool) < count:
            raise ValueError(f"only {len(pool)} unseen {task_type} prompts; need {count}")
        picked += rng.sample(pool, count)
    rng.shuffle(picked)                              # mix task types so neighbours don't anchor each other
    return picked


def write_to_label(picked: list, path: Path = TO_LABEL_PATH) -> None:
    out = [f"Prompts to label ({len(picked)}). Only the prompt is shown; write labels in labels.txt.\n"]
    for number, prompt in enumerate(picked, start=1):
        out.append(f"===== {number}/{len(picked)}  {prompt['id']} =====")
        for message in prompt["messages"]:
            out += [f"[{message['role']}]", message["content"].strip(), ""]
    path.write_text("\n".join(out), encoding="utf-8")


def write_labels_template(picked: list, path: Path = LABELS_PATH) -> None:
    lines = [LABELS_INSTRUCTIONS] + [f"{prompt['id']} " for prompt in picked]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def ids_to_label(path: Path = TO_LABEL_PATH) -> list:
    return [m.group(1) for line in path.read_text(encoding="utf-8").splitlines() if (m := ID_HEADER.match(line))]


def check_labels(expected_ids: list, labels_text: str) -> tuple:
    """(problems, labels) for a labels.txt body: exactly the expected IDs, each with low/medium/high."""
    problems, labels = [], {}
    for number, raw in enumerate(labels_text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        prompt_id = parts[0]
        if prompt_id in labels:
            problems.append(f"line {number}: {prompt_id} is listed twice")
            continue
        if prompt_id not in expected_ids:
            problems.append(f"line {number}: {prompt_id} is not one of the prompts to label")
            continue
        if len(parts) == 1:
            problems.append(f"line {number}: {prompt_id} has no label yet")
            labels[prompt_id] = None
            continue
        if len(parts) > 2 or parts[1].lower() not in VALID_LABELS:
            problems.append(f"line {number}: {prompt_id} label must be low, medium or high, got {' '.join(parts[1:])!r}")
            labels[prompt_id] = None
            continue
        labels[prompt_id] = parts[1].lower()
    for prompt_id in expected_ids:
        if prompt_id not in labels:
            problems.append(f"{prompt_id} is missing from labels.txt")
    return problems, labels


# ---------- Comparison with outcomes ----------

def compare_labels(hand: dict, outcomes: dict) -> dict:
    """Hand labels against outcome labels. hand: id -> label; outcomes: id -> outcome line.

    "too high" means the hand label is a pricier tier than needed (money wasted if routed that way);
    "too low" means a cheaper tier than needed (that tier's answer actually failed). Prompts without an
    outcome label (excluded) are counted separately and left out of every rate.
    """
    rank = {tier: i for i, tier in enumerate(TIER_ORDER)}
    rows, unlabeled = [], []
    for prompt_id, guess in sorted(hand.items()):
        outcome = outcomes.get(prompt_id)
        if not outcome or outcome["label"] is None:
            unlabeled.append(prompt_id)
            continue
        actual = outcome["label"]
        direction = "match" if guess == actual else ("too high" if rank[guess] > rank[actual] else "too low")
        rows.append({"id": prompt_id, "task_type": outcome["task_type"], "hand": guess, "actual": actual,
                     "direction": direction})
    confusion = {h: {a: sum(r["hand"] == h and r["actual"] == a for r in rows) for a in TIER_ORDER}
                 for h in TIER_ORDER}
    return {"rows": rows, "unlabeled": unlabeled, "confusion": confusion,
            "counts": {d: sum(r["direction"] == d for r in rows) for d in ("match", "too high", "too low")}}


def print_comparison(result: dict) -> None:
    rows, counts = result["rows"], result["counts"]
    total = len(rows)
    print(f"Compared {total} hand labels with outcome labels"
          + (f" ({len(result['unlabeled'])} excluded, no outcome label: {', '.join(result['unlabeled'])})"
             if result["unlabeled"] else ""))
    if not total:
        return
    for direction, meaning in (("match", "same tier"), ("too high", "guessed a pricier tier than needed"),
                               ("too low", "guessed a cheaper tier, whose answer actually failed")):
        print(f"  {direction:<9}{counts[direction]:>4}  ({counts[direction] / total:.0%})  {meaning}")

    print("\nRows: hand label, columns: outcome label")
    print(f"{'':<10}" + "".join(f"{a:>8}" for a in TIER_ORDER) + f"{'total':>8}")
    for h in TIER_ORDER:
        line = result["confusion"][h]
        print(f"{h:<10}" + "".join(f"{line[a]:>8}" for a in TIER_ORDER) + f"{sum(line.values()):>8}")
    print(f"{'total':<10}" + "".join(f"{sum(result['confusion'][h][a] for h in TIER_ORDER):>8}" for a in TIER_ORDER)
          + f"{total:>8}")

    print(f"\n{'task_type':<16}{'match':>7}{'too high':>10}{'too low':>9}{'total':>7}")
    for task_type in sorted({r["task_type"] for r in rows}):
        group = [r for r in rows if r["task_type"] == task_type]
        c = {d: sum(r["direction"] == d for r in group) for d in ("match", "too high", "too low")}
        print(f"{task_type:<16}{c['match']:>7}{c['too high']:>10}{c['too low']:>9}{len(group):>7}")

    print("\nDisagreements:")
    for r in sorted(rows, key=lambda r: (r["direction"], r["id"])):
        if r["direction"] != "match":
            print(f"  {r['id']:<26} hand {r['hand']:<7} outcome {r['actual']:<7} ({r['direction']})")


def main():
    parser = argparse.ArgumentParser(description="Blind hand labels for own prompts.")
    parser.add_argument("command", choices=["setup", "check", "compare"])
    args = parser.parse_args()

    if args.command == "setup":
        if LABELS_PATH.exists():
            raise SystemExit(f"{LABELS_PATH.relative_to(PROJECT_ROOT)} already exists; not overwriting your labels")
        LABELS_DIR.mkdir(parents=True, exist_ok=True)
        picked = select_prompts(load_prompts(), seen_ids())
        write_to_label(picked)
        write_labels_template(picked)
        print(f"Wrote {len(picked)} prompts to {TO_LABEL_PATH.relative_to(PROJECT_ROOT)} and "
              f"{LABELS_PATH.relative_to(PROJECT_ROOT)}")
        return

    expected = ids_to_label()
    problems, labels = check_labels(expected, LABELS_PATH.read_text(encoding="utf-8"))
    done = sum(1 for v in labels.values() if v)
    print(f"{done}/{len(expected)} prompts labeled")
    for problem in problems:
        print(f"  {problem}")
    if problems:
        sys.exit(1)
    counts = {label: sum(v == label for v in labels.values()) for label in ("low", "medium", "high")}
    print(f"OK: every prompt has a label {counts}")
    if args.command == "compare":
        outcomes = {}
        for line in OUTCOMES_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip():
                outcome = json.loads(line)
                outcomes[outcome["id"]] = outcome
        print()
        print_comparison(compare_labels(labels, outcomes))


if __name__ == "__main__":
    main()
