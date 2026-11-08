"""Phase 4 Step 0: how often guess_task_type() agrees with the hand-tagged task type.

Usage:
    python -m eval.guesser_agreement --label baseline
    python -m eval.guesser_agreement --label option_d --compare-to baseline
    python -m eval.guesser_agreement --split val --label option_d_val --baseline-rev c61a92a   # ONCE

Splits (the test set is never read):
    fit   train_ids.txt - val_ids.txt (495 prompts). Unlabeled train prompts are included, because agreement
          needs only the hand tag in data/prompts.jsonl, not an outcome label. Keyword tuning happens here.
    val   the 117 validation prompts. Confirmation only: a val run can't be repeated under the same label,
          and its results must not be used to change the keywords.

--baseline-rev also scores the guesser as it was at that git revision (read with `git show`, nothing
checked out), so one run can report old rules and new rules on the same prompts.

Read-only with respect to data/: the only file written is reports/guesser_agreement.json, which keeps one
entry per --label (agreement, per-task accuracy, confusion matrix and every prompt's guess), so a later run
can list exactly which prompts' guesses changed.
"""
import argparse
import json
import subprocess
import types
from collections import Counter
from datetime import datetime, timezone

from app.registry import PROJECT_ROOT
from app.router import features
from app.router.features import TASK_TYPES, guess_task_type, normalize_task_type, prompt_text
from eval.build_dataset import load_prompts
from eval.splits import load_split, load_validation_split

REPORT_PATH = PROJECT_ROOT / "reports" / "guesser_agreement.json"
SPLITS = ("fit", "val")


def agreement_prompts(split: str = "fit") -> list:
    """Hand-tagged prompts in train - val ("fit") or in val, sorted by ID. Never the test set."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    train, test = load_split()
    val = load_validation_split()
    ids = train - val if split == "fit" else val
    assert not ids & test
    return sorted((p for p in load_prompts() if p["id"] in ids and normalize_task_type(p.get("task_type"))),
                  key=lambda p: p["id"])


def guesser_at_revision(rev: str):
    """guess_task_type as it was in app/router/features.py at a git revision (read-only; nothing checked out)."""
    source = subprocess.run(["git", "show", f"{rev}:app/router/features.py"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout
    module = types.ModuleType(f"features_at_{rev}")
    exec(compile(source, f"features.py@{rev}", "exec"), module.__dict__)
    return module.guess_task_type, module.FEATURE_VERSION


def trigger(text: str, task: str):
    """The text that made the guesser pick `task` (None for the "qa" fallback or if rules aren't exposed)."""
    for rule_task, pattern in getattr(features, "GUESS_RULES", []):
        if rule_task == task:
            m = pattern.search(text.lower())
            return m.group(0) if m else None
    return None


def measure(prompts: list, guess=guess_task_type, feature_version: str = None) -> dict:
    guesses, confusion = {}, Counter()
    for p in prompts:
        hand, g = normalize_task_type(p["task_type"]), guess(prompt_text(p))
        guesses[p["id"]] = {"hand": hand, "guess": g}
        confusion[(hand, g)] += 1
    agree = sum(g["hand"] == g["guess"] for g in guesses.values())
    per_task = {}
    for t in TASK_TYPES:
        n = sum(g["hand"] == t for g in guesses.values())
        if n:
            per_task[t] = {"n": n, "correct": confusion[(t, t)], "recall": confusion[(t, t)] / n,
                           "guessed": sum(g["guess"] == t for g in guesses.values())}
    return {
        "n": len(prompts), "agree": agree, "agreement": agree / len(prompts) if prompts else None,
        "feature_version": feature_version or features.FEATURE_VERSION,
        "per_task": per_task,
        "confusion": {h: {g: confusion[(h, g)] for g in TASK_TYPES if confusion[(h, g)]} for h in TASK_TYPES
                      if any(confusion[(h, g)] for g in TASK_TYPES)},
        "guesses": guesses,
    }


def print_confusion(result: dict) -> None:
    short = {t: t[:6] for t in TASK_TYPES}
    print(f"\nconfusion (rows = hand tag, columns = guess)\n{'':<15}" + "".join(f"{short[t]:>8}" for t in TASK_TYPES)
          + f"{'recall':>9}")
    for h in TASK_TYPES:
        row = result["confusion"].get(h)
        if not row:
            continue
        print(f"{h:<15}" + "".join(f"{row.get(g, 0) or '.':>8}" for g in TASK_TYPES)
              + f"{result['per_task'][h]['recall']:>9.0%}")


def print_changes(prompts: list, before: dict, after: dict) -> list:
    text = {p["id"]: prompt_text(p) for p in prompts}
    changed = []
    for pid, now in after["guesses"].items():
        old = before["guesses"].get(pid)
        if old and old["guess"] != now["guess"]:
            changed.append({"id": pid, "hand": now["hand"], "old": old["guess"], "new": now["guess"],
                            "trigger": trigger(text[pid], now["guess"]),
                            "fixed": now["guess"] == now["hand"], "broke": old["guess"] == now["hand"],
                            "preview": " ".join(text[pid].split())[:80]})
    print(f"\n{len(changed)} prompts changed guess "
          f"({sum(c['fixed'] for c in changed)} now correct, {sum(c['broke'] for c in changed)} now wrong, "
          f"{sum(not c['fixed'] and not c['broke'] for c in changed)} wrong before and after)")
    for c in changed:
        mark = "FIXED" if c["fixed"] else ("BROKE" if c["broke"] else "still wrong")
        print(f"  {c['id']:<28} hand={c['hand']:<15} {c['old']} -> {c['new']} [{c['trigger']!r}] {mark}")
    return changed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Task-type guesser agreement on train - val, or val (read-only).")
    parser.add_argument("--label", required=True, help="name for this run in reports/guesser_agreement.json")
    parser.add_argument("--split", default="fit", choices=SPLITS)
    parser.add_argument("--compare-to", default=None, help="an earlier --label to list changed guesses against")
    parser.add_argument("--baseline-rev", default=None,
                        help="also score the guesser from this git revision on the same prompts")
    args = parser.parse_args(argv)

    report = json.loads(REPORT_PATH.read_text()) if REPORT_PATH.exists() else {"runs": {}}
    if args.split == "val" and args.label in report["runs"]:
        raise SystemExit(f"run {args.label!r} already exists; validation runs are one-time confirmations")

    prompts = agreement_prompts(args.split)
    result = {"split": args.split, **measure(prompts)}
    print(f"{result['n']} hand-tagged {args.split} prompts "
          f"({'train split minus validation' if args.split == 'fit' else 'validation'}; test never read)")

    if args.baseline_rev:
        old_guess, old_version = guesser_at_revision(args.baseline_rev)
        base = measure(prompts, old_guess, old_version)
        result["baseline_rules"] = {"git_rev": args.baseline_rev, **base}
        print(f"baseline rules (git {args.baseline_rev}, FEATURE_VERSION {old_version}): "
              f"{base['agree']}/{base['n']} = {base['agreement']:.1%}")
        print_confusion(base)
        result["changed_vs"] = {"label": f"baseline_rules@{args.baseline_rev}",
                                "prompts": print_changes(prompts, base, result)}

    print(f"\ncurrent rules (FEATURE_VERSION {result['feature_version']}): "
          f"{result['agree']}/{result['n']} = {result['agreement']:.1%}")
    print_confusion(result)

    if args.compare_to:
        if args.compare_to not in report["runs"]:
            raise SystemExit(f"no run labelled {args.compare_to!r} in {REPORT_PATH.name}")
        before = report["runs"][args.compare_to]
        print(f"\nvs {args.compare_to!r}: {before['agreement']:.1%} -> {result['agreement']:.1%}")
        result["changed_vs"] = {"label": args.compare_to, "prompts": print_changes(prompts, before, result)}

    report["description"] = ("guess_task_type() vs hand-tagged task_type, on train_ids - val_ids ('fit') or "
                             "val_ids ('val') (eval/guesser_agreement.py); test set never read")
    report["runs"][args.label] = {**result, "created": datetime.now(timezone.utc).isoformat()}
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    shown = REPORT_PATH.relative_to(PROJECT_ROOT) if REPORT_PATH.is_relative_to(PROJECT_ROOT) else REPORT_PATH
    print(f"\nsaved run {args.label!r} to {shown}")


if __name__ == "__main__":
    main()
