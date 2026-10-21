"""Phase 2 Step 4: combine automatic grades and judge grades into one outcome and label per prompt.

Usage:
    python -m eval.build_outcomes

Reads data/prompts.jsonl, data/scores.jsonl (automatic grades) and data/judgments/judgments.jsonl (judge
grades; may be missing or partial), all read-only, and writes data/outcomes.jsonl from scratch: one line per
prompt, in prompts.jsonl order. The same inputs always give the same output.

Label rule (the routing label = the cheapest tier that passes):
    low passed -> "low"; else medium passed -> "medium"; else "high" (also when no tier passed: none_passed).
The extra tier (llama) is recorded but never affects the label. A prompt is excluded (no_reference) if the
judge had no usable reference, and incomplete if a tier the rule needs is still pending or errored; both get
label null.
"""
import json
import sys
from collections import Counter
from pathlib import Path

from app.registry import PROJECT_ROOT
from eval.build_dataset import load_prompts

SCORES_PATH = PROJECT_ROOT / "data" / "scores.jsonl"
JUDGMENTS_PATH = PROJECT_ROOT / "data" / "judgments" / "judgments.jsonl"
OUTCOMES_PATH = PROJECT_ROOT / "data" / "outcomes.jsonl"
TIERS = ("low", "medium", "high", "extra")
LABEL_TIERS = ("low", "medium", "high")
PRE_FAIL_REASONS = {"truncated", "empty", "input too large"}
UNRESOLVED = {"pending", "error"}


# ---------- One tier ----------

def tier_from_score(row: dict) -> dict:
    """A tier's outcome from data/scores.jsonl (automatically graded prompts)."""
    status = row["status"]
    if status == "fail" and row["reason"] in PRE_FAIL_REASONS:
        method = "pre_failed"
    elif status in ("pass", "fail"):
        method = row["scorer"]
    else:
        method = "waiting"
    return {"status": status if status != "needs_judge" else "pending", "method": method,
            "reason": row["reason"], "judge_score": None}


def tier_from_judgment(record: dict) -> dict:
    """A tier's outcome from data/judgments/judgments.jsonl (judge-scored prompts)."""
    method = "assumed" if record["status"] == "assumed_pass" else record["method"]
    return {"status": record["status"], "method": method, "reason": record["reason"],
            "judge_score": record["score"] if record["method"] == "judged" else None}


def judge_prompt_tiers(scores: dict, judgments: dict) -> dict:
    """Outcomes for a judge-scored prompt. Tiers without a judgment yet are pending; the extra tier is
    never judged (option C), so it is only failed (pre-failed) or not_judged."""
    out = {}
    no_reference = any(judgments.get(t, {}).get("status") == "no_reference" for t in LABEL_TIERS)
    for tier in LABEL_TIERS:
        if tier in judgments:
            out[tier] = tier_from_judgment(judgments[tier])
        else:
            out[tier] = {"status": "pending", "method": "needs_judge", "reason": "not judged yet", "judge_score": None}
    extra = scores.get("extra")
    if no_reference:
        out["extra"] = {"status": "no_reference", "method": "no_reference", "reason": "no usable reference", "judge_score": None}
    elif extra and extra["status"] == "fail" and extra["reason"] in PRE_FAIL_REASONS:
        out["extra"] = tier_from_score(extra)
    elif extra and extra["status"] == "pending":
        out["extra"] = tier_from_score(extra)
    else:
        out["extra"] = {"status": "not_judged", "method": "not judged (option C)", "reason": "", "judge_score": None}
    return out


# ---------- Label ----------

def passed(tier: dict):
    """True, False, or None when the outcome isn't known (pending, error, no_reference)."""
    if tier["status"] in ("pass", "assumed_pass"):
        return True
    if tier["status"] == "fail":
        return False
    return None


def label_for(tiers: dict) -> tuple:
    """(label, label_status, none_passed) from the low, medium and high outcomes."""
    if any(tiers[t]["status"] == "no_reference" for t in LABEL_TIERS):
        return None, "excluded_no_reference", False
    for tier in LABEL_TIERS:                       # walk low -> medium -> high, stopping at the first pass
        result = passed(tiers[tier])
        if result is None:
            return None, "incomplete", False
        if result:
            return tier, "labeled", False
    return "high", "labeled", True                 # nobody passed: still labeled high


def build_outcome(prompt: dict, scores: dict, judgments: dict) -> dict:
    """The outcome line for one prompt. scores and judgments map tier -> record for this prompt."""
    if prompt["scoring"] == "judge":
        tiers = judge_prompt_tiers(scores, judgments)
    else:
        tiers = {}
        for tier in TIERS:
            row = scores.get(tier)
            tiers[tier] = tier_from_score(row) if row else {"status": "pending", "method": "waiting",
                                                             "reason": "not scored yet", "judge_score": None}
    label, label_status, none_passed = label_for(tiers)
    return {
        "id": prompt["id"],
        "source": prompt["source"],
        "task_type": prompt["task_type"],
        "scoring": prompt["scoring"],
        "difficulty_guess": prompt["difficulty_guess"],
        "messy": prompt.get("messy"),
        "tiers": tiers,
        "low_pass": passed(tiers["low"]),
        "low_pass_assumed": False,
        "medium_pass": passed(tiers["medium"]),
        "medium_pass_assumed": tiers["medium"]["status"] == "assumed_pass",
        "high_pass": passed(tiers["high"]),
        "label": label,
        "label_status": label_status,
        "none_passed": none_passed,
    }


# ---------- Whole dataset ----------

def read_jsonl(path: Path) -> list:
    """Records from a JSONL file; a half-written final line (a writer mid-append) is ignored."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines):
                raise
    return out


def index(records: list) -> dict:
    """prompt_id -> tier -> newest record."""
    out = {}
    for record in records:
        out.setdefault(record["prompt_id"], {})[record["tier"]] = record
    return out


def build_all(prompts: list, score_rows: list, judgment_records: list) -> tuple:
    """(outcome lines in prompts order, warnings)."""
    ids = [p["id"] for p in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError("prompts.jsonl has duplicate IDs")
    known = set(ids)
    warnings = []
    for name, records in (("scores.jsonl", score_rows), ("judgments.jsonl", judgment_records)):
        unknown = sorted({r["prompt_id"] for r in records} - known)
        if unknown:
            warnings.append(f"{name}: {len(unknown)} prompt IDs not in prompts.jsonl: {', '.join(unknown[:10])}"
                            + (" ..." if len(unknown) > 10 else ""))
    scores, judgments = index(score_rows), index(judgment_records)
    outcomes = [build_outcome(p, scores.get(p["id"], {}), judgments.get(p["id"], {})) for p in prompts]
    assert len(outcomes) == len(prompts) and [o["id"] for o in outcomes] == ids
    return outcomes, warnings


def write_outcomes(outcomes: list, path: Path = OUTCOMES_PATH) -> None:
    path.write_text("".join(json.dumps(o, ensure_ascii=False, sort_keys=True) + "\n" for o in outcomes),
                    encoding="utf-8")


# ---------- Report ----------

def print_breakdown(title: str, labeled: list, key) -> None:
    print(f"\n{title}")
    print(f"{'':<16}{'low':>8}{'medium':>8}{'high':>8}{'total':>8}   (none_passed)")
    for group in sorted({key(o) for o in labeled}, key=str):
        rows = [o for o in labeled if key(o) == group]
        counts = Counter(o["label"] for o in rows)
        print(f"{str(group):<16}{counts['low']:>8}{counts['medium']:>8}{counts['high']:>8}{len(rows):>8}"
              f"   ({sum(o['none_passed'] for o in rows)})")


def main():
    prompts = load_prompts()
    outcomes, warnings = build_all(prompts, read_jsonl(SCORES_PATH), read_jsonl(JUDGMENTS_PATH))
    write_outcomes(outcomes)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    print(f"Wrote {len(outcomes)} outcomes to {OUTCOMES_PATH.relative_to(PROJECT_ROOT)}"
          f" (judgments file {'found' if JUDGMENTS_PATH.exists() else 'not found yet'})")
    status = Counter(o["label_status"] for o in outcomes)
    print("\nlabel_status: " + ", ".join(f"{k} {v}" for k, v in sorted(status.items())))
    labeled = [o for o in outcomes if o["label_status"] == "labeled"]
    print_breakdown("Labels (labeled prompts)", labeled, lambda o: "all")
    print_breakdown("By source", labeled, lambda o: o["source"])
    print_breakdown("By task_type", labeled, lambda o: o["task_type"])
    print_breakdown("By difficulty_guess", labeled, lambda o: o["difficulty_guess"])
    print(f"\nnone_passed: {sum(o['none_passed'] for o in outcomes)} (labeled high)")


if __name__ == "__main__":
    main()
