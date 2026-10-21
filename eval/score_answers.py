"""Phase 2 Step 3: grade the collected answers with the automatic (non-judge) scorers.

Usage:
    python -m eval.score_answers

Reads data/answers/answers.jsonl (read-only; the collector may still be appending to it) and writes
data/scores.jsonl, one line per (prompt_id, tier), from scratch every run. The same answers always
give the same scores.

Each pair gets a status:
    pass / fail    graded by the prompt's scorer (exact_match, field_check, label_match, final_answer)
    fail           also for truncated or empty answers, and inputs the model can never accept
    needs_judge    answered, but the prompt is judge-scored (graded in a later step)
    pending        not answered yet (or the last attempt hit an error that will be retried)
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from app.registry import PROJECT_ROOT, load_registry
from app.verifier.scorers import score_record
from eval.build_dataset import FULL_DIR, load_collection_config, load_prompts

ANSWERS_PATH = FULL_DIR / "answers.jsonl"
SCORES_PATH = PROJECT_ROOT / "data" / "scores.jsonl"
REVIEW_PATH = PROJECT_ROOT / "data" / "scores_review.txt"
AUTO_SCORINGS = {"exact_match", "field_check", "label_match", "final_answer"}


# ---------- Grading one pair ----------

def is_input_too_large(error: dict) -> bool:
    """A permanent rejection: the prompt alone is bigger than the model will ever accept (e.g. Groq 413)."""
    message = error.get("message", "").lower()
    return error.get("kind") == "client" and ("request too large" in message or "error code: 413" in message)


def score_pair(prompt: dict, record: Optional[dict]) -> dict:
    """Grade one prompt's newest answer from one tier. Returns status, scorer, graded value and reason."""
    scoring = prompt["scoring"]
    scorer = scoring if scoring in AUTO_SCORINGS else None
    if record is None:
        return {"status": "pending", "scorer": scorer, "graded": None, "reason": "not answered yet"}
    if record["error"]:
        if is_input_too_large(record["error"]):
            return {"status": "fail", "scorer": scorer, "graded": None, "reason": "input too large"}
        return {"status": "pending", "scorer": scorer, "graded": None,
                "reason": f"last attempt failed ({record['error']['kind']}); will be retried"}
    if record["truncated"]:
        return {"status": "fail", "scorer": scorer, "graded": None, "reason": "truncated"}
    if not (record["answer"] or "").strip():
        return {"status": "fail", "scorer": scorer, "graded": None, "reason": "empty"}
    if scorer is None:
        return {"status": "needs_judge", "scorer": None, "graded": None, "reason": "judge-scored prompt"}
    result = score_record(prompt, record["answer"])
    return {"status": "pass" if result.passed else "fail", "scorer": scorer,
            "graded": result.extracted, "reason": result.reason}


# ---------- Whole dataset ----------

def load_answers(path: Path = ANSWERS_PATH) -> dict:
    """Newest record per (prompt_id, tier). A half-written final line (collector mid-write) is ignored."""
    latest = {}
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                continue
            raise
        latest[(record["prompt_id"], record["tier"])] = record
    return latest


def score_all(prompts: list, tiers: list, latest: dict) -> list:
    rows = []
    for prompt in prompts:
        for tier in tiers:
            record = latest.get((prompt["id"], tier))
            rows.append({
                "prompt_id": prompt["id"],
                "tier": tier,
                "model_id": record["model_id"] if record else None,
                "source": prompt["source"],
                "task_type": prompt["task_type"],
                "difficulty_guess": prompt["difficulty_guess"],
                **score_pair(prompt, record),
            })
    return rows


def write_scores(rows: list, path: Path = SCORES_PATH) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


# ---------- Report ----------

def rate(rows: list) -> str:
    """Pass rate over auto-graded answers. Judge-scored prompts are left out (their only fails so far
    are truncations, counted separately), so a 0% here always means real wrong answers."""
    graded = [r for r in rows if r["status"] in ("pass", "fail") and r["scorer"] is not None]
    if not graded:
        return "-"
    passed = sum(r["status"] == "pass" for r in graded)
    return f"{passed}/{len(graded)} ({passed / len(graded):.0%})"


def print_table(title: str, rows: list, tiers: list, key) -> None:
    groups = sorted({key(r) for r in rows if key(r) is not None})
    print(f"\n{title}")
    print(f"{'':<16}" + "".join(f"{t:>17}" for t in tiers))
    for group in groups:
        print(f"{group:<16}" + "".join(
            f"{rate([r for r in rows if r['tier'] == t and key(r) == group]):>17}" for t in tiers))


def suspicious(rows: list, label_tiers: list) -> list:
    """(prompt_id, why) where every label tier failed, or low and medium passed but high failed."""
    by_prompt = defaultdict(dict)
    for r in rows:
        by_prompt[r["prompt_id"]][r["tier"]] = r
    flagged = []
    for prompt_id, tiers in by_prompt.items():
        statuses = [tiers[t]["status"] for t in label_tiers]
        if all(s == "fail" for s in statuses):
            flagged.append((prompt_id, "every tier failed"))
        elif statuses == ["pass", "pass", "fail"]:
            flagged.append((prompt_id, "low and medium passed, high failed"))
    return flagged


def write_review(flagged: list, prompts: dict, latest: dict, rows: list, tiers: list, path: Path) -> None:
    by_key = {(r["prompt_id"], r["tier"]): r for r in rows}
    out = [f"Possible bad references: {len(flagged)} prompts\n"]
    for number, (prompt_id, why) in enumerate(flagged, start=1):
        prompt = prompts[prompt_id]
        user = next(m["content"] for m in reversed(prompt["messages"]) if m["role"] == "user")
        out += ["=" * 110, f"#{number} {prompt_id}  [{prompt['source']} / {prompt['task_type']} / "
                f"{prompt['scoring']}]  -- {why}", "-" * 110, user.strip(), "",
                f"REFERENCE: {json.dumps(prompt['reference'], ensure_ascii=False)}", ""]
        for tier in tiers:
            row, record = by_key[(prompt_id, tier)], latest.get((prompt_id, tier))
            answer = (record or {}).get("answer") or ""
            out += [f"--- {tier}: {row['status'].upper()} ({row['reason'][:160]})",
                    answer.strip()[-1500:] or "(no answer)", ""]
    path.write_text("\n".join(out), encoding="utf-8")


def report(rows: list, prompts: list, latest: dict, tiers: list, label_tiers: list) -> None:
    status_counts = Counter((r["tier"], r["status"]) for r in rows)
    print(f"Scored {len(rows)} (prompt, tier) pairs -> {SCORES_PATH.relative_to(PROJECT_ROOT)}\n")
    print(f"{'tier':<8}" + "".join(f"{s:>13}" for s in ("pass", "fail", "needs_judge", "pending")))
    for tier in tiers:
        print(f"{tier:<8}" + "".join(f"{status_counts[(tier, s)]:>13}" for s in ("pass", "fail", "needs_judge", "pending")))

    print_table("Pass rate (auto-graded answers only)", rows, tiers, lambda r: "all")
    print_table("By source", rows, tiers, lambda r: r["source"])
    print_table("By task_type", rows, tiers, lambda r: r["task_type"])
    order = {"easy": "1 easy", "medium": "2 medium", "hard": "3 hard"}
    print_table("Own prompts by difficulty_guess", rows, tiers,
                lambda r: order.get(r["difficulty_guess"]) if r["source"] == "own" else None)

    print("\nSpecial fails per tier")
    for reason in ("truncated", "empty", "input too large"):
        print(f"  {reason:<16}" + "".join(
            f"{t}={sum(1 for r in rows if r['tier'] == t and r['status'] == 'fail' and r['reason'] == reason):<5}"
            for t in tiers))

    by_id = {p["id"]: p for p in prompts}
    flagged = suspicious(rows, label_tiers)
    write_review(flagged, by_id, latest, rows, tiers, REVIEW_PATH)
    print(f"\nPossible bad references: {len(flagged)} -> {REVIEW_PATH.relative_to(PROJECT_ROOT)}")
    for prompt_id, why in flagged:
        print(f"  {prompt_id:<26} {why}")


def main():
    config = load_collection_config(load_registry())
    tiers = list(config.tiers)
    prompts = load_prompts()
    latest = load_answers()
    rows = score_all(prompts, tiers, latest)
    write_scores(rows)
    report(rows, prompts, latest, tiers, config.label_tiers)


if __name__ == "__main__":
    main()
