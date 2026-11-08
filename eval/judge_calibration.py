"""Phase 3 Step 4: how much can the LLM judge be trusted?

judge_calibration_sample.py ran the judge, exactly as in Phase 2, on 150 GSM8K/MMLU answers whose real
pass/fail is known from exact match. This compares the two:

    agree          the judge's verdict matches the truth
    too lenient    the judge PASSED a truly wrong answer   (dangerous: bad answers look fine)
    too strict     the judge FAILED a truly right answer   (costly: good cheap answers look bad)

Also broken down by dataset, tier, and whether the reference (the strong model's answer, which the judge
compares against) was itself right; and repeated without the 2 ambiguous MMLU questions from the review.
The "what if the pass line were different" table is safe to read: the sample is train-split only.

Usage:
    python -m eval.judge_calibration
"""
import json
from collections import Counter

from app.registry import PROJECT_ROOT
from app.verifier.judge import load_judge_settings
from eval.run_judge import JudgmentLog

CALIB_DIR = PROJECT_ROOT / "data" / "judge_calibration"
REPORTS_DIR = PROJECT_ROOT / "reports"
AMBIGUOUS = {"mmlu_0030", "mmlu_0079"}   # from data/review_decisions.md


def load_results() -> list:
    version = load_judge_settings().judge_version
    latest = JudgmentLog(CALIB_DIR / "judgments.jsonl").latest().values()
    return [r for r in latest if r.get("judge_version") == version and r["status"] == "scored"]


def summary(records: list) -> dict:
    n = len(records)
    if not n:
        return {"n": 0}
    right = [r for r in records if r["truth_pass"]]
    wrong = [r for r in records if not r["truth_pass"]]
    judged_pass = [r for r in records if r["judge_pass"]]
    return {
        "n": n,
        "agreement": sum(r["judge_pass"] == r["truth_pass"] for r in records) / n,
        "too_lenient": sum(r["judge_pass"] for r in wrong) / len(wrong) if wrong else None,      # of wrong answers
        "too_strict": sum(not r["judge_pass"] for r in right) / len(right) if right else None,   # of right answers
        "pass_means_right": sum(r["truth_pass"] for r in judged_pass) / len(judged_pass) if judged_pass else None,
        "confusion": {"right_judged_pass": sum(r["judge_pass"] for r in right),
                      "right_judged_fail": sum(not r["judge_pass"] for r in right),
                      "wrong_judged_pass": sum(r["judge_pass"] for r in wrong),
                      "wrong_judged_fail": sum(not r["judge_pass"] for r in wrong)},
    }


def line(label: str, s: dict) -> str:
    if not s["n"]:
        return f"  {label:<34} (none)"
    pct = lambda v: "  -  " if v is None else f"{v:5.0%}"
    return (f"  {label:<34} n={s['n']:<4} agree {pct(s['agreement'])}   too lenient {pct(s['too_lenient'])}"
            f"   too strict {pct(s['too_strict'])}")


def threshold_table(records: list) -> list:
    """Agreement if the pass line were 3, 4 or 5 (the score after the constraint cap)."""
    rows = []
    for threshold in (3, 4, 5):
        moved = [{**r, "judge_pass": r["score"] >= threshold} for r in records]
        rows.append({"pass_at": threshold, **summary(moved)})
    return rows


def main():
    records = load_results()
    if not records:
        raise SystemExit("no scored calibration judgments yet; run eval.judge_calibration_sample first")
    pass_line = load_judge_settings().pass_threshold

    report = {"all": summary(records),
              "without_ambiguous": summary([r for r in records if r["prompt_id"] not in AMBIGUOUS])}
    print(f"{len(records)} answers with known truth, judged exactly as in Phase 2 (pass at score >= {pass_line})\n")
    s = report["all"]
    c = s["confusion"]
    print("                      judge PASS   judge FAIL")
    print(f"  truly right       {c['right_judged_pass']:>10}   {c['right_judged_fail']:>10}   <- too strict")
    print(f"  truly wrong       {c['wrong_judged_pass']:>10}   {c['wrong_judged_fail']:>10}")
    print(f"                    {'^ too lenient':>12}\n")
    print(f"When the judge says PASS, the answer is really right {s['pass_means_right']:.0%} of the time.\n")

    print(line("all answers", s))
    print(line("without the 2 ambiguous questions", report["without_ambiguous"]))
    groups = {}
    for key, fn in (("dataset", lambda r: r["source"]), ("tier", lambda r: r["tier"]),
                    ("reference", lambda r: "reference right" if r["reference_correct"] else "reference WRONG")):
        print(f"\nby {key}:")
        for value in sorted({fn(r) for r in records}):
            groups[f"{key}={value}"] = summary([r for r in records if fn(r) == value])
            print(line(value, groups[f"{key}={value}"]))
    report["groups"] = groups

    print("\nscores given (truly right vs truly wrong answers):")
    for label, subset in (("truly right", [r for r in records if r["truth_pass"]]),
                          ("truly wrong", [r for r in records if not r["truth_pass"]])):
        counts = Counter(r["score"] for r in subset)
        print(f"  {label:<12} " + "  ".join(f"{s}:{counts.get(s, 0):>3}" for s in range(1, 6)))

    capped = [r for r in records if r["truth_pass"] and not r["judge_pass"] and r.get("broken_constraints")]
    print(f"\n{len(capped)} of the 'too strict' cases were capped for breaking a constraint "
          "(the answer was right, but the format rule wasn't followed)")

    print("\nif the pass line were different (train-split sample, so it's fine to look):")
    report["thresholds"] = threshold_table(records)
    for t in report["thresholds"]:
        print(f"  pass at >= {t['pass_at']}:  agree {t['agreement']:.0%}   too lenient {t['too_lenient']:.0%}"
              f"   too strict {t['too_strict']:.0%}")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "judge_calibration.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {(REPORTS_DIR / 'judge_calibration.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()