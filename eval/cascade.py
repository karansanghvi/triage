"""Cascade routing, simulated with the Phase 2 answers (no API calls).

Cascade: send the prompt to a starting tier, check the answer with app/verifier/checker.py, and move up
one tier when the check fails. "high" is the last stop, so its answer is always used. Every tier tried is
paid for, and latencies add up, because the tiers run one after another.

Strategies scored (train split, the same score() rules as every other strategy):
    cascade low->medium->high     start cheap, escalate on a failed check
    cascade low->medium           start cheap, escalate to medium at most (never strong)
    cascade medium->high          start at medium, escalate to strong on a failed check
    classifier start + cascade    a tau sweep: start cheap if p(cheap passes) >= tau, else medium; then cascade
    predict-only (Step 3)         the tau sweep from pareto.py, for comparison

Usage:
    python -m eval.cascade                       # checker without constraint lists (realistic)
    python -m eval.cascade --constraints         # also check the hand-written constraint lists
    python -m eval.cascade --classifier classifier_c_logreg_cv:1
"""
import argparse
import json

import numpy as np

from app.registry import PROJECT_ROOT
from app.verifier.checker import check_answer
from eval.baselines import all_baselines
from eval.build_dataset import FULL_DIR, load_prompts
from eval.pareto import TAUS, oof_probabilities, sweep
from eval.routing_sim import TIERS, load_eval_rows
from eval.score_answers import load_answers

REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------- checking every cheap and medium answer once ----------

def check_all(rows: list, use_constraints: bool) -> dict:
    """{(prompt_id, tier): CheckResult} for the low and medium answers."""
    prompts = {p["id"]: p for p in load_prompts()}
    answers = load_answers(FULL_DIR / "answers.jsonl")
    checks = {}
    for r in rows:
        prompt = prompts[r.id]
        for tier in ("low", "medium"):
            a = answers.get((r.id, tier)) or {}
            checks[(r.id, tier)] = check_answer(
                prompt["messages"], a.get("answer"), truncated=bool(a.get("truncated")), error=a.get("error"),
                constraints=prompt.get("constraints") if use_constraints else None)
    return checks


def checker_report(rows: list, checks: dict) -> dict:
    """How good is the checker? 'Flagged' = the check said bad."""
    out = {}
    for tier in ("low", "medium"):
        flagged = np.array([not checks[(r.id, tier)].ok for r in rows])
        failed = np.array([not r.passed[tier] for r in rows])
        out[tier] = {
            "flag_rate": float(flagged.mean()),
            "flagged_truly_bad": float(failed[flagged].mean()) if flagged.any() else None,   # precision
            "bad_caught": float(flagged[failed].mean()) if failed.any() else None,           # recall
            "passed_check_truly_good": float((~failed[~flagged]).mean()) if (~flagged).any() else None,
        }
    reasons = {}
    for (pid, tier), c in checks.items():
        for reason in c.reasons:
            key = "breaks a constraint" if reason.startswith("breaks constraint") else reason
            reasons[(tier, key)] = reasons.get((tier, key), 0) + 1
    out["reasons"] = {f"{t}: {k}": n for (t, k), n in sorted(reasons.items(), key=lambda x: -x[1])}
    return out


# ---------- running a cascade ----------

def cascade_path(row, start: str, checks: dict, top: str = "high") -> list:
    """Tiers tried, in order, starting at `start` and escalating on a failed check, up to `top`."""
    order = TIERS[TIERS.index(start):TIERS.index(top) + 1]
    path = []
    for tier in order:
        path.append(tier)
        if tier == top or checks[(row.id, tier)].ok:
            break
    return path


def score_paths(rows: list, paths: list) -> dict:
    """Like routing_sim.score(), but every tier tried is paid for and latencies add up."""
    final = [p[-1] for p in paths]
    passed = np.array([r.passed[t] for r, t in zip(rows, final)], dtype=float)
    cost = np.array([sum(r.cost[t] for t in p) for r, p in zip(rows, paths)])
    latency = np.array([sum(r.latency_ms[t] or 0 for t in p) for r, p in zip(rows, paths)])
    strong_quality = np.mean([r.passed["high"] for r in rows])
    strong_cost = sum(r.cost["high"] for r in rows)
    return {
        "n": len(rows),
        "quality": float(passed.mean()),
        "quality_retained": float(passed.mean() / strong_quality),
        "cost_per_1k": float(cost.mean() * 1000),
        "cost_saved": float(1 - cost.sum() / strong_cost),
        "latency_p50_ms": float(np.percentile(latency, 50)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "mix": {t: float(np.mean([x == t for x in final])) for t in TIERS},   # where answers came from
        "escalation_rate": float(np.mean([len(p) > 1 for p in paths])),
    }


def fixed_cascades(rows: list, checks: dict) -> dict:
    return {
        "cascade low->medium->high": score_paths(rows, [cascade_path(r, "low", checks) for r in rows]),
        "cascade low->medium": score_paths(rows, [cascade_path(r, "low", checks, top="medium") for r in rows]),
        "cascade medium->high": score_paths(rows, [cascade_path(r, "medium", checks) for r in rows]),
    }


def classifier_cascade_sweep(rows: list, checks: dict, probs: dict) -> list:
    rows = [r for r in rows if r.id in probs]
    points = []
    for tau in TAUS:
        paths = [cascade_path(r, "low" if probs[r.id][0] >= tau else "medium", checks) for r in rows]
        points.append({"tau": tau, **score_paths(rows, paths)})
    return points


# ---------- output ----------

def print_table(results: dict) -> None:
    print(f"{'strategy':<28}{'quality':>9}{'retained':>10}{'$/1k':>9}{'saved':>7}{'p50 ms':>8}{'p95 ms':>8}"
          f"{'escal.':>8}   answers from (low/med/high)")
    for name, r in results.items():
        mix = "/".join(f"{r['mix'][t]:.0%}" for t in TIERS)
        esc = f"{r['escalation_rate']:.0%}" if "escalation_rate" in r else "-"
        print(f"{name:<28}{r['quality']:>9.1%}{r['quality_retained']:>10.1%}{r['cost_per_1k']:>9.4f}"
              f"{r['cost_saved']:>7.0%}{r['latency_p50_ms'] or 0:>8.0f}{r['latency_p95_ms'] or 0:>8.0f}{esc:>8}   {mix}")


def plot(baselines: dict, fixed: dict, curves: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for label, points in curves.items():
        ax.plot([p["cost_per_1k"] for p in points], [p["quality_retained"] * 100 for p in points], lw=2, label=label)
    for name, r in fixed.items():
        ax.scatter(r["cost_per_1k"], r["quality_retained"] * 100, marker="D", s=70, zorder=6, edgecolors="black",
                   label=name)
    markers = {"always_low": ("v", "tab:purple"), "always_medium": ("s", "tab:brown"),
               "always_high": ("^", "tab:green"), "oracle": ("*", "gold")}
    for name, (marker, color) in markers.items():
        b = baselines[name]
        ax.scatter(b["cost_per_1k"], b["quality_retained"] * 100, marker=marker, s=130, zorder=6, color=color,
                   edgecolors="black", linewidths=0.6,
                   label=name.replace("_", " ").replace("oracle", "perfect router (oracle)"))
    ax.set_xlabel("cost per 1,000 requests (USD, Groq paid prices)")
    ax.set_ylabel("quality retained (% of always-strong)")
    ax.set_title("Cascade vs predict-only routing (train split)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nchart saved to {path.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Simulate cascade routing with the Phase 2 answers.")
    parser.add_argument("--classifier", default="classifier_c_logreg_cv:1")
    parser.add_argument("--constraints", action="store_true", help="also check hand-written constraint lists")
    args = parser.parse_args()
    REPORTS_DIR.mkdir(exist_ok=True)

    rows = load_eval_rows("train")
    checks = check_all(rows, args.constraints)
    report = checker_report(rows, checks)

    print(f"{len(rows)} labeled train prompts | checker {'WITH' if args.constraints else 'without'} constraint lists\n")
    print("How good is the checker?")
    for tier in ("low", "medium"):
        c = report[tier]
        print(f"  {tier:<7} flags {c['flag_rate']:.0%} of answers | of those flagged, {c['flagged_truly_bad']:.0%} "
              f"really were bad | catches {c['bad_caught']:.0%} of all bad answers | passed the check and really "
              f"good: {c['passed_check_truly_good']:.0%}")
    print("  most common reasons: " + ", ".join(f"{k} ({n})" for k, n in list(report["reasons"].items())[:6]))

    name, _, version = args.classifier.partition(":")
    probs = oof_probabilities(name, int(version or 1))
    baselines = all_baselines(rows)
    fixed = fixed_cascades(rows, checks)
    predict_only = sweep(rows, probs, random_repeats=20)
    hybrid = classifier_cascade_sweep(rows, checks, probs)

    print()
    print_table({k: baselines[k] for k in ("always_low", "always_medium", "always_high", "oracle")} | fixed)

    medium = baselines["always_medium"]
    for label, points in (("predict-only", predict_only), ("classifier start + cascade", hybrid)):
        affordable = [p for p in points if p["cost_per_1k"] <= medium["cost_per_1k"]]
        best = max(affordable, key=lambda p: p["quality"]) if affordable else None
        if best:
            print(f"\n{label}: best quality at always-medium's cost or less: {best['quality_retained']:.1%} "
                  f"(tau {best['tau']:.2f}, saves {best['cost_saved']:.0%}) vs always-medium {medium['quality_retained']:.1%}")
        at95 = [p for p in points if p["quality_retained"] >= 0.95]
        if at95:
            cheapest = min(at95, key=lambda p: p["cost_per_1k"])
            print(f"{label}: cheapest way to keep 95%: tau {cheapest['tau']:.2f}, saves {cheapest['cost_saved']:.0%}")
        else:
            print(f"{label}: never reaches 95% retained")

    suffix = "_constraints" if args.constraints else ""
    (REPORTS_DIR / f"cascade_train{suffix}.json").write_text(json.dumps(
        {"checker": report, "fixed": fixed, "hybrid": hybrid}, indent=2))
    plot(baselines, fixed, {f"predict-only ({name})": predict_only, "classifier start + cascade": hybrid},
         REPORTS_DIR / f"cascade_train{suffix}.png")


if __name__ == "__main__":
    main()