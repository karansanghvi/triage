"""Phase 3 Step 4: Triage's cost-vs-quality curve, with every baseline on the same chart.
 
For each tau from 0 to 1, route every prompt with choose_tier() (the same function the live router uses)
and score the result with routing_sim.score(). Each tau is one point; together they are Triage's curve.
 
Two modes:
 
  TUNING (train split, run as often as you like):
      python -m eval.pareto --classifiers classifier_c_logreg_cv:1 classifier_b_logreg_cv:1
    Uses the out-of-fold probabilities saved with each model, so every prediction comes from a model that
    never saw that prompt. Use this to pick the winning classifier and the cheap/balanced/high presets.
 
  FINAL (locked test split, run ONCE at the very end):
      python -m eval.pareto --final
    Runs the real predictor (the classifier named in routing.yaml) on the test prompts, twice: with the
    hand-tagged task types and with guessed ones (what production sees). Writes a marker file; running it
    again needs --force, because re-running and adjusting after seeing test results defeats the test set.
 
  SPLIT OPTION (Phase 4):  --split train (default) | fit | val
    train and fit use out-of-fold probabilities (fit = train minus validation). val runs the real Predictor
    for each --classifiers name:version on the validation prompts with hand-tagged task types, because a
    model trained on the fit set has no out-of-fold probabilities for them.

Outputs go to reports/: pareto_<split>.png (the chart) and pareto_<split>.json (every number). On val the
files are pareto_predict_only_val.*, because pareto_val.* is the Phase 4 Step 3 chart (eval/checker_routing.py).
"""

import argparse
import json
import time
from datetime import datetime, timezone

import numpy as np

from app.registry import PROJECT_ROOT
from app.router.quality import choose_tier, load_quality_settings
from eval.baselines import all_baselines, random_same_mix
from eval.routing_sim import TIERS, load_eval_rows, score

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FINAL_MARKER = REPORTS_DIR / "FINAL_TEST_DONE.json"
TAUS = [round(t, 2) for t in np.arange(0, 1.0001, 0.01)]
TARGETS = (0.90, 0.95, 0.98)   # quality-retained levels to report the cheapest tau for


# ---------- probabilities ----------

def oof_probabilities(name: str, version: int) -> dict:
    """{prompt_id: (p_low, p_medium)} from the out-of-fold probabilities saved by train_classifier.py"""
    probs = {}
    for tier in ("low", "medium"):
        path = MODELS_DIR / f"{name}_{tier}_v{version}.json"
        if not path.exists():
            raise SystemExit(f"{path.relative_to(PROJECT_ROOT)} not found")
        probs[tier] = json.loads(path.read_text(encoding="utf-8"))["oof_probabilities"]
    shared = probs["low"].keys() & probs["medium"].keys()
    # Same adjustment as predictor.py: medium never looks less likely to pass than cheap 
    return {i: (probs["low"][i], max(probs["medium"][i], probs["low"][i])) for i in shared}


def test_probabilities(rows: list, guess_task_type: bool) -> tuple:
    """({prompt_id: (p_low, p_medium)}, router overhead in ms per request) from the real live predictor."""
    from app.router.features import prompt_text
    from app.router.predictor import load_predictor
    from eval.build_dataset import load_prompts
    prompts = {p["id"]: p for p in load_prompts()}
    predictor = load_predictor()
    texts =[prompt_text(prompts[r.id]) for r in rows]
    tasks = [None if guess_task_type else prompts[r.id].get("task_type") for r in rows]
    start = time.perf_counter()
    preds = predictor.predict_texts(texts, tasks)
    overhead_ms = (time.perf_counter() - start) * 1000 / len(rows)
    return {r.id: (p.p_low, p.p_medium) for r, p in zip(rows, preds)}, overhead_ms


def predictor_probabilities(name: str, version: int, rows: list) -> dict:
    """{prompt_id: (p_low, p_medium)} from the real Predictor name:version, hand-tagged task types.
    For prompts the model never trained on (validation, for a model trained on the fit set)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")        # the embedding model must already be cached
    from app.router.features import prompt_text
    from app.router.predictor import Predictor
    from eval.build_dataset import load_prompts
    prompts = {p["id"]: p for p in load_prompts()}
    preds = Predictor(name, version).predict_texts([prompt_text(prompts[r.id]) for r in rows],
                                                   [prompts[r.id].get("task_type") for r in rows])
    return {r.id: (p.p_low, p.p_medium) for r, p in zip(rows, preds)}


def split_probabilities(name: str, version: int, split: str, rows: list) -> dict:
    """Out-of-fold probabilities on train / fit; the real Predictor on val."""
    if split == "val":
        return predictor_probabilities(name, version, rows)
    return oof_probabilities(name, version)


def report_stem(split: str) -> str:
    return "pareto_predict_only_val" if split == "val" else f"pareto_{split}"


# ---------- the sweep ----------

def sweep(rows: list, probs: dict, random_repeats: int = 200) -> list:
    """One point per tau: Triage's score, plus random routing with the same tier mix"""
    rows = [r for r in rows if r.id in probs]
    points = []
    for tau in TAUS:
        tiers = [choose_tier(*probs[r.id], tau) for r in rows]
        s = score(rows, tiers)
        rand = random_same_mix(rows, s["mix"], repeats=random_repeats, seed=42)
        points.append({"tau": tau, **s, "random_quality": rand["quality"], "random_cost_per_1k": rand["cost_per_1k"]})
    return points


def summarize(points: list, baselines: dict) -> dict:
    """The numbers worth quoting: cheapest tau for each quality target, and the comparison with always-medium"""
    out ={}
    for target in TARGETS:
        ok = [p for p in points if p["quality_retained"] >= target]
        best = min(ok, key=lambda p: (p["cost_per_1k"], -p["tau"])) if ok else None
        out[f"cheapest_at_{int(target * 100)}pct"] = best and {k: best[k] for k in
                                                             ("tau", "quality_retained", "cost_saved", "mix")}
    medium = baselines["always_medium"]
    affordable = [p for p in points if p["cost_per_1k"] <= medium["cost_per_1k"]]
    best = max(affordable, key=lambda p: p["quality"]) if affordable else None
    out["best_quality_at_always_medium_cost"] = best and {
        "tau": best["tau"], "quality_retained": best["quality_retained"], 
        "cost_saved": best["cost_saved"], "always_medium_retained": medium["quality_retained"]}
    # How much better than blind routing at the same mix, averaged over the curve 
    out["avg_gain_over_random_same_mix"] = float(np.mean([p["quality"] - p["random_quality"]
                                                          for p in points]))
    return out


def preset_points(points: list) -> dict:
    """Where the current routing.yaml presets land on the curve"""
    settings = load_quality_settings()
    by_tau = {p["tau"]: p for p in points}
    return {name: by_tau[min(by_tau, key=lambda t: abs(t - tau))] for name, tau in settings.presets.items()}


# ---------- output ----------

def print_report(label: str, points: list, summary: dict, presets: dict) -> None:
    print(f"\n=== {label} ===")
    for key, val in summary.items():
        if key.startswith("cheapest_at"):
            target = key.split("_")[-1].replace("pct", "%")
            if val: 
                mix = "/".join(f"{val['mix'][t]:.0%}" for t in TIERS)
                print(f" {target} of strong quality: cheapest at tau {val['tau']:.2f} -> saves "
                      f"{val['cost_saved']:.0%} (mix low/med/high {mix})")
            else:
                print(f" {target} of strong quality: never reached")
    b = summary["best_quality_at_always_medium_cost"]
    if b:
        print(f"    at always-medium's cost or less: best quality {b['quality_retained']:.1%} (tau {b['tau']:.2f})"
              f" vs always-medium {b['always_medium_retained']:.1%}")
    print(f"    average quality gain over random routing at the same mix: "
          f"{summary['avg_gain_over_random_same_mix'] * 100:+.1f} points")
    print("  current presets:  " + "   ".join(
            f"{name} (tau {p['tau']:.2f}): {p['quality_retained']:.1%} kept, {p['cost_saved']:.0%} saved"
             for name, p in presets.items()))


def plot(curves: dict, baselines: dict, presets: dict, path, title: str)-> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (label, points) in enumerate(curves.items()):
        x = [p["cost_per_1k"] for p in points]
        line, = ax.plot(x, [p["quality_retained"] * 100 for p in points], lw=2, label=f"Triage: {label}")
        if i == 0: # random routing at the same mix as the first one 
            rq = [p["random_quality"] / p["quality"] * p["quality_retained"] * 100 if p["quality"]
                  else 0 for p in points]
            ax.plot([p["random_cost_per_1k"] for p in points], rq, ls="--", color="grey", label="random routing, same mix")
    
    for name, preset in presets.items():
        ax.scatter(preset["cost_per_1k"], preset["quality_retained"] * 100, s=60, zorder=5, color="black")
        ax.annotate(f"{name} (τ {preset['tau']:.2f})", (preset["cost_per_1k"], preset["quality_retained"] * 100),
                    textcoords="offset points", xytext=(6, -12), fontsize=8)
    markers = {"always_low": ("v", "tab:purple"), "always_medium": ("s", "tab:brown"),
               "always_high": ("^", "tab:green"), "oracle": ("*", "gold")}
    
    for name, (marker, color) in markers.items():
        b = baselines[name]
        ax.scatter(b["cost_per_1k"], b["quality_retained"] * 100, marker=marker, s=130, zorder=6,
                   color=color, edgecolors="black", linewidths=0.6,
                   label=name.replace("_"," ").replace("oracle", "perfect router (oracle)"))
    ax.set_xlabel("cost per 1,000 requests (USD, Groq paid prices)")
    ax.set_ylabel("quality retained (% of always-strong)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"\nchart saved to {path.relative_to(PROJECT_ROOT)}")


# ---------- main ----------


def main():
    parser = argparse.ArgumentParser(description="Sweep tau and draw Triage's cost-quality curve.")
    parser.add_argument("--classifiers", nargs="+", default=["classifier_c_logreg_cv:1"],
                        help="tuning mode: name:version of each classifier to compare (out-of-fold, train split)")
    parser.add_argument("--final", action="store_true", help="the one-time final evaluation on the test split")
    parser.add_argument("--force", action="store_true", help="allow re-running the final evaluation")
    parser.add_argument("--split", default="train", choices=["train", "fit", "val"],
                        help="tuning split (default train, as before; val runs the real predictor)")
    args = parser.parse_args()
    REPORTS_DIR.mkdir(exist_ok=True)
 
    if args.final:
        if FINAL_MARKER.exists() and not args.force:
            raise SystemExit(f"the final test evaluation already ran ({FINAL_MARKER.read_text()}); "
                             "re-running it after seeing the results turns the test set into tuning data. "
                             "Pass --force only for a genuine reason (e.g. a bug fix), and say so in the write-up.")
        rows = load_eval_rows("test", allow_test=True)
        split, curves, overhead = "test", {}, {}
        for label, guess in (("task type given", False), ("task type guessed", True)):
            probs, overhead[label] = test_probabilities(rows, guess)
            curves[label] = sweep(rows, probs)
    else:
        split = args.split
        rows = load_eval_rows(split, allow_val=split == "val")
        curves, overhead = {}, {}
        for spec in args.classifiers:
            name, _, version = spec.partition(":")
            curves[f"{name} v{version or 1}"] = sweep(rows, split_probabilities(name, int(version or 1), split, rows))
 
    baselines = all_baselines(rows)
    print(f"{len(rows)} labeled {split} prompts | always-medium keeps {baselines['always_medium']['quality_retained']:.1%}"
          f" and saves {baselines['always_medium']['cost_saved']:.0%}; perfect router keeps "
          f"{baselines['oracle']['quality_retained']:.1%} and saves {baselines['oracle']['cost_saved']:.0%}")
 
    report = {"split": split, "created": datetime.now(timezone.utc).isoformat(), "baselines": baselines,
              "curves": {}, "router_overhead_ms": overhead}
    first_presets = None
    for label, points in curves.items():
        summary, presets = summarize(points, baselines), preset_points(points)
        first_presets = first_presets or presets
        print_report(label, points, summary, presets)
        report["curves"][label] = {"summary": summary, "presets": presets, "points": points}
    for label, ms in overhead.items():
        print(f"router overhead ({label}): {ms:.1f} ms per request")
 
    (REPORTS_DIR / f"{report_stem(split)}.json").write_text(json.dumps(report, indent=2))
    plot(curves, baselines, first_presets, REPORTS_DIR / f"{report_stem(split)}.png",
         f"Triage: cost vs quality ({split} split, {len(rows)} prompts)")
    if args.final:
        FINAL_MARKER.write_text(json.dumps({"ran_at": report["created"], "prompts": len(rows)}))


if __name__ == "__main__":
    main()