"""Phase 4 Step 4: score the pre-registered strategy on the TEST set, once. No API calls, no tuning.

The strategy is fixed by reports/PHASE4_PREREGISTRATION.json (written by eval/checker_test_run.py before any
test check was made):
    medium_then_check_selective, skip_above 0.90, min_answer_tokens 0, on_error escalate, checker groq-small v3
Comparison points (also fixed, none tuned here): always_medium, always_high, predict-only Triage at the
balanced preset (tau 0.86, chosen in Phase 3), and the oracle.

Scored with eval/routing_sim and eval/checker_routing.score_decisions, at real Groq prices and at price
ratios R = high / medium price (same method as eval/price_sensitivity.py: only the high tier's cost scales).

Before scoring, it verifies that:
  * the items the strategy checks today are exactly the pre-registered ones, with the same checker model,
    version, prompt hash and classifier;
  * every pre-registered item has a finished, ok verdict from that checker on that exact answer; and
  * scoring asks for a check on exactly those items and no others.

Usage:  python -m eval.final_test_phase4 --final
Writes reports/final_test_phase4.json and reports/final_test_phase4.png, then reports/FINAL_TEST_PHASE4_DONE.json.
After that, this script and eval/checker_test_run.py both refuse to run.
"""
import argparse
import json
from datetime import datetime, timezone

import numpy as np

from app.router.quality import choose_tier
from app.verifier.ai_checker import load_checker_settings, prompt_sha256
from eval.baselines import always, oracle
from eval.build_dataset import FULL_DIR, load_prompts
from eval.checker_path import CHECK
from eval.checker_routing import as_flat, medium_then_check_selective_path, prepare, score_decisions, selective_point
from eval.checker_test_run import (CLASSIFIER, FINAL_MARKER, PREREG_PATH, REPORTS_DIR, STRATEGY, log_path,
                                   preregistration, strategy_items, test_probabilities)
from eval.checker_verdicts import load_verdicts, make_estimate, read_records
from eval.price_sensitivity import ACTUAL_RATIO, scale_high
from eval.routing_sim import load_eval_rows, score
from eval.score_answers import load_answers

BALANCED_TAU = 0.86
RATIOS = (ACTUAL_RATIO, 3, 5, 10, 20)
STRATEGIES = ("always_medium", "always_high", "predict_only_balanced", "selective_checking", "oracle")
EPS = 1e-12


class VerificationFailed(SystemExit):
    pass


def verify_preregistration(items: list, settings, saved: dict) -> None:
    now = preregistration(items, settings)
    core = {k: v for k, v in saved.items() if k != "registered_at"}
    if core != now:
        raise VerificationFailed(f"the strategy's inputs differ from the pre-registration: "
                                 f"{sorted(k for k in now if core.get(k) != now[k])}")
    if now["checker_prompt_sha256"] != prompt_sha256():
        raise VerificationFailed("the checker prompt changed since the pre-registration")


def verify_verdicts(verdicts: dict, prereg_ids: list) -> None:
    bad = {k: v.why for k, v in verdicts.items() if k[0] in prereg_ids and (v.why != "ok" or v.estimated)}
    if bad:
        raise VerificationFailed(f"{len(bad)} pre-registered checks are not finished ok verdicts: {list(bad.items())[:5]}")


def selective_decisions(rows, probs, verdicts, format_ok):
    results = [medium_then_check_selective_path(r, verdicts, format_ok, p_medium=probs[r.id][1], tokens=None,
                                                skip_above=STRATEGY["skip_above"],
                                                min_answer_tokens=STRATEGY["min_answer_tokens"],
                                                on_error=STRATEGY["on_error"]) for r in rows]
    return [(final, calls) for final, calls, _ in results], [why for _, _, why in results]


def checked_ids(decisions, rows) -> set:
    return {r.id for r, (_, calls) in zip(rows, decisions) if any(c.kind == CHECK for c in as_flat(calls))}


def score_all(rows, probs, verdicts, format_ok, factor: float = 1.0) -> dict:
    priced = scale_high(rows, factor)
    decisions, skipped = selective_decisions(priced, probs, verdicts, format_ok)
    sel = selective_point(priced, decisions, skipped, verdicts)
    predict = score(priced, [choose_tier(*probs[r.id], BALANCED_TAU) for r in priced])
    out = {"always_medium": score(priced, always(priced, "medium")),
           "always_high": score(priced, always(priced, "high")),
           "predict_only_balanced": {**predict, "tau": BALANCED_TAU},
           "selective_checking": sel,
           "oracle": score(priced, oracle(priced))}
    high = out["always_high"]["cost_per_1k"]
    med = out["always_medium"]["cost_per_1k"]
    for name, s in out.items():
        s["correct"] = round(s["quality"] * s["n"])
        s["pct_of_always_high_cost"] = 100 * s["cost_per_1k"] / high
        s["x_always_medium_cost"] = s["cost_per_1k"] / med
        s["escalated_share"] = s["mix"]["high"] if name != "always_high" else 1.0
    out["selective_checking"]["checked_share"] = sel["checked_share"]
    return out, decisions


def beats_always_medium(s: dict, m: dict) -> bool:
    return (s["correct"] > m["correct"] and s["cost_per_1k"] <= m["cost_per_1k"] + EPS) or \
           (s["correct"] >= m["correct"] and s["cost_per_1k"] < m["cost_per_1k"] - EPS)


def break_even(by_ratio: dict) -> dict:
    """Costs are linear in R; where selective checking and predict-only (balanced) cost the same."""
    r0, r1 = RATIOS[0], RATIOS[-1]
    line = lambda name: [(by_ratio[f"{r:g}"][name]["cost_per_1k"]) for r in (r0, r1)]
    (p0, p1), (s0, s1) = line("predict_only_balanced"), line("selective_checking")
    pb, sb = (p1 - p0) / (r1 - r0), (s1 - s0) / (r1 - r0)
    pa, sa = p0 - pb * r0, s0 - sb * r0
    ratio = None if pb == sb else (sa - pa) / (pb - sb)
    return {"ratio": ratio, "note": "R where selective checking and predict-only (tau 0.86) cost the same "
                                    "(costs are linear in R); compare their correct counts separately"}


def mistakes(rows, decisions, skipped, verdicts, format_ok) -> dict:
    """What happened to each wrong medium answer (and to right ones the strategy escalated anyway)."""
    out = {"medium_wrong": 0, "format_caught": 0, "skipped_confident": 0, "checker_caught": 0,
           "checker_missed": 0, "checker_unknown": 0, "fixed_by_high": 0, "escalated_but_high_also_wrong": 0,
           "right_answers_escalated": 0, "right_answers_escalated_by_checker": 0, "checked": 0,
           "checker_false_alarms": 0}
    for r, (final, calls), why in zip(rows, decisions, skipped):
        was_checked = any(c.kind == CHECK for c in as_flat(calls))
        out["checked"] += was_checked
        v = verdicts.get((r.id, "medium"))
        if r.passed["medium"]:
            if final == "high":
                out["right_answers_escalated"] += 1
                out["right_answers_escalated_by_checker"] += was_checked
                out["checker_false_alarms"] += bool(was_checked and v and v.status == "bad")
            continue
        out["medium_wrong"] += 1
        if not format_ok[(r.id, "medium")]:
            out["format_caught"] += 1
        elif why == "confident":
            out["skipped_confident"] += 1
        elif v.status == "bad":
            out["checker_caught"] += 1
        elif v.status == "good":
            out["checker_missed"] += 1
        else:
            out["checker_unknown"] += 1
        if final == "high":
            out["fixed_by_high" if r.passed["high"] else "escalated_but_high_also_wrong"] += 1
    return out


def validation_reference() -> dict:
    """The validation numbers this was chosen on, for the comparison (read-only)."""
    out = {}
    try:
        pv = json.loads((REPORTS_DIR / "pareto_val.json").read_text())
        sel = next(p for p in pv["selective"]["e. medium_then_check_selective (min_answer_tokens=0)"]
                   if abs(p["skip_above"] - 0.90) < 1e-9)
        po = next(p for p in pv["curves"]["d. predict-only Triage (v2)"] if abs(p["tau"] - BALANCED_TAU) < 1e-9)
        n = pv["n"]
        out = {"n": n, "selective_checking": {"correct": round(sel["quality"] * n), "cost_saved": sel["cost_saved"],
                                               "quality_retained": sel["quality_retained"]},
               "predict_only_balanced": {"correct": round(po["quality"] * n), "cost_saved": po["cost_saved"],
                                         "quality_retained": po["quality_retained"]},
               "always_medium": {"correct": round(pv["baselines"]["always_medium"]["quality"] * n)},
               "always_high": {"correct": round(pv["baselines"]["always_high"]["quality"] * n)}}
        ps = json.loads((REPORTS_DIR / "price_sensitivity_val.json").read_text())
        out["break_even_at_113_correct"] = (ps.get("break_even") or {}).get("113")
    except (OSError, KeyError, StopIteration, ValueError):
        pass
    return out


def plot(report: dict, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = report["real_prices"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(13, 5.5))
    style = {"always_medium": ("s", "tab:brown"), "always_high": ("^", "tab:green"),
             "predict_only_balanced": ("o", "tab:blue"), "selective_checking": ("D", "tab:purple"),
             "oracle": ("*", "gold")}
    labels = {"always_medium": "always medium", "always_high": "always high",
              "predict_only_balanced": "predict-only (tau 0.86)",
              "selective_checking": "selective checking (pre-registered)", "oracle": "oracle"}
    for name, (marker, color) in style.items():
        s = real[name]
        left.scatter(100 * s["cost_saved"], 100 * s["quality_retained"], marker=marker, color=color, s=140,
                     edgecolors="black", zorder=5, label=f"{labels[name]}: {s['correct']}/{s['n']}")
    left.set_xlabel("cost saved vs always-high (%, real Groq prices, checks included)")
    left.set_ylabel("quality retained (% of always-high)")
    left.set_title(f"Test set ({real['always_high']['n']} prompts), real prices (R = 2)")
    left.grid(alpha=0.3)
    left.legend(fontsize=8, loc="lower left")

    ratios = sorted(float(r) for r in report["by_ratio"])
    for name in ("always_high", "predict_only_balanced", "selective_checking", "always_medium"):
        marker, color = style[name]
        ys = [report["by_ratio"][f"{r:g}"][name]["pct_of_always_high_cost"] for r in ratios]
        right.plot(ratios, ys, marker=marker, color=color, lw=2,
                   label=f"{labels[name]} ({real[name]['correct']} correct)")
    right.axvline(ACTUAL_RATIO, color="grey", ls="--", lw=1)
    right.text(ACTUAL_RATIO * 1.03, 95, "actual Groq ratio (2x)", fontsize=8, color="grey", va="top")
    be = report["break_even"]["ratio"]
    if be and ratios[0] <= be <= ratios[-1]:
        right.axvline(be, color="tab:purple", ls=":", lw=1)
        right.text(be * 1.03, 80, f"break-even R = {be:.2f}", fontsize=8, color="tab:purple")
    right.set_xscale("log")
    right.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    right.set_xticks(ratios)
    right.set_xticklabels([f"{r:g}" for r in ratios])
    right.set_xlabel("price ratio R = high-tier price / medium-tier price (log scale)")
    right.set_ylabel("cost (% of always-high at that R)")
    right.set_title("Cost as the high tier gets pricier (settings fixed)")
    right.grid(alpha=0.3)
    right.legend(fontsize=8)
    fig.suptitle("Phase 4 final test (second use of the test set): pre-registered selective checking")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase 4 final: score the pre-registered strategy on the test set.")
    parser.add_argument("--final", action="store_true", help="required: the one Phase 4 test-set scoring")
    args = parser.parse_args(argv)
    if not args.final:
        raise SystemExit("this scores the TEST set once; pass --final")
    if FINAL_MARKER.exists():
        raise SystemExit(f"the Phase 4 final test evaluation already ran ({FINAL_MARKER.read_text().strip()})")
    if not PREREG_PATH.exists():
        raise SystemExit(f"{PREREG_PATH.name} is missing: run eval.checker_test_run --final first")
    saved = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    settings = load_checker_settings()
    model = settings.model_name

    rows = load_eval_rows("test", allow_test=True)
    ids = [r.id for r in rows]
    prompts = {p["id"]: p for p in load_prompts() if p["id"] in set(ids)}
    answers = {k: v for k, v in load_answers(FULL_DIR / "answers.jsonl").items() if k[0] in set(ids)}
    probs = test_probabilities(ids)

    # 1. same prompts, same checker, same classifier as registered
    items, skipped_counts = strategy_items(ids, prompts, answers, probs)
    verify_preregistration(items, settings, saved)
    prereg_ids = set(saved["items"])
    # 2. every registered check has a finished ok verdict, on that exact answer
    keys = [(i, "medium") for i in ids]
    inputs = {(i, "medium"): (prompts[i]["messages"], (answers.get((i, "medium")) or {}).get("answer") or "")
              for i in ids}
    path = log_path(model)
    verdicts = load_verdicts(path, keys, model=model, version=saved["checker_version"],
                             prompt_hash=saved["checker_prompt_sha256"], inputs=inputs,
                             estimate=make_estimate(model, read_records(path)))
    verify_verdicts(verdicts, prereg_ids)
    format_ok, _, _ = prepare(rows, prompts, answers)
    # 3. scoring asks for a check on exactly the registered items
    decisions, skipped = selective_decisions(rows, probs, verdicts, format_ok)
    if checked_ids(decisions, rows) != prereg_ids:
        raise VerificationFailed("scoring would check a different set of prompts than the pre-registration")

    real, _ = score_all(rows, probs, verdicts, format_ok, 1.0)
    by_ratio = {f"{R:g}": score_all(rows, probs, verdicts, format_ok, R / ACTUAL_RATIO)[0] for R in RATIOS}
    for R, res in by_ratio.items():
        m = res["always_medium"]
        res["beats_always_medium"] = [n for n in ("predict_only_balanced", "selective_checking", "oracle")
                                      if beats_always_medium(res[n], m)]
        res["selective_cheaper_than_predict_only"] = (res["selective_checking"]["cost_per_1k"]
                                                      < res["predict_only_balanced"]["cost_per_1k"] - EPS)
    report = {
        "split": "test", "use": "second use of the test set (first: Phase 3 final, reports/FINAL_TEST_DONE.json)",
        "n": len(rows), "created": datetime.now(timezone.utc).isoformat(),
        "preregistration": {"path": str(PREREG_PATH.name), "registered_at": saved.get("registered_at"),
                            "verified": ["same items (" + str(len(prereg_ids)) + ")", "same checker model, version and "
                                         "prompt hash", "same classifier", "every registered check finished ok on "
                                         "that exact answer", "scoring checks exactly the registered items"]},
        "strategy": STRATEGY, "classifier": f"{CLASSIFIER[0]} v{CLASSIFIER[1]}", "predict_only_tau": BALANCED_TAU,
        "selection": dict(skipped_counts),
        "real_prices": real, "by_ratio": by_ratio, "break_even": break_even(by_ratio),
        "medium_mistakes": mistakes(rows, decisions, skipped, verdicts, format_ok),
        "validation_reference": validation_reference(),
        "caveats": [
            "The comparison rests on a small number of wrong medium answers (see medium_mistakes); one or two "
            "prompts more or less would move the break-even ratio.",
            "This is the test set's second use. Phase 3 used it once, and its result was known when Phase 4 began, "
            "so it is no longer completely fresh, though nothing in Phase 4 was tuned on it.",
            "Only price is varied across ratios; real providers also differ in quality gaps and answer lengths.",
            "The checker stays at groq-small's price at every ratio.",
        ],
    }
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "final_test_phase4.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(report, REPORTS_DIR / "final_test_phase4.png")
    with open(FINAL_MARKER, "x", encoding="utf-8") as f:
        f.write(json.dumps({"ran_at": report["created"], "prompts": len(rows), "strategy": STRATEGY}) + "\n")

    print(f"{len(rows)} test prompts | verified against {PREREG_PATH.name}")
    for name in STRATEGIES:
        s = real[name]
        print(f"  {name:24s} {s['correct']:3d}/{s['n']}  kept {s['quality_retained']:6.1%}  saved {s['cost_saved']:6.1%}  "
              f"escalated {s['escalated_share']:5.1%}" + (f"  checked {s['checked_share']:5.1%}" if 'checked_share' in s else ""))
    for R, res in by_ratio.items():
        print(f"  R={R:>3s}: predict-only {res['predict_only_balanced']['pct_of_always_high_cost']:5.1f}% vs selective "
              f"{res['selective_checking']['pct_of_always_high_cost']:5.1f}% of always-high; beats always_medium: "
              f"{res['beats_always_medium'] or 'none'}")
    print("  break-even R:", report["break_even"]["ratio"])
    print("  medium mistakes:", report["medium_mistakes"])
    print("saved reports/final_test_phase4.json, final_test_phase4.png and FINAL_TEST_PHASE4_DONE.json")


if __name__ == "__main__":
    main()
