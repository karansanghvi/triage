"""Phase 4 Step 3 follow-up: price-gap sensitivity on the VALIDATION set. No API calls, no config changes.

Question: does answer-checking beat predict-only routing once the high tier is much more expensive than medium?

R = high-tier price / medium-tier price, input and output prices scaled together. Today groq-large is exactly
2x groq-small on both ($0.15/$0.60 vs $0.075/$0.30 per 1M tokens), and every high answer's price_as_cost is
its own tokens x those prices, so pricing high at R x medium is the stored high cost x R / 2. Low, medium,
the extra model (llama3.2:3b) and the checker (groq-small) keep their real price_as prices.

Quality never depends on R; only cost does. For each R and each quality level k (number correct out of 117),
every strategy family reports its CHEAPEST setting that gets at least k right; predict-only picks its own tau
at each R (today's presets are ignored).

Families (all from eval/checker_routing.py, same verdicts and Phase 2 answers as reports/pareto_val.json)
  always_medium, always_high
  predict_only            classifier v2, tau swept (TAUS)
  medium_then_check       (a), on_error = escalate
  selective               (e), on_error = escalate, min_answer_tokens = 0, skip_above swept (SKIP_ABOVE)
  classifier_start        (c), on_error = escalate, tau swept

Usage:  python -m eval.price_sensitivity [--model groq-small]
Writes reports/price_sensitivity_val.json and reports/price_sensitivity_val.png.
"""
import argparse
import copy
import json
from datetime import datetime, timezone

from app.verifier.ai_checker import load_checker_settings, prompt_sha256
from eval.build_dataset import FULL_DIR, load_prompts
from eval.checker_routing import (CHECKED_TIERS, CLASSIFIER, REPORTS_DIR, SKIP_ABOVE, medium_answer_tokens, prepare,
                                  run_a, score_decisions, sweep_c, sweep_predict_only, sweep_selective,
                                  val_probabilities)
from eval.checker_val_run import log_path
from eval.checker_verdicts import load_verdicts, make_estimate, read_records, summarize_verdicts
from eval.pareto import TAUS
from eval.routing_sim import load_eval_rows, score
from eval.score_answers import load_answers
from eval.splits import assert_no_test_ids
from eval.baselines import always

ACTUAL_RATIO = 2.0                      # groq-large / groq-small, both input and output (config/models.yaml)
RATIOS = (ACTUAL_RATIO, 3, 5, 10, 15, 20, 30)
LEVELS = (110, 111, 112, 113)           # headline quality levels (number correct)
FAMILIES = ("always_medium", "always_high", "predict_only", "medium_then_check", "selective", "classifier_start")
CHECKER_FAMILIES = ("medium_then_check", "selective", "classifier_start")
EPS = 1e-12

CAVEATS = [
    "The whole effect rests on about 11 wrong medium answers out of 117 validation prompts (the checker flags 6, "
    "misses 4, 1 unknown, 1 false alarm); one or two prompts more or less move every crossover.",
    "Settings (tau, skip_above) are chosen on validation and reported on validation, which overstates results; "
    "an untouched split is needed before believing any number here.",
    "Only price is varied. Real providers with a large price gap also differ in the quality gap between tiers and "
    "in answer lengths (tokens), which change both who gets answers right and what each answer costs.",
    "R scales the high tier's price only; the checker stays at groq-small's price, so checking gets relatively "
    "cheaper as R grows. A pricier checker would push any crossover higher.",
]


def scale_high(rows: list, factor: float) -> list:
    """Copies of rows with the high tier's cost multiplied by factor (everything else unchanged)."""
    out = []
    for r in rows:
        c = copy.copy(r)
        c.cost = dict(r.cost)
        c.cost["high"] = r.cost["high"] * factor
        out.append(c)
    return out


def correct(point: dict) -> int:
    return round(point["quality"] * point["n"])


def cheapest_at(points: list, k: int):
    """The cheapest point with at least k correct (ties: more correct first), or None if none reaches k."""
    ok = [p for p in points if correct(p) >= k]
    return min(ok, key=lambda p: (p["cost_per_1k"], -correct(p))) if ok else None


def families_at(rows, probs, verdicts, format_ok, keys, is_comparable, tokens, factor) -> dict:
    """{family: [points]} at one price ratio; each point has its setting under "setting"."""
    priced = scale_high(rows, factor)
    out = {"always_medium": [{**score(priced, always(priced, "medium")), "setting": None}],
           "always_high": [{**score(priced, always(priced, "high")), "setting": None}],
           "predict_only": [{**p, "setting": {"tau": p["tau"]}} for p in sweep_predict_only(priced, probs, TAUS)],
           "medium_then_check": [{**score_decisions(priced, run_a(priced, verdicts, format_ok, "escalate"), verdicts),
                                  "setting": {"on_error": "escalate"}}],
           "selective": [{**p, "setting": {"skip_above": p["skip_above"], "min_answer_tokens": 0}} for p in
                         sweep_selective(priced, probs, verdicts, format_ok, tokens, on_error="escalate",
                                         skip_above=SKIP_ABOVE, min_answer_tokens=(0,))[0]],
           "classifier_start": [{**p, "setting": {"tau": p["tau"]}} for p in
                                sweep_c(priced, probs, verdicts, format_ok, keys, is_comparable, "escalate", TAUS)]}
    return out


def beats_always_medium(fams: dict):
    """Points that get more right at no higher cost than always_medium, or as many right for less."""
    m = fams["always_medium"][0]
    mk, mc = correct(m), m["cost_per_1k"]
    wins = []
    for name, pts in fams.items():
        if name == "always_medium":
            continue
        for p in pts:
            c, k = p["cost_per_1k"], correct(p)
            if (k > mk and c <= mc + EPS) or (k >= mk and c < mc - EPS):
                wins.append({"family": name, "setting": p["setting"], "correct": k,
                             "cost_x_always_medium": c / mc})
    return wins


def analyse(rows, probs, verdicts, format_ok, keys, is_comparable, tokens, ratios=RATIOS, levels=None) -> dict:
    n = len(rows)
    levels = levels or list(range(100, n + 1))
    by_ratio = {}
    for R in ratios:
        fams = families_at(rows, probs, verdicts, format_ok, keys, is_comparable, tokens, R / ACTUAL_RATIO)
        high_cost = fams["always_high"][0]["cost_per_1k"]
        med_cost = fams["always_medium"][0]["cost_per_1k"]
        table = {}
        for k in levels:
            row = {}
            for name in FAMILIES:
                p = cheapest_at(fams[name], k)
                row[name] = None if p is None else {
                    "setting": p["setting"], "correct": correct(p), "cost_per_1k": p["cost_per_1k"],
                    "pct_of_always_high": 100 * p["cost_per_1k"] / high_cost,
                    "x_always_medium": p["cost_per_1k"] / med_cost}
            checkers = [(nm, row[nm]) for nm in CHECKER_FAMILIES if row[nm]]
            best = min(checkers, key=lambda x: x[1]["cost_per_1k"]) if checkers else None
            row["best_checker"] = best and {"family": best[0], **best[1]}
            po = row["predict_only"]
            row["checker_cheaper_than_predict_only"] = (
                None if best is None else True if po is None else best[1]["cost_per_1k"] < po["cost_per_1k"] - EPS)
            table[k] = row
        sel = fams["selective"]
        by_ratio[R] = {
            "always_medium": {"correct": correct(fams["always_medium"][0]), "cost_per_1k": med_cost,
                              "pct_of_always_high": 100 * med_cost / high_cost},
            "always_high": {"correct": correct(fams["always_high"][0]), "cost_per_1k": high_cost},
            "cheapest_to_reach": table,
            "beats_always_medium": beats_always_medium(fams),
            "selective_points": [{"skip_above": p["skip_above"], "correct": correct(p),
                                  "pct_of_always_high": 100 * p["cost_per_1k"] / high_cost,
                                  "checked_share": p["checked_share"]} for p in sel],
            "best_skip_above": {k: (table[k]["selective"] or {}).get("setting", {}) and
                                table[k]["selective"]["setting"]["skip_above"] for k in LEVELS},
        }
    crossover, break_even = {}, {}
    for k in LEVELS:
        hit = [R for R in ratios if by_ratio[R]["cheapest_to_reach"][k]["checker_cheaper_than_predict_only"]]
        crossover[k] = hit[0] if hit else f"no crossover up to {max(ratios)}"
        break_even[k] = hit and break_even_ratio(by_ratio, ratios, k, hit[0])
    return {"by_ratio": by_ratio, "crossover": crossover, "break_even": break_even}


def break_even_ratio(by_ratio: dict, ratios, k: int, first: float):
    """The exact R where the two settings that cross at `first` cost the same. Each setting's cost is linear in R
    (a + b*R: only high answers scale), so two sampled ratios fix each line. None if the lines don't cross."""
    cell = by_ratio[first]["cheapest_to_reach"][k]
    pick = {"predict_only": cell["predict_only"]["setting"] if cell["predict_only"] else None,
            cell["best_checker"]["family"]: cell["best_checker"]["setting"]}
    r0, r1 = ratios[0], ratios[-1]
    if pick["predict_only"] is None:
        return None
    lines = {}
    for fam, setting in pick.items():
        costs = []
        for R in (r0, r1):
            c = by_ratio[R]["cheapest_to_reach"][k][fam]
            if not c or c["setting"] != setting:
                return None                       # the cheapest setting changes with R: no single line
            costs.append(c["cost_per_1k"])
        slope = (costs[1] - costs[0]) / (r1 - r0)
        lines[fam] = (costs[0] - slope * r0, slope)
    (pa, pb), (ca, cb) = lines["predict_only"], lines[cell["best_checker"]["family"]]
    if pb == cb:
        return None
    return {"ratio": (ca - pa) / (pb - cb), "predict_only": pick["predict_only"],
            cell["best_checker"]["family"]: cell["best_checker"]["setting"]}


def plot(result: dict, path, k: int = 113) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ratios = sorted(result["by_ratio"])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    styles = {"always_high": ("tab:green", "--", "^"), "predict_only": ("tab:blue", "-.", "o"),
              "medium_then_check": ("tab:red", "-", "s"), "selective": ("tab:purple", ":", "D"),
              "classifier_start": ("tab:orange", "-", "v")}
    for name, (color, ls, marker) in styles.items():
        xs, ys = [], []
        for R in ratios:
            cell = result["by_ratio"][R]["cheapest_to_reach"][k][name]
            if cell:
                xs.append(R)
                ys.append(cell["pct_of_always_high"])
        if xs:
            label = name.replace("_", " ") + (" (min_answer_tokens 0, best skip_above)" if name == "selective"
                                              else " (best tau at each R)" if name in ("predict_only",
                                                                                       "classifier_start") else "")
            ax.plot(xs, ys, color=color, ls=ls, marker=marker, lw=2, label=label)
    med = [result["by_ratio"][R]["always_medium"] for R in ratios]
    ax.plot(ratios, [m["pct_of_always_high"] for m in med], color="tab:brown", lw=1, ls=":", marker=".",
            label=f"always medium (only {med[0]['correct']} correct: never reaches {k})")
    ax.axvline(ACTUAL_RATIO, color="grey", lw=1.2, ls="--")
    ax.text(ACTUAL_RATIO * 1.03, 92, "actual Groq ratio (2x)", fontsize=8, va="top", color="grey")
    ax.set_xscale("log")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks(ratios)
    ax.set_xticklabels([f"{R:g}" for R in ratios])
    ax.set_xlabel("price ratio R = high-tier price / medium-tier price (log scale)")
    ax.set_ylabel(f"cheapest cost to get {k}/117 right (% of always-high at that R)")
    ax.set_title(f"Price-gap sensitivity (validation, 117 prompts): cost to reach {k} correct")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def load_inputs(model: str):
    rows = load_eval_rows("val", allow_val=True)
    assert_no_test_ids({r.id for r in rows})
    val = {r.id for r in rows}
    prompts = {p["id"]: p for p in load_prompts() if p["id"] in val}
    answers = {k: v for k, v in load_answers(FULL_DIR / "answers.jsonl").items() if k[0] in val}
    format_ok, keys, is_comparable = prepare(rows, prompts, answers)
    settings = load_checker_settings()
    check_keys = [(r.id, t) for r in rows for t in CHECKED_TIERS]
    inputs = {(r.id, t): (prompts[r.id]["messages"], (answers.get((r.id, t)) or {}).get("answer") or "")
              for r in rows for t in CHECKED_TIERS}
    path = log_path(model)
    verdicts = load_verdicts(path, check_keys, model=model, version=settings.checker_version,
                             prompt_hash=prompt_sha256(), inputs=inputs,
                             estimate=make_estimate(model, read_records(path)))
    needed = {k: verdicts[k] for k in check_keys if k[1] == "medium" and format_ok[k]}
    info = summarize_verdicts(needed)
    if info["by_reason"].get("missing"):
        raise SystemExit(f"{info['by_reason']['missing']} needed verdicts are missing")
    return (rows, val_probabilities(rows), verdicts, format_ok, keys, is_comparable,
            medium_answer_tokens(rows, answers), settings, info)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Price-gap sensitivity of checker routing (validation set).")
    parser.add_argument("--model", default=None, help="checker model whose validation verdicts to use")
    args = parser.parse_args(argv)
    settings = load_checker_settings()
    model = args.model or settings.model_name
    rows, probs, verdicts, format_ok, keys, is_comparable, tokens, settings, info = load_inputs(model)
    result = analyse(rows, probs, verdicts, format_ok, keys, is_comparable, tokens)
    report = {"split": "val", "n": len(rows), "created": datetime.now(timezone.utc).isoformat(),
              "question": "does answer-checking beat predict-only routing once the high tier is much pricier?",
              "price_ratio": {"definition": "high-tier price / medium-tier price, input and output scaled together; "
                                            "low, medium, extra and the checker keep their real price_as prices",
                              "actual": ACTUAL_RATIO, "ratios": list(RATIOS)},
              "checker": {"model": model, "version": settings.checker_version, "prompt_sha256": prompt_sha256(),
                          "needed": info},
              "classifier": f"{CLASSIFIER[0]} v{CLASSIFIER[1]}",
              "families": {"predict_only": "tau swept", "medium_then_check": "on_error=escalate",
                           "selective": f"on_error=escalate, min_answer_tokens=0, skip_above in {list(SKIP_ABOVE)}",
                           "classifier_start": "on_error=escalate, tau swept"},
              "headline_levels": list(LEVELS), "caveats": CAVEATS, **result}
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "price_sensitivity_val.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(result, REPORTS_DIR / "price_sensitivity_val.png")

    short = {"predict_only": "predict", "medium_then_check": "m+check", "selective": "selective",
             "classifier_start": "cls_start", "always_high": "always_high"}
    for k in (113, 111):
        print(f"\ncheapest cost to reach {k}/117 correct: % of always_high (x always_medium)")
        print("R      " + "".join(f"{v:>22s}" for v in short.values()))
        for R in RATIOS:
            row = result["by_ratio"][R]["cheapest_to_reach"][k]
            cells = [f"{row[f]['pct_of_always_high']:6.1f}% ({row[f]['x_always_medium']:4.2f}x)" if row[f] else "-"
                     for f in short]
            print(f"{R:<7g}" + "".join(f"{c:>22s}" for c in cells))
    print("\ncrossover (first sampled R where a checker strategy is cheaper):", result["crossover"])
    print("exact break-even R:", {k: v and round(v["ratio"], 2) for k, v in result["break_even"].items()})
    for R in RATIOS:
        b = result["by_ratio"][R]
        print(f"R={R:g}: beats always_medium: {len(b['beats_always_medium'])} points; best skip_above at 113/111: "
              f"{b['best_skip_above'][113]}/{b['best_skip_above'][111]}")
    print("saved reports/price_sensitivity_val.json and price_sensitivity_val.png")


if __name__ == "__main__":
    main()
