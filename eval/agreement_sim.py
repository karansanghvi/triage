"""Agreement cascades, simulated with the Phase 2 answers (no API calls).

Strategies (comparable prompts use agreement; others fall back to "medium, escalate to high if the
format checker flags it", the best format-only cascade):

    agree low+medium          ask low and medium together. Same final answer -> use medium's.
                              Different -> also ask high.
    agree low+extra           ask low and the local extra model together. Same -> use low's (cheap!).
                              Different -> medium, escalating to high if the format checker flags it.
    3-way                     ask low and extra. Same -> low's. Different -> ask medium; if medium matches
                              either of them -> medium's; otherwise -> high.

Calls made together run in parallel (latency = the slower one); later rounds add up.
Every call is paid for. "extra" is llama3.2:3b, run locally in Phase 2; its price_as_cost prices it like a
small paid model, which is the honest choice if you didn't host it yourself.

Usage:
    python -m eval.agreement_sim
"""
import json

import numpy as np

from app.registry import PROJECT_ROOT
from app.verifier.agreement import agree, comparable, final_key
from eval.baselines import all_baselines
from eval.build_dataset import FULL_DIR, load_prompts
from eval.cascade import check_all, print_table
from eval.routing_sim import TIERS, load_eval_rows
from eval.score_answers import load_answers

REPORTS_DIR = PROJECT_ROOT / "reports"
CALLED = ("low", "medium", "high", "extra")


def load_keys(rows: list) -> tuple:
    """({prompt_id: comparable?}, {(prompt_id, tier): final_key}); also adds extra's cost/latency to rows."""
    prompts = {p["id"]: p for p in load_prompts()}
    answers = load_answers(FULL_DIR / "answers.jsonl")
    is_comparable, keys = {}, {}
    for r in rows:
        messages = prompts[r.id]["messages"]
        is_comparable[r.id] = comparable(messages)
        extra = answers.get((r.id, "extra")) or {}
        r.cost["extra"] = float(extra.get("price_as_cost") or 0.0)
        r.latency_ms["extra"] = extra.get("latency_ms")
        for tier in CALLED:
            a = answers.get((r.id, tier)) or {}
            keys[(r.id, tier)] = final_key(messages, a.get("answer"), bool(a.get("truncated")))
    return is_comparable, keys


# ---------- strategies: each returns (final tier, rounds of parallel calls) ----------

def format_fallback(row, checks):
    """Non-comparable prompts: medium, escalate to high if the format checker flags medium's answer."""
    if checks[(row.id, "medium")].ok:
        return "medium", [["medium"]]
    return "high", [["medium"], ["high"]]


def agree_low_medium(row, keys, checks, is_comparable):
    if not is_comparable[row.id]:
        return format_fallback(row, checks)
    if agree(keys[(row.id, "low")], keys[(row.id, "medium")]):
        return "medium", [["low", "medium"]]
    return "high", [["low", "medium"], ["high"]]


def agree_low_extra(row, keys, checks, is_comparable):
    if not is_comparable[row.id]:
        return format_fallback(row, checks)
    if agree(keys[(row.id, "low")], keys[(row.id, "extra")]):
        return "low", [["low", "extra"]]
    final, rounds = format_fallback(row, checks)
    return final, [["low", "extra"]] + rounds


def three_way(row, keys, checks, is_comparable):
    if not is_comparable[row.id]:
        return format_fallback(row, checks)
    low, extra, medium = (keys[(row.id, t)] for t in ("low", "extra", "medium"))
    if agree(low, extra):
        return "low", [["low", "extra"]]
    if agree(medium, low) or agree(medium, extra):
        return "medium", [["low", "extra"], ["medium"]]
    return "high", [["low", "extra"], ["medium"], ["high"]]


STRATEGIES = {"agree low+medium": agree_low_medium, "agree low+extra": agree_low_extra, "3-way agreement": three_way}


def score_rounds(rows: list, decisions: list) -> dict:
    final = [d[0] for d in decisions]
    passed = np.array([r.passed[t] for r, t in zip(rows, final)], dtype=float)
    cost = np.array([sum(r.cost[t] for rnd in rounds for t in rnd) for r, (_, rounds) in zip(rows, decisions)])
    latency = np.array([sum(max((r.latency_ms[t] or 0) for t in rnd) for rnd in rounds)
                        for r, (_, rounds) in zip(rows, decisions)])
    strong_quality = np.mean([r.passed["high"] for r in rows])
    strong_cost = sum(r.cost["high"] for r in rows)
    return {
        "n": len(rows), "quality": float(passed.mean()),
        "quality_retained": float(passed.mean() / strong_quality),
        "cost_per_1k": float(cost.mean() * 1000), "cost_saved": float(1 - cost.sum() / strong_cost),
        "latency_p50_ms": float(np.percentile(latency, 50)), "latency_p95_ms": float(np.percentile(latency, 95)),
        "mix": {t: float(np.mean([x == t for x in final])) for t in TIERS},
        "escalation_rate": float(np.mean([len(rounds) > 1 for _, rounds in decisions])),
    }


# ---------- how good is agreement as a signal? ----------

def agreement_report(rows: list, keys: dict, is_comparable: dict) -> dict:
    comp = [r for r in rows if is_comparable[r.id]]
    out = {"comparable_share": len(comp) / len(rows), "by_source": {}}
    for a, b, judged in (("low", "medium", "medium"), ("low", "extra", "low")):
        same = np.array([agree(keys[(r.id, a)], keys[(r.id, b)]) for r in comp])
        good = np.array([r.passed[judged] for r in comp])
        out[f"{a}+{b}"] = {
            "agree_rate": float(same.mean()),
            f"when_agree_{judged}_really_passed": float(good[same].mean()) if same.any() else None,
            f"{judged}_failures_caught": float((~same)[~good].mean()) if (~good).any() else None,
        }
    for source in sorted({r.source for r in rows}):
        src = [r for r in rows if r.source == source]
        out["by_source"][source] = sum(is_comparable[r.id] for r in src) / len(src)
    return out


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    rows = load_eval_rows("train")
    is_comparable, keys = load_keys(rows)
    checks = check_all(rows, use_constraints=False)
    report = agreement_report(rows, keys, is_comparable)

    print(f"{len(rows)} labeled train prompts | {report['comparable_share']:.0%} have a comparable final answer ("
          + ", ".join(f"{s} {v:.0%}" for s, v in report["by_source"].items()) + ")\n")
    print("How good is agreement as a signal? (comparable prompts only)")
    lm, le = report["low+medium"], report["low+extra"]
    print(f"  low & medium agree on {lm['agree_rate']:.0%} | when they agree, medium really passed "
          f"{lm['when_agree_medium_really_passed']:.0%} | catches {lm['medium_failures_caught']:.0%} of medium's failures")
    print(f"  low & extra  agree on {le['agree_rate']:.0%} | when they agree, low really passed "
          f"{le['when_agree_low_really_passed']:.0%} | catches {le['low_failures_caught']:.0%} of low's failures\n")

    baselines = all_baselines(rows)
    results = {k: baselines[k] for k in ("always_low", "always_medium", "always_high", "oracle")}
    for name, strategy in STRATEGIES.items():
        results[name] = score_rounds(rows, [strategy(r, keys, checks, is_comparable) for r in rows])
    print_table(results)

    # Fair "same cost" bar: blindly mixing always-medium and always-high in the proportion that spends the
    # same as the strategy (a straight line between those two points on the chart).
    medium, high = baselines["always_medium"], baselines["always_high"]
    print()
    for name in STRATEGIES:
        r = results[name]
        share = (r["cost_per_1k"] - medium["cost_per_1k"]) / (high["cost_per_1k"] - medium["cost_per_1k"])
        share = min(max(share, 0.0), 1.0)
        bar = medium["quality_retained"] + share * (high["quality_retained"] - medium["quality_retained"])
        if r["cost_per_1k"] < medium["cost_per_1k"]:
            bar = None   # cheaper than always-medium: compare its quality with always-medium's directly
        gap = None if bar is None else (r["quality_retained"] - bar) * 100
        where = ("cheaper than always-medium" if bar is None else
                 f"{gap:+.1f} points vs blindly mixing medium+high at the same cost")
        print(f"{name}: {r['quality_retained']:.1%} kept, saves {r['cost_saved']:.0%} -> {where}")
    print(f"(always-medium: {medium['quality_retained']:.1%} kept, saves {medium['cost_saved']:.0%})")

    (REPORTS_DIR / "agreement_train.json").write_text(json.dumps({"signal": report, "results": results}, indent=2))


if __name__ == "__main__":
    main()