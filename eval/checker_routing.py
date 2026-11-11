"""Phase 4 Step 3: checker-based routing, simulated on the VALIDATION set. No API calls.

Every strategy is scored on the same validation rows with the Phase 2 answers, and pays for every call it
makes: each tier's answer at its price_as cost, and each AI check at the checker's own price_as cost
(eval/checker_verdicts.Verdict.cost). Calls in one round run in parallel (latency = the slowest); rounds
run one after another (latencies add up).

Strategies
  a. medium_then_check                medium; verify (format, then AI checker); bad -> high.
                                      on_error = escalate | accept for "unknown" verdicts.
                                      The path is eval/checker_path.medium_then_check_path (yours).
  b. medium_then_check_with_agreement like (a), but on comparable prompts (an "Answer:" line or JSON) also pay
                                      for the low answer, in parallel, and use agreement instead of the AI
                                      checker: same final answer -> medium's, different -> high
                                      (reports/PHASE_4.md, Step 1 notes). Other prompts: exactly (a).
  c. classifier_start_then_check      classifier v2 picks the start tier by tau (choose_tier, swept):
                                        low    the free format check, and on comparable prompts agreement
                                               with the extra model (llama3.2:3b, paid for in parallel);
                                               NO AI checker on low answers (PHASE_4.md: a check costs 2-6x
                                               the answer). Bad -> medium, then (a)'s checks.
                                        medium (a)
                                        high   high's answer, unchecked
  e. medium_then_check_selective     like (a), but skip the AI check (accept medium, pay nothing extra) when the
                                      classifier's p_medium >= skip_above, and optionally when medium's answer is
                                      shorter than min_answer_tokens (visible tokens: output minus reasoning).
                                      The free format check always runs. Swept over SKIP_ABOVE x MIN_ANSWER_TOKENS;
                                      skip_above 1.00 means "always check" (exactly strategy a).
  d. baselines                        always_low / always_medium / always_high, the oracle, predict-only
                                      Triage (v2, tau sweep), and random routing at the same cost as each
                                      strategy: the line through always_low, always_medium and always_high.
                                      Mixing two tiers at random in the right shares spends that much and
                                      keeps, on average, the linear mix of their quality.

Usage (after eval/checker_val_run.py has logged the verdicts):
    python -m eval.checker_routing --model groq-small
    python -m eval.checker_routing --model groq-small --allow-missing   # chart even if some checks are missing
    python -m eval.checker_routing --fake-verdicts 0.8,0.1              # plumbing only: synthetic verdicts
        (catch rate, false-alarm rate), written to data/checker_val/fake/, never to reports/

Writes reports/pareto_val.json and reports/pareto_val.png (quality retained vs cost saved; always_medium
is the line to beat). The tau presets are NOT re-picked here.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from app.registry import PROJECT_ROOT
from app.router.quality import choose_tier
from app.verifier.agreement import agree, comparable, final_key
from app.verifier.ai_checker import load_checker_settings, prompt_sha256
from app.verifier.checker import check_answer
from eval.baselines import always, oracle
from eval.build_dataset import FULL_DIR, load_prompts
from eval.checker_path import ANSWER, CHECK, ON_ERROR, Call, medium_then_check_path
from eval.checker_val_run import VAL_DIR, log_path
from eval.checker_verdicts import Verdict, load_verdicts, make_estimate, read_records, summarize_verdicts
from eval.pareto import TAUS
from eval.routing_sim import TIERS, load_eval_rows, score
from eval.score_answers import load_answers
from eval.splits import assert_no_test_ids

REPORTS_DIR = PROJECT_ROOT / "reports"
CLASSIFIER = ("classifier_c_logreg_cv", 2)
CHECKED_TIERS = ("low", "medium")
SKIP_ABOVE = (0.90, 0.92, 0.94, 0.96, 0.98, 1.00)   # 1.00 = never skip = always check
MIN_ANSWER_TOKENS = (0, 100)                         # 0 = never skip for length


# ---------- inputs: format checks, agreement keys, extra's cost ----------

def prepare(rows: list, prompts: dict, answers: dict) -> tuple:
    """(format_ok, keys, is_comparable) for the rows; also adds the extra model's cost and latency to each row.

    format_ok {(id, tier): bool} for low and medium; keys {(id, tier): final_key} for low, medium, extra.
    """
    format_ok, keys, is_comparable = {}, {}, {}
    for r in rows:
        messages = prompts[r.id]["messages"]
        is_comparable[r.id] = comparable(messages)
        extra = answers.get((r.id, "extra")) or {}
        r.cost["extra"] = float(extra.get("price_as_cost") or 0.0)
        r.latency_ms["extra"] = extra.get("latency_ms")
        for tier in ("low", "medium", "extra"):
            a = answers.get((r.id, tier)) or {}
            keys[(r.id, tier)] = final_key(messages, a.get("answer"), bool(a.get("truncated")))
            if tier in CHECKED_TIERS:
                format_ok[(r.id, tier)] = check_answer(messages, a.get("answer"), truncated=bool(a.get("truncated")),
                                                       error=a.get("error")).ok
    return format_ok, keys, is_comparable


# ---------- scoring ----------

def as_rounds(calls) -> list:
    """A flat call list (one after another) as rounds of one call each; rounds pass through unchanged."""
    return [c if isinstance(c, list) else [c] for c in calls]


def call_cost(row, call: Call, verdicts: dict) -> tuple:
    """(dollars, ms) for one call."""
    if call.kind == ANSWER:
        return row.cost[call.tier], row.latency_ms.get(call.tier) or 0.0
    if call.kind == CHECK:
        v = verdicts.get((row.id, call.tier))
        if v is None:
            raise ValueError(f"a check ran on ({row.id}, {call.tier}) but there is no verdict to charge for it; "
                             "load_verdicts returns one for every key, so pass all keys")
        return v.cost, v.latency_ms
    raise ValueError(f"unknown call kind {call.kind!r}")


def score_decisions(rows: list, decisions: list, verdicts: dict) -> dict:
    """decisions[i] = (final tier, calls) for rows[i]; calls is a flat list (sequential) or a list of rounds."""
    if len(rows) != len(decisions):
        raise ValueError("one decision per row")
    final = [d[0] for d in decisions]
    rounds = [as_rounds(d[1]) for d in decisions]
    passed = np.array([r.passed[t] for r, t in zip(rows, final)], dtype=float)
    cost, check_cost, latency, checks = [], [], [], []
    for r, rnds in zip(rows, rounds):
        priced = [[(c, *call_cost(r, c, verdicts)) for c in rnd] for rnd in rnds]
        cost.append(sum(d for rnd in priced for _, d, _ in rnd))
        check_cost.append(sum(d for rnd in priced for c, d, _ in rnd if c.kind == CHECK))
        latency.append(sum(max(ms for _, _, ms in rnd) for rnd in priced if rnd))
        checks.append(sum(c.kind == CHECK for rnd in rnds for c in rnd))
    cost, latency = np.array(cost), np.array(latency)
    strong_quality = np.mean([r.passed["high"] for r in rows])
    strong_cost = sum(r.cost["high"] for r in rows)
    answers_used = [sum(c.kind == ANSWER for rnd in rnds for c in rnd) for rnds in rounds]
    return {
        "n": len(rows),
        "quality": float(passed.mean()),
        "quality_retained": float(passed.mean() / strong_quality) if strong_quality else None,
        "cost_per_1k": float(cost.mean() * 1000),
        "cost_saved": float(1 - cost.sum() / strong_cost) if strong_cost else None,
        "check_cost_per_1k": float(np.mean(check_cost) * 1000),
        "checks_per_request": float(np.mean(checks)),
        "latency_p50_ms": float(np.percentile(latency, 50)),
        "latency_p95_ms": float(np.percentile(latency, 95)),
        "mix": {t: float(np.mean([x == t for x in final])) for t in TIERS},     # where served answers came from
        "escalation_rate": float(np.mean([n > 1 for n in answers_used])),       # paid for more than one answer
    }


# ---------- strategies: each path returns (final tier, calls or rounds) ----------

def medium_then_check_with_agreement_path(row, verdicts, format_ok, keys, is_comparable, on_error="escalate"):
    """(b): agreement between low and medium on comparable prompts; otherwise (a)."""
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}")
    if not is_comparable[row.id]:
        return medium_then_check_path(row, verdicts, format_ok, on_error)
    first = [Call(ANSWER, "low"), Call(ANSWER, "medium")]          # asked together
    if format_ok[(row.id, "medium")] and agree(keys[(row.id, "low")], keys[(row.id, "medium")]):
        return "medium", [first]
    return "high", [first, [Call(ANSWER, "high")]]                  # format fail or disagreement


def classifier_start_then_check_path(row, start, verdicts, format_ok, keys, is_comparable, on_error="escalate"):
    """(c): start at the classifier's tier; low gets format (+ agreement with extra), never the AI checker."""
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}")
    if start == "high":
        return "high", [[Call(ANSWER, "high")]]
    if start == "medium":
        return medium_then_check_path(row, verdicts, format_ok, on_error)
    if start != "low":
        raise ValueError(f"unknown start tier {start!r}")
    if is_comparable[row.id]:
        first = [Call(ANSWER, "low"), Call(ANSWER, "extra")]
        good = format_ok[(row.id, "low")] and agree(keys[(row.id, "low")], keys[(row.id, "extra")])
    else:
        first = [Call(ANSWER, "low")]
        good = format_ok[(row.id, "low")]
    if good:
        return "low", [first]
    final, calls = medium_then_check_path(row, verdicts, format_ok, on_error)
    return final, [first] + as_rounds(calls)


def answer_tokens(answer: dict):
    """Visible tokens of a Phase 2 answer: output tokens minus hidden reasoning (gpt-oss reports both).
    None when unknown (then a length rule never skips)."""
    out = (answer or {}).get("output_tokens")
    if out is None:
        return None
    return max(0, int(out) - int(answer.get("reasoning_tokens") or 0))


def medium_answer_tokens(rows: list, answers: dict) -> dict:
    return {r.id: answer_tokens(answers.get((r.id, "medium"))) for r in rows}


def skip_reason(p_medium, tokens, skip_above: float, min_answer_tokens: int):
    """Why the AI check would be skipped ("confident" / "short"), or None to run it."""
    if skip_above < 1.0 and p_medium is not None and p_medium >= skip_above:
        return "confident"
    if min_answer_tokens and tokens is not None and tokens < min_answer_tokens:
        return "short"
    return None


def medium_then_check_selective_path(row, verdicts, format_ok, *, p_medium, tokens, skip_above: float,
                                     min_answer_tokens: int = 0, on_error: str = "escalate") -> tuple:
    """(e): (final tier, calls, skipped) where skipped is None, "confident" or "short".

    The free format check still runs first: a format failure escalates as in (a), whatever the classifier says.
    Otherwise, if skip_reason() says so, medium's answer is accepted with no AI check (no extra cost);
    if not, this is exactly medium_then_check_path."""
    if on_error not in ON_ERROR:
        raise ValueError(f"on_error must be one of {ON_ERROR}")
    if format_ok[(row.id, "medium")]:
        why = skip_reason(p_medium, tokens, skip_above, min_answer_tokens)
        if why:
            return "medium", [Call(ANSWER, "medium")], why
    final, calls = medium_then_check_path(row, verdicts, format_ok, on_error)
    return final, calls, None


def selective_point(rows, decisions, skipped, verdicts) -> dict:
    """Score one (skip_above, min_answer_tokens) setting, plus what happened to the truly bad answers."""
    s = score_decisions(rows, decisions, verdicts)
    bad = [i for i, r in enumerate(rows) if not r.passed["medium"]]
    checked = [any(c.kind == CHECK for c in as_flat(d[1])) for d in decisions]
    caught = sum(decisions[i][0] != "medium" for i in bad)                 # escalated: by format or checker
    skipped_bad = sum(decisions[i][0] == "medium" and skipped[i] is not None for i in bad)
    return {**s,
            "checked_share": float(np.mean(checked)),
            "escalated_share": s["escalation_rate"],
            "skipped_confident": int(sum(x == "confident" for x in skipped)),
            "skipped_short": int(sum(x == "short" for x in skipped)),
            "bad_total": len(bad), "bad_caught": int(caught), "bad_skipped": int(skipped_bad),
            "bad_missed_by_checker": int(len(bad) - caught - skipped_bad),
            "good_escalated": int(sum(d[0] != "medium" for i, d in enumerate(decisions) if rows[i].passed["medium"]))}


def as_flat(calls) -> list:
    return [c for rnd in as_rounds(calls) for c in rnd]


def sweep_selective(rows, probs, verdicts, format_ok, tokens: dict, *, on_error="escalate",
                    skip_above=SKIP_ABOVE, min_answer_tokens=MIN_ANSWER_TOKENS) -> dict:
    """{min_answer_tokens: [one point per skip_above]} for strategy (e)."""
    out = {}
    for min_tokens in min_answer_tokens:
        points = []
        for above in skip_above:
            results = [medium_then_check_selective_path(r, verdicts, format_ok, p_medium=probs[r.id][1],
                                                        tokens=tokens.get(r.id), skip_above=above,
                                                        min_answer_tokens=min_tokens, on_error=on_error)
                       for r in rows]
            decisions = [(final, calls) for final, calls, _ in results]
            points.append({"skip_above": above, "min_answer_tokens": min_tokens,
                           **selective_point(rows, decisions, [why for _, _, why in results], verdicts)})
        out[min_tokens] = points
    return out


def run_a(rows, verdicts, format_ok, on_error):
    return [medium_then_check_path(r, verdicts, format_ok, on_error) for r in rows]


def run_b(rows, verdicts, format_ok, keys, is_comparable, on_error):
    return [medium_then_check_with_agreement_path(r, verdicts, format_ok, keys, is_comparable, on_error)
            for r in rows]


def sweep_c(rows, probs, verdicts, format_ok, keys, is_comparable, on_error, taus=TAUS) -> list:
    points = []
    for tau in taus:
        decisions = [classifier_start_then_check_path(r, choose_tier(*probs[r.id], tau), verdicts, format_ok,
                                                      keys, is_comparable, on_error) for r in rows]
        points.append({"tau": tau, **score_decisions(rows, decisions, verdicts)})
    return points


def sweep_predict_only(rows, probs, taus=TAUS) -> list:
    return [{"tau": tau, **score(rows, [choose_tier(*probs[r.id], tau) for r in rows])} for tau in taus]


# ---------- baselines and the random-at-same-cost bar ----------

def baselines(rows: list) -> dict:
    out = {f"always_{t}": score(rows, always(rows, t)) for t in TIERS}
    out["oracle"] = score(rows, oracle(rows))
    return out


def random_same_cost(cost_per_1k: float, base: dict):
    """Expected quality retained of random routing that spends cost_per_1k: a random mix of the two adjacent
    always-tier baselines (low/medium below always-medium's cost, medium/high above). None outside the range."""
    pts = sorted(((base[f"always_{t}"]["cost_per_1k"], base[f"always_{t}"]["quality_retained"]) for t in TIERS))
    for (c0, q0), (c1, q1) in zip(pts, pts[1:]):
        if c0 <= cost_per_1k <= c1:
            return q0 if c1 == c0 else q0 + (cost_per_1k - c0) / (c1 - c0) * (q1 - q0)
    return None


def with_random_bar(point: dict, base: dict) -> dict:
    bar = random_same_cost(point["cost_per_1k"], base)
    return {**point, "random_same_cost_retained": bar,
            "gain_over_random_same_cost": None if bar is None else point["quality_retained"] - bar}


def beats_always_medium(point: dict, base: dict) -> bool:
    """At always-medium's cost or less, keeps more quality than always-medium."""
    m = base["always_medium"]
    return point["cost_per_1k"] <= m["cost_per_1k"] and point["quality_retained"] > m["quality_retained"]


# ---------- everything ----------

def simulate(rows: list, probs: dict, verdicts: dict, format_ok: dict, keys: dict, is_comparable: dict,
             taus=TAUS, answer_tokens: dict = None) -> dict:
    """Every strategy and baseline on the same rows. Returns {"baselines", "fixed", "curves", "selective"}.
    answer_tokens {id: visible tokens of medium's answer}; missing or None = unknown (never skipped for length)."""
    assert_no_test_ids({r.id for r in rows})
    missing = [r.id for r in rows if r.id not in probs]
    if missing:
        raise ValueError(f"{len(missing)} rows have no classifier probabilities: {missing[:5]}")
    base = baselines(rows)
    fixed, curves = {}, {}
    for on_error in ON_ERROR:
        fixed[f"a. medium_then_check (on_error={on_error})"] = \
            score_decisions(rows, run_a(rows, verdicts, format_ok, on_error), verdicts)
        fixed[f"b. medium_then_check_with_agreement (on_error={on_error})"] = \
            score_decisions(rows, run_b(rows, verdicts, format_ok, keys, is_comparable, on_error), verdicts)
        curves[f"c. classifier_start_then_check (on_error={on_error})"] = \
            sweep_c(rows, probs, verdicts, format_ok, keys, is_comparable, on_error, taus)
    curves["d. predict-only Triage (v2)"] = sweep_predict_only(rows, probs, taus)
    selective = {f"e. medium_then_check_selective (min_answer_tokens={n})": pts for n, pts in
                 sweep_selective(rows, probs, verdicts, format_ok, answer_tokens or {}).items()}
    fixed = {k: with_random_bar(v, base) for k, v in fixed.items()}
    curves = {k: [with_random_bar(p, base) for p in pts] for k, pts in curves.items()}
    selective = {k: [with_random_bar(p, base) for p in pts] for k, pts in selective.items()}
    return {"baselines": base, "fixed": fixed, "curves": curves, "selective": selective}


def summarize_selective(result: dict) -> dict:
    """Per selective line: every point's headline numbers, and whether any point beats always-medium."""
    base = result["baselines"]
    keys = ("skip_above", "min_answer_tokens", "quality_retained", "cost_saved", "checked_share", "escalated_share",
            "bad_total", "bad_caught", "bad_skipped", "bad_missed_by_checker", "good_escalated")
    return {name: {"beats_always_medium": any(beats_always_medium(p, base) for p in pts),
                   "points": [{k: p[k] for k in keys} for p in pts]}
            for name, pts in result.get("selective", {}).items()}


def summarize(result: dict) -> dict:
    """Per strategy: does it beat always-medium anywhere, and its best quality at always-medium's cost."""
    base, out = result["baselines"], {}
    for name, r in result["fixed"].items():
        out[name] = {"beats_always_medium": beats_always_medium(r, base), "quality_retained": r["quality_retained"],
                     "cost_saved": r["cost_saved"], "gain_over_random_same_cost": r["gain_over_random_same_cost"]}
    for name, pts in result["curves"].items():
        cheap = [p for p in pts if p["cost_per_1k"] <= base["always_medium"]["cost_per_1k"]]
        best = max(cheap, key=lambda p: (p["quality"], -p["cost_per_1k"])) if cheap else None
        out[name] = {"beats_always_medium": any(beats_always_medium(p, base) for p in pts),
                     "best_at_always_medium_cost": best and {k: best[k] for k in
                                                             ("tau", "quality_retained", "cost_saved", "mix")}}
    return out


def plot(result: dict, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base = result["baselines"]
    fig, ax = plt.subplots(figsize=(10, 6.5))
    pct = lambda v: None if v is None else v * 100
    m = base["always_medium"]
    ax.axhline(pct(m["quality_retained"]), color="tab:brown", lw=1.2, ls=":", zorder=1)
    ax.axvline(pct(m["cost_saved"]), color="tab:brown", lw=1.2, ls=":", zorder=1)
    ax.fill_between([pct(m["cost_saved"]), 100], pct(m["quality_retained"]), 110, color="tab:brown", alpha=0.06,
                    zorder=0, label="beats always-medium (cheaper and better)")
    pts = sorted((base[f"always_{t}"]["cost_saved"], base[f"always_{t}"]["quality_retained"]) for t in TIERS)
    ax.plot([pct(x) for x, _ in pts], [pct(y) for _, y in pts], ls="--", color="grey", lw=1,
            label="random routing at the same cost")
    for name, points in result["curves"].items():
        ax.plot([pct(p["cost_saved"]) for p in points], [pct(p["quality_retained"]) for p in points], lw=2,
                label=name, ls="-" if "predict-only" not in name else "-.")
    for name, points in result.get("selective", {}).items():
        ax.plot([pct(p["cost_saved"]) for p in points], [pct(p["quality_retained"]) for p in points], lw=1.6,
                ls=":", marker=".", label=f"{name}, skip_above {points[0]['skip_above']:.2f}-{points[-1]['skip_above']:.2f}")
    fixed_markers = ["o", "P", "D", "X"]
    for (name, r), marker in zip(result["fixed"].items(), fixed_markers):
        ax.scatter(pct(r["cost_saved"]), pct(r["quality_retained"]), marker=marker, s=80, zorder=6,
                   edgecolors="black", linewidths=0.6, label=name)
    markers = {"always_low": ("v", "tab:purple", 110), "always_medium": ("s", "tab:brown", 220),
               "always_high": ("^", "tab:green", 110), "oracle": ("*", "gold", 200)}
    for name, (marker, color, size) in markers.items():
        b = base[name]
        label = {"oracle": "perfect router (oracle)", "always_medium": "always medium: the line to beat"}.get(
            name, name.replace("_", " "))
        ax.scatter(pct(b["cost_saved"]), pct(b["quality_retained"]), marker=marker, s=size, zorder=7, color=color,
                   edgecolors="black", linewidths=1.4 if name == "always_medium" else 0.6, label=label)
    ax.set_xlabel("cost saved vs always-strong (%, Groq paid prices, checks included)")
    ax.set_ylabel("quality retained (% of always-strong)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


# ---------- real data ----------

def val_probabilities(rows: list) -> dict:
    from eval.pareto import predictor_probabilities
    return predictor_probabilities(*CLASSIFIER, rows)


def fake_verdicts(rows: list, catch: float, false_alarm: float, model: str, seed: int = 0) -> dict:
    """Synthetic verdicts from the truth, for plumbing only: a truly bad answer is flagged with probability
    `catch`, a truly good one with `false_alarm`. Costs are estimates."""
    rng = np.random.default_rng(seed)
    est = make_estimate(model)
    out = {}
    for r in rows:
        for tier in CHECKED_TIERS:
            bad = rng.random() < (catch if not r.passed[tier] else false_alarm)
            out[(r.id, tier)] = Verdict("bad" if bad else "good", "fake", est.cost(), est.latency_ms, estimated=True)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Simulate checker-based routing on the validation set.")
    parser.add_argument("--model", default=None, help="checker model whose validation verdicts to use")
    parser.add_argument("--allow-missing", action="store_true", help="chart even if some verdicts are missing")
    parser.add_argument("--fake-verdicts", default=None, metavar="CATCH,FALSE_ALARM",
                        help="synthetic verdicts for plumbing; output goes to data/checker_val/fake/")
    args = parser.parse_args(argv)

    rows = load_eval_rows("val", allow_val=True)
    val = {r.id for r in rows}
    prompts = {p["id"]: p for p in load_prompts() if p["id"] in val}
    answers = {k: v for k, v in load_answers(FULL_DIR / "answers.jsonl").items() if k[0] in val}
    format_ok, keys, is_comparable = prepare(rows, prompts, answers)
    settings = load_checker_settings()
    model = args.model or settings.model_name
    check_keys = [(r.id, t) for r in rows for t in CHECKED_TIERS]

    if args.fake_verdicts:
        catch, false_alarm = (float(x) for x in args.fake_verdicts.split(","))
        verdicts, out_dir, tag = fake_verdicts(rows, catch, false_alarm, model), VAL_DIR / "fake", " FAKE VERDICTS"
    else:
        inputs = {(r.id, t): (prompts[r.id]["messages"], (answers.get((r.id, t)) or {}).get("answer") or "")
                  for r in rows for t in CHECKED_TIERS}
        path = log_path(model)
        verdicts = load_verdicts(path, check_keys, model=model, version=settings.checker_version,
                                 prompt_hash=prompt_sha256(), inputs=inputs,
                                 estimate=make_estimate(model, read_records(path)))
        out_dir, tag = REPORTS_DIR, ""
    # Only verdicts a strategy can actually ask for matter: medium answers that pass the format checker
    needed = {k: verdicts[k] for k in check_keys if k[1] == "medium" and format_ok[k]}
    info = summarize_verdicts(needed)
    print(f"{len(rows)} validation prompts | checker {model} v{settings.checker_version} | medium verdicts needed: "
          f"{info['n']} ({info['good']} good, {info['bad']} bad, {info['unknown']} unknown: {info['by_reason']})")
    if not args.fake_verdicts and info["by_reason"].get("missing") and not args.allow_missing:
        raise SystemExit(f"{info['by_reason']['missing']} needed verdicts are missing; run eval.checker_val_run "
                         f"--model {model} first, or pass --allow-missing")

    result = simulate(rows, val_probabilities(rows), verdicts, format_ok, keys, is_comparable,
                      answer_tokens=medium_answer_tokens(rows, answers))
    report = {"split": "val", "n": len(rows), "created": datetime.now(timezone.utc).isoformat(),
              "fake_verdicts": args.fake_verdicts, "checker": {"model": model, "version": settings.checker_version,
                                                               "prompt_sha256": prompt_sha256(), "needed": info},
              "classifier": f"{CLASSIFIER[0]} v{CLASSIFIER[1]}", "summary": summarize(result),
              "selective_summary": summarize_selective(result), **result}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pareto_val.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot(result, out_dir / "pareto_val.png",
         f"Checker routing vs predict-only (validation, {len(rows)} prompts){tag}")
    for name, s in report["summary"].items():
        print(f"  {name}: beats always-medium: {'yes' if s['beats_always_medium'] else 'no'}")
    for name, s in report["selective_summary"].items():
        print(f"  {name}:")
        for p in s["points"]:
            print(f"    skip_above {p['skip_above']:.2f}: kept {p['quality_retained']:.1%}, saved {p['cost_saved']:.1%}, "
                  f"checked {p['checked_share']:.0%}, escalated {p['escalated_share']:.0%}, bad answers: "
                  f"{p['bad_caught']} caught / {p['bad_skipped']} skipped / {p['bad_missed_by_checker']} missed "
                  f"of {p['bad_total']}")
    print(f"saved {(out_dir / 'pareto_val.json').relative_to(PROJECT_ROOT)} and pareto_val.png")


if __name__ == "__main__":
    main()
