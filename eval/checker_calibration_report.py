"""Phase 4 Step 2, part 3: how good is the AI checker? (no API calls)

Usage:
    python -m eval.checker_calibration_report                       # every model with a log
    python -m eval.checker_calibration_report --models groq-small

Reads data/checker_calibration/sample.json and the per-model logs written by eval.checker_calibration, keeps
only results made by the current checker_version and prompt, and writes reports/checker_calibration.json.

"Positive" = the checker said not good. Confusion counts and rates come from eval.checker_metrics.checker_metrics:
    catch rate    of truly bad answers, the share the checker said not good
    trust rate    when the checker said not good, the share truly bad
    false alarms  of truly good answers, the share the checker said not good
Rows where the checker returned an error are left out of those (the error rate is reported on its own); results
still pending (provider errors, not run yet) are counted, never treated as verdicts.

Every rate has a Wilson 95% interval. Breakdowns: tier, tier x truth source, tier x dataset, the non-comparable
slice (prompts agreement can't handle, where the AI checker does its real work), and tier x task type (counts and
intervals in alphabetical order: strata are small, so they are not ranked).

The sample over-represents truly bad answers and draws different shares of each stratum, so:
    * catch rate and false alarms within one (tier, truth source) stratum need no correction
    * tier-level "pool-weighted" catch / false-alarm rates weight each truth-source stratum by its pool size
    * the trust rate in the sample is inflated by oversampling; it is reported raw (labelled) and reweighted to a
      base rate pi of bad answers:  trust = pi*catch / (pi*catch + (1 - pi)*false_alarm), using the
      pool-weighted rates, with a stratified bootstrap 95% interval. Medium: pi = 15% (all format-passing medium
      answers) and 22% (non-comparable ones); low: pi = 50%.
Also: the flip rate from the repeat runs, and groq-small vs groq-large on exact-match items only (judged labels
come from gpt-oss-120b, the same model as groq-large, so judged items are not a fair comparison).
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Optional

from app.registry import PROJECT_ROOT
from app.verifier.ai_checker import load_checker_settings, prompt_sha256
from eval.check_log import CheckLog, is_finished
from eval.checker_calibration import load_sample, log_path
from eval.checker_calibration_sample import CALIB_DIR, SAMPLE_PATH
from eval.checker_metrics import checker_metrics

REPORT_PATH = PROJECT_ROOT / "reports" / "checker_calibration.json"
MODELS = ("groq-small", "groq-large")
BASE_RATES = {"medium": {"all format-passing medium answers": 0.15, "non-comparable medium answers": 0.22},
              "low": {"all format-passing low answers": 0.50}}
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0


# ---------- statistics ----------

def wilson(k: int, n: int, z: float = 1.96) -> Optional[list]:
    """Wilson 95% interval [low, high] for k successes out of n, or None when n is 0."""
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def reweighted_trust(pi: float, catch: Optional[float], false_alarm: Optional[float]) -> Optional[float]:
    """Trust rate at base rate pi of truly bad answers, from the catch and false-alarm rates."""
    if catch is None or false_alarm is None:
        return None
    flagged = pi * catch + (1 - pi) * false_alarm
    return None if flagged == 0 else pi * catch / flagged


def block(rows: list) -> dict:
    """checker_metrics plus Wilson intervals, for rows that all have a verdict."""
    m = checker_metrics(rows)
    m = dict(m)
    m["catch_rate_ci"] = wilson(m["tp"], m["tp"] + m["fn"])
    m["trust_rate_ci"] = wilson(m["tp"], m["tp"] + m["fp"])
    m["false_alarm_ci"] = wilson(m["fp"], m["fp"] + m["tn"])
    return m


# ---------- rows ----------

def verdict_rows(items: list, latest: dict, model: str, version: int, sha: str) -> list:
    """One row per sample item: the item, plus state (ok, error, pending, missing) and checker_bad."""
    rows = []
    for c in items:
        r = latest.get((c["prompt_id"], c["tier"], model, version))
        if r is not None and r.get("prompt_sha256") != sha:
            r = None                                    # made with another prompt text: not a current result
        if r is None:
            state = "missing"
        elif not is_finished(r):
            state = "pending"
        else:
            state = "ok" if r["status"] == "ok" else "error"
        rows.append({**c, "state": state, "error_kind": r.get("error_kind") if r else None,
                     "checker_bad": (not r["good"]) if state == "ok" else None})
    return rows


def ok_rows(rows: list) -> list:
    return [r for r in rows if r["state"] == "ok"]


def error_rate(rows: list) -> dict:
    done = [r for r in rows if r["state"] in ("ok", "error")]
    errors = [r for r in done if r["state"] == "error"]
    return {"n": len(done), "errors": len(errors),
            "error_rate": len(errors) / len(done) if done else None, "error_rate_ci": wilson(len(errors), len(done)),
            "by_kind": dict(Counter(r["error_kind"] for r in errors)),
            "pending": sum(r["state"] == "pending" for r in rows),
            "missing": sum(r["state"] == "missing" for r in rows)}


def grouped(rows: list, key) -> dict:
    """{group label: metrics block}, labels in sorted order (not ranked)."""
    groups = {}
    for r in ok_rows(rows):
        groups.setdefault(key(r), []).append(r)
    return {label: block(groups[label]) for label in sorted(groups)}


# ---------- pool weighting and reweighted trust ----------

def pool_weighted(rows: list, tier: str, pool_counts: dict) -> dict:
    """Catch and false-alarm rates for a tier, each truth-source stratum weighted by its pool size."""
    out = {}
    for rate, side, num, den in (("catch_rate", "bad", "tp", ("tp", "fn")),
                                 ("false_alarm", "good", "fp", ("fp", "tn"))):
        total, weight = 0.0, 0
        for source in ("exact", "judged"):
            sub = [r for r in ok_rows(rows) if r["tier"] == tier and r["truth_source"] == source]
            if not sub:
                continue
            m = checker_metrics(sub)
            if m[den[0]] + m[den[1]] == 0:
                continue
            n_pool = pool_counts.get(f"{tier}/{source}/{side}", 0)
            total += n_pool * m[num] / (m[den[0]] + m[den[1]])
            weight += n_pool
        out[rate] = total / weight if weight else None
    return out


def bootstrap_trust(rows: list, tier: str, pool_counts: dict, pis: list, reps: int = BOOTSTRAP_REPS,
                    seed: int = BOOTSTRAP_SEED) -> dict:
    """Stratified bootstrap 95% interval of the reweighted trust rate at each base rate in pis."""
    rng = random.Random(seed)
    strata = {}
    for r in ok_rows(rows):
        if r["tier"] == tier:
            strata.setdefault((r["truth_source"], r["truth_bad"]), []).append(r)
    draws = {pi: [] for pi in pis}
    for _ in range(reps):
        sample = [rng.choice(items) for items in strata.values() for _ in items]
        w = pool_weighted(sample, tier, pool_counts)
        for pi in pis:
            t = reweighted_trust(pi, w["catch_rate"], w["false_alarm"])
            if t is not None:
                draws[pi].append(t)
    out = {}
    for pi, values in draws.items():
        values.sort()
        out[pi] = [values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]] if values else None
    return out


def trust_section(rows: list, tier: str, pool_counts: dict) -> dict:
    weighted = pool_weighted(rows, tier, pool_counts)
    raw = block([r for r in ok_rows(rows) if r["tier"] == tier]) if any(r["tier"] == tier for r in ok_rows(rows)) else None
    pis = BASE_RATES[tier]
    intervals = bootstrap_trust(rows, tier, pool_counts, list(pis.values()))
    return {
        "pool_weighted_catch_rate": weighted["catch_rate"], "pool_weighted_false_alarm": weighted["false_alarm"],
        "trust_rate_raw_sample (inflated by oversampling bad answers)": raw["trust_rate"] if raw else None,
        "trust_rate_reweighted": {f"{label} (base rate {pi:.0%})": {
            "trust_rate": reweighted_trust(pi, weighted["catch_rate"], weighted["false_alarm"]),
            "bootstrap_95_ci": intervals[pi]} for label, pi in pis.items()},
    }


# ---------- repeat runs and model comparison ----------

def flip_rate(main: list, repeat: list) -> dict:
    """How often the verdict changed between the main run and the repeat run (items with a verdict in both)."""
    first = {(r["prompt_id"], r["tier"]): r for r in main}
    pairs = [(first[(r["prompt_id"], r["tier"])], r) for r in repeat if (r["prompt_id"], r["tier"]) in first]
    both = [(a, b) for a, b in pairs if a["state"] == "ok" and b["state"] == "ok"]
    flips = sum(a["checker_bad"] != b["checker_bad"] for a, b in both)
    return {"n_repeated": len(pairs), "n_with_both_verdicts": len(both), "flips": flips,
            "flip_rate": flips / len(both) if both else None, "flip_rate_ci": wilson(flips, len(both)),
            "error_in_either_run": sum((a["state"] == "error") or (b["state"] == "error") for a, b in pairs),
            "not_finished_in_either_run": sum(a["state"] in ("pending", "missing") or b["state"] in ("pending", "missing")
                                              for a, b in pairs)}


def compare_models(rows_a: list, rows_b: list, name_a: str, name_b: str) -> dict:
    """Both models on exact-match items only, restricted to items where both gave a verdict."""
    b_by = {(r["prompt_id"], r["tier"]): r for r in rows_b}
    out = {}
    for tier in ("medium", "low"):
        pairs = [(a, b_by[(a["prompt_id"], a["tier"])]) for a in rows_a
                 if a["tier"] == tier and a["truth_source"] == "exact" and (a["prompt_id"], a["tier"]) in b_by]
        pairs = [(a, b) for a, b in pairs if a["state"] == "ok" and b["state"] == "ok"]
        if not pairs:
            out[tier] = {"n": 0}
            continue
        right = lambda r: r["checker_bad"] == r["truth_bad"]
        out[tier] = {
            "n": len(pairs), name_a: block([a for a, _ in pairs]), name_b: block([b for _, b in pairs]),
            "verdicts_agree": sum(a["checker_bad"] == b["checker_bad"] for a, b in pairs) / len(pairs),
            f"only_{name_a}_right": sum(right(a) and not right(b) for a, b in pairs),
            f"only_{name_b}_right": sum(right(b) and not right(a) for a, b in pairs),
        }
    return out


# ---------- one model ----------

def model_report(rows: list, pool_counts: dict) -> dict:
    tiers = sorted({r["tier"] for r in rows}, key=lambda t: ("medium", "low").index(t))
    return {
        "errors": {"all": error_rate(rows), **{t: error_rate([r for r in rows if r["tier"] == t]) for t in tiers}},
        "by_tier (sample, unweighted)": grouped(rows, lambda r: r["tier"]),
        "by_tier_and_truth_source": grouped(rows, lambda r: f"{r['tier']}/{r['truth_source']}"),
        "by_tier_and_dataset": grouped(rows, lambda r: f"{r['tier']}/{r['dataset']}"),
        "non_comparable_slice": {
            "by_tier": grouped([r for r in rows if not r["comparable"]], lambda r: r["tier"]),
            "by_tier_and_truth_source": grouped([r for r in rows if not r["comparable"]],
                                                lambda r: f"{r['tier']}/{r['truth_source']}"),
        },
        "by_tier_and_task_type (counts and intervals; alphabetical, not ranked)":
            grouped(rows, lambda r: f"{r['tier']}/{r['task_type']}"),
        "trust_reweighted": {t: trust_section(rows, t, pool_counts) for t in tiers if ok_rows(rows)},
    }


def build_report(models=MODELS, sample_path: Path = SAMPLE_PATH, calib_dir: Path = CALIB_DIR, settings=None) -> dict:
    settings = settings or load_checker_settings()
    sha = prompt_sha256()
    sample = load_sample(sample_path)
    items = sample["items"]
    repeat_ids = {tuple(x) for x in sample["repeat_items"]}
    report = {"checker_version": settings.checker_version, "prompt_sha256": sha, "sample_size": len(items),
              "sample_counts": sample["sample_counts"], "pool_counts": sample["pool_counts"],
              "positive": "the checker said not good", "models": {}}
    rows_by_model = {}
    for model in models:
        path = log_path(model, False, calib_dir)
        if not path.exists():
            continue
        rows = verdict_rows(items, CheckLog(path).latest(), model, settings.checker_version, sha)
        rows_by_model[model] = rows
        section = model_report(rows, sample["pool_counts"])
        repeat_path = log_path(model, True, calib_dir)
        if repeat_path.exists():
            repeats = verdict_rows([c for c in items if (c["prompt_id"], c["tier"]) in repeat_ids],
                                   CheckLog(repeat_path).latest(), model, settings.checker_version, sha)
            section["repeat_runs"] = flip_rate(rows, repeats)
        report["models"][model] = section
    if all(m in rows_by_model for m in MODELS):
        report["groq-small vs groq-large (exact-match items only)"] = compare_models(
            rows_by_model["groq-small"], rows_by_model["groq-large"], "groq-small", "groq-large")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Report how well the AI checker did on the calibration sample.")
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    args = parser.parse_args(argv)
    report = build_report(args.models)
    if not report["models"]:
        raise SystemExit("no checker results yet; run eval.checker_calibration first")
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for model, section in report["models"].items():
        print(f"\n{model}: errors {section['errors']['all']}")
        for tier, m in section["by_tier (sample, unweighted)"].items():
            print(f"  {tier}: n={m['n']} catch {m['catch_rate']} false alarm {m['false_alarm']}")
    print(f"\nsaved {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
