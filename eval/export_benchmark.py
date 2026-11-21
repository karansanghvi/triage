"""The dashboard's Benchmark tab: one curated JSON from the finished evaluations. No API calls, no tuning.

    venv/bin/python -m eval.export_benchmark
        writes web/src/content/benchmark.json (the dashboard imports it at build time)
        and reports/benchmark_v2_test_curve.json (the same curve and references, kept with the reports)

What it reads:
  * the locked test split (149 labeled prompts), with every tier's recorded pass/fail and paid-price cost;
  * classifier v2's probabilities for those prompts, from the real predictor (no API calls: the embedding model
    must already be cached), exactly as eval/final_test_phase4.py computed them;
  * reports/final_test_phase4.json, reports/judge_calibration.json and the classifier table in reports/PHASE_3.md.

The headline (tau 0.86, the balanced preset) is the pre-registered Phase 4 result. The full v2 curve was
computed afterwards, for reporting only: it is labeled so, and nothing may be tuned on it. This script never
writes the test-set markers (reports/*_DONE.json) or runs the --final scripts. Before writing, it checks that its
tau 0.86 point and the baselines reproduce final_test_phase4.json exactly; any difference stops it.

Needs files that aren't in git (data/answers/answers.jsonl, data/outcomes.jsonl, the cached embedding model), so it
runs on a machine that has them, and its outputs are committed.
"""
import json
import re
from pathlib import Path

from app.registry import PROJECT_ROOT
from app.router.quality import choose_tier, load_quality_settings
from eval.baselines import always, oracle
from eval.routing_sim import TIERS, load_eval_rows, score

REPORTS_DIR = PROJECT_ROOT / "reports"
OUT_PATH = PROJECT_ROOT / "web" / "src" / "content" / "benchmark.json"
REPORT_COPY = REPORTS_DIR / "benchmark_v2_test_curve.json"
CLASSIFIER = ("classifier_c_logreg_cv", 2)
HEADLINE_TAU = 0.86
TAUS = [round(i / 100, 2) for i in range(101)]
DIGITS = 6
SCHEMA = 1
CURVE_PROVENANCE = ("computed after the pre-registered test, for reporting only; not used for tuning")
HEADLINE_PROVENANCE = "pre-registered (reports/PHASE4_PREREGISTRATION.json; scored in reports/final_test_phase4.json)"

# From the written record (reports/PHASE_2.md, PHASE_3.md, PHASE_4.md, config/collection.yaml), cited per value
DATASET_FACTS = {
    "answers_collected": {"from": "2026-10-19", "to": "2026-10-22", "source": "reports/PHASE_2.md"},
    "answers_judged": {"from": "2026-10-24", "to": "2026-11-02", "source": "reports/PHASE_2.md"},
    "sources": {"own (hand-written)": 162, "GSM8K": 200, "MMLU": 200, "Chatbot Arena": 200},
    "lineup": {"low": "ALLaM 2 7B (groq-allam)", "medium": "gpt-oss-20b (groq-small, reasoning effort low)",
               "high": "gpt-oss-120b (groq-large, reasoning effort low)", "provider": "Groq",
               "source": "config/collection.yaml"},
    "prices": "Groq's published paid prices per token (price_as in config/models.yaml); ALLaM 2 7B's price is "
              "pending, so it borrows Llama 3.1 8B's last price",
}
CAVEATS = [
    "The test set is small (149 prompts): differences of one to three points are within noise.",
    "Every tier is a Groq model (ALLaM 2 7B, gpt-oss-20b, gpt-oss-120b). The live router's backups on other "
    "providers (Gemini, OpenAI, Anthropic) were never benchmarked.",
    "Costs are estimates at published paid prices; the runs themselves were on free tiers. ALLaM 2 7B borrows "
    "Llama 3.1 8B's last price because its own is pending.",
    "Quality kept is pass rate relative to always using the strongest model. It can exceed 100%, because a "
    "cheaper model sometimes passes where the strongest fails.",
    "The cheap and high preset values (tau 0.68 and 0.89) were chosen for classifier v1 and haven't been re-tuned "
    "for v2. Their test-set points come from the curve, which was computed after the pre-registered test.",
    "Against always using the cheapest model, Triage costs more, not less: it buys back the quality the cheapest "
    "model loses.",
    "Open-ended answers were graded by an AI judge, which agreed with known answers 70% of the time (mostly by "
    "failing correct answers over a literal format rule). Math and multiple-choice labels are exact-match.",
    "This test set was used twice: once in Phase 3 (classifier v1) and once, pre-registered, in Phase 4 (v2). "
    "Nothing was tuned on it.",
]


class ExportError(SystemExit):
    pass


# ---------- pieces ----------

def r(x):
    return None if x is None else round(float(x), DIGITS) + 0.0      # + 0.0: no "-0.0" in the JSON


def point(rows: list, tiers: list, **extra) -> dict:
    s = score(rows, tiers)
    return {**extra, "quality_kept": r(s["quality_retained"]), "cost_saved": r(s["cost_saved"]),
            "cost_per_1k_usd": r(s["cost_per_1k"]), "correct": round(s["quality"] * s["n"]), "n": s["n"],
            "mix": {t: r(s["mix"][t]) for t in TIERS}}


def curve(rows: list, probs: dict) -> list:
    return [point(rows, [choose_tier(*probs[row.id], tau) for row in rows], tau=tau) for tau in TAUS]


def references(rows: list) -> dict:
    return {"always_cheapest": point(rows, always(rows, "low")), "always_medium": point(rows, always(rows, "medium")),
            "always_strongest": point(rows, always(rows, "high")), "oracle": point(rows, oracle(rows))}


def check_reproduces(headline: dict, refs: dict, saved: dict, eps: float = 1e-9) -> None:
    """Stops unless this run's tau 0.86 point and baselines match reports/final_test_phase4.json."""
    pairs = [("predict_only_balanced", headline), ("always_medium", refs["always_medium"]),
             ("always_high", refs["always_strongest"]), ("oracle", refs["oracle"])]
    for name, mine in pairs:
        theirs = saved[name]
        problems = []
        if mine["correct"] != round(theirs["quality"] * theirs["n"]):
            problems.append(f"correct {mine['correct']} vs {round(theirs['quality'] * theirs['n'])}")
        for key, their_key in (("cost_saved", "cost_saved"), ("quality_kept", "quality_retained"),
                               ("cost_per_1k_usd", "cost_per_1k")):
            if abs(mine[key] - theirs[their_key]) > max(eps, 10 ** -DIGITS):
                problems.append(f"{key} {mine[key]} vs {theirs[their_key]}")
        for t in TIERS:
            if abs(mine["mix"][t] - theirs["mix"][t]) > 10 ** -DIGITS:
                problems.append(f"mix[{t}] {mine['mix'][t]} vs {theirs['mix'][t]}")
        if problems:
            raise ExportError(f"{name} doesn't reproduce reports/final_test_phase4.json: {'; '.join(problems)}")


_NUMBER = re.compile(r"^\*{0,2}([0-9.]+)\*{0,2}$")


def parse_classifier_table(markdown: str) -> list:
    """The table under '## Classifier comparison' in PHASE_3.md: [{name, auc_cheap, auc_medium, brier_cheap,
    brier_medium, chosen}]; chosen = the bold row. Stops if the section or its columns aren't as expected."""
    match = re.search(r"^## Classifier comparison.*?\n\n((?:\|.*\n)+)", markdown, re.M)
    if not match:
        raise ExportError("reports/PHASE_3.md has no '## Classifier comparison' table")
    lines = [line.strip() for line in match.group(1).strip().splitlines()]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    expected = ["Classifier", "Cheap ROC AUC", "Medium ROC AUC", "Cheap Brier", "Medium Brier"]
    if header != expected:
        raise ExportError(f"the classifier table's columns changed: {header}")
    out = []
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        numbers = [_NUMBER.match(c) for c in cells[1:]]
        if len(cells) != 5 or not all(numbers):
            raise ExportError(f"can't read classifier table row: {line}")
        name = cells[0]
        out.append({"name": name.strip("*").strip(), "auc_cheap": float(numbers[0].group(1)),
                    "auc_medium": float(numbers[1].group(1)), "brier_cheap": float(numbers[2].group(1)),
                    "brier_medium": float(numbers[3].group(1)), "chosen": name.startswith("**")})
    if sum(row["chosen"] for row in out) != 1:
        raise ExportError("the classifier table should have exactly one bold (chosen) row")
    return out


def judge_summary(calibration: dict) -> dict:
    a = calibration["all"]
    right_ref = calibration["groups"]["reference=reference right"]
    return {"n": a["n"], "agreement": r(a["agreement"]), "passed_wrong_answers": r(a["too_lenient"]),
            "failed_correct_answers": r(a["too_strict"]), "pass_means_right": r(a["pass_means_right"]),
            "passed_wrong_answers_when_reference_right": r(right_ref["too_lenient"]),
            "split": "train (GSM8K and MMLU answers with known truth)", "source": "reports/judge_calibration.json"}


def dataset_summary(test_rows: list) -> dict:
    split = json.loads((PROJECT_ROOT / "data" / "splits" / "split_info.json").read_text(encoding="utf-8"))
    val = json.loads((PROJECT_ROOT / "data" / "splits" / "val_info.json").read_text(encoding="utf-8"))
    labeled_train = val["pool_size"]
    return {"prompts": split["total"], "labeled": labeled_train + len(test_rows), "train_labeled": labeled_train,
            "fit": val["fit"], "validation": val["val"], "test_labeled": len(test_rows), **DATASET_FACTS}


def presets(points: list) -> dict:
    by_tau = {p["tau"]: p for p in points}
    out = {}
    for name, tau in load_quality_settings().presets.items():
        p = dict(by_tau[round(tau, 2)])
        p["provenance"] = HEADLINE_PROVENANCE if round(tau, 2) == HEADLINE_TAU else CURVE_PROVENANCE
        out[name] = p
    return out


def build(rows: list, probs: dict, phase4: dict, phase3_md: str, calibration: dict) -> dict:
    points = curve(rows, probs)
    refs = references(rows)
    headline = dict(next(p for p in points if p["tau"] == HEADLINE_TAU))
    check_reproduces(headline, refs, phase4["real_prices"])
    cheapest = refs["always_cheapest"]
    return {
        "schema": SCHEMA,
        "classifier": f"{CLASSIFIER[0]} v{CLASSIFIER[1]}",
        "split": {"name": "test", "n": len(rows), "scored": phase4["created"][:10]},
        "headline": {**headline, "baseline": "always-strongest", "provenance": HEADLINE_PROVENANCE},
        "vs_always_cheapest": {
            "cost_multiple": r(headline["cost_per_1k_usd"] / cheapest["cost_per_1k_usd"]),
            "quality_kept_points_gained": r(headline["quality_kept"] - cheapest["quality_kept"]),
            "note": "Triage costs more than always using the cheapest model; it buys back the quality it loses."},
        "references": refs,
        "presets": presets(points),
        "tier_mix_default": headline["mix"],
        "curve": {"provenance": CURVE_PROVENANCE, "points": points},
        "classifiers": {"method": "5-fold cross-validation on the train split (598 labeled prompts)",
                        "source": "reports/PHASE_3.md", "rows": parse_classifier_table(phase3_md)},
        "judge": judge_summary(calibration),
        "dataset": dataset_summary(rows),
        "caveats": CAVEATS,
    }


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    from eval.pareto import predictor_probabilities
    # The test split, for reporting only: the same rows final_test_phase4.py scored. Nothing here tunes anything.
    rows = load_eval_rows("test", allow_test=True)
    probs = predictor_probabilities(*CLASSIFIER, rows)
    phase4 = json.loads((REPORTS_DIR / "final_test_phase4.json").read_text(encoding="utf-8"))
    calibration = json.loads((REPORTS_DIR / "judge_calibration.json").read_text(encoding="utf-8"))
    data = build(rows, probs, phase4, (REPORTS_DIR / "PHASE_3.md").read_text(encoding="utf-8"), calibration)
    write_json(OUT_PATH, data)
    write_json(REPORT_COPY, {k: data[k] for k in ("schema", "classifier", "split", "headline", "references",
                                                  "presets", "curve")})
    h = data["headline"]
    print(f"{OUT_PATH.relative_to(PROJECT_ROOT)}: tau {h['tau']} keeps {h['quality_kept']:.1%} of always-strongest "
          f"quality and saves {h['cost_saved']:.1%} ({h['correct']}/{h['n']} correct); reproduces "
          f"final_test_phase4.json")
    print(f"{REPORT_COPY.relative_to(PROJECT_ROOT)}: the v2 curve ({len(data['curve']['points'])} points), "
          f"{CURVE_PROVENANCE}")


if __name__ == "__main__":
    main()
