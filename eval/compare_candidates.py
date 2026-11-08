"""Phase 4 Step 0, Part F: compare candidate classifiers on the VALIDATION set. No API calls.

Usage:
    python -m eval.compare_candidates
    python -m eval.compare_candidates --candidates classifier_c_logreg_cv:2 classifier_c_logreg_cv_guessed:1

For each candidate, prompts are routed twice with the real Predictor (the same code the live router uses):
    guessed  task type from guess_task_type()  (what production sees for most requests)
    hand     the hand-tagged task type          (requests that send X-Triage-Task)
and scored with routing_sim.score() at one tau (default: the current `balanced` preset in routing.yaml).

Also reported:
    log loss / Brier / ROC AUC of each tier's raw probability against that tier's pass/fail on validation
    the candidates' out-of-fold (CV) metrics on the fit set, next to v1's (v1's OOF covers all 598 train
    prompts; it is also recomputed on the 481 fit prompts so the numbers are comparable)

Writes reports/candidates_val.json. Never reads the test split. Picks no winner.
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from app.registry import PROJECT_ROOT
from app.router.features import prompt_text
from app.router.predictor import MODELS_DIR, Predictor
from app.router.quality import choose_tier, load_quality_settings, resolve_tau
from eval.baselines import always, oracle
from eval.build_dataset import load_prompts
from eval.routing_sim import OUTCOMES_PATH, TIERS, load_eval_rows, score
from eval.splits import load_split, load_validation_split

REPORT_PATH = PROJECT_ROOT / "reports" / "candidates_val.json"
DEFAULT_CANDIDATES = ["classifier_c_logreg_cv:2", "classifier_c_logreg_cv_guessed:1"]
REFERENCE_OOF = "classifier_c_logreg_cv:1"     # the Phase 3 model, for the OOF comparison only
PROB_TIERS = ("low", "medium")


def parse(spec: str) -> tuple:
    name, version = spec.rsplit(":", 1)
    return name, int(version)


def outcomes_by_id(ids: set) -> dict:
    out = {}
    for line in OUTCOMES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            if o["id"] in ids:
                out[o["id"]] = o
    return out


def prob_metrics(y: list, p: list) -> dict:
    y, p = np.array(y, dtype=int), np.clip(np.array(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "pass_rate": float(y.mean()),
            "log_loss": float(log_loss(y, p, labels=[0, 1])), "brier": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None}


def evaluate(predictor: Predictor, rows: list, prompts: dict, outcomes: dict, tau: float, mode: str) -> dict:
    texts = [prompt_text(prompts[r.id]) for r in rows]
    tasks = [None if mode == "guessed" else prompts[r.id]["task_type"] for r in rows]
    preds = predictor.predict_texts(texts, tasks)
    tiers = [choose_tier(p.p_low, p.p_medium, tau) for p in preds]
    result = {"routing": score(rows, tiers), "probabilities": {}}
    raw = {"low": [p.p_low for p in preds], "medium": [p.raw_p_medium for p in preds]}
    for tier in PROB_TIERS:
        keep = [(outcomes[r.id][f"{tier}_pass"], prob) for r, prob in zip(rows, raw[tier])
                if outcomes[r.id].get(f"{tier}_pass") is not None]
        result["probabilities"][tier] = prob_metrics([int(bool(y)) for y, _ in keep], [p for _, p in keep])
    if mode == "guessed":
        hand = [prompts[r.id]["task_type"] for r in rows]
        result["task_type_agreement"] = float(np.mean([p.task_type == h for p, h in zip(preds, hand)]))
    return result


def oof_metrics(name: str, version: int, ids: set = None) -> dict:
    """CV metrics saved with a model; with ids, recomputed from its out-of-fold probabilities on those ids."""
    out = {}
    for tier in PROB_TIERS:
        meta = json.loads((MODELS_DIR / f"{name}_{tier}_v{version}.json").read_text(encoding="utf-8"))
        if ids is None:
            cv = meta["cv"]
            out[tier] = {k: cv[k] for k in ("n", "pass_rate", "log_loss", "brier", "roc_auc", "accuracy")}
            continue
        probs = {i: p for i, p in meta["oof_probabilities"].items() if i in ids}
        outcomes = outcomes_by_id(set(probs))
        keep = [(int(bool(outcomes[i][f"{tier}_pass"])), p) for i, p in probs.items()
                if outcomes[i].get(f"{tier}_pass") is not None]
        m = prob_metrics([y for y, _ in keep], [p for _, p in keep])
        m["accuracy"] = float(np.mean([(p >= 0.5) == y for y, p in keep]))
        out[tier] = m
    return out


def print_table(report: dict) -> None:
    print(f"\nVALIDATION ({report['n_val']} prompts), tau = {report['tau']} ({report['tau_preset']})")
    print(f"{'strategy':<44}{'quality':>8}{'kept':>8}{'saved':>7}{'LL low':>8}{'LL med':>8}"
          f"{'AUC low':>9}{'AUC med':>9}   mix low/med/high")
    for name, r in report["baselines"].items():
        mix = "/".join(f"{r['mix'][t]:.0%}" for t in TIERS)
        print(f"{name:<44}{r['quality']:>8.1%}{r['quality_retained']:>8.1%}{r['cost_saved']:>7.0%}"
              f"{'':>8}{'':>8}{'':>9}{'':>9}   {mix}")
    for cand, modes in report["candidates"].items():
        for mode, res in modes.items():
            r, p = res["routing"], res["probabilities"]
            mix = "/".join(f"{r['mix'][t]:.0%}" for t in TIERS)
            print(f"{cand + ' | ' + mode + ' tasks':<44}{r['quality']:>8.1%}{r['quality_retained']:>8.1%}"
                  f"{r['cost_saved']:>7.0%}{p['low']['log_loss']:>8.3f}{p['medium']['log_loss']:>8.3f}"
                  f"{p['low']['roc_auc']:>9.3f}{p['medium']['roc_auc']:>9.3f}   {mix}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare candidate classifiers on the validation set.")
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES, help="name:version ...")
    parser.add_argument("--quality", default="balanced", help="preset name or a number for tau")
    args = parser.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")        # embedding model must already be cached: no downloads

    _, test = load_split()
    val_ids = load_validation_split()
    rows = load_eval_rows("val")
    assert {r.id for r in rows} <= val_ids and not {r.id for r in rows} & test
    prompts = {p["id"]: p for p in load_prompts() if p["id"] in val_ids}
    outcomes = outcomes_by_id(val_ids)
    tau = resolve_tau(args.quality, load_quality_settings())

    report = {
        "split": "val (data/splits/val_ids.txt)", "n_val": len(rows), "tau": tau, "tau_preset": args.quality,
        "note": "tau presets were chosen for classifier v1; they will be re-picked on validation in Step 3",
        "created": datetime.now(timezone.utc).isoformat(),
        "baselines": {f"always_{t}": score(rows, always(rows, t)) for t in TIERS} | {"oracle": score(rows, oracle(rows))},
        "candidates": {}, "oof_fit": {},
    }
    fit_ids = load_split()[0] - val_ids
    for spec in args.candidates:
        name, version = parse(spec)
        predictor = Predictor(name, version)
        report["candidates"][spec] = {mode: evaluate(predictor, rows, prompts, outcomes, tau, mode)
                                      for mode in ("guessed", "hand")}
        report["oof_fit"][spec] = {"trained_on": predictor.meta["low"].get("trained_on"),
                                   "task_types": predictor.meta["low"].get("task_types", "hand"),
                                   "saved_cv": oof_metrics(name, version)}
    ref_name, ref_version = parse(REFERENCE_OOF)
    report["oof_fit"][REFERENCE_OOF] = {"trained_on": "train (all 598, validation included)", "task_types": "hand",
                                        "saved_cv": oof_metrics(ref_name, ref_version),
                                        "recomputed_on_fit_ids": oof_metrics(ref_name, ref_version, fit_ids)}

    print_table(report)
    print("\nOUT-OF-FOLD (5-fold CV on the training prompts; not validation)")
    print(f"{'model':<36}{'source':<14}{'tier':<8}{'n':>5}{'log loss':>10}{'Brier':>8}{'AUC':>7}{'acc':>7}")
    for spec, info in report["oof_fit"].items():
        for source in ("saved_cv", "recomputed_on_fit_ids"):
            for tier, m in info.get(source, {}).items():
                label = "CV as saved" if source == "saved_cv" else "on fit ids"
                print(f"{spec:<36}{label:<14}{tier:<8}{m['n']:>5}{m['log_loss']:>10.3f}{m['brier']:>8.3f}"
                      f"{m['roc_auc']:>7.3f}{m['accuracy']:>7.3f}")
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nsaved {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
