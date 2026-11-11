"""Phase 4 Step 9: safe retraining of the live classifier. Local training and evaluation only: no API calls.

Usage:
    python -m scripts.retrain --dry-run      # train and evaluate, print the decision; writes only the candidate files
    python -m scripts.retrain                # the same, then replace the live model if the rule says so

Steps
  1. Training data
       original  the fit set (labeled train minus validation), hand-tagged features + embeddings + outcome
                 labels: exactly what eval/train_classifier.py uses for classifier_c_logreg_cv
       new       background labels (the request log's `labels` table) joined to their request rows, only where
                 the row is training_eligible, its feature_version and embedding_model match the current code,
                 and both vectors are stored. A label is for the tier that answered (served low -> a low
                 label, medium -> a medium label); nothing is inferred for the other tier.
       leak guard  every validation and test prompt is HMAC'd with TRIAGE_HASH_SECRET (the API's secret), and
                 any live row with a matching prompt_hmac is dropped, so a validation or test question sent
                 through the API can never become training data. Only those prompts' text is read, to hash it;
                 the test set's outcomes are never loaded.
     NOTE: live rows' task_type columns come from the keyword guesser unless the caller sent X-Triage-Task,
     while the original data uses hand tags. Labels are also noisy (graded against the strong model's answer;
     see app/api/background.py).
  2. No new eligible labels: "nothing new, keeping the current model", a history entry, no training.
  3. Train a candidate: same model type and settings as the live classifier (classifier_c_logreg_cv:
     scaler + LogisticRegressionCV, features + embeddings), saved as the next version. Existing model files are
     never overwritten (files are opened in exclusive-create mode).
  4. Evaluate the live model and the candidate on the VALIDATION set (eval/routing_sim) at each quality preset
     in routing.yaml: quality_retained and cost_saved, plus per-tier log loss, Brier and ROC AUC.
  5. Decide with scripts/retrain_decision.should_replace on the balanced preset.
  6. Replace: point routing.yaml's classifier.version at the new version (temp file + rename, comments kept).
     Keep: change nothing. Either way, append an entry to reports/retrain_history.jsonl.
     --dry-run writes nothing but the candidate model files.

The API server loads the classifier once at startup: a new version is only used after a restart.
"""
import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.api.request_log import (DEFAULT_LOG_DB, HASH_SECRET_ENV, LOG_DB_ENV, SQLiteRequestLog,
                                 embedding_from_bytes, prompt_hmac)
from app.registry import PROJECT_ROOT
from app.router.embeddings import MODEL_NAME as EMBEDDING_MODEL
from app.router.features import FEATURE_NAMES, FEATURE_VERSION
from app.router.quality import ROUTING_CONFIG_PATH, choose_tier, load_quality_settings
from eval import train_classifier as tc
from eval.splits import assert_no_test_ids, load_split, load_validation_split
from scripts import retrain_decision

HISTORY_PATH = PROJECT_ROOT / "reports" / "retrain_history.jsonl"
TIERS = ("low", "medium")
EMBEDDING_DIM = 384
DECISION_PRESET = "balanced"


# ---------- 1. training data ----------

def load_original(include_val: bool = False) -> dict:
    """{tier: {"X", "y", "ids"}} for the fit set, built exactly like eval/train_classifier.train_tier."""
    rows = tc.load_training_rows(include_val=include_val)
    out = {}
    for tier in TIERS:
        keep = [o for o in rows if o.get(f"{tier}_pass") is not None]
        ids = [o["id"] for o in keep]
        X, names, versions = tc.feature_matrix("both", ids, "hand")
        if names[:len(FEATURE_NAMES)] != FEATURE_NAMES or X.shape[1] != len(FEATURE_NAMES) + EMBEDDING_DIM:
            raise SystemExit("data/features.npz + embeddings.npz don't have the columns the live code builds; "
                             "rebuild them before retraining")
        if versions != {"features": FEATURE_VERSION, "embeddings": EMBEDDING_MODEL}:
            raise SystemExit(f"training data versions {versions} don't match the current code "
                             f"(features {FEATURE_VERSION}, embeddings {EMBEDDING_MODEL})")
        out[tier] = {"X": X, "y": np.array([int(bool(o[f"{tier}_pass"])) for o in keep]), "ids": ids}
    assert_no_test_ids({i for t in out.values() for i in t["ids"]})
    return out


def canonical_messages(messages: list) -> list:
    """The messages exactly as the API hashes them (only role and content, as its request model keeps)."""
    return [{"role": m["role"], "content": m["content"]} for m in messages]


def forbidden_hmacs(secret: bytes, prompts: Optional[list] = None) -> dict:
    """{prompt_hmac: "validation" | "test"} for every validation and test prompt."""
    from eval.build_dataset import load_prompts
    _, test_ids = load_split()
    val_ids = load_validation_split()
    out = {}
    for p in prompts if prompts is not None else load_prompts():
        split = "test" if p["id"] in test_ids else "validation" if p["id"] in val_ids else None
        if split:
            out[prompt_hmac(canonical_messages(p["messages"]), secret)] = split
    return out


def load_new_labels(log, forbidden: dict) -> tuple:
    """({tier: {"X", "y", "request_ids", "task_type_guessed"}}, exclusion counts) from the labels table."""
    new = {t: {"X": [], "y": [], "request_ids": [], "task_type_guessed": []} for t in TIERS}
    excluded = {}

    def drop(reason):
        excluded[reason] = excluded.get(reason, 0) + 1

    for label in log.labels():
        if label.status != "labeled" or label.passed is None:
            drop("not_labeled"); continue
        row = log.get(label.request_id)
        if row is None:
            drop("request_row_missing"); continue
        if not row.training_eligible:
            drop("not_training_eligible"); continue
        if row.feature_version != FEATURE_VERSION or row.embedding_model != EMBEDDING_MODEL:
            drop("wrong_feature_or_embedding_version"); continue
        if row.features is None or row.embedding is None or len(row.features) != len(FEATURE_NAMES):
            drop("vectors_missing"); continue
        embedding = embedding_from_bytes(row.embedding)
        if embedding.shape != (EMBEDDING_DIM,):
            drop("vectors_missing"); continue
        if row.prompt_hmac in forbidden:
            drop(f"leak_{forbidden[row.prompt_hmac]}"); continue
        tier = row.tier                               # the tier that answered: a label for that tier only
        if tier not in TIERS or label.served_tier not in (None, tier):
            drop("not_a_low_or_medium_answer"); continue
        new[tier]["X"].append(np.concatenate([np.asarray(row.features, dtype=np.float64),
                                              embedding.astype(np.float64)]))
        new[tier]["y"].append(int(bool(label.passed)))
        new[tier]["request_ids"].append(row.request_id)
        new[tier]["task_type_guessed"].append(bool(row.task_type_guessed))
    for t in TIERS:
        width = len(FEATURE_NAMES) + EMBEDDING_DIM
        new[t]["X"] = np.array(new[t]["X"], dtype=np.float64).reshape(-1, width)
        new[t]["y"] = np.array(new[t]["y"], dtype=int)
    return new, excluded


def open_log(path: Optional[Path]):
    """The request log, or None if there is no database yet (never creates one)."""
    path = Path(path or os.environ.get(LOG_DB_ENV) or DEFAULT_LOG_DB)
    return SQLiteRequestLog(path) if path.exists() else None


def data_counts(original: dict, new: dict, excluded: dict) -> dict:
    return {
        "original": {t: len(original[t]["ids"]) for t in TIERS},
        "new_labels": {t: int(len(new[t]["y"])) for t in TIERS},
        "new_label_pass_rate": {t: (float(new[t]["y"].mean()) if len(new[t]["y"]) else None) for t in TIERS},
        "new_task_types_guessed": {t: int(sum(new[t]["task_type_guessed"])) for t in TIERS},
        "excluded": dict(sorted(excluded.items())),
        "note": "live rows use guessed task types unless X-Triage-Task was sent; the original data uses hand tags",
    }


# ---------- 3. training the candidate ----------

def model_files(name: str, version: int, models_dir: Path) -> list:
    return [models_dir / f"{name}_{tier}_v{version}{ext}" for tier in TIERS for ext in (".joblib", ".json")]


def next_free_version(name: str, models_dir: Path) -> int:
    """One past the highest version of THIS name for either tier (classifier_c_logreg_cv_guessed doesn't count)."""
    pattern = re.compile(rf"^{re.escape(name)}_(?:{'|'.join(TIERS)})_v(\d+)\.(?:joblib|json)$")
    used = [int(m.group(1)) for p in Path(models_dir).iterdir() if (m := pattern.match(p.name))] \
        if Path(models_dir).exists() else []
    return max(used, default=0) + 1


def train_candidate(original: dict, new: dict, *, name: str, models_dir: Path, kind: str = "logreg_cv") -> dict:
    """Fit one model per tier on original + new rows; save as the next version. Returns {"version", "tiers"}."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    version = next_free_version(name, models_dir)
    if any(p.exists() for p in model_files(name, version, models_dir)):
        raise SystemExit(f"{name} v{version} files already exist; refusing to overwrite a model")
    summary = {}
    for tier in TIERS:
        X = np.vstack([original[tier]["X"], new[tier]["X"]])
        y = np.concatenate([original[tier]["y"], new[tier]["y"]])
        cv = StratifiedKFold(n_splits=tc.FOLDS, shuffle=True, random_state=tc.SEED)
        oof = cross_val_predict(tc.make_model(kind), X, y, cv=cv, method="predict_proba")[:, 1]
        model = tc.make_model(kind).fit(X, y)
        step = tc.logistic_step(model)
        ids = list(original[tier]["ids"]) + [f"live:{r}" for r in new[tier]["request_ids"]]
        record = {
            "name": name, "tier": tier, "feature_set": "both", "model": kind, "pca": None,
            "chosen_C": float(step.C_[0]) if step is not None and hasattr(step, "C_") else None,
            "trained_on": "fit+live", "task_types": "hand (original) + guessed or given (live)",
            "versions": {"features": FEATURE_VERSION, "embeddings": EMBEDDING_MODEL},
            "columns": f"{X.shape[1]} columns",
            "rows": {"original": len(original[tier]["ids"]), "new_labels": int(len(new[tier]["y"]))},
            "cv": {"folds": tc.FOLDS, "seed": tc.SEED, **tc.evaluate(y, oof)},
            "train_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
            "trained_at": datetime.now(timezone.utc).isoformat(), "version": version,
        }
        stem = models_dir / f"{name}_{tier}_v{version}"
        with open(stem.with_suffix(".joblib"), "xb") as f:          # "x": fail rather than overwrite
            joblib.dump(model, f)
        with open(stem.with_suffix(".json"), "x", encoding="utf-8") as f:
            f.write(json.dumps(record, indent=2))
        summary[tier] = {"rows": record["rows"], "cv_log_loss": record["cv"]["log_loss"],
                         "cv_roc_auc": record["cv"]["roc_auc"]}
    return {"version": version, "tiers": summary}


# ---------- 4. evaluating on validation ----------

def prob_metrics(y, p) -> dict:
    y, p = np.array(y, dtype=int), np.clip(np.array(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"n": int(len(y)), "log_loss": float(log_loss(y, p, labels=[0, 1])),
            "brier": float(brier_score_loss(y, p)),
            "roc_auc": float(roc_auc_score(y, p)) if len(set(y)) == 2 else None}


def default_predictor(name: str, version: int, models_dir: Path):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from app.router.predictor import Predictor
    return Predictor(name, version, models_dir=models_dir)


def evaluate_on_validation(name: str, version: int, *, models_dir: Path = tc.MODELS_DIR,
                           make_predictor: Callable = default_predictor, rows=None, prompts=None) -> dict:
    """{"presets": {preset: {tau, quality_retained, cost_saved, mix}}, "tiers": {tier: log loss, Brier, AUC}}
    for one model version, on the validation split only, with the live Predictor (hand-tagged task types)."""
    from app.router.features import prompt_text
    from eval.build_dataset import load_prompts
    from eval.routing_sim import load_eval_rows, score
    rows = rows if rows is not None else load_eval_rows("val", allow_val=True)
    assert_no_test_ids({r.id for r in rows})
    if prompts is None:
        wanted = {r.id for r in rows}
        prompts = {p["id"]: p for p in load_prompts() if p["id"] in wanted}
    predictor = make_predictor(name, version, models_dir)
    preds = predictor.predict_texts([prompt_text(prompts[r.id]) for r in rows],
                                    [prompts[r.id].get("task_type") for r in rows])
    presets = {}
    for preset, tau in load_quality_settings().presets.items():
        s = score(rows, [choose_tier(p.p_low, p.p_medium, tau) for p in preds])
        presets[preset] = {"tau": tau, "quality_retained": s["quality_retained"], "cost_saved": s["cost_saved"],
                           "mix": s["mix"]}
    tiers = {"low": prob_metrics([r.passed["low"] for r in rows], [p.p_low for p in preds]),
             "medium": prob_metrics([r.passed["medium"] for r in rows], [p.raw_p_medium for p in preds])}
    return {"version": version, "presets": presets, "tiers": tiers}


# ---------- 6. routing.yaml and history ----------

def set_classifier_version(path: Path, version: int) -> None:
    """Change only classifier.version in routing.yaml (comments kept), via a temp file and an atomic rename."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    start = next((i for i, l in enumerate(lines) if re.match(r"^classifier:\s*(#.*)?$", l)), None)
    if start is None:
        raise SystemExit(f"{path}: no top-level `classifier:` section")
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if line.strip() and not line[0].isspace() and not line.lstrip().startswith("#"):
            break                                                    # left the section
        m = re.match(r"^(\s+version:\s*)(\d+)(.*)$", line.rstrip("\n"))
        if m:
            lines[i] = f"{m.group(1)}{version}{m.group(3)}" + ("\n" if line.endswith("\n") else "")
            break
    else:
        raise SystemExit(f"{path}: no `version:` under `classifier:`")
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def append_history(path: Path, entry: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------- the whole run ----------

def run(*, dry_run: bool = False, log_db: Optional[Path] = None, models_dir: Path = tc.MODELS_DIR,
        routing_path: Path = ROUTING_CONFIG_PATH, history_path: Path = HISTORY_PATH, secret: Optional[bytes] = None,
        load_original_fn: Callable = load_original, train_fn: Callable = train_candidate,
        evaluate_fn: Callable = evaluate_on_validation, decide_fn: Optional[Callable] = None,
        prompts: Optional[list] = None, say=print) -> dict:
    from app.router.predictor import load_classifier_config
    name, current_version = load_classifier_config(routing_path)
    original = load_original_fn()
    log = open_log(log_db)
    new, excluded = {t: {"X": np.zeros((0, len(FEATURE_NAMES) + EMBEDDING_DIM)), "y": np.zeros(0, dtype=int),
                         "request_ids": [], "task_type_guessed": []} for t in TIERS}, {}
    try:
        if log is not None and log.labels():
            secret = secret or os.environ.get(HASH_SECRET_ENV, "").encode("utf-8")
            if not secret:
                raise SystemExit(f"set {HASH_SECRET_ENV} (the API's secret): it is needed to keep validation and "
                                 "test prompts out of the training data")
            new, excluded = load_new_labels(log, forbidden_hmacs(secret, prompts))
    finally:
        if log is not None:
            log.close()
    counts = data_counts(original, new, excluded)
    entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "dry_run": dry_run, "classifier": name,
             "current_version": current_version, "candidate_version": None, "data": counts}
    say(f"original fit-set rows: {counts['original']} | new labels: {counts['new_labels']} | excluded: "
        f"{counts['excluded'] or 'none'}")

    if sum(counts["new_labels"].values()) == 0:
        reason = "nothing new, keeping the current model"
        entry.update(decision="keep", reason=reason, trained=False)
        say(reason)
        if not dry_run:
            append_history(history_path, entry)
        return entry

    candidate = train_fn(original, new, name=name, models_dir=models_dir)
    entry["candidate_version"] = candidate["version"]
    entry["training"] = candidate["tiers"]
    say(f"candidate {name} v{candidate['version']} saved in {models_dir}")
    current_metrics = evaluate_fn(name, current_version, models_dir=models_dir)
    candidate_metrics = evaluate_fn(name, candidate["version"], models_dir=models_dir)
    entry["metrics"] = {"current": current_metrics, "candidate": candidate_metrics}
    decide = decide_fn or retrain_decision.should_replace
    replace, reason = decide(current_metrics["presets"][DECISION_PRESET], candidate_metrics["presets"][DECISION_PRESET])
    entry.update(decision="replace" if replace else "keep", reason=reason, trained=True)
    for preset in current_metrics["presets"]:
        c, k = current_metrics["presets"][preset], candidate_metrics["presets"][preset]
        say(f"  {preset:<9} quality {c['quality_retained']:.1%} -> {k['quality_retained']:.1%}   "
            f"saved {c['cost_saved']:.0%} -> {k['cost_saved']:.0%}")
    say(f"decision: {entry['decision']} ({reason})" + ("  [dry run: nothing changed]" if dry_run else ""))
    if dry_run:
        return entry
    if replace:
        set_classifier_version(routing_path, candidate["version"])
        say(f"routing.yaml now uses {name} v{candidate['version']}; restart the API server to load it")
    append_history(history_path, entry)
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description="Retrain the live classifier with background labels, safely.")
    parser.add_argument("--dry-run", action="store_true", help="train and evaluate; write only the candidate files")
    parser.add_argument("--log-db", type=Path, default=None, help=f"request log (default: ${LOG_DB_ENV} or "
                        f"{DEFAULT_LOG_DB.relative_to(PROJECT_ROOT)})")
    args = parser.parse_args(argv)
    entry = run(dry_run=args.dry_run, log_db=args.log_db)
    print(json.dumps({k: entry[k] for k in ("decision", "reason", "data")}, indent=2))


if __name__ == "__main__":
    main()
