"""Phase 3 Step 2: train "will this tier pass?" classifiers on the TRAIN split only.
 
One binary model per lower tier:  low ("will cheap pass?")  and  medium ("will medium pass?").
 
Feature sets:
    features   Classifier A: handcrafted features (data/features.npz)
    task_only  sanity baseline: just the one-hot task type columns from features.npz
    embeddings Classifier B: sentence-transformer embeddings (data/embeddings.npz, from embeddings.py)
    both       Classifier C: features + embeddings side by side
 
Usage:
    python -m eval.train_classifier --features features              # Classifier A
    python -m eval.train_classifier --features task_only             # baseline to beat
    python -m eval.train_classifier --features both --model gboost   # Classifier C, gradient boosting
    python -m eval.train_classifier --features embeddings --model logreg_cv --pca 32   # tuned C, 32 components
 
Honest-evaluation rules:
    * Only labeled prompts in the train split are loaded; assert_no_test_ids() guards that.
    * By default the validation IDs (data/splits/val_ids.txt) are held out too, so models train on the
      "fit" set and can be compared on validation. --include-val trains on all labeled train prompts.
    * All numbers printed here are 5-fold CROSS-VALIDATION on the train split. Comparing A/B/C and picking
      a winner uses these. The locked test set is used once, at the end, by a separate script.
    * No class reweighting: tau compares these probabilities directly, so they must stay honest.
 
Saves, per tier:  models/<name>_<tier>_v<N>.joblib  (fitted on all train prompts)
                  models/<name>_<tier>_v<N>.json    (CV metrics, settings, out-of-fold probabilities)
"""

import argparse
import hashlib 
import json
from datetime import datetime, timezone 
from pathlib import Path 

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
 
from app.registry import PROJECT_ROOT
from eval.splits import assert_no_test_ids, fit_ids, load_split

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
 
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_PATH = DATA_DIR / "features.npz"                 # task_<t> one-hot from the hand tags
FEATURES_GUESSED_PATH = DATA_DIR / "features_guessed.npz" # task_<t> one-hot from guess_task_type()
TASK_TYPE_SOURCES = {"hand": FEATURES_PATH, "guessed": FEATURES_GUESSED_PATH}
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npz"
OUTCOMES_PATH = DATA_DIR / "outcomes.jsonl"
TIERS = ("low", "medium")
SEED = 42
FOLDS = 5
NAMES = {"features": "classifier_a", "embeddings": "classifier_b", "both": "classifier_c",
         "task_only": "baseline_task_only"}

# ---------- Loading ----------

def load_npx(path: Path) -> tuple:
    """({prompt_id: row}, column names, version) from a .npz written by features.py or embeddings.py"""
    if not path.exists():
        raise SystemExit(f"{path.relative_to(PROJECT_ROOT)} not found. Build it first")
    data = np.load(path, allow_pickle=False)
    rows = {str(i): x for i, x in zip(data["ids"], data["X"])}
    names = [str(n) for n in data["feature_names"]] if "feature_names" in data else \
        [f"emb_{i}" for i in range(data["X"].shape[1])]
    version = str(data["feature_version"]) if "feature_version" in data else \
        (str(data["model_name"]) if "model_name" in data else "unknown")
    return rows, names, version

def feature_matrix(feature_set: str, ids: list, task_types: str = "hand") -> tuple:
    """(X, column names, version info) for the given prompt IDs, in that order.
    task_types: "hand" (features.npz) or "guessed" (features_guessed.npz, built with --guess-task-type)"""
    if feature_set in ("features", "task_only"):
        rows, names, version = load_npx(TASK_TYPE_SOURCES[task_types])
        X = np.array([rows[i] for i in ids], dtype=np.float64)
        if feature_set == "task_only":
            keep = [j for j, n in enumerate(names) if n.startswith("task_")]
            X, names = X[:, keep], [names[j] for j in keep]
        return X, names, {"features": version}
    if feature_set == "embeddings":
        rows, names, version = load_npx(EMBEDDINGS_PATH)
        return np.array([rows[i] for i in ids], dtype=np.float64), names, {"embeddings": version}
    if feature_set == "both":
        Xf, nf, vf = feature_matrix("features", ids, task_types)
        Xe, ne, ve = feature_matrix("embeddings", ids)
        return np.hstack([Xf, Xe]), nf + ne, {**vf, **ve}
    raise ValueError(f"unknown feature set {feature_set!r}")

def load_training_rows(include_val: bool = False) -> list:
    """Labelled outcomes in the train split, sorted by ID so runs are reproducible.

    By default the validation IDs (data/splits/val_ids.txt) are left out, so this returns the "fit" set;
    include_val=True returns every labeled train prompt (e.g. a final refit after tuning on validation)."""
    train_ids, _ = load_split()
    outcomes = [json.loads(l) for l in OUTCOMES_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = sorted((o for o in outcomes if o["label_status"] == "labeled" and o["id"] in train_ids),
                  key=lambda o: o["id"])
    assert_no_test_ids({o["id"] for o in rows})
    if not include_val:
        keep = fit_ids(o["id"] for o in rows)
        rows = [o for o in rows if o["id"] in keep]
    return rows

# ---------- Model ----------

def make_model(kind: str, pca: int = None):
    """kind: logreg, logreg_cv or gboost. pca: if set, scale and squeeze the columns down to that many
    principal components first (useful for the 384 embedding columns)."""
    # Scaling puts token counts (hundreds) and 0/1 flags on the same footing.
    steps = [StandardScaler()]
    if pca:
        steps.append(PCA(n_components=pca, random_state=SEED))
    if kind == "logreg":
        return make_pipeline(*steps, LogisticRegression(max_iter=2000, C=1.0))
    if kind == "logreg_cv":
        # Tries 12 regularization strengths with an inner 5-fold CV on the training folds only, scored by
        # log loss (tau needs honest probabilities), and keeps the best
        inner = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
        return make_pipeline(*steps, LogisticRegressionCV(Cs=12, cv=inner, scoring="neg_log_loss", max_iter=2000))
    if kind == "gboost":
        model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, random_state=SEED)
        return make_pipeline(*steps, model) if pca else model
    raise ValueError(f"unknown model {kind!r}")


def logistic_step(model):
    """The fitted LogisticRegression / LogisticRegressionCV inside a pipeline, or None."""
    if not hasattr(model, "named_steps"):
        return None
    for step in model.named_steps.values():
        if isinstance(step, (LogisticRegression, LogisticRegressionCV)):
            return step
    return None

def evaluate(y: np.ndarray, prob:np.ndarray) -> dict:
    """Metrics for out-of-fold predictions. 'Positive' = the tier passes; predicted pass if prob >= 0.5"""
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "n": int(len(y)),
        "pass_rate": float(y.mean()),
        "always_pass_accuracy": float(max(y.mean(), 1 - y.mean())),   # the accuracy a useless model gets
        "accuracy": float(accuracy_score(y, pred)),
        "precision_pass": float(precision_score(y, pred, zero_division=0)),
        "recall_pass": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)) if len(set(y)) == 2 else None,
        "brier": float(brier_score_loss(y, prob)),
        "log_loss": float(log_loss(y, prob, labels=[0, 1])),
        # rows = truth (fail, pass), columns = prediction (fail, pass)
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

def top_weights(model, names: list, k: int = 8) -> list:
    """Largest logistic-regression weights (on scaled features), for explaining what the model learned.
    Empty for gradient boosting, and after PCA (the weights belong to components, not named features)."""
    step = logistic_step(model)
    if step is None or "pca" in model.named_steps:
        return []
    coef = step.coef_[0]
    order = np.argsort(-np.abs(coef))[:k]
    return [(names[j], float(coef[j])) for j in order]

def next_version(name: str, tier: str) -> int:
    existing = [int(p.stem.rsplit("_v", 1)[1]) for p in MODELS_DIR.glob(f"{name}_{tier}_v*.joblib")
                if p.stem.rsplit("_v", 1)[1].isdigit()]
    return max(existing, default=0) + 1

# ---------- Main ----------

def train_tier(tier: str, rows: list, feature_set: str, kind: str, name: str, save: bool,
               pca: int = None, trained_on: str = "fit", task_types: str = "hand") -> dict:
    keep = [o for o in rows if o.get(f"{tier}_pass") is not None]
    ids = [o["id"] for o in keep]
    y = np.array([int(bool(o[f"{tier}_pass"])) for o in keep])
    X, names, versions = feature_matrix(feature_set, ids, task_types)
    if pca and pca >= X.shape[1]:
        raise SystemExit(f"--pca {pca} must be smaller than the {X.shape[1]} columns of --features {feature_set}")

    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    oof = cross_val_predict(make_model(kind, pca), X, y, cv=cv, method="predict_proba")[:, 1]
    metrics = evaluate(y, oof)
    
    # Per task type, so a strong overall number can't hide a weak task type 
    by_task = {}
    for task in sorted({o["task_type"] for o in keep}):
        mask = np.array([o["task_type"] == task for o in keep])
        if mask.sum() >= 5:
            by_task[task] = {"n": int(mask.sum()), "pass_rate": float(y[mask].mean()),
                             "accuracy": float(((oof[mask] >= 0.5) == y[mask]).mean())}
    
    model = make_model(kind, pca).fit(X, y)
    weights = top_weights(model, names)
    step = logistic_step(model)
    chosen_c = float(step.C_[0]) if isinstance(step, LogisticRegressionCV) else None

    m = metrics
    print(f"\n=== {name} | tier={tier} | {kind} | {X.shape[1]} columns"
          + (f" -> {pca} PCA components" if pca else "") + f" | {m['n']} train prompts "
          f"({m['pass_rate']:.0%} pass) ===")
    if chosen_c is not None:
        print(f"chosen C {chosen_c:.4g} (from 12 strengths, inner CV on log loss)")
    print(f"accuracy {m['accuracy']:.3f}  (always-guess-majority gets {m['always_pass_accuracy']:.3f})")
    print(f"precision(pass) {m['precision_pass']:.3f}   recall(pass) {m['recall_pass']:.3f}   "
          f"ROC AUC {m['roc_auc'] or float('nan'):.3f}   Brier {m['brier']:.3f}")
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    print(f"confusion matrix          pred fail  pred pass\n  truly fail           {tn:>9}  {fp:>9}"
          f"   <- {fp} would get a bad answer\n  truly pass           {fn:>9}  {tp:>9}"
          f"   <- {fn} would cost more than needed")
    print("by task type: " + ", ".join(f"{t} {v['accuracy']:.2f} (n={v['n']})" for t, v in by_task.items()))
    
    if weights:
        print("biggest weights (+ = more likely to pass): "
            + ", ".join(f"{n} {w:+.3f}" for n, w in weights))
    
    record = {
        "name": name, "tier": tier, "feature_set": feature_set, "model": kind, "pca": pca, "chosen_C": chosen_c,
        "trained_on": trained_on,   # "fit" = labeled train minus validation; "train" = all labeled train (--include-val)
        "task_types": task_types,   # where the task_<t> columns came from at training time: hand tags or guesses
        "versions": versions,
        "columns": names if X.shape[1] <= 50 else f"{X.shape[1]} columns",
        "cv": {"folds": FOLDS, "seed": SEED, **metrics}, "by_task_type": by_task, "top_weights": weights,
        "train_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "oof_probabilities": {i: round(float(p), 5) for i, p in zip(ids, oof)}, # for tuning tau on train later 
    }
    if save: 
        MODELS_DIR.mkdir(exist_ok=True)
        version = next_version(name, tier)
        stem = MODELS_DIR / f"{name}_{tier}_v{version}"
        joblib.dump(model, stem.with_suffix(".joblib"))
        stem.with_suffix(".json").write_text(json.dumps({**record, "version": version}, indent=2))
        print(f"saved {stem.relative_to(PROJECT_ROOT)}.joblib/.json")
    return record

def main():
    parser = argparse.ArgumentParser(description="Train will-this-tier-pass classifiers (train split, CV metrics).")
    parser.add_argument("--features", default="features", choices=list(NAMES))
    parser.add_argument("--model", default="logreg", choices=["logreg", "logreg_cv", "gboost"])
    parser.add_argument("--pca", type=int, default=None, help="reduce the columns to this many PCA components first")
    parser.add_argument("--no-save", action="store_true", help="just print metrics")
    parser.add_argument("--task-types", default="hand", choices=list(TASK_TYPE_SOURCES),
                        help="task_<t> columns from hand tags (features.npz) or from guess_task_type() "
                             "(features_guessed.npz); guessed models are saved with a _guessed suffix")
    parser.add_argument("--include-val", action="store_true",
                        help="train on ALL labeled train prompts, validation included (default: fit set only)")
    args = parser.parse_args()
    if args.pca is not None and args.pca < 1:
        raise SystemExit("--pca must be a positive whole number")

    name = (NAMES[args.features] + ("" if args.model == "logreg" else f"_{args.model}")
            + (f"_pca{args.pca}" if args.pca else "") + ("_guessed" if args.task_types == "guessed" else ""))
    if args.task_types == "guessed" and args.features in ("embeddings",):
        raise SystemExit("--task-types guessed only matters for feature sets with task columns")
    rows = load_training_rows(include_val=args.include_val)
    if not rows:
        raise SystemExit("no labeled train prompts found; check data/outcomes.jsonl and data/splits/")
    trained_on = "train" if args.include_val else "fit"
    print(f"{len(rows)} labeled {trained_on} prompts"
          + ("" if args.include_val else " (validation held out)")
          + f"; metrics are {FOLDS}-fold cross-validation on these only")
    for tier in TIERS:
        train_tier(tier, rows, args.features, args.model, name, save=not args.no_save, pca=args.pca,
                   trained_on=trained_on, task_types=args.task_types)


if __name__ == "__main__":
    main()