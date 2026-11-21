"""
predictor.py: turns a prompt into "chance the cheap tier passes" and "chance the medium tier passes".
 
    prompt -> features.py (14 clues) + embeddings.py (384 meaning numbers)
           -> the classifier named in routing.yaml -> p_low, p_medium
           -> quality.py compares them with tau -> low / medium / high
 
Everything is loaded once (load_predictor() is cached). A classifier that is a StandardScaler + logistic regression
runs as plain numpy maths from its exported .npz (scripts/export_numpy_classifier.py), so the server never imports
scikit-learn (and scipy): about 150 MB of memory. The .npz records the SHA-256 of the .joblib it came from; if it
doesn't match, or there is no .npz, the .joblib is loaded with scikit-learn as before. At load time the predictor refuses to start if
the saved models don't match the current code (feature version, embedding model, number of columns),
because a silently mismatched model is one of the most common real-world ML bugs.
 
Try it:
    python -m app.router.predictor "Summarize this email in one sentence: ..."
    python -m app.router.predictor "What is 17 * 23?" --quality high --task reasoning
"""

from __future__ import annotations
 
import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional
 
import joblib
import numpy as np
import yaml
 
from app.registry import PROJECT_ROOT
from app.router.features import (FEATURE_NAMES, FEATURE_VERSION, extract_features, guess_task_type,
                                 normalize_task_type, prompt_text, to_vector)
from app.router.quality import ROUTING_CONFIG_PATH, RoutingDecision, load_quality_settings, resolve_tau, route
 
MODELS_DIR = PROJECT_ROOT / "models"
TIERS = ("low", "medium")
USES_FEATURES = {"features", "task_only", "both"}
USES_EMBEDDINGS = {"embeddings", "both"}


logger = logging.getLogger("triage.predictor")


class PredictorError(RuntimeError):
    """The configured classifier is missing or doesn't match the current one"""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NumpyLogisticPipeline:
    """StandardScaler -> binary logistic regression, as numpy: predict_proba(X)[:, 1] = 1 / (1 + e^-(z @ w + b)),
    z = (X - mean) / scale. The same maths scikit-learn runs (tests/test_numpy_classifier.py checks it)."""

    def __init__(self, mean: np.ndarray, scale: np.ndarray, coef: np.ndarray, intercept: float):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.coef = np.asarray(coef, dtype=np.float64)
        self.intercept = float(intercept)
        self.n_features_in_ = self.coef.shape[0]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = (np.asarray(X, dtype=np.float64) - self.mean) / self.scale
        logit = z @ self.coef + self.intercept
        # expit without overflow warnings: e^-x for x >= 0, e^x / (1 + e^x) below
        p = np.empty_like(logit)
        pos = logit >= 0
        p[pos] = 1.0 / (1.0 + np.exp(-logit[pos]))
        e = np.exp(logit[~pos])
        p[~pos] = e / (1.0 + e)
        return np.column_stack([1.0 - p, p])

    @classmethod
    def from_sklearn(cls, pipeline) -> "NumpyLogisticPipeline":
        """From a fitted Pipeline([StandardScaler, LogisticRegression(CV)]) with classes [0, 1]"""
        steps = [step for _, step in pipeline.steps]
        if len(steps) != 2 or type(steps[0]).__name__ != "StandardScaler" or not hasattr(steps[1], "coef_"):
            raise PredictorError(f"can't export {[type(s).__name__ for s in steps]}: only StandardScaler + "
                                 "logistic regression")
        scaler, clf = steps
        if list(clf.classes_) != [0, 1] or clf.coef_.shape[0] != 1:
            raise PredictorError(f"expected a binary classifier with classes [0, 1], got {list(clf.classes_)}")
        n = clf.coef_.shape[1]
        mean = scaler.mean_ if scaler.with_mean and scaler.mean_ is not None else np.zeros(n)
        scale = scaler.scale_ if scaler.with_std and scaler.scale_ is not None else np.ones(n)
        return cls(mean, scale, clf.coef_[0], clf.intercept_[0])

    def save(self, path: Path, source_sha256: str) -> None:
        np.savez(path, mean=self.mean, scale=self.scale, coef=self.coef, intercept=np.array(self.intercept),
                 source_sha256=np.array(source_sha256))

    @classmethod
    def load(cls, path: Path) -> tuple["NumpyLogisticPipeline", str]:
        with np.load(path, allow_pickle=False) as data:
            return cls(data["mean"], data["scale"], data["coef"], float(data["intercept"])), str(data["source_sha256"])


def export_numpy(stem: Path) -> Optional[Path]:
    """Write <stem>.npz next to <stem>.joblib if the model can run as numpy; None (nothing written) if it can't
    (gradient boosting, a model without a scaler). Exact maths, so routing doesn't change (see the parity test)."""
    joblib_path = stem.with_suffix(".joblib")
    try:
        model = NumpyLogisticPipeline.from_sklearn(joblib.load(joblib_path))
    except (PredictorError, AttributeError):
        return None
    out = stem.with_suffix(".npz")
    model.save(out, file_sha256(joblib_path))
    return out


def load_classifier_file(stem: Path):
    """The fitted model for one tier: the numpy export when it matches the .joblib, else the .joblib itself"""
    joblib_path, npz_path = stem.with_suffix(".joblib"), stem.with_suffix(".npz")
    if npz_path.exists():
        model, source = NumpyLogisticPipeline.load(npz_path)
        if source == file_sha256(joblib_path):
            return model
        logger.warning("%s doesn't match %s (re-export it with scripts/export_numpy_classifier.py); loading the "
                       ".joblib with scikit-learn instead", npz_path.name, joblib_path.name)
    return joblib.load(joblib_path)


@dataclass(frozen=True)
class Prediction:
    p_low: float                # chance the cheap tier passes 
    p_medium: float             # chance the medium tier passes (never below p_low, see predict_texts)
    raw_p_medium: float         # what the medium model said before that adjustment 
    task_type: str              
    task_type_guessed: bool     # True if nobody told us the task type and the keyword guesser picked it 
    # The inputs the classifier saw, kept so the request log can store them without computing them twice
    # (Phase 4 Step 8 storage). None when this predictor's feature set doesn't include them.
    features: Optional[tuple] = field(default=None, compare=False, repr=False)       # the 14 FEATURE_NAMES values
    embedding: Optional[np.ndarray] = field(default=None, compare=False, repr=False)  # float32, embeddings.MODEL_NAME


class Predictor:
    def __init__(self, name: str, version: int, models_dir: Path = MODELS_DIR):
        self.name = name 
        self.version = version
        self.models, self.meta = {}, {}
        for tier in TIERS:
            stem = Path(models_dir) / f"{name}_{tier}_v{version}"
            missing = [p.name for p in (stem.with_suffix(".joblib"), stem.with_suffix(".json")) if not p.exists()]
            if missing:
                raise PredictorError(
                    f"model file(s) {', '.join(missing)} not found in {models_dir}. The live classifier's files "
                    f"are committed to git (see .gitignore); a deploy must include them. Restore them with "
                    f"`git checkout -- models/{stem.name}.*`, train them with eval/train_classifier.py, or point "
                    f"routing.yaml's classifier at a version that exists")
            self.meta[tier] = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            # The file name picks the model; the version saved INSIDE the JSON must agree with it too,
            # so a renamed or copied file can't quietly serve a different model than routing.yaml names
            if self.meta[tier].get("version") != version:
                raise PredictorError(f"{stem.name}.json records version {self.meta[tier].get('version')!r}, "
                                     f"but version {version!r} was requested (routing.yaml classifier.version)")
            self.models[tier] = load_classifier_file(stem)

        feature_sets = {m["feature_set"] for m in self.meta.values()}
        if len(feature_sets) != 1:
            raise PredictorError(f"the low and medium models use different feature sets: {feature_sets}")
        self.feature_set = feature_sets.pop()
        self._check_versions()
        if self.feature_set == "task_only":
            self.task_columns = [c for c in self.meta["low"]["columns"] if c.startswith("task_")]
        self._check_columns()
    
    # ---------- load-time safety checks ----------
    
    def _check_versions(self) -> None:
        for tier, meta in self.meta.items():
            versions = meta.get("versions", {})
            if self.feature_set in USES_FEATURES and versions.get("features") != FEATURE_VERSION:
                raise PredictorError(f"{tier} model was trained on feature version {versions.get('features')!r}, "
                                     f"but features.py is version {FEATURE_VERSION!r}; retrain it")
            if self.feature_set in USES_EMBEDDINGS:
                from app.router.embeddings import MODEL_NAME
                if versions.get("embeddings") != MODEL_NAME:
                    raise PredictorError(f"{tier} model was trained on embeddings from "
                                         f"{versions.get('embeddings')!r}, but embeddings.py uses {MODEL_NAME!r}")
    
    def _check_columns(self) -> None:
        """Build one row and make sure each model expects exactly that many columns.
        This also loads the embedding model now, so the first real request isn't slow."""
        width = self._rows(["warm-up"],[None])[0].shape[1]
        for tier, model in self.models.items():
            expected = getattr(model, "n_features_in_", width)
            if expected != width:
                raise PredictorError(f"{tier} model expects {expected} columns, but the predictor builds {width}")
    
    # ---------- building inputs exactly like training did ----------
    
    def _rows(self, texts: list, task_types: list) -> tuple:
        """(X, final task types, guessed flags). Column order matches eval/train_classifier.feature_matrix:
            handcrafted features first, then embeddings"""
        tasks, guessed = [], []
        for text, given in zip(texts, task_types):
            task = normalize_task_type(given)
            guessed.append(task is None)
            tasks.append(task or guess_task_type(text))
        
        parts = []
        if self.feature_set in USES_FEATURES:
            feats = [extract_features(t, task) for t, task in zip(texts, tasks)]
            if self.feature_set == "task_only":
                parts.append(np.array([[f[c] for c in self.task_columns] for f in feats], dtype=np.float64))
            else:
                parts.append(np.array([to_vector(f) for f in feats], dtype=np.float64))
        if self.feature_set in USES_EMBEDDINGS:
            from app.router.embeddings import embed_texts
            from app.router.embeddings import BATCH_SIZE, LIVE_BATCH_SIZE
            vecs, _ = embed_texts(list(texts), batch_size=LIVE_BATCH_SIZE if len(texts) == 1 else BATCH_SIZE)
            parts.append(vecs.astype(np.float64))
        return np.hstack(parts), tasks, guessed
    
    # ---------- predicting ----------
    
    def predict_texts(self, texts: list, task_types: Optional[list] = None) -> list:
        """Predictions for many prompts at once (faster: embeddings are computed in one batch)"""
        task_types = task_types or [None] * len(texts)
        X, tasks, guessed = self._rows(texts, task_types)
        p_low = self.models["low"].predict_proba(X)[:, 1]
        p_med = self.models["medium"].predict_proba(X)[:, 1]
        features, embeddings = self._split_inputs(X)
        # The two models are trained seperately and can disagree oddly ("cheap 70%, medium 60%")
        # Medium is the stronger model, so it should never look less likely to pass than cheap 
        return [Prediction(p_low=float(lo), p_medium=float(max(md, lo)), raw_p_medium=float(md), task_type=task, task_type_guessed=g,
                           features=features[i], embedding=embeddings[i])
                for i, (lo, md, task, g) in enumerate(zip(p_low, p_med, tasks, guessed))]

    def _split_inputs(self, X: np.ndarray) -> tuple:
        """Per row: (the 14 handcrafted features or None, the float32 embedding or None), sliced out of the
        matrix the models just used, so nothing is computed twice. Column order is _rows' order."""
        n = len(X)
        width = 0
        features = [None] * n
        if self.feature_set in USES_FEATURES:
            width = len(self.task_columns) if self.feature_set == "task_only" else len(FEATURE_NAMES)
            if self.feature_set != "task_only":
                features = [tuple(float(v) for v in row) for row in X[:, :width]]
        embeddings = [None] * n
        if self.feature_set in USES_EMBEDDINGS:
            # embed_texts returns float32; _rows widened it to float64 exactly, so this narrowing is lossless
            embeddings = [row.astype(np.float32) for row in X[:, width:]]
        return features, embeddings
    
    def predict_text(self, text: str, task_type: Optional[str] = None) -> Prediction:
        return self.predict_texts([text], [task_type])[0]
    
    def predict_messages(self, messages: list, task_type: Optional[str] = None) -> Prediction:
        """For chat-style requests: uses the same text extraction as training (the user turns)"""
        return self.predict_text(prompt_text({"messages": messages}), task_type)


 # ---------- loading from config ----------
 
def load_classifier_config(path: Path = ROUTING_CONFIG_PATH) -> tuple:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    section = config.get("classifier")
    if not isinstance(section, dict) or "name" not in section or "version" not in section:
        raise PredictorError(f"{path}: needs a `classifier:` section with `name` and `version`")
    return str(section["name"]), int(section["version"])


@lru_cache(maxsize=1)
def load_predictor() -> Predictor:
    """The predictor named in routing.yaml, loaded once and reused"""
    name, version = load_classifier_config()
    return Predictor(name, version)


def decide(messages: list, quality: Optional[str] = None, task_type: Optional[str] = None) -> tuple:
    """The whole routing decision for one request: (RoutingDecision, Prediction).
    quality is the X-Triage-Quality header value (or None for the default).
    Raises QualityError for a bad header, so an API can answer 400 before doing any work."""
    tau = resolve_tau(quality, load_quality_settings())
    prediction = load_predictor().predict_messages(messages, task_type)
    return route(prediction.p_low, prediction.p_medium, tau), prediction


def main() -> None:
    parser = argparse.ArgumentParser(description="Show Triage's routing decision for one prompt.")
    parser.add_argument("prompt")
    parser.add_argument("--quality", default=None, help="cheap | balanced | high | a number 0-1")
    parser.add_argument("--task", default=None, help="task type, if known (otherwise it is guessed)")
    args = parser.parse_args()
    
    decision, p = decide([{"role": "user", "content": args.prompt}], args.quality, args.task)
    predictor = load_predictor()
    print(f"classifier: {predictor.name} v{predictor.version} ({predictor.feature_set})")
    print(f"task type:  {p.task_type}" + (" (guessed)" if p.task_type_guessed else ""))
    print(f"p(cheap passes) = {p.p_low:.2f}")
    print(f"p(medium passes) = {p.p_medium:.2f}"
          + (f" (raised from {p.raw_p_medium:.2f} to stay >= cheap)" if p.raw_p_medium < p.p_medium else ""))
    print(f"tau = {decision.tau:.2f} -> {decision.tier.upper()} ({decision.reason})")

if __name__ == "__main__":
    main()