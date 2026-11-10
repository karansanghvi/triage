"""
predictor.py: turns a prompt into "chance the cheap tier passes" and "chance the medium tier passes".
 
    prompt -> features.py (14 clues) + embeddings.py (384 meaning numbers)
           -> the classifier named in routing.yaml -> p_low, p_medium
           -> quality.py compares them with tau -> low / medium / high
 
Everything is loaded once (load_predictor() is cached). At load time the predictor refuses to start if
the saved models don't match the current code (feature version, embedding model, number of columns),
because a silently mismatched model is one of the most common real-world ML bugs.
 
Try it:
    python -m app.router.predictor "Summarize this email in one sentence: ..."
    python -m app.router.predictor "What is 17 * 23?" --quality high --task reasoning
"""

from __future__ import annotations
 
import argparse
import json
from dataclasses import dataclass
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


class PredictorError(RuntimeError):
    """The configured classifier is missing or doesn't match the current one"""


@dataclass(frozen=True)
class Prediction:
    p_low: float                # chance the cheap tier passes 
    p_medium: float             # chance the medium tier passes (never below p_low, see predict_texts)
    raw_p_medium: float         # what the medium model said before that adjustment 
    task_type: str              
    task_type_guessed: bool     # True if nobody told us the task type and the keyword guesser picked it 


class Predictor:
    def __init__(self, name: str, version: int, models_dir: Path = MODELS_DIR):
        self.name = name 
        self.version = version
        self.models, self.meta = {}, {}
        for tier in TIERS:
            stem = Path(models_dir) / f"{name}_{tier}_v{version}"
            if not stem.with_suffix(".joblib").exists() or not stem.with_suffix(".json").exists():
                raise PredictorError(f"{stem.name}.joblib/.json not found in {models_dir}; "
                                     "train it with eval/train_classifier.py or fix routing.yaml")
            self.meta[tier] = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
            # The file name picks the model; the version saved INSIDE the JSON must agree with it too,
            # so a renamed or copied file can't quietly serve a different model than routing.yaml names
            if self.meta[tier].get("version") != version:
                raise PredictorError(f"{stem.name}.json records version {self.meta[tier].get('version')!r}, "
                                     f"but version {version!r} was requested (routing.yaml classifier.version)")
            self.models[tier] = joblib.load(stem.with_suffix(".joblib"))

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
            vecs, _ = embed_texts(list(texts))
            parts.append(vecs.astype(np.float64))
        return np.hstack(parts), tasks, guessed
    
    # ---------- predicting ----------
    
    def predict_texts(self, texts: list, task_types: Optional[list] = None) -> list:
        """Predictions for many prompts at once (faster: embeddings are computed in one batch)"""
        task_types = task_types or [None] * len(texts)
        X, tasks, guessed = self._rows(texts, task_types)
        p_low = self.models["low"].predict_proba(X)[:, 1]
        p_med = self.models["medium"].predict_proba(X)[:, 1]
        # The two models are trained seperately and can disagree oddly ("cheap 70%, medium 60%")
        # Medium is the stronger model, so it should never look less likely to pass than cheap 
        return [Prediction(p_low=float(lo), p_medium=float(max(md, lo)), raw_p_medium=float(md), task_type=task, task_type_guessed=g)
                for lo, md, task, g in zip(p_low, p_med, tasks, guessed)]
    
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