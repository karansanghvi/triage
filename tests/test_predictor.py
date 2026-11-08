"""Predictor tests with tiny fake models trained on handcrafted features, so no embedding download is needed."""
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from app.router.features import FEATURE_NAMES, FEATURE_VERSION
from app.router.predictor import Predictor, PredictorError


def save_fake_models(models_dir, name="fake", feature_set="features", feature_version=FEATURE_VERSION,
                     columns=FEATURE_NAMES, flip_medium=False, file_version=1, recorded_version="same"):
    rng = np.random.default_rng(0)
    X = rng.random((60, len(columns)))
    recorded = file_version if recorded_version == "same" else recorded_version
    for tier in ("low", "medium"):
        y = (X[:, 0] > 0.5).astype(int)
        if tier == "medium" and flip_medium:
            y = 1 - y   # deliberately makes medium disagree with low
        model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
        joblib.dump(model, models_dir / f"{name}_{tier}_v{file_version}.joblib")
        meta = {"feature_set": feature_set, "versions": {"features": feature_version}, "columns": list(columns)}
        if recorded is not None:
            meta["version"] = recorded
        (models_dir / f"{name}_{tier}_v{file_version}.json").write_text(json.dumps(meta))


def test_predicts_probabilities(tmp_path):
    save_fake_models(tmp_path)
    p = Predictor("fake", 1, tmp_path).predict_text("Summarize this article in two sentences.")
    assert 0.0 <= p.p_low <= 1.0 and 0.0 <= p.p_medium <= 1.0


def test_medium_never_below_cheap(tmp_path):
    save_fake_models(tmp_path, flip_medium=True)
    predictor = Predictor("fake", 1, tmp_path)
    for p in predictor.predict_texts(["a", "Write a poem about rain " * 20, "What is 2+2?"]):
        assert p.p_medium >= p.p_low
        assert p.raw_p_medium <= p.p_medium


def test_task_type_given_or_guessed(tmp_path):
    save_fake_models(tmp_path)
    predictor = Predictor("fake", 1, tmp_path)
    given = predictor.predict_text("Who wrote Hamlet?", task_type="qa")
    guessed = predictor.predict_text("Write a Python function to reverse a list.")
    assert (given.task_type, given.task_type_guessed) == ("qa", False)
    assert (guessed.task_type, guessed.task_type_guessed) == ("coding", True)


def test_messages_use_user_turns(tmp_path):
    save_fake_models(tmp_path)
    predictor = Predictor("fake", 1, tmp_path)
    messages = [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "Hi there"}]
    assert predictor.predict_messages(messages) == predictor.predict_text("Hi there")


def test_task_only_models(tmp_path):
    task_columns = [c for c in FEATURE_NAMES if c.startswith("task_")]
    save_fake_models(tmp_path, feature_set="task_only", columns=task_columns)
    assert 0.0 <= Predictor("fake", 1, tmp_path).predict_text("Who wrote Hamlet?").p_low <= 1.0


def test_missing_model_fails_at_startup(tmp_path):
    with pytest.raises(PredictorError, match="not found"):
        Predictor("nothing", 1, tmp_path)


def test_stale_feature_version_fails_at_startup(tmp_path):
    save_fake_models(tmp_path, feature_version="0")
    with pytest.raises(PredictorError, match="feature version"):
        Predictor("fake", 1, tmp_path)


def test_wrong_column_count_fails_at_startup(tmp_path):
    save_fake_models(tmp_path, columns=FEATURE_NAMES[:5])
    with pytest.raises(PredictorError, match="columns"):
        Predictor("fake", 1, tmp_path)

# ---------- the version recorded inside the model JSON must match the requested version ----------

def test_recorded_version_must_match_requested_version(tmp_path):
    save_fake_models(tmp_path, file_version=2, recorded_version=1)       # e.g. a v1 file copied to _v2
    with pytest.raises(PredictorError, match="records version 1, but version 2 was requested"):
        Predictor("fake", 2, tmp_path)


def test_missing_recorded_version_fails(tmp_path):
    save_fake_models(tmp_path, recorded_version=None)
    with pytest.raises(PredictorError, match="records version None"):
        Predictor("fake", 1, tmp_path)


def test_matching_recorded_version_loads(tmp_path):
    save_fake_models(tmp_path, file_version=2)
    assert Predictor("fake", 2, tmp_path).version == 2


# ---------- the real Phase 4 models ----------

def test_real_v1_is_refused_after_feature_version_bump():
    with pytest.raises(PredictorError, match="feature version '2'"):
        Predictor("classifier_c_logreg_cv", 1)


@pytest.mark.parametrize("name, version, task_types", [
    ("classifier_c_logreg_cv", 2, "hand"), ("classifier_c_logreg_cv_guessed", 1, "guessed")])
def test_real_fit_set_candidates_load(monkeypatch, name, version, task_types):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")          # embedding model must be cached; never download
    p = Predictor(name, version)
    assert p.meta["low"]["version"] == version and p.meta["low"]["trained_on"] == "fit"
    assert p.meta["low"]["task_types"] == task_types and p.meta["low"]["versions"]["features"] == FEATURE_VERSION
    assert len(p.meta["low"]["oof_probabilities"]) == 481
