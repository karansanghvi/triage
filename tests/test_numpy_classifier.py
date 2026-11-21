"""The live classifier as numpy (what the server runs) against the scikit-learn pipeline it was exported from.

On real inputs (the dataset's prompts, with real embeddings): probabilities within 1e-9, and the same routing
decision at every quality preset. Plus the export's safety rules: an .npz whose recorded .joblib hash doesn't match
is ignored, and a model that can't run as numpy is left alone.
"""
import json
import shutil

import joblib
import numpy as np
import pytest

from app.registry import PROJECT_ROOT
from app.router.embeddings import EmbeddingModelMissing
from app.router.predictor import (MODELS_DIR, TIERS, NumpyLogisticPipeline, Predictor, export_numpy, file_sha256,
                                  load_classifier_config, load_classifier_file)
from app.router.quality import load_quality_settings, route

NAME, VERSION = load_classifier_config()


@pytest.fixture(scope="module")
def real_inputs():
    """(X, sklearn models, numpy models) for every prompt in data/prompts.jsonl, built exactly as live requests are"""
    try:
        predictor = Predictor(NAME, VERSION)
    except EmbeddingModelMissing as error:
        pytest.skip(f"embedding model not available offline: {error}")
    records = [json.loads(line) for line in (PROJECT_ROOT / "data" / "prompts.jsonl").read_text().splitlines() if line]
    from app.router.features import prompt_text
    texts = [prompt_text(r) for r in records]
    X, _, _ = predictor._rows(texts, [r.get("task_type") for r in records])
    sk = {tier: joblib.load(MODELS_DIR / f"{NAME}_{tier}_v{VERSION}.joblib") for tier in TIERS}
    return X, sk, predictor.models


def test_the_server_runs_the_live_classifier_as_numpy(real_inputs):
    _, _, models = real_inputs
    assert all(isinstance(m, NumpyLogisticPipeline) for m in models.values())


def test_probabilities_match_scikit_learn_within_1e_9(real_inputs):
    X, sk, models = real_inputs
    assert X.shape[0] >= 700
    for tier in TIERS:
        expected = sk[tier].predict_proba(X)[:, 1]
        got = models[tier].predict_proba(X)[:, 1]
        assert np.max(np.abs(expected - got)) < 1e-9, tier


def test_routing_decisions_are_identical_at_every_preset(real_inputs):
    X, sk, models = real_inputs
    presets = load_quality_settings().presets
    assert presets
    lo_sk, md_sk = sk["low"].predict_proba(X)[:, 1], sk["medium"].predict_proba(X)[:, 1]
    lo_np, md_np = models["low"].predict_proba(X)[:, 1], models["medium"].predict_proba(X)[:, 1]
    for name, tau in presets.items():
        for i in range(len(X)):
            a = route(lo_sk[i], max(md_sk[i], lo_sk[i]), tau).tier
            b = route(lo_np[i], max(md_np[i], lo_np[i]), tau).tier
            assert a == b, (name, i)


def test_an_npz_whose_joblib_changed_is_ignored(tmp_path, caplog):
    stem = tmp_path / f"{NAME}_low_v{VERSION}"
    for suffix in (".joblib", ".npz"):
        shutil.copy(MODELS_DIR / f"{NAME}_low_v{VERSION}{suffix}", stem.with_suffix(suffix))
    assert isinstance(load_classifier_file(stem), NumpyLogisticPipeline)
    model = joblib.load(stem.with_suffix(".joblib"))
    model.steps[-1][1].intercept_ = model.steps[-1][1].intercept_ + 1.0       # a "retrained" model
    joblib.dump(model, stem.with_suffix(".joblib"))
    with caplog.at_level("WARNING"):
        loaded = load_classifier_file(stem)
    assert not isinstance(loaded, NumpyLogisticPipeline)
    assert "doesn't match" in caplog.text


def test_no_npz_loads_the_joblib(tmp_path):
    stem = tmp_path / f"{NAME}_low_v{VERSION}"
    shutil.copy(MODELS_DIR / f"{NAME}_low_v{VERSION}.joblib", stem.with_suffix(".joblib"))
    assert not isinstance(load_classifier_file(stem), NumpyLogisticPipeline)


def test_export_records_the_joblib_hash_and_round_trips(tmp_path):
    stem = tmp_path / "m_low_v1"
    shutil.copy(MODELS_DIR / f"{NAME}_low_v{VERSION}.joblib", stem.with_suffix(".joblib"))
    out = export_numpy(stem)
    model, source = NumpyLogisticPipeline.load(out)
    assert source == file_sha256(stem.with_suffix(".joblib"))
    X = np.random.default_rng(0).normal(size=(50, model.n_features_in_))
    ref = joblib.load(stem.with_suffix(".joblib")).predict_proba(X)[:, 1]
    assert np.max(np.abs(model.predict_proba(X)[:, 1] - ref)) < 1e-9


def test_a_model_that_cant_run_as_numpy_is_left_alone(tmp_path):
    source = MODELS_DIR / "classifier_c_gboost_low_v1.joblib"
    if not source.exists():
        pytest.skip("no gradient-boosting model file here")
    stem = tmp_path / "gb_low_v1"
    shutil.copy(source, stem.with_suffix(".joblib"))
    assert export_numpy(stem) is None and not stem.with_suffix(".npz").exists()


def test_the_committed_exports_match_the_committed_models():
    for tier in TIERS:
        stem = MODELS_DIR / f"{NAME}_{tier}_v{VERSION}"
        _, source = NumpyLogisticPipeline.load(stem.with_suffix(".npz"))
        assert source == file_sha256(stem.with_suffix(".joblib")), f"re-export {stem.name}"
