"""The server's embedding model (plain transformers) against sentence-transformers, which produced the training
embeddings. Same vectors within float32 rounding on the dataset's prompts (long ones chunked), and the same routing
decision for every prompt at every quality preset when each pipeline feeds the classifier.

Needs sentence-transformers (a test dependency; the server doesn't install-time need it) and the model cached.
"""
import json

import numpy as np
import pytest

from app.registry import PROJECT_ROOT
from app.router import embeddings
from app.router.features import prompt_text

st = pytest.importorskip("sentence_transformers")


@pytest.fixture(scope="module")
def both():
    try:
        light = embeddings.get_model()
    except embeddings.EmbeddingModelMissing as error:
        pytest.skip(str(error))
    try:
        reference = st.SentenceTransformer(embeddings.MODEL_NAME, revision=embeddings.MODEL_REVISION)
    except OSError as error:
        pytest.skip(f"sentence-transformers can't load the model offline: {error}")
    records = [json.loads(l) for l in (PROJECT_ROOT / "data" / "prompts.jsonl").read_text().splitlines() if l]
    texts = [prompt_text(r) for r in records] + ["  padded text \n", "", "x", "word " * 900]
    return light, reference, texts


def _reference_embed(reference, light, texts):
    """embed_texts' chunk-and-average, with sentence-transformers doing the encoding"""
    out = []
    for text in texts:
        pieces = embeddings.chunk_text(text or "", light)
        vecs = reference.encode(pieces, convert_to_numpy=True, normalize_embeddings=True)
        w = np.array([max(len(p), 1) for p in pieces], dtype=np.float64)
        v = (vecs * w[:, None]).sum(0)
        out.append(v / (np.linalg.norm(v) or 1))
    return np.array(out, dtype=np.float32)


def test_the_same_settings_as_the_sentence_transformers_model(both):
    light, reference, _ = both
    assert light.max_seq_length == reference.max_seq_length == 256


def test_vectors_match_sentence_transformers(both):
    light, reference, texts = both
    ours, chunks = embeddings.embed_texts(texts, light, batch_size=embeddings.LIVE_BATCH_SIZE)
    theirs = _reference_embed(reference, light, texts)
    assert max(chunks) > 1                                        # the chunked path is covered
    assert np.max(np.abs(ours - theirs)) < 1e-6


def test_routing_decisions_are_identical_at_every_preset(both):
    import joblib
    from app.router.features import extract_features, normalize_task_type, guess_task_type, to_vector
    from app.router.predictor import MODELS_DIR, load_classifier_config
    from app.router.quality import load_quality_settings, route
    light, reference, texts = both
    texts = texts[:-4]                                             # the dataset's prompts
    records = [json.loads(l) for l in (PROJECT_ROOT / "data" / "prompts.jsonl").read_text().splitlines() if l]
    tasks = [normalize_task_type(r.get("task_type")) or guess_task_type(t) for r, t in zip(records, texts)]
    feats = np.array([to_vector(extract_features(t, k)) for t, k in zip(texts, tasks)], dtype=np.float64)
    name, version = load_classifier_config()
    models = {tier: joblib.load(MODELS_DIR / f"{name}_{tier}_v{version}.joblib") for tier in ("low", "medium")}

    def probabilities(vectors):
        X = np.hstack([feats, vectors.astype(np.float64)])
        lo = models["low"].predict_proba(X)[:, 1]
        return lo, np.maximum(models["medium"].predict_proba(X)[:, 1], lo)

    ours = probabilities(embeddings.embed_texts(texts, light)[0])
    theirs = probabilities(_reference_embed(reference, light, texts))
    for preset, tau in load_quality_settings().presets.items():
        a = [route(lo, md, tau).tier for lo, md in zip(*ours)]
        b = [route(lo, md, tau).tier for lo, md in zip(*theirs)]
        assert a == b, preset


def test_a_missing_model_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    with pytest.raises(embeddings.EmbeddingModelMissing, match="fetch_embedding_model"):
        embeddings.MiniLMEncoder(revision="0" * 40)


def test_torch_threads_setting(monkeypatch):
    import torch
    before = torch.get_num_threads()
    try:
        monkeypatch.delenv(embeddings.TORCH_THREADS_ENV, raising=False)
        assert embeddings.set_torch_threads() == 1 and torch.get_num_threads() == 1
        monkeypatch.setenv(embeddings.TORCH_THREADS_ENV, "2")
        assert embeddings.set_torch_threads() == 2
        for bad in ("0", "two"):
            monkeypatch.setenv(embeddings.TORCH_THREADS_ENV, bad)
            with pytest.raises(ValueError):
                embeddings.set_torch_threads()
    finally:
        torch.set_num_threads(before)
