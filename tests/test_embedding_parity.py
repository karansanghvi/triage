"""The server's embedding model (onnxruntime + tokenizers + numpy) against the torch/transformers pipeline it
replaced (tests/reference_encoder.py) and against sentence-transformers, which produced the training embeddings.

On all 762 dataset prompts plus edge cases (long chunked prompts, Unicode, empty text): identical chunk strings,
vectors within 1e-5, and the same routing decision for every prompt at every quality preset.

Needs torch, transformers and sentence-transformers (requirements-dev.txt; the server doesn't install them) and
both models cached; skipped otherwise.
"""
import json

import numpy as np
import pytest

from app.registry import PROJECT_ROOT
from app.router import embeddings
from app.router.features import prompt_text

TOLERANCE = 1e-5
RECORDS = [json.loads(l) for l in (PROJECT_ROOT / "data" / "prompts.jsonl").read_text().splitlines() if l]
EDGE_CASES = ["  padded text \n", "", "x", "word " * 900,
              "Ünïcödé café — naïve “quotes” don't , . ! 中文字符 emoji 🎉 " * 40,
              "tabs\tand\nnew\nlines\r\n and   spaces", "a" * 5000, "Hello [SEP] [CLS] [PAD] [UNK] [MASK]"]


@pytest.fixture(scope="module")
def encoders():
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.reference_encoder import TorchMiniLMEncoder
    try:
        ours = embeddings.get_model()
    except embeddings.EmbeddingModelMissing as error:
        pytest.skip(str(error))
    try:
        reference = TorchMiniLMEncoder()
    except OSError as error:
        pytest.skip(f"the torch reference can't load the model offline: {error}")
    return ours, reference


@pytest.fixture(scope="module")
def texts():
    assert len(RECORDS) == 762
    return [prompt_text(r) for r in RECORDS] + EDGE_CASES


@pytest.fixture(scope="module")
def vectors(encoders, texts):
    ours, reference = encoders
    a, chunks = embeddings.embed_texts(texts, ours, batch_size=embeddings.LIVE_BATCH_SIZE)
    b, _ = embeddings.embed_texts(texts, reference, batch_size=embeddings.LIVE_BATCH_SIZE)
    return a, b, chunks


def test_chunk_strings_are_identical(encoders, texts):
    ours, reference = encoders
    different = [i for i, t in enumerate(texts) if embeddings.chunk_text(t, ours) != embeddings.chunk_text(t, reference)]
    assert different == []


def test_vectors_match_the_torch_pipeline(vectors, record_property):
    ours, theirs, chunks = vectors
    assert sum(c > 1 for c in chunks) >= 50 and max(chunks) >= 9       # the chunked path is well covered
    worst = float(np.max(np.abs(ours - theirs)))
    record_property("max_abs_diff", worst)
    print(f"\nmax |onnx - torch| over {len(ours)} texts: {worst:.3e}")
    assert worst < TOLERANCE


def test_batch_size_doesnt_change_the_result(encoders, texts):
    ours, _ = encoders
    sample = texts[:40] + EDGE_CASES
    one, _ = embeddings.embed_texts(sample, ours, batch_size=1)
    many, _ = embeddings.embed_texts(sample, ours, batch_size=embeddings.BATCH_SIZE)
    assert np.max(np.abs(one - many)) < TOLERANCE                      # padding never leaks into the mean


def test_routing_decisions_are_identical_at_every_preset(vectors, texts):
    from app.router.features import extract_features, guess_task_type, normalize_task_type, to_vector
    from app.router.predictor import Predictor, load_classifier_config
    from app.router.quality import load_quality_settings, route
    ours, theirs, _ = vectors
    texts = texts[:len(RECORDS)]
    tasks = [normalize_task_type(r.get("task_type")) or guess_task_type(t) for r, t in zip(RECORDS, texts)]
    feats = np.array([to_vector(extract_features(t, k)) for t, k in zip(texts, tasks)], dtype=np.float64)
    models = Predictor(*load_classifier_config()).models          # what the server runs (numpy)

    def probabilities(vecs):
        X = np.hstack([feats, vecs[:len(RECORDS)].astype(np.float64)])
        lo = models["low"].predict_proba(X)[:, 1]
        return lo, np.maximum(models["medium"].predict_proba(X)[:, 1], lo)

    a, b = probabilities(ours), probabilities(theirs)
    presets = load_quality_settings().presets
    assert presets
    for preset, tau in presets.items():
        assert [route(lo, md, tau).tier for lo, md in zip(*a)] == [route(lo, md, tau).tier for lo, md in zip(*b)], \
            preset


def test_vectors_match_sentence_transformers(encoders, texts):
    st = pytest.importorskip("sentence_transformers")
    ours, _ = encoders
    try:
        reference = st.SentenceTransformer(embeddings.MODEL_NAME, revision=embeddings.MODEL_REVISION)
    except OSError as error:
        pytest.skip(f"sentence-transformers can't load the model offline: {error}")
    assert reference.max_seq_length == ours.max_seq_length == 256
    sample = texts[:100] + EDGE_CASES
    got, _ = embeddings.embed_texts(sample, ours)
    expected = []
    for text in sample:
        pieces = embeddings.chunk_text(text or "", ours)
        vecs = reference.encode(pieces, convert_to_numpy=True, normalize_embeddings=True)
        w = np.array([max(len(p), 1) for p in pieces], dtype=np.float64)
        v = (vecs * w[:, None]).sum(0)
        expected.append(v / (np.linalg.norm(v) or 1))
    assert np.max(np.abs(got - np.array(expected, dtype=np.float32))) < TOLERANCE


# ---------- the encoder's settings (no torch needed) ----------

@pytest.fixture(scope="module")
def onnx_model():
    try:
        return embeddings.get_model()
    except embeddings.EmbeddingModelMissing as error:
        pytest.skip(str(error))


def test_inputs_are_cut_at_256_tokens_not_the_tokenizer_files_128(onnx_model):
    """tokenizer.json in the model repo truncates and pads at 128; the encoder must override both"""
    text = "word " * 200                                   # 200 tokens: one chunk, longer than 128
    assert len(onnx_model.token_ids(text)) == 200
    assert embeddings.chunk_text(text, onnx_model) == [text]
    enc = onnx_model.tokenizer.encode(text)
    assert len(enc.ids) == 202 and sum(enc.attention_mask) == 202          # [CLS] + 200 + [SEP]: not cut to 128
    assert len(onnx_model.tokenizer.encode("word " * 400).ids) == 256
    short = onnx_model.tokenizer.encode_batch(["hi", "hello there friend"])
    assert len(short[0].ids) == len(short[1].ids) < 128                     # padded to the batch, not to 128


def test_one_thread_cpu_only_fp32(onnx_model):
    assert onnx_model.threads == 1
    assert onnx_model.session.get_providers() == ["CPUExecutionProvider"]
    assert [i.type for i in onnx_model.session.get_inputs()] == ["tensor(int64)"] * 3
    assert onnx_model.session.get_outputs()[0].type == "tensor(float)"


def test_the_model_files_are_pinned_to_one_revision():
    assert embeddings.MODEL_FILES == ("onnx/model.onnx", "tokenizer.json")
    assert len(embeddings.MODEL_REVISION) == 40


def test_a_missing_model_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    import huggingface_hub.constants as c
    monkeypatch.setattr(c, "HF_HUB_OFFLINE", True, raising=False)
    monkeypatch.setattr(c, "HF_HUB_CACHE", str(tmp_path / "hub"), raising=False)
    with pytest.raises(embeddings.EmbeddingModelMissing, match="fetch_embedding_model"):
        embeddings.MiniLMEncoder(revision="0" * 40)


def test_threads_setting(monkeypatch):
    monkeypatch.delenv(embeddings.THREADS_ENV, raising=False)
    assert embeddings.THREADS_ENV == "TRIAGE_EMBEDDING_THREADS"
    assert embeddings.embedding_threads() == 1
    monkeypatch.setenv(embeddings.THREADS_ENV, "2")
    assert embeddings.embedding_threads() == 2
    for bad in ("0", "two"):
        monkeypatch.setenv(embeddings.THREADS_ENV, bad)
        with pytest.raises(ValueError):
            embeddings.embedding_threads()


def test_the_server_never_imports_torch_or_transformers():
    import subprocess
    import sys
    code = ("import sys; from app.router.embeddings import get_model, embed_one; embed_one('hi'); "
            "import app.api.main; bad = [m for m in ('torch', 'transformers', 'scipy', 'sentence_transformers') "
            "if m in sys.modules]; print(bad); sys.exit(1 if bad else 0)")
    result = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
                            env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"})
    assert result.returncode == 0, result.stdout + result.stderr
