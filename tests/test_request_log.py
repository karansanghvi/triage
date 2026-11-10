"""The SQLite request log, JSON request lines and Step 8 storage (Phase 4 Steps 8 and 10). Fake models only.

These use the real route_and_stream (app/api/core.py), not the reference version.
"""
import io
import json
import logging
import sqlite3
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth, key_id
from app.api.fake_models import FakeBehavior
from app.api.main import create_app, request_logger
from app.api.request_log import (SCHEMA_VERSION, InMemoryRequestLog, RequestRow, SQLiteRequestLog,
                                 embedding_from_bytes, embedding_to_bytes, is_training_eligible, open_request_log)
from app.router.embeddings import MODEL_NAME
from app.router.features import FEATURE_NAMES, FEATURE_VERSION
from app.router.predictor import Prediction
from tests.api_helpers import ADMIN_KEY, API_KEY, OTHER_KEY, make_deps

AUTH = Auth([API_KEY, OTHER_KEY], [ADMIN_KEY])
SECRET_PROMPT = "ZEBRA-PROMPT-TEXT-918273 please add 2 and 3"
SECRET_ANSWER = "UNICORN-ANSWER-TEXT-556677"
ASK = [{"role": "user", "content": SECRET_PROMPT}]


class VectorPredictor:
    """Like the real predictor: returns the features and float32 embedding it 'computed' with the Prediction."""

    def __init__(self, p_low=0.4, p_medium=0.95, seed=0):
        rng = np.random.default_rng(seed)
        self.p_low, self.p_medium = p_low, p_medium
        self.features = tuple(float(x) for x in rng.normal(size=len(FEATURE_NAMES)) * 1e3)
        self.embedding = rng.normal(size=384).astype(np.float32)
        self.calls = 0

    def predict_messages(self, messages, task_type=None):
        self.calls += 1
        return Prediction(self.p_low, self.p_medium, self.p_medium, task_type or "qa", task_type is None,
                          features=self.features, embedding=self.embedding)


def app_with(tmp_path, script=None, predictor=None, name="requests.sqlite3"):
    log = SQLiteRequestLog(tmp_path / name)
    deps = make_deps(script, predictor=predictor or VectorPredictor(), log=log)
    return TestClient(create_app(deps=deps, auth=AUTH)), deps, log


def post(http, body=None, key=API_KEY, headers=None):
    return http.post("/v1/chat/completions", json=body or {"model": "auto", "messages": ASK},
                     headers={"Authorization": f"Bearer {key}"} | (headers or {}))


def stream(http, body=None):
    with http.stream("POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"},
                     json=body or {"model": "auto", "messages": ASK, "stream": True}) as r:
        return r.headers, "".join(r.iter_text())


def auth(key=API_KEY):
    return {"Authorization": f"Bearer {key}"}


def row(i=0, **kw):
    base = dict(request_id=f"r{i}", timestamp=f"2026-11-09T00:00:{i % 60:02d}+00:00", key_id="abcd1234",
                requested_model="auto", stream=False)
    return RequestRow(**(base | kw))


# ---------- schema ----------

def test_schema_tables_indexes_version_and_wal(tmp_path):
    log = SQLiteRequestLog(tmp_path / "r.sqlite3")
    db = sqlite3.connect(tmp_path / "r.sqlite3")
    assert db.execute("SELECT version FROM schema_version").fetchall() == [(SCHEMA_VERSION,)]
    cols = [c[1] for c in db.execute("PRAGMA table_info(requests)")]
    assert cols[0] == "id" and cols[1:] == [f for f in RequestRow.__dataclass_fields__]
    indexes = {i[1] for i in db.execute("PRAGMA index_list(requests)")}
    assert {"idx_requests_request_id", "idx_requests_key_id", "idx_requests_timestamp"} <= indexes
    assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    log.close()


def test_a_database_from_another_schema_version_is_refused(tmp_path):
    SQLiteRequestLog(tmp_path / "r.sqlite3").close()
    db = sqlite3.connect(tmp_path / "r.sqlite3")
    db.execute("UPDATE schema_version SET version = 99")
    db.commit()
    with pytest.raises(RuntimeError, match="schema version 99"):
        SQLiteRequestLog(tmp_path / "r.sqlite3")


def test_log_location_and_kind_come_from_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIAGE_LOG_DB", str(tmp_path / "env.sqlite3"))
    log = open_request_log()
    assert isinstance(log, SQLiteRequestLog) and log.path == tmp_path / "env.sqlite3"
    log.close()
    monkeypatch.setenv("TRIAGE_LOG", "memory")
    assert isinstance(open_request_log(), InMemoryRequestLog)


# ---------- survives a restart ----------

def test_rows_survive_closing_and_reopening(tmp_path):
    log = SQLiteRequestLog(tmp_path / "r.sqlite3")
    written = [row(i, tier="medium", fell_back=False, attempts=[{"model": "groq-small", "outcome": "success",
                                                                 "error_kind": None}], price_as_cost=0.001 * i)
               for i in range(5)]
    for r in written:
        log.write(r)
    log.close()
    again = SQLiteRequestLog(tmp_path / "r.sqlite3")
    assert again.rows() == written and again.get("r3") == written[3]
    assert again.rows(key_id="abcd1234") == written and again.rows(key_id="other") == []


def test_stats_and_request_lookup_are_the_same_after_a_restart(tmp_path):
    http, deps, log = app_with(tmp_path, {"groq-small": FakeBehavior(text=SECRET_ANSWER)})
    post(http)
    post(http, {"model": "groq-large", "messages": ASK})
    stream(http)
    post(http, key=OTHER_KEY)
    post(http, {"model": "nope", "messages": ASK})                            # an error row too
    rid = post(http).headers["x-triage-request-id"]
    before = {k: http.get("/v1/stats", headers=auth(k)).json() for k in (API_KEY, OTHER_KEY, ADMIN_KEY)}
    looked_up = http.get(f"/v1/requests/{rid}", headers=auth()).json()
    assert before[ADMIN_KEY]["requests"] == 6 and before[API_KEY]["requests"] == 5 and before[API_KEY]["errors"] == 1
    log.close()

    http2, _, _ = app_with(tmp_path)                                          # a fresh server, same file
    after = {k: http2.get("/v1/stats", headers=auth(k)).json() for k in (API_KEY, OTHER_KEY, ADMIN_KEY)}
    assert after == before
    assert http2.get(f"/v1/requests/{rid}", headers=auth()).json() == looked_up
    assert http2.get(f"/v1/requests/{rid}", headers=auth(OTHER_KEY)).status_code == 404


# ---------- many writers ----------

def test_many_threads_writing_at_once_lose_no_rows(tmp_path):
    log = SQLiteRequestLog(tmp_path / "r.sqlite3")
    start = threading.Barrier(16)

    def writer(t):
        start.wait()
        for i in range(50):
            log.write(row(t * 1000 + i, key_id=f"key{t:04d}"))
    threads = [threading.Thread(target=writer, args=(t,)) for t in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ids = [r.request_id for r in log.rows()]
    assert len(ids) == 800 == len(set(ids))
    assert all(len(log.rows(key_id=f"key{t:04d}")) == 50 for t in range(16))


def test_two_connections_to_the_same_file_lose_no_rows(tmp_path):
    """Like two server processes sharing one file: WAL plus a busy timeout serialize their writes."""
    a, b = SQLiteRequestLog(tmp_path / "r.sqlite3"), SQLiteRequestLog(tmp_path / "r.sqlite3")

    def writer(log, offset):
        for i in range(200):
            log.write(row(offset + i))
    threads = [threading.Thread(target=writer, args=(a, 0)), threading.Thread(target=writer, args=(b, 10_000))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(SQLiteRequestLog(tmp_path / "r.sqlite3").rows()) == 400


def test_concurrent_api_requests_each_write_one_row(tmp_path):
    http, deps, log = app_with(tmp_path)
    results = []

    def client():
        results.append(post(http).status_code)
    threads = [threading.Thread(target=client) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == [200] * 12 and len(log.rows()) == 12


# ---------- Step 8 storage: features and embedding ----------

def test_features_and_embedding_round_trip_exactly(tmp_path):
    rng = np.random.default_rng(1)
    feats = [float(x) for x in rng.normal(size=14) * 1e6] + []
    feats[0] = 1 / 3                                                     # awkward floats survive JSON exactly
    emb = rng.normal(size=384).astype(np.float32)
    log = SQLiteRequestLog(tmp_path / "r.sqlite3")
    log.write(row(features=feats, embedding=embedding_to_bytes(emb), feature_version=FEATURE_VERSION,
                  embedding_model=MODEL_NAME))
    log.close()
    back = SQLiteRequestLog(tmp_path / "r.sqlite3").get("r0")
    assert back.features == feats
    out = embedding_from_bytes(back.embedding)
    assert out.dtype == np.float32 and out.shape == (384,) and np.array_equal(out, emb)
    assert len(back.embedding) == 384 * 4


def test_auto_requests_store_the_predictors_own_vectors(tmp_path):
    pred = VectorPredictor()
    http, deps, log = app_with(tmp_path, predictor=pred)
    rid = post(http).headers["x-triage-request-id"]
    r = log.get(rid)
    assert r.features == list(pred.features) and len(r.features) == 14
    assert np.array_equal(embedding_from_bytes(r.embedding), pred.embedding)
    assert (r.feature_version, r.embedding_model) == (FEATURE_VERSION, MODEL_NAME)
    assert pred.calls == 1                                                # predicted once, nothing recomputed
    public = http.get(f"/v1/requests/{rid}", headers=auth()).json()
    assert public["vectors_stored"] is True                               # the API shows only that they exist
    assert "features" not in public and "embedding" not in public


def test_streamed_auto_requests_store_vectors_too(tmp_path):
    http, deps, log = app_with(tmp_path)
    h, body = stream(http)
    r = log.get(h["x-triage-request-id"])
    assert r.status == "ok" and r.features is not None and r.embedding is not None


def test_pass_through_and_error_rows_have_no_features(tmp_path):
    http, deps, log = app_with(tmp_path, {"groq-small": FakeBehavior(fail_after=1),
                                          "gemini-flash": FakeBehavior(fail_before="server")})
    passthrough = post(http, {"model": "groq-large", "messages": ASK}).headers["x-triage-request-id"]
    post(http, {"model": "nope", "messages": ASK})                        # 404
    stream(http)                                                          # groq-small breaks mid-stream
    post(http, {"model": "auto", "messages": ASK}, headers={"X-Triage-Quality": "0.99"})   # high: ok
    rows = log.rows()
    assert [r.status for r in rows] == ["ok", "error", "error", "ok"]
    for r in rows[:3]:
        assert (r.features, r.embedding, r.feature_version, r.embedding_model) == (None, None, None, None), r.status
    assert log.get(passthrough).requested_model == "groq-large"
    assert rows[3].features is not None                                   # the auto request that answered


def test_real_predictor_vectors_are_reused_not_recomputed(tmp_path, monkeypatch):
    """With the real classifier v2: the stored vectors are the ones it predicted from, embedded once."""
    from app.api.deps import load_predictor_checked
    from app.router import embeddings
    from app.router.features import extract_features, prompt_text, to_vector
    predictor, _ = load_predictor_checked()
    calls = []
    real = embeddings.embed_texts
    monkeypatch.setattr(embeddings, "embed_texts", lambda texts, model=None: calls.append(texts) or real(texts, model))
    http, deps, log = app_with(tmp_path, predictor=predictor)
    rid = post(http, headers={"X-Triage-Task": "reasoning"}).headers["x-triage-request-id"]
    assert len(calls) == 1                                                # one embedding for the whole request
    r = log.get(rid)
    text = prompt_text({"messages": ASK})
    assert r.features == to_vector(extract_features(text, "reasoning"))
    assert np.array_equal(embedding_from_bytes(r.embedding), real([text])[0][0])


# ---------- training_eligible ----------

@pytest.mark.parametrize("case,expected", [
    ("auto, Phase 2 settings, answered", True),
    ("auto, streamed, answered", True),
    ("fell back to another model", False),
    ("temperature differs from training", False),
    ("max_tokens differs from training", False),
    ("every model failed", False),
    ("failed mid-stream", False),
    ("pass-through", False),
    ("unknown model", False),
])
def test_training_eligible(tmp_path, case, expected):
    script = {"fell back to another model": {"groq-small": FakeBehavior(fail_before="server")},
              "every model failed": {"groq-small": FakeBehavior(fail_before="server"),
                                     "gemini-flash": FakeBehavior(fail_before="server")},
              "failed mid-stream": {"groq-small": FakeBehavior(fail_after=1)}}.get(case)
    http, deps, log = app_with(tmp_path, script)
    body = {"model": "auto", "messages": ASK}
    if case == "temperature differs from training":
        body["temperature"] = 0.5
    elif case == "max_tokens differs from training":
        body["max_tokens"] = 256
    elif case == "pass-through":
        body["model"] = "groq-small"
    elif case == "unknown model":
        body["model"] = "nope"
    if "stream" in case:
        stream(http, body | {"stream": True})
    else:
        post(http, body)
    r = log.rows()[-1]
    assert r.training_eligible is expected, (case, r.status, r.fell_back, r.settings_match_training)


def test_training_eligible_rule():
    ok = row(status="ok", fell_back=False, settings_match_training=True)
    assert is_training_eligible(ok)
    for change in ({"requested_model": "groq-small"}, {"status": "error"}, {"fell_back": True},
                   {"fell_back": None}, {"settings_match_training": False}, {"settings_match_training": None}):
        assert not is_training_eligible(RequestRow(**(ok.to_dict() | change))), change


# ---------- no keys, prompt text or answer text anywhere ----------

@pytest.fixture
def json_lines():
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    request_logger.addHandler(handler)
    yield buffer
    request_logger.removeHandler(handler)


def test_no_key_prompt_or_answer_text_in_the_database_or_the_json_logs(tmp_path, json_lines):
    script = {m: FakeBehavior(text=SECRET_ANSWER) for m in ("groq-allam", "groq-small", "groq-large", "gemini-flash")}
    http, deps, log = app_with(tmp_path, script)
    r = post(http)
    assert r.json()["choices"][0]["message"]["content"] == SECRET_ANSWER     # the answer really went through
    stream(http)
    post(http, {"model": "groq-large", "messages": ASK, "stream": False})
    post(http, {"model": "auto", "messages": ASK}, key=ADMIN_KEY)
    post(http, key="sk-wrong-key-never-stored")                              # 401: not logged at all
    post(http, {"model": "nope", "messages": ASK})
    log.close()
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("requests.sqlite3"))
    assert len(raw) > 10_000
    lines = json_lines.getvalue()
    assert len(lines.strip().splitlines()) == 5
    for secret in (API_KEY, ADMIN_KEY, "sk-wrong-key-never-stored", SECRET_PROMPT, "ZEBRA-PROMPT", SECRET_ANSWER,
                   "UNICORN-ANSWER"):
        assert secret.encode() not in raw, secret
        assert secret not in lines, secret
    assert key_id(API_KEY).encode() in raw                                   # only the short hash is kept


def test_json_log_lines_have_the_request_summary(tmp_path, json_lines):
    http, deps, log = app_with(tmp_path)
    rid = post(http).headers["x-triage-request-id"]
    post(http, {"model": "nope", "messages": ASK})
    first, second = [json.loads(l) for l in json_lines.getvalue().splitlines()]
    assert first["request_id"] == rid and first["tier"] == "medium" and first["served_model"] == "groq-small"
    assert first["status"] == "ok" and first["latency_ms"] > 0 and first["price_as_cost"] > 0
    assert first["training_eligible"] is True and first["error_kind"] is None
    assert second["status"] == "error" and second["error_kind"] == "model_not_found"
    assert not {"messages", "content", "text", "features", "embedding", "prompt_hmac", "prompt_sha256"} & set(first)


# ---------- prompt HMAC (keyed with TRIAGE_HASH_SECRET) ----------

def test_same_prompt_gives_different_values_under_different_secrets():
    from app.api.request_log import prompt_hmac
    a, b = prompt_hmac(ASK, b"secret-one"), prompt_hmac(ASK, b"secret-two")
    assert a != b and len(a) == len(b) == 64
    assert prompt_hmac(ASK, b"secret-one") == a                           # stable under one secret
    assert prompt_hmac([{"role": "user", "content": "something else"}], b"secret-one") != a
    import hashlib
    plain = hashlib.sha256(json.dumps(ASK, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    assert plain not in (a, b)                                            # not the guessable plain hash


def test_the_app_stores_the_hmac_under_its_own_secret(tmp_path):
    from app.api.request_log import prompt_hmac
    values = []
    for i, secret in enumerate((b"server-secret-a", b"server-secret-b")):
        log = SQLiteRequestLog(tmp_path / f"r{i}.sqlite3")
        http = TestClient(create_app(deps=make_deps(predictor=VectorPredictor(), log=log, hash_secret=secret),
                                     auth=AUTH))
        rid = post(http).headers["x-triage-request-id"]
        values.append(log.get(rid).prompt_hmac)
        assert values[-1] == prompt_hmac(ASK, secret)
        raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith(f"r{i}.sqlite3"))
        assert secret not in raw                                          # the secret itself is never stored
    assert values[0] != values[1]


def test_sqlite_log_refuses_to_start_without_a_hash_secret(tmp_path, monkeypatch):
    from app.api import deps as deps_module
    from app.api.fake_models import FakeModels
    monkeypatch.delenv("TRIAGE_HASH_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="TRIAGE_HASH_SECRET"):
        deps_module.build_deps(VectorPredictor(), FakeModels(), log=SQLiteRequestLog(tmp_path / "r.sqlite3"))
    monkeypatch.setenv("TRIAGE_LOG_DB", str(tmp_path / "s.sqlite3"))
    monkeypatch.setattr(deps_module, "load_predictor_checked", lambda: pytest.fail("checked the secret first"))
    with pytest.raises(RuntimeError, match="TRIAGE_HASH_SECRET"):
        deps_module.load_deps()
    # The in-memory log works without one (a random secret per process: nothing outlives it anyway)
    memory = deps_module.build_deps(VectorPredictor(), FakeModels(), log=InMemoryRequestLog())
    assert len(memory.hash_secret) == 32
    monkeypatch.setenv("TRIAGE_HASH_SECRET", "configured-secret")
    assert deps_module.build_deps(VectorPredictor(), FakeModels(),
                                  log=SQLiteRequestLog(tmp_path / "t.sqlite3")).hash_secret == b"configured-secret"


def test_schema_v2_has_prompt_hmac_and_no_plain_hash(tmp_path):
    SQLiteRequestLog(tmp_path / "r.sqlite3").close()
    db = sqlite3.connect(tmp_path / "r.sqlite3")
    cols = {c[1] for c in db.execute("PRAGMA table_info(requests)")}
    assert SCHEMA_VERSION == 2 and "prompt_hmac" in cols and "prompt_sha256" not in cols


# ---------- stats and lookups don't read the vectors ----------

def test_stats_and_request_lookup_never_select_the_vector_columns(tmp_path):
    http, deps, log = app_with(tmp_path)
    rid = post(http).headers["x-triage-request-id"]
    assert log.get(rid).embedding is not None                              # they are stored
    statements = []
    log._db.set_trace_callback(statements.append)
    assert http.get("/v1/stats", headers=auth()).status_code == 200
    assert http.get("/v1/stats", headers=auth(ADMIN_KEY)).status_code == 200
    assert http.get(f"/v1/requests/{rid}", headers=auth()).status_code == 200
    log._db.set_trace_callback(None)
    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 3
    for s in selects:
        columns = {c.strip() for c in s.split("FROM")[0].replace("SELECT", "", 1).split(",")}
        assert not {"embedding", "features"} & columns, s
        assert {"feature_version", "embedding_model", "price_as_cost"} <= columns


def test_reads_without_vectors_still_return_everything_else(tmp_path):
    for log in (SQLiteRequestLog(tmp_path / "r.sqlite3"), InMemoryRequestLog()):
        full = row(features=[1.0] * 14, embedding=embedding_to_bytes(np.ones(384)), feature_version="3",
                   embedding_model=MODEL_NAME, price_as_cost=0.5)
        log.write(full)
        light = log.get("r0", include_vectors=False)
        assert (light.features, light.embedding) == (None, None)
        assert light.price_as_cost == 0.5 and light.feature_version == "3"
        assert log.rows(include_vectors=False)[0] == light and log.get("r0") == full   # nothing was lost
