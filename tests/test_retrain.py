"""scripts/retrain.py (Phase 4 Step 9): data gathering, leak guard, candidate files, decision, routing.yaml.

No API calls. Fake labels in a temporary request-log database; routing.yaml, models and history in tmp_path.
The decision rule is swapped for a reference copy where needed (the real one is yours: test_retrain_decision.py).
"""
import json
import shutil

import numpy as np
import pytest

from app.api.request_log import LabelRow, RequestRow, SQLiteRequestLog, embedding_to_bytes, prompt_hmac
from app.router.embeddings import MODEL_NAME
from app.router.features import FEATURE_NAMES, FEATURE_VERSION
from app.router.predictor import Prediction, load_classifier_config
from app.router.quality import ROUTING_CONFIG_PATH
from eval.splits import load_split, load_validation_split
from scripts import retrain
from scripts import retrain_decision

SECRET = b"retrain-test-secret"
WIDTH = len(FEATURE_NAMES) + 384
NAME = "classifier_c_logreg_cv"


def reference_should_replace(current, candidate, *, min_cost_gain=0.01, quality_tolerance=0.0):
    q0, q1, c0, c1 = current["quality_retained"], candidate["quality_retained"], current["cost_saved"], candidate["cost_saved"]
    if q1 >= q0 - quality_tolerance and c1 - c0 >= min_cost_gain:
        return True, "replace: rule a"
    if q1 > q0 and c1 >= c0:
        return True, "replace: rule b"
    return False, "keep: neither rule"


# ---------- fakes ----------

def synthetic_original(n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for tier in retrain.TIERS:
        X = rng.normal(size=(n, WIDTH))
        y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 0).astype(int)
        out[tier] = {"X": X, "y": y, "ids": [f"orig_{tier}_{i}" for i in range(n)]}
    return out


def live_row(i, *, tier="medium", eligible=True, feature_version=FEATURE_VERSION, embedding_model=MODEL_NAME,
             hmac="h", vectors=True, guessed=True, seed=0):
    rng = np.random.default_rng(seed + i)
    return RequestRow(request_id=f"req{i}", timestamp="2026-11-10T00:00:00+00:00", key_id="abcd1234",
                      requested_model="auto", stream=False, status="ok", tier=tier, served_model="groq-small",
                      fell_back=False, settings_match_training=True, prompt_hmac=f"{hmac}{i}",
                      task_type_guessed=guessed, training_eligible=eligible,
                      features=[float(x) for x in rng.normal(size=14)] if vectors else None,
                      embedding=embedding_to_bytes(rng.normal(size=384).astype(np.float32)) if vectors else None,
                      feature_version=feature_version, embedding_model=embedding_model)


def write_db(path, rows, passed=None, labels=None):
    log = SQLiteRequestLog(path)
    for i, row in enumerate(rows):
        log.write(row)
        log.write_label(labels[i] if labels else LabelRow(
            request_id=row.request_id, created_at="2026-11-10T01:00:00+00:00", status="labeled",
            passed=True if passed is None else passed[i], judge_score=5, served_tier=row.tier, calls=2, tokens=100))
    log.close()
    return path


def metrics(quality, cost):
    return {"presets": {p: {"tau": 0.5, "quality_retained": quality, "cost_saved": cost, "mix": {}}
                        for p in ("cheap", "balanced", "high")},
            "tiers": {"low": {"log_loss": 0.5}, "medium": {"log_loss": 0.4}}}


def rigged(current, candidate):
    """evaluate_fn returning fixed metrics: the live version gets `current`, anything newer `candidate`."""
    def evaluate(name, version, *, models_dir):
        return {"version": version, **(current if version == 2 else candidate)}
    return evaluate


@pytest.fixture
def env(tmp_path):
    routing = tmp_path / "routing.yaml"
    shutil.copy(ROUTING_CONFIG_PATH, routing)
    models = tmp_path / "models"
    models.mkdir()
    return {"routing": routing, "models": models, "history": tmp_path / "retrain_history.jsonl",
            "db": tmp_path / "requests.sqlite3", "tmp": tmp_path}


def run(env, **kw):
    kw.setdefault("load_original_fn", synthetic_original)
    kw.setdefault("decide_fn", reference_should_replace)
    kw.setdefault("secret", SECRET)
    kw.setdefault("prompts", [])
    return retrain.run(log_db=env["db"], models_dir=env["models"], routing_path=env["routing"],
                       history_path=env["history"], say=lambda m: None, **kw)


def history(env):
    return [json.loads(l) for l in env["history"].read_text().splitlines()] if env["history"].exists() else []


# ---------- zero new labels ----------

@pytest.mark.parametrize("db", ["missing", "empty", "only ineligible"])
def test_zero_new_labels_keeps_without_training(env, db):
    if db == "empty":
        SQLiteRequestLog(env["db"]).close()
    elif db == "only ineligible":
        write_db(env["db"], [live_row(0, eligible=False)])
    before = env["routing"].read_text()
    trained = []
    entry = run(env, train_fn=lambda *a, **k: trained.append(1))
    assert trained == [] and entry["trained"] is False
    assert (entry["decision"], entry["reason"]) == ("keep", "nothing new, keeping the current model")
    assert env["routing"].read_text() == before and list(env["models"].iterdir()) == []
    [h] = history(env)
    assert h["decision"] == "keep" and h["candidate_version"] is None and h["data"]["new_labels"] == {"low": 0, "medium": 0}
    if db == "missing":
        assert not env["db"].exists()                                   # never creates a database


# ---------- rigged candidates ----------

def test_a_better_candidate_replaces_the_live_model(env):
    write_db(env["db"], [live_row(i, tier="medium" if i % 2 else "low") for i in range(6)])
    before = env["routing"].read_text()
    entry = run(env, evaluate_fn=rigged(metrics(0.95, 0.30), metrics(0.96, 0.35)))
    assert entry["decision"] == "replace" and entry["candidate_version"] == 1
    assert load_classifier_config(env["routing"]) == (NAME, 1)
    changed = [(a, b) for a, b in zip(before.splitlines(), env["routing"].read_text().splitlines()) if a != b]
    assert changed == [("  version: 2", "  version: 1")]                  # only that line: comments kept
    assert not list(env["tmp"].glob(".routing.yaml.*"))                   # no temp file left behind
    [h] = history(env)
    assert h["decision"] == "replace" and h["reason"] == "replace: rule a"
    assert (h["current_version"], h["candidate_version"]) == (2, 1)
    assert h["metrics"]["current"]["presets"]["balanced"]["cost_saved"] == 0.30
    assert h["metrics"]["candidate"]["presets"]["cheap"]["quality_retained"] == 0.96
    assert h["data"]["new_labels"] == {"low": 3, "medium": 3} and h["data"]["original"] == {"low": 60, "medium": 60}


def test_a_worse_candidate_is_kept_but_its_files_are_saved(env):
    write_db(env["db"], [live_row(i) for i in range(4)])
    before = env["routing"].read_text()
    entry = run(env, evaluate_fn=rigged(metrics(0.95, 0.30), metrics(0.94, 0.33)))
    assert entry["decision"] == "keep" and env["routing"].read_text() == before
    assert sorted(p.name for p in env["models"].iterdir()) == sorted(
        f"{NAME}_{t}_v1{e}" for t in ("low", "medium") for e in (".joblib", ".json"))
    [h] = history(env)
    assert h["decision"] == "keep" and h["reason"] == "keep: neither rule" and h["candidate_version"] == 1


def test_the_balanced_preset_decides(env):
    write_db(env["db"], [live_row(0)])
    seen = []
    current, candidate = metrics(0.95, 0.30), metrics(0.95, 0.30)
    candidate["presets"]["balanced"] = {"quality_retained": 0.99, "cost_saved": 0.5}
    run(env, evaluate_fn=rigged(current, candidate),
        decide_fn=lambda c, k: seen.append((c, k)) or (False, "keep"))
    assert seen == [(current["presets"]["balanced"], candidate["presets"]["balanced"])]


def test_the_default_decision_is_should_replace(env, monkeypatch):
    write_db(env["db"], [live_row(0)])
    calls = []
    monkeypatch.setattr(retrain_decision, "should_replace", lambda c, k: calls.append(1) or (False, "keep: spy"))
    entry = run(env, decide_fn=None, evaluate_fn=rigged(metrics(0.9, 0.3), metrics(0.9, 0.3)))
    assert calls == [1] and entry["reason"] == "keep: spy"


# ---------- --dry-run ----------

@pytest.mark.parametrize("better", [True, False])
def test_dry_run_changes_nothing_but_the_candidate_files(env, better):
    write_db(env["db"], [live_row(i) for i in range(4)])
    before_routing = env["routing"].read_text()
    before_labels = SQLiteRequestLog(env["db"]).labels()
    entry = run(env, dry_run=True, evaluate_fn=rigged(metrics(0.95, 0.30),
                                                      metrics(0.97, 0.40) if better else metrics(0.90, 0.30)))
    assert entry["decision"] == ("replace" if better else "keep") and entry["dry_run"] is True
    assert env["routing"].read_text() == before_routing
    assert not env["history"].exists()
    assert len(list(env["models"].iterdir())) == 4                        # the candidate files only
    assert SQLiteRequestLog(env["db"]).labels() == before_labels           # the log is only read
    others = {p.name for p in env["tmp"].iterdir()} - {"routing.yaml", "models"}
    assert {n for n in others if not n.startswith("requests.sqlite3")} == set()


def test_dry_run_with_nothing_new_writes_nothing(env):
    run(env, dry_run=True)
    assert not env["history"].exists() and list(env["models"].iterdir()) == []


# ---------- which live rows count ----------

def test_only_eligible_current_version_rows_with_vectors_count(env):
    rows = [live_row(0), live_row(1, tier="low"),
            live_row(2, eligible=False),
            live_row(3, feature_version="2"),
            live_row(4, embedding_model="some-other-embedder"),
            live_row(5, vectors=False),
            live_row(6, tier="high")]
    labels = [LabelRow(request_id=r.request_id, created_at="t", status="labeled", passed=True, served_tier=r.tier)
              for r in rows]
    rows.append(live_row(7))
    labels.append(LabelRow(request_id="req7", created_at="t", status="skipped", skip_reason="daily_cap"))
    write_db(env["db"], rows, labels=labels)
    log = SQLiteRequestLog(env["db"])
    new, excluded = retrain.load_new_labels(log, {})
    assert [len(new[t]["y"]) for t in ("low", "medium")] == [1, 1]
    assert excluded == {"not_training_eligible": 1, "wrong_feature_or_embedding_version": 2, "vectors_missing": 1,
                        "not_a_low_or_medium_answer": 1, "not_labeled": 1}


def test_a_label_applies_only_to_the_tier_that_answered(env):
    write_db(env["db"], [live_row(0, tier="low"), live_row(1, tier="low"), live_row(2, tier="medium")],
             passed=[False, True, True])
    new, _ = retrain.load_new_labels(SQLiteRequestLog(env["db"]), {})
    assert new["low"]["y"].tolist() == [0, 1] and new["medium"]["y"].tolist() == [1]
    assert new["low"]["request_ids"] == ["req0", "req1"] and new["medium"]["request_ids"] == ["req2"]


def test_live_rows_are_the_stored_vectors(env):
    row = live_row(0)
    write_db(env["db"], [row])
    new, _ = retrain.load_new_labels(SQLiteRequestLog(env["db"]), {})
    x = new["medium"]["X"][0]
    assert x.shape == (WIDTH,) and x[:14].tolist() == row.features
    assert np.array_equal(x[14:].astype(np.float32), np.frombuffer(row.embedding, dtype="<f4"))


def test_validation_and_test_prompts_sent_through_the_api_are_excluded(env):
    from eval.build_dataset import load_prompts
    prompts = load_prompts()
    _, test_ids = load_split()
    val_ids = load_validation_split()
    val_p = next(p for p in prompts if p["id"] in val_ids)
    test_p = next(p for p in prompts if p["id"] in test_ids)
    fit_p = next(p for p in prompts if p["id"] not in val_ids and p["id"] not in test_ids)
    rows = []
    for i, p in enumerate((val_p, test_p, fit_p)):
        r = live_row(i)
        # what the API stores: the HMAC of the messages as its request model keeps them (role + content)
        r.prompt_hmac = prompt_hmac([{"role": m["role"], "content": m["content"]} for m in p["messages"]], SECRET)
        rows.append(r)
    write_db(env["db"], rows)
    forbidden = retrain.forbidden_hmacs(SECRET, prompts)
    assert len(forbidden) == len(val_ids) + len(test_ids)
    new, excluded = retrain.load_new_labels(SQLiteRequestLog(env["db"]), forbidden)
    assert excluded == {"leak_validation": 1, "leak_test": 1}
    assert new["medium"]["request_ids"] == ["req2"]                       # only the fit prompt is kept
    # under a different secret the HMACs don't match, which is why retraining needs the API's secret
    assert not set(retrain.forbidden_hmacs(b"another-secret", prompts)) & {r.prompt_hmac for r in rows}


def test_labels_without_the_secret_refuse_to_run(env, monkeypatch):
    write_db(env["db"], [live_row(0)])
    monkeypatch.delenv("TRIAGE_HASH_SECRET", raising=False)
    with pytest.raises(SystemExit, match="TRIAGE_HASH_SECRET"):
        run(env, secret=None)


def test_data_counts_record_guessed_task_types(env):
    write_db(env["db"], [live_row(0, guessed=True), live_row(1, guessed=False), live_row(2, guessed=True)])
    entry = run(env, dry_run=True, evaluate_fn=rigged(metrics(0.9, 0.3), metrics(0.9, 0.3)))
    d = entry["data"]
    assert d["new_task_types_guessed"] == {"low": 0, "medium": 2} and "hand tags" in d["note"]
    assert d["new_label_pass_rate"]["medium"] == 1.0 and d["new_label_pass_rate"]["low"] is None


# ---------- the original data ----------

def test_the_original_fit_set_is_always_included(env):
    original = retrain.load_original()
    fit = {i for t in original.values() for i in t["ids"]}
    assert not fit & load_validation_split() and not fit & load_split()[1]
    assert len(original["low"]["ids"]) == len(original["medium"]["ids"]) == 481
    assert original["low"]["X"].shape == (481, WIDTH)
    seen = {}

    def spy_train(orig, new, *, name, models_dir):
        seen.update({t: (orig[t]["X"].shape[0], new[t]["X"].shape[0]) for t in retrain.TIERS})
        return {"version": 3, "tiers": {}}
    write_db(env["db"], [live_row(0), live_row(1, tier="low")])
    run(env, load_original_fn=lambda: original, train_fn=spy_train, dry_run=True,
        evaluate_fn=rigged(metrics(0.9, 0.3), metrics(0.9, 0.3)))
    assert seen == {"low": (481, 1), "medium": (481, 1)}


def test_the_original_data_is_built_exactly_like_train_classifier():
    from eval import train_classifier as tc
    original = retrain.load_original()
    rows = tc.load_training_rows()
    keep = [o for o in rows if o.get("medium_pass") is not None]
    X, _, _ = tc.feature_matrix("both", [o["id"] for o in keep], "hand")
    assert np.array_equal(original["medium"]["X"], X)
    assert original["medium"]["y"].tolist() == [int(bool(o["medium_pass"])) for o in keep]


# ---------- the test set is never loaded ----------

def test_the_test_set_is_never_loaded(env, monkeypatch):
    import eval.routing_sim as routing_sim
    calls = []
    real = routing_sim.load_eval_rows

    def guarded(split="train", allow_test=False, allow_val=False):
        calls.append(split)
        if split == "test" or allow_test:
            raise AssertionError("retrain loaded the test split")
        return real(split, allow_test=allow_test, allow_val=allow_val)
    monkeypatch.setattr(routing_sim, "load_eval_rows", guarded)

    class FakePredictor:
        def predict_texts(self, texts, tasks):
            return [Prediction(0.5, 0.9, 0.9, "qa", False) for _ in texts]
    result = retrain.evaluate_on_validation(NAME, 2, make_predictor=lambda *a: FakePredictor())
    assert calls == ["val"]
    assert set(result["presets"]) == {"cheap", "balanced", "high"} and set(result["tiers"]) == {"low", "medium"}
    # training data: the fit set, with no test (or validation) IDs
    original = retrain.load_original()
    _, test_ids = load_split()
    assert not {i for t in original.values() for i in t["ids"]} & test_ids


# ---------- candidate files ----------

def test_the_candidate_is_trained_like_the_live_model_and_never_overwrites(env):
    original, new = synthetic_original(40), {t: {"X": np.random.default_rng(1).normal(size=(3, WIDTH)),
                                                 "y": np.array([1, 0, 1]), "request_ids": ["a", "b", "c"],
                                                 "task_type_guessed": [True] * 3} for t in retrain.TIERS}
    (env["models"] / f"{NAME}_low_v4.json").write_text("{}")             # an existing file must not be touched
    out = retrain.train_candidate(original, new, name=NAME, models_dir=env["models"])
    assert out["version"] == 5
    assert (env["models"] / f"{NAME}_low_v4.json").read_text() == "{}"
    meta = json.loads((env["models"] / f"{NAME}_medium_v5.json").read_text())
    assert meta["version"] == 5 and meta["feature_set"] == "both" and meta["model"] == "logreg_cv"
    assert meta["versions"] == {"features": FEATURE_VERSION, "embeddings": MODEL_NAME}
    assert meta["rows"] == {"original": 40, "new_labels": 3}
    import joblib
    model = joblib.load(env["models"] / f"{NAME}_medium_v5.joblib")
    assert model.n_features_in_ == WIDTH
    # loads in the live Predictor, which runs its own version and column checks
    from app.router.predictor import Predictor
    Predictor(NAME, 5, models_dir=env["models"])
    # a second run takes the next number
    assert retrain.train_candidate(original, new, name=NAME, models_dir=env["models"])["version"] == 6


def test_an_existing_target_file_stops_training(env, monkeypatch):
    monkeypatch.setattr(retrain, "next_free_version", lambda name, d: 7)
    (env["models"] / f"{NAME}_medium_v7.joblib").write_bytes(b"precious")
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        retrain.train_candidate(synthetic_original(30), {t: {"X": np.zeros((0, WIDTH)), "y": np.zeros(0, int),
                                                             "request_ids": [], "task_type_guessed": []}
                                                         for t in retrain.TIERS}, name=NAME, models_dir=env["models"])
    assert (env["models"] / f"{NAME}_medium_v7.joblib").read_bytes() == b"precious"


# ---------- routing.yaml ----------

def test_set_classifier_version_changes_one_line_only(env):
    before = env["routing"].read_text()
    retrain.set_classifier_version(env["routing"], 9)
    after = env["routing"].read_text()
    assert load_classifier_config(env["routing"]) == (NAME, 9)
    diff = [(a, b) for a, b in zip(before.splitlines(), after.splitlines()) if a != b]
    assert diff == [("  version: 2", "  version: 9")] and len(before.splitlines()) == len(after.splitlines())
    assert "# The trained classifier the live router uses" in after          # comments kept
    from app.api.config_state import ActiveConfig
    from app.registry import load_registry
    ActiveConfig.from_file(load_registry(), env["routing"])                 # still a valid routing config
