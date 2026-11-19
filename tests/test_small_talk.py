"""Small talk: a pure greeting or pleasantry goes to the low tier without the classifier (routing.yaml small_talk).
Fake models and a fake predictor; no network."""
import copy

import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.background import eligible as label_eligible
from app.api.fake_models import FakeBehavior
from app.api.main import create_app
from app.api.request_log import is_training_eligible
from app.api.stats import summarize
from app.registry import load_registry
from app.reliability import load_routing, routing_from_dict
from app.router.small_talk import is_small_talk, normalize
from tests.api_helpers import API_KEY, FakePredictor, make_deps, routing_raw

REGISTRY = load_registry()
PHRASES = ["hi", "hey", "hello", "hiya", "yo", "good morning", "good afternoon", "good evening", "thanks",
           "thank you", "thx", "ty", "ok", "okay", "cool", "great", "nice", "bye", "goodbye"]
HIGH = FakePredictor(0.1, 0.2)          # the classifier would pick high for anything


def variations(phrase):
    return [phrase, phrase.upper(), phrase.title(), f"  {phrase}  ", f"{phrase}!", f"{phrase}!!!", f"{phrase}.",
            f"{phrase}?", f"{phrase} :)", f"{phrase} 👋", f"{phrase}🙏🏽", f"{phrase}...", f"{phrase}\n",
            phrase.replace(" ", "   ")]


def app(script=None, predictor=None, raw=None, **kw):
    predictor = predictor or FakePredictor(0.1, 0.2)
    deps = make_deps(script, predictor=predictor, raw=raw, **kw)
    return TestClient(create_app(deps=deps, auth=Auth([API_KEY]))), deps, predictor


def chat(http, text=None, *, messages=None, model="auto", headers=None, **body):
    messages = messages or [{"role": "user", "content": text}]
    return http.post("/v1/chat/completions", json={"model": model, "messages": messages, **body},
                     headers={"Authorization": f"Bearer {API_KEY}"} | (headers or {}))


# ---------- matching ----------

def test_the_configured_list_is_the_one_asked_for():
    assert load_routing(REGISTRY).small_talk_phrases == frozenset(PHRASES)


@pytest.mark.parametrize("phrase", PHRASES)
def test_every_phrase_matches_with_case_space_punctuation_and_emoji_variations(phrase):
    for text in variations(phrase):
        assert is_small_talk([{"role": "user", "content": text}], frozenset(PHRASES)), repr(text)


@pytest.mark.parametrize("text", [
    "hi, can you help with X", "hi can you help", "hello world", "hey there", "thanks for the help",
    "thank you so much for explaining recursion", "ok so what is 2 + 2?", "good morning! what's the weather?",
    "great, now write a poem", "bye-bye", "nice work, one more question", "yo what's up", "hi hi", "h i",
    "hii", "okk", "", "   ", "👋", "Write a haiku about saying hello",
])
def test_messages_that_merely_contain_a_greeting_dont_match(text):
    assert not is_small_talk([{"role": "user", "content": text}], frozenset(PHRASES))


def test_only_the_latest_user_message_counts():
    after_work = [{"role": "user", "content": "Explain how a B-tree splits a node."},
                  {"role": "assistant", "content": "A long explanation."}, {"role": "user", "content": "Thanks!"}]
    after_hi = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "Explain how a B-tree splits a node."}]
    assert is_small_talk(after_work, frozenset(PHRASES))
    assert not is_small_talk(after_hi, frozenset(PHRASES))


def test_normalize():
    assert normalize("  Good   MORNING!! 👋🏽 ") == "good morning"
    assert normalize("ok.") == "ok" and normalize("...") == ""


# ---------- routing through the API ----------

@pytest.mark.parametrize("phrase", PHRASES)
def test_each_phrase_routes_low_without_the_classifier(phrase):
    http, deps, predictor = app()
    for text in (phrase, f"{phrase.title()}!!", f"{phrase.upper()} 🙂"):
        r = chat(http, text)
        assert r.status_code == 200, text
        assert (r.headers["x-triage-tier"], r.headers["x-triage-routed-by"]) == ("low", "small_talk"), text
        assert r.headers["x-triage-model"] == "groq-allam"
    assert predictor.calls == []                       # never asked


def test_a_message_that_merely_contains_a_greeting_goes_through_the_classifier():
    http, deps, predictor = app()
    r = chat(http, "hi, can you help with X")
    assert (r.headers["x-triage-tier"], r.headers["x-triage-routed-by"]) == ("high", "classifier")
    assert len(predictor.calls) == 1


def test_the_response_says_small_talk_and_has_no_probabilities():
    http, _, _ = app()
    r = chat(http, "Thanks!")
    triage = r.json()["triage"]
    assert triage["routed_by"] == "small_talk" and triage["tier"] == "low"
    assert (triage["p_low"], triage["p_medium"], triage["tau"]) == (None, None, None)
    assert not {"x-triage-p-low", "x-triage-p-medium", "x-triage-tau"} & set(r.headers)
    assert r.headers["x-triage-reason"].startswith("small talk")
    assert chat(http, "Explain B-trees").json()["triage"]["routed_by"] == "classifier"


def test_the_quality_dial_doesnt_change_small_talk():
    http, _, _ = app()
    assert chat(http, "hello", headers={"X-Triage-Quality": "0.99"}).headers["x-triage-tier"] == "low"


@pytest.mark.parametrize("stream", [False, True])
def test_named_models_are_unaffected(stream):
    http, deps, predictor = app()
    r = chat(http, "hi", model="groq-small", stream=stream)
    assert r.status_code == 200 and r.headers["x-triage-model"] == "groq-small"
    assert r.headers["x-triage-tier"] == "medium" and r.headers["x-triage-routed-by"] == "requested_model"
    (row,) = deps.log.rows()
    assert row.routed_by == "requested_model"


def test_streaming_small_talk_routes_low_too():
    http, deps, predictor = app()
    r = chat(http, "hey!", stream=True)
    assert r.status_code == 200 and r.headers["x-triage-routed-by"] == "small_talk"
    assert r.headers["x-triage-tier"] == "low" and predictor.calls == []


def test_the_low_chain_falls_back_as_usual():
    http, deps, _ = app({"groq-allam": FakeBehavior(fail_before="server")})
    r = chat(http, "hello")
    assert r.status_code == 200 and r.headers["x-triage-model"] == "ollama-llama"
    assert r.headers["x-triage-fallback"] == "groq-allam->ollama-llama"


def test_when_the_whole_low_chain_fails_the_usual_error_comes_back_and_the_log_says_small_talk():
    http, deps, _ = app({n: FakeBehavior(fail_before="server") for n in REGISTRY})
    r = chat(http, "hello")
    assert r.status_code == 503 and r.json()["error"]["code"] == "all_models_failed"
    (row,) = deps.log.rows()
    assert row.routed_by == "small_talk" and row.tier == "low"


def test_byok_small_talk_uses_the_users_low_model():
    http, _, _ = app()
    r = chat(http, "hi", headers={"x-gemini-key": "sk-test-SECRET"})
    assert (r.headers["x-triage-model"], r.headers["x-triage-routed-by"]) == ("gemini-flash-lite", "small_talk")


def test_answer_checking_is_unchanged_for_small_talk():
    http, deps, _ = app()
    r = chat(http, "thanks", headers={"X-Triage-Verify": "sync"})
    assert r.headers["x-triage-verified"] == "sync" and r.headers["x-triage-routed-by"] == "small_talk"
    (row,) = deps.log.rows()
    assert row.verified == "sync" and row.decided_by not in (None, "small_talk")    # the check's own verdict


# ---------- logging and exclusion ----------

def test_the_log_row_says_small_talk_with_no_classifier_fields():
    http, deps, _ = app()
    chat(http, "Good morning!")
    (row,) = deps.log.rows()
    assert row.routed_by == "small_talk" and row.tier == "low"
    assert (row.tau, row.p_low, row.p_medium, row.classifier, row.task_type) == (None, None, None, None, None)
    assert row.features is None and row.embedding is None


def test_small_talk_is_never_training_data_but_the_same_request_otherwise_would_be():
    http, deps, _ = app()
    chat(http, "Explain how a B-tree splits a node.")
    chat(http, "thanks")
    normal, small = deps.log.rows()
    assert normal.training_eligible is True and is_training_eligible(normal)
    assert small.training_eligible is False and not is_training_eligible(small)
    # identical but for routed_by: only that excludes it
    import dataclasses
    assert is_training_eligible(dataclasses.replace(small, routed_by="classifier"))


def test_small_talk_is_never_sampled_for_background_labels():
    http, deps, _ = app(predictor=FakePredictor(0.4, 0.95))      # medium: labels skip high answers
    chat(http, "Explain how a B-tree splits a node.")
    chat(http, "bye")
    normal, small = deps.log.rows()
    assert label_eligible(normal) and not label_eligible(small)


def test_stats_leave_small_talk_out_and_count_it_separately():
    http, deps, _ = app()
    chat(http, "Explain how a B-tree splits a node.")
    chat(http, "hi")
    chat(http, "thank you")
    stats = summarize(deps.log.rows())
    assert stats["requests"] == 1 and stats["small_talk_requests"] == 2
    assert stats["tier_mix"] == {"high": 1.0}
    r = http.get("/v1/stats", headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.json()["small_talk_requests"] == 2 and r.json()["requests"] == 1


def test_the_json_log_line_has_routed_by(caplog):
    import json
    import logging
    http, _, _ = app()
    with caplog.at_level(logging.INFO, logger="triage.requests"):
        chat(http, "yo")
    (line,) = [json.loads(r.getMessage()) for r in caplog.records if r.name == "triage.requests"]
    assert line["routed_by"] == "small_talk"


# ---------- config ----------

def test_without_the_section_there_is_no_small_talk():
    raw = copy.deepcopy(routing_raw())
    raw.pop("small_talk", None)
    http, _, predictor = app(raw=raw)
    r = chat(http, "hi")
    assert r.headers["x-triage-routed-by"] == "classifier" and len(predictor.calls) == 1


def test_the_list_can_be_edited():
    raw = copy.deepcopy(routing_raw())
    raw["small_talk"] = {"phrases": ["Howdy!", "cheers"]}
    assert routing_from_dict(raw, REGISTRY).small_talk_phrases == frozenset({"howdy", "cheers"})
    http, _, _ = app(raw=raw)
    assert chat(http, "Cheers!").headers["x-triage-routed-by"] == "small_talk"
    assert chat(http, "hi").headers["x-triage-routed-by"] == "classifier"


@pytest.mark.parametrize("section, message", [
    ({"phrases": "hi"}, "list of strings"), ({"phrases": [1]}, "list of strings"),
    ({"phrases": ["!!"]}, "empty once normalized"), ({"words": ["hi"]}, "one key"), (["hi"], "one key"),
])
def test_bad_small_talk_config_is_refused(section, message):
    raw = copy.deepcopy(routing_raw())
    raw["small_talk"] = section
    with pytest.raises(ValueError, match=message):
        routing_from_dict(raw, REGISTRY)
