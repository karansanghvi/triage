"""Verification in the live API (Phase 4 Step 7): sync mode and background labels. Fake models only.

Uses the real route_and_stream. Fake models tell AI-checker and judge calls apart from answers by their
system prompt (app/api/fake_models.py), so a test can make the checker say "bad" without touching answers.
"""
import copy
import random
from datetime import datetime, timedelta, timezone

import openai
import pytest
from fastapi.testclient import TestClient

from app.api import background as background_module
from app.api.auth import Auth
from app.api.background import BackgroundLabeler
from app.api.fake_models import FakeBehavior, fake_text
from app.api.main import create_app
from app.api.request_log import InMemoryRequestLog, SQLiteRequestLog
from app.api.verification import DEFAULT_SECTION, VerificationConfigError, verification_from_dict
from app.cache import ResponseCache
from app.registry import load_registry
from tests.api_helpers import ADMIN_KEY, API_KEY, FakePredictor, make_deps, routing_raw

AUTH = Auth([API_KEY], [ADMIN_KEY])
RAIN = [{"role": "user", "content": "Write two sentences about rain."}]
ASK = [{"role": "user", "content": "What is 2 + 3? End with: Answer: <number>"}]
BAD_CHECK = '{"good": false, "reason": "the answer is wrong"}'
MEDIUM, LOW, HIGH = FakePredictor(0.4, 0.95), FakePredictor(0.95, 0.97), FakePredictor(0.1, 0.2)   # tau 0.86


def app_with(script=None, *, predictor=MEDIUM, on_error=None, background=None, sample_rate=0.0, labeler=None,
             log=None, cache=None, skip_check_above=None):
    raw = routing_raw()
    raw["verification"] = copy.deepcopy(DEFAULT_SECTION)
    # These tests are about the check itself, so medium always checks (None) unless a test says otherwise; the
    # shipped default (0.90) would skip it for MEDIUM's confident p_medium 0.95
    raw["verification"]["sync"]["tiers"]["medium"]["skip_check_above"] = skip_check_above
    if on_error:
        raw["verification"]["sync"]["on_error"] = on_error
    if background:
        raw["verification"]["background"].update(background)
    deps = make_deps(script, predictor=predictor, raw=raw, sample_rate=sample_rate, labeler=labeler, log=log,
                     cache=cache)
    return TestClient(create_app(deps=deps, auth=AUTH)), deps


def post(http, messages=RAIN, verify="sync", model="auto", **body):
    headers = {"Authorization": f"Bearer {API_KEY}"}
    if verify:
        headers["X-Triage-Verify"] = verify
    return http.post("/v1/chat/completions", json={"model": model, "messages": messages, **body}, headers=headers)


def calls(deps, kind=None, model=None):
    return [c for c in deps.fake_models.calls if (kind is None or c["kind"] == kind)
            and (model is None or c["model"] == model)]


def last_row(deps):
    return deps.log.rows()[-1]


# ---------- THE DONE-WHEN TEST ----------

def test_done_when_a_bad_medium_answer_is_caught_and_the_high_answer_is_served():
    http, deps = app_with({"groq-small": FakeBehavior(check_reply=BAD_CHECK)})
    client = openai.OpenAI(base_url="http://testserver/v1", api_key=API_KEY, http_client=http, max_retries=0,
                           default_headers={"X-Triage-Verify": "sync"})
    raw = client.chat.completions.with_raw_response.create(model="auto", messages=RAIN)
    completion = raw.parse()
    assert completion.choices[0].message.content == fake_text("groq-large")       # the high model's answer
    assert completion.model == "groq-large"
    h = raw.headers
    assert (h["X-Triage-Escalated"], h["X-Triage-Verified"], h["X-Triage-Check"]) == ("true", "sync", "ai_checker")
    assert (h["X-Triage-Model"], h["X-Triage-Tier"]) == ("groq-large", "high")
    assert "medium->high" in h["X-Triage-Reason"]
    r = last_row(deps)
    assert (r.escalated, r.escalated_from, r.escalated_to) == (True, "medium", "high")
    assert (r.decided_by, r.checker_verdict, r.checker_version) == ("ai_checker", "bad", 3)
    assert r.served_model == "groq-large" and r.tier == "high" and r.training_eligible is False
    # the answer calls were medium then high; the checker ran once, on medium's answer
    assert [c["model"] for c in calls(deps, "answer")] == ["groq-small", "groq-large"]
    assert [c["model"] for c in calls(deps, "check")] == ["groq-small"]


# ---------- sync mode: medium ----------

def test_a_good_medium_answer_stays_at_medium():
    http, deps = app_with()
    r = post(http)
    assert r.json()["choices"][0]["message"]["content"] == fake_text("groq-small")
    h = r.headers
    assert (h["x-triage-escalated"], h["x-triage-check"], h["x-triage-tier"]) == ("false", "ai_checker", "medium")
    row = last_row(deps)
    assert (row.escalated, row.decided_by, row.checker_verdict, row.escalated_to) == (False, "ai_checker", "good", None)
    assert row.training_eligible is True                                           # not escalated: still eligible
    assert calls(deps, model="groq-large") == []


def test_a_format_failure_escalates_without_calling_the_ai_checker():
    http, deps = app_with({"groq-small": FakeBehavior(text="five")})                # no "Answer:" line
    r = post(http, messages=ASK)
    assert r.headers["x-triage-escalated"] == "true" and r.headers["x-triage-check"] == "format"
    assert r.json()["model"] == "groq-large"
    assert calls(deps, "check") == []                                              # the free check decided
    row = last_row(deps)
    assert (row.decided_by, row.checker_verdict, row.check_cost, row.checker_version) == ("format", "bad", 0.0, None)


@pytest.mark.parametrize("failure", [FakeBehavior(check_reply="not json at all"), FakeBehavior(check_fail="server")])
def test_a_checker_error_escalates_by_default(failure):
    http, deps = app_with({"groq-small": failure})
    r = post(http)
    assert r.headers["x-triage-check"] == "ai_checker_error" and r.headers["x-triage-escalated"] == "true"
    assert r.json()["model"] == "groq-large"                     # not gemini-flash: the groq provider is fine
    assert last_row(deps).checker_verdict == "error"


def test_a_tripped_groq_small_breaker_still_escalates_to_groq_large():
    """The checker's server errors (retried 3 times) trip groq-small's own breaker. The escalation to high
    must still reach groq-large, which is on the same provider."""
    http, deps = app_with({"groq-small": FakeBehavior(check_fail="server")})
    r = post(http)
    assert deps.client.breaker.is_open("model:groq-small") and not deps.client.breaker.is_open("groq")
    assert r.json()["model"] == "groq-large" and r.headers["x-triage-escalated"] == "true"
    # the next request: groq-small is skipped for its own answer too, so the medium tier falls back to
    # gemini-flash; high still goes straight to groq-large
    assert post(http, verify=None).json()["model"] == "gemini-flash"
    high = make_deps(predictor=HIGH)
    high.client = deps.client                                     # the same breaker state
    http_high = TestClient(create_app(deps=high, auth=AUTH))
    assert post(http_high, verify=None).json()["model"] == "groq-large"


@pytest.mark.parametrize("failure", [FakeBehavior(check_reply="not json at all"), FakeBehavior(check_fail="timeout")])
def test_a_checker_error_is_accepted_when_on_error_is_accept(failure):
    http, deps = app_with({"groq-small": failure}, on_error="accept")
    r = post(http)
    assert r.headers["x-triage-check"] == "ai_checker_error" and r.headers["x-triage-escalated"] == "false"
    assert r.json()["model"] == "groq-small"
    row = last_row(deps)
    assert row.decided_by == "ai_checker_error" and row.escalated is False


# ---------- sync mode: low and high ----------

def test_a_low_answer_gets_only_the_free_format_check():
    http, deps = app_with(predictor=LOW)
    r = post(http)
    assert r.headers["x-triage-tier"] == "low" and r.headers["x-triage-check"] == "format"
    assert r.headers["x-triage-escalated"] == "false"
    assert calls(deps, "check") == []                                              # never the AI checker on low
    assert last_row(deps).decided_by == "format"


def test_a_bad_low_answer_escalates_to_medium_which_gets_the_medium_checks():
    # medium's own fake answer has no "Answer:" line either, so medium's format check sends it on to high
    http, deps = app_with({"groq-allam": FakeBehavior(text="five")}, predictor=LOW)
    r = post(http, messages=ASK)
    assert r.json()["model"] == "groq-large" and r.headers["x-triage-check"] == "format"
    row = last_row(deps)
    assert (row.escalated_from, row.escalated_to, row.decided_by) == ("low", "high", "format")
    assert calls(deps, "check") == []
    # with a well-formatted medium answer, medium's checks pass and it stops at medium
    http, deps = app_with({"groq-allam": FakeBehavior(text="five"),
                           "groq-small": FakeBehavior(text="2 + 3 = 5.\nAnswer: 5")}, predictor=LOW)
    r = post(http, messages=ASK)
    assert r.json()["model"] == "groq-small" and r.headers["x-triage-escalated"] == "true"
    assert [c["model"] for c in calls(deps, "check")] == ["groq-small"]


def test_low_to_medium_to_high():
    http, deps = app_with({"groq-allam": FakeBehavior(text="five"),
                           "groq-small": FakeBehavior(text="2 + 3 = 5.\nAnswer: 5", check_reply=BAD_CHECK)},
                          predictor=LOW)
    r = post(http, messages=ASK)
    assert r.json()["model"] == "groq-large"
    row = last_row(deps)
    assert (row.escalated_from, row.escalated_to, row.decided_by) == ("low", "high", "ai_checker")
    assert [c["model"] for c in calls(deps, "answer")] == ["groq-allam", "groq-small", "groq-large"]


def test_a_high_answer_is_never_checked():
    http, deps = app_with(predictor=HIGH)
    r = post(http)
    assert r.headers["x-triage-tier"] == "high" and r.headers["x-triage-check"] == "none"
    assert calls(deps, "check") == [] and last_row(deps).decided_by == "none"


# ---------- sync mode: what it refuses ----------

def test_pass_through_with_sync_is_a_400():
    http, deps = app_with()
    r = post(http, model="groq-small")
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["param"] == "X-Triage-Verify" and err["code"] == "unsupported_combination" and '"auto"' in err["message"]
    assert deps.fake_models.calls == []


def test_sync_with_streaming_is_still_a_400():
    http, _ = app_with()
    assert post(http, stream=True).status_code == 400


# ---------- costs, latency and log fields ----------

def test_cost_header_and_log_include_the_check_and_the_escalation():
    http, deps = app_with({"groq-small": FakeBehavior(check_reply=BAD_CHECK)})
    r = post(http)
    row = last_row(deps)
    medium_cost, high_cost = (deps.log.get(post(http, verify=None, model=m).headers["x-triage-request-id"])
                              .price_as_cost for m in ("groq-small", "groq-large"))   # the same answers, alone
    assert row.check_cost > 0 and row.check_latency_ms >= 0
    assert row.escalation_extra_cost == pytest.approx(high_cost)
    assert row.escalation_extra_latency_ms >= 0
    assert row.price_as_cost == pytest.approx(medium_cost + row.check_cost + high_cost)
    # the headers carry 8 decimal places of dollars
    assert float(r.headers["x-triage-cost"]) == pytest.approx(row.price_as_cost, abs=1e-8)
    assert float(r.headers["x-triage-saved"]) == pytest.approx(row.strong_cost - row.price_as_cost, abs=1e-8)
    assert row.saved < 0                          # medium + check + high costs more than high alone: said honestly
    assert (row.verified, row.mode) == ("sync", "api")


def test_a_kept_answer_still_pays_for_its_check():
    http, deps = app_with()
    post(http)
    row = last_row(deps)
    alone = deps.log.get(post(http, verify=None, model="groq-small").headers["x-triage-request-id"]).price_as_cost
    assert row.price_as_cost == pytest.approx(alone + row.check_cost) and row.check_cost > 0
    assert row.escalation_extra_cost == 0


def test_training_eligible_is_false_after_escalation():
    http, deps = app_with({"groq-small": FakeBehavior(check_reply=BAD_CHECK)})
    post(http)
    assert last_row(deps).escalated is True and last_row(deps).training_eligible is False


def test_checker_calls_use_the_reliable_client_timeout_and_cache(tmp_path):
    http, deps = app_with(cache=ResponseCache(tmp_path / "cache.sqlite3"))
    post(http)
    post(http)
    checks = calls(deps, "check")
    assert len(checks) == 1                                                        # the second check was cached
    routing = deps.config.snapshot().routing
    assert checks[0]["timeout"] == routing.timeout_for("groq-small")
    assert checks[0]["reasoning_effort"] == "low"
    assert last_row(deps).check_cost == 0                                          # a cached check costs nothing


def test_if_escalation_finds_no_model_the_last_answer_is_served():
    http, deps = app_with({"groq-small": FakeBehavior(check_reply=BAD_CHECK),
                           "groq-large": FakeBehavior(fail_before="server"),
                           "gemini-flash": FakeBehavior(fail_before="server")})
    r = post(http)
    assert r.status_code == 200 and r.json()["model"] == "groq-small"
    assert r.headers["x-triage-escalated"] == "false"
    row = last_row(deps)
    assert row.error_kind == "all_models_failed:high" and row.status == "ok" and row.checker_verdict == "bad"


# ---------- the routing.yaml section ----------

def test_routing_yaml_verification_section_is_the_default_policy():
    from app.api.config_state import ActiveConfig
    policy = ActiveConfig.from_file(load_registry()).snapshot().verification
    assert policy == verification_from_dict({"verification": DEFAULT_SECTION}) == verification_from_dict({})
    assert policy.sync.tiers["low"].checks == ("format",) and policy.sync.tiers["high"].checks == ()
    assert policy.sync.tiers["medium"].checks == ("format", "ai_checker") and policy.sync.on_error == "escalate"
    assert policy.background.sample_rate == 0.1


@pytest.mark.parametrize("change", [
    lambda v: v["sync"].update(agreement_on_comparable=True),
    lambda v: v["sync"].update(on_error="retry"),
    lambda v: v["sync"]["tiers"]["medium"].update(escalate_to="low"),
    lambda v: v["sync"]["tiers"]["medium"].update(checks=["format", "vibes"]),
    lambda v: v["sync"]["tiers"]["high"].update(checks=["format"]),           # checks with nowhere to escalate
    lambda v: v["sync"]["tiers"].pop("low"),
    lambda v: v["background"].update(sample_rate=1.5),
    lambda v: v["background"].update(daily_max_calls=-1),
    lambda v: v.update(extra=1),
])
def test_bad_verification_config_is_rejected(change):
    section = copy.deepcopy(DEFAULT_SECTION)
    change(section)
    with pytest.raises(VerificationConfigError):
        verification_from_dict({"verification": section})


# ---------- background labels ----------

class CountingRandom(random.Random):
    def __init__(self, seed=0):
        super().__init__(seed)
        self.draws = 0

    def random(self):
        self.draws += 1
        return super().random()


def bg_app(script=None, *, rate=1.0, rng=None, background=None, log=None, predictor=MEDIUM, clock=None, cache=None):
    labeler = BackgroundLabeler(rng or CountingRandom(), **({"clock": clock} if clock else {}))
    return app_with(script, predictor=predictor, sample_rate=rate, labeler=labeler, background=background, log=log,
                    cache=cache)


def test_a_sampled_request_gets_a_label_row():
    http, deps = bg_app()
    r = post(http, verify=None)
    labels = deps.log.labels(r.headers["x-triage-request-id"])
    assert len(labels) == 1
    label = labels[0]
    assert (label.status, label.passed, label.judge_score) == ("labeled", True, 5)
    assert (label.judge_version, label.judge_model, label.reference_model) == (3, "groq-large", "groq-large")
    assert (label.served_model, label.served_tier, label.calls) == ("groq-small", "medium", 2)
    assert label.tokens > 0 and label.cost > 0 and label.created_at
    strong = [c for c in calls(deps, "answer") if c["model"] == "groq-large"][0]
    assert (strong["temperature"], strong["max_tokens"]) == (0.0, 4000)            # Phase 2 settings
    judge_call = calls(deps, "judge")[0]
    assert judge_call["timeout"] == deps.judge_settings.timeout_seconds


def test_a_bad_grade_is_a_failed_label():
    http, deps = bg_app({"groq-large": FakeBehavior(judge_reply='{"constraints": [], "score": 2, "reason": "worse"}')})
    post(http, verify=None)
    label = deps.log.labels()[0]
    assert (label.passed, label.judge_score) == (False, 2)


def test_sampling_follows_the_rate():
    rate, n = 0.25, 120
    http, deps = bg_app(rate=rate, rng=CountingRandom(seed=7), background={"daily_max_calls": 10_000,
                                                                          "daily_max_tokens": 100_000})
    for _ in range(n):
        post(http, verify=None)
    replay = random.Random(7)
    expected = sum(replay.random() < rate for _ in range(n))
    assert len(deps.log.labels()) == expected and 15 <= expected <= 45
    assert deps.labeler.rng.draws == n


@pytest.mark.parametrize("case", ["high tier", "fell back", "pass-through", "sync", "model error"])
def test_ineligible_requests_are_never_sampled(case):
    script = {"fell back": {"groq-small": FakeBehavior(fail_before="server")},
              "model error": {"groq-small": FakeBehavior(fail_before="server"),
                              "gemini-flash": FakeBehavior(fail_before="server")}}.get(case)
    http, deps = bg_app(script, predictor=HIGH if case == "high tier" else MEDIUM)
    post(http, verify="sync" if case == "sync" else None, model="groq-small" if case == "pass-through" else "auto")
    assert deps.log.labels() == [] and deps.labeler.rng.draws == 0


def test_a_cache_hit_is_never_sampled_for_a_background_label(tmp_path):
    http, deps = bg_app(cache=ResponseCache(tmp_path / "cache.sqlite3"))
    first = post(http, verify=None)
    repeat = post(http, verify=None)
    assert (first.headers["x-triage-cache"], repeat.headers["x-triage-cache"]) == ("miss", "hit")
    assert len(deps.log.labels(first.headers["x-triage-request-id"])) == 1
    assert deps.log.labels(repeat.headers["x-triage-request-id"]) == []
    assert deps.labeler.rng.draws == 1                                   # the repeat never reached sampling


def test_the_daily_cap_stops_sampling_and_logs_the_skip():
    http, deps = bg_app(background={"daily_max_calls": 4})
    for _ in range(4):
        post(http, verify=None)
    labels = deps.log.labels()
    assert [l.status for l in labels] == ["labeled", "labeled", "skipped", "skipped"]
    assert [l.skip_reason for l in labels[2:]] == ["daily_cap", "daily_cap"]
    assert all(l.calls == 0 for l in labels[2:])
    assert len(calls(deps, model="groq-large")) == 4                    # 2 labels x (strong answer + judge)


def test_the_token_cap_counts_too():
    http, deps = bg_app(background={"daily_max_tokens": 1})
    post(http, verify=None)
    post(http, verify=None)
    assert [l.status for l in deps.log.labels()] == ["labeled", "skipped"]


def test_the_cap_resets_the_next_utc_day_and_survives_a_restart(tmp_path):
    now = [datetime(2026, 11, 10, 23, 0, tzinfo=timezone.utc)]
    log = SQLiteRequestLog(tmp_path / "r.sqlite3")
    http, deps = bg_app(background={"daily_max_calls": 2}, log=log, clock=lambda: now[0])
    post(http, verify=None)
    post(http, verify=None)
    assert [l.status for l in log.labels()] == ["labeled", "skipped"]
    # a restart the same day: the new labeler reads today's usage back from the labels table
    http2, _ = bg_app(background={"daily_max_calls": 2}, log=log, clock=lambda: now[0])
    post(http2, verify=None)
    assert log.labels()[-1].skip_reason == "daily_cap"
    now[0] += timedelta(hours=2)                                         # past midnight UTC
    post(http2, verify=None)
    assert log.labels()[-1].status == "labeled"


def test_the_users_response_is_identical_with_and_without_sampling():
    def strip(r):
        body = r.json()
        body.pop("id"), body.pop("created")
        headers = {k: v for k, v in r.headers.items() if k not in ("x-triage-request-id", "date", "content-length")}
        return body, headers
    off, deps_off = bg_app(rate=0.0)
    on, deps_on = bg_app(rate=1.0)
    assert strip(post(off, verify=None)) == strip(post(on, verify=None))
    assert deps_off.log.labels() == [] and len(deps_on.log.labels()) == 1

    def streamed(http):
        with http.stream("POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"},
                         json={"model": "auto", "messages": RAIN, "stream": True}) as r:
            text = "".join(r.iter_text())
        return [line.split('"created"')[1].split(",", 1)[1] if '"created"' in line else line
                for line in text.splitlines()]
    assert streamed(off) == streamed(on)
    assert len(deps_on.log.labels()) == 2                                  # streams get labels too


def test_a_streamed_answer_is_graded_on_its_full_text(monkeypatch):
    seen = []
    real = background_module.judge
    monkeypatch.setattr(background_module, "judge",
                        lambda messages, constraints, reference, candidate, *a, **kw:
                        seen.append(candidate) or real(messages, constraints, reference, candidate, *a, **kw))
    http, deps = bg_app()
    with http.stream("POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"},
                     json={"model": "auto", "messages": RAIN, "stream": True}) as r:
        "".join(r.iter_text())
    assert seen == [fake_text("groq-small")]


@pytest.mark.parametrize("breakage", ["strong model fails", "judge fails", "judge raises", "label write fails"])
def test_a_failing_background_job_never_affects_the_response(breakage, monkeypatch):
    script = {"strong model fails": {"groq-large": FakeBehavior(fail_before="server")},
              "judge fails": {"groq-large": FakeBehavior(judge_fail="client")}}.get(breakage)
    baseline, _ = bg_app(rate=0.0)
    expected = post(baseline, verify=None).json()["choices"]
    http, deps = bg_app(script)
    if breakage == "judge raises":
        monkeypatch.setattr(background_module, "judge", lambda *a, **kw: 1 / 0)
    if breakage == "label write fails":
        monkeypatch.setattr(deps.log, "write_label", lambda label: 1 / 0)
    r = post(http, verify=None)
    assert r.status_code == 200 and r.json()["choices"] == expected
    assert post(http, verify=None).status_code == 200                       # and the server keeps serving
    labels = deps.log.labels()
    if breakage == "label write fails":
        assert labels == []
    else:
        assert labels[0].status == "skipped" and labels[0].skip_reason, breakage
        assert labels[0].skip_reason in ("model_failed", "error:ZeroDivisionError")


SECRET_PROMPT = "OCELOT-PROMPT-TEXT-4411 describe the rain"
SECRET_ANSWER = "PANGOLIN-ANSWER-TEXT-2299"
SECRET_REFERENCE = "NARWHAL-REFERENCE-TEXT-7733"


def test_no_text_in_the_labels_table_or_anywhere_in_the_database(tmp_path):
    log = SQLiteRequestLog(tmp_path / "r.sqlite3")
    http, deps = bg_app({"groq-small": FakeBehavior(text=SECRET_ANSWER),
                         "groq-large": FakeBehavior(text=SECRET_REFERENCE)}, log=log)
    secret = [{"role": "user", "content": SECRET_PROMPT}]
    assert post(http, messages=secret, verify=None).json()["choices"][0]["message"]["content"] == SECRET_ANSWER
    with http.stream("POST", "/v1/chat/completions", headers={"Authorization": f"Bearer {API_KEY}"},
                     json={"model": "auto", "messages": secret, "stream": True}) as r:
        "".join(r.iter_text())
    assert len(log.labels()) == 2 and all(l.status == "labeled" for l in log.labels())
    log.close()
    raw = b"".join(p.read_bytes() for p in tmp_path.iterdir() if p.name.startswith("r.sqlite3"))
    for text in (SECRET_PROMPT, "OCELOT-PROMPT", SECRET_ANSWER, "PANGOLIN-ANSWER", SECRET_REFERENCE, "NARWHAL",
                 "fake judge", API_KEY):
        assert text.encode() not in raw, text


def test_labels_table_schema(tmp_path):
    import sqlite3
    SQLiteRequestLog(tmp_path / "r.sqlite3").close()
    db = sqlite3.connect(tmp_path / "r.sqlite3")
    cols = [c[1] for c in db.execute("PRAGMA table_info(labels)")]
    assert cols == ["id", "request_id", "created_at", "status", "skip_reason", "passed", "judge_score",
                    "judge_version", "judge_model", "reference_model", "served_model", "served_tier", "calls",
                    "tokens", "cost"]
    assert {"idx_labels_request_id", "idx_labels_created_at"} <= {i[1] for i in db.execute("PRAGMA index_list(labels)")}
    req_cols = {c[1] for c in db.execute("PRAGMA table_info(requests)")}
    assert {"escalated_from", "escalated_to", "escalation_extra_cost", "escalation_extra_latency_ms", "check_cost",
            "check_latency_ms"} <= req_cols


# ---------- the background budget check ----------

def test_routing_yaml_background_caps_are_a_quarter_of_groq_large():
    from app.api.config_state import ActiveConfig
    from app.api.verification import daily_token_budget
    bg = ActiveConfig.from_file(load_registry()).snapshot().verification.background
    assert (bg.daily_max_calls, bg.daily_max_tokens) == (50, 50_000)
    assert bg.daily_max_tokens == daily_token_budget("groq-large") // 4


@pytest.mark.parametrize("tokens,ok", [(50_000, True), (100_000, True), (100_001, False), (200_000, False)])
def test_startup_rejects_a_token_cap_above_half_the_models_daily_budget(tokens, ok):
    if ok:
        app_with(background={"daily_max_tokens": tokens})
        return
    with pytest.raises(VerificationConfigError, match="half of groq-large's daily budget"):
        app_with(background={"daily_max_tokens": tokens})


def test_the_budget_check_covers_the_judge_model_too(tmp_path):
    from app.api.verification import BackgroundPolicy, check_background_budget
    limits = tmp_path / "collection.yaml"
    limits.write_text("limits:\n  groq-large: {tpd: 1000000}\n  judge-model: {tpd: 20000}\n  local: {}\n")
    policy = BackgroundPolicy(daily_max_tokens=15_000)
    check_background_budget(policy, ["groq-large", "local"], limits)             # no tpd for local: no limit
    with pytest.raises(VerificationConfigError, match="judge-model"):
        check_background_budget(policy, ["groq-large", "judge-model"], limits)
