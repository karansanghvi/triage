"""Selective checking (Phase 4): skip the AI check on confident or short medium answers.

Simulation (eval/checker_routing.py strategy e) with fake verdicts and your medium_then_check_path, and
sync mode (app/api/verification.py) with fake models. No API calls.
"""
import copy

import pytest
from fastapi.testclient import TestClient

from app.api.auth import Auth
from app.api.fake_models import FakeBehavior, fake_text
from app.api.main import create_app
from app.api.verification import DEFAULT_SECTION, TierPolicy, VerificationConfigError, verification_from_dict
from eval import checker_routing as cr
from eval.checker_path import ANSWER, CHECK, Call
from eval.checker_verdicts import Verdict
from eval.routing_sim import EvalRow
from tests.api_helpers import ADMIN_KEY, API_KEY, FakePredictor, make_deps, routing_raw

A_MED, C_MED, A_HIGH = Call(ANSWER, "medium"), Call(CHECK, "medium"), Call(ANSWER, "high")


# ---------- simulation ----------

def row(pid, medium=True):
    return EvalRow(id=pid, task_type="qa", source="gsm8k", label="medium" if medium else "high",
                   passed={"low": False, "medium": medium, "high": True},
                   cost={"low": 0.0001, "medium": 0.001, "high": 0.01},
                   latency_ms={"low": 100.0, "medium": 400.0, "high": 2000.0})


def verdict(status, cost=0.0002):
    return Verdict(status, "ok" if status != "unknown" else "missing", cost, 300.0)


def selective(r, status="bad", *, p=0.95, tokens=200, above=0.96, min_tokens=0, fmt=True, on_error="escalate"):
    return cr.medium_then_check_selective_path(r, {(r.id, "medium"): verdict(status)}, {(r.id, "medium"): fmt},
                                               p_medium=p, tokens=tokens, skip_above=above,
                                               min_answer_tokens=min_tokens, on_error=on_error)


def test_a_confident_answer_skips_the_check_and_pays_nothing_extra():
    r = row("p1")
    final, calls, why = selective(r, "bad", p=0.97, above=0.96)        # the checker would have said bad
    assert (final, calls, why) == ("medium", [A_MED], "confident")
    vs = {("p1", "medium"): verdict("bad")}
    s = cr.score_decisions([r], [(final, calls)], vs)
    assert s["cost_per_1k"] == pytest.approx(1.0) and s["check_cost_per_1k"] == 0 and s["checks_per_request"] == 0


def test_a_borderline_answer_is_checked_exactly_like_strategy_a():
    r = row("p1")
    vs, fmt = {("p1", "medium"): verdict("bad")}, {("p1", "medium"): True}
    final, calls, why = selective(r, "bad", p=0.95, above=0.96)
    assert why is None and (final, calls) == cr.medium_then_check_path(r, vs, fmt) == ("high", [A_MED, C_MED, A_HIGH])
    assert selective(r, "bad", p=0.96, above=0.96)[2] == "confident"     # at the threshold: skipped


def test_a_format_failure_still_escalates_even_when_confident():
    final, calls, why = selective(row("p1"), "good", p=0.999, above=0.90, fmt=False)
    assert (final, calls, why) == ("high", [A_MED, A_HIGH], None)


def test_short_answers_skip_when_min_answer_tokens_is_set():
    r = row("p1")
    assert selective(r, "bad", p=0.5, tokens=50, min_tokens=100)[2] == "short"
    assert selective(r, "bad", p=0.5, tokens=99, min_tokens=100)[:2] == ("medium", [A_MED])
    assert selective(r, "bad", p=0.5, tokens=100, min_tokens=100)[2] is None      # not shorter: checked
    assert selective(r, "bad", p=0.5, tokens=50, min_tokens=0)[2] is None         # 0 = off
    assert selective(r, "bad", p=0.5, tokens=None, min_tokens=100)[2] is None     # unknown length: checked


@pytest.mark.parametrize("p", [0.0, 0.5, 0.999, 1.0])
def test_skip_above_one_never_skips(p):
    assert selective(row("p1"), "bad", p=p, above=1.0)[2] is None


def synthetic(n=60):
    rows, probs, vs, fmt, tokens = [], {}, {}, {}, {}
    for i in range(n):
        pid = f"p{i:02d}"
        r = row(pid, medium=i % 4 != 0)
        rows.append(r)
        probs[pid] = (0.3, 0.85 + 0.15 * i / (n - 1))                  # p_medium from 0.85 to exactly 1.0
        fmt[(pid, "medium")] = i % 13 != 0
        fmt[(pid, "low")] = True
        status = ("bad" if i % 3 else "good") if not r.passed["medium"] else ("bad" if i % 10 == 1 else "good")
        vs[(pid, "medium")] = verdict("unknown" if i % 17 == 5 else status)
        vs[(pid, "low")] = verdict("good")
        tokens[pid] = 40 + 7 * i
    return rows, probs, vs, fmt, tokens


@pytest.mark.parametrize("min_tokens", [0, 100])
def test_skip_above_one_matches_medium_then_check_exactly(min_tokens):
    rows, probs, vs, fmt, tokens = synthetic()
    sweep = cr.sweep_selective(rows, probs, vs, fmt, tokens if min_tokens == 0 else {}, skip_above=[1.0],
                               min_answer_tokens=[min_tokens])
    point = sweep[min_tokens][0]
    a = cr.score_decisions(rows, cr.run_a(rows, vs, fmt, "escalate"), vs)
    assert {k: point[k] for k in a} == a
    assert point["skipped_confident"] == point["skipped_short"] == 0


def test_the_sweep_covers_the_grid_and_accounts_for_every_bad_answer():
    rows, probs, vs, fmt, tokens = synthetic()
    sweep = cr.sweep_selective(rows, probs, vs, fmt, tokens)
    assert sorted(sweep) == [0, 100]
    for min_tokens, points in sweep.items():
        assert [p["skip_above"] for p in points] == [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]
        for p in points:
            assert p["min_answer_tokens"] == min_tokens
            assert p["bad_total"] == sum(not r.passed["medium"] for r in rows)
            assert p["bad_caught"] + p["bad_skipped"] + p["bad_missed_by_checker"] == p["bad_total"]
            assert p["checked_share"] == pytest.approx(p["checks_per_request"])
            assert p["escalated_share"] == p["escalation_rate"]
            for key in ("quality_retained", "cost_saved", "checked_share", "escalated_share", "bad_caught",
                        "bad_skipped", "good_escalated"):
                assert key in p
        checked = [p["checked_share"] for p in points]
        assert checked == sorted(checked)                    # a higher threshold never checks fewer answers
        assert points[-1]["bad_skipped"] == 0 or min_tokens == 100
    # the length rule only ever removes checks
    for a, b in zip(sweep[0], sweep[100]):
        assert b["checked_share"] <= a["checked_share"] and b["skipped_short"] >= 0


def test_simulate_adds_the_selective_lines_and_the_chart_draws_them(tmp_path, monkeypatch):
    rows, probs, vs, fmt, tokens = synthetic()
    keys = {(r.id, t): "N:1" for r in rows for t in ("low", "medium", "extra")}
    comp = {r.id: False for r in rows}
    result = cr.simulate(rows, probs, vs, fmt, keys, comp, taus=[0.0, 0.5, 1.0], answer_tokens=tokens)
    assert sorted(result["selective"]) == ["e. medium_then_check_selective (min_answer_tokens=0)",
                                           "e. medium_then_check_selective (min_answer_tokens=100)"]
    assert all(len(pts) == 6 and "random_same_cost_retained" in pts[0] for pts in result["selective"].values())
    summary = cr.summarize_selective(result)
    assert set(summary) == set(result["selective"]) and len(summary[next(iter(summary))]["points"]) == 6
    cr.plot(result, tmp_path / "pareto_val.png", "test")
    assert (tmp_path / "pareto_val.png").stat().st_size > 10_000
    # the old parts of the report are unchanged in shape
    assert len(result["fixed"]) == 4 and len(result["curves"]) == 3


def test_answer_tokens_are_the_visible_tokens():
    assert cr.answer_tokens({"output_tokens": 122, "reasoning_tokens": 51}) == 71
    assert cr.answer_tokens({"output_tokens": 30}) == 30
    assert cr.answer_tokens({"output_tokens": None}) is None and cr.answer_tokens(None) is None


# ---------- sync mode ----------

AUTH = Auth([API_KEY], [ADMIN_KEY])
RAIN = [{"role": "user", "content": "Write two sentences about rain."}]
ASK = [{"role": "user", "content": "What is 2 + 3? End with: Answer: <number>"}]
BAD_CHECK = '{"good": false, "reason": "wrong"}'


def app_with(script=None, *, p_medium=0.95, p_low=0.4, skip_above=None, min_tokens=0):
    raw = routing_raw()
    raw["verification"] = copy.deepcopy(DEFAULT_SECTION)
    medium = raw["verification"]["sync"]["tiers"]["medium"]
    medium["skip_check_above"], medium["min_answer_tokens"] = skip_above, min_tokens
    deps = make_deps(script, predictor=FakePredictor(p_low, p_medium), raw=raw)
    return TestClient(create_app(deps=deps, auth=AUTH)), deps


def post(http, messages=RAIN):
    return http.post("/v1/chat/completions", json={"model": "auto", "messages": messages},
                     headers={"Authorization": f"Bearer {API_KEY}", "X-Triage-Verify": "sync"})


def checks(deps):
    return [c for c in deps.fake_models.calls if c["kind"] == "check"]


def test_default_skips_above_0_90_with_no_length_rule():
    from app.api.config_state import ActiveConfig
    from app.registry import load_registry
    medium = ActiveConfig.from_file(load_registry()).snapshot().verification.sync.tiers["medium"]
    assert (medium.skip_check_above, medium.min_answer_tokens) == (0.90, 0)
    assert medium == TierPolicy(("format", "ai_checker"), "high", skip_check_above=0.90, min_answer_tokens=0)
    assert DEFAULT_SECTION["sync"]["tiers"]["medium"]["skip_check_above"] == 0.90
    http, deps = app_with({"groq-small": FakeBehavior(check_reply=BAD_CHECK)}, p_medium=0.999, skip_above=None)
    r = post(http)                                                       # very confident, but skipping is off
    assert r.headers["x-triage-check"] == "ai_checker" and r.json()["model"] == "groq-large" and len(checks(deps)) == 1


def test_a_confident_medium_answer_skips_the_check_in_sync_mode():
    http, deps = app_with({"groq-small": FakeBehavior(check_reply=BAD_CHECK)}, p_medium=0.97, skip_above=0.96)
    r = post(http)
    assert r.json()["model"] == "groq-small" and r.json()["choices"][0]["message"]["content"] == fake_text("groq-small")
    h = r.headers
    assert (h["x-triage-check"], h["x-triage-escalated"], h["x-triage-verified"]) == ("skipped_confident", "false", "sync")
    assert "skipped_confident" in h["x-triage-reason"]
    assert checks(deps) == []                                            # the checker was never called
    row = deps.log.rows()[-1]
    assert (row.decided_by, row.checker_verdict, row.check_cost, row.checker_version) == ("skipped_confident", None, 0.0, None)
    alone = deps.log.get(TestClient(create_app(deps=deps, auth=AUTH)).post(
        "/v1/chat/completions", json={"model": "groq-small", "messages": RAIN},
        headers={"Authorization": f"Bearer {API_KEY}"}).headers["x-triage-request-id"]).price_as_cost
    assert row.price_as_cost == pytest.approx(alone)                    # nothing extra paid


def test_a_borderline_medium_answer_is_still_checked():
    http, deps = app_with({"groq-small": FakeBehavior(check_reply=BAD_CHECK)}, p_medium=0.90, skip_above=0.96)
    r = post(http)
    assert r.headers["x-triage-check"] == "ai_checker" and r.json()["model"] == "groq-large" and len(checks(deps)) == 1


def test_a_short_answer_skips_the_check_in_sync_mode():
    # the fake answer has about 17 visible tokens
    http, deps = app_with({"groq-small": FakeBehavior(check_reply=BAD_CHECK)}, p_medium=0.90, min_tokens=100, skip_above=None)
    r = post(http)
    assert r.headers["x-triage-check"] == "skipped_short" and r.json()["model"] == "groq-small"
    assert checks(deps) == [] and deps.log.rows()[-1].decided_by == "skipped_short"
    long_answer = FakeBehavior(text="word " * 600, check_reply=BAD_CHECK)
    http, deps = app_with({"groq-small": long_answer}, p_medium=0.90, min_tokens=100, skip_above=None)
    assert post(http).headers["x-triage-check"] == "ai_checker"


def test_skip_check_above_one_always_checks():
    http, deps = app_with(p_medium=1.0, skip_above=1.0)
    assert post(http).headers["x-triage-check"] == "ai_checker" and len(checks(deps)) == 1


def test_a_format_failure_still_escalates_when_confident():
    http, deps = app_with({"groq-small": FakeBehavior(text="five")}, p_medium=0.99, skip_above=0.9)
    r = post(http, ASK)
    assert r.headers["x-triage-check"] == "format" and r.json()["model"] == "groq-large" and checks(deps) == []


def test_a_low_answer_escalated_to_medium_can_skip_mediums_check():
    http, deps = app_with({"groq-allam": FakeBehavior(text="five"),
                           "groq-small": FakeBehavior(text="2 + 3 = 5.\nAnswer: 5", check_reply=BAD_CHECK)},
                          p_low=0.95, p_medium=0.97, skip_above=0.96)
    r = post(http, ASK)
    assert r.json()["model"] == "groq-small" and r.headers["x-triage-check"] == "skipped_confident"
    assert r.headers["x-triage-escalated"] == "true" and checks(deps) == []


@pytest.mark.parametrize("tier,change", [
    ("medium", {"skip_check_above": 1.5}),
    ("medium", {"skip_check_above": 0}),
    ("medium", {"skip_check_above": -0.1}),
    ("medium", {"skip_check_above": "high"}),
    ("medium", {"min_answer_tokens": -1}),
    ("medium", {"min_answer_tokens": 2.5}),
    ("low", {"skip_check_above": 0.9}),                 # low runs no ai_checker: nothing to skip
    ("low", {"min_answer_tokens": 50}),
])
def test_bad_selective_settings_are_rejected_at_startup(tier, change):
    section = copy.deepcopy(DEFAULT_SECTION)
    section["sync"]["tiers"][tier].update(change)
    with pytest.raises(VerificationConfigError):
        verification_from_dict({"verification": section})


def test_good_selective_settings_load():
    section = copy.deepcopy(DEFAULT_SECTION)
    section["sync"]["tiers"]["medium"].update(skip_check_above=0.95, min_answer_tokens=100)
    medium = verification_from_dict({"verification": section}).sync.tiers["medium"]
    assert (medium.skip_check_above, medium.min_answer_tokens) == (0.95, 100)
