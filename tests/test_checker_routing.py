"""Phase 4 Step 3: split option, checker verdicts, validation checker run, strategies, chart. No API calls.

Strategy (a)'s path is written by hand (tests/test_checker_path.py). The tests here swap in a reference
version so they test everything built around it.
"""
import json
import sys

import pytest

from app.providers.base import Response
from app.providers.common import ProviderError
from app.verifier.ai_checker import CheckerSettings, prompt_sha256
from eval import checker_routing as cr
from eval import checker_val_run as vr
from eval.check_log import CheckLog, input_sha256
from eval.checker_path import ANSWER, CHECK, Call
from eval.checker_verdicts import CheckEstimate, Verdict, load_verdicts, make_estimate, summarize_verdicts
from eval.routing_sim import EvalRow, load_eval_rows, score
from eval.splits import load_split, load_validation_split

SETTINGS = CheckerSettings(model_name="groq-small", checker_version=3)
A = lambda t: Call(ANSWER, t)
C = lambda t: Call(CHECK, t)


def reference_path(row, verdicts, format_ok, on_error="escalate"):
    if on_error not in ("escalate", "accept"):
        raise ValueError(on_error)
    if not format_ok[(row.id, "medium")]:
        return "high", [A("medium"), A("high")]
    v = verdicts.get((row.id, "medium"))
    status = v.status if v else "unknown"
    good = status == "good" or (status == "unknown" and on_error == "accept")
    return ("medium", [A("medium"), C("medium")]) if good else ("high", [A("medium"), C("medium"), A("high")])


@pytest.fixture
def ref(monkeypatch):
    monkeypatch.setattr(cr, "medium_then_check_path", reference_path)


def row(pid, low=False, medium=True, high=True):
    label = "low" if low else "medium" if medium else "high"
    return EvalRow(id=pid, task_type="qa", source="gsm8k", label=label,
                   passed={"low": low, "medium": medium, "high": high},
                   cost={"low": 0.0001, "medium": 0.001, "high": 0.01, "extra": 0.00005},
                   latency_ms={"low": 100.0, "medium": 400.0, "high": 2000.0, "extra": 150.0})


def ok_verdict(status, cost=0.0002, latency=300.0):
    return Verdict(status, "ok" if status != "unknown" else "missing", cost, latency)


# ---------- 1. split option ----------

def test_val_split_needs_the_explicit_flag_and_test_stays_blocked():
    with pytest.raises(PermissionError):
        load_eval_rows("val")
    with pytest.raises(PermissionError):
        load_eval_rows("test")
    val = load_eval_rows("val", allow_val=True)
    assert len(val) == 117 and {r.id for r in val} <= load_validation_split()
    assert not {r.id for r in val} & load_split()[1]
    assert len(load_eval_rows("fit")) == 481                       # fit needs no flag


def test_pareto_report_names_keep_the_old_ones_and_avoid_the_step3_chart():
    from eval.pareto import report_stem
    assert report_stem("train") == "pareto_train" and report_stem("test") == "pareto_test"
    assert report_stem("fit") == "pareto_fit" and report_stem("val") == "pareto_predict_only_val"


def test_split_probabilities_use_oof_on_train_and_fit():
    from eval.pareto import oof_probabilities, split_probabilities
    assert split_probabilities("classifier_c_logreg_cv", 1, "fit", []) == oof_probabilities("classifier_c_logreg_cv", 1)


@pytest.mark.parametrize("module", ["eval.pareto", "eval.cascade"])
def test_simulators_accept_fit_and_val_but_not_test(module, monkeypatch):
    import importlib
    mod = importlib.import_module(module)
    monkeypatch.setattr(sys, "argv", ["x", "--split", "test"])
    with pytest.raises(SystemExit):
        mod.main()


def test_agreement_sim_rejects_the_test_split():
    from eval import agreement_sim
    with pytest.raises(SystemExit):
        agreement_sim.main(["--split", "test"])


# ---------- 2. verdicts from a CheckLog file ----------

MSGS = [{"role": "user", "content": "What is 2 + 3? End with: Answer: <number>"}]


def record(pid, tier="medium", good=True, status="ok", error_kind=None, model="groq-small", version=3,
           prompt_hash=None, cost=0.0003, latency=420.0, answer="Answer: 5", output_tokens=90):
    return {"prompt_id": pid, "tier": tier, "status": status, "good": good if status == "ok" else None,
            "reason": "r", "checker_version": version, "checker_model": model,
            "prompt_sha256": prompt_hash or prompt_sha256(), "input_tokens": 400, "output_tokens": output_tokens,
            "reasoning_tokens": 40, "attempts": 1, "error_kind": error_kind, "reported_cost": cost,
            "latency_ms": latency, "raw": "", "input_sha256": input_sha256(MSGS, answer), "api_calls": 1,
            "timestamp": "2026-11-09T00:00:00+00:00"}


def write_log(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def verdicts_from(tmp_path, records, keys, **kw):
    path = write_log(tmp_path / "checks.jsonl", records)
    est = CheckEstimate(input_price=0.075, output_price=0.30, output_tokens=100, latency_ms=500.0)
    return load_verdicts(path, keys, model=kw.pop("model", "groq-small"), version=kw.pop("version", 3),
                         prompt_hash=kw.pop("prompt_hash", prompt_sha256()), estimate=est, **kw)


def test_good_bad_and_their_logged_cost(tmp_path):
    v = verdicts_from(tmp_path, [record("p1", good=True), record("p2", good=False, cost=0.0005, latency=700)],
                      [("p1", "medium"), ("p2", "medium")])
    assert v[("p1", "medium")] == Verdict("good", "ok", 0.0003, 420.0)
    assert v[("p2", "medium")] == Verdict("bad", "ok", 0.0005, 700.0)


def test_errors_are_unknown_never_dropped_and_still_charged(tmp_path):
    recs = [record("p1", status="error", error_kind="invalid_json", cost=0.0004),
            record("p2", status="error", error_kind="provider:timeout", cost=0.0)]
    v = verdicts_from(tmp_path, recs, [("p1", "medium"), ("p2", "medium")])
    assert v[("p1", "medium")].status == "unknown" and v[("p1", "medium")].why == "error:invalid_json"
    assert v[("p1", "medium")].cost == 0.0004 and not v[("p1", "medium")].estimated
    assert v[("p2", "medium")].status == "unknown" and "provider:timeout" in v[("p2", "medium")].why


def test_missing_records_are_unknown_with_an_estimated_cost(tmp_path):
    v = verdicts_from(tmp_path, [], [("p1", "medium"), ("p1", "low")])
    assert set(v) == {("p1", "medium"), ("p1", "low")}
    m = v[("p1", "medium")]
    assert (m.status, m.why, m.estimated) == ("unknown", "missing", True)
    assert m.cost == pytest.approx((600 * 0.075 + 100 * 0.30) / 1e6) and m.latency_ms == 500.0


def test_no_log_file_at_all_gives_all_missing(tmp_path):
    est = CheckEstimate(0.075, 0.30)
    v = load_verdicts(tmp_path / "nope.jsonl", [("p1", "medium")], model="groq-small", version=3,
                      prompt_hash=prompt_sha256(), estimate=est)
    assert v[("p1", "medium")].why == "missing" and not (tmp_path / "nope.jsonl").exists()


@pytest.mark.parametrize("change", [{"model": "groq-large"}, {"version": 2}, {"prompt_hash": "other"}])
def test_other_model_version_or_prompt_hash_counts_as_missing(tmp_path, change):
    v = verdicts_from(tmp_path, [record("p1", **change)], [("p1", "medium")])
    assert v[("p1", "medium")].why == "missing"


def test_newest_record_wins(tmp_path):
    recs = [record("p1", status="error", error_kind="provider:rate_limit"), record("p1", good=False)]
    assert verdicts_from(tmp_path, recs, [("p1", "medium")])[("p1", "medium")].status == "bad"


def test_a_record_for_a_different_answer_is_stale(tmp_path):
    inputs = {("p1", "medium"): (MSGS, "Answer: 6"), ("p2", "medium"): (MSGS, "Answer: 5")}
    v = verdicts_from(tmp_path, [record("p1"), record("p2")], list(inputs), inputs=inputs)
    assert v[("p1", "medium")].why == "missing" and v[("p2", "medium")].status == "good"


def test_estimate_uses_price_as_and_medians_of_ok_records():
    est = make_estimate("groq-small", [record("a", output_tokens=80, latency=300), record("b", output_tokens=120,
                                                                                        latency=500),
                                       record("c", status="error", error_kind="truncated", output_tokens=1024)])
    assert (est.input_price, est.output_price) == (0.075, 0.30)
    assert est.output_tokens == 100 and est.latency_ms == 400.0
    assert make_estimate("groq-large").output_price == 0.60


def test_summarize_verdicts_counts():
    s = summarize_verdicts({1: ok_verdict("good"), 2: ok_verdict("bad"), 3: Verdict("unknown", "missing", estimated=True)})
    assert (s["n"], s["good"], s["bad"], s["unknown"], s["estimated_cost"]) == (3, 1, 1, 1, 1)


# ---------- 3. the validation checker run ----------

@pytest.fixture(scope="module")
def real_val():
    return vr.load_inputs()


def test_val_items_are_the_format_passing_low_and_medium_answers(real_val):
    rows, prompts, answers, items, skipped = real_val
    assert len(rows) == 117
    by_tier = {t: sum(c["tier"] == t for c in items) for t in ("low", "medium")}
    assert by_tier == {"low": 106, "medium": 117}
    assert dict(skipped) == {"low: format checker fails it": 11}
    ids = {c["prompt_id"] for c in items}
    assert ids <= load_validation_split() and not ids & load_split()[1]


def test_val_items_refuse_non_validation_prompts(real_val):
    _, prompts, answers, _, _ = real_val
    fit = [r for r in load_eval_rows("fit")][:1]
    with pytest.raises(ValueError):
        vr.val_items(fit, prompts, answers)


def test_dry_run_counts_calls_and_writes_nothing(real_val, tmp_path):
    rows, prompts, answers, items, skipped = real_val
    out = vr.dry_run("groq-small", items, skipped, prompts, answers, len(rows), val_dir=tmp_path / "v", settings=SETTINGS)
    assert out["calls_to_make"] == 223 and out["calls_to_make_by_tier"] == {"low": 106, "medium": 117}
    assert out["already_finished"] == 0 and out["estimated_cost_usd"] > 0
    assert not (tmp_path / "v").exists()
    medium_only = [c for c in items if c["tier"] == "medium"]
    assert vr.dry_run("groq-small", medium_only, skipped, prompts, answers, len(rows), val_dir=tmp_path / "v",
                      settings=SETTINGS)["calls_to_make"] == 117


PROMPTS = {f"p{i}": {"messages": [{"role": "user", "content": f"question p{i}"}]} for i in range(3)}
ANSWERS = {(f"p{i}", "medium"): {"answer": f"answer {i}"} for i in range(3)}
ITEMS = [{"prompt_id": f"p{i}", "tier": "medium"} for i in range(3)]


class FakeSend:
    def __init__(self, script=None):
        self.calls, self.script = [], dict(script or {})

    def __call__(self, messages, model_name, settings):
        pid = next(p for p in PROMPTS if f"question {p}" in messages[1]["content"])
        self.calls.append(pid)
        item = self.script.get(pid, '{"good": true, "reason": "fine"}')
        if isinstance(item, list):
            item = item.pop(0) if len(item) > 1 else item[0]
        if isinstance(item, Exception):
            raise item
        return Response(text=item, model_name=model_name, model_id="x", provider="groq", input_tokens=100,
                        output_tokens=20, latency_ms=5.0, cost=0.0, reported_cost=0.00001, finish_reason="stop",
                        reasoning_tokens=10)


def val_run(tmp_path, send, **kw):
    return vr.run("groq-small", ITEMS, PROMPTS, ANSWERS, send_fn=send, use_pacer=False, val_dir=tmp_path,
                  settings=SETTINGS, sleep=lambda s: None, say=lambda m: None, **kw)


def test_val_run_is_resumable_and_logs_errors(tmp_path):
    send = FakeSend({"p1": "not json", "p2": [ProviderError("groq", "timeout", "slow"), '{"good": false, "reason": "x"}']})
    summary = val_run(tmp_path, send, passes=2)
    assert summary["real_calls"] == 4 and send.calls.count("p2") == 2
    log = CheckLog(vr.log_path("groq-small", tmp_path))
    assert [r["error_kind"] for r in log.records if r["prompt_id"] == "p2"] == ["provider:timeout", None]
    assert val_run(tmp_path, FakeSend())["real_calls"] == 0          # everything finished, errors included


def test_val_run_limit_and_its_log_feeds_load_verdicts(tmp_path):
    assert val_run(tmp_path, FakeSend(), limit=2)["real_calls"] == 2
    assert val_run(tmp_path, FakeSend())["real_calls"] == 1
    inputs = {("p%d" % i, "medium"): (PROMPTS["p%d" % i]["messages"], ANSWERS[("p%d" % i, "medium")]["answer"])
              for i in range(3)}
    v = load_verdicts(vr.log_path("groq-small", tmp_path), list(inputs), model="groq-small", version=3,
                      prompt_hash=prompt_sha256(), inputs=inputs)
    assert {x.status for x in v.values()} == {"good"} and all(x.cost == 0.00001 for x in v.values())


def test_val_run_refuses_while_a_calibration_run_holds_the_model(monkeypatch, tmp_path):
    monkeypatch.setattr(vr, "load_inputs", lambda tiers: ([], {}, {}, [], {}))
    monkeypatch.setattr(vr, "running_pid", lambda path: 12345 if "checker_calibration" in str(path) else None)
    monkeypatch.setattr(vr, "VAL_DIR", tmp_path)
    with pytest.raises(SystemExit, match="calibration run"):
        vr.main(["--model", "groq-small"])


# ---------- scoring ----------

def test_score_decisions_rounds_run_in_parallel_and_add_up():
    r = row("p1")
    vs = {("p1", "medium"): ok_verdict("bad")}
    s = cr.score_decisions([r], [("high", [[A("low"), A("medium")], [C("medium")], [A("high")]])], vs)
    assert s["cost_per_1k"] == pytest.approx((0.0001 + 0.001 + 0.0002 + 0.01) * 1000)
    assert s["latency_p50_ms"] == pytest.approx(400 + 300 + 2000)
    assert s["escalation_rate"] == 1 and s["checks_per_request"] == 1


def test_a_check_without_a_verdict_cannot_be_scored():
    with pytest.raises(ValueError, match="no verdict"):
        cr.score_decisions([row("p1")], [("medium", [A("medium"), C("medium")])], {})


def test_score_decisions_matches_routing_sim_for_single_answers():
    rows = [row("p1"), row("p2", medium=False)]
    tiers = ["medium", "high"]
    got = cr.score_decisions(rows, [(t, [A(t)]) for t in tiers], {})
    want = score(rows, tiers)
    for k in ("quality", "quality_retained", "cost_per_1k", "cost_saved", "latency_p50_ms", "mix"):
        assert got[k] == pytest.approx(want[k])


# ---------- (b) agreement on comparable prompts ----------

def b_inputs(agree=True, comparable=True, medium_format=True):
    keys = {("p1", "low"): "N:5", ("p1", "medium"): "N:5" if agree else "N:6", ("p1", "extra"): "N:5"}
    fmt = {("p1", "medium"): medium_format, ("p1", "low"): True}
    return {("p1", "medium"): ok_verdict("good")}, fmt, keys, {"p1": comparable}


def test_b_comparable_agreement_keeps_medium_and_pays_for_both_answers_in_parallel(ref):
    vs, fmt, keys, comp = b_inputs(agree=True)
    final, rounds = cr.medium_then_check_with_agreement_path(row("p1"), vs, fmt, keys, comp)
    assert (final, rounds) == ("medium", [[A("low"), A("medium")]])
    s = cr.score_decisions([row("p1")], [(final, rounds)], vs)
    assert s["cost_per_1k"] == pytest.approx(1.1) and s["latency_p50_ms"] == 400 and s["checks_per_request"] == 0


def test_b_comparable_disagreement_or_format_failure_goes_to_high_without_a_check(ref):
    for kw in ({"agree": False}, {"medium_format": False}):
        vs, fmt, keys, comp = b_inputs(**kw)
        final, rounds = cr.medium_then_check_with_agreement_path(row("p1"), vs, fmt, keys, comp)
        assert final == "high" and C("medium") not in [c for rnd in rounds for c in rnd]


def test_b_missing_final_answer_counts_as_disagreement(ref):
    vs, fmt, keys, comp = b_inputs()
    keys[("p1", "low")] = None
    assert cr.medium_then_check_with_agreement_path(row("p1"), vs, fmt, keys, comp)[0] == "high"


def test_b_non_comparable_prompts_are_exactly_strategy_a(ref):
    vs, fmt, keys, comp = b_inputs(comparable=False)
    for on_error in ("escalate", "accept"):
        assert cr.medium_then_check_with_agreement_path(row("p1"), vs, fmt, keys, comp, on_error) == \
               reference_path(row("p1"), vs, fmt, on_error)


# ---------- (c) classifier start, no AI checker on low ----------

def c_path(start, agree=True, comparable=True, low_format=True, verdict="good", on_error="escalate"):
    keys = {("p1", "low"): "N:5", ("p1", "extra"): "N:5" if agree else "N:7", ("p1", "medium"): "N:5"}
    fmt = {("p1", "low"): low_format, ("p1", "medium"): True}
    vs = {("p1", "medium"): ok_verdict(verdict)}
    return cr.classifier_start_then_check_path(row("p1"), start, vs, fmt, keys, {"p1": comparable}, on_error)


def test_c_start_high_is_one_unchecked_answer(ref):
    assert c_path("high") == ("high", [[A("high")]])


def test_c_start_medium_is_strategy_a(ref):
    assert c_path("medium", verdict="bad") == ("high", [A("medium"), C("medium"), A("high")])


def test_c_low_with_agreement_from_extra(ref):
    assert c_path("low") == ("low", [[A("low"), A("extra")]])
    final, rounds = c_path("low", agree=False, verdict="good")
    assert final == "medium" and rounds == [[A("low"), A("extra")], [A("medium")], [C("medium")]]


def test_c_low_non_comparable_gets_only_the_format_check(ref):
    assert c_path("low", comparable=False) == ("low", [[A("low")]])
    final, rounds = c_path("low", comparable=False, low_format=False, verdict="bad")
    assert final == "high" and rounds == [[A("low")], [A("medium")], [C("medium")], [A("high")]]


def test_c_never_runs_the_ai_checker_on_low_answers(ref):
    rows, probs, vs, fmt, keys, comp = synthetic()
    for tau in (0.0, 0.3, 0.6, 1.0):
        for r in rows:
            start = cr.choose_tier(*probs[r.id], tau)
            _, rounds = cr.classifier_start_then_check_path(r, start, vs, fmt, keys, comp)
            assert C("low") not in [c for rnd in cr.as_rounds(rounds) for c in rnd]


def test_c_and_b_reject_a_bad_on_error(ref):
    with pytest.raises(ValueError):
        c_path("low", on_error="maybe")
    vs, fmt, keys, comp = b_inputs()
    with pytest.raises(ValueError):
        cr.medium_then_check_with_agreement_path(row("p1"), vs, fmt, keys, comp, "maybe")


# ---------- (d) baselines and the same-cost random bar ----------

BASE = {"always_low": {"cost_per_1k": 0.1, "quality_retained": 0.5},
        "always_medium": {"cost_per_1k": 1.0, "quality_retained": 0.9},
        "always_high": {"cost_per_1k": 10.0, "quality_retained": 1.0}}


def test_random_same_cost_interpolates_between_adjacent_tiers():
    assert cr.random_same_cost(1.0, BASE) == pytest.approx(0.9)
    assert cr.random_same_cost(0.55, BASE) == pytest.approx(0.7)
    assert cr.random_same_cost(5.5, BASE) == pytest.approx(0.95)
    assert cr.random_same_cost(0.05, BASE) is None and cr.random_same_cost(11, BASE) is None


def test_beats_always_medium_needs_cheaper_and_better():
    assert cr.beats_always_medium({"cost_per_1k": 0.9, "quality_retained": 0.95}, BASE)
    assert not cr.beats_always_medium({"cost_per_1k": 1.1, "quality_retained": 0.99}, BASE)
    assert not cr.beats_always_medium({"cost_per_1k": 0.5, "quality_retained": 0.9}, BASE)


def synthetic(n=40):
    rows, probs, vs, fmt, keys, comp = [], {}, {}, {}, {}, {}
    for i in range(n):
        pid = f"p{i:02d}"
        r = row(pid, low=i % 3 == 0, medium=i % 5 != 0, high=i % 7 != 0)
        rows.append(r)
        probs[pid] = (i / n, min(0.99, i / n + 0.3))
        comp[pid] = i % 2 == 0
        for t in ("low", "medium"):
            fmt[(pid, t)] = i % 11 != 0
            vs[(pid, t)] = ok_verdict("bad" if not r.passed[t] and i % 4 else "good")
        vs[(pid, "medium")] = ok_verdict("unknown") if i % 13 == 0 else vs[(pid, "medium")]
        keys.update({(pid, "low"): "N:1" if r.passed["low"] else "N:2", (pid, "medium"): "N:1",
                     (pid, "extra"): "N:1" if i % 6 == 0 else "N:3"})
    return rows, probs, vs, fmt, keys, comp


def test_simulate_scores_every_strategy_on_the_same_rows(ref):
    rows, probs, vs, fmt, keys, comp = synthetic()
    taus = [0.0, 0.5, 1.0]
    result = cr.simulate(rows, probs, vs, fmt, keys, comp, taus=taus)
    assert set(result["baselines"]) == {"always_low", "always_medium", "always_high", "oracle"}
    assert len(result["fixed"]) == 4 and len(result["curves"]) == 3
    assert all(r["n"] == 40 for r in result["fixed"].values())
    assert all(len(pts) == 3 and all(p["n"] == 40 for p in pts) for pts in result["curves"].values())
    a_esc, a_acc = (result["fixed"][f"a. medium_then_check (on_error={e})"] for e in ("escalate", "accept"))
    assert a_esc["mix"]["high"] > a_acc["mix"]["high"]                         # unknowns escalate only in one
    assert a_esc["check_cost_per_1k"] > 0
    predict = result["curves"]["d. predict-only Triage (v2)"]
    assert predict[1]["cost_per_1k"] == pytest.approx(
        score(rows, [cr.choose_tier(*probs[r.id], 0.5) for r in rows])["cost_per_1k"])
    # tau 1.0 sends everything to high: identical to always_high, whatever the checker says
    c_top = result["curves"]["c. classifier_start_then_check (on_error=escalate)"][2]
    assert c_top["cost_per_1k"] == pytest.approx(result["baselines"]["always_high"]["cost_per_1k"])
    assert "random_same_cost_retained" in a_esc and "gain_over_random_same_cost" in predict[0]
    summary = cr.summarize(result)
    assert set(summary) == set(result["fixed"]) | set(result["curves"])


def test_simulate_refuses_rows_without_probabilities(ref):
    rows, probs, vs, fmt, keys, comp = synthetic()
    del probs[rows[0].id]
    with pytest.raises(ValueError, match="probabilities"):
        cr.simulate(rows, probs, vs, fmt, keys, comp, taus=[0.5])


def test_chart_is_written(ref, tmp_path):
    rows, probs, vs, fmt, keys, comp = synthetic()
    result = cr.simulate(rows, probs, vs, fmt, keys, comp, taus=[0.0, 0.5, 1.0])
    cr.plot(result, tmp_path / "pareto_val.png", "test")
    assert (tmp_path / "pareto_val.png").stat().st_size > 10_000


def test_fake_verdicts_follow_the_given_rates():
    rows = [row(f"p{i}", low=i % 2 == 0, medium=i % 2 == 0) for i in range(2000)]
    vs = cr.fake_verdicts(rows, catch=0.8, false_alarm=0.1, model="groq-small")
    bad = [vs[(r.id, "medium")].status == "bad" for r in rows if not r.passed["medium"]]
    good = [vs[(r.id, "medium")].status == "bad" for r in rows if r.passed["medium"]]
    assert sum(bad) / len(bad) == pytest.approx(0.8, abs=0.04)
    assert sum(good) / len(good) == pytest.approx(0.1, abs=0.03)
    assert all(v.estimated and v.cost > 0 for v in vs.values())
