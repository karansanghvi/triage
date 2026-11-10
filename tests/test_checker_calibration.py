"""Checker calibration: sample, runner and report (Phase 4 Step 2). Fake send only: no API calls.

The report tests swap in a reference checker_metrics so they test the report's own logic; the real
checker_metrics has its own tests in test_checker_metrics.py.
"""
import json
import random

import pytest

from app.providers.base import Response
from app.providers.common import ProviderError
from app.registry import PROJECT_ROOT
from app.verifier.ai_checker import CheckerSettings, prompt_sha256
from eval import checker_calibration as cc
from eval import checker_calibration_report as rep
from eval import checker_calibration_sample as cs
from eval.check_log import CheckLog
from eval.splits import load_split, load_validation_split

SETTINGS = CheckerSettings(model_name="groq-small", checker_version=3)


# ---------- sample: pure functions ----------

def outcome(pid, low=("pass", "final_answer"), medium=("pass", "final_answer"), task="qa", source="gsm8k"):
    tier = lambda s, m: {"status": s, "method": m, "reason": "r", "judge_score": None}
    return {"id": pid, "label_status": "labeled", "task_type": task, "source": source,
            "tiers": {"low": tier(*low), "medium": tier(*medium), "high": tier("pass", "final_answer")}}


ASK = [{"role": "user", "content": "What is 2 + 3? End with: Answer: <number>"}]
ESSAY = [{"role": "user", "content": "Write a poem about rain."}]


def test_find_candidates_applies_every_filter():
    outcomes = [
        outcome("p_ok", low=("fail", "final_answer"), medium=("pass", "final_answer")),
        outcome("p_assumed", low=("pass", "judged"), medium=("assumed_pass", "assumed")),
        outcome("p_pre", low=("fail", "pre_failed"), medium=("fail", "judged")),
        outcome("mmlu_0079"),
        outcome("gsm8k_0142"),                                     # medium is a smoke answer, low is not
        outcome("p_val"),                                          # not in the fit set given below
    ]
    prompts = {o["id"]: {"messages": ESSAY if o["id"] == "p_pre" else ASK} for o in outcomes}
    answers = {(o["id"], t): {"answer": "Answer: 5", "model_name": "m"} for o in outcomes for t in ("low", "medium")}
    answers[("p_ok", "low")] = {"answer": "five", "model_name": "m"}          # no Answer: line -> format flag
    fit = {o["id"] for o in outcomes} - {"p_val"}
    cands, dropped = cs.find_candidates(outcomes, prompts, answers, fit)
    got = {(c["prompt_id"], c["tier"]) for c in cands}
    assert got == {("p_ok", "medium"), ("p_assumed", "low"), ("p_pre", "medium"), ("gsm8k_0142", "low")}
    assert dropped == {"format_flagged": 1, "assumed_pass": 1, "pre_failed": 1, "excluded_prompt": 2,
                       "excluded_smoke": 1}
    by = {(c["prompt_id"], c["tier"]): c for c in cands}
    assert by[("p_pre", "medium")]["truth_source"] == "judged" and by[("p_pre", "medium")]["truth_bad"]
    assert by[("p_ok", "medium")]["comparable"] is True and by[("p_pre", "medium")]["comparable"] is False


def synthetic(n_per=30):
    out = []
    for tier in ("low", "medium"):
        for source in ("exact", "judged"):
            for bad in (False, True):
                for i in range(n_per):
                    out.append({"prompt_id": f"{tier}_{source}_{bad}_{i:02d}", "tier": tier, "truth_source": source,
                                "truth_bad": bad})
    return out


def test_select_sample_follows_the_plan_and_seed():
    plan = {("medium", "exact", "bad"): "all", ("medium", "exact", "good"): 10, ("low", "judged", "bad"): 5}
    a = cs.select_sample(synthetic(), plan, seed=42)
    assert cs.counts(a) == {"low/judged/bad": 5, "medium/exact/bad": 30, "medium/exact/good": 10}
    assert a == cs.select_sample(list(reversed(synthetic())), plan, seed=42)      # order doesn't matter
    assert a != cs.select_sample(synthetic(), plan, seed=7)


def test_select_sample_refuses_a_too_small_stratum():
    with pytest.raises(ValueError, match="plan asks for 31"):
        cs.select_sample(synthetic(), {("low", "exact", "bad"): 31})


def test_repeats_are_half_bad_half_good_and_seeded():
    sample = cs.select_sample(synthetic(), {k: 5 for k in [("low", "exact", "bad"), ("low", "exact", "good"),
                                                          ("medium", "judged", "bad"), ("medium", "judged", "good")]})
    picked = cs.select_repeats(sample, 8, seed=1)
    bad = {(c["prompt_id"], c["tier"]) for c in sample if c["truth_bad"]}
    assert len(picked) == 8 and sum(p in bad for p in picked) == 4
    assert picked == cs.select_repeats(sample, 8, seed=1)


# ---------- sample: the real one (reads fit-set outcomes only) ----------

@pytest.fixture(scope="module")
def real_sample():
    return cs.build_sample()


def test_real_sample_counts(real_sample):
    assert real_sample["sample_counts"] == {
        "low/exact/bad": 25, "low/exact/good": 25, "low/judged/bad": 25, "low/judged/good": 25,
        "medium/exact/bad": 26, "medium/exact/good": 100, "medium/judged/bad": 35, "medium/judged/good": 80}
    assert len(real_sample["items"]) == 341
    pool = real_sample["pool_counts"]
    for key in ("medium/exact/bad", "medium/judged/bad", "medium/judged/good"):      # "all" strata
        assert real_sample["sample_counts"][key] == pool[key]


def test_real_sample_is_fit_only_and_excludes_smoke_and_ambiguous(real_sample):
    train, test = load_split()
    ids = {c["prompt_id"] for c in real_sample["items"]}
    assert not ids & test and not ids & load_validation_split() and ids <= train
    pairs = {(c["prompt_id"], c["tier"]) for c in real_sample["items"]}
    assert not pairs & cs.EXCLUDED_ANSWERS and "mmlu_0079" not in ids
    assert len(pairs) == len(real_sample["items"])                                   # no duplicates
    assert len(real_sample["repeat_items"]) == 20
    assert {tuple(x) for x in real_sample["repeat_items"]} <= pairs


def test_make_sample_refuses_to_redraw(tmp_path, monkeypatch, real_sample):
    monkeypatch.setattr(cs, "build_sample", lambda seed: real_sample)
    path = tmp_path / "sample.json"
    cs.make_sample(42, False, path)
    with pytest.raises(SystemExit, match="cherry-picking"):
        cs.make_sample(42, False, path)
    cs.make_sample(42, True, path)                                                  # --force redraws


# ---------- runner ----------

ITEMS = [{"prompt_id": f"p{i}", "tier": "medium", "truth_source": "exact", "truth_bad": i % 2 == 0,
          "truth_reason": "", "task_type": "qa", "dataset": "gsm8k", "comparable": False, "answer_model": "m"}
         for i in range(4)]


def write_sample(tmp_path, items=ITEMS, repeat=(("p0", "medium"), ("p1", "medium"))):
    path = tmp_path / "sample.json"
    path.write_text(json.dumps({"items": items, "repeat_items": [list(x) for x in repeat],
                                "sample_counts": cs.counts(items), "pool_counts": cs.counts(items)}))
    return path


class FakeSend:
    def __init__(self, script=None):
        self.calls, self.script = [], dict(script or {})

    def __call__(self, messages, model_name, settings):
        pid = next(p for p in ("p0", "p1", "p2", "p3") if f"question {p}" in messages[1]["content"])
        self.calls.append((pid, model_name))
        item = self.script.get(pid, '{"good": true, "reason": "fine"}')
        if isinstance(item, list):
            item = item.pop(0) if len(item) > 1 else item[0]
        if isinstance(item, Exception):
            raise item
        return Response(text=item, model_name=model_name, model_id="x", provider="groq", input_tokens=100,
                        output_tokens=20, latency_ms=5.0, cost=0.0, reported_cost=0.0, finish_reason="stop",
                        reasoning_tokens=10)


PROMPTS = {f"p{i}": {"messages": [{"role": "user", "content": f"question p{i}"}]} for i in range(4)}
ANSWERS = {(f"p{i}", "medium"): {"answer": f"answer {i}"} for i in range(4)}


def run(tmp_path, send, **kw):
    kw.setdefault("sample_path", write_sample(tmp_path))
    return cc.run(kw.pop("model", "groq-small"), send_fn=send, use_pacer=False, calib_dir=tmp_path, prompts=PROMPTS,
                  answers=ANSWERS, settings=SETTINGS, sleep=lambda s: None, say=lambda m: None, **kw)


def test_run_checks_everything_and_a_rerun_costs_nothing(tmp_path):
    send = FakeSend()
    summary = run(tmp_path, send)
    assert summary["real_calls"] == 4 and summary["ok"] == 4
    assert len(CheckLog(cc.log_path("groq-small", False, tmp_path)).records) == 4
    again = FakeSend()
    assert run(tmp_path, again)["real_calls"] == 0 and again.calls == []


def test_errors_are_recorded_not_dropped(tmp_path):
    send = FakeSend({"p1": "not json", "p2": '{"good": "yes", "reason": "x"}'})
    summary = run(tmp_path, send)
    assert summary["error:invalid_json"] == 1 and summary["error:bad_shape"] == 1
    latest = CheckLog(cc.log_path("groq-small", False, tmp_path)).latest()
    assert {k[0]: r["error_kind"] for k, r in latest.items()} == {"p0": None, "p1": "invalid_json",
                                                                  "p2": "bad_shape", "p3": None}
    assert run(tmp_path, FakeSend())["real_calls"] == 0                            # deterministic errors are finished


def test_provider_errors_are_logged_and_retried(tmp_path):
    send = FakeSend({"p3": [ProviderError("groq", "timeout", "slow"), '{"good": false, "reason": "wrong"}']})
    summary = run(tmp_path, send, passes=2)
    log = CheckLog(cc.log_path("groq-small", False, tmp_path))
    p3 = [r for r in log.records if r["prompt_id"] == "p3"]
    assert [r["error_kind"] for r in p3] == ["provider:timeout", None]             # both kept in the log
    assert summary["real_calls"] == 4 and send.calls.count(("p3", "groq-small")) == 2


def test_limit_caps_real_checks(tmp_path):
    send = FakeSend()
    assert run(tmp_path, send, limit=2)["real_calls"] == 2
    assert run(tmp_path, FakeSend())["real_calls"] == 2                            # resumes with the rest


def test_models_use_separate_logs(tmp_path):
    run(tmp_path, FakeSend())
    send = FakeSend()
    assert run(tmp_path, send, model="groq-large")["real_calls"] == 4
    assert {m for _, m in send.calls} == {"groq-large"}
    assert cc.log_path("groq-large", False, tmp_path).exists() and cc.log_path("groq-small", False, tmp_path).exists()


def test_repeat_mode_reruns_only_finished_repeat_items_in_a_separate_log(tmp_path):
    assert run(tmp_path, FakeSend(), repeat=True)["real_calls"] == 0               # main run not done yet
    run(tmp_path, FakeSend())
    send = FakeSend()
    assert run(tmp_path, send, repeat=True)["real_calls"] == 2                     # never reused from the main log
    assert sorted(p for p, _ in send.calls) == ["p0", "p1"]
    assert len(CheckLog(cc.log_path("groq-small", True, tmp_path)).records) == 2
    assert run(tmp_path, FakeSend(), repeat=True)["real_calls"] == 0               # the repeat itself is resumable


def test_daily_budget_waits_or_stops(tmp_path):
    class Full:
        def __init__(self): self.room = 0
        def daily_budget_left(self, estimate):
            self.room += 1
            return self.room > 1                                                   # full once, then room
        def seconds_until_daily_room(self, estimate): return 10.0
        def wait_for_minute_slot(self, estimate): return 0.0
        def record(self, calls, tokens): pass
    waited = []
    paced = cc.PacedSend(Full(), FakeSend(), wait=True, say=lambda m: None, waiter=waited.append)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "question p0"}]
    paced(messages, "groq-small", SETTINGS)
    assert len(waited) == 1 and paced.calls == 1
    stop = cc.PacedSend(Full(), FakeSend(), wait=False)
    with pytest.raises(cc.BudgetExhausted):
        stop(messages, "groq-small", SETTINGS)


def test_runner_never_writes_to_data(tmp_path):
    data = PROJECT_ROOT / "data"
    snap = lambda: {str(p): p.stat().st_mtime_ns for p in data.rglob("*") if p.is_file()}
    before = snap()
    run(tmp_path, FakeSend())
    assert snap() == before


# ---------- report (with a reference checker_metrics) ----------

def reference_metrics(rows):
    rows = list(rows)
    tp = sum(r["truth_bad"] and r["checker_bad"] for r in rows)
    fp = sum(not r["truth_bad"] and r["checker_bad"] for r in rows)
    tn = sum(not r["truth_bad"] and not r["checker_bad"] for r in rows)
    fn = sum(r["truth_bad"] and not r["checker_bad"] for r in rows)
    div = lambda a, b: a / b if b else None
    return {"n": len(rows), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "catch_rate": div(tp, tp + fn),
            "trust_rate": div(tp, tp + fp), "false_alarm": div(fp, fp + tn)}


@pytest.fixture
def metrics(monkeypatch):
    monkeypatch.setattr(rep, "checker_metrics", reference_metrics)


def test_wilson_known_values():
    assert rep.wilson(0, 0) is None
    lo, hi = rep.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert rep.wilson(0, 20)[0] == 0.0 and rep.wilson(20, 20)[1] == 1.0


def test_reweighted_trust_formula():
    assert rep.reweighted_trust(0.15, 0.6, 0.1) == pytest.approx(0.15 * 0.6 / (0.15 * 0.6 + 0.85 * 0.1))
    assert rep.reweighted_trust(0.5, 0.0, 0.0) is None
    assert rep.reweighted_trust(0.5, None, 0.1) is None


def row(tier, source, bad, checker_bad, state="ok", **kw):
    return {"prompt_id": kw.get("pid", f"{tier}{source}{bad}{random.random()}"), "tier": tier, "truth_source": source,
            "truth_bad": bad, "checker_bad": checker_bad if state == "ok" else None, "state": state,
            "error_kind": kw.get("kind"), "dataset": kw.get("dataset", "gsm8k"), "task_type": kw.get("task", "qa"),
            "comparable": kw.get("comparable", True)}


def test_errors_are_excluded_from_metrics_and_reported_separately(metrics):
    rows = [row("medium", "exact", True, True), row("medium", "exact", False, False),
            row("medium", "exact", True, None, state="error", kind="truncated"),
            row("medium", "exact", False, None, state="pending", kind="provider:timeout"),
            row("medium", "exact", False, None, state="missing")]
    r = rep.model_report(rows, {"medium/exact/bad": 10, "medium/exact/good": 10})
    assert r["by_tier (sample, unweighted)"]["medium"]["n"] == 2
    e = r["errors"]["all"]
    assert (e["n"], e["errors"], e["pending"], e["missing"], e["by_kind"]) == (3, 1, 1, 1, {"truncated": 1})
    assert e["error_rate"] == pytest.approx(1 / 3)


def test_pool_weighting_and_raw_trust_label(metrics):
    # exact: catch 1/2, false alarm 0/2; judged: catch 2/2, false alarm 2/2
    rows = ([row("medium", "exact", True, True), row("medium", "exact", True, False),
             row("medium", "exact", False, False), row("medium", "exact", False, False)]
            + [row("medium", "judged", True, True)] * 2 + [row("medium", "judged", False, True)] * 2)
    pool = {"medium/exact/bad": 30, "medium/judged/bad": 10, "medium/exact/good": 300, "medium/judged/good": 100}
    w = rep.pool_weighted(rows, "medium", pool)
    assert w["catch_rate"] == pytest.approx((30 * 0.5 + 10 * 1.0) / 40)
    assert w["false_alarm"] == pytest.approx((300 * 0.0 + 100 * 1.0) / 400)
    t = rep.trust_section(rows, "medium", pool)
    assert t["trust_rate_raw_sample (inflated by oversampling bad answers)"] == pytest.approx(3 / 5)
    key = "all format-passing medium answers (base rate 15%)"
    assert t["trust_rate_reweighted"][key]["trust_rate"] == pytest.approx(
        rep.reweighted_trust(0.15, w["catch_rate"], w["false_alarm"]))
    lo, hi = t["trust_rate_reweighted"][key]["bootstrap_95_ci"]
    assert 0.0 <= lo <= hi <= 1.0
    assert set(t["trust_rate_reweighted"]) == {key, "non-comparable medium answers (base rate 22%)"}


def test_breakdowns_and_non_comparable_slice(metrics):
    rows = [row("medium", "exact", True, True, dataset="mmlu", task="qa", comparable=True),
            row("medium", "judged", False, True, dataset="arena", task="coding", comparable=False),
            row("low", "exact", False, False, dataset="own", task="extraction", comparable=False)]
    r = rep.model_report(rows, {})
    assert list(r["by_tier_and_dataset"]) == ["low/own", "medium/arena", "medium/mmlu"]          # sorted, not ranked
    assert list(r["by_tier_and_task_type (counts and intervals; alphabetical, not ranked)"]) == \
           ["low/extraction", "medium/coding", "medium/qa"]
    assert set(r["non_comparable_slice"]["by_tier"]) == {"low", "medium"}
    assert r["non_comparable_slice"]["by_tier"]["medium"]["fp"] == 1
    m = r["by_tier_and_truth_source"]["medium/exact"]
    assert m["catch_rate_ci"] == rep.wilson(1, 1) and m["false_alarm_ci"] is None


def test_flip_rate():
    main = [row("medium", "exact", True, True, pid="a"), row("medium", "exact", False, False, pid="b"),
            row("medium", "exact", False, False, pid="c"), row("medium", "exact", True, None, state="error", pid="d")]
    again = [row("medium", "exact", True, False, pid="a"), row("medium", "exact", False, False, pid="b"),
             row("medium", "exact", False, None, state="error", pid="c"), row("medium", "exact", True, True, pid="d")]
    f = rep.flip_rate(main, again)
    assert (f["n_repeated"], f["n_with_both_verdicts"], f["flips"], f["error_in_either_run"]) == (4, 2, 1, 2)
    assert f["flip_rate"] == 0.5


def test_model_comparison_uses_exact_match_items_only(metrics):
    small = [row("medium", "exact", True, False, pid="a"), row("medium", "exact", False, False, pid="b"),
             row("medium", "judged", True, True, pid="c")]
    large = [row("medium", "exact", True, True, pid="a"), row("medium", "exact", False, False, pid="b"),
             row("medium", "judged", True, False, pid="c")]
    c = rep.compare_models(small, large, "groq-small", "groq-large")
    assert c["medium"]["n"] == 2 and c["low"] == {"n": 0}
    assert c["medium"]["only_groq-large_right"] == 1 and c["medium"]["only_groq-small_right"] == 0
    assert c["medium"]["verdicts_agree"] == 0.5


def test_build_report_end_to_end_with_fake_runs(tmp_path, metrics):
    sample_path = write_sample(tmp_path)
    for model in ("groq-small", "groq-large"):
        run(tmp_path, FakeSend({"p0": '{"good": false, "reason": "bad"}'}), model=model, sample_path=sample_path)
        run(tmp_path, FakeSend(), model=model, sample_path=sample_path, repeat=True)
    report = rep.build_report(sample_path=sample_path, calib_dir=tmp_path, settings=SETTINGS)
    assert report["prompt_sha256"] == prompt_sha256() and set(report["models"]) == {"groq-small", "groq-large"}
    m = report["models"]["groq-small"]["by_tier (sample, unweighted)"]["medium"]
    assert (m["tp"], m["fn"], m["tn"], m["fp"]) == (1, 1, 2, 0)
    assert report["models"]["groq-small"]["repeat_runs"]["n_with_both_verdicts"] == 2
    assert report["groq-small vs groq-large (exact-match items only)"]["medium"]["n"] == 4


def test_results_from_another_prompt_are_not_counted(tmp_path, metrics, monkeypatch):
    sample_path = write_sample(tmp_path)
    run(tmp_path, FakeSend(), sample_path=sample_path)
    monkeypatch.setattr(rep, "prompt_sha256", lambda: "a different prompt")
    report = rep.build_report(models=["groq-small"], sample_path=sample_path, calib_dir=tmp_path, settings=SETTINGS)
    assert report["models"]["groq-small"]["errors"]["all"]["missing"] == 4
