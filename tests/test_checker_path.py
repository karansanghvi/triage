"""Strategy (a)'s path, medium_then_check_path (Phase 4 Step 3). These fail until eval/checker_path.py is
written by hand.

medium answer -> format check (free) -> AI check (paid) -> high if bad; "unknown" follows on_error.
"""
import pytest

from eval.checker_path import ANSWER, CHECK, Call, medium_then_check_path
from eval.checker_routing import score_decisions
from eval.checker_verdicts import Verdict
from eval.routing_sim import EvalRow

A_MED, C_MED, A_HIGH = Call(ANSWER, "medium"), Call(CHECK, "medium"), Call(ANSWER, "high")


def row(pid="p1", medium_pass=True):
    return EvalRow(id=pid, task_type="qa", source="gsm8k", label="medium" if medium_pass else "high",
                   passed={"low": False, "medium": medium_pass, "high": True},
                   cost={"low": 0.0001, "medium": 0.001, "high": 0.01},
                   latency_ms={"low": 100.0, "medium": 400.0, "high": 2000.0})


def verdicts(status, pid="p1", why=None, cost=0.0002, latency=300.0):
    why = why or ("ok" if status != "unknown" else "error:invalid_json")
    return {(pid, "medium"): Verdict(status, why, cost, latency)}


FORMAT_OK = {("p1", "medium"): True}
FORMAT_FAIL = {("p1", "medium"): False}


# ---------- the AI checker's verdict ----------

def test_good_verdict_stops_at_medium():
    assert medium_then_check_path(row(), verdicts("good"), FORMAT_OK) == ("medium", [A_MED, C_MED])


def test_bad_verdict_escalates_to_high():
    assert medium_then_check_path(row(), verdicts("bad"), FORMAT_OK) == ("high", [A_MED, C_MED, A_HIGH])


@pytest.mark.parametrize("why", ["error:invalid_json", "error:provider:timeout (unfinished)", "missing"])
def test_unknown_verdict_escalates_when_on_error_is_escalate(why):
    got = medium_then_check_path(row(), verdicts("unknown", why=why), FORMAT_OK, on_error="escalate")
    assert got == ("high", [A_MED, C_MED, A_HIGH])                 # the failed check is still paid for


@pytest.mark.parametrize("why", ["error:truncated", "missing"])
def test_unknown_verdict_keeps_medium_when_on_error_is_accept(why):
    got = medium_then_check_path(row(), verdicts("unknown", why=why), FORMAT_OK, on_error="accept")
    assert got == ("medium", [A_MED, C_MED])


def test_default_on_error_is_escalate():
    assert medium_then_check_path(row(), verdicts("unknown"), FORMAT_OK)[0] == "high"


def test_absent_verdict_counts_as_unknown():
    assert medium_then_check_path(row(), {}, FORMAT_OK, on_error="escalate") == ("high", [A_MED, C_MED, A_HIGH])
    assert medium_then_check_path(row(), {}, FORMAT_OK, on_error="accept") == ("medium", [A_MED, C_MED])


def test_on_error_only_matters_for_unknown_verdicts():
    for on_error in ("escalate", "accept"):
        assert medium_then_check_path(row(), verdicts("good"), FORMAT_OK, on_error)[0] == "medium"
        assert medium_then_check_path(row(), verdicts("bad"), FORMAT_OK, on_error)[0] == "high"


def test_bad_on_error_value_raises():
    with pytest.raises(ValueError):
        medium_then_check_path(row(), verdicts("good"), FORMAT_OK, on_error="retry")


# ---------- the free format check comes first ----------

@pytest.mark.parametrize("status", ["good", "bad", "unknown"])
@pytest.mark.parametrize("on_error", ["escalate", "accept"])
def test_format_failure_escalates_without_calling_the_ai_checker(status, on_error):
    got = medium_then_check_path(row(), verdicts(status), FORMAT_FAIL, on_error)
    assert got == ("high", [A_MED, A_HIGH])                        # no Call(CHECK, ...) at all


def test_format_failure_needs_no_verdict():
    assert medium_then_check_path(row(), {}, FORMAT_FAIL, on_error="accept") == ("high", [A_MED, A_HIGH])


# ---------- shape of the result ----------

def test_calls_are_call_tuples_in_order_and_high_is_never_checked():
    final, calls = medium_then_check_path(row(), verdicts("bad"), FORMAT_OK)
    assert all(isinstance(c, Call) for c in calls)
    assert calls[0] == A_MED and calls[-1] == A_HIGH
    assert Call(CHECK, "high") not in calls and isinstance(calls, list)


def test_uses_the_row_id_to_find_its_own_verdict():
    vs = {**verdicts("bad", pid="p1"), **verdicts("good", pid="p2")}
    fmt = {("p1", "medium"): True, ("p2", "medium"): True}
    assert medium_then_check_path(row("p1"), vs, fmt)[0] == "high"
    assert medium_then_check_path(row("p2"), vs, fmt)[0] == "medium"


def test_never_looks_at_the_truth():
    for status in ("good", "bad", "unknown"):
        assert medium_then_check_path(row(medium_pass=True), verdicts(status), FORMAT_OK) == \
               medium_then_check_path(row(medium_pass=False), verdicts(status), FORMAT_OK)


# ---------- costs include the check ----------

def test_cost_and_latency_include_the_check_when_it_stops_at_medium():
    r, vs = row(), verdicts("good", cost=0.0002, latency=300.0)
    s = score_decisions([r], [medium_then_check_path(r, vs, FORMAT_OK)], vs)
    assert s["cost_per_1k"] == pytest.approx((0.001 + 0.0002) * 1000)
    assert s["check_cost_per_1k"] == pytest.approx(0.0002 * 1000)
    assert s["latency_p50_ms"] == pytest.approx(400 + 300)
    assert s["checks_per_request"] == 1 and s["escalation_rate"] == 0


def test_cost_and_latency_add_up_on_escalation():
    r, vs = row(), verdicts("bad", cost=0.0002, latency=300.0)
    s = score_decisions([r], [medium_then_check_path(r, vs, FORMAT_OK)], vs)
    assert s["cost_per_1k"] == pytest.approx((0.001 + 0.0002 + 0.01) * 1000)
    assert s["latency_p50_ms"] == pytest.approx(400 + 300 + 2000)
    assert s["mix"]["high"] == 1 and s["escalation_rate"] == 1


def test_a_failed_check_is_still_charged():
    r, vs = row(), verdicts("unknown", cost=0.00015, latency=250.0)
    s = score_decisions([r], [medium_then_check_path(r, vs, FORMAT_OK, on_error="accept")], vs)
    assert s["cost_per_1k"] == pytest.approx((0.001 + 0.00015) * 1000)


def test_format_failure_costs_no_check():
    r, vs = row(), verdicts("good", cost=0.0002)
    s = score_decisions([r], [medium_then_check_path(r, vs, FORMAT_FAIL)], vs)
    assert s["cost_per_1k"] == pytest.approx((0.001 + 0.01) * 1000)
    assert s["check_cost_per_1k"] == 0 and s["checks_per_request"] == 0
