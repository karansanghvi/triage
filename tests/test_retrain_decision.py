"""should_replace (Phase 4 Step 9). These fail until scripts/retrain_decision.py is written by hand.

Rule: replace if (a) quality no worse (within tolerance) AND cost_saved better by >= min_cost_gain, or
(b) quality better AND cost_saved no worse. Otherwise keep.
"""
import copy

import pytest

from scripts.retrain_decision import should_replace


def m(quality, cost):
    return {"quality_retained": quality, "cost_saved": cost}


BASE = m(0.95, 0.40)


def decide(candidate, current=BASE, **kw):
    decision, reason = should_replace(current, candidate, **kw)
    assert isinstance(decision, bool)
    assert isinstance(reason, str) and reason.strip() and "\n" not in reason
    return decision


# ---------- keep ----------

def test_a_tie_keeps():
    assert decide(m(0.95, 0.40)) is False


def test_a_cost_gain_below_min_cost_gain_keeps():
    assert decide(m(0.95, 0.405)) is False                      # +0.5 points < 1 point
    assert decide(m(0.95, 0.4099)) is False


def test_a_quality_drop_beyond_the_tolerance_keeps_even_with_a_big_cost_gain():
    assert decide(m(0.949, 0.60)) is False                      # default tolerance is 0
    assert decide(m(0.94, 0.60), quality_tolerance=0.005) is False


def test_better_quality_but_worse_cost_keeps():
    assert decide(m(0.97, 0.39)) is False


def test_worse_on_both_keeps():
    assert decide(m(0.94, 0.35)) is False


def test_worse_quality_and_same_cost_keeps():
    assert decide(m(0.94, 0.40)) is False


def test_same_quality_and_worse_cost_keeps():
    assert decide(m(0.95, 0.30)) is False


# ---------- replace: rule (a) ----------

def test_rule_a_same_quality_and_enough_cost_gain_replaces():
    assert decide(m(0.95, 0.42)) is True


def test_rule_a_cost_gain_of_exactly_min_cost_gain_replaces():
    assert decide(m(0.95, 0.50), current=m(0.95, 0.25), min_cost_gain=0.25) is True   # exact in binary


def test_rule_a_quality_drop_within_the_tolerance_still_replaces():
    assert decide(m(0.946, 0.45), quality_tolerance=0.005) is True
    assert decide(m(0.945, 0.45), current=m(0.95, 0.40), quality_tolerance=0.25) is True


def test_rule_a_respects_a_custom_min_cost_gain():
    assert decide(m(0.95, 0.43), min_cost_gain=0.05) is False
    assert decide(m(0.95, 0.46), min_cost_gain=0.05) is True


# ---------- replace: rule (b) ----------

def test_rule_b_better_quality_and_same_cost_replaces():
    assert decide(m(0.96, 0.40)) is True


def test_rule_b_better_quality_and_a_tiny_cost_gain_replaces():
    assert decide(m(0.96, 0.401)) is True                       # below min_cost_gain, but rule (b) applies


def test_rule_b_any_quality_gain_counts():
    assert decide(m(0.9501, 0.40)) is True


def test_better_on_both_replaces():
    assert decide(m(0.97, 0.50)) is True


# ---------- inputs ----------

@pytest.mark.parametrize("side", ["current", "candidate"])
@pytest.mark.parametrize("missing", ["quality_retained", "cost_saved"])
def test_a_missing_key_raises(side, missing):
    broken = {k: v for k, v in BASE.items() if k != missing}
    args = (broken, BASE) if side == "current" else (BASE, broken)
    with pytest.raises(KeyError):
        should_replace(*args)


def test_extra_keys_are_ignored_and_inputs_are_not_modified():
    current = {**BASE, "mix": {"low": 0.1}, "tau": 0.86}
    candidate = {**m(0.96, 0.40), "mix": {"low": 0.2}}
    before = copy.deepcopy((current, candidate))
    assert should_replace(current, candidate)[0] is True
    assert (current, candidate) == before


def test_the_reason_differs_between_replace_and_keep():
    _, replace_reason = should_replace(BASE, m(0.96, 0.40))
    _, keep_reason = should_replace(BASE, m(0.95, 0.40))
    assert replace_reason != keep_reason
