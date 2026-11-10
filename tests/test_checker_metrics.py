"""checker_metrics (Phase 4 Step 2). These fail until eval/checker_metrics.py is written by hand.

"Positive" = the checker said not good (checker_bad True).
"""
import pytest

from eval.checker_metrics import checker_metrics

KEYS = {"n", "tp", "fp", "tn", "fn", "catch_rate", "trust_rate", "false_alarm"}


def rows(tp=0, fp=0, tn=0, fn=0):
    return ([{"truth_bad": True, "checker_bad": True}] * tp + [{"truth_bad": False, "checker_bad": True}] * fp
            + [{"truth_bad": False, "checker_bad": False}] * tn + [{"truth_bad": True, "checker_bad": False}] * fn)


def test_returns_exactly_the_documented_keys():
    assert set(checker_metrics(rows(1, 1, 1, 1))) == KEYS


def test_known_counts():
    m = checker_metrics(rows(tp=30, fp=10, tn=50, fn=10))
    assert (m["n"], m["tp"], m["fp"], m["tn"], m["fn"]) == (100, 30, 10, 50, 10)
    assert m["catch_rate"] == pytest.approx(30 / 40)       # tp / (tp + fn)
    assert m["trust_rate"] == pytest.approx(30 / 40)       # tp / (tp + fp)
    assert m["false_alarm"] == pytest.approx(10 / 60)      # fp / (fp + tn)


def test_rates_are_not_mixed_up():
    m = checker_metrics(rows(tp=6, fp=2, tn=9, fn=3))
    assert m["catch_rate"] == pytest.approx(6 / 9)
    assert m["trust_rate"] == pytest.approx(6 / 8)
    assert m["false_alarm"] == pytest.approx(2 / 11)


def test_counts_do_not_depend_on_row_order():
    data = rows(tp=3, fp=1, tn=4, fn=2)
    assert checker_metrics(data) == checker_metrics(list(reversed(data)))


def test_accepts_any_iterable_and_extra_fields():
    data = [{**r, "prompt_id": f"p{i}", "tier": "medium"} for i, r in enumerate(rows(tp=2, tn=3))]
    m = checker_metrics(iter(data))
    assert (m["n"], m["tp"], m["tn"]) == (5, 2, 3)


def test_empty_input_gives_zero_counts_and_no_rates():
    assert checker_metrics([]) == {"n": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0,
                                   "catch_rate": None, "trust_rate": None, "false_alarm": None}


def test_all_truly_good_answers():
    m = checker_metrics(rows(fp=2, tn=8))
    assert m["catch_rate"] is None                         # no truly bad answers: tp + fn = 0
    assert m["false_alarm"] == pytest.approx(0.2)
    assert m["trust_rate"] == 0.0                          # flagged 2, none truly bad


def test_all_truly_bad_answers():
    m = checker_metrics(rows(tp=7, fn=3))
    assert m["catch_rate"] == pytest.approx(0.7)
    assert m["trust_rate"] == 1.0
    assert m["false_alarm"] is None                        # no truly good answers: fp + tn = 0


def test_checker_never_says_not_good():
    m = checker_metrics(rows(tn=5, fn=4))
    assert m["trust_rate"] is None                         # nothing flagged: tp + fp = 0
    assert m["catch_rate"] == 0.0 and m["false_alarm"] == 0.0


def test_checker_always_says_not_good():
    m = checker_metrics(rows(tp=4, fp=6))
    assert m["catch_rate"] == 1.0 and m["false_alarm"] == 1.0
    assert m["trust_rate"] == pytest.approx(0.4)


def test_zero_rates_are_zero_not_none():
    m = checker_metrics(rows(fp=3, fn=2))                  # wrong on everything
    assert m["catch_rate"] == 0.0 and m["trust_rate"] == 0.0 and m["false_alarm"] == 1.0


def test_perfect_checker():
    m = checker_metrics(rows(tp=5, tn=5))
    assert (m["catch_rate"], m["trust_rate"], m["false_alarm"]) == (1.0, 1.0, 0.0)


def test_single_rows():
    assert checker_metrics(rows(tp=1))["tp"] == 1
    assert checker_metrics(rows(fp=1))["fp"] == 1
    assert checker_metrics(rows(tn=1))["tn"] == 1
    assert checker_metrics(rows(fn=1))["fn"] == 1


def test_counts_are_ints_and_rates_are_floats_or_none():
    m = checker_metrics(rows(tp=1, fp=1, tn=1, fn=1))
    assert all(type(m[k]) is int for k in ("n", "tp", "fp", "tn", "fn"))
    assert all(isinstance(m[k], float) for k in ("catch_rate", "trust_rate", "false_alarm"))
