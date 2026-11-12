"""Price-gap sensitivity helpers (eval/price_sensitivity.py). No data files, no API calls."""
from types import SimpleNamespace

from eval.price_sensitivity import beats_always_medium, cheapest_at, correct, scale_high


def pt(right, cost, n=10, setting=None):
    return {"quality": right / n, "n": n, "cost_per_1k": cost, "setting": setting}


def test_scale_high_only_changes_the_high_tier_and_leaves_the_input_alone():
    row = SimpleNamespace(id="p", cost={"low": 1.0, "medium": 2.0, "high": 4.0, "extra": 0.5})
    (scaled,) = scale_high([row], 2.5)
    assert scaled.cost == {"low": 1.0, "medium": 2.0, "high": 10.0, "extra": 0.5}
    assert row.cost["high"] == 4.0


def test_cheapest_at_picks_the_cheapest_point_that_reaches_the_level():
    pts = [pt(9, 5.0, setting="a"), pt(8, 1.0, setting="b"), pt(10, 7.0, setting="c"), pt(9, 4.0, setting="d")]
    assert cheapest_at(pts, 9)["setting"] == "d"
    assert cheapest_at(pts, 8)["setting"] == "b"
    assert cheapest_at(pts, 11) is None
    assert correct(pts[0]) == 9


def test_beats_always_medium_needs_more_right_at_no_more_cost_or_same_right_for_less():
    fams = {"always_medium": [pt(8, 2.0)],
            "x": [pt(9, 2.0, setting="same cost, better"), pt(8, 1.5, setting="same right, cheaper"),
                  pt(9, 2.1, setting="better but pricier"), pt(8, 2.0, setting="a tie")]}
    wins = {w["setting"] for w in beats_always_medium(fams)}
    assert wins == {"same cost, better", "same right, cheaper"}
