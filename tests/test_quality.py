import pytest

from app.router.quality import (QualityError, QualitySettings, choose_tier, load_quality_settings, resolve_tau,
                                route)

SETTINGS = QualitySettings(default="balanced", presets={"cheap": 0.4, "balanced": 0.6, "high": 0.8})


# ---------- choose_tier ----------

def test_cheapest_tier_that_clears_tau():
    assert choose_tier(0.62, 0.91, 0.6) == "low"
    assert choose_tier(0.62, 0.91, 0.8) == "medium"
    assert choose_tier(0.62, 0.91, 0.95) == "high"


def test_exactly_tau_counts_as_clearing_it():
    assert choose_tier(0.6, 0.0, 0.6) == "low"


def test_extremes():
    assert choose_tier(0.0, 0.0, 0.0) == "low"      # tau 0: everything goes cheap
    assert choose_tier(0.99, 0.99, 1.0) == "high"   # tau 1: everything goes strong


def test_higher_tau_never_picks_a_cheaper_tier():
    order = {"low": 0, "medium": 1, "high": 2}
    taus = [i / 20 for i in range(21)]
    tiers = [order[choose_tier(0.55, 0.85, t)] for t in taus]
    assert tiers == sorted(tiers)


def test_bad_tau_is_rejected():
    with pytest.raises(QualityError):
        choose_tier(0.5, 0.5, 1.5)


def test_route_explains_itself():
    decision = route(0.3, 0.9, 0.6)
    assert decision.tier == "medium"
    assert "medium 0.90" in decision.reason


# ---------- resolve_tau ----------

@pytest.mark.parametrize("header, tau", [
    (None, 0.6), ("", 0.6), ("  ", 0.6),            # missing -> default (balanced)
    ("cheap", 0.4), ("HIGH", 0.8), (" Balanced ", 0.6),
    ("0.75", 0.75), ("0", 0.0), ("1", 1.0), (0.3, 0.3),
])
def test_resolve_tau(header, tau):
    assert resolve_tau(header, SETTINGS) == tau


@pytest.mark.parametrize("header", ["banana", "1.7", "-0.1", "nan", "inf", "60%"])
def test_resolve_tau_rejects_nonsense(header):
    with pytest.raises(QualityError):
        resolve_tau(header, SETTINGS)


def test_numeric_default():
    assert resolve_tau(None, QualitySettings(default="0.7", presets={"cheap": 0.4})) == 0.7


# ---------- config ----------

def test_load_settings(tmp_path):
    path = tmp_path / "routing.yaml"
    path.write_text("quality:\n  default: high\n  presets: {cheap: 0.4, balanced: 0.6, high: 0.8}\n")
    settings = load_quality_settings(path)
    assert resolve_tau(None, settings) == 0.8


@pytest.mark.parametrize("yaml_text", [
    "other: 1\n",                                                        # no quality section
    "quality:\n  default: balanced\n  presets: {cheap: 1.4}\n",          # preset out of range
    "quality:\n  default: turbo\n  presets: {cheap: 0.4}\n",             # default isn't a preset or number
])
def test_bad_config_fails_at_startup(tmp_path, yaml_text):
    path = tmp_path / "routing.yaml"
    path.write_text(yaml_text)
    with pytest.raises(QualityError):
        load_quality_settings(path)