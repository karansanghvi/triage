"""
quality.py: Triage's quality dial (tau) and routing rule.
 
Routing rule: send the request to the cheapest tier whose predicted pass probability is at least tau.
    low (cheap) first, then medium; if neither clears tau, high (strong) is the fallback.
 
    tau near 0: aggressive, almost everything goes to cheap (lowest cost, most risk)
    tau near 1: cautious, almost everything goes to strong (highest cost, highest quality)
 
Each request can choose tau with the header  X-Triage-Quality: cheap | balanced | high | <number 0-1>.
Without the header, the default from config/routing.yaml is used.
 
The same choose_tier() is used by the live router AND by eval/pareto.py, so the Pareto chart measures
exactly what Triage does in production.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import yaml

from app.registry import PROJECT_ROOT

ROUTING_CONFIG_PATH = PROJECT_ROOT / "config" / "routing.yaml"
QUALITY_HEADER = "X-Triage-Quality"
TIERS = ("low", "medium", "high") # cheapest first 

class QualityError(ValueError):
     """A quality setting that isn't a known preset or a number from 0 to 1 (an API would return 400)."""


# ---------- Settings ----------
 
@dataclass(frozen=True)
class QualitySettings:
    default: str                 # a preset name, or a number written as text, e.g. "0.7"
    presets: dict                # {"cheap": 0.4, "balanced": 0.6, "high": 0.8}
    

def _check_tau(value: float, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
        or not 0.0 <= value <= 1.0:
            raise QualityError(f"{where}: must be a number from 0 to 1, got {value!r}")
    return float(value)

def load_quality_settings(path: Path = ROUTING_CONFIG_PATH) -> QualitySettings:
    """Read the `quality:` section of routing.yaml and validate it, so a bad config fails at startup"""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return quality_settings_from_dict(config, path)


def quality_settings_from_dict(config: dict, path="routing.yaml") -> QualitySettings:
    """Validate the `quality:` section of an already-parsed routing.yaml (also used by live config updates)"""
    section = (config or {}).get("quality")
    if not isinstance(section, dict):
        raise QualityError(f"{path}: missing a `quality:` section")
    presets = section.get("presets") or {}
    if not isinstance(presets, dict) or not presets:
        raise QualityError(f"{path}: quality.presets must map names to numbers")
    presets = {str(name).lower():_check_tau(v, f"quality.presets.{name}") for name, v in presets.items()}
    settings = QualitySettings(default=str(section.get("default", "balanced")).lower(), presets=presets)
    resolve_tau(None, settings) # the default itself must be valid 
    return settings


# ---------- Header -> tau ----------

def resolve_tau(value: Optional[Union[str, float]], settings: QualitySettings) -> float:
    """Turn a header value (or None) into tau.
 
    None or blank -> the configured default. A preset name (any case) -> its number.
    A number from 0 to 1 (as text or a float) -> that number. Anything else -> QualityError.
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        value = settings.default
    if isinstance(value, str):
        text = value.strip().lower()
        if text in settings.presets:
            return settings.presets[text]
        try:
            value = float(text)
        except ValueError:
            options = " | ".join(settings.presets)
            raise QualityError(f"{QUALITY_HEADER} must be {options} or a number from 0 to 1, got {text!r}") \
                from None
    return _check_tau(value, QUALITY_HEADER)


# ---------- The routing rule ----------

@dataclass(frozen=True)
class RoutingDecision:
    tier: str                     # low, medium or high
    tau: float
    p_low: float
    p_medium: float
    
    @property
    def reason(self) -> str:
        # 3 decimals: at 2, a probability just under tau could read the same as tau ("medium 0.86 < tau 0.86")
        if self.tier == "low":
            return f"cheap pass chance {self.p_low:.3f} >= tau {self.tau:.3f}"
        if self.tier == "medium":
            return f"cheap {self.p_low:.3f} < tau {self.tau:.3f}; medium {self.p_medium:.3f} >= tau"
        return f"cheap {self.p_low:.3f} and medium {self.p_medium:.3f} both < tau {self.tau:.3f} using strong"

def choose_tier(p_low: float, p_medium: float, tau: float) -> str:
    """The cheapest tier whose predicted pass probability is at least tau; "high" if none is"""
    tau = _check_tau(tau, "tau")
    if p_low >= tau:
        return "low"
    if p_medium >= tau:
        return "medium"
    return "high"

def route(p_low: float, p_medium: float, tau: float) -> RoutingDecision:
    """choose_tier() plus the numbers behind it, for logs and response headers"""
    return RoutingDecision(tier=choose_tier(p_low, p_medium, tau), tau=float(tau), p_low=float(p_low), p_medium=float(p_medium))