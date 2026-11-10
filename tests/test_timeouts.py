"""Latency report and live timeouts (Phase 4 Step 0, Part E)."""
import json
import math
import random

import pytest

from app.providers.common import ProviderError
from app.registry import PROJECT_ROOT, ModelConfig, load_registry
from app.reliability import BreakerSettings, ReliableClient, RetryPolicy, RoutingConfig, load_routing
from eval.latency_report import REPORT_PATH, build_report, latency_stats


def rec(ms, tier="high", cache=False, error=None, truncated=False, model="groq-large"):
    return {"latency_ms": ms, "tier": tier, "cache_hit": cache, "error": error, "truncated": truncated,
            "model_name": model}


# ---------- latency_report ----------

def test_stats_drop_cache_hits_and_errors_but_keep_truncated():
    records = [rec(m) for m in range(100, 1100, 100)]                     # 100..1000 ms
    records += [rec(5, cache=True), rec(None, error={"kind": "server"}), rec(99999, truncated=True)]
    s = latency_stats(records)
    assert (s["excluded_cache_hits"], s["excluded_errors"]) == (1, 1)
    assert s["n"] == 11 and s["truncated_included"] == 1
    assert s["max_ms"] == 99999
    assert s["p50_ms"] == 600.0
    assert s["p95_ms_without_truncated"] < 1000


def test_report_uses_only_the_given_ids():
    answers = {("train_1", "high"): rec(1000), ("train_2", "high"): rec(2000), ("test_1", "high"): rec(50000)}
    report = build_report(answers, {"train_1", "train_2"})
    assert report["tiers"]["high"]["n"] == 2
    assert report["tiers"]["high"]["max_ms"] == 2000


def test_saved_report_is_train_only():
    report = json.loads(REPORT_PATH.read_text())
    assert report["split"].startswith("train")
    assert "cache_hit" in report["excluded"] and "error" in report["excluded"]


# ---------- routing.yaml timeouts follow the report ----------

@pytest.fixture(scope="module")
def routing():
    return load_routing(load_registry())


def seconds(ms):
    """Rounded up to whole seconds, plus 1 s of margin."""
    return math.ceil(ms / 1000) + 1


# Which percentile each model's timeout follows. groq-large deliberately uses p99 (see routing.yaml).
PERCENTILE = {"groq-large": "p99_ms", "ollama-llama": "p95_ms"}
# Models allowed a timeout without report data, each documented in routing.yaml.
NO_DATA_EXCEPTIONS = {"gemini-flash"}


def report_by_model():
    tiers = json.loads(REPORT_PATH.read_text())["tiers"]
    return {s["models"][0]: s for s in tiers.values() if len(s["models"]) == 1}


def test_default_timeout_is_groq_large_p95_rounded_up_plus_1s(routing):
    assert routing.timeout_seconds == seconds(report_by_model()["groq-large"]["p95_ms"])


def test_every_model_timeout_follows_the_report(routing):
    stats = report_by_model()
    for name, timeout in routing.model_timeouts.items():
        if name in NO_DATA_EXCEPTIONS:
            assert name not in stats, f"{name} now has latency data; set its timeout from the report"
            continue
        assert name in PERCENTILE, f"{name}: say which percentile its timeout follows"
        assert timeout == seconds(stats[name][PERCENTILE[name]]), name


def test_groq_large_uses_p99_and_no_phase2_call_exceeds_it(routing):
    high = report_by_model()["groq-large"]
    assert routing.timeout_for("groq-large") == seconds(high["p99_ms"]) == 10
    assert high["max_ms"] < routing.timeout_for("groq-large") * 1000


def test_ollama_and_gemini_timeouts(routing):
    assert routing.timeout_for("ollama-llama") == 32 > routing.timeout_seconds
    assert routing.timeout_for("gemini-flash") == 30


def test_default_timeout_covers_every_phase2_cheap_and_medium_call(routing):
    tiers = json.loads(REPORT_PATH.read_text())["tiers"]
    for tier in ("low", "medium"):
        assert tiers[tier]["max_ms"] < routing.timeout_seconds * 1000, tier


# ---------- per-model timeouts in ReliableClient ----------

def model(name, provider):
    return ModelConfig(name=name, provider=provider, model_id=name, tier="low", input_price=0, output_price=0)


REGISTRY = {"groq-allam": model("groq-allam", "groq"), "ollama-llama": model("ollama-llama", "ollama")}


def test_each_model_in_a_chain_gets_its_own_timeout():
    calls = []

    def send(messages, m, stream=False, **settings):
        calls.append((m.name, settings["timeout"]))
        if m.name == "groq-allam":
            raise ProviderError("groq", "timeout", "slow")
        return "ok"

    routing = RoutingConfig(chains={"low": ["groq-allam", "ollama-llama"]}, retry=RetryPolicy(max_attempts=1),
                            breaker=BreakerSettings(), timeout_seconds=8, model_timeouts={"ollama-llama": 32})
    ReliableClient(REGISTRY, routing, send_fn=send, sleep=lambda s: None, rng=random.Random(0)).send(
        [{"role": "user", "content": "hi"}], "low")
    assert calls == [("groq-allam", 8), ("ollama-llama", 32)]


def test_caller_timeout_still_wins():
    seen = []
    routing = RoutingConfig(chains={"low": ["ollama-llama"]}, retry=RetryPolicy(max_attempts=1),
                            breaker=BreakerSettings(), timeout_seconds=8, model_timeouts={"ollama-llama": 32})
    client = ReliableClient(REGISTRY, routing,
                            send_fn=lambda msgs, m, stream=False, **s: seen.append(s["timeout"]) or "ok")
    client.send([{"role": "user", "content": "hi"}], "low", timeout=180)
    assert seen == [180]


@pytest.mark.parametrize("extra, message", [
    ("model_timeouts: {nope: 5}\n", "unknown model"),
    ("model_timeouts: {ollama-llama: 0}\n", "above 0"),
    ("model_timeouts: [ollama-llama]\n", "mapping"),
])
def test_bad_model_timeouts_are_rejected(tmp_path, extra, message):
    path = tmp_path / "routing.yaml"
    path.write_text("fallback_chains:\n  low: [groq-allam]\n  medium: [groq-allam]\n  high: [ollama-llama]\n" + extra)
    with pytest.raises(ValueError, match=message):
        load_routing(REGISTRY, path)


def test_latency_script_writes_nothing_to_data(monkeypatch, tmp_path):
    from eval import latency_report
    data = PROJECT_ROOT / "data"
    snap = lambda: {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in data.rglob("*") if p.is_file()}
    before = snap()
    monkeypatch.setattr(latency_report, "REPORT_PATH", tmp_path / "latency.json")
    monkeypatch.setattr(latency_report, "PROJECT_ROOT", tmp_path)          # for the relative path it prints
    latency_report.main()
    assert snap() == before
    assert json.loads((tmp_path / "latency.json").read_text())["tiers"]["high"]["n"] > 0
