"""Phase 4 Step 0: per-tier latency from the Phase 2 answers, for setting live timeouts.

Usage:
    python -m eval.latency_report

Reads data/answers/answers.jsonl (newest record per (prompt_id, tier), like routing_sim), keeps prompts in
the TRAIN split only (data/splits/train_ids.txt; the test set is never read), and drops cache hits (no real
call was timed) and errors (no answer). Truncated answers are kept: they were real calls, and they are the
slowest ones, which is exactly what a timeout has to allow for. Writes reports/latency_p95.json only.

Caveat: Phase 2 ran all tiers at once on one machine, with reasoning_effort "low" and max_tokens 4000 (the
same settings live calls now use). Local-model (ollama) latencies include that contention.
"""
import json
from datetime import datetime, timezone

import numpy as np

from app.registry import PROJECT_ROOT
from eval.build_dataset import FULL_DIR
from eval.score_answers import load_answers
from eval.splits import load_split

REPORT_PATH = PROJECT_ROOT / "reports" / "latency_p95.json"
TIER_ORDER = ("low", "medium", "high", "extra")


def latency_stats(records: list) -> dict:
    """Latency percentiles for one tier's newest records (cache hits and errors are counted, then dropped)."""
    timed = [r for r in records if not r.get("error") and not r.get("cache_hit") and r.get("latency_ms")]
    ms = np.array([r["latency_ms"] for r in timed], dtype=float)
    untruncated = np.array([r["latency_ms"] for r in timed if not r.get("truncated")], dtype=float)
    pct = lambda a, q: round(float(np.percentile(a, q)), 1) if len(a) else None
    return {
        "models": sorted({r["model_name"] for r in records}),
        "records": len(records),
        "excluded_cache_hits": sum(bool(r.get("cache_hit")) for r in records),
        "excluded_errors": sum(bool(r.get("error")) for r in records),
        "n": int(len(ms)),
        "truncated_included": sum(bool(r.get("truncated")) for r in timed),
        "p50_ms": pct(ms, 50), "p90_ms": pct(ms, 90), "p95_ms": pct(ms, 95), "p99_ms": pct(ms, 99),
        "max_ms": round(float(ms.max()), 1) if len(ms) else None,
        "p95_ms_without_truncated": pct(untruncated, 95),
        "share_over_p95_rounded_plus_1s": (round(float(np.mean(ms > (np.ceil(np.percentile(ms, 95) / 1000) + 1) * 1000)), 4)
                                           if len(ms) else None),
    }


def build_report(answers: dict, train_ids: set) -> dict:
    by_tier = {}
    for (pid, tier), record in answers.items():
        if pid in train_ids:
            by_tier.setdefault(tier, []).append(record)
    tiers = {t: latency_stats(by_tier[t]) for t in TIER_ORDER if t in by_tier}
    return {
        "source": "data/answers/answers.jsonl (newest record per prompt and tier)",
        "split": "train (data/splits/train_ids.txt); test never read",
        "excluded": "cache_hit == true, error != null",
        "percentile_method": "numpy linear interpolation",
        "created": datetime.now(timezone.utc).isoformat(),
        "tiers": tiers,
    }


def main():
    train_ids, _ = load_split()
    report = build_report(load_answers(FULL_DIR / "answers.jsonl"), train_ids)
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{'tier':<8}{'model':<14}{'n':>5}{'cache':>7}{'err':>5}{'trunc':>7}"
          f"{'p50':>9}{'p90':>9}{'p95':>9}{'p99':>9}{'max':>10}   (ms)")
    for tier, s in report["tiers"].items():
        print(f"{tier:<8}{','.join(s['models']):<14}{s['n']:>5}{s['excluded_cache_hits']:>7}{s['excluded_errors']:>5}"
              f"{s['truncated_included']:>7}{s['p50_ms']:>9.0f}{s['p90_ms']:>9.0f}{s['p95_ms']:>9.0f}"
              f"{s['p99_ms']:>9.0f}{s['max_ms']:>10.0f}")
    print(f"\nsaved {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
