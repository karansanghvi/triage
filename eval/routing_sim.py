"""Routing simulator: score any routing strategy with the real Phase 2 results. No API calls.
 
Every labeled prompt already has every tier's answer, its pass/fail, its realistic paid cost
(price_as_cost: tokens x the model's published paid price, from config/models.yaml) and its latency.
So "what if this prompt had gone to tier X?" can be answered by looking it up.
 
baselines.py and pareto.py both score strategies with score(), so every strategy, Triage included,
is measured exactly the same way.
 
The test split can only be loaded with allow_test=True. That flag belongs in the FINAL evaluation
only; tuning (choosing tau presets, picking classifier B or C) must use the train split.
"""

import json
from dataclasses import dataclass, field

import numpy as np

from app.registry import PROJECT_ROOT
from eval.build_dataset import FULL_DIR
from eval.score_answers import load_answers
from eval.splits import load_split

OUTCOMES_PATH = PROJECT_ROOT / "data" / "outcomes.jsonl"
TIERS = ("low", "medium", "high")


@dataclass
class EvalRow:
    id: str
    task_type: str
    source: str
    label: str              # cheapest tier that passed ("high" if none did)
    passed: dict = field(default_factory=dict)  # tier -> bool
    cost: dict = field(default_factory=dict)    # tier -> dollars at paid prices 
    latency_ms: dict = field(default_factory=dict)  # tier -> milliseconds (None if unknown)


def load_eval_rows(split: str = "train", allow_test: bool = False) -> list:
    """Labeled prompts in one split, with every tier's pass/fail, cost and latency"""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    if split == "test" and not allow_test:
        raise PermissionError("the test split is for the final evaluation only; pass allow_test=True there")
    train_ids, test_ids = load_split()
    wanted = train_ids if split == "train" else test_ids
    answers = load_answers(FULL_DIR / "answers.jsonl")
    
    rows, missing = [], 0
    for line in OUTCOMES_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o["label_status"] != "labeled" or o["id"] not in wanted:
            continue
        row = EvalRow(id=o["id"], task_type=o.get("task_type", ""), source=o.get("source", ""), label=o["label"]) 
        for tier in TIERS:
            row.passed[tier] = bool(o.get(f"{tier}_pass"))
            answer = answers.get((o["id"], tier)) or {}
            cost = answer.get("price_as_cost")
            if cost is None:
                missing += 1
            row.cost[tier] = float(cost or 0.0)
            row.latency_ms[tier] = answer.get("latency_ms")
        rows.append(row)
    if missing:
        print(f"note: {missing} (prompt, tier) answers had no price_as_count; counted as $0")
    return sorted(rows, key=lambda r: r.id)


def score(rows: list, tiers: list) -> dict:
    """Quality, cost and latency if prompt i is sent to tiers[i]."""
    if len(rows) != len(tiers):
        raise ValueError("one time per row")
    passed = np.array([r.passed[t] for r, t in zip(rows, tiers)], dtype=float)
    cost = np.array([r.cost[t] for r, t in zip(rows, tiers)])
    latency = np.array([r.latency_ms[t] for r, t in zip(rows, tiers) if r.latency_ms[t] is not None])
    strong_quality = np.mean([r.passed["high"] for r in rows])
    strong_cost = sum(r.cost["high"] for r in rows)
    return {
        "n": len(rows),
        "quality": float(passed.mean()),                                   # share of answers that passed
        "quality_retained": float(passed.mean() / strong_quality) if strong_quality else None,
        "cost_per_1k": float(cost.mean() * 1000),                          # dollars per 1,000 requests
        "cost_saved": float(1 - cost.sum() / strong_cost) if strong_cost else None,   # vs always-strong
        "latency_p50_ms": float(np.percentile(latency, 50)) if len(latency) else None,
        "latency_p95_ms": float(np.percentile(latency, 95)) if len(latency) else None,
        "mix": {t: float(np.mean([x == t for x in tiers])) for t in TIERS},  # share sent to each tier
    }