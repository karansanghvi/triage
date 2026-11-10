"""Strategy (a)'s core path: medium answer, then verify, then high if needed (Phase 4 Step 3).

The function body is left for you to write; tests/test_checker_path.py fails until then.
eval/checker_routing.py scores what it returns and builds strategies (b) and (c) on top of it.
"""
from typing import NamedTuple

ANSWER, CHECK = "answer", "check"
ON_ERROR = ("escalate", "accept")


class Call(NamedTuple):
    """One paid call: kind ANSWER (that tier's Phase 2 answer) or CHECK (the AI checker on that tier's answer)."""
    kind: str
    tier: str


def medium_then_check_path(row, verdicts: dict, format_ok: dict, on_error: str = "escalate") -> tuple:
    """Strategy (a) for one prompt: answer with medium, verify that answer, escalate to high if it's bad.

    row        an eval.routing_sim.EvalRow (only row.id is needed; do NOT look at row.passed or row.label:
               a router can't see the truth)
    verdicts   {(prompt_id, tier): eval.checker_verdicts.Verdict}; Verdict.status is "good", "bad" or
               "unknown". A key that is absent counts as "unknown".
    format_ok  {(prompt_id, tier): bool}, the free format checker's result (app/verifier/checker.check_answer)
    on_error   what to do when the AI checker's verdict is "unknown" (an error or a missing check):
                 "escalate"  treat it as bad: also answer with high
                 "accept"    treat it as good: keep medium's answer
               anything else -> ValueError

    Order, as in app/verifier/pipeline.verify (cheapest check first):
      1. pay for medium's answer
      2. format check (free, not a Call): if it fails, escalate to high WITHOUT running the AI checker
      3. otherwise pay for the AI check on medium's answer and read verdicts[(row.id, "medium")]:
           good    -> stop at medium
           bad     -> escalate to high
           unknown -> follow on_error
      high's answer is final and is never checked.

    Returns (final_tier, calls):
      final_tier  "medium" or "high", the tier whose answer is served
      calls       list of Call, in the order they are made (they run one after another), e.g.
                  [Call(ANSWER, "medium"), Call(CHECK, "medium"), Call(ANSWER, "high")]
    Scoring (eval/checker_routing.score_decisions) charges each ANSWER at row.cost[tier] and each CHECK at
    verdicts[(row.id, tier)].cost, and adds up their latencies.
    """
    if on_error not in ON_ERROR:
      raise ValueError(f"on_error must be one of {ON_ERROR}, got {on_error!r}")
    
    key = (row.id, "medium")
    calls = [Call(ANSWER, "medium")]      # 1. always pay for medium's answer
    
    # 2. free format check: if it fails, escalate without paying for the AI checker
    if not format_ok[key]:
        calls.append(Call(ANSWER, "high"))
        return "high", calls

    # 3. pay for the AI check and read its verdict (missing key = unknown)
    calls.append(Call(CHECK, "medium"))
    verdict = verdicts.get(key)
    status = verdict.status if verdict is not None else "unknown"

    if status == "good":
        return "medium", calls                  # medium's answer is served

    if status == "bad" or on_error == "escalate":
        calls.append(Call(ANSWER, "high"))      # bad, or unknown treated as bad
        return "high", calls

    return "medium", calls                      # unknown, and on_error == "accept"
