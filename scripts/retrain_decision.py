"""The replace-or-keep rule for retraining (Phase 4 Step 9). Used by scripts/retrain.py.

should_replace() is left for you to write; tests/test_retrain_decision.py fails until then.
"""


def should_replace(current: dict, candidate: dict, *, min_cost_gain: float = 0.01,
                   quality_tolerance: float = 0.0) -> tuple:
    """Should the candidate classifier replace the live one? Both are measured on the VALIDATION set at the
    `balanced` quality preset (eval/routing_sim.score, via scripts/retrain.evaluate_on_validation).

    current, candidate   {"quality_retained": float, "cost_saved": float}, as fractions (0.95 = 95%).
                         Extra keys are allowed and ignored. A missing key raises KeyError.
    min_cost_gain        how much more cost_saved a candidate must show to win on cost alone (0.01 = 1 point)
    quality_tolerance    how much quality_retained a candidate may lose and still count as "no worse"

    Replace only if the candidate is
      (a) no worse on quality_retained (candidate >= current - quality_tolerance)
          AND better on cost_saved by at least min_cost_gain (candidate - current >= min_cost_gain), OR
      (b) better on quality_retained (candidate > current)
          AND no worse on cost_saved (candidate >= current).
    Otherwise keep the current model (a tie keeps it: retraining must earn its place).

    Returns (decision, reason): decision True = replace, False = keep; reason is one line (no newline) saying
    which rule applied, or why neither did, e.g. "keep: quality 93.8% -> 93.1% is below the 0.0-point tolerance".
    The inputs are not modified.
    """
    EPS = 1e-12 # protects exactly equal comparisons from tiny float rounding errors 
    
    # Read all four values first: a missing key raises KeyError, as the tests expect 
    q_old, q_new = current["quality_retained"], candidate["quality_retained"]
    c_old, c_new = current["cost_saved"], candidate["cost_saved"]
    
    quality_ok = q_new >= q_old - quality_tolerance - EPS # no worse on quality 
    cost_gain = c_new - c_old
    
    # Rule (a): no worse on quality, and clearly cheaper 
    if quality_ok and cost_gain >= min_cost_gain - EPS:
        return True, (f"replace (rule a): cost saved {c_old:.1%} -> {c_new:.1%} "
                      f"(+{cost_gain * 100:.1f} points), quality {q_old:.1%} -> {q_new:.1%}")
    
    # Rule (b): better quality, and no more expensive 
    if q_new > q_old and c_new >= c_old - EPS:
        return True, (f"replace (rule b): quality {q_old:.1%} -> {q_new:.1%} "
                      f"at no worse cost ({c_old:.1%} -> {c_new:.1%} saved)")
    
    # Neither rule: keep the current model, and say why 
    if not quality_ok:
        return False, (f"keep: quality {q_old:.1%} -> {q_new:.1%} is below "
                       f"the {quality_tolerance * 100:.1f}-point tolerance")
    return False, (f"keep: cost gain {cost_gain * 100:+.1f} points is under the "
                   f"{min_cost_gain * 100:.1f}-point minimum, and quality isn't better at no worse cost")