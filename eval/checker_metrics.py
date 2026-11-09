"""Confusion counts and rates for the AI checker (Phase 4 Step 2). To be written by hand."""


def checker_metrics(rows) -> dict:
    """Confusion counts and rates for AI-checker verdicts against known truth.

    rows: an iterable of dicts, each with
        "truth_bad":   True if the answer is truly bad (exact-match fail, or a judged fail)
        "checker_bad": True if the checker said "not good"
    Rows where the checker returned an error must be excluded BEFORE calling this; every row here has a
    real verdict.

    "Positive" means the checker said not good:
        tp = checker said not good, answer truly bad
        fp = checker said not good, answer truly good
        tn = checker said good,     answer truly good
        fn = checker said good,     answer truly bad

    Returns a dict:
        n            number of rows
        tp, fp, tn, fn
        catch_rate   tp / (tp + fn)   of truly bad answers, the share the checker flagged (recall)
        trust_rate   tp / (tp + fp)   when the checker said not good, the share truly bad (precision)
        false_alarm  fp / (fp + tn)   of truly good answers, the share the checker flagged
    A rate whose denominator is 0 is None (never 0.0, never an exception).
    """
    raise NotImplementedError("checker_metrics is written by hand; see the docstring")
