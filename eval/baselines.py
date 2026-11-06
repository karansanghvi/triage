"""Phase 3 Step 4: the simple strategies Triage has to beat.

    always_low / always_medium / always_high   every prompt to one tier
    oracle                                     each prompt to its label (cheapest tier that passed):
                                               the most any router could save; nothing real beats it
    random_same_mix                            random tiers in the same proportions as a given router,
                                               so it spends about the same; averaged over many seeds

Usage:
    python -m eval.baselines                 # on the train split (safe, nothing is learned here)
    python -m eval.baselines --split test    # final evaluation only

RouteLLM is added separately (it needs its own install).
"""
import argparse

import numpy as np

from eval.routing_sim import TIERS, load_eval_rows, score


def always(rows: list, tier: str) -> list:
    return [tier] * len(rows)


def oracle(rows: list) -> list:
    return [r.label for r in rows]


def random_same_mix(rows: list, mix: dict, repeats: int = 1000, seed: int = 42) -> dict:
    """Average score of random routing that sends the same share of prompts to each tier as `mix`.

    The exact counts are kept (e.g. 30% low, 55% medium, 15% high); only WHICH prompts go where is
    random. If a router can't beat this at the same mix, its predictions aren't adding anything.
    """
    rng = np.random.default_rng(seed)
    n = len(rows)
    counts = {t: int(round(mix.get(t, 0) * n)) for t in TIERS}
    counts["high"] = n - counts["low"] - counts["medium"]   # rounding leftovers go to high
    base = [t for t in TIERS for _ in range(counts[t])]
    runs = [score(rows, list(rng.permutation(base))) for _ in range(repeats)]
    keys = ["quality", "quality_retained", "cost_per_1k", "cost_saved", "latency_p50_ms", "latency_p95_ms"]
    out = {k: float(np.mean([r[k] for r in runs])) for k in keys}
    out["quality_std"] = float(np.std([r["quality"] for r in runs]))
    out["n"], out["mix"] = n, {t: counts[t] / n for t in TIERS}
    return out


def all_baselines(rows: list) -> dict:
    results = {f"always_{t}": score(rows, always(rows, t)) for t in TIERS}
    results["oracle"] = score(rows, oracle(rows))
    # Example random mix: the oracle's own mix, i.e. "spend like a perfect router, but choose blindly"
    results["random_oracle_mix"] = random_same_mix(rows, results["oracle"]["mix"])
    return results


def print_table(results: dict) -> None:
    print(f"{'strategy':<20}{'quality':>9}{'retained':>10}{'$/1k req':>10}{'saved':>8}"
          f"{'p50 ms':>9}{'p95 ms':>9}   mix (low/med/high)")
    for name, r in results.items():
        mix = "/".join(f"{r['mix'][t]:.0%}" for t in TIERS)
        print(f"{name:<20}{r['quality']:>9.1%}{r['quality_retained']:>10.1%}{r['cost_per_1k']:>10.4f}"
              f"{r['cost_saved']:>8.0%}{r['latency_p50_ms'] or 0:>9.0f}{r['latency_p95_ms'] or 0:>9.0f}   {mix}")


def main():
    parser = argparse.ArgumentParser(description="Score the baseline routing strategies.")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    args = parser.parse_args()
    rows = load_eval_rows(args.split, allow_test=args.split == "test")
    print(f"{len(rows)} labeled {args.split} prompts; cost at Groq paid prices (price_as), no API calls\n")
    print_table(all_baselines(rows))


if __name__ == "__main__":
    main()