"""Estimate the token cost of LLM-judging the judge-scored prompts, without calling any model.

Usage:
    python -m eval.judge_cost [--low-judge-fail 0.6] [--retry-rate 0.10]

Reads data/prompts.jsonl and data/answers/answers.jsonl (read-only). For each judge-scored prompt it
builds the exact judge messages the judge would send (app/verifier/judge.py) and converts characters to
tokens with ratios calibrated on Groq's own token counts for gpt-oss-120b in this run. Answers that
aren't collected yet are estimated from the average of collected answers of the same tier, source and
task type (falling back to the tier and source, then to pilot 2).

Options compared:
    A  judge low and medium on every judge-scored prompt
    B  A, minus candidates that already failed (truncated, empty, input too large) and no_reference prompts
    C  B, but judge medium only when low failed (assume medium passes when low passes)
    D  C, plus judging the extra tier (llama)
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from app.registry import PROJECT_ROOT
from app.verifier.judge import JUDGE_SYSTEM_PROMPT, build_judge_messages, load_judge_settings
from eval.build_dataset import FULL_DIR, PILOT2_DIR, load_prompts
from eval.score_answers import is_input_too_large, load_answers

DAILY_TOKENS = 200_000                  # gpt-oss-120b free-tier tokens per day (rolling 24h)
FOLLOW_UP_TOKENS = 45                   # the retry's follow-up message
JUDGED_TIERS = ("low", "medium", "extra")


# ---------- Token calibration ----------

def fit_line(pairs: list) -> tuple:
    """Least-squares (intercept, slope) for tokens = intercept + slope * chars."""
    mean_x = statistics.mean(c for c, _ in pairs)
    mean_y = statistics.mean(t for _, t in pairs)
    slope = sum((c - mean_x) * (t - mean_y) for c, t in pairs) / sum((c - mean_x) ** 2 for c, _ in pairs)
    return mean_y - slope * mean_x, slope


def calibrate(records: list, prompts: dict) -> dict:
    """chars->tokens ratios from gpt-oss-120b's reported counts: prompts (with chat overhead) and answer text."""
    prompt_pairs, answer_pairs = [], []
    for r in records:
        if r["model_name"] != "groq-large" or r["error"] or r["cache_hit"] or r["prompt_id"] not in prompts:
            continue
        prompt = prompts[r["prompt_id"]]
        prompt_pairs.append((sum(len(m["content"]) for m in prompt["messages"]), r["input_tokens"]))
        if r["answer"] and r["reasoning_tokens"] is not None:
            answer_pairs.append((len(r["answer"]), r["output_tokens"] - r["reasoning_tokens"]))
    overhead, per_char = fit_line(prompt_pairs)
    _, answer_per_char = fit_line(answer_pairs)
    return {"overhead": overhead, "prompt_per_char": per_char, "answer_per_char": answer_per_char,
            "samples": len(prompt_pairs)}


# ---------- Answer statistics ----------

def complete(record) -> bool:
    return bool(record) and not record["error"] and not record["truncated"] and bool((record["answer"] or "").strip())


def prefailed(record) -> bool:
    """Already a fail before judging: truncated, empty, or an input the model can never accept."""
    if not record:
        return False
    if record["error"]:
        return is_input_too_large(record["error"])
    return record["truncated"] or not (record["answer"] or "").strip()


class AnswerStats:
    """Average answer length and pre-judge failure rate per (tier, source[, task_type]), for estimating
    answers that haven't been collected yet."""

    def __init__(self, latest: dict, pilot: dict, prompts: dict):
        self.length = defaultdict(list)
        self.failed = defaultdict(list)
        for source_records in (latest, pilot):
            for (pid, tier), record in source_records.items():
                if pid not in prompts or (record["error"] and not is_input_too_large(record["error"])):
                    continue
                p = prompts[pid]
                for key in ((tier, p["source"], p["task_type"]), (tier, p["source"]), (tier,)):
                    self.failed[key].append(prefailed(record))
                    if complete(record):
                        self.length[key].append(len(record["answer"]))

    def _lookup(self, table: dict, tier: str, prompt: dict):
        for key in ((tier, prompt["source"], prompt["task_type"]), (tier, prompt["source"]), (tier,)):
            if table.get(key):
                return statistics.mean(table[key])
        return None

    def avg_length(self, tier: str, prompt: dict) -> float:
        return self._lookup(self.length, tier, prompt) or 2000.0

    def fail_rate(self, tier: str, prompt: dict) -> float:
        return self._lookup(self.failed, tier, prompt) or 0.0


# ---------- Estimate ----------

def verdict_chars(n_constraints: int) -> int:
    """Length of a typical verdict JSON: constraint checks plus a one-sentence reason."""
    return 60 + 160 + n_constraints * 55


def estimate(prompts: list, latest: dict, stats: AnswerStats, cal: dict, low_judge_fail: float,
             retry_rate: float, judge_reasoning: float) -> tuple:
    options = {key: {"calls": 0.0, "tokens": 0.0, "retry_tokens": 0.0} for key in "ABCD"}
    facts = defaultdict(float)

    for prompt in prompts:
        high = latest.get((prompt["id"], "high"))
        if high:
            p_ref = 1.0 if complete(high) else 0.0
            ref_chars = len(high["answer"] or "") if not high["error"] else 0
            facts["no_reference (collected)"] += 1 - p_ref
        else:
            p_ref = 1 - stats.fail_rate("high", prompt)
            ref_chars = stats.avg_length("high", prompt)
            facts["no_reference (expected, high not collected)"] += 1 - p_ref
            facts["high answers not collected yet"] += 1

        base = build_judge_messages(prompt["messages"], prompt["constraints"], "", "")
        base_chars = sum(len(m["content"]) for m in base)
        out_tokens = judge_reasoning + verdict_chars(len(prompt["constraints"])) * cal["answer_per_char"]

        call = {}     # tier -> (probability the candidate is judgeable, tokens for one call)
        for tier in JUDGED_TIERS:
            record = latest.get((prompt["id"], tier))
            if record and not (record["error"] and not is_input_too_large(record["error"])):
                p_ok = 1.0 if complete(record) else 0.0
                cand_chars = len(record["answer"] or "") if not record["error"] else 0
                if tier == "low" and prefailed(record):
                    facts["low answers already failed before judging"] += 1
            else:
                p_ok = 1 - stats.fail_rate(tier, prompt)
                cand_chars = stats.avg_length(tier, prompt)
                facts[f"{tier} answers not collected yet"] += 1
            in_tokens = (cal["overhead"] + base_chars * cal["prompt_per_char"]
                         + (ref_chars + cand_chars) * cal["answer_per_char"])
            call[tier] = (p_ok, in_tokens + out_tokens)
        retry_extra = verdict_chars(len(prompt["constraints"])) * cal["answer_per_char"] + FOLLOW_UP_TOKENS

        def add(option: str, calls: float, tokens_per_call: float) -> None:
            options[option]["calls"] += calls
            options[option]["tokens"] += calls * tokens_per_call
            options[option]["retry_tokens"] += calls * retry_rate * (tokens_per_call + retry_extra)

        (p_low, t_low), (p_med, t_med), (p_ext, t_ext) = call["low"], call["medium"], call["extra"]
        add("A", 1, t_low)
        add("A", 1, t_med)
        for option in "BCD":
            add(option, p_ref * p_low, t_low)
        add("B", p_ref * p_med, t_med)
        # C/D: medium only when low failed, either before judging or by the judge
        p_low_failed = (1 - p_low) + p_low * low_judge_fail
        for option in "CD":
            add(option, p_ref * p_med * p_low_failed, t_med)
        add("D", p_ref * p_ext, t_ext)
    return options, facts


def main():
    parser = argparse.ArgumentParser(description="Estimate judge token cost (no model calls).")
    parser.add_argument("--low-judge-fail", type=float, default=0.6,
                        help="share of judgeable low answers the judge fails (drives options C and D)")
    parser.add_argument("--retry-rate", type=float, default=0.10, help="share of judge calls that need a retry")
    # gpt-oss-120b at reasoning_effort=low averaged 16-131 reasoning tokens per prompt type in this run;
    # a judge call checks several constraints, so assume the top of that range.
    parser.add_argument("--judge-reasoning", type=float, default=150, help="reasoning tokens per judge call")
    args = parser.parse_args()

    all_prompts = load_prompts()
    by_id = {p["id"]: p for p in all_prompts}
    judged = [p for p in all_prompts if p["scoring"] == "judge"]
    latest = load_answers(FULL_DIR / "answers.jsonl")
    pilot = load_answers(PILOT2_DIR / "answers.jsonl")
    records = [json.loads(line) for line in (FULL_DIR / "answers.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()][:-1]   # skip the last line in case the collector is mid-write
    cal = calibrate(records, by_id)
    settings = load_judge_settings()
    options, facts = estimate(judged, latest, AnswerStats(latest, pilot, by_id), cal, args.low_judge_fail,
                              args.retry_rate, args.judge_reasoning)

    print(f"Judge-scored prompts: {len(judged)} ({sum(p['source'] == 'own' for p in judged)} own, "
          f"{sum(p['source'] == 'arena' for p in judged)} arena) | judge model: {settings.model_name}")
    print(f"Token calibration from {cal['samples']} gpt-oss-120b answers: prompt ~{cal['overhead']:.0f} + "
          f"chars/{1 / cal['prompt_per_char']:.2f}, answer text chars/{1 / cal['answer_per_char']:.2f}; "
          f"judge output ~{args.judge_reasoning:.0f} reasoning tokens + the verdict")
    print(f"Assumptions: judge fails {args.low_judge_fail:.0%} of judgeable low answers (C, D); "
          f"{args.retry_rate:.0%} of calls need a retry\n")
    for key, value in sorted(facts.items()):
        print(f"  {key}: {value:.1f}")
    labels = {"A": "low + medium, every prompt", "B": "A minus pre-failed / no_reference",
              "C": "B, medium only if low failed", "D": "C + extra (llama)"}
    print(f"\n{'option':<38}{'calls':>8}{'tokens':>12}{'days':>7}{'+ retries':>12}{'days':>7}")
    for key in "ABCD":
        o = options[key]
        total = o["tokens"] + o["retry_tokens"]
        print(f"{key}  {labels[key]:<35}{o['calls']:>8.0f}{o['tokens']:>12,.0f}{o['tokens'] / DAILY_TOKENS:>7.2f}"
              f"{total:>12,.0f}{total / DAILY_TOKENS:>7.2f}")
    print(f"\nDays = tokens / {DAILY_TOKENS:,} (gpt-oss-120b's free daily tokens). Judging shares that budget with the "
          "collector's high tier, so these days start after collection finishes.")


if __name__ == "__main__":
    main()
