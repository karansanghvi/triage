"""Phase 4 Step 1 smoke test: a few REAL AI-checker calls, to check the plumbing and measure real cost.

Usage:
    python -m eval.checker_smoke

Five fit-set answers (picked once with a seeded draw; see ITEMS), checked by groq-small through CheckLog, so a
rerun reuses the logged results and costs nothing. Hard cap: MAX_CALLS real calls in total, retries included;
the run stops before a call that would go over it. Calls are paced with the Phase 2 Pacer against groq-small's
limits, counting its past usage (collector answers and earlier checks). If the daily budget is used up, it stops
instead of waiting.

Not an accuracy measurement: five calls say nothing about how often the checker is right.
Writes data/checks/checks.jsonl (via CheckLog) and reports/checker_smoke.json.
"""
import json
from datetime import datetime

from app.registry import PROJECT_ROOT, load_registry
from app.verifier.ai_checker import default_send, load_checker_settings
from app.verifier.checker import check_answer
from eval.build_dataset import FULL_DIR, Pacer, load_collection_config, load_prompts
from eval.check_log import CHECKS_PATH, CheckLog, logged_check
from eval.score_answers import load_answers
from eval.splits import fit_ids, load_split, load_validation_split

OUTCOMES_PATH = PROJECT_ROOT / "data" / "outcomes.jsonl"
REPORT_PATH = PROJECT_ROOT / "reports" / "checker_smoke.json"
MAX_CALLS = 5

# Picked with random.Random(42).choice over each category's sorted fit-set candidates (val and test excluded,
# and the old calibration items now in val dropped):
#   a  GSM8K medium, exact-match pass                         (120 candidates)
#   b  GSM8K medium, exact-match fail                         (8)
#   c  GSM8K low, truly right, failed by the Phase 3 judge on "ends with 'Answer: <number>'" because the
#      Answer: line carries extra text around the number (the format trap)              (12)
#   d  MMLU medium, exact-match fail                          (17)
#   e  open_ended medium, judged fail                         (3)
ITEMS = [("a", "gsm8k_0142", "medium"), ("b", "gsm8k_0076", "medium"), ("c", "gsm8k_0014", "low"),
         ("d", "mmlu_0111", "medium"), ("e", "arena_0031", "medium")]


class CallCapReached(Exception):
    pass


class PacedCappedSend:
    """default_send behind the Pacer and a hard call cap; remembers each call's latency."""

    def __init__(self, pacer: Pacer, max_calls: int):
        self.pacer, self.max_calls, self.calls, self.latencies = pacer, max_calls, 0, []

    def __call__(self, messages, model_name, settings):
        if self.calls >= self.max_calls:
            raise CallCapReached(f"stopping: {self.max_calls} real calls already made")
        estimate = sum(len(m["content"]) for m in messages) // 4 + settings.max_tokens
        if not self.pacer.daily_budget_left(estimate):
            raise CallCapReached(f"stopping: {model_name}'s daily budget is used up")
        self.pacer.wait_for_minute_slot(estimate)
        self.calls += 1
        response = default_send(messages, model_name, settings)
        self.pacer.record(1, response.input_tokens + response.output_tokens)
        self.latencies.append(response.latency_ms)
        return response


def usage_history(model_name: str, log: CheckLog) -> list:
    """(UTC time, api calls, tokens) already spent on this model: collector answers and earlier checks."""
    history = []
    for line in (FULL_DIR / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line) if line.strip() else {}
        if r.get("model_name") == model_name and r.get("api_calls"):
            history.append((datetime.fromisoformat(r["timestamp"]), r["api_calls"],
                            (r["input_tokens"] or 0) + (r["output_tokens"] or 0)))
    history += [(datetime.fromisoformat(r["timestamp"]), r["api_calls"], r["input_tokens"] + r["output_tokens"])
                for r in log.records if r["checker_model"] == model_name and r.get("api_calls")]
    return history


def fit_truth(ids: set) -> dict:
    """Outcomes for the given ids, which must all be fit-set ids (checked before anything is read)."""
    train, _ = load_split()
    fit = fit_ids(train)
    assert ids <= fit and not ids & load_validation_split(), "smoke items must be fit-set prompts"
    out = {}
    for line in OUTCOMES_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            o = json.loads(line)
            if o["id"] in ids:
                out[o["id"]] = o
    return out


def main():
    settings = load_checker_settings()
    model = settings.model_name
    ids = {pid for _, pid, _ in ITEMS}
    truth = fit_truth(ids)
    prompts = {p["id"]: p for p in load_prompts()}
    answers = load_answers(FULL_DIR / "answers.jsonl")
    log = CheckLog(CHECKS_PATH)
    pacer = Pacer(load_collection_config(load_registry()).limits[model], usage_history(model, log))
    send = PacedCappedSend(pacer, MAX_CALLS)

    rows = []
    for label, pid, tier in ITEMS:
        messages, answer = prompts[pid]["messages"], answers[(pid, tier)]
        before = len(send.latencies)
        try:
            result, reused = logged_check(log, pid, tier, messages, answer["answer"], model=model, send=send,
                                          settings=settings)
        except CallCapReached as stop:
            print(stop)
            break
        outcome = truth[pid]["tiers"][tier]
        rows.append({
            "item": label, "prompt_id": pid, "source": truth[pid]["source"], "task_type": truth[pid]["task_type"],
            "tier": tier, "truth": outcome["status"], "truth_method": outcome["method"],
            "format_check_ok": check_answer(messages, answer["answer"], truncated=bool(answer.get("truncated"))).ok,
            "status": result.status, "good": result.good, "reason": result.reason, "error_kind": result.error_kind,
            "attempts": result.attempts, "reused_from_log": reused,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "reasoning_tokens": result.reasoning_tokens,
            "latency_ms": round(sum(send.latencies[before:]), 1),
            "check_cost": result.reported_cost, "answer_cost": answer.get("price_as_cost"),
            "answer_model": answer["model_name"],
        })
        verdict = "error" if result.status != "ok" else ("good" if result.good else "not good")
        print(f"[{label}] {pid} {tier}: truth {outcome['status']}, checker {verdict}"
              + (" (from log)" if reused else f", {result.attempts} call(s)") + f" | {result.reason}")

    report = {"checker_model": model, "checker_version": settings.checker_version, "max_calls": MAX_CALLS,
              "real_calls": send.calls, "rows": rows,
              "note": "plumbing and cost check only; 5 calls say nothing about accuracy"}
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n{send.calls} real call(s); saved {REPORT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
