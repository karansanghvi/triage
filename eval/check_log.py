"""Offline log of AI-checker results, so reruns cost nothing (Phase 4 Step 1).

Append-only JSONL (default data/checks/checks.jsonl), like run_judge.JudgmentLog. Records are keyed by
(prompt_id, tier, checker_model, checker_version) and the newest record per key wins.

logged_check() reuses a logged result instead of calling the model when the key matches AND the result is
finished AND it was made from exactly the same checker input (input_sha256 = hash of the messages sent, which
covers the system prompt, the conversation and the answer). So a re-collected answer or an edited prompt is
checked again rather than served stale.

Finished = status "ok", or a deterministic "error" (truncated, empty, invalid_json, bad_shape): at temperature 0
the same request would fail the same way, so retrying only costs money. Provider errors (timeouts, rate limits)
are not finished and are retried on the next run.
"""
import dataclasses
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from app.registry import PROJECT_ROOT
from app.verifier.ai_checker import (AICheckResult, CheckerSettings, build_checker_messages, check_answer_ai,
                                     load_checker_settings)

CHECKS_DIR = PROJECT_ROOT / "data" / "checks"
CHECKS_PATH = CHECKS_DIR / "checks.jsonl"
RESULT_FIELDS = [f.name for f in dataclasses.fields(AICheckResult)]


def input_sha256(messages: list, answer: str) -> str:
    """SHA-256 of the exact checker messages for this conversation and answer."""
    text = json.dumps(build_checker_messages(messages, answer), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def is_finished(record: dict) -> bool:
    return record["status"] == "ok" or not str(record.get("error_kind") or "").startswith("provider")


def make_check_record(prompt_id: str, tier: str, result: AICheckResult, input_hash: str) -> dict:
    return {"prompt_id": prompt_id, "tier": tier, **dataclasses.asdict(result), "input_sha256": input_hash,
            "api_calls": result.attempts, "timestamp": datetime.now(timezone.utc).isoformat()}


def result_from_record(record: dict) -> AICheckResult:
    """The logged result, with zero usage: reusing it costs nothing."""
    kept = {k: record[k] for k in RESULT_FIELDS if k in record}
    return AICheckResult(**{**kept, "input_tokens": 0, "output_tokens": 0, "reasoning_tokens": None,
                            "reported_cost": 0.0, "attempts": 0})


class CheckLog:
    """Append-only checks.jsonl; the newest record per (prompt_id, tier, checker_model, checker_version) wins."""

    def __init__(self, path: Path = CHECKS_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()] \
            if self.path.exists() else []

    @staticmethod
    def key(record: dict) -> tuple:
        return record["prompt_id"], record["tier"], record["checker_model"], record["checker_version"]

    def latest(self) -> dict:
        out = {}
        for r in self.records:
            out[self.key(r)] = r
        return out

    def finished(self, checker_model: Optional[str] = None, checker_version: Optional[int] = None) -> dict:
        """Finished results; with checker_model / checker_version, only those made by that model / version."""
        return {k: r for k, r in self.latest().items() if is_finished(r)
                and (checker_model is None or r["checker_model"] == checker_model)
                and (checker_version is None or r["checker_version"] == checker_version)}

    def append(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records.append(record)


def logged_check(log: CheckLog, prompt_id: str, tier: str, messages: list, answer: str, *,
                 model: Optional[str] = None, send: Optional[Callable] = None,
                 settings: Optional[CheckerSettings] = None) -> tuple:
    """(AICheckResult, reused). A matching finished record is returned with zero cost and no call."""
    settings = settings or load_checker_settings()
    model = model or settings.model_name
    input_hash = input_sha256(messages, answer)
    record = log.latest().get((prompt_id, tier, model, settings.checker_version))
    if record is not None and is_finished(record) and record.get("input_sha256") == input_hash:
        return result_from_record(record), True
    result = check_answer_ai(messages, answer, model=model, send=send, settings=settings)
    log.append(make_check_record(prompt_id, tier, result, input_hash))
    return result, False
