"""Background labels (Phase 4 Step 7): grade a sample of live answers after they have been sent.

For a sampled model="auto" request (verify off, served tier not high, no fallback, answered without error),
once the response has been fully sent (streams included):
    1. ask the strong model (the high tier's first model) the same question, with the Phase 2 settings
    2. grade the served answer with the judge (app/verifier/judge.py, v3), using that answer as the reference
    3. write a row to the `labels` table, linked by request_id: passed, judge score, versions and models,
       how many calls and tokens it used, and no text

The user's response never changes and never waits for this: it runs as a Starlette background task, which
starts after the last byte of the response has been sent. Anything that goes wrong is caught and written as a
"skipped" label (with a reason), so a background failure can't affect a response or crash the server.

Prompt and answer text are not stored anywhere, so the job uses the copies still in memory from the request
and drops them when it ends.

A daily cap (UTC day) limits background calls and tokens. Once it is reached, sampled requests are written as
skipped (reason daily_cap) without any calls, until the next day.

NOISE WARNING: these labels compare the served answer with the strong model's answer, which can be wrong too.
Phase 3 found the judge passes about 75% of wrong answers when the reference itself is wrong. Treat a
background "passed" as "agrees with the strong model", not as ground truth.
"""
import logging
import random
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import anyio

from app.api.request_log import LabelRow
from app.reliability import AllModelsFailed
from app.verifier.judge import judge

logger = logging.getLogger("triage.labels")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def eligible(row) -> bool:
    """Requests whose answer the classifier's labels would describe, and which aren't already strong. Cache hits
    are skipped: the answer is a repeat of one already served (and possibly already labeled)."""
    # BYOK and demo requests never: a background job must never get a user's key, and demo labels would spend the
    # free quota on strangers' prompts
    return (row.mode not in ("byok", "demo") and row.requested_model == "auto" and row.status == "ok" and row.verified == "off"
            and row.tier in ("low", "medium") and row.fell_back is False and not row.escalated
            and not row.cache_hit)


class JudgeSend:
    """The judge's send(messages, settings) -> text, through the reliable client (judge.yaml's timeout,
    since long answers take longer to grade than the per-model default). Counts calls, tokens and cost."""

    def __init__(self, client):
        self.client = client
        self.calls = self.tokens = 0
        self.cost = 0.0

    def __call__(self, messages, settings):
        response_format = {"type": settings.response_format} if settings.response_format else None
        result = self.client.send_model(messages, settings.model_name, stream=False,
                                        temperature=settings.temperature, max_tokens=settings.max_tokens,
                                        timeout=settings.timeout_seconds, response_format=response_format)
        r = result.response
        self.calls += 1
        self.tokens += (r.input_tokens or 0) + (r.output_tokens or 0)
        self.cost += r.reported_cost
        return r.text


class BackgroundLabeler:
    """One per server. rng decides sampling (injectable for tests); clock gives the current UTC time."""

    def __init__(self, rng: Optional[random.Random] = None, clock: Callable[[], datetime] = _now):
        self.rng = rng or random.Random()
        self.clock = clock
        self._lock = threading.Lock()
        self._day = None
        self._calls = self._tokens = 0

    # ----- the daily cap -----

    def _day_start(self) -> str:
        now = self.clock()
        return now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()

    def _refresh(self, log) -> None:
        """At the first job of a UTC day, load that day's usage from the labels table (survives restarts)."""
        day = self._day_start()
        if day != self._day:
            self._day = day
            self._calls, self._tokens = log.label_usage_since(day)

    def _reserve(self, log, policy) -> bool:
        with self._lock:
            self._refresh(log)
            if self._calls + 2 > policy.daily_max_calls or self._tokens >= policy.daily_max_tokens:
                return False
            self._calls += 2                      # held for the strong answer and the judge; corrected after
            return True

    def _settle(self, reserved: int, calls: int, tokens: int) -> None:
        with self._lock:
            self._calls += calls - reserved
            self._tokens += tokens

    def usage_today(self, log) -> tuple:
        with self._lock:
            self._refresh(log)
            return self._calls, self._tokens

    # ----- one job -----

    async def run(self, deps, row, messages: list, answer: Optional[str]) -> None:
        """The background task. Never raises."""
        try:
            await self._run(deps, row, messages, answer)
        except Exception as error:                # last resort: a broken label must not break anything else
            logger.warning("background label for %s failed outside the job: %s", row.request_id,
                           type(error).__name__)
        finally:
            messages = answer = None              # drop the only copies of the text

    async def _run(self, deps, row, messages, answer) -> None:
        policy = deps.config.snapshot().verification.background
        if not eligible(row) or answer is None or policy.sample_rate <= 0:
            return
        if self.rng.random() >= policy.sample_rate:
            return
        label = LabelRow(request_id=row.request_id, created_at=self.clock().isoformat(), status="skipped",
                         served_model=row.served_model, served_tier=row.tier)
        if not self._reserve(deps.log, policy):
            label.skip_reason = "daily_cap"
            self._write(deps, label)
            return
        send = JudgeSend(deps.client)
        calls = tokens = 0
        try:
            strong_name = deps.strong_model()
            strong = await anyio.to_thread.run_sync(lambda: deps.client.send_model(messages, strong_name, stream=False))
            reference = strong.response
            calls, tokens = 1, (reference.input_tokens or 0) + (reference.output_tokens or 0)
            label.reference_model = strong.model.name
            label.cost = reference.reported_cost
            truncated = (reference.finish_reason or "").lower() in ("length", "max_tokens")
            settings = deps.judge_settings
            result = await anyio.to_thread.run_sync(lambda: judge(messages, [], reference.text, answer, settings,
                                                                  reference_truncated=truncated, send=send))
            calls, tokens = calls + send.calls, tokens + send.tokens
            label.cost += send.cost
            label.judge_model, label.judge_version = settings.model_name, settings.judge_version
            if result.status == "scored":
                label.status, label.passed, label.judge_score = "labeled", result.passed, result.score
            else:
                label.skip_reason = result.status            # no_reference or error (unparseable verdict)
        except AllModelsFailed:
            calls, tokens = calls + send.calls, tokens + send.tokens
            label.skip_reason = "model_failed"
        except Exception as error:
            calls, tokens = calls + send.calls, tokens + send.tokens
            label.skip_reason = f"error:{type(error).__name__}"
        finally:
            self._settle(2, calls, tokens)
        label.calls, label.tokens = calls, tokens
        self._write(deps, label)

    @staticmethod
    def _write(deps, label: LabelRow) -> None:
        try:
            deps.log.write_label(label)
        except Exception as error:
            logger.warning("couldn't write the background label for %s: %s", label.request_id, type(error).__name__)
            return
        logger.info('{"request_id":"%s","label":"%s","reason":%s}', label.request_id, label.status,
                    "null" if label.skip_reason is None else f'"{label.skip_reason}"')
