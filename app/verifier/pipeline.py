"""Answer verification pipeline: cheapest check first (Phase 4 Step 1).

    (a) format checker (checker.check_answer, free)     flags -> bad,  decided_by "format"
    (b) agreement (agreement.py, free given a 2nd answer) only if other_answer is given and the conversation
                                                         asks for a comparable final answer:
                                                         same final answer -> good, different -> bad,
                                                         decided_by "agreement"
    (c) AI checker (ai_checker.py, one paid call)        only if ai_check: good/bad, decided_by "ai_checker";
                                                         a checker error -> good None, "ai_checker_error"
    (d) nothing applied                                  good None, decided_by "none"

good=None means "unknown"; what the router does with it (escalate or not) is decided in Step 3, not here.
In (b), an other_answer without a usable final answer counts as a disagreement, as in eval/agreement_sim.py.
"""
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.verifier.agreement import agree, comparable, final_key
from app.verifier.ai_checker import AICheckResult, CheckerSettings, check_answer_ai, load_checker_settings
from app.verifier.checker import check_answer


@dataclass(frozen=True)
class VerifyResult:
    good: Optional[bool]
    decided_by: str                       # format, agreement, ai_checker, ai_checker_error, none
    reasons: tuple = field(default_factory=tuple)
    ai: Optional[AICheckResult] = None    # the AI check, when it ran
    # The AI check's cost fields (zero / None when it didn't run)
    checker_model: Optional[str] = None
    checker_version: Optional[int] = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: Optional[int] = None
    reported_cost: float = 0.0
    api_calls: int = 0


def verify(messages: list, answer: Optional[str], *, other_answer: Optional[str] = None, truncated: bool = False,
           error=None, ai_check: bool = True, other_truncated: bool = False, checker_model: Optional[str] = None,
           send: Optional[Callable] = None, settings: Optional[CheckerSettings] = None) -> VerifyResult:
    # (a) free format checker
    fmt = check_answer(messages, answer, truncated=truncated, error=error)
    if not fmt.ok:
        return VerifyResult(False, "format", fmt.reasons)

    # (b) agreement with a second, independent answer
    if other_answer is not None and comparable(messages):
        mine, theirs = final_key(messages, answer, truncated), final_key(messages, other_answer, other_truncated)
        if agree(mine, theirs):
            return VerifyResult(True, "agreement", (f"both answers give {mine}",))
        return VerifyResult(False, "agreement", (f"final answers differ: {mine} vs {theirs}",))

    # (c) AI checker
    if ai_check:
        settings = settings or load_checker_settings()
        model = checker_model or settings.model_name
        ai = check_answer_ai(messages, answer, model=model, send=send, settings=settings)
        cost = dict(ai=ai, checker_model=ai.checker_model, checker_version=ai.checker_version,
                    input_tokens=ai.input_tokens, output_tokens=ai.output_tokens,
                    reasoning_tokens=ai.reasoning_tokens, reported_cost=ai.reported_cost, api_calls=ai.attempts)
        if ai.status != "ok":
            return VerifyResult(None, "ai_checker_error", (ai.reason,), **cost)
        return VerifyResult(ai.good, "ai_checker", (ai.reason,), **cost)

    # (d) nothing applied
    return VerifyResult(None, "none")
