"""Triage request headers: X-Triage-Quality, X-Triage-Task, X-Triage-Verify. Bad values -> 400."""
from typing import Optional

from app.api.errors import APIError, bad_request
from app.router.features import TASK_TYPES, normalize_task_type
from app.router.quality import QualityError, QualitySettings, resolve_tau

QUALITY, TASK, VERIFY = "X-Triage-Quality", "X-Triage-Task", "X-Triage-Verify"
VERIFY_MODES = ("off", "sync")


def parse_quality(value: Optional[str], settings: QualitySettings) -> float:
    try:
        return resolve_tau(value, settings)
    except QualityError:
        options = " | ".join(settings.presets)
        raise bad_request(f"{QUALITY} must be {options} or a number from 0 to 1, got {value!r}", QUALITY) from None


def parse_task(value: Optional[str]) -> Optional[str]:
    if value is None or not value.strip():
        return None
    task = normalize_task_type(value)
    if task is None:
        raise bad_request(f"{TASK} must be one of {', '.join(TASK_TYPES)} (or a known alias), got {value!r}", TASK)
    return task


def parse_verify(value: Optional[str], stream: bool) -> str:
    mode = (value or "off").strip().lower()
    if mode not in VERIFY_MODES:
        raise bad_request(f"{VERIFY} must be off or sync, got {value!r}", VERIFY)
    if mode == "sync" and stream:
        raise bad_request(f"{VERIFY}: sync can't be combined with stream=true: the answer has to be checked "
                          "before any of it is sent", VERIFY, "unsupported_combination")
    if mode == "sync":
        raise APIError(501, f"{VERIFY}: sync is not available yet (Phase 4 Step 7). Use off, or leave the header out.",
                       "not_implemented", VERIFY, "not_available_yet")
    return mode
