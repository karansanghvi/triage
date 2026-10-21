import json

import pytest

from eval.score_answers import load_answers, score_all, score_pair

EXACT = {"id": "own_qa_001", "source": "own", "task_type": "qa", "difficulty_guess": "easy",
         "scoring": "exact_match", "reference": "14"}
JUDGE = {"id": "own_creative_003", "source": "own", "task_type": "creative", "difficulty_guess": "easy",
         "scoring": "judge", "reference": None}


def record(answer="14", truncated=False, error=None, finish="stop"):
    return {"prompt_id": "own_qa_001", "tier": "low", "model_id": "m", "answer": answer,
            "truncated": truncated, "error": error, "finish_reason": finish}


def test_correct_answer_passes():
    result = score_pair(EXACT, record("14"))
    assert result["status"] == "pass"
    assert result["scorer"] == "exact_match"


def test_wrong_answer_fails_with_the_scorer_reason():
    result = score_pair(EXACT, record("15"))
    assert result["status"] == "fail"
    assert "expected" in result["reason"]


def test_truncated_answer_fails_even_if_it_contains_the_right_value():
    result = score_pair(EXACT, record("14", truncated=True, finish="length"))
    assert (result["status"], result["reason"]) == ("fail", "truncated")


@pytest.mark.parametrize("answer", ["", "   \n", None])
def test_empty_answer_fails(answer):
    result = score_pair(EXACT, record(answer))
    assert (result["status"], result["reason"]) == ("fail", "empty")


def test_request_too_large_counts_as_a_fail_for_that_tier():
    error = {"kind": "client", "message": "[groq] client: Error code: 413 - {'error': {'message': "
                                          "'Request too large for model `allam-2-7b` ... TPM: Limit 6000, Requested 7842"}
    result = score_pair(EXACT, record(None, error=error))
    assert (result["status"], result["reason"]) == ("fail", "input too large")


def test_unanswered_prompt_is_pending_not_a_fail():
    assert score_pair(EXACT, None)["status"] == "pending"


@pytest.mark.parametrize("kind", ["rate_limit", "timeout", "server", "connection", "circuit_open"])
def test_retryable_errors_are_pending(kind):
    result = score_pair(EXACT, record(None, error={"kind": kind, "message": "x"}))
    assert result["status"] == "pending"


def test_other_client_errors_are_pending_not_input_too_large():
    result = score_pair(EXACT, record(None, error={"kind": "client", "message": "invalid model"}))
    assert result["status"] == "pending"


def test_judge_prompts_need_the_judge():
    assert score_pair(JUDGE, record("a nice tagline"))["status"] == "needs_judge"


def test_judge_prompts_still_fail_when_truncated_or_empty():
    assert score_pair(JUDGE, record("x", truncated=True))["reason"] == "truncated"
    assert score_pair(JUDGE, record(""))["reason"] == "empty"
    assert score_pair(JUDGE, None)["status"] == "pending"


def test_scoring_is_deterministic_and_covers_every_pair(tmp_path):
    answers = tmp_path / "answers.jsonl"
    rows = [record("15"), record("14")]                       # newest record for the key wins
    answers.write_text("".join(json.dumps(r) + "\n" for r in rows) + '{"prompt_id": "half-writ')  # partial last line
    latest = load_answers(answers)
    first = score_all([EXACT, JUDGE], ["low", "medium"], latest)
    second = score_all([EXACT, JUDGE], ["low", "medium"], load_answers(answers))
    assert first == second
    assert [(r["prompt_id"], r["tier"], r["status"]) for r in first] == [
        ("own_qa_001", "low", "pass"), ("own_qa_001", "medium", "pending"),
        ("own_creative_003", "low", "pending"), ("own_creative_003", "medium", "pending"),
    ]
