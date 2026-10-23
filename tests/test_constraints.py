import json

import pytest

from app.verifier.constraints import (
    check_constraint, count_characters, count_sentences, count_words, parse_constraint,
)
from app.verifier.judge import JudgeSettings, apply_code_checks, judge

# ---------- counting ----------

@pytest.mark.parametrize("text, words", [
    ("one two three", 3),
    ("  spaced   out\nacross lines ", 4),
    ("an 18-inch, hand-wash-only board", 4),            # hyphenated words are one word
    ("**Answer:** 42", 2),                                # markdown markers attached to a word don't add words
    ("- item one\n- item two", 4),                         # a lone "-" is not a word
    ("Price: $1,200 — final.", 3),                        # a lone dash is not a word
    ("", 0),
])
def test_word_count(text, words):
    assert count_words(text) == words


def test_character_count_trims_outer_whitespace_and_counts_inner_spaces():
    assert count_characters("  ab cd  \n") == 5


@pytest.mark.parametrize("text, sentences", [
    ("One. Two! Three?", 3),
    ("Dr. Smith paid $3.50 for it. Done.", 2),            # abbreviation and decimal don't end sentences
    ("It costs 2.5 times more, e.g. in Q3.", 1),
    ("No final punctuation", 1),
    ('He said "stop." Then left.', 2),
])
def test_sentence_count(text, sentences):
    assert count_sentences(text) == sentences


@pytest.mark.parametrize("text", ["- one.\n- two.", "# Title\nText.", "Para one.\n\nPara two.", "```\ncode.\n```"])
def test_sentence_count_is_unreliable_for_non_prose(text):
    assert count_sentences(text) is None


# ---------- parsing ----------

@pytest.mark.parametrize("constraint, parsed", [
    ("under 80 words", ("words", "<", 80, None)),
    ("max 20 words", ("words", "<=", 20, None)),
    ("maximum 8 words", ("words", "<=", 8, None)),
    ("50 words or fewer", ("words", "<=", 50, None)),
    ("80–120 words", ("words", "range", 80, 120)),
    ("under 60 characters including spaces", ("characters", "<", 60, None)),
    ("280 characters or fewer including the URL", ("characters", "<=", 280, None)),
    ("exactly 3 bullets", ("bullets", "==", 3, None)),
    ("at most 4 bullets", ("bullets", "<=", 4, None)),
    ("exactly 2 sentences", ("sentences", "==", 2, None)),
    ("one sentence", ("sentences", "==", 1, None)),
    ("1–2 sentences", ("sentences", "range", 1, 2)),
])
def test_countable_constraints_are_parsed(constraint, parsed):
    assert parse_constraint(constraint) == parsed


@pytest.mark.parametrize("constraint", [
    "each bullet under 12 words",          # scoped to parts of the answer
    "Customer: exactly 2 sentences",
    "Short: 20 words max",
    "description 20–40 words",
    "title 60 characters or fewer",
    "includes 18 x 12 in",                 # not a count at all
    "plain language",
    "no exclamation marks",
])
def test_unparseable_or_scoped_constraints_are_left_to_the_judge(constraint):
    assert parse_constraint(constraint) is None
    assert check_constraint(constraint, "any answer at all.") is None


# ---------- checking ----------

def test_word_limits():
    seventy_nine, eighty = " ".join(["w"] * 79), " ".join(["w"] * 80)
    assert check_constraint("under 80 words", seventy_nine) is True
    assert check_constraint("under 80 words", eighty) is False            # "under" is strict
    assert check_constraint("max 80 words", eighty) is True               # "max" includes the limit
    assert check_constraint("80–120 words", seventy_nine) is False


def test_character_limits():
    assert check_constraint("under 60 characters including spaces", "x" * 59) is True
    assert check_constraint("under 60 characters including spaces", "x" * 60) is False
    assert check_constraint("170 characters or fewer including spaces", "x" * 170) is True


def test_bullet_counts():
    three = "- one\n- two\n- three"
    assert check_constraint("exactly 3 bullets", three) is True
    assert check_constraint("exactly 3 bullets", three + "\n- four") is False
    assert check_constraint("exactly 3 bullets", "1. one\n2) two\n• three") is True   # numbered and • bullets count
    assert check_constraint("at most 4 bullets", three) is True
    assert check_constraint("exactly 3 bullets", "- one\n  - nested\n- two") is None  # nested list: judge decides


def test_sentence_counts():
    assert check_constraint("exactly 2 sentences", "First one. Second one.") is True
    assert check_constraint("exactly 2 sentences", "One. Two. Three.") is False
    assert check_constraint("exactly 2 sentences", "- One.\n- Two.") is None              # not prose: judge decides


# ---------- overriding the judge ----------

def test_code_check_overrides_the_judge_both_ways():
    constraints = ["under 10 words", "warm tone"]
    broken, checks = apply_code_checks(constraints, "too " * 12, judge_broken=())          # judge said all met
    assert broken == ("under 10 words",)
    assert checks[0] == {"constraint": "under 10 words", "met": False, "judge_said_met": True,
                         "detail": "words=12 (limit < 10)"}
    broken, _ = apply_code_checks(constraints, "short and sweet", judge_broken=("under 10 words",))
    assert broken == ()                                                                    # judge was wrong: met


def test_judge_keeps_its_verdict_on_constraints_it_owns():
    broken, checks = apply_code_checks(["warm tone"], "anything", judge_broken=("warm tone",))
    assert broken == ("warm tone",) and checks == ()


SETTINGS = JudgeSettings(model_name="fake", temperature=0, reasoning_effort="low", max_tokens=500,
                         max_attempts=2, pass_threshold=4)


def test_override_turns_a_pass_into_a_fail_through_the_cap():
    """The judge says every constraint is met and scores 5, but the answer is over the word limit."""
    constraints = ["under 80 words", "mentions 24 hours cold"]
    verdict = json.dumps({"constraints": [{"constraint": c, "met": True} for c in constraints],
                          "score": 5, "reason": "meets all constraints"})
    candidate = "word " * 88 + "keeps drinks cold for 24 hours"
    result = judge([{"role": "user", "content": "x"}], constraints, "ref", candidate, SETTINGS,
                   send=lambda messages, settings: verdict)
    assert result.judge_score == 5
    assert result.score == 3 and result.passed is False
    assert result.broken_constraints == ("under 80 words",)
    assert result.judge_broken_constraints == ()


def test_override_can_also_remove_a_wrong_cap():
    constraints = ["under 50 words"]
    verdict = json.dumps({"constraints": [{"constraint": "under 50 words", "met": False}], "score": 5, "reason": "x"})
    result = judge([{"role": "user", "content": "x"}], constraints, "ref", "A short answer.", SETTINGS,
                   send=lambda messages, settings: verdict)
    assert (result.score, result.passed, result.broken_constraints) == (5, True, ())
