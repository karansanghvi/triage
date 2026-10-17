from decimal import Decimal

import pytest

from app.verifier.scorers import extract_final_answer, parse_number, score_final_answer

REASONING = "First, 3 boxes of 6 is 18.\nThen nothing else changes.\n"


@pytest.mark.parametrize("output, reference", [
    (REASONING + "Answer: 18", "18"),                       # plain number
    (REASONING + "Answer: 18.0", "18"),                     # decimal form of an integer
    (REASONING + "Answer: 18", "18.00"),                    # reference written with decimals
    (REASONING + "Answer: 2.75", "2.75"),                   # real decimal
    (REASONING + "Answer: .5", "0.5"),                      # leading-dot decimal
    (REASONING + "Answer: -42", "-42"),                     # negative
    (REASONING + "Answer: -0.25", "-0.25"),                 # negative decimal
    (REASONING + "Answer: -$7.50", "-7.5"),                 # negative with a dollar sign
    (REASONING + "Answer: 1,800", "1800"),                  # thousands comma in the answer
    (REASONING + "Answer: 1800", "1,800"),                  # thousands comma in the reference
    (REASONING + "Answer: 1,234,567.89", "1234567.89"),     # several commas plus decimals
    (REASONING + "Answer: $18", "18"),                      # dollar sign
    (REASONING + "Answer: $1,545.00", "1545.00"),           # dollar sign, comma, and decimals
    (REASONING + "Answer: 18 apples", "18"),                # trailing unit
    (REASONING + "Answer: 18 km/h.", "18"),                 # unit plus trailing period
    (REASONING + "Answer: about 18 apples in total", "18"), # words before and after
    (REASONING + "answer: 18", "18"),                       # lowercase marker
    (REASONING + "ANSWER: 18", "18"),                       # uppercase marker
    (REASONING + "**Answer:** 18", "18"),                   # markdown bold around the marker
    (REASONING + "Final Answer: 18", "18"),                 # marker inside a longer label
    ("Answer: 18", "18"),                                   # no reasoning at all
    ("Answer: 18\nHope that helps!", "18"),                 # text on the following line is ignored
])
def test_numeric_answers_that_pass(output, reference):
    result = score_final_answer(output, reference)
    assert result.passed, result.reason


def test_last_answer_line_wins():
    output = "My first guess.\nAnswer: 12\nWait, I forgot the tax.\nAnswer: 18"
    assert score_final_answer(output, "18").passed
    assert not score_final_answer(output, "12").passed


def test_earlier_mention_of_answer_does_not_count_over_the_final_line():
    output = "The answer: probably 12, but let me check.\nRecomputing...\nAnswer: 18"
    assert extract_final_answer(output) == "18"


def test_missing_answer_line_fails():
    result = score_final_answer("3 boxes of 6 is 18, so the total is 18.", "18")
    assert not result.passed
    assert result.extracted is None
    assert result.reason == "no 'Answer:' line"


def test_answer_line_with_no_number_fails():
    result = score_final_answer(REASONING + "Answer: eighteen", "18")
    assert not result.passed
    assert result.reason == "no number after 'Answer:'"


@pytest.mark.parametrize("output", [
    REASONING + "Answer:\n18",                       # value on the next line
    REASONING + "**Answer:**\n18",                   # bold marker, value below
    REASONING + "**Answer:**\n\n18",                 # blank line, then the value
    REASONING + "Answer:   \n  \n\n**18**",          # several blank lines, bold value
    REASONING + "**Answer:**\r\n\r\n$18 total",      # Windows line endings
])
def test_value_on_a_later_line_when_answer_line_is_empty(output):
    result = score_final_answer(output, "18")
    assert result.passed, result.reason


def test_same_line_value_takes_priority_over_next_line():
    assert extract_final_answer("Answer: 18\n20") == "18"


def test_only_the_next_non_empty_line_is_used():
    # The line after the marker has no number, so it fails; later lines aren't searched.
    result = score_final_answer("**Answer:**\n\nI'm not sure.\n18", "18")
    assert not result.passed
    assert result.extracted == "I'm not sure."


@pytest.mark.parametrize("output", [
    REASONING + "Answer:",              # nothing after the marker at all
    REASONING + "**Answer:**\n\n   \n",  # only blank lines after it
])
def test_nothing_after_answer_fails(output):
    result = score_final_answer(output, "18")
    assert not result.passed
    assert result.extracted == ""


@pytest.mark.parametrize("output, reference", [
    (REASONING + "Answer: 17", "18"),          # wrong number
    (REASONING + "Answer: 18.5", "18"),        # close but not equal
    (REASONING + "Answer: -18", "18"),         # wrong sign
    (REASONING + "Answer: 180", "18"),         # extra digit
    (REASONING + "Answer: 1,800", "18"),       # commas don't make 1,800 equal 18
])
def test_wrong_numbers_fail(output, reference):
    result = score_final_answer(output, reference)
    assert not result.passed


def test_first_number_after_marker_is_graded():
    # "first number" rule: the 3 is graded, not the 18
    assert not score_final_answer("Answer: 3 boxes, so 18", "18").passed


@pytest.mark.parametrize("text, expected", [
    ("18", Decimal("18")),
    ("$1,800.50 total", Decimal("1800.50")),
    ("-$7", Decimal("-7")),
    ("−12", Decimal("-12")),   # Unicode minus sign
    ("no digits here", None),
])
def test_parse_number(text, expected):
    assert parse_number(text) == expected


@pytest.mark.parametrize("value", [
    "B", "B)", "(B)", "B) Paris", "**B**", "B.", "Option B", "The answer is B", "[B]", "b", "(b).",
])
def test_choice_letter_forms_that_pass(value):
    result = score_final_answer(f"Thinking it over.\nAnswer: {value}", "B")
    assert result.passed, result.reason
    assert result.extracted == "B"


def test_choice_letter_on_the_next_line():
    assert score_final_answer("**Answer:**\n\n(B) Paris", "B").passed


def test_letters_in_the_reasoning_are_never_used():
    # "A" in "A dog" appears before the Answer line; only the Answer value counts.
    output = "A dog is a mammal, so option A looks tempting.\nAnswer: B"
    assert score_final_answer(output, "B").passed
    assert not score_final_answer(output, "A").passed


def test_reasoning_letter_does_not_rescue_a_missing_answer_line():
    assert not score_final_answer("A dog is a mammal. I pick B.", "B").passed


@pytest.mark.parametrize("value, reason", [
    ("Paris", "no A-D letter after 'Answer:'"),   # no letter at all
    ("E", "no A-D letter after 'Answer:'"),       # outside A-D
    ("Bob", "no A-D letter after 'Answer:'"),     # capital B inside a word isn't standalone
])
def test_choice_answers_without_a_letter_fail(value, reason):
    result = score_final_answer(f"Answer: {value}", "B")
    assert not result.passed
    assert result.reason == reason


@pytest.mark.parametrize("value", ["C", "C) Paris", "(D)"])
def test_wrong_choice_letter_fails(value):
    assert not score_final_answer(f"Answer: {value}", "B").passed


def test_first_standalone_letter_wins():
    # "I think C, not B": C is the first standalone A-D letter ("I" isn't A-D).
    assert score_final_answer("Answer: I think C, not B", "C").passed


@pytest.mark.parametrize("output, reference, passed", [
    ("Answer: B", "B", True),
    ("answer: b.", "B", True),
    ("Answer: 2026-05-25", "2026-05-25", True),
    ("Answer: C", "B", False),
    ("Answer:", "B", False),
])
def test_text_references_use_exact_match_normalization(output, reference, passed):
    assert score_final_answer(output, reference).passed is passed
