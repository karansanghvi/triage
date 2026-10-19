import json
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


# ---------- exact_match ----------

from app.verifier.scorers import (  # noqa: E402
    score_exact_match, score_field_check, score_label_match, score_record,
)


@pytest.mark.parametrize("output, reference", [
    ("14", "14"),
    ("  14 \n", "14"),                          # surrounding whitespace
    ("14.", "14"),                              # one trailing period
    ("YES", "yes"),                             # case-insensitive
    ("Ctrl + Shift + E", "Ctrl+Shift+E"),       # spaces around +
    ("O-101, O-102,O-105", "O-101,O-102,O-105"),  # spaces around , and -
    ("08 : 50", "08:50"),                       # spaces around :
    ("2026 - 05 - 25", "2026-05-25"),
    ("not stated", "not stated"),
    ("DHL", "DHL"),
    ("Not stated.", "not stated"),              # case and trailing period together
])
def test_exact_match_passes_after_normalization(output, reference):
    assert score_exact_match(output, reference).passed


@pytest.mark.parametrize("output, reference", [
    ("May 25, 2026", "2026-05-25"),   # dates must match the requested format
    ("8:50", "08:50"),                # so must times
    ("14 days", "14"),                # extra words
    ("14..", "14"),                   # only one trailing period is ignored
    ("$60.00", "60.00"),              # no currency stripping
    ("**14**", "14"),                 # no markdown stripping
])
def test_exact_match_fails_on_anything_else(output, reference):
    assert not score_exact_match(output, reference).passed


# ---------- label_match ----------

@pytest.mark.parametrize("output, passed", [
    ("billing", True),
    ("Billing", True),
    (" billing.\n", True),
    ("open_ended", False),
    ("billing, technical", False),
    ("The label is billing", False),
    ("", False),
])
def test_label_match(output, passed):
    assert score_label_match(output, "billing").passed is passed


# ---------- field_check ----------

INVOICE = {
    "invoice_number": "AN-2026-311",
    "vendor": "Harbor Line Printing Co.",
    "issue_date": "2026-10-05",
    "total": 1881.0,
    "paid": False,
    "po": None,
    "address": {"street": "77 Harrow Court, Unit 12", "zip": "53711"},
    "line_items": [
        {"description": "Window display design", "quantity": 1},
        {"description": "Printed vinyl panels", "quantity": 8},
    ],
    "skills": ["Go", "PostgreSQL", "Tri-County"],
}
FUZZY = ["vendor", "address.street", "line_items.description", "skills"]


def answer(**changes):
    data = json.loads(json.dumps(INVOICE))
    for path, value in changes.items():
        target = data
        *parents, last = path.split("__")
        for key in parents:
            target = target[int(key)] if key.isdigit() else target[key]
        if value is DELETE:
            del target[int(last) if last.isdigit() else last]
        else:
            target[int(last) if last.isdigit() else last] = value
    return json.dumps(data)


DELETE = object()


def test_field_check_exact_copy_passes():
    assert score_field_check(json.dumps(INVOICE), INVOICE, FUZZY).passed


@pytest.mark.parametrize("changes", [
    {"vendor": "harbor line printing co"},                  # fuzzy: case and punctuation
    {"vendor": "  Harbor   Line Printing Co  "},            # fuzzy: whitespace
    {"address__street": "77 harrow court unit 12"},         # fuzzy nested path
    {"line_items__0__description": "window DISPLAY design"},  # fuzzy path through a list
    {"skills": ["postgresql", "tri county", "GO"]},          # fuzzy list: unordered set, punctuation
    {"total": 1881},                                         # number compared by value
])
def test_field_check_tolerated_differences(changes):
    result = score_field_check(answer(**changes), INVOICE, FUZZY)
    assert result.passed, result.reason


@pytest.mark.parametrize("changes, reason_part", [
    ({"invoice_number": "an-2026-311"}, "invoice_number"),        # strict ID: case matters
    ({"issue_date": "05.10.2026"}, "issue_date"),                  # strict date
    ({"total": 1881.01}, "total"),                                 # strict amount
    ({"total": "1881.0"}, "total"),                                # number as string
    ({"paid": 0}, "paid"),                                         # boolean vs number
    ({"paid": "false"}, "paid"),                                   # boolean as string
    ({"po": ""}, "po"),                                            # null must stay null
    ({"address__zip": 53711}, "address.zip"),                      # strict nested field type
    ({"line_items__1__quantity": 9}, "line_items.quantity"),       # strict field inside a list
    ({"skills": ["Go", "PostgreSQL"]}, "skills"),                  # fuzzy set still needs every item
    ({"vendor": "Harbour Line Printing"}, "vendor"),               # fuzzy is not spelling-tolerant
    ({"vendor": DELETE}, "missing keys"),
    ({"extra_field": 1}, "unexpected keys"),
])
def test_field_check_failures(changes, reason_part):
    result = score_field_check(answer(**changes), INVOICE, FUZZY)
    assert not result.passed
    assert reason_part in result.reason


def test_field_check_list_order_is_strict_for_objects():
    swapped = dict(INVOICE, line_items=list(reversed(INVOICE["line_items"])))
    assert not score_field_check(json.dumps(swapped), INVOICE, FUZZY).passed


def test_field_check_without_fuzzy_fields_is_fully_strict():
    assert not score_field_check(answer(vendor="harbor line printing co"), INVOICE).passed


def test_field_check_accepts_a_json_code_fence():
    assert score_field_check("```json\n" + json.dumps(INVOICE) + "\n```", INVOICE, FUZZY).passed
    assert score_field_check("```\n" + json.dumps(INVOICE) + "\n```", INVOICE, FUZZY).passed


@pytest.mark.parametrize("output", [
    "Here is the JSON: " + json.dumps(INVOICE),   # prose around the JSON
    "{'vendor': 'x'}",                           # Python dict, not JSON
    "",
    "[1, 2]",                                    # valid JSON but not the right shape
])
def test_field_check_rejects_non_json_or_wrong_shape(output):
    assert not score_field_check(output, INVOICE, FUZZY).passed


def test_field_check_fuzzy_null_must_be_null():
    reference = {"candidate_phone": None, "name": "Rosa"}
    assert score_field_check('{"candidate_phone": null, "name": "rosa"}', reference, ["candidate_phone", "name"]).passed
    assert not score_field_check('{"candidate_phone": "555-0148", "name": "Rosa"}', reference, ["candidate_phone"]).passed


# ---------- dispatch ----------

def test_score_record_dispatches_by_scoring_method():
    assert score_record({"scoring": "exact_match", "reference": "14"}, "14.").passed
    assert score_record({"scoring": "label_match", "reference": "billing"}, "Billing").passed
    assert score_record({"scoring": "final_answer", "reference": "18"}, "x\nAnswer: 18").passed
    assert score_record({"scoring": "field_check", "reference": {"a": "X"}, "fuzzy_fields": ["a"]}, '{"a": "x"}').passed
    assert score_record({"scoring": "judge", "reference": None}, "anything") is None


# ---------- Unicode spaces and dashes ----------

from app.verifier.scorers import normalize_unicode  # noqa: E402

NBSP, NARROW, THIN = " ", " ", " "
HYPHEN, NB_HYPHEN, FIGURE, EN, EM, MINUS = "‐", "‑", "‒", "–", "—", "−"


def test_normalize_unicode_maps_every_listed_character():
    assert normalize_unicode(f"a{NBSP}b{NARROW}c{THIN}d") == "a b c d"
    assert normalize_unicode(f"{HYPHEN}{NB_HYPHEN}{FIGURE}{EN}{EM}{MINUS}") == "------"


def test_normalize_unicode_leaves_other_characters_alone():
    # horizontal bar, fullwidth hyphen-minus, ideographic space, accented letters
    assert normalize_unicode("―－　Tomás") == "―－　Tomás"


@pytest.mark.parametrize("output, reference", [
    (f"2026{NB_HYPHEN}10{NB_HYPHEN}19{NARROW}15:30", "2026-10-19 15:30"),
    (f"O{EN}101,O{EN}102", "O-101,O-102"),
    (f"{MINUS}42", "-42"),
    (f"Ctrl{NBSP}+{NBSP}Shift{THIN}+E", "Ctrl+Shift+E"),
])
def test_exact_match_normalizes_unicode(output, reference):
    assert score_exact_match(output, reference).passed


def test_exact_match_unicode_does_not_hide_a_wrong_value():
    assert not score_exact_match(f"2026{NB_HYPHEN}10{NB_HYPHEN}18{NARROW}15:30", "2026-10-19 15:30").passed


def test_label_match_normalizes_unicode():
    assert score_label_match(f"{NBSP}wrong_item{NARROW}", "wrong_item").passed


@pytest.mark.parametrize("output, reference", [
    (f"Answer:{NARROW}{MINUS}18", "-18"),
    (f"Answer: {EN}0.25", "-0.25"),
    (f"Answer:{NBSP}$1,800", "1800"),
    (f"Answer:{NBSP}2026{NB_HYPHEN}05{NB_HYPHEN}25", "2026-05-25"),
    (f"Answer:{NARROW}B", "B"),
])
def test_final_answer_normalizes_unicode(output, reference):
    assert score_final_answer(output, reference).passed


def test_field_check_normalizes_unicode_in_strict_fields():
    reference = {"tracking": "SHP-90217", "date": "2026-10-05", "range": "$112,000-$128,000", "name": "Tri-County Freight"}
    output = json.dumps({"tracking": f"SHP{NB_HYPHEN}90217", "date": f"2026{EN}10{EN}05",
                         "range": f"$112,000{EM}$128,000", "name": f"Tri{NB_HYPHEN}County{NBSP}Freight"})
    assert score_field_check(output, reference, ["name"]).passed
    # still strict: a different value fails even after normalization
    wrong = json.dumps({"tracking": f"SHP{NB_HYPHEN}90218", "date": "2026-10-05", "range": "$112,000-$128,000",
                        "name": "Tri-County Freight"})
    assert not score_field_check(wrong, reference, ["name"]).passed


def test_field_check_normalizes_unicode_in_fuzzy_lists():
    reference = {"skills": ["A/B testing", "Go"]}
    assert score_field_check(json.dumps({"skills": ["go", f"A/B{NBSP}testing"]}), reference, ["skills"]).passed
