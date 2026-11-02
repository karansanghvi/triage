from app.router.features import (
    FEATURE_NAMES,
    count_constraints,
    count_reasoning_words,
    extract_features,
    format_complexity,
    guess_task_type,
    has_context,
    to_vector,
)

def test_vector_matches_feature_names():
    feats = extract_features("What is the capital of India?")
    assert len(to_vector(feats)) == len(FEATURE_NAMES)

def test_exactly_one_task_flag():
    feats = extract_features("Summarize this.", task_type="summarization")
    assert sum(v for k, v in feats.items() if k.startswith("task_")) == 1.0 
    assert feats["task_summarization"] == 1.0 

def test_unknown_task_type_is_guessed():
    feats = extract_features("Classify the sentiment of this review.", task_type="???")
    assert feats["task_classification"] == 1.0 

def test_reasoning_words():
    assert count_reasoning_words("Compare A and B and justify your answer.") == 2
    assert count_reasoning_words("Fix the spelling.") == 0 

def test_constraints():
    text = "Answer in under 50 words. Do not use jargon.\n- be polite\n- be brief"
    assert count_constraints(text) == 4

def test_format_complexity_levels():
    assert format_complexity("Tell me a joke.") == 0 
    assert format_complexity("Give me a bullet list of tips.") == 1
    assert format_complexity("Put the results in a table.") == 2
    assert format_complexity("Return valid JSON with name and email.") == 3

def test_has_context():
    assert has_context("What is photosynthesis?") == 0 
    article = "Summarize this article:\n" + "The city council met on Tuesday." * 10
    assert has_context(article) == 1

def test_guess_task_type():
    assert guess_task_type("Extract the name and date from this invoice.") == "extraction"
    assert guess_task_type("Write a poem about rain.") == "creative"
    assert guess_task_type("If 3 apples cost $6, how much do 5 cost?") == "reasoning"
    assert guess_task_type("Who wrote Hamlet?") == "qa"
    assert guess_task_type("Write a Python function to reverse a list.") == "coding"