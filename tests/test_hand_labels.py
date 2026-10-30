from eval.hand_labels import check_labels, compare_labels, select_prompts

IDS = ["own_qa_001", "own_creative_002", "own_extraction_003"]


def test_complete_labels_pass_and_notes_are_ignored():
    text = "# instructions\n\nown_qa_001 low\nown_creative_002 HIGH  # tough tone rules\nown_extraction_003 medium\n"
    problems, labels = check_labels(IDS, text)
    assert problems == []
    assert labels == {"own_qa_001": "low", "own_creative_002": "high", "own_extraction_003": "medium"}


def test_blank_labels_are_reported():
    problems, _ = check_labels(IDS, "own_qa_001 \nown_creative_002 low\nown_extraction_003 low\n")
    assert problems == ["line 1: own_qa_001 has no label yet"]


def test_bad_label_missing_extra_and_duplicate_ids():
    text = "own_qa_001 hard\nown_creative_002 low\nown_creative_002 low\nown_qa_999 low\n"
    problems, _ = check_labels(IDS, text)
    assert any("must be low, medium or high" in p for p in problems)
    assert any("listed twice" in p for p in problems)
    assert any("own_qa_999 is not one of the prompts" in p for p in problems)
    assert any("own_extraction_003 is missing" in p for p in problems)


def test_selection_is_stratified_excludes_seen_and_is_reproducible():
    types = ["extraction", "classification", "summarization", "qa", "reasoning", "creative"]
    prompts = [{"id": f"own_{t}_{n:03d}", "source": "own", "task_type": t} for t in types for n in range(1, 21)]
    seen = {"own_qa_001", "own_qa_002"}
    first = select_prompts(prompts, seen, size=50, seed=42)
    assert first == select_prompts(prompts, seen, size=50, seed=42)
    assert len({p["id"] for p in first}) == 50
    assert not seen & {p["id"] for p in first}
    counts = sorted(sum(p["task_type"] == t for p in first) for t in types)
    assert counts == [8, 8, 8, 8, 9, 9]


def test_compare_counts_direction_and_skips_excluded():
    outcomes = {
        "own_qa_001": {"label": "low", "task_type": "qa"},
        "own_qa_002": {"label": "medium", "task_type": "qa"},
        "own_creative_001": {"label": "high", "task_type": "creative"},
        "own_creative_002": {"label": None, "task_type": "creative"},
    }
    hand = {"own_qa_001": "low", "own_qa_002": "high", "own_creative_001": "medium", "own_creative_002": "low"}
    result = compare_labels(hand, outcomes)
    assert result["counts"] == {"match": 1, "too high": 1, "too low": 1}
    assert result["unlabeled"] == ["own_creative_002"]
    assert result["confusion"]["high"]["medium"] == 1 and result["confusion"]["medium"]["high"] == 1
