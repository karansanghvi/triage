"""Task-type guesser rules and the read-only agreement script."""
import json
import subprocess
import types

import pytest

from app.registry import PROJECT_ROOT
from app.router.features import FEATURE_VERSION, GUESS_RULES, guess_task_type, prompt_text

BASELINE_REV = "c61a92a"     # the guesser rules this phase keeps (option D was reverted)


def test_feature_version_is_3():
    assert FEATURE_VERSION == "3"


# ---------- identical to the c61a92a rules ----------

def test_guesses_identical_to_baseline_rules_on_all_prompts():
    """Compares prompt TEXT only: no labels, outcomes or splits are read."""
    source = subprocess.run(["git", "show", f"{BASELINE_REV}:app/router/features.py"], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True).stdout
    old = types.ModuleType("features_baseline")
    exec(compile(source, "features_baseline", "exec"), old.__dict__)
    lines = (PROJECT_ROOT / "data" / "prompts.jsonl").read_text(encoding="utf-8").splitlines()
    texts = [prompt_text(json.loads(line)) for line in lines if line.strip()]
    assert len(texts) == 762
    differ = [i for i, t in enumerate(texts) if old.guess_task_type(t) != guess_task_type(t)]
    assert differ == []


# ---------- rule order ----------

def test_rule_order():
    assert [task for task, _ in GUESS_RULES] == [
        "coding", "extraction", "summarization", "classification", "creative", "reasoning"]


@pytest.mark.parametrize("text, expected", [
    ("Extract the fields from this Python script.", "coding"),          # coding beats extraction
    ("Summarize what this function does.", "coding"),                   # coding beats summarization
    ("Summarize the fields of this form.", "extraction"),              # extraction beats summarization
    ("Summarize the reviews and label the sentiment.", "summarization"),  # summarization beats classification
    ("Classify this poem: write a poem about rain.", "classification"),   # classification beats creative
    ("Write a story about how many stars exist.", "creative"),           # creative beats reasoning
    ("How many legs do 3 spiders have?", "reasoning"),
    ("Who painted the Mona Lisa?", "qa"),                               # nothing matches: qa
])
def test_first_matching_rule_wins(text, expected):
    assert guess_task_type(text) == expected


# ---------- false positives found while trying extra coding keywords ----------

@pytest.mark.parametrize("text", [
    "What is the probability of a Type I error?",                       # statistics, not code
    "Everything outside the earth, with the single exception of the earth itself.",
    "How did markets react to the announcement?",
    "Requests are limited per API key.",
    "A TCP connection to a database proxy.",
])
def test_known_false_positives_stay_non_coding(text):
    assert guess_task_type(text) != "coding"


def test_coding_rule_reports_the_trigger():
    rule = dict(GUESS_RULES)["coding"]
    assert rule.search("write a python script").group(0) == "python"
    assert rule.search("tell me a joke") is None


# ---------- agreement script is read-only for data/ ----------

def _snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mtime_ns, p.stat().st_size)
            for p in sorted(root.rglob("*")) if p.is_file()}


def test_agreement_script_never_writes_to_data(tmp_path, monkeypatch):
    from eval import guesser_agreement
    data_dir = PROJECT_ROOT / "data"
    report = tmp_path / "guesser_agreement.json"
    monkeypatch.setattr(guesser_agreement, "REPORT_PATH", report)

    before = _snapshot(data_dir)
    guesser_agreement.main(["--label", "t1"])
    guesser_agreement.main(["--label", "t2", "--compare-to", "t1", "--baseline-rev", BASELINE_REV])
    assert _snapshot(data_dir) == before

    saved = json.loads(report.read_text())
    assert set(saved["runs"]) == {"t1", "t2"}
    assert saved["runs"]["t1"]["split"] == "fit"
    assert saved["runs"]["t2"]["changed_vs"]["prompts"] == []        # reverted rules == baseline rules


def test_agreement_prompts_never_include_test_ids():
    from eval.guesser_agreement import agreement_prompts
    from eval.splits import load_split, load_validation_split
    train, test = load_split()
    val = load_validation_split()
    fit_ids = {p["id"] for p in agreement_prompts("fit")}
    val_ids = {p["id"] for p in agreement_prompts("val")}
    assert not fit_ids & test and not val_ids & test
    assert not fit_ids & val and val_ids <= val


def test_validation_run_cannot_be_repeated(tmp_path, monkeypatch):
    from eval import guesser_agreement
    report = tmp_path / "guesser_agreement.json"
    report.write_text(json.dumps({"runs": {"v": {}}}))
    monkeypatch.setattr(guesser_agreement, "REPORT_PATH", report)
    with pytest.raises(SystemExit, match="one-time"):
        guesser_agreement.main(["--split", "val", "--label", "v"])
