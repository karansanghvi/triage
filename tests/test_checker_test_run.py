"""Guards and item selection for the Phase 4 final test run (eval/checker_test_run.py). No API calls, and no
real test data: every test uses made-up prompts and answers."""
import json
from types import SimpleNamespace

import pytest

from eval import checker_test_run as ctr

SETTINGS = SimpleNamespace(model_name="groq-small", checker_version=3)
ASK = [{"role": "user", "content": "What is 2 + 3? End with: Answer: <number>"}]
RAIN = [{"role": "user", "content": "Write two sentences about rain."}]


def inputs():
    prompts = {"a": {"messages": RAIN}, "b": {"messages": RAIN}, "c": {"messages": ASK}, "d": {"messages": RAIN}}
    answers = {("a", "medium"): {"answer": "Rain falls. It is wet."},
               ("b", "medium"): {"answer": "Rain falls. It is wet."},
               ("c", "medium"): {"answer": "five"},                        # fails the format check
               }                                                           # d: no medium answer
    probs = {"a": (0.2, 0.50), "b": (0.2, 0.90), "c": (0.2, 0.10), "d": (0.2, 0.10)}
    return ["a", "b", "c", "d"], prompts, answers, probs


def test_only_answers_the_strategy_would_check_are_selected():
    items, skipped = ctr.strategy_items(*inputs())
    assert items == [{"prompt_id": "a", "tier": "medium"}]               # b is exactly at 0.90: skipped
    assert sum(skipped.values()) == 3 and any("0.90" in k for k in skipped)


def test_the_first_real_run_writes_the_preregistration_and_a_resume_must_match(tmp_path):
    prereg, marker = tmp_path / "prereg.json", tmp_path / "done.json"
    items = [{"prompt_id": "a", "tier": "medium"}]
    ctr.check_guards(items, SETTINGS, prereg_path=prereg, marker=marker, real=False)
    assert not prereg.exists()                                           # a dry run never registers
    ctr.check_guards(items, SETTINGS, prereg_path=prereg, marker=marker, real=True)
    saved = json.loads(prereg.read_text())
    assert saved["strategy"] == ctr.STRATEGY and saved["items"] == ["a"] and "registered_at" in saved
    ctr.check_guards(items, SETTINGS, prereg_path=prereg, marker=marker, real=True)      # a resume: same items
    with pytest.raises(ctr.FinalRunRefused, match="items"):
        ctr.check_guards(items + [{"prompt_id": "z", "tier": "medium"}], SETTINGS, prereg_path=prereg, marker=marker,
                         real=True)


def test_it_refuses_after_the_final_evaluation_ran(tmp_path):
    marker = tmp_path / "done.json"
    marker.write_text('{"ran_at": "x"}')
    with pytest.raises(ctr.FinalRunRefused, match="already ran"):
        ctr.check_guards([], SETTINGS, prereg_path=tmp_path / "p.json", marker=marker, real=False)


def test_it_refuses_a_different_checker(tmp_path):
    with pytest.raises(ctr.FinalRunRefused, match="pre-registered"):
        ctr.check_guards([], SimpleNamespace(model_name="groq-small", checker_version=4),
                         prereg_path=tmp_path / "p.json", marker=tmp_path / "d.json", real=False)


def test_final_flag_is_required():
    with pytest.raises(SystemExit, match="--final"):
        ctr.main([])


def test_labeled_test_ids_reads_only_labels_and_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(ctr, "load_split", lambda: ({"t1"}, {"x1", "x2", "x3"}))
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(o) for o in [
        {"id": "x1", "label_status": "labeled", "medium_pass": True},
        {"id": "x2", "label_status": "unlabeled"},
        {"id": "t1", "label_status": "labeled"},
        {"id": "x3", "label_status": "labeled", "medium_pass": False}]))
    assert ctr.labeled_test_ids(path) == ["x1", "x3"]
