import json

import pytest

from eval.make_split import split_ids, write_split
from eval.splits import TestSetLeakError, assert_no_test_ids, load_split

SOURCES = {"own": ["qa", "creative"], "gsm8k": ["reasoning"], "arena": ["coding", "classification"]}


def records():
    out = []
    sizes = {("own", "qa"): 27, ("own", "creative"): 27, ("gsm8k", "reasoning"): 200,
             ("arena", "coding"): 47, ("arena", "classification"): 1}
    for (source, task), n in sizes.items():
        out += [{"id": f"{source}_{task}_{i:04d}", "source": source, "task_type": task} for i in range(n)]
    return out


def test_train_and_test_cover_everything_without_overlap():
    train, test = split_ids(records())
    assert not set(train) & set(test)
    assert sorted(train + test) == sorted(r["id"] for r in records())


def test_split_is_stratified_80_20():
    train, test = split_ids(records())
    stratum = {r["id"]: (r["source"], r["task_type"]) for r in records()}
    count = lambda ids, key: sum(stratum[i] == key for i in ids)
    assert count(test, ("gsm8k", "reasoning")) == 40          # 20% of 200
    assert count(test, ("own", "qa")) == 5                     # round(5.4)
    assert count(test, ("arena", "coding")) == 9               # round(9.4)
    assert count(test, ("arena", "classification")) == 0       # a stratum of 1 stays in train


def test_split_is_reproducible_and_seed_dependent():
    assert split_ids(records(), seed=42) == split_ids(records(), seed=42)
    assert split_ids(records(), seed=42) != split_ids(records(), seed=7)


def test_input_order_does_not_change_the_split():
    assert split_ids(list(reversed(records()))) == split_ids(records())


def test_split_never_reads_labels_or_scores():
    with_labels = [dict(r, label="high", score=1) for r in records()]
    assert split_ids(with_labels) == split_ids(records())


def test_refuses_to_overwrite_an_existing_split(tmp_path):
    write_split(records(), tmp_path)
    before = (tmp_path / "test_ids.txt").read_text()
    with pytest.raises(FileExistsError, match="locked"):
        write_split(records(), tmp_path, seed=7)
    assert (tmp_path / "test_ids.txt").read_text() == before


def test_refuses_if_even_one_split_file_exists(tmp_path):
    (tmp_path / "split_info.json").write_text("{}")
    with pytest.raises(FileExistsError):
        write_split(records(), tmp_path)


def test_load_split_round_trips_and_guard_raises_on_test_ids(tmp_path):
    info = write_split(records(), tmp_path)
    train, test = load_split(tmp_path)
    assert (len(train), len(test)) == (info["train"], info["test"])
    assert_no_test_ids(sorted(train)[:20], tmp_path)                          # training IDs are fine
    with pytest.raises(TestSetLeakError, match="1 test-set prompt IDs"):
        assert_no_test_ids(sorted(train)[:5] + [sorted(test)[0]], tmp_path)


def test_tampered_test_file_is_detected(tmp_path):
    write_split(records(), tmp_path)
    path = tmp_path / "test_ids.txt"
    ids = path.read_text().split()
    path.write_text("\n".join(ids[1:]) + "\n")                                 # quietly drop one test ID
    with pytest.raises(ValueError, match="hash"):
        load_split(tmp_path)


def test_real_locked_split_is_consistent_with_prompts():
    from eval.build_dataset import load_prompts
    train, test = load_split()
    prompt_ids = {p["id"] for p in load_prompts()}
    assert train | test == prompt_ids
    assert not train & test
    info = json.loads((__import__("eval.splits", fromlist=["SPLITS_DIR"]).SPLITS_DIR / "split_info.json").read_text())
    assert info["seed"] == 42
