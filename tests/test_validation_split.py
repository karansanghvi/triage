import json

import pytest

from eval.make_split import write_split
from eval.make_validation_split import labeled_train_records, write_validation_split
from eval.splits import TestSetLeakError, fit_ids, load_split, load_validation_split

SIZES = {("own", "qa"): 27, ("own", "creative"): 27, ("gsm8k", "reasoning"): 200,
         ("arena", "coding"): 47, ("arena", "classification"): 1}


def all_records():
    out = []
    for (source, task), n in SIZES.items():
        out += [{"id": f"{source}_{task}_{i:04d}", "source": source, "task_type": task} for i in range(n)]
    return out


@pytest.fixture
def splits(tmp_path):
    """A train/test split in tmp_path, plus outcomes where every 10th prompt is unlabeled."""
    write_split(all_records(), tmp_path)
    outcomes = tmp_path / "outcomes.jsonl"
    outcomes.write_text("\n".join(json.dumps({**r, "label_status": "excluded" if n % 10 == 0 else "labeled"})
                                  for n, r in enumerate(all_records())) + "\n")
    return tmp_path, outcomes


def test_pool_is_labeled_train_only(splits):
    d, outcomes = splits
    train, test = load_split(d)
    pool = {r["id"] for r in labeled_train_records(outcomes, d)}
    assert pool <= train and not pool & test
    assert len(pool) < len(train)                       # unlabeled train prompts are left out


def test_val_is_subset_of_train_with_no_test_overlap(splits):
    d, outcomes = splits
    info = write_validation_split(labeled_train_records(outcomes, d), d)
    train, test = load_split(d)
    val = load_validation_split(d)
    assert val <= train
    assert not val & test
    assert len(val) == info["val"] and info["fit"] + info["val"] == info["pool_size"]


def test_val_is_stratified_20_percent(splits):
    d, outcomes = splits
    info = write_validation_split(labeled_train_records(outcomes, d), d)
    for key, c in info["counts"].items():
        assert c["val"] == round(0.2 * (c["fit"] + c["val"])), key


def test_fit_set_is_pool_minus_val(splits):
    d, outcomes = splits
    pool = {r["id"] for r in labeled_train_records(outcomes, d)}
    write_validation_split(labeled_train_records(outcomes, d), d)
    val = load_validation_split(d)
    assert fit_ids(pool, d) == pool - val
    _, test = load_split(d)
    with pytest.raises(TestSetLeakError):
        fit_ids(pool | {sorted(test)[0]}, d)


def test_refuses_to_overwrite_validation_split(splits):
    d, outcomes = splits
    records = labeled_train_records(outcomes, d)
    write_validation_split(records, d)
    before = (d / "val_ids.txt").read_text()
    with pytest.raises(FileExistsError, match="locked"):
        write_validation_split(records, d, seed=7)
    assert (d / "val_ids.txt").read_text() == before


def test_refuses_if_only_val_info_exists(splits):
    d, outcomes = splits
    (d / "val_info.json").write_text("{}")
    with pytest.raises(FileExistsError):
        write_validation_split(labeled_train_records(outcomes, d), d)


def test_does_not_touch_train_test_files(splits):
    d, outcomes = splits
    before = {n: (d / n).read_text() for n in ("train_ids.txt", "test_ids.txt", "split_info.json")}
    write_validation_split(labeled_train_records(outcomes, d), d)
    assert {n: (d / n).read_text() for n in before} == before


def test_tampered_val_file_is_detected(splits):
    d, outcomes = splits
    write_validation_split(labeled_train_records(outcomes, d), d)
    ids = (d / "val_ids.txt").read_text().split()
    (d / "val_ids.txt").write_text("\n".join(ids[1:]) + "\n")
    with pytest.raises(ValueError, match="hash"):
        load_validation_split(d)


def test_val_containing_a_test_id_is_rejected(splits):
    """Even with a matching hash, a val file that includes a test ID must be refused."""
    import hashlib
    d, outcomes = splits
    write_validation_split(labeled_train_records(outcomes, d), d)
    _, test = load_split(d)
    ids = sorted((d / "val_ids.txt").read_text().split() + [sorted(test)[0]])
    (d / "val_ids.txt").write_text("\n".join(ids) + "\n")
    info = json.loads((d / "val_info.json").read_text())
    info["val"], info["val_ids_sha256"] = len(ids), hashlib.sha256("\n".join(ids).encode()).hexdigest()
    (d / "val_info.json").write_text(json.dumps(info))
    with pytest.raises(TestSetLeakError):
        load_validation_split(d)


def test_real_validation_split_is_consistent():
    from eval.train_classifier import load_training_rows
    train, test = load_split()
    val = load_validation_split()
    assert val <= train and not val & test
    fit_rows, all_rows = load_training_rows(), load_training_rows(include_val=True)
    fit, everything = {o["id"] for o in fit_rows}, {o["id"] for o in all_rows}
    assert len(everything) == 598
    assert fit == everything - val and val <= everything
    assert not (fit | val) & test


def test_routing_sim_fit_and_val_splits():
    from eval.routing_sim import load_eval_rows
    train, test = load_split()
    val = load_validation_split()
    fit_rows, val_rows = load_eval_rows("fit"), load_eval_rows("val", allow_val=True)
    fit, v = {r.id for r in fit_rows}, {r.id for r in val_rows}
    assert (len(fit), len(v)) == (481, 117)
    assert v <= val and not fit & val and not (fit | v) & test
    assert fit | v == {r.id for r in load_eval_rows("train")}
