"""Read the locked train/test split, and guard training and tuning code against test-set leaks.

Every piece of code that trains, tunes or selects anything must pass the prompt IDs it uses through
assert_no_test_ids() first:

    from eval.splits import load_split, assert_no_test_ids
    train_ids, test_ids = load_split()
    assert_no_test_ids(ids_used_for_training)
"""
import hashlib
import json
from pathlib import Path

from app.registry import PROJECT_ROOT

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"


class TestSetLeakError(ValueError):
    """Raised when a test-set prompt ID is used where only training IDs are allowed."""

    __test__ = False   # not a pytest test class, despite the name


def _read_ids(path: Path) -> set:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def load_split(splits_dir: Path = SPLITS_DIR) -> tuple:
    """(train IDs, test IDs) as sets. Raises if the files are missing, overlap, or don't match the recorded hash."""
    train = _read_ids(splits_dir / "train_ids.txt")
    test = _read_ids(splits_dir / "test_ids.txt")
    info = json.loads((splits_dir / "split_info.json").read_text(encoding="utf-8"))
    if train & test:
        raise ValueError(f"train and test overlap: {sorted(train & test)[:5]}")
    digest = hashlib.sha256("\n".join(sorted(test)).encode("utf-8")).hexdigest()
    if digest != info["test_ids_sha256"]:
        raise ValueError("test_ids.txt doesn't match the hash in split_info.json; the locked split was changed")
    if (len(train), len(test)) != (info["train"], info["test"]):
        raise ValueError("split sizes don't match split_info.json")
    return train, test


def assert_no_test_ids(ids, splits_dir: Path = SPLITS_DIR) -> None:
    """Raise TestSetLeakError if any of ids is in the test set."""
    _, test = load_split(splits_dir)
    leaked = sorted(set(ids) & test)
    if leaked:
        raise TestSetLeakError(f"{len(leaked)} test-set prompt IDs passed to training/tuning code: "
                               f"{', '.join(leaked[:10])}" + (" ..." if len(leaked) > 10 else ""))
