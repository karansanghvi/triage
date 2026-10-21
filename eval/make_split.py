"""Phase 2 Step 5: make the locked 80/20 train/test split of prompt IDs.

Usage:
    python -m eval.make_split

Stratified by (source, task_type) with seed 42, using only each prompt's ID, source and task_type, never
labels or scores. Writes data/splits/train_ids.txt, data/splits/test_ids.txt and data/splits/split_info.json
(with a SHA-256 hash of the sorted test IDs). It refuses to run if any of those files already exists, so the
split can't be redone by accident. Use eval/splits.py to read it.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from app.registry import PROJECT_ROOT
from eval.build_dataset import load_prompts

SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
SEED = 42
TEST_FRACTION = 0.2
FILES = ("train_ids.txt", "test_ids.txt", "split_info.json")


def split_ids(records: list, seed: int = SEED, test_fraction: float = TEST_FRACTION) -> tuple:
    """(train IDs, test IDs), both sorted. records are dicts with id, source and task_type only.

    Within each (source, task_type) stratum, round(test_fraction * size) IDs go to test, so every stratum is
    split as close to 80/20 as its size allows (a stratum of 1 or 2 stays entirely in train).
    """
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate prompt IDs")
    strata = {}
    for r in records:
        strata.setdefault((r["source"], r["task_type"]), []).append(r["id"])
    rng = random.Random(seed)
    train, test = [], []
    for key in sorted(strata):                      # fixed order, so one seeded generator is reproducible
        members = sorted(strata[key])
        rng.shuffle(members)
        n_test = round(test_fraction * len(members))
        test += members[:n_test]
        train += members[n_test:]
    return sorted(train), sorted(test)


def test_hash(test_ids: list) -> str:
    return hashlib.sha256("\n".join(sorted(test_ids)).encode("utf-8")).hexdigest()


def split_info(records: list, train: list, test: list, seed: int, test_fraction: float) -> dict:
    stratum = {r["id"]: f"{r['source']}/{r['task_type']}" for r in records}
    train_counts, test_counts = Counter(stratum[i] for i in train), Counter(stratum[i] for i in test)
    return {
        "seed": seed,
        "test_fraction": test_fraction,
        "stratified_by": ["source", "task_type"],
        "total": len(train) + len(test),
        "train": len(train),
        "test": len(test),
        "counts": {key: {"train": train_counts[key], "test": test_counts[key]} for key in sorted(set(stratum.values()))},
        "test_ids_sha256": test_hash(test),
    }


def write_split(records: list, out_dir: Path = SPLITS_DIR, seed: int = SEED, test_fraction: float = TEST_FRACTION) -> dict:
    """Write the three split files. Raises FileExistsError if any already exists."""
    existing = [name for name in FILES if (out_dir / name).exists()]
    if existing:
        raise FileExistsError(f"split already exists ({', '.join(existing)} in {out_dir}); "
                              "the split is locked and must not be redone")
    train, test = split_ids(records, seed, test_fraction)
    info = split_info(records, train, test, seed, test_fraction)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train_ids.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (out_dir / "test_ids.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
    (out_dir / "split_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main():
    records = [{"id": p["id"], "source": p["source"], "task_type": p["task_type"]} for p in load_prompts()]
    try:
        info = write_split(records)
    except FileExistsError as error:
        raise SystemExit(str(error))
    print(f"Split {info['total']} prompts: {info['train']} train, {info['test']} test (seed {info['seed']})")
    print(f"test IDs SHA-256: {info['test_ids_sha256']}")
    print(f"\n{'source/task_type':<26}{'train':>7}{'test':>6}")
    for key, c in info["counts"].items():
        print(f"{key:<26}{c['train']:>7}{c['test']:>6}")


if __name__ == "__main__":
    main()
