"""Phase 4 Step 0: carve a locked validation set out of the labeled TRAIN prompts.

Usage:
    python -m eval.make_validation_split

Pool: the labeled train prompts (label_status == "labeled" and in data/splits/train_ids.txt), the same
598 prompts eval/train_classifier.py trains on. 20% of them, stratified by (source, task_type) with seed 42,
go to validation, using make_split.split_ids (the same per-stratum shuffle as the train/test split).

Writes data/splits/val_ids.txt and data/splits/val_info.json (with a SHA-256 hash of the sorted val IDs).
Never touches train_ids.txt, test_ids.txt or split_info.json, and refuses to run if a val file already
exists, so the validation set can't be redone by accident. Use eval/splits.py to read it.

    fit set = labeled train - val    (what classifiers are trained on by default)
    val set                          (where tau presets and classifier choices are tuned)
    test set                         (untouched until the final evaluation)
"""
import json
from collections import Counter
from pathlib import Path

from app.registry import PROJECT_ROOT
from eval.make_split import split_ids, test_hash
from eval.splits import SPLITS_DIR, assert_no_test_ids, load_split

OUTCOMES_PATH = PROJECT_ROOT / "data" / "outcomes.jsonl"
SEED = 42
VAL_FRACTION = 0.2
VAL_FILES = ("val_ids.txt", "val_info.json")


def labeled_train_records(outcomes_path: Path = OUTCOMES_PATH, splits_dir: Path = SPLITS_DIR) -> list:
    """[{id, source, task_type}] for the labeled train prompts (same filter as train_classifier.load_training_rows)."""
    train_ids, _ = load_split(splits_dir)
    records = []
    for line in Path(outcomes_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if o["label_status"] == "labeled" and o["id"] in train_ids:
            records.append({"id": o["id"], "source": o["source"], "task_type": o["task_type"]})
    assert_no_test_ids({r["id"] for r in records}, splits_dir)
    return sorted(records, key=lambda r: r["id"])


def val_info(records: list, fit: list, val: list, seed: int, fraction: float) -> dict:
    stratum = {r["id"]: f"{r['source']}/{r['task_type']}" for r in records}
    fit_counts, val_counts = Counter(stratum[i] for i in fit), Counter(stratum[i] for i in val)
    return {
        "seed": seed,
        "val_fraction": fraction,
        "stratified_by": ["source", "task_type"],
        "pool": "labeled train prompts (outcomes.jsonl label_status == 'labeled' and in train_ids.txt)",
        "pool_size": len(records),
        "fit": len(fit),
        "val": len(val),
        "counts": {key: {"fit": fit_counts[key], "val": val_counts[key]} for key in sorted(set(stratum.values()))},
        "val_ids_sha256": test_hash(val),       # same hash recipe as the test IDs: sha256 of the sorted IDs
    }


def write_validation_split(records: list, out_dir: Path = SPLITS_DIR, seed: int = SEED,
                           fraction: float = VAL_FRACTION) -> dict:
    """Write val_ids.txt and val_info.json. Raises FileExistsError if either already exists."""
    existing = [name for name in VAL_FILES if (out_dir / name).exists()]
    if existing:
        raise FileExistsError(f"validation split already exists ({', '.join(existing)} in {out_dir}); "
                              "it is locked and must not be redone")
    fit, val = split_ids(records, seed, fraction)
    info = val_info(records, fit, val, seed, fraction)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "val_ids.txt").write_text("\n".join(val) + "\n", encoding="utf-8")
    (out_dir / "val_info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


def main():
    records = labeled_train_records()
    try:
        info = write_validation_split(records)
    except FileExistsError as error:
        raise SystemExit(str(error))
    print(f"Split {info['pool_size']} labeled train prompts: {info['fit']} fit, {info['val']} val (seed {info['seed']})")
    print(f"val IDs SHA-256: {info['val_ids_sha256']}")
    print(f"\n{'source/task_type':<26}{'fit':>6}{'val':>6}")
    for key, c in info["counts"].items():
        print(f"{key:<26}{c['fit']:>6}{c['val']:>6}")


if __name__ == "__main__":
    main()
