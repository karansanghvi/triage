"""Turn public benchmark rows into prompt records (see data/schema.md).

Usage:
    python -m eval.benchmarks

Writes one file per source under data/benchmarks/, overwriting it each run. Rows are
picked with a fixed seed, so rerunning produces an identical file.

Downloads go to the Hugging Face cache (~/.cache/huggingface by default), outside the repo.
"""
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "test"
GSM8K_SAMPLE_SIZE = 200
SEED = 42

GSM8K_INSTRUCTION = (
    "Work through it step by step, then give the final answer on the last line in the form: Answer: <number>"
)

# A plain number after commas are removed: optional minus, digits, optional decimals
PLAIN_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def gsm8k_reference(answer: str, index: int) -> str:
    """The number after "####" in a GSM8K answer, commas removed. Raises if it isn't a number."""
    if "####" not in answer:
        raise ValueError(f"GSM8K row {index}: answer has no '####' line")
    value = answer.rsplit("####", 1)[1].strip().replace(",", "")
    if not PLAIN_NUMBER.fullmatch(value):
        raise ValueError(f"GSM8K row {index}: reference {value!r} is not a valid number")
    return value


def build_gsm8k_records(sample_size: int = GSM8K_SAMPLE_SIZE, seed: int = SEED) -> list:
    rows = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split=GSM8K_SPLIT)
    # Without replacement; the records keep the order the rows were drawn in.
    indices = random.Random(seed).sample(range(len(rows)), sample_size)
    records = []
    for number, index in enumerate(indices, start=1):
        row = rows[index]
        records.append({
            "id": f"gsm8k_{number:04d}",
            "source": "gsm8k",
            "task_type": "reasoning",
            "messages": [{"role": "user", "content": f"{row['question']}\n\n{GSM8K_INSTRUCTION}"}],
            "reference": gsm8k_reference(row["answer"], index),
            "scoring": "final_answer",
            "difficulty_guess": "unrated",
            "messy": False,
            "constraints": ["ends with 'Answer: <number>'"],
            "notes": f"GSM8K ({GSM8K_DATASET}, {GSM8K_CONFIG}/{GSM8K_SPLIT}) row {index}: grade-school math word problem",
            "meta": {"dataset": GSM8K_DATASET, "config": GSM8K_CONFIG, "split": GSM8K_SPLIT, "index": index},
        })
    return records


def write_jsonl(records: list, path: Path) -> None:
    """Overwrite path with one JSON record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    records = build_gsm8k_records()
    path = BENCHMARKS_DIR / "gsm8k.jsonl"
    write_jsonl(records, path)
    print(f"Wrote {len(records)} GSM8K records to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
