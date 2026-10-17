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

MMLU_DATASET = "cais/mmlu"
MMLU_CONFIG = "all"
MMLU_SPLIT = "test"
MMLU_SUBJECT_COUNT = 57
MMLU_LETTERS = "ABCD"
# 200 questions over 57 subjects: 29 subjects get 4, the other 28 get 3
MMLU_LARGER_SUBJECTS = 29
MMLU_PER_LARGER_SUBJECT = 4
MMLU_PER_SMALLER_SUBJECT = 3

# Rows (indices into the cais/mmlu all/test split) that are broken and must never be used.
# If one is drawn, a replacement is drawn from the same subject.
MMLU_EXCLUDED_ROWS = {
    # high_school_macroeconomics: option D reads "both A and D are correct", which refers to
    # itself, yet D is the official answer (almost certainly a typo for "A and C").
    3895: "self-referential option D marked as the answer",
}

GSM8K_INSTRUCTION = (
    "Work through it step by step, then give the final answer on the last line in the form: Answer: <number>"
)

MMLU_INSTRUCTION = "End your reply with a final line in the form: Answer: <letter>"

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


def mmlu_prompt(question: str, choices: list) -> str:
    options = "\n".join(f"{letter}) {choice}" for letter, choice in zip(MMLU_LETTERS, choices))
    return f"{question}\n\n{options}\n\n{MMLU_INSTRUCTION}"


def mmlu_reference(row: dict, index: int) -> str:
    """The correct letter for a row. Raises if the row doesn't have exactly 4 choices and a 0-3 answer."""
    if len(row["choices"]) != len(MMLU_LETTERS):
        raise ValueError(f"MMLU row {index}: expected 4 choices, got {len(row['choices'])}")
    answer = row["answer"]
    if not isinstance(answer, int) or not 0 <= answer < len(MMLU_LETTERS):
        raise ValueError(f"MMLU row {index}: answer {answer!r} is not an index from 0 to 3")
    return MMLU_LETTERS[answer]


def build_mmlu_records(seed: int = SEED) -> list:
    rows = load_dataset(MMLU_DATASET, MMLU_CONFIG, split=MMLU_SPLIT)
    indices_by_subject = {}
    for index, subject in enumerate(rows["subject"]):
        indices_by_subject.setdefault(subject, []).append(index)
    subjects = sorted(indices_by_subject)  # sorted first, so the shuffle is reproducible
    if len(subjects) != MMLU_SUBJECT_COUNT:
        raise ValueError(f"expected {MMLU_SUBJECT_COUNT} MMLU subjects, found {len(subjects)}")
    random.Random(seed).shuffle(subjects)

    picks = []  # (subject, row index), in subject order, then draw order within each subject
    for position, subject in enumerate(subjects):
        count = MMLU_PER_LARGER_SUBJECT if position < MMLU_LARGER_SUBJECTS else MMLU_PER_SMALLER_SUBJECT
        # Each subject is sampled with a fresh generator on the same seed, without replacement.
        rng = random.Random(seed)
        pool = indices_by_subject[subject]
        drawn = rng.sample(pool, count)
        # Swap any excluded row for a fresh draw in the same position, so the other picks
        # (and every record ID) stay where they were.
        for slot, index in enumerate(drawn):
            if index in MMLU_EXCLUDED_ROWS:
                unused = [i for i in pool if i not in drawn and i not in MMLU_EXCLUDED_ROWS]
                drawn[slot] = rng.choice(unused)
        picks.extend((subject, index) for index in drawn)

    records = []
    for number, (subject, index) in enumerate(picks, start=1):
        row = rows[index]
        records.append({
            "id": f"mmlu_{number:04d}",
            "source": "mmlu",
            "task_type": "qa",
            "messages": [{"role": "user", "content": mmlu_prompt(row["question"], row["choices"])}],
            "reference": mmlu_reference(row, index),
            "scoring": "final_answer",
            "difficulty_guess": "unrated",
            "messy": False,
            "constraints": ["ends with 'Answer: <letter>'"],
            "notes": f"MMLU ({MMLU_DATASET}, {MMLU_CONFIG}/{MMLU_SPLIT}) row {index}: multiple choice, {subject}",
            "meta": {"dataset": MMLU_DATASET, "config": MMLU_CONFIG, "split": MMLU_SPLIT, "index": index,
                     "subject": subject},
        })
    return records


def write_jsonl(records: list, path: Path) -> None:
    """Overwrite path with one JSON record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    for name, label, build in [("gsm8k", "GSM8K", build_gsm8k_records), ("mmlu", "MMLU", build_mmlu_records)]:
        records = build()
        path = BENCHMARKS_DIR / f"{name}.jsonl"
        write_jsonl(records, path)
        print(f"Wrote {len(records)} {label} records to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
