"""Check prompt dataset files against data/schema.md.

Usage:
    python scripts/validate_prompts.py [path ...]

With no paths, checks data/own_prompts.jsonl. With several paths, IDs must be
unique across all of them. Exits 1 if any errors are found; near-duplicates and
long prompts are reported as warnings only.
"""
import argparse
import difflib
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "own_prompts.jsonl"

REQUIRED_FIELDS = [
    "id", "source", "task_type", "messages", "reference",
    "scoring", "difficulty_guess", "constraints", "notes",
]
OPTIONAL_FIELDS = ["fuzzy_fields", "messy", "meta"]
SOURCES = ["own", "gsm8k", "mmlu", "arena"]
BENCHMARK_SOURCES = [s for s in SOURCES if s != "own"]
META_STRING_FIELDS = ["dataset", "config", "split"]
TASK_TYPES = ["extraction", "classification", "summarization", "qa", "reasoning", "creative"]
SCORINGS = ["field_check", "label_match", "exact_match", "final_answer", "judge"]
DIFFICULTIES = ["easy", "medium", "hard"]
# Benchmark records may also be "unrated": their difficulty is measured later, not guessed.
BENCHMARK_DIFFICULTIES = DIFFICULTIES + ["unrated"]
ROLES = {"system", "user", "assistant"}

# Which scoring methods each task type may use.
ALLOWED_SCORING = {
    "extraction": {"field_check"},
    "classification": {"label_match"},
    "summarization": {"judge"},
    "qa": {"exact_match", "final_answer", "judge"},
    "reasoning": {"exact_match", "final_answer", "judge"},
    "creative": {"judge"},
}

ID_PATTERN = re.compile(r"^own_(" + "|".join(TASK_TYPES) + r")_(\d{3})$")
BENCHMARK_ID_PATTERN = re.compile(r"^(" + "|".join(BENCHMARK_SOURCES) + r")_(\d{4})$")
DUPLICATE_THRESHOLD = 0.85
MAX_WORDS = 1200


def load_records(path, errors):
    """Return (location, record) pairs for every line that parses as a JSON object."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            location = f"{path.name}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"{location}: invalid JSON ({error})")
                continue
            if not isinstance(record, dict):
                errors.append(f"{location}: line is not a JSON object")
                continue
            records.append((location, record))
    return records


def check_messages(messages, where, errors):
    if not isinstance(messages, list) or not messages:
        errors.append(f"{where}: messages must be a non-empty list")
        return
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            errors.append(f"{where}: messages[{index}] must have exactly 'role' and 'content'")
            continue
        if message["role"] not in ROLES:
            errors.append(f"{where}: messages[{index}] has unknown role {message['role']!r}")
        if not isinstance(message["content"], str) or not message["content"].strip():
            errors.append(f"{where}: messages[{index}] has empty content")
        if message["role"] == "system" and index != 0:
            errors.append(f"{where}: system message must come first")
    if isinstance(messages[-1], dict) and messages[-1].get("role") != "user":
        errors.append(f"{where}: last message must be from the user")


def check_reference(record, where, errors):
    task_type = record["task_type"]
    scoring = record["scoring"]
    reference = record["reference"]

    if scoring not in ALLOWED_SCORING.get(task_type, set()):
        errors.append(f"{where}: {task_type} records cannot use scoring {scoring!r}")

    if scoring == "judge" and reference is not None:
        errors.append(f"{where}: judge records must have reference null")
    if scoring in ("exact_match", "final_answer") and not (isinstance(reference, str) and reference.strip()):
        errors.append(f"{where}: {scoring} records need a non-empty string reference")
    if scoring == "final_answer" and not any(
        isinstance(m, dict) and m.get("role") == "user" and "Answer:" in str(m.get("content"))
        for m in record["messages"]
    ):
        errors.append(f"{where}: final_answer records need a user message telling the model to end with 'Answer:'")
    if task_type == "extraction" and not isinstance(reference, dict):
        errors.append(f"{where}: extraction reference must be a JSON object")
    if task_type == "classification":
        if not (isinstance(reference, str) and reference.strip()):
            errors.append(f"{where}: classification reference must be a non-empty string")
        else:
            prompt_text = " ".join(
                m["content"] for m in record["messages"]
                if isinstance(m, dict) and isinstance(m.get("content"), str)
            ).lower()
            if reference.lower() not in prompt_text:
                errors.append(f"{where}: label {reference!r} does not appear in the prompt")


def resolve_path(value, keys):
    """Return the values a dotted path reaches, fanning out over lists. None if the path is missing."""
    if not keys:
        return [value]
    if isinstance(value, list):
        found = []
        for item in value:
            item_values = resolve_path(item, keys)
            if item_values is None:
                return None
            found.extend(item_values)
        return found
    if isinstance(value, dict) and keys[0] in value:
        return resolve_path(value[keys[0]], keys[1:])
    return None


def is_text(value):
    return value is None or isinstance(value, str) or (
        isinstance(value, list) and all(isinstance(v, str) for v in value)
    )


def check_fuzzy_fields(record, where, errors):
    if "fuzzy_fields" not in record:
        return
    fuzzy_fields = record["fuzzy_fields"]
    if record["scoring"] != "field_check":
        errors.append(f"{where}: fuzzy_fields is only allowed with field_check scoring")
        return
    if not isinstance(fuzzy_fields, list) or not all(isinstance(f, str) and f for f in fuzzy_fields):
        errors.append(f"{where}: fuzzy_fields must be a list of non-empty strings")
        return
    if len(set(fuzzy_fields)) != len(fuzzy_fields):
        errors.append(f"{where}: fuzzy_fields has duplicates")
    if not isinstance(record["reference"], dict):
        return
    for path in fuzzy_fields:
        values = resolve_path(record["reference"], path.split("."))
        if values is None:
            errors.append(f"{where}: fuzzy field {path!r} not found in reference")
        elif not all(is_text(v) for v in values):
            errors.append(f"{where}: fuzzy field {path!r} must point to text, not numbers/booleans/objects")


def check_id(record, where, errors):
    source = record["source"]
    if source not in SOURCES:
        errors.append(f"{where}: source {source!r} not in {SOURCES}")
        return
    if source == "own":
        match = ID_PATTERN.match(str(record["id"]))
        if not match:
            errors.append(f"{where}: id does not match own_<task_type>_<3 digits>")
        elif match.group(1) != record["task_type"]:
            errors.append(f"{where}: id says {match.group(1)!r} but task_type is {record['task_type']!r}")
        return
    match = BENCHMARK_ID_PATTERN.match(str(record["id"]))
    if not match:
        errors.append(f"{where}: id does not match {source}_<4 digits>")
    elif match.group(1) != source:
        errors.append(f"{where}: id prefix {match.group(1)!r} does not match source {source!r}")


def check_meta(record, where, errors, warnings):
    """meta is required for benchmark sources. Own records may have one, with no required fields."""
    source = record["source"]
    if "meta" not in record:
        if source in BENCHMARK_SOURCES:
            errors.append(f"{where}: {source} records need a meta object")
        return
    meta = record["meta"]
    if not isinstance(meta, dict):
        errors.append(f"{where}: meta must be an object")
        return
    if source == "own":
        return
    for field in META_STRING_FIELDS:
        if field not in meta:
            errors.append(f"{where}: meta.{field} is missing")
        elif not (isinstance(meta[field], str) and meta[field].strip()):
            errors.append(f"{where}: meta.{field} must be a non-empty string")
    if "index" not in meta:
        errors.append(f"{where}: meta.index is missing")
    # bool is a subclass of int, so rule it out explicitly.
    elif not isinstance(meta["index"], int) or isinstance(meta["index"], bool) or meta["index"] < 0:
        errors.append(f"{where}: meta.index must be a non-negative integer")
    if source == "mmlu":
        if not (isinstance(meta.get("subject"), str) and meta["subject"].strip()):
            errors.append(f"{where}: mmlu records need meta.subject as a non-empty string")
    elif "subject" in meta:
        errors.append(f"{where}: meta.subject is only allowed on mmlu records")
    extra = sorted(set(meta) - set(META_STRING_FIELDS) - {"index", "subject"})
    if extra:
        warnings.append(f"{where}: unexpected meta fields {extra}")


def check_record(location, record, errors, warnings):
    missing = [field for field in REQUIRED_FIELDS if field not in record]
    if missing:
        errors.append(f"{location}: missing fields {missing}")
        return False
    extra = sorted(set(record) - set(REQUIRED_FIELDS) - set(OPTIONAL_FIELDS))
    if extra:
        warnings.append(f"{location}: unexpected fields {extra}")

    where = f"{location} ({record['id']})"
    ok = True
    difficulties = BENCHMARK_DIFFICULTIES if record["source"] in BENCHMARK_SOURCES else DIFFICULTIES
    for field, allowed in [
        ("task_type", TASK_TYPES), ("scoring", SCORINGS), ("difficulty_guess", difficulties),
    ]:
        if record[field] not in allowed:
            errors.append(f"{where}: {field} {record[field]!r} not in {allowed}")
            ok = False
    if not ok:
        return False

    check_id(record, where, errors)
    check_meta(record, where, errors, warnings)
    if not isinstance(record["constraints"], list) or not all(
        isinstance(c, str) and c.strip() for c in record["constraints"]
    ):
        errors.append(f"{where}: constraints must be a list of non-empty strings")
    if not isinstance(record["notes"], str) or not record["notes"].strip():
        errors.append(f"{where}: notes must be a non-empty string")
    if "messy" in record and not isinstance(record["messy"], bool):
        errors.append(f"{where}: messy must be true or false")

    check_messages(record["messages"], where, errors)
    if isinstance(record["messages"], list) and record["messages"]:
        check_reference(record, where, errors)
        check_fuzzy_fields(record, where, errors)
        words = sum(
            len(m["content"].split()) for m in record["messages"]
            if isinstance(m, dict) and isinstance(m.get("content"), str)
        )
        if words > MAX_WORDS:
            warnings.append(f"{where}: {words} words, over the ~{MAX_WORDS} word limit")
    return True


def user_text(record):
    return "\n".join(
        m["content"] for m in record["messages"]
        if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)
    )


def find_near_duplicates(records):
    """Return (id_a, id_b, ratio) for pairs whose user content is too similar.

    Only records from the same source are compared: benchmark prompts share boilerplate
    instructions with each other, and checking across sources would only slow things down.
    """
    by_source = {}
    for record in records:
        by_source.setdefault(record["source"], []).append((record["id"], user_text(record)))
    pairs = []
    for texts in by_source.values():
        pairs.extend(similar_pairs(texts))
    return pairs


def similar_pairs(texts):
    pairs = []
    for (id_a, text_a), (id_b, text_b) in combinations(texts, 2):
        matcher = difflib.SequenceMatcher(None, text_a, text_b)
        # Cheap upper bounds first; ratio() is slow on long texts.
        if matcher.real_quick_ratio() <= DUPLICATE_THRESHOLD:
            continue
        if matcher.quick_ratio() <= DUPLICATE_THRESHOLD:
            continue
        ratio = matcher.ratio()
        if ratio > DUPLICATE_THRESHOLD:
            pairs.append((id_a, id_b, ratio))
    return pairs


def print_counts(records):
    counts = Counter((r["task_type"], r["difficulty_guess"]) for r in records)
    task_types = [t for t in TASK_TYPES if any(key[0] == t for key in counts)]
    # Only show the "unrated" column when some record uses it
    columns = [d for d in BENCHMARK_DIFFICULTIES if d in DIFFICULTIES or any(key[1] == d for key in counts)]
    header = f"{'task_type':<16}" + "".join(f"{d:>8}" for d in columns) + f"{'total':>8}"
    print(header)
    print("-" * len(header))
    for task_type in task_types:
        row = [counts[(task_type, d)] for d in columns]
        print(f"{task_type:<16}" + "".join(f"{n:>8}" for n in row) + f"{sum(row):>8}")
    totals = [sum(counts[(t, d)] for t in task_types) for d in columns]
    print("-" * len(header))
    print(f"{'total':<16}" + "".join(f"{n:>8}" for n in totals) + f"{sum(totals):>8}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path, default=[DEFAULT_PATH])
    args = parser.parse_args()

    errors, warnings, valid = [], [], []
    for path in args.paths:
        if not path.exists():
            errors.append(f"{path}: file not found")
            continue
        for location, record in load_records(path, errors):
            if check_record(location, record, errors, warnings):
                valid.append((location, record))

    id_locations = {}
    for location, record in valid:
        id_locations.setdefault(record["id"], []).append(location)
    for record_id, locations in id_locations.items():
        if len(locations) > 1:
            errors.append(f"duplicate id {record_id!r} at {', '.join(locations)}")

    records = [record for _, record in valid]
    for id_a, id_b, ratio in find_near_duplicates(records):
        warnings.append(f"near-duplicate user content: {id_a} vs {id_b} ({ratio:.2f})")

    print(f"Checked {len(records)} records from {', '.join(str(p) for p in args.paths)}\n")
    if records:
        print_counts(records)
        tagged = [r for r in records if "messy" in r]
        if tagged:
            messy = sum(r["messy"] is True for r in tagged)
            print(f"\nmessy: {messy} of {len(tagged)} tagged records ({len(records) - len(tagged)} untagged)")
        print()
    for warning in warnings:
        print(f"WARNING: {warning}")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"\n{len(errors)} error(s), {len(warnings)} warning(s)")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
