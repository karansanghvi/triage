"""Turn public benchmark rows into prompt records (see data/schema.md).

Usage:
    python -m eval.benchmarks                    # GSM8K and MMLU
    python -m eval.benchmarks arena-candidates   # Arena review files (gitignored)
    python -m eval.benchmarks arena              # final Arena file, after review marks
    python -m eval.benchmarks merge              # combine every source into data/prompts.jsonl

Writes one file per source under data/benchmarks/, overwriting it each run. Rows are
picked with a fixed seed, so rerunning produces an identical file.

Downloads go to the Hugging Face cache (~/.cache/huggingface by default), outside the repo.
"""
import argparse
import json
import random
import re
from collections import Counter
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

ARENA_DATASET = "lmarena-ai/arena-human-preference-140k"
ARENA_CONFIG = "default"
ARENA_SPLIT = "train"
ARENA_CANDIDATES = 300
ARENA_FINAL_SIZE = 200
ARENA_MARKS_PATH = BENCHMARKS_DIR / "arena_marks.txt"

# The combined dataset, built from these files in this order
MERGED_PATH = PROJECT_ROOT / "data" / "prompts.jsonl"
MERGE_SOURCES = [
    PROJECT_ROOT / "data" / "own_prompts.jsonl",
    BENCHMARKS_DIR / "gsm8k.jsonl",
    BENCHMARKS_DIR / "mmlu.jsonl",
    BENCHMARKS_DIR / "arena.jsonl",
]
ARENA_MIN_WORDS = 5
ARENA_MAX_WORDS = 1200
ARENA_COLUMNS = ["conversation_a", "language", "evaluation_order", "is_code", "category_tag"]

# Keyword filters for Arena prompts. A prompt matching any of these is dropped, and every matching
# reason is recorded. They are deliberately broad: the surviving candidates still get a human review.
ARENA_EXCLUSIONS = {
    "jailbreak": re.compile(
        r"jailbreak|\bDAN\b|do anything now|god ?mode|developer mode|\[dev prompt\]|"
        r"\b(ignore|disregard|forget)\b.{0,20}\b(previous|prior|above|all|your)\b.{0,20}\b(instructions|rules|prompts?|guidelines)\b|"
        r"\b(reveal|show|print|repeat|leak)\b.{0,15}\b(system prompt|your instructions|hidden prompt)\b|"
        r"\b(without|no)\b.{0,15}\b(filters?|restrictions|censorship|ethical|moral)\b.{0,15}\b(limits?|guidelines|restrictions|filters?)?\b.{0,5}(mode|unrestricted)|"
        r"\bunfiltered\b|\buncensored\b|\bunrestricted (mode|ai)\b|you are no longer (an ai|bound)",
        re.IGNORECASE),
    "image_file_or_browsing": re.compile(
        r"\b(generate|create|draw|make|design|render|paint|produce|sketch)\b (me )?(an? |the |some )?"
        r"(image|picture|photo|illustration|drawing|logo|painting|sketch|portrait|icon|wallpaper|meme)s?\b|"
        r"\b(attached|uploaded|this|the following) (image|picture|photo|screenshot|pdf|file|document|spreadsheet)\b(?! (format|name|path|system|extension))|"
        r"\b(browse|search the (web|internet)|look (it )?up online|go to (this|the) (link|url|website|site))\b|"
        r"https?://\S+",
        re.IGNORECASE),
    "sexual": re.compile(
        r"\bsex(y|ual|ually)?\b|\bporn|\bnsfw\b|\bnude\b|\bnaked\b|\berotic|\bfetish|\borgasm|\bhorny\b|"
        r"\bboobs?\b|\bpenis\b|\bvagina|\bblowjob|\bhentai\b|\bstrippers?\b|\bstrip club|\bonlyfans\b|\bxxx\b|"
        r"\bincest\b|\brape\b|\bmilf\b",
        re.IGNORECASE),
    "hateful": re.compile(
        r"\bn[i1]gg(er|a)s?\b|\bf[a@]gg?ots?\b|\bretards?\b|\bk[i1]kes?\b|\bch[i1]nks?\b|\bsp[i1]cs?\b|"
        r"\btr[a@]nn(y|ies)\b|heil hitler|white power|\b(hate|kill) (all )?(jews|muslims|blacks|gays|women|immigrants)\b",
        re.IGNORECASE),
    "abuse": re.compile(
        r"\b(malware|ransomware|keylogger|trojan|botnet|spyware|rootkit|malicious|computer virus|ddos|zero-day|"
        r"phishing|scam(s|mer|mers|ming)?|fraud(ulent)?|carding|money laundering|launder(ing)? money|"
        r"fake (ids?|passports?|documents|reviews|invoices))\b|"
        r"\bhack(ing)?\b.{0,20}\b(into|someone|account|accounts|wifi|wi-fi|password|passwords|phone|email|instagram|"
        r"facebook|whatsapp|server)\b|"
        r"\b(bypass(ing)?|circumvent(ing)?|get(ting)? around|evad(e|ing)|disabl(e|ing))\b.{0,30}\b(security|authentication|"
        r"login|paywall|drm|captcha|antivirus|anti-cheat|anticheat|2fa|license check|activation|ban|bans|banned|block|"
        r"blocked|restrictions?)\b|"
        r"\bcrack(ed|ing)?\b.{0,15}\b(software|password|passwords|license|wifi|wi-fi|games?)\b|"
        r"\bsteal(ing)?\b.{0,20}\b(passwords?|credentials|data|accounts?|identity|credit cards?)\b",
        re.IGNORECASE),
    "depends_on_context": re.compile(
        r"\b(as (i|we) (said|mentioned|discussed)|(your|the) (previous|last|earlier) (answer|response|message|reply|output)|"
        r"continue (from )?where you left off|you (said|told me|wrote|mentioned) (earlier|before|above)|same as before|"
        r"try again|regenerate|(the|that) (story|code|text|answer|list) (you|we) (wrote|made|gave))\b|^\s*(continue|go on|more|again|why\??|ok|yes|no)\s*[.!?]*\s*$",
        re.IGNORECASE),
    "personal_info": re.compile(
        r"[\w.+-]+@[\w-]+\.[\w.-]+|"                                          # email address
        r"(?<!\d)(\+?\d{1,3}[\s.-])?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\d)|"       # phone number
        r"\b\d{1,5} (?:[A-Z][a-z]+ ){1,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b"),
}

# Task-type hints, checked in this order after the dataset's own tags. First match wins.
CODE_HINT = re.compile(
    r"```|\b(python|javascript|typescript|java|c\+\+|c#|golang|rust|kotlin|swift|php|ruby|sql|html|css|react|vue|"
    r"angular|node\.?js|django|flask|bash|shell script|powershell|regex|api|json|yaml|docker|kubernetes|git|"
    r"function|class|method|variable|compile|compiler|debug|bug|stack ?trace|exception|syntax|script|code|"
    r"algorithm|library|framework|repo|database|query|frontend|backend|linux|terminal|excel formula|vba)\b",
    re.IGNORECASE)
SUMMARIZE_HINT = re.compile(r"\b(summari[sz]e|summary|tl;?dr|key points|condense)\b", re.IGNORECASE)
EXTRACT_HINT = re.compile(
    r"\b(extract|pull out|parse)\b|\bconvert\b.{0,40}\b(json|csv|table)\b|\blist (all|every) (the )?(names|dates|numbers|emails)\b",
    re.IGNORECASE)
CLASSIFY_HINT = re.compile(
    r"\b(classify|categori[sz]e|which category|sentiment|is (this|it) (positive|negative|spam))\b", re.IGNORECASE)
CREATIVE_HINT = re.compile(
    r"\b(write|compose|create|draft|come up with|generate)\b.{0,40}\b(story|poem|lyrics|song|haiku|limerick|joke|"
    r"slogan|tagline|caption|speech|toast|script|dialogue|essay|letter|email|tweet|post|bio|description|names?|"
    r"title|headline|prompt|rap)\b|\b(rewrite|rephrase|paraphrase|reword|proofread|polish|make (it|this) sound|"
    r"translate|roleplay|role-play|pretend you|brainstorm)\b",
    re.IGNORECASE)
MATH_HINT = re.compile(
    r"\b(calculate|compute|solve|equation|integral|derivative|probability|prove|theorem)\b|\d+\s*[-+*/^×]\s*\d+|\d+(?:\.\d+)?\s*% of\b",
    re.IGNORECASE)
# Weaker math words only count when the prompt also has at least two numbers
WORD_PROBLEM_HINT = re.compile(
    r"\b(how (many|much|far|long|fast|old)|percent(age)?|average|ratio|sum of|total|convert)\b", re.IGNORECASE)
NUMBER_TOKEN = re.compile(r"\d+(?:\.\d+)?")
ADVICE_HINT = re.compile(
    r"\b(should i|should we|best way|best (time|place|option|tool|choice)|recommend|suggest|tips?|advice|"
    r"opinion|what do you think|thoughts on|how (can|do|should) (i|we)|help me (decide|choose|plan)|worth it|"
    r"pros and cons|plan|itinerary|ideas?|vs\.?|versus|better|would you|do you (like|think|believe))\b",
    re.IGNORECASE)
QUESTION_HINT = re.compile(
    r"^\s*(what|who|whom|when|where|which|why|how|is|are|was|were|does|do|did|can|could|define|explain|"
    r"tell me (about|what|who|why|how))\b|\?\s*$",
    re.IGNORECASE)

# Phrases that state an explicit, checkable rule; copied into constraints verbatim.
CONSTRAINT_PATTERNS = [
    re.compile(r"\b(?:in|under|within|less than|fewer than|no more than|at most|maximum of|max|exactly|at least)\s+"
               r"\d+\s+(?:words?|sentences?|characters?|paragraphs?|lines?|bullet points?|bullets?|points?|items?)\b",
               re.IGNORECASE),
    re.compile(r"\b\d+\s+(?:words?|sentences?|paragraphs?|bullet points?|bullets?)\s+(?:or less|or fewer|max(?:imum)?|long)\b",
               re.IGNORECASE),
    re.compile(r"\b(?:in|as)\s+(?:a\s+)?(?:json|table|bullet(?:ed)? (?:points|list)|numbered list|markdown|csv|yaml)\b",
               re.IGNORECASE),
    re.compile(r"\b(?:do not|don't|without)\s+(?:use|using|mention|mentioning|include|including)\b[^.\n]{0,40}",
               re.IGNORECASE),
]

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


def arena_first_prompt(conversation: list):
    """Text of the first user message, or None if it has non-text parts (images) or no text."""
    first = next((m for m in conversation if m["role"] == "user"), None)
    if first is None:
        return None
    parts = first["content"] or []
    if any(part.get("type") != "text" for part in parts):
        return None
    text = "\n\n".join(part.get("text") or "" for part in parts).strip()
    return text or None


def arena_exclusion_reasons(text: str) -> list:
    return [reason for reason, pattern in ARENA_EXCLUSIONS.items() if pattern.search(text)]


def tag_flag(category_tag, group: str, flag: str) -> bool:
    return bool(((category_tag or {}).get(group) or {}).get(flag))


def suggest_task_type(text: str, is_code: bool, category_tag) -> tuple:
    """(task_type, reason) from the dataset's tags and simple keyword rules. Only a suggestion for review."""
    code_hint = CODE_HINT.search(text)
    if is_code and code_hint:
        return "coding", f"is_code=true + keyword '{code_hint.group()}'"
    if code_hint and code_hint.group().lower() in {"```", "python", "javascript", "sql", "function", "code", "regex",
                                                 "bug", "debug", "compile", "stack trace", "script"}:
        return "coding", f"keyword '{code_hint.group()}'"
    for task_type, pattern in [("summarization", SUMMARIZE_HINT), ("extraction", EXTRACT_HINT),
                               ("classification", CLASSIFY_HINT)]:
        match = pattern.search(text)
        if match:
            return task_type, f"keyword '{match.group()}'"
    if tag_flag(category_tag, "creative_writing_v0.1", "creative_writing"):
        return "creative", "creative_writing tag"
    match = CREATIVE_HINT.search(text)
    if match:
        return "creative", f"keyword '{match.group()[:40]}'"
    if tag_flag(category_tag, "math_v0.1", "math"):
        return "reasoning", "math tag"
    match = MATH_HINT.search(text)
    if match:
        return "reasoning", f"keyword '{match.group()}'"
    match = WORD_PROBLEM_HINT.search(text)
    if match and len(NUMBER_TOKEN.findall(text)) >= 2:
        return "reasoning", f"keyword '{match.group()}' + numbers"
    if is_code:
        return "coding", "is_code=true, no code keyword (check)"
    match = ADVICE_HINT.search(text)
    if match:
        return "open_ended", f"advice/opinion keyword '{match.group()}'"
    if QUESTION_HINT.search(text):
        return "qa", "question form"
    return "open_ended", "no rule matched (default)"


def arena_constraints(text: str) -> list:
    found = []
    for pattern in CONSTRAINT_PATTERNS:
        for match in pattern.finditer(text):
            phrase = match.group().strip()
            if phrase not in found:
                found.append(phrase)
    return found


def normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def build_arena_candidates(count: int = ARENA_CANDIDATES, seed: int = SEED) -> tuple:
    """Return (candidate records, review rows, excluded rows, stage counts)."""
    rows = load_dataset(ARENA_DATASET, ARENA_CONFIG, split=ARENA_SPLIT, columns=ARENA_COLUMNS)
    stages = Counter(total=len(rows))
    passing = []  # (index, text) that survive every filter
    excluded = []
    seen = set()
    for index, row in enumerate(rows):
        if row["evaluation_order"] != 1:
            stages["dropped: not first turn"] += 1
            continue
        if row["language"] != "en":
            stages["dropped: not English"] += 1
            continue
        text = arena_first_prompt(row["conversation_a"])
        if text is None:
            stages["dropped: not text-only"] += 1
            continue
        words = len(text.split())
        if not ARENA_MIN_WORDS <= words <= ARENA_MAX_WORDS:
            stages["dropped: under 5 words" if words < ARENA_MIN_WORDS else "dropped: over 1,200 words"] += 1
            continue
        key = normalize_for_dedupe(text)
        if key in seen:
            stages["dropped: duplicate"] += 1
            continue
        seen.add(key)
        reasons = arena_exclusion_reasons(text)
        if reasons:
            stages["dropped: keyword filter"] += 1
            for reason in reasons:
                stages[f"  keyword: {reason}"] += 1
            excluded.append({"index": index, "reasons": reasons, "text": text[:300]})
            continue
        passing.append(index)
    stages["passed all filters"] = len(passing)

    records, review = [], []
    for number, index in enumerate(random.Random(seed).sample(passing, count), start=1):
        row = rows[index]
        text = arena_first_prompt(row["conversation_a"])
        task_type, reason = suggest_task_type(text, row["is_code"], row["category_tag"])
        records.append({
            "id": f"arena_{number:04d}",
            "source": "arena",
            "task_type": task_type,
            "messages": [{"role": "user", "content": text}],
            "reference": None,
            "scoring": "judge",
            "difficulty_guess": "unrated",
            "constraints": arena_constraints(text),
            "notes": f"Arena ({ARENA_DATASET}, {ARENA_CONFIG}/{ARENA_SPLIT}) row {index}: real first-turn user prompt",
            "meta": {"dataset": ARENA_DATASET, "config": ARENA_CONFIG, "split": ARENA_SPLIT, "index": index,
                     "is_code": bool(row["is_code"]), "category_tag": row["category_tag"]},
        })
        review.append({"number": number, "task_type": task_type, "reason": reason, "text": text})
    return records, review, excluded, stages


def read_arena_marks(path: Path, candidate_count: int) -> dict:
    """Parse review marks ("131 drop", "261 open_ended") into {candidate number: action}. Raises on bad lines."""
    from scripts.validate_prompts import TASK_TYPES
    marks = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2 or not parts[0].isdigit():
            raise ValueError(f"{path.name}:{line_number}: expected '<number> drop' or '<number> <task_type>', got {raw!r}")
        number, action = int(parts[0]), parts[1]
        if not 1 <= number <= candidate_count:
            raise ValueError(f"{path.name}:{line_number}: candidate {number} is outside 1-{candidate_count}")
        if action != "drop" and action not in TASK_TYPES:
            raise ValueError(f"{path.name}:{line_number}: unknown action {action!r}")
        if number in marks:
            raise ValueError(f"{path.name}:{line_number}: candidate {number} is marked twice")
        marks[number] = action
    return marks


def build_arena_records(size: int = ARENA_FINAL_SIZE, seed: int = SEED) -> tuple:
    """Apply the review marks to the candidates and draw the final records. Returns (records, summary)."""
    candidates, _, _, _ = build_arena_candidates()
    marks = read_arena_marks(ARENA_MARKS_PATH, len(candidates))
    kept = []
    summary = Counter()
    for number, record in enumerate(candidates, start=1):
        action = marks.get(number)
        if action == "drop":
            summary["dropped"] += 1
            continue
        if action is not None:
            summary["relabeled" if action != record["task_type"] else "relabel matched suggestion"] += 1
            record = dict(record, task_type=action)
        kept.append(record)
    if len(kept) < size:
        raise ValueError(f"only {len(kept)} candidates remain after marks; need {size}")
    summary["remaining"] = len(kept)
    records = []
    for number, record in enumerate(random.Random(seed).sample(kept, size), start=1):
        records.append(dict(record, id=f"arena_{number:04d}"))
    return records, summary


def write_arena_review(review: list, path: Path) -> None:
    lines = [f"Arena candidates for review ({len(review)}). Mark drops and task_type changes by candidate number.\n"]
    for row in review:
        lines.append("=" * 100)
        lines.append(f"#{row['number']:03d}  suggested: {row['task_type']}  ({row['reason']})")
        lines.append("-" * 100)
        lines.append(row["text"])
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_jsonl(records: list, path: Path) -> None:
    """Overwrite path with one JSON record per line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_benchmark(name: str, label: str, build) -> None:
    records = build()
    path = BENCHMARKS_DIR / f"{name}.jsonl"
    write_jsonl(records, path)
    print(f"Wrote {len(records)} {label} records to {path.relative_to(PROJECT_ROOT)}")


def write_arena_candidates() -> None:
    records, review, excluded, stages = build_arena_candidates()
    paths = {
        "candidates": BENCHMARKS_DIR / "arena_candidates.jsonl",
        "review": BENCHMARKS_DIR / "arena_review.txt",
        "excluded": BENCHMARKS_DIR / "arena_excluded.jsonl",
    }
    write_jsonl(records, paths["candidates"])
    write_arena_review(review, paths["review"])
    write_jsonl(excluded, paths["excluded"])
    order = ["total", "dropped: not first turn", "dropped: not English", "dropped: not text-only",
             "dropped: under 5 words", "dropped: over 1,200 words", "dropped: duplicate", "dropped: keyword filter"]
    order += [f"  keyword: {reason}" for reason in ARENA_EXCLUSIONS] + ["passed all filters"]
    for stage in order:
        print(f"{stage:<36}{stages[stage]:>8}")
    print("suggested task types:", dict(Counter(r["task_type"] for r in records).most_common()))
    for path in paths.values():
        print(f"Wrote {path.relative_to(PROJECT_ROOT)}")


def write_arena() -> None:
    records, summary = build_arena_records()
    path = BENCHMARKS_DIR / "arena.jsonl"
    write_jsonl(records, path)
    print("marks:", dict(summary))
    print("final task types:", dict(Counter(r["task_type"] for r in records).most_common()))
    print(f"Wrote {len(records)} Arena records to {path.relative_to(PROJECT_ROOT)}")


def merge_prompts() -> None:
    """Concatenate the source files into MERGED_PATH, copying each record's line unchanged."""
    lines, ids = [], set()
    for path in MERGE_SOURCES:
        count = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record_id = json.loads(line)["id"]  # parse only to check; the original line is what's written
            if record_id in ids:
                raise ValueError(f"duplicate id {record_id!r} in {path.name}")
            ids.add(record_id)
            lines.append(line)
            count += 1
        print(f"{path.relative_to(PROJECT_ROOT)}: {count} records")
    MERGED_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} records to {MERGED_PATH.relative_to(PROJECT_ROOT)}")


def main():
    parser = argparse.ArgumentParser(description="Build benchmark prompt files.")
    parser.add_argument("targets", nargs="*", default=["gsm8k", "mmlu"],
                        choices=["gsm8k", "mmlu", "arena-candidates", "arena", "merge"])
    args = parser.parse_args()
    if "gsm8k" in args.targets:
        write_benchmark("gsm8k", "GSM8K", build_gsm8k_records)
    if "mmlu" in args.targets:
        write_benchmark("mmlu", "MMLU", build_mmlu_records)
    if "arena-candidates" in args.targets:
        write_arena_candidates()
    if "arena" in args.targets:
        write_arena()
    if "merge" in args.targets:  # last, so it picks up anything rebuilt above
        merge_prompts()


if __name__ == "__main__":
    main()
