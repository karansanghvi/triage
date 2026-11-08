"""
features.py: handcrafted prompt features for Triage's router.
 
Turns one prompt into a short row of numbers the classifier can learn from.
Everything here uses ONLY the prompt text (plus an optional task type),
so it can run at request time, before any model is called.
 
Usage (batch, for training):
    python features.py --input data/prompts.jsonl --output data/features.npz
 
Usage (in code, at request time):
    from features import extract_features, to_vector
    vec = to_vector(extract_features("Summarize this email: ..."))
"""

from __future__ import annotations

import argparse
import json
import math
import re 
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Fixed vocabularies. Changing any of these changes the model's inputs,
#    so bump your feature version if you edit them after training.
# ---------------------------------------------------------------------------
 
FEATURE_VERSION = "3"   # 3: Phase 4 Step 0 retrain on the fit set (features and guesses same as v2)
 
TASK_TYPES = ["extraction", "summarization", "classification", "reasoning", 
              "creative", "qa", "coding", "open_ended"]
 
# Different spellings people (or your dataset) might use for the same task.
TASK_ALIASES = {
    "q&a": "qa",
    "q_and_a": "qa",
    "question_answering": "qa",
    "math": "reasoning",
    "summarisation": "summarization",
}
 
# Words that suggest the model has to think, not just look something up.
REASONING_RE = re.compile(
    r"\b(?:analy[sz]e|analysis|compare|contrast|prove|proof|justify|evaluate|why|"
    r"derive|step[- ]by[- ]step|trade-?offs?|pros and cons|critique|infer|"
    r"calculate|solve|reason(?:ing)?)\b",
    re.IGNORECASE,
)
 
# Phrases that add a rule the answer has to obey.
CONSTRAINT_RE = re.compile(
    r"\b(?:must|should|do not|don't|never|always|exactly|only|at most|at least|"
    r"no more than|without|ensure|make sure|"
    r"(?:under|within|fewer than|less than|no longer than) \d+)\b",
    re.IGNORECASE,
)
 
# Lines that start like "- item", "* item", "1. item", "2) item".
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", re.MULTILINE)
 
# Words that point at material included in the prompt.
CONTEXT_MARKER_RE = re.compile(
    r"\b(?:below|following|above|given text|"
    r"this (?:text|article|passage|email|document|review|paragraph|code|transcript))\b",
    re.IGNORECASE,
)
 
# Output format levels, checked from strictest to loosest; first match wins.
FORMAT_LEVELS = [
    (3, re.compile(r"\b(?:json|yaml|xml|schema|csv|sql|regex|code|function)\b", re.IGNORECASE)),
    (2, re.compile(r"\b(?:table|columns?|sections?|headings?|template|format(?:ted)? as)\b", re.IGNORECASE)),
    (1, re.compile(r"\b(?:list|bullets?|bullet points|numbered|steps|one (?:word|sentence|line)|key points)\b", re.IGNORECASE)),
]
 
# Rough tokenizer: words and individual punctuation marks.
# Not identical to any model's tokenizer, but consistent, fast and offline.
TOKEN_RE = re.compile(r"\w+|[^\w\s]")
 
 
# ---------------------------------------------------------------------------
# 2. One small function per feature. Each is easy to test on its own.
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    return len(TOKEN_RE.findall(text))

def count_reasoning_words(text: str) -> int:
    return len(REASONING_RE.findall(text))

def count_constraints(text: str) -> int:
    """Rule phrases + list-style instruction lines."""
    return len(CONSTRAINT_RE.findall(text)) + len(LIST_ITEM_RE.findall(text))

def has_context(text: str) -> int:
    """1 if the prompt includes material to work on (an article, code, data), else 0."""
    if "```" in text or '"""' in text:
        return 1
    if re.search(r'"[^"]{150,}"', text):  # a long quoted passage
        return 1
    # A long chunk after "...:" + newline, or after a blank line.
    parts = re.split(r":\s*\n|\n\s*\n", text, maxsplit=1)
    if len(parts) == 2 and len(parts[1].strip()) >= 200:
        return 1
    # "Summarize the following ..." in a prompt that is long enough to contain it.
    if CONTEXT_MARKER_RE.search(text) and len(text) >= 300:
        return 1
    return 0

def format_complexity(text: str) -> int:
    """0 = plain, 1 = list, 2 = table/structure, 3 = JSON/code/strict schema"""
    for level, pattern in FORMAT_LEVELS:
        if pattern.search(text):
            return level
    return 0 

def normalize_task_type(task_type: str | None) -> str | None:
    if not task_type:
        return None
    t = task_type.strip().lower()
    t = TASK_ALIASES.get(t, t)
    return t if t in TASK_TYPES else None

# Task-type guess rules, checked in order on the lowercased prompt; the first match wins, "qa" if none.
# Same guesses as the original if-chain (commit c61a92a). Extra coding keywords were tried in Phase 4
# Step 0 and reverted: they gained on train - val but lost on validation (reports/guesser_agreement.json).
GUESS_RULES = [
    ("coding", re.compile(r"\b(?:python|javascript|typescript|java|sql|regex|function|refactor|debug|bug|"
                          r"code|script|compile|stack trace)\b|```")),
    ("extraction", re.compile(r"\b(?:extract|pull out|find all|fields?)\b")),
    ("summarization", re.compile(r"\bsummar|\btl;?dr\b|\bkey points\b")),
    ("classification", re.compile(r"\b(?:classify|categori[sz]e|label|sentiment|which category)\b")),
    ("creative", re.compile(r"\bwrite (?:a|an|me a) (?:poem|story|song|haiku|limerick)|\b(?:slogan|tagline|imagine)\b")),
    ("reasoning", re.compile(r"\b(?:calculate|solve|prove|how many|how much|step[- ]by[- ]step)\b|\d+\s*[-+*/x]\s*\d+")),
]


def guess_task_type(text: str) -> str:
    """Keyword guess for production, where nobody hand-tags the request."""
    t = text.lower()
    for task, pattern in GUESS_RULES:
        if pattern.search(t):
            return task
    return "qa"

# ---------------------------------------------------------------------------
# 3. Put it all together.
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "token_count",
    "log_token_count",
    "reasoning_words",
    "constraint_count",
    "has_context",
    "format_complexity",
] + [f"task_{t}" for t in TASK_TYPES]

def extract_features(prompt: str, task_type: str | None = None) -> dict[str, float]:
    """Return {feature_name: value} for one prompt.
 
    If task_type is missing or unknown, it is guessed from the text.
    """
    text = prompt or ""
    task = normalize_task_type(task_type) or guess_task_type(text)
    n_tokens = count_tokens(text)
 
    feats = {
        "token_count": float(n_tokens),
        # Logistic regression copes better with log(length): 50 vs 100 tokens
        # matters more than 2000 vs 2050.
        "log_token_count": math.log1p(n_tokens),
        "reasoning_words": float(count_reasoning_words(text)),
        "constraint_count": float(count_constraints(text)),
        "has_context": float(has_context(text)),
        "format_complexity": float(format_complexity(text)),
    }
    # One-hot task type: exactly one of these is 1.0.
    for t in TASK_TYPES:
        feats[f"task_{t}"] = 1.0 if t == task else 0.0
    return feats
 
 
def to_vector(feats: dict[str, float]) -> list[float]:
    """Dict -> list in the fixed FEATURE_NAMES order (order matters to the model)."""
    return [feats[name] for name in FEATURE_NAMES]
 
 
# ---------------------------------------------------------------------------
# 4. Batch mode: compute features for the whole dataset and save them.
# ---------------------------------------------------------------------------
 
def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def prompt_text(record: dict) -> str:
    """Join the user turns of a chat-style record into one string"""
    return "\n\n".join(
        m["content"] for m in record["messages"]
        if m.get("role") == "user" and isinstance(m.get("content"), str)
    )
 
 
def main() -> None:
    p = argparse.ArgumentParser(description="Compute handcrafted features for every prompt.")
    p.add_argument("--input", default="data/prompts.jsonl")
    p.add_argument("--output", default="data/features.npz")
    p.add_argument("--id-field", default="id")
    p.add_argument("--task-field", default="task_type")
    p.add_argument(
        "--guess-task-type",
        action="store_true",
        help="Ignore hand tags and use guess_task_type(), i.e. exactly what production will see.",
    )
    args = p.parse_args()
 
    records = load_jsonl(Path(args.input))
    ids, rows = [], []
    tagged = agree = 0
 
    for r in records:
        prompt = prompt_text(r)
        hand = normalize_task_type(r.get(args.task_field))
        if hand is None and not args.guess_task_type:
            raise ValueError(f"Prompt {r[args.id_field]} has no valid task type and --guess-task-type is not set.")
        guessed = guess_task_type(prompt)
        if hand is not None:
            tagged += 1
            agree += guessed == hand
 
        task = guessed if (args.guess_task_type or hand is None) else hand
        rows.append(to_vector(extract_features(prompt, task)))
        ids.append(str(r[args.id_field]))
 
    X = np.array(rows, dtype=np.float32)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        ids=np.array(ids),
        X=X,
        feature_names=np.array(FEATURE_NAMES),
        feature_version=np.array(FEATURE_VERSION),
    )
 
    print(f"Saved {X.shape[0]} prompts x {X.shape[1]} features -> {out}")
    print("\nFeature averages (sanity check):")
    for name, mean in zip(FEATURE_NAMES, X.mean(axis=0)):
        print(f"  {name:<20} {mean:8.2f}")
    if tagged:
        print(f"\nTask-type guess matches your hand tag on {agree}/{tagged} prompts ({agree / tagged:.0%}).")
 
 
if __name__ == "__main__":
    main()