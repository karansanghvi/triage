""".env.example lists every environment variable the code reads, and leaves every secret empty."""
import re
from pathlib import Path

from app.registry import PROJECT_ROOT

EXAMPLE = PROJECT_ROOT / ".env.example"
SECRETS = ("TRIAGE_API_KEYS", "TRIAGE_ADMIN_KEYS", "TRIAGE_HASH_SECRET", "TRIAGE_ENCRYPTION_KEYS", "DATABASE_URL",
           "TRIAGE_TEST_DATABASE_URL", "GROQ_API_KEY", "GEMINI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY",
           "ANTHROPIC_API_KEY")


def example_values() -> dict:
    values = {}
    for line in EXAMPLE.read_text().splitlines():
        if line and not line.startswith("#"):
            name, _, value = line.partition("=")
            values[name.strip()] = value.strip()
    return values


def variables_in_code() -> set:
    found = set()
    for folder in ("app", "scripts", "migrations"):
        for path in (PROJECT_ROOT / folder).rglob("*.py"):
            found |= set(re.findall(r'"((?:TRIAGE|OPENAI|ANTHROPIC|GROQ|GEMINI|OPENROUTER|OLLAMA|HF)_[A-Z_]+|'
                                    r'DATABASE_URL)"', path.read_text()))
    found.add("TRIAGE_TEST_DATABASE_URL")                           # read by tests/test_db_postgres.py
    return found


def test_every_variable_the_code_reads_is_in_env_example():
    missing = variables_in_code() - set(example_values())
    assert not missing, f"add to .env.example: {sorted(missing)}"


def test_every_variable_has_a_comment_above_it_or_in_its_group():
    lines = EXAMPLE.read_text().splitlines()
    for i, line in enumerate(lines):
        if line and not line.startswith("#"):
            above = lines[i - 1] if i else ""
            assert above.startswith("#") or (lines[i - 1] and not lines[i - 1].startswith("#")), line


def test_secrets_are_empty():
    values = example_values()
    for name in SECRETS:
        assert values[name] == "", name
