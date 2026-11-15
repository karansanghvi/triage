"""scripts/import_sqlite_log.py: copy a pre-database-layer request log into the database. No network."""
import hashlib
import json
import sqlite3
from dataclasses import fields

import numpy as np
import pytest

from app import db
from app.api.request_log import LabelRow, RequestRow, SQLiteRequestLog, SQLRequestLog, embedding_to_bytes
from scripts import import_sqlite_log as importer

OLD_TYPES = {str: "TEXT", int: "INTEGER", float: "REAL", bool: "INTEGER", bytes: "BLOB", list: "TEXT"}


def _old_type(f):
    from typing import get_args
    args = [a for a in get_args(f.type) if a is not type(None)]
    return OLD_TYPES[args[0] if args else f.type]


def legacy_file(path, version=4, rows=(), labels=()):
    """A request log as the code before the database layer wrote it: schema_version table, bools as 0/1, JSON
    as text. Version 3 had no preferred_provider column; no version had user_id."""
    skip = {"user_id"} | ({"preferred_provider"} if version < 4 else set())
    cols = [f for f in fields(RequestRow) if f.name not in skip]
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    raw.execute("INSERT INTO schema_version VALUES (?)", (version,))
    raw.execute("CREATE TABLE requests (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                + ", ".join(f"{f.name} {_old_type(f)}" for f in cols) + ")")
    raw.execute("CREATE TABLE labels (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                + ", ".join(f"{f.name} {_old_type(f)}" for f in fields(LabelRow)) + ")")
    for r in rows:
        values = []
        for f in cols:
            v = getattr(r, f.name)
            values.append(json.dumps(v) if f.name in ("attempts", "features") and v is not None
                          else int(v) if isinstance(v, bool) else v)
        raw.execute(f"INSERT INTO requests ({', '.join(f.name for f in cols)}) VALUES ({', '.join('?' * len(cols))})",
                    values)
    for l in labels:
        names = [f.name for f in fields(LabelRow)]
        raw.execute(f"INSERT INTO labels ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})",
                    [int(v) if isinstance(v, bool) else v for v in (getattr(l, n) for n in names)])
    raw.commit()
    raw.close()


def row(i, **kw):
    base = dict(request_id=f"r{i}", timestamp=f"2026-11-0{i + 1}T00:00:00+00:00", key_id="abcd1234",
                requested_model="auto", stream=bool(i % 2), tier="low", fell_back=False, cache_hit=False,
                attempts=[{"model": "groq-allam", "outcome": "success", "error_kind": None}], price_as_cost=0.001 * i,
                p_low=0.9, p_medium=0.95, tau=0.86, mode="api", training_eligible=True,
                features=[0.5] * 14, embedding=embedding_to_bytes(np.arange(384) / 384), feature_version="3",
                embedding_model="all-MiniLM-L6-v2")
    return RequestRow(**(base | kw))


def label(rid, **kw):
    return LabelRow(**(dict(request_id=rid, created_at="2026-11-02T00:00:00+00:00", status="labeled", passed=True,
                            judge_score=5, calls=2, tokens=300, cost=0.002) | kw))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def target(tmp_path):
    engine = db.open_engine(db.sqlite_url(tmp_path / "triage.sqlite3"))
    yield engine
    engine.dispose()


def test_rows_and_labels_are_copied_exactly(tmp_path, target):
    old = tmp_path / "old.sqlite3"
    rows = [row(0), row(1, preferred_provider="anthropic", mode="byok", features=None, embedding=None,
                        feature_version=None, embedding_model=None, training_eligible=False), row(2)]
    legacy_file(old, rows=rows, labels=[label("r0"), label("r2", passed=False, judge_score=2)])
    before = digest(old)
    counts = importer.import_log(old, target)
    assert counts == {"schema_version": 4, "rows_in_file": 3, "imported": 3, "already_there": 0,
                      "labels_imported": 2, "would_import": None}
    log = SQLRequestLog(target)
    assert log.rows() == rows                                   # every field, bools, JSON and vectors intact
    assert all(r.user_id is None for r in log.rows())
    assert [(l.request_id, l.passed) for l in log.labels()] == [("r0", True), ("r2", False)]
    assert digest(old) == before                                # the old file is only read


def test_running_again_imports_nothing_twice(tmp_path, target):
    old = tmp_path / "old.sqlite3"
    legacy_file(old, rows=[row(0), row(1)], labels=[label("r0")])
    importer.import_log(old, target)
    counts = importer.import_log(old, target)
    assert (counts["imported"], counts["already_there"], counts["labels_imported"]) == (0, 2, 0)
    log = SQLRequestLog(target)
    assert len(log.rows()) == 2 and len(log.labels()) == 1


def test_a_partly_imported_file_adds_only_the_new_rows_and_their_labels(tmp_path, target):
    first, second = tmp_path / "a.sqlite3", tmp_path / "b.sqlite3"
    legacy_file(first, rows=[row(0)], labels=[label("r0")])
    legacy_file(second, rows=[row(0), row(1)], labels=[label("r0"), label("r1")])
    importer.import_log(first, target)
    counts = importer.import_log(second, target)
    assert (counts["imported"], counts["labels_imported"]) == (1, 1)


def test_a_version_3_file_has_no_preferred_provider(tmp_path, target):
    old = tmp_path / "old.sqlite3"
    legacy_file(old, version=3, rows=[row(0)])
    assert importer.import_log(old, target)["schema_version"] == 3
    (copied,) = SQLRequestLog(target).rows()
    assert copied.preferred_provider is None and copied.p_low == 0.9


def test_dry_run_writes_nothing(tmp_path, target):
    old = tmp_path / "old.sqlite3"
    legacy_file(old, rows=[row(0), row(1)])
    counts = importer.import_log(old, target, dry_run=True)
    assert (counts["would_import"], counts["imported"]) == (2, 0)
    assert SQLRequestLog(target).rows() == []


def test_a_file_that_is_not_an_old_log_is_refused(tmp_path, target):
    new_format = tmp_path / "new.sqlite3"
    SQLiteRequestLog(new_format).close()
    with pytest.raises(SystemExit, match="nothing to import"):
        importer.import_log(new_format, target)


def test_main_imports_into_database_url_and_prints_counts(tmp_path, monkeypatch, capsys):
    old = tmp_path / "old.sqlite3"
    legacy_file(old, rows=[row(0)])
    monkeypatch.setenv("DATABASE_URL", db.sqlite_url(tmp_path / "into.sqlite3"))
    importer.main([str(old)])
    out = json.loads(capsys.readouterr().out)
    assert out["imported"] == 1 and out["into"].endswith("into.sqlite3")
    assert [r.request_id for r in SQLiteRequestLog(tmp_path / "into.sqlite3").rows()] == ["r0"]


def test_the_printed_target_hides_a_password(tmp_path, monkeypatch, capsys):
    old = tmp_path / "old.sqlite3"
    legacy_file(old, rows=[row(0)])
    monkeypatch.setattr(db, "open_engine", lambda url: db.create_engine(db.sqlite_url(tmp_path / "x.sqlite3")))
    monkeypatch.setattr(importer, "import_log", lambda *a, **k: {"imported": 0})
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:SECRETPW@db.example/triage")
    importer.main([str(old)])
    out = json.loads(capsys.readouterr().out)
    assert out["into"] == "postgresql+psycopg://user:***@db.example/triage"
