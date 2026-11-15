"""Copy a request log from before the database layer (its own SQLite file with a schema_version table, e.g.
data/logs/requests.sqlite3) into the app's database (DATABASE_URL, default data/triage.sqlite3).

    venv/bin/python scripts/import_sqlite_log.py data/logs/requests.sqlite3 [--dry-run]

- The old file is opened read-only and never changed.
- Safe to run again: rows whose request_id is already in the database are skipped, and so are their labels
  (labels have no id of their own, so a label is copied only with its request, in the same run).
- All-or-nothing: everything is written in one transaction.
- Any old schema version works: a column the old file doesn't have (e.g. preferred_provider before version 4)
  is NULL, as are the new user_id values. Columns the code no longer knows are ignored.
- For Postgres, the database must already be at the latest revision (alembic upgrade head).
"""
import argparse
import json
import sqlite3
import sys
from dataclasses import fields
from pathlib import Path
from typing import get_args

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select                                          # noqa: E402

from app import db                                                     # noqa: E402
from app.api.request_log import LabelRow, RequestRow, _to_sql          # noqa: E402

_JSON = {"attempts", "features"}


def _is_bool(f) -> bool:
    return f.type is bool or bool in get_args(f.type)


def _value(f, raw):
    """An old SQLite value as the dataclass holds it (bools were stored as 0/1, JSON as text)."""
    if raw is None:
        return None
    if f.name in _JSON:
        return json.loads(raw)
    if _is_bool(f):
        return bool(raw)
    return raw


def read_legacy(path: Path) -> tuple:
    """(schema version, [RequestRow], {request_id: [LabelRow]}) from an old log, opened read-only."""
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in source.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "schema_version" not in tables or "requests" not in tables:
            raise SystemExit(f"{path} is not a request log from before the database layer (no schema_version "
                             "table); nothing to import")
        version = source.execute("SELECT version FROM schema_version").fetchone()[0]
        rows = [RequestRow(**r) for r in _read(source, "requests", RequestRow)]
        labels = {}
        if "labels" in tables:
            for label in _read(source, "labels", LabelRow):
                labels.setdefault(label["request_id"], []).append(LabelRow(**label))
        return version, rows, labels
    finally:
        source.close()


def _read(source, table: str, dataclass_type) -> list:
    present = {c[1] for c in source.execute(f"PRAGMA table_info({table})")}
    wanted = [f for f in fields(dataclass_type) if f.name in present]
    found = source.execute(f"SELECT {', '.join(f.name for f in wanted)} FROM {table} ORDER BY id").fetchall()
    out = []
    for values in found:
        data = {f.name: _value(f, v) for f, v in zip(wanted, values)}
        if dataclass_type is RequestRow:
            data["attempts"] = data.get("attempts") or []
            data["training_eligible"] = bool(data.get("training_eligible"))
        out.append(data)
    return out


def import_log(path: Path, engine, dry_run: bool = False) -> dict:
    """Copy the rows (and their labels) that aren't in `engine`'s database yet. Returns the counts."""
    version, rows, labels = read_legacy(path)
    requests, label_table = db.requests_table, db.labels_table
    with engine.begin() as connection:
        existing = set(connection.execute(select(requests.c.request_id)).scalars())
        new = [r for r in rows if r.request_id not in existing]
        new_labels = [label for r in new for label in labels.get(r.request_id, [])]
        if not dry_run:
            if new:
                connection.execute(requests.insert(), [
                    {f.name: _to_sql(f.name, getattr(r, f.name)) for f in fields(RequestRow)} for r in new])
            if new_labels:
                connection.execute(label_table.insert(), [
                    {f.name: _to_sql(f.name, getattr(l, f.name)) for f in fields(LabelRow)} for l in new_labels])
    return {"schema_version": version, "rows_in_file": len(rows), "imported": 0 if dry_run else len(new),
            "already_there": len(rows) - len(new), "labels_imported": 0 if dry_run else len(new_labels),
            "would_import": len(new) if dry_run else None}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", type=Path, help="the old request log, e.g. data/logs/requests.sqlite3")
    parser.add_argument("--dry-run", action="store_true", help="count what would be imported; write nothing")
    args = parser.parse_args(argv)
    if not args.path.exists():
        raise SystemExit(f"{args.path} doesn't exist")
    url = db.database_url()
    engine = db.open_engine(url)
    try:
        counts = import_log(args.path, engine, dry_run=args.dry_run)
    finally:
        engine.dispose()
    from sqlalchemy.engine import make_url
    target = make_url(url).render_as_string(hide_password=True)          # never print a database password
    print(json.dumps({"into": target, **{k: v for k, v in counts.items() if v is not None}}, indent=2))


if __name__ == "__main__":
    main()
