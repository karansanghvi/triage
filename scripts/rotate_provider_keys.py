"""Re-encrypt every saved provider key with the first key in TRIAGE_ENCRYPTION_KEYS (step 2 of a rotation).

    TRIAGE_ENCRYPTION_KEYS="k2:<new>,k1:<old>" venv/bin/python scripts/rotate_provider_keys.py [--dry-run]

1. Check: every row must decrypt with one of the configured keys. If any can't (its key id isn't configured, or
   the data is damaged), nothing is changed and the script exits with status 1: drop no key yet.
2. Re-encrypt, in batches of --batch-size rows, one transaction per batch: each row not already on the first
   key is decrypted and encrypted again with it (a fresh nonce). A row whose ciphertext changed meanwhile (the
   user saved a new key) is left alone: it's already on the first key.

Safe to run again: rows already on the first key are skipped. Prints only counts and key ids, never a key.
Uses DATABASE_URL (default data/triage.sqlite3); Postgres must already be at the latest migration.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import and_, select, update                                # noqa: E402

from app import db                                                         # noqa: E402
from app.accounts.encryption import DecryptionError, KeyRing, Sealed       # noqa: E402

DEFAULT_BATCH = 100


def _sealed(row) -> Sealed:
    return Sealed(bytes(row.ciphertext), bytes(row.nonce), row.enc_key_id)


def check(engine, keyring: KeyRing) -> dict:
    """{"rows", "unreadable", "by_key": {key id: count}}. Decrypts each row to prove it can, keeping nothing."""
    t = db.provider_keys_table
    counts, unreadable, total = {}, 0, 0
    with engine.connect() as connection:
        for row in connection.execute(select(t.c.user_id, t.c.provider, t.c.ciphertext, t.c.nonce, t.c.enc_key_id)):
            total += 1
            counts[row.enc_key_id] = counts.get(row.enc_key_id, 0) + 1
            try:
                keyring.decrypt(_sealed(row), row.user_id, row.provider)
            except DecryptionError:
                unreadable += 1
    return {"rows": total, "unreadable": unreadable, "by_key": counts}


def rotate(engine, keyring: KeyRing, batch_size: int = DEFAULT_BATCH, dry_run: bool = False) -> dict:
    """Re-encrypt every row that isn't on keyring.primary_id. Returns counts; raises SystemExit (changing
    nothing) if any row can't be decrypted."""
    before = check(engine, keyring)
    result = {"primary": keyring.primary_id, "rows": before["rows"], "by_key_before": before["by_key"],
              "already_current": before["by_key"].get(keyring.primary_id, 0), "re_encrypted": 0,
              "changed_meanwhile": 0, "dry_run": dry_run}
    if before["unreadable"]:
        result["unreadable"] = before["unreadable"]
        print(json.dumps(result, indent=2))
        raise SystemExit(f"refusing to rotate: {before['unreadable']} saved key(s) can't be decrypted with the "
                         "configured keys. Nothing was changed; keep every old key in TRIAGE_ENCRYPTION_KEYS.")
    if dry_run:
        result["would_re_encrypt"] = before["rows"] - result["already_current"]
        return result
    t = db.provider_keys_table
    while True:
        with engine.begin() as connection:              # one transaction per batch
            rows = connection.execute(
                select(t.c.user_id, t.c.provider, t.c.ciphertext, t.c.nonce, t.c.enc_key_id)
                .where(t.c.enc_key_id != keyring.primary_id).order_by(t.c.user_id, t.c.provider)
                .limit(batch_size)).fetchall()
            if not rows:
                break
            for row in rows:
                plain = keyring.decrypt(_sealed(row), row.user_id, row.provider)
                new = keyring.encrypt(plain, row.user_id, row.provider)
                plain = None
                changed = connection.execute(
                    update(t).where(and_(t.c.user_id == row.user_id, t.c.provider == row.provider,
                                         t.c.enc_key_id == row.enc_key_id, t.c.nonce == row.nonce))
                    .values(ciphertext=new.ciphertext, nonce=new.nonce, enc_key_id=new.key_id)).rowcount
                if changed:
                    result["re_encrypted"] += 1
                else:
                    result["changed_meanwhile"] += 1
    result["by_key_after"] = check(engine, keyring)["by_key"]
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dry-run", action="store_true", help="check every row and count; change nothing")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="rows per transaction")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be at least 1")
    keyring = KeyRing.from_env()                # refuses a missing or malformed TRIAGE_ENCRYPTION_KEYS
    engine = db.open_engine(db.database_url())
    try:
        result = rotate(engine, keyring, args.batch_size, args.dry_run)
    finally:
        engine.dispose()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
