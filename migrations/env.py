"""Alembic environment for Triage (app/db.py).

From the command line (a deploy step):  DATABASE_URL=postgresql+psycopg://... alembic upgrade head
From code (SQLite on open, tests):      app.db.upgrade(engine), which passes its own connection.
"""
from alembic import context

from app.db import database_url, metadata
from sqlalchemy import create_engine

config = context.config
target_metadata = metadata


def run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata,
                      render_as_batch=connection.dialect.name == "sqlite")   # SQLite: ALTER via table copies
    with context.begin_transaction():
        context.run_migrations()


connection = config.attributes.get("connection")
if connection is not None:
    run_migrations(connection)
elif context.is_offline_mode():
    context.configure(url=database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(database_url())
    with engine.connect() as conn:
        run_migrations(conn)
        conn.commit()
    engine.dispose()
