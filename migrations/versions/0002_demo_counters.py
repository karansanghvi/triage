"""Demo and public-BYOK usage counters on the database layer (Phase 5, accounts PR 2).

Same tables the DemoLimiter kept in its own SQLite file (data/demo/usage.sqlite3). Its counters are not copied:
they are per UTC day and kept 7 days, so the new tables simply start empty.

Revision ID: 0002
Revises: 0001
"""
import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "visitor_day",
        sa.Column("day", sa.Text(), primary_key=True),             # "YYYY-MM-DD", UTC
        sa.Column("kind", sa.Text(), primary_key=True),            # demo | byok
        sa.Column("visitor", sa.Text(), primary_key=True),         # visitor_id(): a keyed hash, never an IP
        sa.Column("requests", sa.Integer(), nullable=False),
    )
    op.create_table(
        "demo_day",
        sa.Column("day", sa.Text(), primary_key=True),
        sa.Column("requests", sa.Integer(), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False),         # settled actuals plus open reservations
        sa.Column("calls", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("demo_day")
    op.drop_table("visitor_day")
