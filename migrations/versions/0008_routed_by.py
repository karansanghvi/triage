"""requests.routed_by: what decided the tier (classifier | small_talk | requested_model). Phase 5, small talk.

Revision ID: 0008
Revises: 0007
"""
import sqlalchemy as sa
from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # batch mode: SQLite can't ALTER TABLE ... ADD COLUMN inside every Alembic operation
    with op.batch_alter_table("requests") as batch:
        batch.add_column(sa.Column("routed_by", sa.Text()))


def downgrade() -> None:
    with op.batch_alter_table("requests") as batch:
        batch.drop_column("routed_by")
