"""Daily chat request counts per account (Phase 5, accounts PR 6). Deleted with the user.

Revision ID: 0006
Revises: 0005
"""
import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "account_day",
        sa.Column("day", sa.Text(), primary_key=True),                 # "YYYY-MM-DD", UTC
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id", name="fk_account_day_user_id_users",
                                                      ondelete="CASCADE"), primary_key=True),
        sa.Column("requests", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("account_day")
