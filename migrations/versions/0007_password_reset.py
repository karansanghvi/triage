"""Password reset by emailed code (Phase 5, accounts).

password_reset_codes: one row per user (a new code replaces the earlier one). Only an HMAC of the 6-digit code
is stored (TRIAGE_HASH_SECRET, with the user id), never the code. last_sent_at drives the 60-second resend
cooldown and outlives the code. password_reset_tokens: one row per user; only the SHA-256 of the "rst_" token.
Neither table holds an email address. Both are deleted with the user.

Revision ID: 0007
Revises: 0006
"""
import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "password_reset_codes",
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id", name="fk_password_reset_codes_user_id_users",
                                                      ondelete="CASCADE"), primary_key=True),
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("code_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True)),
        sa.Column("last_sent_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "password_reset_tokens",
        sa.Column("token_hash", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id", name="fk_password_reset_tokens_user_id_users",
                                                      ondelete="CASCADE"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_password_reset_tokens_user_id", "password_reset_tokens", ["user_id"], unique=True)


def downgrade() -> None:
    op.drop_table("password_reset_tokens")
    op.drop_table("password_reset_codes")
