"""Account API keys (Phase 5, accounts PR 5).

The full key ("trk_" + 43 base62 characters) is never stored: only its public id ("trk_" + the next 8
characters, unique, shown in lists) and an HMAC-SHA256 of the full key with TRIAGE_HASH_SECRET. Revoking sets
revoked_at and keeps the row (the request log may name its public id). Deleted with the user.

Revision ID: 0005
Revises: 0004
"""
import sqlalchemy as sa
from alembic import op

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "api_keys",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id", name="fk_api_keys_user_id_users",
                                                      ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("public_id", sa.Text(), nullable=False),
        sa.Column("secret_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True)),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_api_keys_public_id", "api_keys", ["public_id"], unique=True)
    op.create_index("idx_api_keys_user_id", "api_keys", ["user_id"])


def downgrade() -> None:
    op.drop_table("api_keys")
