"""Users, sessions and sign-in rate limits (Phase 5, accounts PR 3).

users.id is a UUID kept as text (the same type on SQLite and Postgres, and the type of requests.user_id, which
now references it: deleting a user sets their request rows' user_id to NULL, keeping the rows).
auth_attempts holds one row per counted attempt; `subject` is always a keyed hash (the visitor_id of an IP, or
an HMAC of a normalized email), never an IP or an email.

Revision ID: 0003
Revises: 0002
"""
import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text()),
        sa.Column("preferred_provider", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_users_email", "users", ["email"], unique=True)
    op.create_table(
        "sessions",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id", name="fk_sessions_user_id_users",
                                                      ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_sessions_token_hash", "sessions", ["token_hash"], unique=True)
    op.create_index("idx_sessions_user_id", "sessions", ["user_id"])
    op.create_table(
        "auth_attempts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("kind", sa.Text(), nullable=False),          # signup_ip | login_ip | login_failure_email
        sa.Column("subject", sa.Text(), nullable=False),       # a keyed hash, never an IP or an email
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("idx_auth_attempts_lookup", "auth_attempts", ["kind", "subject", "created_at"])
    with op.batch_alter_table("requests") as batch:            # SQLite: rebuilds the table, keeping every row
        batch.create_foreign_key("fk_requests_user_id_users", "users", ["user_id"], ["id"], ondelete="SET NULL")


def downgrade() -> None:
    with op.batch_alter_table("requests") as batch:
        batch.drop_constraint("fk_requests_user_id_users", type_="foreignkey")
    op.drop_table("auth_attempts")
    op.drop_table("sessions")
    op.drop_table("users")
