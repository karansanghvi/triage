"""Saved provider keys (Phase 5, accounts PR 4).

One row per user and provider (saving again replaces it), deleted with the user. The key itself is only ever
stored encrypted (app/accounts/encryption.py): ciphertext + nonce, and the id of the encryption key used, which
lives in TRIAGE_ENCRYPTION_KEYS, never here. last4 is all that's kept in plain form, for display.

Revision ID: 0004
Revises: 0003
"""
import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "provider_keys",
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.id", name="fk_provider_keys_user_id_users",
                                                      ondelete="CASCADE"), primary_key=True),
        sa.Column("provider", sa.Text(), primary_key=True),          # openai | anthropic | groq | gemini
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("nonce", sa.LargeBinary(), nullable=False),
        sa.Column("enc_key_id", sa.Text(), nullable=False),
        sa.Column("last4", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),              # valid | invalid | unverified
        sa.Column("status_detail", sa.Text()),                       # short and redacted
        sa.Column("verified_at", sa.DateTime(timezone=True)),        # the last check that got an answer
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("provider_keys")
