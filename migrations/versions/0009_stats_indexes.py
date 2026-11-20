"""Indexes for the dashboard's stats queries (Phase 5).

    idx_requests_model_ts     (requested_model, timestamp)  GET /v1/stats/public: model="auto" rows in a time range
    idx_requests_user_ts      (user_id, timestamp)          GET /v1/account/usage: one account's rows in a time range
    idx_requests_user_id_id   (user_id, id)                 GET /v1/account/requests: newest first, by keyset cursor

Revision ID: 0009
Revises: 0008
"""
from alembic import op

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index("idx_requests_model_ts", "requests", ["requested_model", "timestamp"])
    op.create_index("idx_requests_user_ts", "requests", ["user_id", "timestamp"])
    op.create_index("idx_requests_user_id_id", "requests", ["user_id", "id"])


def downgrade() -> None:
    op.drop_index("idx_requests_user_id_id", table_name="requests")
    op.drop_index("idx_requests_user_ts", table_name="requests")
    op.drop_index("idx_requests_model_ts", table_name="requests")
