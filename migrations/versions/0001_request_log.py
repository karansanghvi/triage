"""Request log and labels on the database layer (Phase 5, accounts PR 1).

The same columns the SQLite request log had at its schema version 4, plus requests.user_id (nullable; set by
account requests in a later PR). A pre-Alembic SQLite log is not upgraded in place: scripts/import_sqlite_log.py
copies its rows.

Revision ID: 0001
Revises:
"""
import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "requests",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.Text(), nullable=False),
        sa.Column("timestamp", sa.Text()),
        sa.Column("key_id", sa.Text()),
        sa.Column("requested_model", sa.Text()),
        sa.Column("stream", sa.Boolean()),
        sa.Column("status", sa.Text()),
        sa.Column("http_status", sa.Integer()),
        sa.Column("error_kind", sa.Text()),
        sa.Column("quality", sa.Text()),
        sa.Column("tau", sa.Double()),
        sa.Column("task_type", sa.Text()),
        sa.Column("task_type_guessed", sa.Boolean()),
        sa.Column("classifier", sa.Text()),
        sa.Column("config_version", sa.Integer()),
        sa.Column("p_low", sa.Double()),
        sa.Column("p_medium", sa.Double()),
        sa.Column("tier", sa.Text()),
        sa.Column("primary_model", sa.Text()),
        sa.Column("served_model", sa.Text()),
        sa.Column("fell_back", sa.Boolean()),
        sa.Column("attempts", sa.Text()),
        sa.Column("temperature", sa.Double()),
        sa.Column("max_tokens", sa.Integer()),
        sa.Column("settings_match_training", sa.Boolean()),
        sa.Column("cache_hit", sa.Boolean()),
        sa.Column("input_tokens", sa.Integer()),
        sa.Column("output_tokens", sa.Integer()),
        sa.Column("reasoning_tokens", sa.Integer()),
        sa.Column("tokens_estimated", sa.Boolean()),
        sa.Column("cost", sa.Double()),
        sa.Column("price_as_cost", sa.Double()),
        sa.Column("strong_cost", sa.Double()),
        sa.Column("saved", sa.Double()),
        sa.Column("latency_ms", sa.Double()),
        sa.Column("ttft_ms", sa.Double()),
        sa.Column("prompt_hmac", sa.Text()),
        sa.Column("verified", sa.Text()),
        sa.Column("escalated", sa.Boolean()),
        sa.Column("checker_verdict", sa.Text()),
        sa.Column("decided_by", sa.Text()),
        sa.Column("checker_version", sa.Integer()),
        sa.Column("mode", sa.Text()),
        sa.Column("preferred_provider", sa.Text()),
        sa.Column("user_id", sa.Text()),
        sa.Column("escalated_from", sa.Text()),
        sa.Column("escalated_to", sa.Text()),
        sa.Column("escalation_extra_cost", sa.Double()),
        sa.Column("escalation_extra_latency_ms", sa.Double()),
        sa.Column("check_cost", sa.Double()),
        sa.Column("check_latency_ms", sa.Double()),
        sa.Column("features", sa.Text()),
        sa.Column("embedding", sa.LargeBinary()),
        sa.Column("feature_version", sa.Text()),
        sa.Column("embedding_model", sa.Text()),
        sa.Column("training_eligible", sa.Boolean()),
    )
    op.create_index("idx_requests_request_id", "requests", ["request_id"], unique=True)
    op.create_index("idx_requests_key_id", "requests", ["key_id"])
    op.create_index("idx_requests_timestamp", "requests", ["timestamp"])
    op.create_index("idx_requests_user_id", "requests", ["user_id"])
    op.create_table(
        "labels",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("request_id", sa.Text()),
        sa.Column("created_at", sa.Text()),
        sa.Column("status", sa.Text()),
        sa.Column("skip_reason", sa.Text()),
        sa.Column("passed", sa.Boolean()),
        sa.Column("judge_score", sa.Integer()),
        sa.Column("judge_version", sa.Integer()),
        sa.Column("judge_model", sa.Text()),
        sa.Column("reference_model", sa.Text()),
        sa.Column("served_model", sa.Text()),
        sa.Column("served_tier", sa.Text()),
        sa.Column("calls", sa.Integer()),
        sa.Column("tokens", sa.Integer()),
        sa.Column("cost", sa.Double()),
    )
    op.create_index("idx_labels_request_id", "labels", ["request_id"])
    op.create_index("idx_labels_created_at", "labels", ["created_at"])


def downgrade() -> None:
    op.drop_table("labels")
    op.drop_table("requests")
