"""requests.status "stopped": a request the client stopped part-way (the web chat's Stop) is its own outcome, not
a failure. Rows logged before this as status "error" with error_kind "client_disconnected" become "stopped".

Revision ID: 0010
Revises: 0009
"""
from alembic import op

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("UPDATE requests SET status = 'stopped' WHERE status = 'error' AND error_kind = 'client_disconnected'")


def downgrade() -> None:
    op.execute("UPDATE requests SET status = 'error' WHERE status = 'stopped'")
