"""
Database migration: Add shareholding pipeline processing field to Filing model.
"""
from alembic import op
import sqlalchemy as sa


def upgrade():
    """Add shareholding pipeline processing field."""
    op.add_column('filings', sa.Column('processed_with_shareholding_pipeline', sa.Boolean(), default=False))


def downgrade():
    """Remove shareholding pipeline processing field."""
    op.drop_column('filings', 'processed_with_shareholding_pipeline')