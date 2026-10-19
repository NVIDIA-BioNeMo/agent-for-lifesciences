# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add training_queries table for DPO data collection

Revision ID: 0002
Revises: 0001
Create Date: 2026-01-18

This migration adds the training_queries table for storing test queries
used in DPO training data collection. Each query can be tested with different
prompt modes (oracle/weak) to generate chosen/rejected response pairs.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0002'
down_revision: Union[str, None] = '0001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create training_queries table."""
    op.create_table(
        'training_queries',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('expected_tool', sa.String(length=256), nullable=True),
        sa.Column('category', sa.String(length=64), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, default=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_training_queries_category'),
        'training_queries',
        ['category'],
        unique=False
    )
    
    # Add description column to data_artifacts if not exists
    # (this was added to the model but may not have been in the original migration)
    try:
        op.add_column(
            'data_artifacts',
            sa.Column('description', sa.String(length=256), nullable=True)
        )
    except Exception:
        # Column may already exist
        pass


def downgrade() -> None:
    """Drop training_queries table."""
    op.drop_index(op.f('ix_training_queries_category'), table_name='training_queries')
    op.drop_table('training_queries')
    
    # Note: We don't remove the description column from data_artifacts
    # as it may contain data
