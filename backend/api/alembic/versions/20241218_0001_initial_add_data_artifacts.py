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

"""Add data_artifacts table for tool data storage

Revision ID: 0001
Revises: 
Create Date: 2024-12-18

This migration adds the data_artifacts table which enables tools to store
and retrieve large data artifacts (like SDF files, PDB structures) via
references instead of passing raw data through the LLM.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create data_artifacts table."""
    op.create_table(
        'data_artifacts',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('data_type', sa.String(length=32), nullable=False),
        sa.Column('content', sa.LargeBinary(), nullable=False),
        sa.Column('metadata_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(
        op.f('ix_data_artifacts_data_type'),
        'data_artifacts',
        ['data_type'],
        unique=False
    )


def downgrade() -> None:
    """Drop data_artifacts table."""
    op.drop_index(op.f('ix_data_artifacts_data_type'), table_name='data_artifacts')
    op.drop_table('data_artifacts')



