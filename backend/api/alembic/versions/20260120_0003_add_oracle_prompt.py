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

"""Add oracle_prompt column to training_queries table

Revision ID: 0003
Revises: 0002
Create Date: 2026-01-20

This migration adds the oracle_prompt field to training_queries for storing
per-query oracle prompts used in DPO training data collection.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '0003'
down_revision: Union[str, None] = '0002'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add oracle_prompt and oracle_prompt_updated_at columns."""
    op.add_column(
        'training_queries',
        sa.Column('oracle_prompt', sa.Text(), nullable=True)
    )
    op.add_column(
        'training_queries',
        sa.Column('oracle_prompt_updated_at', sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    """Remove oracle_prompt and oracle_prompt_updated_at columns."""
    op.drop_column('training_queries', 'oracle_prompt_updated_at')
    op.drop_column('training_queries', 'oracle_prompt')
