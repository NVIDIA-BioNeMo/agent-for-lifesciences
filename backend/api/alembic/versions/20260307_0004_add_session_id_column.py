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

"""Add session_id column to data_artifacts table

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-07

Denormalizes session_id from metadata_json into a real column to enable
efficient SQL-level filtering instead of fetching all rows and filtering
in Python.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '0004'
down_revision: Union[str, None] = '0003'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'data_artifacts',
        sa.Column('session_id', sa.String(64), nullable=True)
    )
    op.create_index(
        'ix_data_artifacts_session_id',
        'data_artifacts',
        ['session_id']
    )


def downgrade() -> None:
    op.drop_index('ix_data_artifacts_session_id', table_name='data_artifacts')
    op.drop_column('data_artifacts', 'session_id')
