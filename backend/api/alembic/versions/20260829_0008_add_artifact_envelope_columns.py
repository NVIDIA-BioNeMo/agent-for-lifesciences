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

"""Add scientific artifact envelope columns to data_artifacts

Revision ID: 0008
Revises: 0007
Create Date: 2026-08-29

P0 architecture-contracts work (see internal_docs/docs/0001-architecture-boundaries.md).
Adds structured, nullable provenance/units/uncertainty/validation columns
alongside the existing freeform metadata_json, so producer tools can record
lineage without every reader of metadata_json changing. Nullable everywhere:
existing rows and existing INSERTs that don't set these columns are
unaffected.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "data_artifacts",
        sa.Column("provenance_json", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "data_artifacts",
        sa.Column("units", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "data_artifacts",
        sa.Column("uncertainty", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "data_artifacts",
        sa.Column("validation_json", postgresql.JSONB(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("data_artifacts", "validation_json")
    op.drop_column("data_artifacts", "uncertainty")
    op.drop_column("data_artifacts", "units")
    op.drop_column("data_artifacts", "provenance_json")
