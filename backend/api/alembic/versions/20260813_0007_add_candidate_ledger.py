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

"""Add the candidate ledger

Revision ID: 0007
Revises: 0006
Create Date: 2026-08-13

The findings ledger records one row per tool execution, which is the right grain
for "what did I do" and the wrong one for "which molecule was best". A molecule
accumulates evidence across many calls -- descriptors from one, an ADMET panel
from another, an affinity from a third -- so it gets its own table keyed by
canonical SMILES, upserted as each measurement arrives.

The unique constraint on (session_id, smiles) is the whole point: it is what
turns repeated measurement into one comparable row instead of a pile of
fragments a reader has to reassemble.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "candidates",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("smiles", sa.Text(), nullable=False),
        sa.Column("origin", sa.String(128), nullable=False, server_default=""),
        sa.Column("properties", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("scores", postgresql.JSONB(), nullable=False, server_default="{}"),
        sa.Column("gates", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column("failures", postgresql.JSONB(), nullable=False, server_default="[]"),
        # Nullable on purpose: "not judged yet" is not "failed".
        sa.Column("passed", sa.Boolean(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "session_id", "smiles", name="uq_candidates_session_smiles"
        ),
    )
    op.create_index("ix_candidates_session_id", "candidates", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_candidates_session_id", table_name="candidates")
    op.drop_table("candidates")
