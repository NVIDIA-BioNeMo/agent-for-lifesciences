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

"""Add durable job semantics columns to jobs

Revision ID: 0009
Revises: 0008
Create Date: 2026-08-29

P0 architecture-contracts work (see internal_docs/docs/0001-architecture-boundaries.md).
Adds session_id/request_id (so jobs are queryable per-session like artifacts
already are), upstream_job_id and poll_after (so an NVCF 202/queued response
can be recorded instead of only raised as an error), and the
QUEUED_UPSTREAM enum value. All new columns are nullable; existing rows and
existing INSERTs that don't set them are unaffected.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ALTER TYPE ... ADD VALUE cannot run inside the transaction Alembic
    # normally wraps a migration in on Postgres.
    with op.get_context().autocommit_block():
        op.execute("ALTER TYPE jobstatus ADD VALUE IF NOT EXISTS 'queued_upstream'")
    op.add_column("jobs", sa.Column("session_id", sa.String(64), nullable=True))
    op.add_column("jobs", sa.Column("request_id", sa.String(64), nullable=True))
    op.add_column("jobs", sa.Column("upstream_job_id", sa.String(255), nullable=True))
    op.add_column(
        "jobs", sa.Column("poll_after", sa.DateTime(timezone=True), nullable=True)
    )
    op.create_index("ix_jobs_session_id", "jobs", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_jobs_session_id", table_name="jobs")
    op.drop_column("jobs", "poll_after")
    op.drop_column("jobs", "upstream_job_id")
    op.drop_column("jobs", "request_id")
    op.drop_column("jobs", "session_id")
    # Postgres cannot drop an enum value; the 'queued_upstream' label is left
    # in place on downgrade (harmless if simply unused).
