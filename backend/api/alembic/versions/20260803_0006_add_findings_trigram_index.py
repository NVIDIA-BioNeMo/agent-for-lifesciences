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

"""Enable trigram search over the findings ledger

Revision ID: 0006
Revises: 0005
Create Date: 2026-08-03

The storage-state envelope truncates at 20 artifacts and the session log at
40 findings. Raising those caps only moves the cliff; giving the agent a way
to search past them removes it.

pg_trgm is what makes that search tolerant of how the model phrases a query --
"binding affinity" finding a summary that says "affinity", and a misspelling
finding it anyway. CREATE EXTENSION needs privileges the application role may
not have, so this migration tolerates failure: search falls back to substring
matching, which is narrower but correct.
"""
import logging
from typing import Sequence, Union

from alembic import op

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return

    try:
        op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    except Exception as exc:
        logger.warning(
            f"Could not create the pg_trgm extension ({exc}). Session-history "
            f"search will match substrings only. Grant the application role "
            f"the privilege, or create the extension as superuser, to enable "
            f"fuzzy ranking."
        )
        return

    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_findings_summary_trgm "
        "ON findings USING gin (summary gin_trgm_ops)"
    )


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        return
    op.execute("DROP INDEX IF EXISTS ix_findings_summary_trgm")
