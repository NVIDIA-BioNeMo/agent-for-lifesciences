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

"""A second write to a row must not raise, and must not lose the row.

A live run logged `Failed to capture candidates from admet_predict` and the
molecule's ADMET panel silently never reached the candidate ledger. The cause
was not in the capture logic: `Candidate.updated_at` carries
`onupdate=func.now()`, a value only the database computes, so SQLAlchemy
expires the attribute after an UPDATE flush. `ConversationStore.upsert_candidate`
flushes and immediately calls `candidate.to_dict()`, which reads `updated_at`
and triggers a lazy refresh -- IO from a sync context under an async session,
i.e. `MissingGreenlet`.

The shape of the bug is what made it survive a full 20-task run unnoticed:
SQLAlchemy 2.0 fetches server defaults via RETURNING on INSERT but not UPDATE,
so the *first* write of a molecule succeeded and only the *second* failed. The
existing ledger tests only exercised unpersisted objects, so nothing caught it.

Blast radius, for why this is worth a dedicated test: `capture_candidates`
wraps the whole loop and its commit in one try/except, so a raise on row N
discards rows 1..N-1 too. The molecule is then absent from `query_candidates`,
`best_candidate`, and the `[CANDIDATES]` prompt block -- the ledger stops being
able to rank on exactly the scores it exists to rank on, silently.

SQLite reproduces this faithfully: the expiry is SQLAlchemy's post-flush
behaviour, not a Postgres quirk.
"""

import pytest
import pytest_asyncio
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles

from api.db.models import Base, Candidate, Conversation, Job

pytest.importorskip("aiosqlite", reason="async SQLite driver for the DB-free suite")


# The models are typed for Postgres; teach those two types a SQLite spelling so
# the schema can be created in-process. Nothing else about the models changes,
# and the behaviour under test (post-flush attribute expiry) is dialect-neutral.
@compiles(JSONB, "sqlite")
def _jsonb_as_json(element, compiler, **kw):  # pragma: no cover - trivial
    return "JSON"


@compiles(UUID, "sqlite")
def _uuid_as_char(element, compiler, **kw):  # pragma: no cover - trivial
    return "CHAR(36)"


@pytest_asyncio.fixture
async def session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as s:
        yield s
    await engine.dispose()


def _candidate(**overrides) -> Candidate:
    fields = dict(
        session_id="s1",
        smiles="CCO",
        origin="molecule_gates",
        properties={},
        scores={},
        gates=[],
        failures=[],
        notes="",
    )
    fields.update(overrides)
    return Candidate(**fields)


@pytest.mark.asyncio
class TestSerialisingAfterAnUpdate:
    async def test_a_candidate_survives_being_written_twice(self, session):
        """The exact production sequence: molecule_gates, then admet_predict."""
        candidate = _candidate()
        session.add(candidate)
        await session.flush()
        assert candidate.to_dict()["smiles"] == "CCO"  # first write: always worked

        candidate.scores = {"hERG": 0.5}
        await session.flush()

        # Before the fix this raised MissingGreenlet and the panel was lost.
        assert candidate.to_dict()["scores"] == {"hERG": 0.5}

    async def test_updated_at_is_readable_rather_than_expired(self, session):
        candidate = _candidate()
        session.add(candidate)
        await session.flush()

        candidate.notes = "gated"
        await session.flush()

        assert candidate.to_dict()["updated_at"] is not None

    async def test_the_same_holds_for_the_other_models_with_sql_side_onupdate(
        self, session
    ):
        # Job and Conversation share Candidate's shape, so they shared the bug.
        job = Job(tool_name="vina_dock", arguments={})
        conversation = Conversation(session_id="s1")
        session.add_all([job, conversation])
        await session.flush()

        job.error = "boom"
        conversation.title = "renamed"
        await session.flush()

        assert job.to_dict()["updated_at"] is not None
        assert conversation.to_dict()["updated_at"] is not None


class TestTheGuardItself:
    def test_eager_defaults_is_inherited_by_every_model(self):
        """Pins the fix at the Base, so a new model cannot reintroduce this."""
        for model in (Candidate, Job, Conversation):
            assert model.__mapper__.eager_defaults is True
