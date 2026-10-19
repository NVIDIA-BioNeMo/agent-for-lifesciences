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

"""Read and write the server-side conversation record.

Deliberately explicit tables rather than a LangGraph checkpointer blob. A
checkpointer would be less code, but the record has to be *queried* -- for the
findings ledger, for session-history retrieval, and for training export -- and
an opaque pickled blob supports none of that.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from sqlalchemy import desc, func, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from api.db.models import Candidate, Conversation, Finding, Message, Turn

logger = logging.getLogger(__name__)

# Word-similarity threshold. Low enough that a typo or a partial word still
# matches, high enough that an unrelated finding does not. pg_trgm's own
# default for word similarity is 0.6, which rejects most single-character
# misspellings.
_TRIGRAM_THRESHOLD = 0.45

# Whether pg_trgm exists, per database URL. The answer cannot change while the
# process runs, and the probe is a round trip we would otherwise pay per search.
_TRIGRAM_SUPPORT: dict[str, bool] = {}


class ConversationStore:
    """Persistence for transcripts and findings, scoped to a DB session.

    Example:
        store = ConversationStore(session)
        conversation = await store.get_or_create_conversation(sid, "nemotron")
        turn = await store.start_turn(conversation.id)
        await store.append_message(turn.id, role="user", content="dock it")
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    # -- Conversations -------------------------------------------------

    async def get_active_conversation(
        self, session_id: str
    ) -> Optional[Conversation]:
        """Return the newest conversation for a session, if any."""
        result = await self._session.execute(
            select(Conversation)
            .where(Conversation.session_id == session_id)
            .order_by(desc(Conversation.created_at))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_or_create_conversation(
        self, session_id: str, model_id: Optional[str] = None
    ) -> Conversation:
        """Find the session's active conversation, or start one.

        A model switch starts a new conversation against the same session so
        that artifacts and findings -- keyed on ``session_id`` and
        model-independent -- carry across, while the transcript, whose chat
        template and tokenization are not, begins fresh.

        Args:
            session_id: Client session identifier.
            model_id: Model this turn will be held with.

        Returns:
            The conversation to append to.
        """
        existing = await self.get_active_conversation(session_id)
        if existing is not None and (
            model_id is None or existing.model_id == model_id
        ):
            return existing

        if existing is not None:
            logger.info(
                f"Model switched for session {session_id}: "
                f"{existing.model_id} -> {model_id}. Starting a new transcript; "
                f"artifacts and findings carry across."
            )

        conversation = Conversation(session_id=session_id, model_id=model_id)
        self._session.add(conversation)
        await self._session.flush()
        return conversation

    # -- Turns and messages --------------------------------------------

    async def start_turn(self, conversation_id: uuid.UUID) -> Turn:
        """Open the next turn in a conversation."""
        result = await self._session.execute(
            select(func.coalesce(func.max(Turn.seq), 0)).where(
                Turn.conversation_id == conversation_id
            )
        )
        turn = Turn(
            conversation_id=conversation_id,
            seq=(result.scalar() or 0) + 1,
        )
        self._session.add(turn)
        await self._session.flush()
        return turn

    async def append_message(
        self,
        turn_id: uuid.UUID,
        *,
        role: str,
        content: str = "",
        reasoning: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> Message:
        """Append a message to a turn, preserving everything the model produced."""
        result = await self._session.execute(
            select(func.count(Message.id)).where(Message.turn_id == turn_id)
        )
        message = Message(
            turn_id=turn_id,
            seq=result.scalar() or 0,
            role=role,
            content=content or "",
            reasoning=reasoning,
            tool_calls=tool_calls,
            tool_call_id=tool_call_id,
            name=name,
            extra=extra,
        )
        self._session.add(message)
        await self._session.flush()
        return message

    async def update_tool_message(
        self, turn_id: uuid.UUID, tool_call_id: str, content: str
    ) -> bool:
        """Replace the content of an already-recorded tool message.

        Structure queries are recorded while still holding a
        ``[PENDING:<query_id>]`` marker and are only resolved once the frontend
        posts the selection back.

        Returns:
            Whether a matching message was found.
        """
        result = await self._session.execute(
            select(Message).where(
                Message.turn_id == turn_id,
                Message.tool_call_id == tool_call_id,
            )
        )
        message = result.scalar_one_or_none()
        if message is None:
            return False
        message.content = content
        await self._session.flush()
        return True

    async def get_transcript(self, session_id: str) -> list[dict[str, Any]]:
        """Return the active conversation's messages in order.

        Each entry carries its ``turn_seq`` so the caller can reason about
        turn boundaries -- which is what compaction and the degradation ladder
        operate on.
        """
        conversation = await self.get_active_conversation(session_id)
        if conversation is None:
            return []

        result = await self._session.execute(
            select(Message, Turn.seq)
            .join(Turn, Message.turn_id == Turn.id)
            .where(Turn.conversation_id == conversation.id)
            .order_by(Turn.seq, Message.seq)
        )
        return [
            {**message.to_dict(), "turn_seq": turn_seq}
            for message, turn_seq in result.all()
        ]

    # -- Findings ------------------------------------------------------

    async def record_finding(
        self,
        session_id: str,
        *,
        tool_name: str,
        summary: str,
        conversation_id: Optional[uuid.UUID] = None,
        turn_seq: int = 1,
        iteration: int = 0,
        arguments: Optional[dict[str, Any]] = None,
        input_refs: Optional[list[str]] = None,
        output_refs: Optional[list[str]] = None,
        scalars: Optional[dict[str, Any]] = None,
        success: bool = True,
    ) -> Finding:
        """Record one tool execution in the ledger."""
        finding = Finding(
            session_id=session_id,
            conversation_id=conversation_id,
            turn_seq=turn_seq,
            iteration=iteration,
            tool_name=tool_name,
            arguments=arguments or {},
            input_refs=input_refs or [],
            output_refs=output_refs or [],
            scalars=scalars or {},
            summary=summary,
            success=success,
        )
        self._session.add(finding)
        await self._session.flush()
        return finding

    async def get_findings(
        self, session_id: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        """Return a session's findings oldest-first.

        Findings are keyed on ``session_id`` rather than conversation so they
        survive a model switch: the chemistry does not change when the model
        does.
        """
        result = await self._session.execute(
            select(Finding)
            .where(Finding.session_id == session_id)
            .order_by(Finding.created_at, Finding.turn_seq, Finding.iteration)
            .limit(limit)
        )
        return [finding.to_dict() for finding in result.scalars().all()]

    async def search_findings(
        self,
        session_id: str,
        *,
        query: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find a session's earlier results by text, by tool, or both.

        Ranked by trigram similarity against the summary when PostgreSQL's
        ``pg_trgm`` extension is available, so "affinty" still finds the
        docking runs. Without the extension -- SQLite in tests, or a database
        where the migration could not create it -- this degrades to a
        substring match, which is narrower but never wrong.

        Structured, not embedded, on purpose: a session holds tens to hundreds
        of findings, the fields are typed, and a SQL predicate is auditable in
        Phoenix in a way a vector neighbourhood is not.

        Args:
            session_id: Client session identifier.
            query: Free text matched against the summary and tool name.
            tool_name: Restrict to one tool.
            limit: Maximum rows to return.

        Returns:
            Matching findings, best match first.
        """
        statement = select(Finding).where(Finding.session_id == session_id)

        if tool_name:
            statement = statement.where(Finding.tool_name == tool_name)

        if not query:
            statement = statement.order_by(
                desc(Finding.created_at), desc(Finding.turn_seq)
            )
            result = await self._session.execute(statement.limit(limit))
            return [finding.to_dict() for finding in result.scalars().all()]

        pattern = f"%{query}%"
        matches = Finding.summary.ilike(pattern) | Finding.tool_name.ilike(pattern)

        if await self._trigram_available():
            # word_similarity, not similarity: the query is a few words and the
            # summary is a whole line, so whole-string similarity is near zero
            # for even a perfect match. word_similarity scores the query
            # against the best-matching run of words inside the summary, which
            # is the question actually being asked.
            similarity = func.word_similarity(query, Finding.summary)
            # Substring OR fuzzy, so an exact phrase matches regardless of how
            # the fuzzy threshold happens to score it.
            statement = statement.where(matches | (similarity > _TRIGRAM_THRESHOLD))
            statement = statement.order_by(desc(similarity), desc(Finding.created_at))
        else:
            statement = statement.where(matches)
            statement = statement.order_by(desc(Finding.created_at))

        result = await self._session.execute(statement.limit(limit))
        return [finding.to_dict() for finding in result.scalars().all()]

    # -- Candidates ----------------------------------------------------

    async def upsert_candidate(
        self,
        session_id: str,
        *,
        smiles: str,
        origin: Optional[str] = None,
        properties: Optional[dict[str, Any]] = None,
        scores: Optional[dict[str, Any]] = None,
        gates: Optional[list[dict[str, Any]]] = None,
        failures: Optional[list[str]] = None,
        passed: Optional[bool] = None,
        notes: Optional[str] = None,
    ) -> dict[str, Any]:
        """Record what is now known about one molecule, merging with what was.

        The merge itself lives in :func:`api.conversations.candidates.apply_update`
        so that this and the in-memory ledger training uses cannot drift. See it
        for what an omitted field means and why gates replace rather than merge.

        Args:
            session_id: Session the candidate belongs to.
            smiles: The molecule. Canonicalize before calling so that the same
                molecule written two ways lands on one row.
            origin: Provenance. Only set on first write.
            properties: Descriptors to merge in.
            scores: Rankable values to merge in.
            gates: Per-gate judgement, replacing any previous one.
            failures: Constraint violations, replacing any previous list.
            passed: Overall verdict. None leaves the existing verdict alone.
            notes: Free text, replacing any previous note.

        Returns:
            The candidate as stored, after the merge.
        """
        from api.conversations.candidates import apply_update

        result = await self._session.execute(
            select(Candidate).where(
                Candidate.session_id == session_id, Candidate.smiles == smiles
            )
        )
        candidate = result.scalar_one_or_none()

        merged = apply_update(
            candidate.to_dict() if candidate is not None else None,
            {
                "origin": origin,
                "properties": properties,
                "scores": scores,
                "gates": gates,
                "failures": failures,
                "passed": passed,
                "notes": notes,
            },
        )

        if candidate is None:
            candidate = Candidate(session_id=session_id, smiles=smiles, **merged)
            self._session.add(candidate)
        else:
            # Assigned rather than mutated in place: SQLAlchemy does not track
            # in-place changes to a JSONB value, so a mutation would be lost.
            for field, value in merged.items():
                setattr(candidate, field, value)

        await self._session.flush()
        return candidate.to_dict()

    async def get_candidates(
        self, session_id: str, limit: int = 500
    ) -> list[dict[str, Any]]:
        """Every candidate in the session, first seen first.

        Args:
            session_id: Session to read.
            limit: Maximum rows. Generous because the rows are small and the
                point of the ledger is not losing any of them.
        """
        result = await self._session.execute(
            select(Candidate)
            .where(Candidate.session_id == session_id)
            .order_by(Candidate.created_at)
            .limit(limit)
        )
        return [candidate.to_dict() for candidate in result.scalars().all()]

    async def _trigram_available(self) -> bool:
        """Whether this database can do fuzzy ranking.

        Checked rather than attempted. A failed statement aborts the enclosing
        transaction in PostgreSQL, so "try the fancy query and fall back on
        error" would take the caller's whole unit of work down with it.
        """
        bind = self._session.get_bind()
        dialect = getattr(getattr(bind, "dialect", None), "name", "")
        if dialect != "postgresql":
            return False

        key = str(getattr(bind, "url", dialect))
        if key in _TRIGRAM_SUPPORT:
            return _TRIGRAM_SUPPORT[key]

        try:
            result = await self._session.execute(
                text("SELECT 1 FROM pg_extension WHERE extname = 'pg_trgm'")
            )
            available = result.scalar() is not None
        except Exception:
            logger.debug("pg_trgm probe failed; using substring search", exc_info=True)
            available = False

        if not available:
            logger.info(
                "pg_trgm is not installed; session-history search will match "
                "substrings only. Fuzzy ranking needs the extension."
            )
        _TRIGRAM_SUPPORT[key] = available
        return available
