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

"""Write a turn to the durable record as it unfolds.

Recording incrementally rather than dumping the final graph state at the end
is deliberate. A user who interrupts a long tool chain, or a request that dies
mid-stream, still leaves behind everything that had happened up to that point.
Under end-of-run persistence those turns would vanish from the record entirely,
and the trajectory they contain -- which docks ran, what they scored -- is
exactly what the next turn needs.

Each write takes its own short-lived session and commits. The overhead is
irrelevant next to a tool call that takes seconds, and it means a crash cannot
roll back a fold that genuinely happened.
"""

from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar
from typing import Any, Optional

from api.conversations.store import ConversationStore

logger = logging.getLogger(__name__)

# Set for the duration of a request so the tool registry, which is several
# layers below the streaming loop, can file findings without every caller in
# between having to pass a recorder through. Mirrors how storage and session
# id already reach the tools.
_recorder_ctx: ContextVar[Optional["TurnRecorder"]] = ContextVar(
    "turn_recorder", default=None
)


def get_recorder() -> Optional["TurnRecorder"]:
    """Return the recorder for the in-flight request, if there is one."""
    return _recorder_ctx.get()


def set_recorder(recorder: Optional["TurnRecorder"]) -> Any:
    """Bind a recorder to this request. Returns a token for ``reset_recorder``."""
    return _recorder_ctx.set(recorder)


def reset_recorder(token: Any) -> None:
    """Unbind the request's recorder."""
    _recorder_ctx.reset(token)


# Count of ledger writes that failed and were swallowed, by operation.
#
# Swallowing is the right behaviour (see TurnRecorder's docstring) but a log
# line is not a signal anyone watches: a real bug -- every second write of a
# molecule raising MissingGreenlet, silently dropping whole ADMET panels out of
# the candidate ledger -- survived a full 20-task run because WARNING was the
# only trace it left. A counter makes "the ledger is quietly failing" something
# a health check or a dashboard can see.
#
# Process-local and unsynchronised: mutated only from the event loop, and an
# exact count matters far less than the difference between zero and non-zero.
_swallowed: dict[str, int] = {}


def _note_swallowed(operation: str) -> None:
    """Record that a ledger write failed and was suppressed."""
    _swallowed[operation] = _swallowed.get(operation, 0) + 1


def swallowed_writes() -> dict[str, int]:
    """Ledger writes that failed silently, by operation. Empty is healthy."""
    return dict(_swallowed)


def reset_swallowed_writes() -> None:
    """Clear the counters. For tests."""
    _swallowed.clear()


class TurnRecorder:
    """Appends one turn's messages and findings to the durable record.

    Every method swallows its own failures. The record is valuable, but not so
    valuable that failing to write it should take down a request that has
    already done real work -- a dock that ran is worth more than the row
    describing it.

    Example:
        recorder = await TurnRecorder.open(factory, session_id, "nemotron")
        await recorder.record_user("dock aspirin into 1ABC")
        await recorder.record_assistant(content="", tool_calls=[...])
    """

    def __init__(
        self,
        session_factory: Any,
        session_id: str,
        conversation_id: uuid.UUID,
        turn_id: uuid.UUID,
        turn_seq: int,
    ):
        self._session_factory = session_factory
        self.session_id = session_id
        self.conversation_id = conversation_id
        self.turn_id = turn_id
        self.turn_seq = turn_seq
        self.iteration = 0

    @classmethod
    async def open(
        cls,
        session_factory: Any,
        session_id: str,
        model_id: Optional[str] = None,
    ) -> "TurnRecorder":
        """Start a new turn, creating the conversation if this is the first one."""
        async with session_factory() as session:
            store = ConversationStore(session)
            conversation = await store.get_or_create_conversation(session_id, model_id)
            turn = await store.start_turn(conversation.id)
            await session.commit()
            return cls(
                session_factory,
                session_id,
                conversation.id,
                turn.id,
                turn.seq,
            )

    async def _append(self, **kwargs: Any) -> None:
        try:
            async with self._session_factory() as session:
                store = ConversationStore(session)
                await store.append_message(self.turn_id, **kwargs)
                await session.commit()
        except Exception:
            _note_swallowed("append_message")
            logger.warning("Failed to record message in conversation", exc_info=True)

    async def record_user(self, content: str) -> None:
        """Record the message that opened this turn."""
        await self._append(role="user", content=content)

    async def record_assistant(
        self,
        content: str = "",
        reasoning: Optional[str] = None,
        tool_calls: Optional[list[dict[str, Any]]] = None,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record an assistant message, chain of thought and tool calls included.

        Reasoning is stored because models in the Nemotron family are shown
        their own chain of thought on the next iteration; dropping it here
        would put production out of parity with training, where the Responses
        API loop carries reasoning items forward already.
        """
        await self._append(
            role="assistant",
            content=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            extra=extra,
        )

    async def record_tool(
        self,
        content: str,
        tool_call_id: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        """Record a tool result."""
        await self._append(
            role="tool", content=content, tool_call_id=tool_call_id, name=name
        )

    async def update_tool(self, tool_call_id: str, content: str) -> None:
        """Replace a tool result that was a placeholder when first recorded.

        Structure queries return ``[PENDING:<query_id>]`` and are patched once
        the frontend posts the selection back, so the record has to be patched
        the same way or it keeps the placeholder forever.
        """
        try:
            async with self._session_factory() as session:
                store = ConversationStore(session)
                await store.update_tool_message(self.turn_id, tool_call_id, content)
                await session.commit()
        except Exception:
            _note_swallowed("update_tool_message")
            logger.warning("Failed to update recorded tool message", exc_info=True)

    async def get_findings(self) -> list[dict[str, Any]]:
        """Read back this session's ledger, oldest first."""
        async with self._session_factory() as session:
            return await ConversationStore(session).get_findings(self.session_id)

    async def search_findings(
        self,
        query: Optional[str] = None,
        tool_name: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search this session's ledger, best match first."""
        async with self._session_factory() as session:
            return await ConversationStore(session).search_findings(
                self.session_id, query=query, tool_name=tool_name, limit=limit
            )

    async def record_finding(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: Any,
    ) -> None:
        """Extract and file the facts a tool call produced.

        Findings key on ``session_id`` rather than conversation, so a model
        switch keeps them: the chemistry does not change when the model does.
        """
        from api.conversations.findings import extract_finding

        try:
            finding = extract_finding(tool_name, arguments, result)
            async with self._session_factory() as session:
                store = ConversationStore(session)
                await store.record_finding(
                    self.session_id,
                    tool_name=tool_name,
                    conversation_id=self.conversation_id,
                    turn_seq=self.turn_seq,
                    iteration=self.iteration,
                    **finding,
                )
                await session.commit()
        except Exception:
            _note_swallowed("record_finding")
            logger.warning(f"Failed to record finding for {tool_name}", exc_info=True)

    async def get_candidates(self) -> list[dict[str, Any]]:
        """Read back this session's candidate ledger, first seen first."""
        async with self._session_factory() as session:
            return await ConversationStore(session).get_candidates(self.session_id)

    async def upsert_candidate(self, **fields: Any) -> Optional[dict[str, Any]]:
        """Record what is now known about one molecule."""
        try:
            async with self._session_factory() as session:
                store = ConversationStore(session)
                stored = await store.upsert_candidate(self.session_id, **fields)
                await session.commit()
                return stored
        except Exception:
            _note_swallowed("upsert_candidate")
            logger.warning("Failed to record candidate", exc_info=True)
            return None

    async def capture_candidates(self, tool_name: str, result: Any) -> None:
        """File whatever a tool result says about specific molecules.

        Called for every tool call, and silent for the tools that say nothing.
        Automatic rather than agent-driven on purpose: a tool already returned
        these numbers structurally, and asking the model to copy them into a
        second call would spend tokens to introduce transcription errors.
        """
        from api.conversations.candidates import canonical, capture_from_result

        try:
            rows = capture_from_result(tool_name, result)
            if not rows:
                return
            async with self._session_factory() as session:
                store = ConversationStore(session)
                for row in rows:
                    smiles = canonical(row.get("smiles", ""))
                    if not smiles:
                        continue
                    await store.upsert_candidate(
                        self.session_id, **{**row, "smiles": smiles}
                    )
                await session.commit()
        except Exception:
            _note_swallowed("capture_candidates")
            logger.warning(
                f"Failed to capture candidates from {tool_name}", exc_info=True
            )
