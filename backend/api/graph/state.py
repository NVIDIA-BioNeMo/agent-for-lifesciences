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

"""LangGraph state definition for Aevom.

Defines the typed state that flows through the agent graph. State refresh
from Redis/PostgreSQL is handled by a dedicated graph node, not inline.
"""

import logging
from operator import add as add_op
from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class AevomState(TypedDict):
    """Typed state for the Aevom agent graph.

    Attributes:
        messages: Conversation history managed by LangGraph's add_messages reducer.
        viewer_state: Current Molstar viewer state (structures loaded, selections).
        storage_state: Available data artifacts in the storage layer.
        selections: Active molecular selections for analysis.
        request_id: Per-request ID used as Redis key for viewer/storage state.
        session_id: Client session identifier for artifact scoping.
        findings: Ledger rows for the [SESSION LOG] block. The only part of the
            envelope that is memory rather than a snapshot of current state.
        tool_budgets: Remaining per-tool call budgets, empty when none are
            configured.
        candidates: Candidate ledger rows for the [CANDIDATES] block. Memory,
            like findings, rather than a snapshot of current state.
        exclusions: Rules derived from the failing candidates, for the
            [EXCLUSIONS] block. Derived rather than stored, so it is always
            consistent with the ledger it came from.
        nim_notices: NIMs registered for the session but not yet confirmed,
            whose tools are therefore withheld; for the [NIM STATE] line.

    Services (storage, DB session factory) deliberately do not live here.
    They are not state -- they never change during a run and cannot be
    serialised -- so they travel in ``config["configurable"]`` instead. That
    keeps every field of this TypedDict something a checkpointer could
    actually persist, should we want resumption later.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    viewer_state: dict
    storage_state: dict
    selections: list[dict]
    findings: list[dict]
    tool_budgets: list[dict]
    candidates: list[dict]
    exclusions: list[dict]
    nim_notices: list[dict]
    request_id: str
    session_id: str
    # Token usage accumulators (summed across iterations via add_op reducer)
    total_episode_input_tokens: Annotated[int, add_op]
    total_episode_output_tokens: Annotated[int, add_op]
    # Iteration counter — incremented by agent_node, checked by _should_continue
    iteration_count: Annotated[int, add_op]


async def refresh_state_from_backends(
    state: AevomState, config: Optional[dict] = None
) -> dict:
    """Refresh viewer_state, storage_state, and selections from Redis/DB.

    Called as a graph node after every tool-execution cycle so the next
    LLM call sees up-to-date state.

    Args:
        state: Current graph state.
        config: LangGraph runnable config. ``configurable["storage"]`` carries
            the request's DataStorageService.

    Returns a partial state update dict (only the keys that changed).
    """
    from api.tools.viewer_state import get_current_viewer_state
    from api.tools.storage_state import (
        refresh_storage_state,
        get_current_storage_state,
        StorageState,
    )
    from api.selections import get_available_selections
    from api.redis import get_redis

    redis_client = get_redis()
    request_id: str = state["request_id"]
    session_id: str = state["session_id"]
    storage = (config or {}).get("configurable", {}).get("storage")

    # Commit the DB session so artifacts stored during this iteration are
    # visible to other DB sessions (sandbox python_execute, frontend polling).
    if storage and hasattr(storage, "commit"):
        await storage.commit()

    # Refresh storage state
    if storage:
        await refresh_storage_state(redis_client, storage, session_id)
        storage_state_obj: Optional[StorageState] = await get_current_storage_state(
            redis_client, session_id
        )
    else:
        storage_state_obj = StorageState(is_empty=True)

    storage_dict: dict = (
        storage_state_obj.to_dict()
        if storage_state_obj and hasattr(storage_state_obj, "to_dict")
        else {"is_empty": True, "artifacts": []}
    )

    # Refresh viewer state
    viewer_obj = await get_current_viewer_state(request_id)
    if viewer_obj and hasattr(viewer_obj, "to_dict"):
        viewer_dict = viewer_obj.to_dict()
    elif viewer_obj:
        viewer_dict = {
            "is_empty": viewer_obj.is_empty,
            "structures": [
                {
                    "label": s.label,
                    "structure_type": s.structure_type,
                    "format": s.format,
                    "chain_ids": s.chain_ids,
                    "residue_count": s.residue_count,
                    "atom_count": s.atom_count,
                    "has_polymer": s.has_polymer,
                    "has_ligand": s.has_ligand,
                    "has_water": s.has_water,
                }
                for s in viewer_obj.structures
            ],
            "has_selection": viewer_obj.has_selection,
            "selection_info": viewer_obj.selection_info,
        }
    else:
        viewer_dict = {"is_empty": True, "structures": []}

    # Refresh selections
    sel_objs = await get_available_selections(redis_client, session_id)
    selections_list = [
        s.to_dict() if hasattr(s, "to_dict") else {
            "reference": s.reference,
            "molql": s.molql,
            "action": s.action,
            "atom_count": s.atom_count,
            "residue_count": s.residue_count,
            "summary": s.summary,
        }
        for s in sel_objs
    ]

    update: dict = {
        "viewer_state": viewer_dict,
        "storage_state": storage_dict,
        "selections": selections_list,
        "tool_budgets": await _load_tool_budgets(session_id),
        "nim_notices": await _load_nim_notices(session_id),
    }

    # Re-read the ledger rather than accumulating it in state, because tools
    # file findings from places whose state never reaches this graph -- the
    # workflow subgraphs most of all.
    findings = await _load_findings(session_id, config)
    if findings is not None:
        update["findings"] = findings

    candidates = await _load_candidates(session_id, config)
    if candidates is not None:
        update["candidates"] = candidates
        update["exclusions"] = derive_exclusions(candidates)
    return update


def derive_exclusions(candidates: list[dict]) -> list[dict]:
    """Render the rules implied by the failing candidates for the prompt block.

    Derived here from candidates already in hand rather than loaded separately:
    an exclusion is a statement about the ledger, so reading it from anywhere
    else would let the two disagree.
    """
    from api.conversations.exclusions import derive

    try:
        return [
            {**rule.to_dict(), "description": rule.describe()}
            for rule in derive(candidates)
        ]
    except Exception:
        logger.warning("Could not derive exclusions", exc_info=True)
        return []


async def _load_nim_notices(session_id: str) -> list[dict]:
    """NIMs the user registered but has not confirmed, for the prompt.

    Only NIMs with an adapter are listed: one without an adapter would add no
    tools even once confirmed, so telling the model to ask for it would send
    the user to a button that does nothing. Unreadable state is "no notice"
    rather than a failure of the turn.
    """
    from api.nim import get_nim_session_store

    try:
        servers = await get_nim_session_store().get_servers(session_id)
        notices = []
        for server in servers:
            if server.confirmed:
                continue
            offered = server.to_response()
            if not offered["adapter"]:
                continue
            notices.append({"name": server.name, "tools": offered["tools"]})
    except Exception:
        logger.warning("Could not read NIM registrations", exc_info=True)
        return []
    return notices


async def _load_tool_budgets(session_id: str) -> list[dict]:
    """Read remaining call budgets for the prompt block.

    Empty when no budget is configured, which is the default, so a deployment
    without budgets carries no block at all.
    """
    from api.tools.budget import get_budget_states
    from api.tools.registry import get_registry

    try:
        states = await get_budget_states(session_id)
    except Exception:
        logger.warning("Could not read tool budgets", exc_info=True)
        return []

    # Whether a call covers one input or many changes what "three calls left"
    # means, so the block needs it alongside the count.
    try:
        registry = get_registry()
    except Exception:
        registry = None
    return [
        {
            "tool_name": state.tool_name,
            "limit": state.limit,
            "used": state.used,
            "batch_argument": _batch_argument(registry, state.tool_name),
        }
        for state in states
    ]


def _batch_argument(registry, tool_name: str) -> Optional[str]:
    """Whether one call of this tool covers many inputs. None when unknown."""
    tool = registry.get(tool_name) if registry else None
    return getattr(tool, "batch_argument", None) if tool else None


async def _load_findings(
    session_id: str, config: Optional[dict]
) -> Optional[list[dict]]:
    """Read the session's findings ledger.

    Returns:
        The ledger rows, or None when they could not be read. None means
        "unknown", and callers leave existing state alone -- returning an empty
        list would silently erase a session log the model had been relying on.
    """
    from api.conversations import ConversationStore, get_recorder

    session_factory = (config or {}).get("configurable", {}).get("db_session_factory")

    try:
        if session_factory is not None:
            async with session_factory() as db_session:
                return await ConversationStore(db_session).get_findings(session_id)
        # Workflow subgraphs run without a factory in their config; the
        # request's recorder carries one.
        recorder = get_recorder()
        if recorder is not None:
            return await recorder.get_findings()
    except Exception:
        logger.warning("Could not load findings for the session log", exc_info=True)
    return None


async def _load_candidates(
    session_id: str, config: Optional[dict]
) -> Optional[list[dict]]:
    """Read the session's candidate ledger.

    Returns:
        The ledger rows, or None when they could not be read. None means
        "unknown" and leaves existing state alone, for the same reason as
        findings: replacing a populated block with an empty one would tell the
        agent it has found nothing.
    """
    from api.conversations import ConversationStore, get_recorder
    from api.conversations.candidates import get_candidates_source

    session_factory = (config or {}).get("configurable", {}).get("db_session_factory")

    try:
        if session_factory is not None:
            async with session_factory() as db_session:
                return await ConversationStore(db_session).get_candidates(session_id)
        recorder = get_recorder()
        if recorder is not None:
            return await recorder.get_candidates()
        source = get_candidates_source()
        if source is not None:
            return list(source())
    except Exception:
        logger.warning("Could not load the candidate ledger", exc_info=True)
    return None
