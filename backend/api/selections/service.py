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

"""Store a posted selection and register it for the session.

The body of ``POST /api/selection``, kept out of ``api.main`` so it can be
tested with a fake storage and a fake Redis.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable, Optional

from .models import SelectionData
from .registry import register_selection
from .residues import format_residue_listing, residues_from_atoms

logger = logging.getLogger(__name__)

# Storage-strip descriptions are one line; the listing is the interesting part.
MAX_DESCRIPTION_LEN = 240


def selection_description(data: SelectionData, residue_listing: str) -> str:
    """Storage-strip description: who made it, what it is called, what it holds."""
    who = "User selection" if data.provenance == "user" else "Agent selection"
    label = f" '{data.name}'" if data.name else ""
    detail = residue_listing or data.summary
    return f"{who}{label}: {detail}"[:MAX_DESCRIPTION_LEN]


async def store_selection_data(
    data: SelectionData,
    *,
    storage: Any,
    redis_client: Optional[Any],
    session_id: str,
    acknowledge: Optional[Callable[[str], Awaitable[bool]]] = None,
    commit: Optional[Callable[[], Awaitable[Any]]] = None,
) -> dict[str, Any]:
    """Persist ``data`` as a ``selection`` artifact and register it.

    Args:
        data: The posted selection. Counts are rewritten for a user pick.
        storage: A ``DataStorageService`` (or anything with its ``store``).
        redis_client: Client for the per-session selection registry.
        session_id: Session the artifact and registry entry are scoped to.
        acknowledge: Called with ``data.query_id`` once the selection is
            registered, to release the tool call waiting on it. Never called
            for a user's pick, which has no ``query_id``.
        commit: Awaited after the artifact is stored and before it is
            registered, so a registry entry never points at an uncommitted row.

    Returns:
        ``reference``, the counts, ``provenance`` and ``residue_listing``.
    """
    residues = residues_from_atoms(a.model_dump() for a in data.atoms)
    if data.provenance == "user":
        # A hand-made pick is complete (the viewer's atom cap is far above
        # anything clicked residue by residue), so the counts are derived from
        # the atoms rather than taken on the client's word.
        data.atom_count = len(data.atoms)
        data.residue_count = len(residues)
        if not data.summary:
            data.summary = (
                f"Selected {data.atom_count} atoms in {data.residue_count} residues"
            )
    residue_listing = format_residue_listing(residues)

    metadata: dict[str, Any] = {
        "query_id": data.query_id,
        "molql": data.molql,
        "action": data.action,
        "atom_count": data.atom_count,
        "residue_count": data.residue_count,
        "provenance": data.provenance,
        "scope": data.scope,
    }
    if data.name:
        metadata["name"] = data.name
    if data.parent:
        metadata["parent"] = data.parent
    if data.role:
        metadata["role"] = data.role
        if data.value is not None:
            metadata["role_value"] = data.value
    if session_id:
        metadata["session_id"] = session_id

    ref = await storage.store(
        data_type="selection",
        content=data.model_dump_json().encode("utf-8"),
        metadata=metadata,
        description=selection_description(data, residue_listing),
    )
    if commit is not None:
        await commit()

    # Register in the selection registry for access across conversation turns.
    # Without Redis (a deployment-less eval, a unit test) the artifact still
    # exists and can be passed by reference; it just is not listed in the
    # [SELECTION STATE] block.
    if redis_client is not None:
        await register_selection(
            redis_client,
            session_id=session_id or "default",
            reference=ref,
            molql=data.molql,
            action=data.action,
            atom_count=data.atom_count,
            residue_count=data.residue_count,
            summary=data.summary,
            query_id=data.query_id,
            provenance=data.provenance,
            name=data.name,
            parent=data.parent,
            scope=data.scope,
            residue_listing=residue_listing,
            role=data.role,
            role_value=data.value,
        )

    # Release the waiting tool call, if any. This resolves the race where
    # python_execute runs before the selection is ready.
    acknowledged = False
    if data.query_id and acknowledge is not None:
        acknowledged = await acknowledge(data.query_id)

    logger.info(
        f"Stored and registered selection: {ref} ({data.atom_count} atoms, "
        f"{data.residue_count} residues, provenance={data.provenance}, "
        f"query_id={data.query_id}, ack={acknowledged})"
    )

    return {
        "reference": ref,
        "atom_count": data.atom_count,
        "residue_count": data.residue_count,
        "provenance": data.provenance,
        "residue_listing": residue_listing,
        "role": data.role,
    }


async def store_built_selection(
    atoms: list[dict[str, Any]],
    *,
    storage: Any,
    redis_client: Any,
    session_id: str,
    provenance: str = "agent",
    scope: str = "residue",
    name: Optional[str] = None,
    parent: Optional[str] = None,
    summary: str = "",
    molql: str = "",
    commit: Optional[Callable[[], Awaitable[Any]]] = None,
) -> dict[str, Any]:
    """Store a selection built server-side from ``atoms`` (viewer ``AtomData`` shape).

    The path for selections that never touched a viewer: the agent's
    ``select_residues`` and set-algebra tools, and the sandbox's
    ``aevom.store_selection``. The counts are derived here, because there is no
    client to take them from, and no ``query_id`` is involved.
    """
    data = SelectionData(
        atoms=atoms,
        atom_count=len(atoms),
        residue_count=0,
        summary=summary,
        molql=molql,
        provenance=provenance,
        scope=scope,
        name=name,
        parent=parent,
    )
    data.residue_count = len(residues_from_atoms(a.model_dump() for a in data.atoms))
    if not data.summary:
        data.summary = f"Selected {data.atom_count} atoms in {data.residue_count} residues"
    return await store_selection_data(
        data,
        storage=storage,
        redis_client=redis_client,
        session_id=session_id,
        commit=commit,
    )
