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

"""Stands in for the browser so viewer-dependent tool calls can complete.

``structure_query`` is the one tool that does not finish inside the backend. It
emits an SSE ``tool_event``, registers a pending op in Redis, and blocks in
``_await_pending_ops_node`` until something POSTs the selection to
``/api/selection``. In production that something is Molstar in the user's
browser (``frontend/src/App.tsx``). With no browser attached the op times out
after 10s and the tool returns an error, which measurably changes the run: in a
"residues within 5A of chain A residue 20" trace the model burned three
timeouts and then reimplemented the neighbour search by hand in
``python_execute``.

This bridge closes that gap without touching the backend. It executes the same
MolQL through ``services/molstar-headless``, which compiles expressions with the
identical ``Script(molql, 'mol-script')`` call the frontend uses, and posts the
result in the same ``SelectionData`` shape. From the backend's point of view a
browser is attached.

Deliberately mirrored from the frontend:

- The structure queried is whatever the preceding ``visualize_pdb`` or
  ``visualize_3d`` event loaded, not whatever ``pdb_ref`` names. The browser can
  only query what is in the viewer, and a query against a structure the user
  cannot see would be a different measurement. Either event replaces the
  viewer's contents, as ``loadPdb``/``loadSdf`` do.
- Atoms are capped at 10,000 (``MAX_ATOMS`` in ``App.tsx``).
- A MolQL expression that fails to compile still gets a POST, with zero atoms
  and an explanatory summary. Staying silent would leave the backend blocked for
  its full timeout and turn a bad query into an indistinguishable hang.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from .http import HttpError, post_json

logger = logging.getLogger(__name__)

# frontend/src/App.tsx: const MAX_ATOMS = 10000
MAX_ATOMS = 10000


class MolstarBridge:
    """Executes viewer queries on the backend's behalf, as the browser would.

    Attributes:
        loaded_pdb: The structure the viewer currently shows. ``None`` until a
            ``visualize_pdb`` event arrives, which is also the browser's state.
        events: One record per handled query, for the trajectory. The verifier
            reads these to tell "the model never asked" from "the model asked
            and the query matched nothing".
    """

    def __init__(
        self, api_base: str, molstar_base: str, owner_token: str | None = None
    ):
        self._api_base = api_base.rstrip("/")
        self._owner_token = owner_token
        self._molstar_base = molstar_base.rstrip("/")
        self.loaded_pdb: str | None = None
        self.loaded_label: str | None = None
        self.events: list[dict[str, Any]] = []

    def note_tool_event(self, event: dict[str, Any], session_id: str) -> None:
        """Handle one SSE ``tool_event``, mirroring the frontend's dispatch."""
        kind = event.get("type")
        if kind in ("visualize_pdb", "visualize_3d", "visualize_sdf"):
            self._load_structure(event)
        elif kind == "structure_query":
            self._run_query(event, session_id)
        elif kind == "show_selection":
            # Display only: the selection was already stored by the tool, so
            # there is nothing to post back. Recorded so a run can show that
            # the agent pointed at something.
            self.events.append(
                {
                    "kind": "viewer_show",
                    "selection_ref": event.get("selection_ref"),
                    "residue_count": len(event.get("residues") or []),
                    # Residues with only some atoms chosen are listed as atoms.
                    "atom_count": len(event.get("atoms") or []),
                }
            )

    def _load_structure(self, event: dict[str, Any]) -> None:
        # visualize_pdb carries PDB or mmCIF under "pdb"; visualize_3d carries
        # MOL/SDF under "sdf" (backend/api/graph/tools.py). App.tsx dispatches
        # on exactly those two keys, and both of its load paths clear the viewer
        # first -- so whichever arrives last is the only thing queryable, and
        # this replaces rather than accumulates.
        payload = event.get("pdb") or event.get("sdf")
        self.loaded_label = event.get("label") or self.loaded_label
        if payload:
            self.loaded_pdb = payload
        self.events.append(
            {
                "kind": "viewer_load",
                "type": event.get("type"),
                "label": event.get("label"),
                "has_pdb": bool(payload),
            }
        )

    def _run_query(self, event: dict[str, Any], session_id: str) -> None:
        query = event.get("query") or {}
        molql = query.get("molql")
        query_id = event.get("query_id")
        action = query.get("action") or "select"
        name = query.get("name") or None
        # The event's session_id is the graph's, which is authoritative: App.tsx
        # prefers it over React state for exactly this reason.
        target_session = event.get("session_id") or session_id

        if not query_id:
            # Nothing to acknowledge; the backend did not register a pending op.
            self.events.append({"kind": "structure_query", "skipped": "no_query_id"})
            return

        if not molql:
            # The frontend's guard is `event.type === "structure_query" &&
            # event.query?.molql`, so with no expression it posts nothing and the
            # backend's pending op times out. Posting here would be kinder but
            # would hide a backend bug that production users would feel.
            self.events.append(
                {"kind": "structure_query", "query_id": query_id, "error": "no_molql"}
            )
            return

        if not self.loaded_pdb:
            self._post_selection(
                target_session,
                query_id,
                molql=molql,
                action=action,
                atoms=[],
                atom_count=0,
                residue_count=0,
                summary="Selected 0 atoms",
                name=name,
            )
            self.events.append(
                {
                    "kind": "structure_query",
                    "query_id": query_id,
                    "molql": molql,
                    "error": "no_structure_loaded",
                }
            )
            return

        try:
            result = post_json(
                f"{self._molstar_base}/query",
                {"pdb_data": self.loaded_pdb, "molql": molql},
                timeout=60.0,
            )
        except HttpError as e:
            # molstar-headless answers 422 with a populated error body for an
            # expression it cannot parse or compile -- the same class of
            # failure the frontend reports with a "MolQL error:" summary.
            result = _error_result(e.body, molql)
        except Exception as e:
            result = _error_result(str(e), molql)

        atoms = result.get("atoms") or []
        truncated = atoms[:MAX_ATOMS]
        if len(atoms) > MAX_ATOMS:
            logger.warning(
                "Selection truncated: %d atoms -> %d", len(atoms), MAX_ATOMS
            )

        self._post_selection(
            target_session,
            query_id,
            molql=molql,
            action=action,
            atoms=truncated,
            # atom_count reports the true size even when the payload is capped,
            # matching the frontend's use of queryResult.atomCount.
            atom_count=int(result.get("atom_count") or len(atoms)),
            residue_count=int(result.get("residue_count") or 0),
            summary=str(result.get("summary") or f"Selected {len(atoms)} atoms"),
            name=name,
        )
        self.events.append(
            {
                "kind": "structure_query",
                "query_id": query_id,
                "molql": molql,
                "action": action,
                "atom_count": int(result.get("atom_count") or len(atoms)),
                "residue_count": int(result.get("residue_count") or 0),
                "summary": result.get("summary"),
                "error": result.get("error"),
            }
        )

    def _post_selection(
        self,
        session_id: str,
        query_id: str,
        *,
        molql: str,
        action: str,
        atoms: list[dict],
        atom_count: int,
        residue_count: int,
        summary: str,
        name: Optional[str] = None,
    ) -> None:
        """POST to /api/selection, which acknowledges the backend's pending op."""
        url = f"{self._api_base}/api/selection?session_id={session_id}"
        payload = {
            "query_id": query_id,
            "molql": molql,
            "action": action,
            "atoms": [_atom_payload(atom) for atom in atoms],
            "atom_count": atom_count,
            "residue_count": residue_count,
            "summary": summary,
        }
        if name:
            payload["name"] = name
        try:
            post_json(url, payload, timeout=120.0, token=self._owner_token)
        except Exception as e:
            # Swallowing this is right: the backend's pending op will time out
            # on its own, which is the same outcome as a browser that failed to
            # post. Raising would abort a turn that is otherwise recoverable.
            logger.error("Failed to post selection for %s: %s", query_id, e)
            self.events.append(
                {"kind": "selection_post_failed", "query_id": query_id, "error": str(e)}
            )


def _atom_payload(atom: dict) -> dict:
    """Project an atom onto the backend's AtomData model.

    molstar-headless already emits this shape, but building it explicitly means
    an extra field on either side cannot cause a 422 that looks like a viewer
    failure.
    """
    residue = atom.get("residue") or {}
    residue_payload = {
        "name": residue.get("name", ""),
        "seqNumber": int(residue.get("seqNumber") or 0),
        "chainId": residue.get("chainId", ""),
        "insCode": residue.get("insCode") or "",
        "entityType": residue.get("entityType", ""),
    }
    # Author numbering rides beside the label pair. Forwarded only when the
    # viewer sent it, so an older molstar-headless image still posts cleanly
    # -- and a number is never invented where none was reported.
    if residue.get("authSeqNumber") is not None:
        residue_payload["authSeqNumber"] = int(residue["authSeqNumber"])
    if residue.get("authChainId") is not None:
        residue_payload["authChainId"] = str(residue["authChainId"])
    return {
        "name": atom.get("name", ""),
        "element": atom.get("element", ""),
        "id": int(atom.get("id") or 0),
        "coords": [float(value) for value in (atom.get("coords") or [0.0, 0.0, 0.0])],
        "residue": residue_payload,
    }


# Marks a selection whose expression never ran. Must match MOLQL_ERROR_PREFIX in
# backend/api/selections/format.py and App.tsx, which is how the backend tells a
# syntax error the model can fix from a query that legitimately matched nothing.
_MOLQL_ERROR_PREFIX = "MolQL error:"
_HEADLESS_ERROR_PREFIX = "MolQL query failed:"


def _error_message(body: str) -> str:
    """The engine's message from a molstar-headless error body.

    The service answers `{"error": "MolQL query failed: <reason>", ...}`; fall
    back to the raw text for anything else (a connection error, a proxy page).
    """
    try:
        parsed = json.loads(body)
        message = parsed.get("error") if isinstance(parsed, dict) else None
    except ValueError:
        message = None
    message = str(message or body).strip()
    if message.startswith(_HEADLESS_ERROR_PREFIX):
        message = message[len(_HEADLESS_ERROR_PREFIX):].strip()
    return message[:500]


def _error_result(detail: str, molql: str) -> dict:
    """A failed compile, reported the way the frontend reports it.

    The summary reaches the model as tool output, so it carries the engine's
    message under the same prefix App.tsx uses; without it the model sees only
    "0 atoms" and cannot tell a syntax error from an empty result.
    """
    message = _error_message(detail)
    return {
        "atoms": [],
        "atom_count": 0,
        "residue_count": 0,
        "summary": f"{_MOLQL_ERROR_PREFIX} {message}",
        "error": f"{molql}: {message}",
    }
