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

"""Wire models for a selection posted by the frontend.

These live here, not in ``api.main``, so the code that stores a selection can
be exercised without importing the whole FastAPI app (which pulls in Redis and
the database).
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .roles import get_role


class ResidueData(BaseModel):
    """Residue information from a selection."""

    name: str  # 3-letter residue code (e.g., ALA, HEM)
    # seqNumber / chainId are LABEL numbering (label_seq_id / label_asym_id) as
    # emitted by the viewer. They are what AtomWorks/RFD4 resolve ``res_id``
    # against. Legacy payloads and hand-written fixtures may carry author
    # numbering here instead, which is why the auth* fields below exist.
    seqNumber: int  # Sequence number
    chainId: str  # Chain ID (e.g., A, B)
    insCode: str = ""  # Insertion code
    entityType: str  # polymer, non-polymer, water
    # Author numbering (auth_seq_id / auth_asym_id): what the user sees in the
    # viewer and what MolQL ``atom.resno`` / ``atom.chain`` match. Optional so
    # payloads written before the two systems were separated still validate.
    authSeqNumber: int | None = None
    authChainId: str | None = None


class AtomData(BaseModel):
    """Atom information from a selection, including 3D coordinates."""

    name: str  # Atom name (e.g., CA, CB, N)
    element: str  # Element symbol (e.g., C, N, O)
    id: int  # Atom ID
    coords: list[float]  # [x, y, z] coordinates in Angstroms
    residue: ResidueData


class SelectionData(BaseModel):
    """A selection posted from the frontend.

    Two producers share this shape. After a ``structure_query`` the viewer
    posts what the agent's MolQL matched (``provenance="agent"``, with the
    ``query_id`` that unblocks the waiting tool call). When the user picks
    residues by hand it posts them with ``provenance="user"`` and no
    ``query_id`` -- there is no tool call to correlate with and nothing to
    acknowledge. Everything after ``summary`` is optional so payloads from
    before either existed still validate.
    """

    query_id: str | None = None  # Correlates with the tool call ID; absent for user picks
    molql: str = ""  # The MolQL query that was executed; empty for user picks
    action: str = "select"  # The action performed (highlight, focus, select, info)
    atoms: list[AtomData]  # Full atom info with coordinates
    atom_count: int
    residue_count: int
    summary: str = ""  # Human-readable summary
    provenance: Literal["user", "agent"] = "agent"
    name: str | None = Field(default=None, max_length=80)  # Short label
    parent: str | None = None  # selection_ ref this was edited or derived from
    scope: Literal["atom", "residue"] = "atom"  # How it was made
    # The part this selection plays in a design (a key from ``roles``), and the
    # role's parameter: a Float group, a Burial level, or a Keep layout. Absent
    # for a plain selection. Posting a role with no atoms *removes* the role.
    role: str | None = None
    value: Any = None

    @field_validator("role")
    @classmethod
    def _known_role(cls, role: str | None) -> str | None:
        if role in (None, ""):
            return None
        resolved = get_role(role)
        if resolved is None:
            raise ValueError(f"Unknown role '{role}'.")
        return resolved.key
