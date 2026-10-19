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

"""Read atoms out of a stored structure, for building selections server-side.

A selection made by hand arrives from the viewer with its atoms already
resolved. One the agent asks for by residue number does not -- there is no
viewer in the loop, in training or in an eval -- so the atoms have to come
from the stored structure itself. This reads exactly what that needs from an
mmCIF or PDB file: coordinates, atom and residue names, and both numbering
systems. It is not a general parser; anything it does not understand is
skipped rather than guessed at.

The atoms it yields are shaped like the viewer's (see
:class:`~api.selections.models.AtomData`), so a selection built here and one
posted by Mol* are the same artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from ..storage.mmcif_loop import as_text, atom_site_columns, iter_atom_site_rows

_WATER = frozenset({"HOH", "WAT", "DOD", "H2O"})
_NULL = frozenset({".", "?"})


@dataclass(frozen=True)
class StructureAtom:
    """One atom of a stored structure, in both numbering systems."""

    name: str
    element: str
    id: int
    x: float
    y: float
    z: float
    label_chain: str
    label_seq: Optional[int]
    auth_chain: str
    auth_seq: int
    ins_code: str
    comp_id: str
    entity_type: str

    @property
    def residue_key(self) -> tuple[str, int, str]:
        """Identity of the residue in author numbering."""
        return (self.auth_chain, self.auth_seq, self.ins_code)

    def to_payload(self) -> dict[str, Any]:
        """The viewer's ``AtomData`` shape, so the artifact is the same either way."""
        return {
            "name": self.name,
            "element": self.element,
            "id": self.id,
            "coords": [self.x, self.y, self.z],
            "residue": {
                "name": self.comp_id,
                # label numbering in seqNumber/chainId, as the viewer emits it;
                # a non-polymer has no label_seq_id, which the viewer reports as 0.
                "seqNumber": self.label_seq if self.label_seq is not None else 0,
                "chainId": self.label_chain,
                "authSeqNumber": self.auth_seq,
                "authChainId": self.auth_chain,
                "insCode": self.ins_code,
                "entityType": self.entity_type,
            },
        }


def _clean(value: str) -> str:
    return "" if value in _NULL else value


def _entity_type(group: str, comp: str, label_seq: Optional[int]) -> str:
    if comp.upper() in _WATER:
        return "water"
    if group == "HETATM" and label_seq is None:
        return "non-polymer"
    return "polymer" if label_seq is not None else "non-polymer"


def _parse_mmcif(text: str) -> list[StructureAtom]:
    columns = atom_site_columns(text)
    if not columns:
        return []
    index = {name: i for i, name in enumerate(columns)}

    def col(row: list[str], *names: str) -> str:
        for name in names:
            i = index.get(name)
            if i is not None:
                return row[i]
        return "."

    atoms: list[StructureAtom] = []
    first_model: Optional[str] = None
    for row in iter_atom_site_rows(text):
        model = col(row, "pdbx_PDB_model_num")
        if first_model is None:
            first_model = model
        if model != first_model:
            break
        altloc = _clean(col(row, "label_alt_id"))
        if altloc and altloc not in ("A", "1"):
            continue
        try:
            x, y, z = (float(col(row, c)) for c in ("Cartn_x", "Cartn_y", "Cartn_z"))
            label_seq_raw = _clean(col(row, "label_seq_id"))
            label_seq = int(label_seq_raw) if label_seq_raw else None
            auth_seq = int(col(row, "auth_seq_id") if col(row, "auth_seq_id") not in _NULL else label_seq_raw)
            atom_id = int(col(row, "id"))
        except (TypeError, ValueError):
            continue
        label_chain = _clean(col(row, "label_asym_id"))
        auth_chain = _clean(col(row, "auth_asym_id")) or label_chain
        comp = _clean(col(row, "label_comp_id", "auth_comp_id"))
        atoms.append(
            StructureAtom(
                name=_clean(col(row, "label_atom_id", "auth_atom_id")),
                element=_clean(col(row, "type_symbol")),
                id=atom_id,
                x=x, y=y, z=z,
                label_chain=label_chain,
                label_seq=label_seq,
                auth_chain=auth_chain,
                auth_seq=auth_seq,
                ins_code=_clean(col(row, "pdbx_PDB_ins_code")),
                comp_id=comp,
                entity_type=_entity_type(col(row, "group_PDB"), comp, label_seq),
            )
        )
    return atoms


def _parse_pdb(text: str) -> list[StructureAtom]:
    """Fixed-column PDB. Label numbering is the author numbering, as Mol* does."""
    atoms: list[StructureAtom] = []
    for line in text.splitlines():
        record = line[:6].strip()
        if record == "ENDMDL":
            break
        if record not in ("ATOM", "HETATM") or len(line) < 54:
            continue
        altloc = line[16].strip()
        if altloc and altloc not in ("A", "1"):
            continue
        try:
            auth_seq = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            atom_id = int(line[6:11])
        except ValueError:
            continue
        comp = line[17:20].strip()
        chain = line[21].strip() or "A"
        element = line[76:78].strip() if len(line) >= 78 else ""
        name = line[12:16].strip()
        if not element:
            element = name.lstrip("0123456789")[:1]
        polymer = record == "ATOM"
        atoms.append(
            StructureAtom(
                name=name,
                element=element,
                id=atom_id,
                x=x, y=y, z=z,
                label_chain=chain,
                label_seq=auth_seq if polymer else None,
                auth_chain=chain,
                auth_seq=auth_seq,
                ins_code=line[26].strip(),
                comp_id=comp,
                entity_type="water" if comp.upper() in _WATER else ("polymer" if polymer else "non-polymer"),
            )
        )
    return atoms


def parse_structure_atoms(content: "str | bytes") -> list[StructureAtom]:
    """Atoms of the first model of an mmCIF or PDB file, alternate A only.

    Returns an empty list when the content is neither, which callers report as
    "no atoms found" rather than as a parse failure -- the usual cause is a
    reference to something that is not a structure.
    """
    text = as_text(content)
    if not text:
        return []
    head = text[:4096]
    if "_atom_site." in text or head.lstrip().startswith("data_"):
        return _parse_mmcif(text)
    return _parse_pdb(text)
