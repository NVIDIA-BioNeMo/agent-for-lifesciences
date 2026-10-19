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

"""Residue-level view of a stored selection.

A ``selection`` artifact is atom-level: it stores every atom the viewer
reported. Almost every consumer wants residues -- the prompt block, the
hotspot builder, set algebra, the sandbox helper, the viewer's ``show``
command -- and each of them used to re-derive them from ``atoms[].residue``
with its own idea of which numbering ``seqNumber`` was. This module is the one
place that does it.

Every residue carries both numbering systems:

* **label** (``label_asym_id`` / ``label_seq_id``): what AtomWorks and RFD4
  resolve ``res_id`` against.
* **author** (``auth_asym_id`` / ``auth_seq_id``): what the viewer shows the
  user and what MolQL ``atom.resno`` / ``atom.chain`` match. This is also the
  numbering used when talking *to* the model or the user.

Selections stored before the author fields existed carry only the label pair
(see :class:`~api.main.ResidueData`). For those, ``auth_known`` is False and
the author fields repeat the label ones, so callers can still list residues
but know the numbering is not independently confirmed.

No Redis, no FastAPI: pure functions, so training and deployment share them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Optional

_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}  # fmt: skip

# Heavy atoms in each standard residue, as the PDB lists them (without the
# terminal OXT). A residue with fewer selected atoms than this is *partial*:
# somebody chose atoms, not the residue, and the listing says which.
STANDARD_HEAVY_ATOMS: dict[str, int] = {
    "ALA": 5, "ARG": 11, "ASN": 8, "ASP": 8, "CYS": 6,
    "GLN": 9, "GLU": 9, "GLY": 4, "HIS": 10, "ILE": 8,
    "LEU": 8, "LYS": 9, "MET": 8, "PHE": 11, "PRO": 7,
    "SER": 6, "THR": 7, "TRP": 14, "TYR": 12, "VAL": 7,
}  # fmt: skip

# Listing is a reminder of what was selected, not a transcript of it.
DEFAULT_MAX_LISTED = 12

# Atom names shown in one token before the rest are counted: ``Y56[OH,CZ,CE1,CE2,+2]``.
MAX_ATOMS_IN_TOKEN = 4


@dataclass(frozen=True)
class ResidueRef:
    """One residue of a selection, in both numbering systems.

    Attributes:
        label_chain: ``label_asym_id``.
        label_seq: ``label_seq_id``, or None for a non-polymer (a ligand or
            water has no label sequence number).
        auth_chain: ``auth_asym_id``.
        auth_seq: ``auth_seq_id``.
        ins_code: Insertion code, empty when there is none.
        comp_id: Three-letter residue name.
        entity_type: ``polymer``, ``non-polymer`` or ``water``.
        atoms: Names of the atoms of this residue that the selection holds, in
            the order reported, without duplicates (alternate locations repeat
            a name). Empty when the residue was built without that information.
        auth_known: False when the selection predates the author fields, in
            which case ``auth_*`` repeat the label values.
    """

    label_chain: str
    label_seq: Optional[int]
    auth_chain: str
    auth_seq: int
    ins_code: str = ""
    comp_id: str = ""
    entity_type: str = "polymer"
    auth_known: bool = True
    atoms: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[str, int, str]:
        """Identity of the residue within one structure, in author numbering."""
        return (self.auth_chain, self.auth_seq, self.ins_code)

    @property
    def is_polymer(self) -> bool:
        return self.entity_type == "polymer" and self.label_seq is not None

    @property
    def partial(self) -> bool:
        """True when the selection holds only some of this residue's atoms.

        Judged against the standard heavy-atom count, so only the twenty
        standard residues can be partial: for anything else the selection does
        not know how many atoms the residue has, and calls it whole.
        """
        expected = STANDARD_HEAVY_ATOMS.get(self.comp_id.upper())
        return bool(self.atoms) and expected is not None and len(self.atoms) < expected

    @property
    def token(self) -> str:
        """Short display form in author numbering.

        ``Y56`` for a whole residue, ``HEM301`` for a ligand, and
        ``Y56[OH,CZ]`` when only some atoms were chosen.
        """
        letter = _THREE_TO_ONE.get(self.comp_id.upper(), self.comp_id or "?")
        base = f"{letter}{self.auth_seq}{self.ins_code}"
        if not self.partial:
            return base
        shown = list(self.atoms[:MAX_ATOMS_IN_TOKEN])
        hidden = len(self.atoms) - len(shown)
        if hidden:
            shown.append(f"+{hidden}")
        return f"{base}[{','.join(shown)}]"

    def to_dict(self) -> dict[str, Any]:
        return {
            "label_chain": self.label_chain,
            "label_seq": self.label_seq,
            "auth_chain": self.auth_chain,
            "auth_seq": self.auth_seq,
            "ins_code": self.ins_code,
            "comp_id": self.comp_id,
            "entity_type": self.entity_type,
            "auth_known": self.auth_known,
            "atoms": list(self.atoms),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResidueRef":
        return cls(
            label_chain=str(data["label_chain"]),
            label_seq=data.get("label_seq"),
            auth_chain=str(data["auth_chain"]),
            auth_seq=int(data["auth_seq"]),
            ins_code=str(data.get("ins_code") or ""),
            comp_id=str(data.get("comp_id") or ""),
            entity_type=str(data.get("entity_type") or "polymer"),
            auth_known=bool(data.get("auth_known", True)),
            atoms=tuple(str(n) for n in data.get("atoms") or ()),
        )


def _residue_from_payload(residue: dict[str, Any]) -> Optional[ResidueRef]:
    """Read one ``atoms[].residue`` dict, or None if it names no residue."""
    chain = str(residue.get("chainId") or "").strip()
    try:
        seq = int(residue.get("seqNumber"))
    except (TypeError, ValueError):
        return None
    if not chain:
        return None

    entity_type = str(residue.get("entityType") or "polymer")
    # Mol* reports label_seq_id 0 for a residue with none.
    label_seq: Optional[int] = seq if entity_type == "polymer" and seq > 0 else None

    auth_chain_raw = residue.get("authChainId")
    auth_seq_raw = residue.get("authSeqNumber")
    auth_known = auth_chain_raw is not None and auth_seq_raw is not None
    if auth_known:
        try:
            auth_chain, auth_seq = str(auth_chain_raw).strip(), int(auth_seq_raw)
        except (TypeError, ValueError):
            auth_chain, auth_seq, auth_known = chain, seq, False
    else:
        auth_chain, auth_seq = chain, seq

    return ResidueRef(
        label_chain=chain,
        label_seq=label_seq,
        auth_chain=auth_chain,
        auth_seq=auth_seq,
        ins_code=str(residue.get("insCode") or "").strip(),
        comp_id=str(residue.get("name") or "").strip(),
        entity_type=entity_type,
        auth_known=auth_known,
    )


def residues_from_atoms(atoms: Iterable[dict[str, Any]]) -> list[ResidueRef]:
    """Distinct residues of ``atoms``, in the order the viewer reported them.

    Each residue carries the names of its atoms that appear in ``atoms``.
    """
    order: list[ResidueRef] = []
    names: dict[tuple[str, int, str], list[str]] = {}
    for atom in atoms:
        ref = _residue_from_payload(atom.get("residue") or {})
        if ref is None:
            continue
        seen = names.get(ref.key)
        if seen is None:
            names[ref.key] = seen = []
            order.append(ref)
        name = str(atom.get("name") or "").strip()
        if name and name not in seen:
            seen.append(name)
    return [replace(ref, atoms=tuple(names[ref.key])) for ref in order]


def residues_from_selection(selection: dict[str, Any]) -> list[ResidueRef]:
    """Distinct residues of a stored ``selection`` artifact."""
    return residues_from_atoms(selection.get("atoms") or [])


def _runs(residues: list[ResidueRef]) -> list[list[ResidueRef]]:
    """Group consecutive residues of one chain, so 40 in a row is one token."""
    runs: list[list[ResidueRef]] = []
    for residue in residues:
        if runs:
            last = runs[-1][-1]
            if (
                residue.auth_chain == last.auth_chain
                and residue.auth_seq == last.auth_seq + 1
                and not residue.ins_code
                and not last.ins_code
                and residue.is_polymer
                and last.is_polymer
                # A run reads "these residues, whole"; a partial one has atoms
                # to name, so it stands on its own.
                and not residue.partial
                and not last.partial
            ):
                runs[-1].append(residue)
                continue
        runs.append([residue])
    return runs


def format_residue_listing(
    residues: list[ResidueRef], max_listed: int = DEFAULT_MAX_LISTED
) -> str:
    """Compact, chain-grouped listing in author numbering.

    ``A: Y56 M115 S120-T125 (6 res) | B: K7`` -- a run of consecutive residues
    is one token, and past ``max_listed`` tokens the rest are counted rather
    than named. Empty string for no residues.
    """
    if not residues:
        return ""

    by_chain: dict[str, list[ResidueRef]] = {}
    for residue in residues:
        by_chain.setdefault(residue.auth_chain, []).append(residue)

    shown = 0
    hidden = 0
    parts: list[str] = []
    for chain, members in by_chain.items():
        tokens: list[str] = []
        for run in _runs(members):
            if shown >= max_listed:
                hidden += len(run)
                continue
            shown += 1
            if len(run) == 1:
                tokens.append(run[0].token)
            else:
                tokens.append(f"{run[0].token}-{run[-1].token} ({len(run)} res)")
        if tokens:
            parts.append(f"{chain}: {' '.join(tokens)}")

    listing = " | ".join(parts)
    if hidden:
        listing += f" (+{hidden} more residues)"
    return listing
