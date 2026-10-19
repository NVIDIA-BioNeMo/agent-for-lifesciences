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

"""Translate author residue numbering into label numbering.

An mmCIF carries two numbering systems for the same residue. ``auth_seq_id``
and ``auth_asym_id`` are what the depositor wrote: they match the paper, they
can start at any number, they can skip, and they can repeat with an insertion
code. ``label_seq_id`` and ``label_asym_id`` are the programmatic pair: one
chain per entity instance, residues numbered 1..n with no gaps.

Everything on either side of this module has an opinion about which it means:

* **Mol\\*** shows the user author numbering, and MolQL's ``atom.resno`` /
  ``atom.chain`` match it, so a hotspot picked or queried by eye is in author
  numbering. The selection payload it posts carries both: ``authSeqNumber`` /
  ``authChainId`` are the author ids, and ``seqNumber`` / ``chainId`` are the
  label ids (an earlier version of this docstring claimed the latter were
  author ids, which they never were -- verified on 5O45, where querying
  ``atom.resno 56`` returns a payload with ``seqNumber 40``). Selections
  written before the author fields existed carry only the label pair.
* **AtomWorks**, which parses the condition specs RFD4-Proteina consumes,
  resolves ``sel()`` ``res_id`` against ``label_seq_id`` and prefers
  ``label_asym_id`` for ``chain_id``. Its documentation is explicit: "``res_id``
  means the mmCIF ``label_seq_id``, never the author deposited ``auth_seq_id``."

For a structure where the two agree -- a cropped construct renumbered from 1 --
the distinction is invisible. For most PDB entries they differ, often by a
constant offset, sometimes not even by that. Emitting a ``sel()`` with author
numbers against a structure where they disagree does not fail: it silently
conditions the wrong residues, and the design comes back wrong with no error
anywhere. Hence this module, and hence its test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .mmcif_loop import as_text, atom_site_columns, iter_atom_site_rows

# mmCIF's two spellings of "no value". A residue with no label_seq_id is a
# non-polymer: ligands and waters are numbered only by the author.
_NULL_TOKENS = frozenset({".", "?"})


@dataclass(frozen=True)
class ResidueKey:
    """A residue as the viewer names it.

    Attributes:
        chain_id: ``auth_asym_id``, the chain letter the user sees.
        seq_id: ``auth_seq_id``, the residue number the user sees.
        ins_code: Insertion code, empty when there is none. Two residues can
            share an author number and differ only here.
    """

    chain_id: str
    seq_id: int
    ins_code: str = ""


@dataclass(frozen=True)
class LabelResidue:
    """A residue as AtomWorks names it.

    ``seq_id`` is None for a non-polymer, which has no ``label_seq_id``. Such
    a residue cannot be named by ``res_id`` in a ``sel()`` at all; it is
    selected by ``res_name`` or ``is_ligand`` instead.
    """

    chain_id: str
    seq_id: Optional[int]
    comp_id: str = ""


@dataclass
class NumberingMap:
    """Author-to-label residue numbering for one structure.

    Attributes:
        residues: Author residue key -> its label counterpart.
        identical: True when every polymer residue has the same chain id and
            number in both systems. A caller can skip translation entirely in
            that case, and the design tool says so in its provenance metadata.
    """

    residues: dict[ResidueKey, LabelResidue] = field(default_factory=dict)
    identical: bool = True
    #: ``(label chain, label_seq_id)`` of each residue that is *not* bonded to
    #: the next one, though the numbering says it should be. See
    #: :meth:`split_span`.
    breaks: dict[tuple[str, int], float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.residues)

    def split_span(self, chain: str, first: int, last: int) -> list[tuple[int, int]]:
        """Cut a label-numbered span wherever the chain is physically broken.

        A contig names a span like ``B1-213``, which tells the model every
        neighbouring pair is bonded. When residues are unmodelled the
        coordinates jump while the numbering may not: a PDB converted to
        mmCIF is renumbered 1..n over the *modelled* residues, so the gap is
        invisible in the ids. The model then rejects the structure, naming a
        bond of several angstroms. Spans are therefore cut at each such break,
        and the caller sends the pieces as separate segments.
        """
        cuts = sorted(seq for (c, seq) in self.breaks if c == chain and first <= seq < last)
        pieces: list[tuple[int, int]] = []
        start = first
        for seq in cuts:
            pieces.append((start, seq))
            start = seq + 1
        pieces.append((start, last))
        return pieces

    def break_notes(self, chain: str, first: int, last: int) -> list[str]:
        """One sentence per break inside a span, in the numbering the user sees."""
        by_label = {
            (label.chain_id, label.seq_id): key
            for key, label in self.residues.items()
            if label.seq_id is not None
        }
        notes = []
        for (c, seq), distance in sorted(self.breaks.items()):
            if c != chain or not first <= seq < last:
                continue
            before, after = by_label.get((c, seq)), by_label.get((c, seq + 1))
            if before and after:
                where = (
                    f"{before.chain_id}{before.seq_id}{before.ins_code} and "
                    f"{after.chain_id}{after.seq_id}{after.ins_code}"
                )
            else:
                where = f"{c}{seq} and {c}{seq + 1}"
            notes.append(
                f"Chain {before.chain_id if before else c} is broken between {where} "
                f"({distance:.1f} \u00c5 apart: residues are unmodelled there), so it is "
                "sent as separate segments rather than as one bonded chain."
            )
        return notes

    def lookup(
        self, chain_id: str, seq_id: int, ins_code: str = ""
    ) -> Optional[LabelResidue]:
        """Find a residue by author numbering.

        Falls back to ignoring the insertion code, because Mol* reports an
        empty one for residues that have none while some files write ``.``,
        and a caller should not have to know which.
        """
        key = ResidueKey(chain_id, seq_id, _clean_ins_code(ins_code))
        found = self.residues.get(key)
        if found is not None:
            return found
        if key.ins_code:
            return self.residues.get(ResidueKey(chain_id, seq_id, ""))
        return None

    @property
    def chain_ids(self) -> list[str]:
        """Author chain ids present, in first-seen order."""
        seen: dict[str, None] = {}
        for key in self.residues:
            seen.setdefault(key.chain_id, None)
        return list(seen)

    def label_chain(self, auth_chain_id: str) -> Optional[str]:
        """The label chain id for an author chain, if it is unambiguous.

        One author chain can span several label chains -- a protein and the
        waters beside it are separate entities sharing one author letter. When
        that happens the polymer's chain is the answer, because a design
        condition is about the polymer.
        """
        candidates = [
            label.chain_id
            for key, label in self.residues.items()
            if key.chain_id == auth_chain_id and label.seq_id is not None
        ]
        if not candidates:
            candidates = [
                label.chain_id
                for key, label in self.residues.items()
                if key.chain_id == auth_chain_id
            ]
        if not candidates:
            return None
        # Ties are broken by frequency: the polymer dominates the atom count.
        return max(set(candidates), key=candidates.count)

    def polymer_span(self, auth_chain_id: str) -> Optional[tuple[int, int]]:
        """Lowest and highest ``label_seq_id`` of a chain's polymer residues.

        This is what a contig needs: ``A17-131`` names a label-numbered range,
        so the span has to be read off the same system the contig resolves in.
        """
        numbers = [
            label.seq_id
            for key, label in self.residues.items()
            if key.chain_id == auth_chain_id and label.seq_id is not None
        ]
        return (min(numbers), max(numbers)) if numbers else None


def build_numbering_map(content: str | bytes) -> NumberingMap:
    """Read the author-to-label residue map out of an mmCIF's ``_atom_site``.

    Rows are per atom, so each residue is seen many times; the first sighting
    wins and the rest confirm it. A file with no ``_atom_site`` loop, or none
    of the four columns, yields an empty map -- which callers must treat as
    "cannot translate", not as "nothing to translate".
    """
    text = as_text(content)
    if not text:
        return NumberingMap()

    columns = atom_site_columns(text)
    if not columns:
        return NumberingMap()

    index = {name: position for position, name in enumerate(columns)}
    auth_chain_at = index.get("auth_asym_id", index.get("label_asym_id"))
    auth_seq_at = index.get("auth_seq_id")
    label_chain_at = index.get("label_asym_id", index.get("auth_asym_id"))
    label_seq_at = index.get("label_seq_id")
    ins_code_at = index.get("pdbx_PDB_ins_code")
    comp_at = index.get("label_comp_id", index.get("auth_comp_id"))

    if auth_chain_at is None or auth_seq_at is None or label_chain_at is None:
        return NumberingMap()

    mapping: dict[ResidueKey, LabelResidue] = {}
    identical = True

    # The backbone atoms that bond a residue to its neighbours, by label id, in
    # file order, for finding breaks the numbering does not show.
    atom_at = index.get("label_atom_id")
    coord_at = [index.get(axis) for axis in ("Cartn_x", "Cartn_y", "Cartn_z")]
    can_measure = (
        atom_at is not None
        and label_seq_at is not None
        and all(at is not None for at in coord_at)
    )
    backbone: dict[tuple[str, int], dict[str, tuple[float, float, float]]] = {}
    residue_order: list[tuple[str, int]] = []

    for values in iter_atom_site_rows(text):
        if can_measure:
            label_seq_value = _as_int(values[label_seq_at])
            if label_seq_value is not None and values[atom_at] in ("C", "N"):
                residue_id = (values[label_chain_at], label_seq_value)
                try:
                    xyz = tuple(float(values[at]) for at in coord_at)  # type: ignore[index]
                except ValueError:
                    xyz = None
                if xyz is not None:
                    if residue_id not in backbone:
                        backbone[residue_id] = {}
                        residue_order.append(residue_id)
                    backbone[residue_id].setdefault(values[atom_at], xyz)  # type: ignore[arg-type]
            elif label_seq_value is not None:
                residue_id = (values[label_chain_at], label_seq_value)
                if residue_id not in backbone:
                    backbone[residue_id] = {}
                    residue_order.append(residue_id)

        auth_seq = _as_int(values[auth_seq_at])
        if auth_seq is None:
            continue

        key = ResidueKey(
            chain_id=values[auth_chain_at],
            seq_id=auth_seq,
            ins_code=_clean_ins_code(
                values[ins_code_at] if ins_code_at is not None else ""
            ),
        )
        if key in mapping:
            continue

        label_seq = (
            _as_int(values[label_seq_at]) if label_seq_at is not None else None
        )
        label = LabelResidue(
            chain_id=values[label_chain_at],
            seq_id=label_seq,
            comp_id=values[comp_at] if comp_at is not None else "",
        )
        mapping[key] = label

        # Non-polymers have no label_seq_id in either system, so they cannot
        # disagree and must not be what flips this flag.
        if label.seq_id is not None and (
            label.seq_id != key.seq_id or label.chain_id != key.chain_id
        ):
            identical = False

    return NumberingMap(
        residues=mapping, identical=identical, breaks=_find_breaks(backbone, residue_order)
    )


# A peptide bond is 1.33 A; the model's own polymerization check allows 2.0.
MAX_PEPTIDE_BOND = 2.0


def _find_breaks(
    backbone: dict[tuple[str, int], dict[str, tuple[float, float, float]]],
    order: list[tuple[str, int]],
) -> dict[tuple[str, int], float]:
    """Residues whose carbonyl C is too far from the next residue's N to be bonded.

    Only residues whose label ids are consecutive are compared: a jump in the
    ids is already a gap and is split without help. Residues missing either
    atom are left alone, since there is nothing to measure.
    """
    breaks: dict[tuple[str, int], float] = {}
    for here, there in zip(order, order[1:]):
        if here[0] != there[0] or there[1] != here[1] + 1:
            continue
        c = backbone[here].get("C")
        n = backbone[there].get("N")
        if c is None or n is None:
            continue
        distance = sum((a - b) ** 2 for a, b in zip(c, n)) ** 0.5
        if distance > MAX_PEPTIDE_BOND:
            breaks[here] = distance
    return breaks


def _as_int(token: str) -> Optional[int]:
    if token in _NULL_TOKENS:
        return None
    try:
        return int(token)
    except (TypeError, ValueError):
        return None


def _clean_ins_code(value: str) -> str:
    """Normalise mmCIF's null tokens and Mol*'s empty string to ``""``."""
    token = (value or "").strip()
    return "" if token in _NULL_TOKENS else token
