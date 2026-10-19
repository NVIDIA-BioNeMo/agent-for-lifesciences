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

"""Build and combine selections without a viewer.

Everything here is a pure function over atom payloads (the viewer's
``AtomData`` shape), so it runs identically in the deployed agent, in an eval
with no browser, and in NeMo-Gym training. The viewer is only ever asked to
*show* the result.

Scope is the residue unless atoms were chosen: set operations compare
residues by their author identity ``(chain, number, insertion code)``, and
``expand`` grows by whole residues. When either operand of a set operation is
atom-level (some residues hold only some of their atoms), the operation
compares atoms by ``(residue, atom name)`` instead, so subtracting ``CA`` from
a residue leaves the rest of it. Nothing here ever fabricates an atom a
structure does not have.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .residues import _residue_from_payload
from .structure import StructureAtom

# "56", "52A" (insertion code), "100-120"
_SPEC = re.compile(r"^\s*(-?\d+)([A-Za-z]?)\s*(?:-\s*(-?\d+)([A-Za-z]?))?\s*$")

# Atom keywords, which stand for sets of names. Hydrogens are in neither:
# structures from the PDB do not have them, and a hotspot is heavy atoms.
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})
_NOT_SIDECHAIN = BACKBONE_ATOMS | {"OXT"}
ATOM_KEYWORDS = ("backbone", "sidechain")

OPERATIONS = ("union", "intersect", "subtract", "expand")


class SelectionError(ValueError):
    """A request that cannot be satisfied, phrased for the model to act on."""


@dataclass(frozen=True)
class ResidueSpan:
    """An inclusive range of residue numbers (one residue when start == end)."""

    start: int
    end: int
    ins_code: str = ""
    # Atoms to keep in these residues: names and/or the keywords ``backbone``
    # and ``sidechain``. None means no opinion, so the call-wide ``atoms``
    # argument (or every atom) applies.
    atoms: Optional[tuple[str, ...]] = None

    def __str__(self) -> str:
        base = (
            f"{self.start}{self.ins_code}"
            if self.start == self.end
            else f"{self.start}-{self.end}"
        )
        return f"{base}:{','.join(self.atoms)}" if self.atoms else base

    def contains(self, number: int, ins_code: str) -> bool:
        if self.start == self.end:
            return number == self.start and (not self.ins_code or ins_code == self.ins_code)
        return self.start <= number <= self.end


def atom_tokens(value: Any) -> Optional[tuple[str, ...]]:
    """Atom names and keywords from ``["CA", "OH"]``, ``"CA,OH"`` or ``"sidechain"``.

    Raises:
        SelectionError: If ``value`` is not a name, a list of names, or empty.
    """
    if value is None:
        return None
    if isinstance(value, str):
        raw: list[Any] = [t for t in re.split(r"[,\s]+", value) if t]
    elif isinstance(value, (list, tuple)):
        raw = list(value)
    else:
        raise SelectionError(
            "atoms must be a list of atom names such as ['CA', 'OH'], or one of: "
            + ", ".join(ATOM_KEYWORDS)
            + "."
        )
    tokens = tuple(str(t).strip().upper() for t in raw if str(t).strip())
    if not tokens:
        return None
    return tokens


def _atom_filter(tokens: Optional[Sequence[str]]):
    """A predicate over an upper-case atom name, or None to keep every atom."""
    if not tokens:
        return None
    names = {t for t in tokens if t.lower() not in ATOM_KEYWORDS}
    backbone = any(t.lower() == "backbone" for t in tokens)
    sidechain = any(t.lower() == "sidechain" for t in tokens)

    def keep(name: str) -> bool:
        if name in names:
            return True
        if backbone and name in BACKBONE_ATOMS:
            return True
        return sidechain and name not in _NOT_SIDECHAIN and not name.startswith("H")

    return keep


def parse_residue_spec(items: Iterable[Any]) -> list[ResidueSpan]:
    """Parse ``[56, "100-120", "52A", "56:OH,CZ", "100-120:sidechain"]`` into spans.

    A ``:`` ends the residue part and starts the atoms to keep in it.

    Raises:
        SelectionError: On anything else, naming the offending item.
    """
    spans: list[ResidueSpan] = []
    for item in items:
        if isinstance(item, bool):
            raise SelectionError(f"Residue '{item}' is not a number or range.")
        if isinstance(item, int):
            spans.append(ResidueSpan(item, item))
            continue
        text, _, atom_part = str(item).partition(":")
        match = _SPEC.match(text)
        if not match:
            raise SelectionError(
                f"Could not read residue '{item}'. Use numbers like 56, ranges "
                "like '100-120', an insertion code like '52A', or residue:atoms "
                "like '56:OH,CZ'."
            )
        atoms = atom_tokens(atom_part) if ":" in str(item) else None
        if ":" in str(item) and atoms is None:
            raise SelectionError(
                f"'{item}' names no atoms after the ':'. Write '56:OH,CZ', or "
                "just '56' for the whole residue."
            )
        start, ins_a, end, _ins_b = match.groups()
        if end is None:
            spans.append(ResidueSpan(int(start), int(start), ins_a, atoms))
        else:
            low, high = int(start), int(end)
            if low > high:
                raise SelectionError(
                    f"Range '{item}' runs backwards; write it low to high."
                )
            spans.append(ResidueSpan(low, high, atoms=atoms))
    if not spans:
        raise SelectionError("No residues were given.")
    return spans


@dataclass
class ResidueSelectionResult:
    """What a by-number selection found and what it could not."""

    atoms: list[StructureAtom] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    residue_count: int = 0
    # True when any atom filter applied, so the result may hold only part of
    # some residues and should be stored as an atom-level selection.
    atom_level: bool = False


def select_residues(
    structure: Sequence[StructureAtom],
    chain: str,
    spans: Sequence[ResidueSpan],
    *,
    numbering: str = "author",
    atom_names: Optional[Sequence[str] | str] = None,
) -> ResidueSelectionResult:
    """Pick residues of one chain by number.

    Args:
        structure: Atoms of the structure.
        chain: Chain id, in the same numbering system as ``numbering``.
        spans: Residues and ranges to take.
        numbering: ``"author"`` (what the viewer shows; MolQL ``atom.resno``)
            or ``"label"`` (what RFD4's ``res_id`` means).
        atom_names: Keep only these atoms of each residue (``["CA"]``), or the
            keyword ``backbone`` or ``sidechain``. None keeps them all. A span
            that names its own atoms (``"56:OH"``) uses those instead.

    Raises:
        SelectionError: If ``chain`` is not in the structure or ``numbering``
            is not one of the two systems.
    """
    if numbering not in ("author", "label"):
        raise SelectionError("numbering must be 'author' or 'label'.")
    author = numbering == "author"

    def chain_of(a: StructureAtom) -> str:
        return a.auth_chain if author else a.label_chain

    chains = sorted({chain_of(a) for a in structure})
    if chain not in chains:
        raise SelectionError(
            f"Chain '{chain}' is not in this structure ({numbering} chain ids: "
            f"{', '.join(chains) or 'none'})."
        )

    global_filter = _atom_filter(atom_tokens(atom_names))
    span_filters = [_atom_filter(span.atoms) if span.atoms else None for span in spans]
    explicit = [
        {t for t in (span.atoms or ()) if t.lower() not in ATOM_KEYWORDS} for span in spans
    ]
    matched_names: list[set[str]] = [set() for _ in spans]

    result = ResidueSelectionResult(
        atom_level=global_filter is not None or any(span.atoms for span in spans)
    )
    hit = [False] * len(spans)
    kept_residues: set[tuple[str, int, str]] = set()

    for atom in structure:
        if chain_of(atom) != chain or atom.entity_type == "water":
            continue
        number = atom.auth_seq if author else atom.label_seq
        if number is None:
            continue
        name = atom.name.upper()
        residue_matched = False
        keep = False
        for i, span in enumerate(spans):
            if not span.contains(number, atom.ins_code if author else ""):
                continue
            hit[i] = True
            residue_matched = True
            # A span's own atoms win over the call-wide ones.
            wanted = span_filters[i] if span.atoms else global_filter
            if wanted is None or wanted(name):
                keep = True
                if name in explicit[i]:
                    matched_names[i].add(name)
        if not residue_matched or not keep:
            continue
        result.atoms.append(atom)
        kept_residues.add(atom.residue_key)

    result.residue_count = len(kept_residues)
    result.missing = [str(span) for span, found in zip(spans, hit) if not found]
    # An atom named for a residue that has no such atom is as much a miss as
    # a residue that is not there, and the model should be told which.
    for i, span in enumerate(spans):
        if hit[i]:
            result.missing.extend(
                f"{ResidueSpan(span.start, span.end, span.ins_code)}:{n}"
                for n in sorted(explicit[i] - matched_names[i])
            )
    return result


def _residue_key(atom: dict[str, Any]) -> Optional[tuple[str, int, str]]:
    ref = _residue_from_payload(atom.get("residue") or {})
    return ref.key if ref is not None else None


def _atom_key(atom: dict[str, Any]) -> Optional[tuple[tuple[str, int, str], str]]:
    key = _residue_key(atom)
    return (key, str(atom.get("name", ""))) if key is not None else None


def combine(
    operation: str,
    first: list[dict[str, Any]],
    second: list[dict[str, Any]],
    *,
    atom_level: bool = False,
) -> list[dict[str, Any]]:
    """``union``, ``intersect`` or ``subtract`` of two atom lists.

    By residue unless ``atom_level``: ``intersect`` and ``subtract`` then keep
    or drop whole residues of the *first* operand. With ``atom_level`` they
    compare ``(residue, atom name)``, so ``intersect`` keeps the atoms both
    selections hold and ``subtract`` removes only the second's atoms.
    ``union`` keeps the first operand's atoms and adds any of the second's that
    the first lacks, either way.
    """
    if operation == "union":
        seen = {k for a in first if (k := _atom_key(a)) is not None}
        extra = [a for a in second if (k := _atom_key(a)) is not None and k not in seen]
        return list(first) + extra

    if operation in ("intersect", "subtract"):
        if atom_level:
            other = {k for a in second if (k := _atom_key(a)) is not None}
            key = _atom_key
        else:
            other = {k for a in second if (k := _residue_key(a)) is not None}
            key = _residue_key
        if operation == "intersect":
            return [a for a in first if key(a) in other]
        return [a for a in first if (k := key(a)) is not None and k not in other]

    raise SelectionError(
        f"Unknown operation '{operation}'. Use one of: {', '.join(OPERATIONS)}."
    )


def expand(
    selected: list[dict[str, Any]],
    structure: Sequence[StructureAtom],
    radius: float,
    *,
    whole_residues: bool = True,
    include_original: bool = True,
) -> list[dict[str, Any]]:
    """Grow a selection to everything within ``radius`` angstroms of it.

    Args:
        selected: The selection's atoms.
        structure: The structure to look in. Waters are never added.
        radius: Distance in angstroms from any selected atom.
        whole_residues: Return every atom of each residue that has one within
            reach (the usual meaning of a neighbour shell). False returns only
            the atoms that are themselves within reach.
        include_original: Keep the selection itself in the result.

    Raises:
        SelectionError: If the radius is not positive, or the selection has no
            coordinates to measure from.
    """
    import numpy as np

    if not 0 < radius <= 30:
        raise SelectionError("radius must be between 0 and 30 angstroms.")
    origin = np.array([a["coords"] for a in selected if a.get("coords")], dtype=float)
    if origin.size == 0:
        raise SelectionError("The selection has no atoms to measure from.")

    candidates = [a for a in structure if a.entity_type != "water"]
    if not candidates:
        return list(selected) if include_original else []
    points = np.array([(a.x, a.y, a.z) for a in candidates], dtype=float)

    limit = radius * radius
    near = np.zeros(len(candidates), dtype=bool)
    # Bound the working set: rows x selection-size distances at a time.
    rows = max(1, 4_000_000 // max(1, len(origin)))
    for start in range(0, len(candidates), rows):
        block = points[start : start + rows]
        d2 = ((block[:, None, :] - origin[None, :, :]) ** 2).sum(axis=2)
        near[start : start + rows] = d2.min(axis=1) <= limit

    if whole_residues:
        reach = {a.residue_key for a, flag in zip(candidates, near) if flag}
        grown = [a.to_payload() for a in candidates if a.residue_key in reach]
    else:
        grown = [a.to_payload() for a, flag in zip(candidates, near) if flag]

    if include_original:
        return combine("union", list(selected), grown)
    original = {k for a in selected if (k := _residue_key(a)) is not None}
    return [a for a in grown if _residue_key(a) not in original]
