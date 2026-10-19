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

"""Turn a design spec's roles into what a design engine is told.

Two stages, kept apart so each can be tested alone:

1. **Resolve.** A stored selection becomes a list of residues in both
   numberings (:func:`resolve_selection`). The viewer reports *author*
   numbering and RFD4 reads *label* numbering, so every residue is translated
   through the target's :class:`NumberingMap` and anything that cannot be
   mapped is dropped *and said so* -- a silently dropped residue changes the
   design.
2. **Compile.** The resolved layers become an engine's request:
   :func:`compile_rfd4` builds a contig and a ``conditions`` block, and
   :func:`compile_rfdiffusion1` a contig string and hotspot list.

Compiling never raises for a bad design; it returns ``errors`` the caller must
treat as fatal and ``warnings`` it should show, so the model can act on them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from api.storage.mmcif_numbering import NumberingMap

from .design_checks import role_findings
from .design_spec import DEFAULT_LINKER, Layout, SpecLayer
from .residues import STANDARD_HEAVY_ATOMS
from .roles import (
    ENGINE_RFD4,
    ENGINE_RFDIFFUSION1,
    ROLE_ORDER,
    ROLES,
    SUPPORT_DOWNSTREAM,
    SUPPORT_IMPLIED,
    SUPPORT_NONE,
    SUPPORT_RESIDUE,
    VALUE_ENUM,
    apply_atom_rule,
)

#: More named atoms than this and a selector names the whole residue.
MAX_NAMED_ATOMS = 8

#: Backbone atoms. A side-chain atom in a Fix position pick reveals which
#: residue it is, so it implies Fix sequence.
BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O", "OXT"})

MODE_BINDER = "binder"
MODE_MOTIF = "motif"


# --- Resolving a selection -------------------------------------------------


def author_identity(residue: dict[str, Any]) -> Optional[tuple[str, int]]:
    """The residue's author chain and number, or None if it names neither.

    Selections made since the two numbering systems were separated carry
    ``authChainId`` / ``authSeqNumber`` explicitly. Older payloads carry only
    ``chainId`` / ``seqNumber``, which the translation has always read as
    author numbering -- ambiguous, because the viewer emitted label ids there,
    but there is nothing better to read.
    """
    auth_chain = residue.get("authChainId")
    auth_seq = residue.get("authSeqNumber")
    if auth_chain is not None and auth_seq is not None:
        chain_id, raw_seq = str(auth_chain).strip(), auth_seq
    else:
        chain_id, raw_seq = (
            str(residue.get("chainId") or "").strip(),
            residue.get("seqNumber"),
        )
    try:
        seq_number = int(raw_seq)
    except (TypeError, ValueError):
        return None
    if not chain_id:
        return None
    return chain_id, seq_number


def reported_label(residue: dict[str, Any]) -> Optional[tuple[str, int]]:
    """The label chain and number the viewer reported, when it separated them."""
    if residue.get("authChainId") is None or residue.get("authSeqNumber") is None:
        return None
    try:
        return str(residue.get("chainId") or "").strip(), int(residue.get("seqNumber"))
    except (TypeError, ValueError):
        return None


@dataclass
class ResolvedResidue:
    """One picked residue, in both numberings, with the atoms picked in it."""

    author_chain: str
    author_seq: int
    comp: str
    atoms: list[str] = field(default_factory=list)
    label_chain: Optional[str] = None
    label_seq: Optional[int] = None
    ins_code: str = ""

    @property
    def author(self) -> str:
        return f"{self.author_chain}{self.author_seq}"

    @property
    def label(self) -> str:
        return f"{self.label_chain}{self.label_seq}"

    @property
    def key(self) -> tuple[str, int]:
        """Label-numbered identity; what RFD4 resolves."""
        return (self.label_chain or "", self.label_seq if self.label_seq is not None else 0)

    @property
    def is_whole(self) -> bool:
        """Whether the pick covers the whole residue rather than chosen atoms."""
        heavy = {a for a in self.atoms if not a.upper().startswith("H")}
        standard = STANDARD_HEAVY_ATOMS.get(self.comp.upper())
        if standard is not None:
            return len(heavy) >= standard
        return len(heavy) > MAX_NAMED_ATOMS


@dataclass
class ResolvedSelection:
    """A selection resolved onto a target, plus what could not be placed."""

    residues: list[ResolvedResidue] = field(default_factory=list)
    #: Author ids of residues the viewer's label numbering disagrees on.
    disagreements: set[str] = field(default_factory=set)
    unmapped: set[str] = field(default_factory=set)
    non_polymer: set[str] = field(default_factory=set)
    empty: bool = False

    @property
    def chains(self) -> list[str]:
        out: list[str] = []
        for residue in self.residues:
            if residue.label_chain and residue.label_chain not in out:
                out.append(residue.label_chain)
        return out

    def translated(self) -> list[ResolvedResidue]:
        """Residues whose label id differs from the author id."""
        return [r for r in self.residues if r.author != r.label]

    def tail_notes(self, what: str = "hotspot") -> list[str]:
        """Notes about residues that were moved, dropped or mismatched."""
        notes: list[str] = []
        if self.disagreements:
            notes.append(
                "The viewer's label numbering for "
                + ", ".join(sorted(self.disagreements))
                + " differs from the target structure's. The "
                f"{what}s were translated from the viewer's author numbering, which is "
                "stable across formats, so they are correct for target_ref; the "
                "selection was probably made on a different copy of the structure."
            )
        if self.unmapped:
            notes.append(
                "Dropped "
                + ", ".join(sorted(self.unmapped))
                + ": not found in the target structure. The selection may have been "
                "made on a different structure than target_ref."
            )
        if self.non_polymer:
            notes.append(
                "Dropped "
                + ", ".join(sorted(self.non_polymer))
                + f": {what}s must be protein residues, and these are not."
            )
        return notes


def resolve_selection(
    selection: dict[str, Any], numbering: Optional[NumberingMap]
) -> ResolvedSelection:
    """Place a stored selection's atoms on a target, residue by residue.

    Residues are identified by their **author** chain and number, stable
    across file formats, and translated to label numbering through
    ``numbering``. The viewer's own ``chainId`` / ``seqNumber`` are label ids
    and are used only to notice a selection made on a different copy of the
    structure. Payloads that predate the author fields are read as author
    numbering throughout.

    With ``numbering=None`` (a PDB target, which has no label numbering) the
    residues keep their author ids only and polymer-ness is read from the
    payload's ``entityType``.
    """
    out = ResolvedSelection()
    atoms = selection.get("atoms") or []
    if not atoms:
        out.empty = True
        return out

    grouped: dict[tuple, ResolvedResidue] = {}
    for atom in atoms:
        residue = atom.get("residue") or {}
        ins_code = str(residue.get("insCode") or "").strip()
        author = author_identity(residue)
        if author is None:
            continue
        chain_id, seq_number = author
        comp = str(residue.get("name") or "").upper()

        label_chain: Optional[str] = None
        label_seq: Optional[int] = None
        if numbering is not None:
            label = numbering.lookup(chain_id, seq_number, ins_code)
            if label is None:
                out.unmapped.add(f"{chain_id}{seq_number}")
                continue
            reported = reported_label(residue)
            if reported is not None and reported != (label.chain_id, label.seq_id):
                out.disagreements.add(f"{chain_id}{seq_number}")
            if label.seq_id is None:
                out.non_polymer.add(f"{chain_id}{seq_number} ({label.comp_id})")
                continue
            label_chain, label_seq = label.chain_id, label.seq_id
            comp = comp or str(label.comp_id or "").upper()
        elif residue.get("entityType") not in (None, "polymer"):
            out.non_polymer.add(f"{chain_id}{seq_number} ({comp})")
            continue

        key = (label_chain, label_seq, chain_id, seq_number)
        resolved = grouped.get(key)
        if resolved is None:
            resolved = grouped[key] = ResolvedResidue(
                author_chain=chain_id,
                author_seq=seq_number,
                comp=comp,
                label_chain=label_chain,
                label_seq=label_seq,
                ins_code=ins_code,
            )
        name = str(atom.get("name") or "").strip()
        if name and name not in resolved.atoms:
            resolved.atoms.append(name)
    out.residues = list(grouped.values())
    return out


# --- Selector formatting ---------------------------------------------------


def merge_ranges(numbers: Iterable[int]) -> list[tuple[int, int]]:
    """Sorted, merged inclusive ranges: ``[3, 4, 5, 9]`` -> ``[(3, 5), (9, 9)]``."""
    ranges: list[tuple[int, int]] = []
    for number in sorted(set(numbers)):
        if ranges and number == ranges[-1][1] + 1:
            ranges[-1] = (ranges[-1][0], number)
        else:
            ranges.append((number, number))
    return ranges


def _range_text(low: int, high: int) -> str:
    return str(low) if low == high else f"{low}-{high}"


def piece(chain: str, low: int, high: int) -> str:
    """A contig token for a kept stretch: ``A5-9``, or ``A5`` for one residue."""
    return f"{chain}{low}" if low == high else f"{chain}{low}-{high}"


def residue_selector(chain: str, numbers: Iterable[int]) -> str:
    """``sel('A/*/[10-20, 25]')`` for whole residues of one chain."""
    ranges = merge_ranges(numbers)
    if len(ranges) == 1 and ranges[0][0] == ranges[0][1]:
        return f"sel('{chain}/*/{ranges[0][0]}')"
    return f"sel('{chain}/*/[{', '.join(_range_text(a, b) for a, b in ranges)}]')"


def atom_selector(chain: str, seq: int, names: list[str]) -> str:
    """``sel('A/*/12/[CA,CB]')``: named atoms of one residue; no names, all of it."""
    if not names:
        return f"sel('{chain}/*/{seq}')"
    atom_part = f"/[{','.join(names)}]" if len(names) > 1 else f"/{names[0]}"
    return f"sel('{chain}/*/{seq}{atom_part}')"


def _by_chain(residues: Iterable[ResolvedResidue]) -> dict[str, list[ResolvedResidue]]:
    out: dict[str, list[ResolvedResidue]] = {}
    for residue in residues:
        out.setdefault(residue.label_chain or "", []).append(residue)
    return out


def _whole_residue_selectors(residues: Iterable[ResolvedResidue]) -> list[str]:
    return [
        residue_selector(chain, (r.label_seq for r in members if r.label_seq is not None))
        for chain, members in _by_chain(residues).items()
    ]


def _atom_selectors(
    residues: Iterable[ResolvedResidue], *, names: Optional[dict[tuple, list[str]]] = None
) -> list[str]:
    """One selector per residue, naming atoms when a handful were picked."""
    out: list[str] = []
    for residue in residues:
        picked = (names or {}).get(residue.key, residue.atoms)
        if picked and len(picked) <= MAX_NAMED_ATOMS and not residue.is_whole:
            out.append(atom_selector(residue.label_chain or "", residue.label_seq or 0, picked))
        else:
            out.append(atom_selector(residue.label_chain or "", residue.label_seq or 0, []))
    return out


# --- Compiling for RFD4 ----------------------------------------------------


@dataclass
class LayerInput:
    """A spec layer with the selection artifact it names."""

    layer: SpecLayer
    selection: dict[str, Any]


@dataclass
class RolePlan:
    """What an engine will be told, and what the user should know about it."""

    contig: str = ""
    conditions: dict[str, Any] = field(default_factory=dict)
    #: Label chains the design keeps (the target's chains, or the motif's).
    chains: list[str] = field(default_factory=list)
    hotspot_selectors: list[str] = field(default_factory=list)
    hotspot_residue_count: int = 0
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: One line per role: what it will do.
    summary: list[str] = field(default_factory=list)
    #: Roles present, as ``(role, value)``.
    roles: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _gap(low: int, high: int) -> str:
    return str(low) if low == high else f"{low}-{high}"


def _kept_runs(
    kept: list[ResolvedResidue], numbering: Optional[NumberingMap] = None
) -> list[tuple[str, int, int]]:
    """Contiguous kept ranges as ``(label chain, first, last)``, in order.

    Chains stay in the order they were first picked, ranges ascend within a
    chain. Ranges that are not adjacent stay separate: closing the gap would
    tell the model that distant residues are sequence neighbours. A range is
    also cut where the structure is physically broken (unmodelled residues
    that the numbering does not show), for the same reason.
    """
    runs: list[tuple[str, int, int]] = []
    for chain, members in _by_chain(kept).items():
        for low, high in merge_ranges(r.label_seq for r in members if r.label_seq is not None):
            pieces = numbering.split_span(chain, low, high) if numbering else [(low, high)]
            runs.extend((chain, a, b) for a, b in pieces)
    return runs


def _chain_labels(numbering: NumberingMap, chain: str) -> list[int]:
    return sorted(
        label.seq_id
        for label in numbering.residues.values()
        if label.chain_id == chain and label.seq_id is not None
    )


def _binder_target(
    keep: list[ResolvedResidue],
    hotspot: list[ResolvedResidue],
    numbering: NumberingMap,
    plan: "RolePlan",
) -> Optional[tuple[list[tuple[str, int, int]], str]]:
    """The target as one bonded stretch per chain, and the selector holding it.

    The model reads each ``/0`` segment as its own output chain named for the
    chain it came from, and refuses two segments with one name ("Contig
    produces two separate output chains both named 'A'"). A binder's target is
    therefore one contiguous stretch per chain:

    * With a Keep, the stretch runs from the chain's first kept residue to its
      last. Residues between two kept stretches are real residues of the
      structure, so carrying them in is faithful, and says so in a note.
    * With no Keep, it is the whole chain.
    * Where the structure is physically broken (unmodelled residues the
      numbering does not show), the stretch cannot cross the break. The side
      holding the kept residues or the hotspot is carried and the other side is
      dropped, loudly; residues needed on both sides are an error.

    Returns None, with the reason on ``plan.errors``, when that is impossible.
    """
    required = _by_chain(keep or hotspot)
    runs: list[tuple[str, int, int]] = []
    selectors: list[str] = []
    for chain, members in required.items():
        needed = sorted({r.label_seq for r in members if r.label_seq is not None})
        if not needed:
            continue
        labels = _chain_labels(numbering, chain)
        first, last = (needed[0], needed[-1]) if keep else (labels[0], labels[-1])
        pieces = numbering.split_span(chain, first, last)
        holding = [
            (a, b) for a, b in pieces if any(a <= n <= b for n in needed)
        ]
        if len(holding) > 1:
            edge = holding[0][1]
            plan.errors.append(
                f"{'Keep' if keep else 'The hotspot'} is on both sides of a break in chain "
                f"{members[0].author_chain}: the structure has unmodelled residues after "
                f"{members[0].author_chain}{numbering_author(numbering, chain, edge)}. A binder's target "
                "has to be one unbroken stretch per chain, so use "
                f"{'a Keep' if keep else 'hotspots'} on one side of it."
            )
            return None
        a, b = holding[0]
        if keep:
            a, b = max(a, first), min(b, last)
            inside = sum(1 for n in needed if a <= n <= b)
            filled = (b - a + 1) - inside
            if filled > 0:
                plan.notes.append(
                    f"Keep on chain {members[0].author_chain} has gaps; the {filled} residue(s) "
                    "between the kept stretches are carried too, because the target has to be "
                    "one unbroken stretch per chain. They are held fixed like the rest."
                )
        else:
            dropped = [p for p in pieces if p != (a, b)]
            author_chain = members[0].author_chain
            for da, db in dropped:
                plan.warnings.append(
                    f"Chain {members[0].author_chain} is broken in the structure (unmodelled "
                    f"residues), so only the part holding the hotspot is used; "
                    f"{author_chain}{numbering_author(numbering, chain, da)}-"
                    f"{author_chain}{numbering_author(numbering, chain, db)} "
                    "is left out. Add a Keep on the other side to use it instead."
                )
        runs.append((chain, a, b))
        whole = labels and (a, b) == (labels[0], labels[-1]) and len(pieces) == 1
        selectors.append(
            f"sel('{chain}')"
            if whole and not keep
            else residue_selector(chain, range(a, b + 1))
        )
    return runs, " or ".join(selectors)


def numbering_author(numbering: NumberingMap, chain: str, label_seq: int) -> str:
    """A label residue as the user sees it, e.g. ``H135``."""
    for key, label in numbering.residues.items():
        if label.chain_id == chain and label.seq_id == label_seq:
            return f"{key.seq_id}{key.ins_code}"
    return str(label_seq)


def _binder_contig(
    runs: list[tuple[str, int, int]], binder_min: int, binder_max: int
) -> str:
    """A binder to design, then each kept run of target as its own chain."""
    segments = [_gap(binder_min, binder_max)]
    segments.extend(piece(chain, low, high) for chain, low, high in runs)
    return ", /0, ".join(segments)


def _motif_contig(runs: list[tuple[str, int, int]], layout: Layout) -> str:
    """Kept pieces joined by designed linkers; one output chain per source chain."""
    chains: list[str] = []
    for chain, _, _ in runs:
        if chain not in chains:
            chains.append(chain)
    run_texts: list[str] = []
    linker_index = 0
    for ci, chain in enumerate(chains):
        tokens: list[str] = []
        if ci == 0 and layout.n_term:
            tokens.append(_gap(*layout.n_term))
        pieces = [r for r in runs if r[0] == chain]
        for pi, (_, low, high) in enumerate(pieces):
            if pi > 0:
                tokens.append(_gap(*layout.linker(linker_index)))
                linker_index += 1
            tokens.append(piece(chain, low, high))
        if ci == len(chains) - 1 and layout.c_term:
            tokens.append(_gap(*layout.c_term))
        run_texts.append(", ".join(tokens))
    return ", /0, ".join(run_texts)


def _numbers_text(residues: list[ResolvedResidue], limit: int = 8) -> str:
    labels = [r.label or r.author for r in residues]
    if len(labels) <= limit:
        return " ".join(labels)
    return " ".join(labels[:limit]) + f" (+{len(labels) - limit} more)"


def compile_rfd4(
    layers: list[LayerInput],
    numbering: NumberingMap,
    *,
    mode: str = MODE_BINDER,
    binder_min: int = 60,
    binder_max: int = 120,
    layout: Optional[Layout] = None,
) -> RolePlan:
    """Compile a spec's layers to an RFD4 contig and ``conditions``.

    Binder mode holds the kept target still (backbone coordinates and
    sequence) and designs a new chain against the hotspot. Motif mode holds
    the kept residues the same way but joins them with designed linkers, so
    the result is one protein that carries the motif.

    A Fix position or Fix sequence layer *adds* to the target's defaults in
    binder mode (a side chain, say) and *replaces* them in motif mode, where
    choosing which kept residues stay put is the point.
    """
    plan = RolePlan()
    layout = layout or Layout()

    # Resolve every layer and apply its role's atom rule.
    resolved: dict[tuple[str, str], list[ResolvedResidue]] = {}
    layer_of: dict[tuple[str, str], SpecLayer] = {}
    for item in layers:
        layer = item.layer
        role = ROLES[layer.condition]
        selection = resolve_selection(item.selection, numbering)
        label = role.label if not layer.key[1] else f"{role.label} ({layer.key[1]})"
        for note in selection.tail_notes(role.label.lower()):
            plan.notes.append(f"{label}: {note}")
        translated = selection.translated()
        for residue in translated[:6]:
            plan.notes.append(
                f"{label}: Residue {residue.author} in the viewer is {residue.label} in "
                f"the numbering RFD4 uses; the {role.label.lower()} was translated."
            )
        if len(translated) > 6:
            plan.notes.append(
                f"{label}: {len(translated) - 6} more residue(s) were translated the same way."
            )
        residues = selection.residues
        if role.is_enum:
            variant = role.variant(layer.value)
            rule = variant.atom_rule if variant else None
        else:
            rule = role.atom_rule
        if rule is not None:
            kept_residues: list[ResolvedResidue] = []
            dropped: list[str] = []
            for residue in residues:
                narrowed = apply_atom_rule(rule, residue.comp, residue.atoms)
                if narrowed is None:
                    dropped.append(residue.author)
                    continue
                if set(narrowed) != set(residue.atoms):
                    residue = ResolvedResidue(
                        author_chain=residue.author_chain,
                        author_seq=residue.author_seq,
                        comp=residue.comp,
                        atoms=narrowed,
                        label_chain=residue.label_chain,
                        label_seq=residue.label_seq,
                        ins_code=residue.ins_code,
                    )
                    plan.notes.append(
                        f"{label}: {residue.author} was narrowed to the atoms the "
                        f"role applies to ({', '.join(narrowed)})."
                    )
                kept_residues.append(residue)
            if dropped:
                plan.errors.append(
                    f"{label}: {', '.join(dropped)} cannot play this role (the picked "
                    "atoms do not include what it needs)."
                )
            residues = kept_residues
        if not residues:
            if not selection.empty:
                plan.errors.append(f"{label}: no usable residues remained.")
            continue
        resolved[layer.key] = residues
        layer_of[layer.key] = layer
        plan.roles.append(layer.key)

    def residues_of(role: str) -> list[ResolvedResidue]:
        out: dict[tuple[str, int], ResolvedResidue] = {}
        for (cond, _), items in resolved.items():
            if cond == role:
                for residue in items:
                    out.setdefault(residue.key, residue)
        return list(out.values())

    keep = residues_of("keep")
    hotspot = residues_of("hotspot")
    fix_position = residues_of("fix_position")
    fix_sequence = residues_of("fix_sequence")

    # --- Cross-role checks ------------------------------------------------
    findings = role_findings(
        {
            role: {r.key for r in residues_of(role)}
            for role in ROLE_ORDER
            if residues_of(role)
        },
        names={r.key: r.author for items in resolved.values() for r in items},
    )
    plan.errors.extend(findings.errors)
    plan.warnings.extend(findings.warnings)

    if mode == MODE_BINDER and not hotspot:
        plan.errors.append(
            "A binder design needs a Hotspot: the residues the binder should touch."
        )
    if mode == MODE_MOTIF and not keep:
        plan.errors.append(
            "Motif scaffolding needs a Keep: the residues of the motif to carry into the design."
        )
    if plan.errors:
        return plan

    # --- Contig and the kept part of the target ---------------------------
    if mode == MODE_BINDER:
        target = _binder_target(keep, hotspot, numbering, plan)
        if target is None:
            return plan
        runs, kept_selector = target
        plan.chains = list(dict.fromkeys(chain for chain, _, _ in runs))
        plan.contig = _binder_contig(runs, binder_min, binder_max)
    else:
        runs = _kept_runs(keep, numbering)
        plan.chains = list(dict.fromkeys(chain for chain, _, _ in runs))
        for chain, low, high in _kept_runs(keep):
            plan.notes.extend(numbering.break_notes(chain, low, high))
        plan.contig = _motif_contig(runs, layout)
        if len(runs) > len(plan.chains):
            plan.notes.append(
                f"Keep has {len(runs)} separate stretches; they are joined by linkers "
                "of the lengths in the layout."
            )
        kept_selector = " or ".join(_whole_residue_selectors(keep))

    # --- Conditions -------------------------------------------------------
    conditions = plan.conditions

    def add(annotation: str, value: str, entry: dict[str, Any]) -> None:
        conditions.setdefault(annotation, {}).setdefault(value, []).append(entry)

    holds_coordinates = f"({kept_selector}) and is_protein_backbone"
    explicit_position = mode == MODE_MOTIF and bool(fix_position)
    explicit_sequence = mode == MODE_MOTIF and bool(fix_sequence)

    sequence_extra: list[ResolvedResidue] = []
    side_chain_fixed = [
        r for r in fix_position if r.atoms and any(a not in BACKBONE_ATOMS for a in r.atoms)
    ]
    if side_chain_fixed:
        have = {r.key for r in fix_sequence}
        sequence_extra = [r for r in side_chain_fixed if r.key not in have]
        if sequence_extra:
            plan.notes.append(
                "Fix position on side-chain atoms reveals those residues' identity, so "
                f"their sequence is fixed too ({_numbers_text(sequence_extra)})."
            )

    if explicit_position:
        for selector in _atom_selectors(fix_position):
            add("C_CRD", "true", {"select": selector})
        plan.notes.append(
            "Fix position replaces the default of holding every kept residue's backbone: "
            "only the residues you fixed keep their coordinates."
        )
    else:
        add("C_CRD", "true", {"select": holds_coordinates})
        for selector in _atom_selectors(fix_position):
            add("C_CRD", "true", {"select": selector})

    if explicit_sequence:
        plan.notes.append(
            "Fix sequence replaces the default of holding every kept residue's identity."
        )
        for selector in _whole_residue_selectors(list(fix_sequence) + sequence_extra):
            add("C_SEQ", "true", {"select": selector})
    else:
        add("C_SEQ", "true", {"select": kept_selector})
        for selector in _whole_residue_selectors(fix_sequence):
            add("C_SEQ", "true", {"select": selector})

    if hotspot:
        plan.hotspot_residue_count = len(hotspot)
        for residue in hotspot:
            names = residue.atoms
            if names and len(names) <= MAX_NAMED_ATOMS:
                selector = atom_selector(residue.label_chain or "", residue.label_seq or 0, names)
            else:
                selector = atom_selector(residue.label_chain or "", residue.label_seq or 0, [])
                standard = STANDARD_HEAVY_ATOMS.get(residue.comp)
                if names and standard is not None and len(names) < standard:
                    plan.notes.append(
                        f"Residue {residue.author}: {len(names)} atoms were chosen, more "
                        f"than the {MAX_NAMED_ATOMS} a hotspot names individually, so RFD4 "
                        "is told the whole residue."
                    )
            plan.hotspot_selectors.append(selector)
            add("C_HOT", "true", {"select": selector})

    # Float: one group per layer, so residues sharing a group keep their spacing.
    for key, items in resolved.items():
        if key[0] != "float":
            continue
        group = key[1] or "motif"
        for selector in _whole_residue_selectors(items):
            add("C_IDX", "false", {"select": selector, "group": group})

    # Enumerated and chemistry roles.
    _compile_value_roles(plan, resolved, add)

    # Summaries.
    for key in plan.roles:
        role = ROLES[key[0]]
        items = resolved[key]
        suffix = f" ({key[1]})" if key[1] else ""
        plan.summary.append(
            f"{role.label}{suffix}: {len(items)} residue(s) -- {_numbers_text(items)}"
        )
    return plan


def _compile_value_roles(plan: RolePlan, resolved, add) -> None:
    """Emit the enumerated and chemistry roles, merging overlaps into BOTH."""
    # Group by role, then variant.
    by_role: dict[str, dict[str, list[ResolvedResidue]]] = {}
    for (role_key, value), items in resolved.items():
        role = ROLES[role_key]
        if role_key in {"hotspot", "keep", "fix_position", "fix_sequence", "float"}:
            continue
        by_role.setdefault(role_key, {})[value or "true"] = items

    for role_key, variants in by_role.items():
        role = ROLES[role_key]
        # DONOR+ACCEPTOR and AROMATIC+CATION on the same atoms mean BOTH; two
        # separate entries would not combine, and the last would win.
        merged = {value: {r.key: r for r in items} for value, items in variants.items()}
        pair = {"hbond": ("DONOR", "ACCEPTOR"), "pication": ("AROMATIC", "CATION")}.get(role_key)
        if pair and all(v in merged for v in pair):
            a, b = pair
            both: dict[tuple[str, int], ResolvedResidue] = {}
            for key in list(merged[a]):
                if key in merged[b]:
                    shared = [
                        atom for atom in merged[a][key].atoms if atom in merged[b][key].atoms
                    ]
                    if not shared:
                        continue
                    base = merged[a][key]
                    both[key] = ResolvedResidue(
                        author_chain=base.author_chain,
                        author_seq=base.author_seq,
                        comp=base.comp,
                        atoms=shared,
                        label_chain=base.label_chain,
                        label_seq=base.label_seq,
                    )
            if both:
                plan.notes.append(
                    f"{role.label}: {len(both)} residue(s) were marked as both {a.lower()} "
                    f"and {b.lower()}, so they are sent as BOTH."
                )
                for key, residue in both.items():
                    for value in (a, b):
                        current = merged[value][key]
                        remaining = [x for x in current.atoms if x not in residue.atoms]
                        if remaining:
                            merged[value][key] = ResolvedResidue(
                                author_chain=current.author_chain,
                                author_seq=current.author_seq,
                                comp=current.comp,
                                atoms=remaining,
                                label_chain=current.label_chain,
                                label_seq=current.label_seq,
                            )
                        else:
                            del merged[value][key]
                merged.setdefault("BOTH", {}).update(both)
        for value, members in merged.items():
            if not members:
                continue
            level = value if role.value_kind == VALUE_ENUM else "true"
            if role.granularity == "residue":
                selectors = _whole_residue_selectors(members.values())
            else:
                selectors = [
                    atom_selector(r.label_chain or "", r.label_seq or 0, r.atoms)
                    for r in members.values()
                ]
            for selector in selectors:
                add(role.annotation, level, {"select": selector})


# --- Compiling for RFdiffusion v1 ------------------------------------------


@dataclass
class RFdiffusion1Plan:
    """A contig string and hotspot list for RFdiffusion v1."""

    contigs: str = ""
    hotspot_res: list[str] = field(default_factory=list)
    #: ProteinMPNN fixed positions, ``A:30-35``-style, from Fix sequence.
    fixed_positions: list[str] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def compile_rfdiffusion1(
    layers: list[LayerInput],
    target_numbering: Optional[NumberingMap] = None,
    *,
    mode: str = MODE_BINDER,
    binder_min: int = 60,
    binder_max: int = 120,
    layout: Optional[Layout] = None,
) -> RFdiffusion1Plan:
    """Compile a spec's layers for RFdiffusion v1 (author numbering, PDB).

    Roles map as the engine allows (see ``roles.ENGINES``): Keep becomes the
    contig; Hotspot becomes ``hotspot_res`` at residue level; Fix position is
    implied by Keep; Fix sequence is returned as ProteinMPNN fixed positions;
    Float, chemistry and secondary structure cannot be honoured and are
    reported as dropped, never silently.
    """
    plan = RFdiffusion1Plan()
    layout = layout or Layout()
    resolved: dict[str, list[ResolvedResidue]] = {}
    seen_roles: dict[str, str] = {}

    for item in layers:
        role = ROLES[item.layer.condition]
        support, note = role.support(ENGINE_RFDIFFUSION1)
        if support == SUPPORT_NONE:
            plan.warnings.append(
                f"{role.label} was dropped: RFdiffusion cannot take it. {note}".strip()
            )
            continue
        selection = resolve_selection(item.selection, target_numbering)
        for tail in selection.tail_notes(role.label.lower()):
            plan.notes.append(f"{role.label}: {tail}")
        if not selection.residues:
            continue
        resolved.setdefault(role.key, [])
        known = {(r.author_chain, r.author_seq) for r in resolved[role.key]}
        for residue in selection.residues:
            if (residue.author_chain, residue.author_seq) not in known:
                resolved[role.key].append(residue)
        seen_roles[role.key] = note

        if support == SUPPORT_RESIDUE:
            widened = [
                r for r in selection.residues if r.atoms and not r.is_whole
            ]
            if widened:
                plan.notes.append(
                    f"{role.label}: chosen atoms on {len(widened)} residue(s) were widened to "
                    "whole residues, which is what RFdiffusion takes."
                )
        elif support == SUPPORT_IMPLIED:
            plan.notes.append(f"{role.label}: {note}")
        elif support == SUPPORT_DOWNSTREAM:
            plan.notes.append(f"{role.label}: {note}")

    keep = resolved.get("keep", [])
    hotspot = resolved.get("hotspot", [])
    if mode == MODE_BINDER and not hotspot:
        plan.errors.append("A binder design needs a Hotspot: the residues the binder should touch.")
    if mode == MODE_MOTIF and not keep:
        plan.errors.append("Motif scaffolding needs a Keep: the residues of the motif.")

    findings = role_findings(
        {
            role: {(r.author_chain, r.author_seq) for r in residues}
            for role, residues in resolved.items()
        },
        names={(r.author_chain, r.author_seq): r.author for rs in resolved.values() for r in rs},
    )
    plan.errors.extend(findings.errors)
    plan.warnings.extend(findings.warnings)
    if plan.errors:
        return plan

    def author_runs(residues: list[ResolvedResidue]) -> list[tuple[str, int, int]]:
        runs: list[tuple[str, int, int]] = []
        by_chain: dict[str, list[int]] = {}
        for r in residues:
            by_chain.setdefault(r.author_chain, []).append(r.author_seq)
        for chain, numbers in by_chain.items():
            for low, high in merge_ranges(numbers):
                runs.append((chain, low, high))
        return runs

    if keep:
        runs = author_runs(keep)
        plan.chains = list(dict.fromkeys(chain for chain, _, _ in runs))
        if mode == MODE_BINDER:
            pieces = [piece(c, a, b) for c, a, b in runs] + [_gap(binder_min, binder_max)]
            plan.contigs = "/0 ".join(pieces)
        else:
            chain_texts: list[str] = []
            linker_index = 0
            for ci, chain in enumerate(plan.chains):
                tokens: list[str] = []
                if ci == 0 and layout.n_term:
                    tokens.append(_gap(*layout.n_term))
                for pi, (c, a, b) in enumerate([r for r in runs if r[0] == chain]):
                    if pi > 0:
                        tokens.append(_gap(*layout.linker(linker_index)))
                        linker_index += 1
                    tokens.append(piece(c, a, b))
                if ci == len(plan.chains) - 1 and layout.c_term:
                    tokens.append(_gap(*layout.c_term))
                chain_texts.append("/".join(tokens))
            plan.contigs = "/0 ".join(chain_texts)
    else:
        plan.chains = list(dict.fromkeys(r.author_chain for r in hotspot))
        plan.errors.append(
            "RFdiffusion needs a Keep to know which part of the target to use; "
            "its contig names the residues explicitly."
        )
        return plan

    plan.hotspot_res = [r.author for r in hotspot]
    fixed = resolved.get("fix_sequence", [])
    by_chain: dict[str, list[int]] = {}
    for r in fixed:
        by_chain.setdefault(r.author_chain, []).append(r.author_seq)
    for chain, numbers in by_chain.items():
        for low, high in merge_ranges(numbers):
            plan.fixed_positions.append(f"{chain}:{_range_text(low, high)}")

    for role_key, residues in resolved.items():
        role = ROLES[role_key]
        plan.summary.append(
            f"{role.label}: {len(residues)} residue(s) -- "
            + " ".join(r.author for r in residues[:8])
            + (f" (+{len(residues) - 8} more)" if len(residues) > 8 else "")
        )
    return plan


__all__ = [
    "BACKBONE_ATOMS",
    "DEFAULT_LINKER",
    "ENGINE_RFD4",
    "LayerInput",
    "MODE_BINDER",
    "MODE_MOTIF",
    "RFdiffusion1Plan",
    "ResolvedResidue",
    "ResolvedSelection",
    "RolePlan",
    "atom_selector",
    "author_identity",
    "compile_rfd4",
    "compile_rfdiffusion1",
    "merge_ranges",
    "reported_label",
    "residue_selector",
    "resolve_selection",
]
