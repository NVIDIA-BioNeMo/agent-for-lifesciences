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

"""The catalog of roles a selection can play in a design.

A selection is only a set of atoms. A *role* says what a design should do
with them: engage them (hotspot), keep them (a motif), hold their
coordinates, hold their sequence, let them float to a new position, or
require a particular chemistry of them. This module is the single place that
vocabulary lives, so the backend compiler, the validators, the language model
and the viewer all read one definition.

Each role carries:

* a plain **label** for the button (what a person would say) and the RFD4
  **annotation** it compiles to, kept in the tooltip so the two vocabularies
  stay linked;
* the **granularity** it accepts and an **atom rule** saying which picks are
  valid for it. The viewer shows a role's button only when the current pick
  satisfies the rule, so a pick that cannot be a pi-stack never offers one;
* how it behaves on each **engine** (RFD4-Proteina, RFdiffusion v1), so a
  role the engine cannot honour is flagged rather than silently dropped.

Roles are *not* the layout (the lengths of the pieces between kept segments)
or the centroid. Those are parameters of the design, not residue sets, and
they are carried separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# --- Engines ---------------------------------------------------------------

ENGINE_RFD4 = "rfd4"
ENGINE_RFDIFFUSION1 = "rfdiffusion1"
DEFAULT_ENGINE = ENGINE_RFD4

#: How an engine treats a role.
SUPPORT_NATIVE = "native"  # expressed directly, at the granularity picked
SUPPORT_RESIDUE = "residue"  # honoured, but atoms are widened to whole residues
SUPPORT_IMPLIED = "implied"  # a consequence of another role (Keep fixes coordinates)
SUPPORT_DOWNSTREAM = "downstream"  # honoured by a later stage, not by this engine
SUPPORT_NONE = "none"  # cannot be honoured; dropped with a warning


@dataclass(frozen=True)
class Engine:
    """What an engine accepts, as far as selections are concerned."""

    key: str
    label: str
    structure_format: str  # "mmcif" or "pdb"
    numbering: str  # "label" (RFD4) or "author" (RFdiffusion)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "structure_format": self.structure_format,
            "numbering": self.numbering,
            "note": self.note,
        }


ENGINES: dict[str, Engine] = {
    ENGINE_RFD4: Engine(
        ENGINE_RFD4,
        "RFD4-Proteina",
        "mmcif",
        "label",
        "Takes a condition spec: every role is expressed directly, at atom level.",
    ),
    ENGINE_RFDIFFUSION1: Engine(
        ENGINE_RFDIFFUSION1,
        "RFdiffusion",
        "pdb",
        "author",
        "Takes a contig string and hotspot residues. Atom-level roles, chemistry "
        "and secondary structure are not available; hotspots are whole residues.",
    ),
}


# --- Atom rules ------------------------------------------------------------
#
# Heavy-atom names only: the picker's atoms and the compiler's both come from
# the structure, and hydrogens are never conditioned on.

#: Aromatic rings, by residue. A residue may offer more than one ring.
RING_GROUPS: dict[str, list[list[str]]] = {
    "PHE": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TYR": [["CG", "CD1", "CD2", "CE1", "CE2", "CZ"]],
    "TRP": [
        ["CG", "CD1", "NE1", "CE2", "CD2"],
        ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    ],
    "HIS": [["CG", "ND1", "CD2", "CE1", "NE2"]],
}

POSITIVE_GROUPS: dict[str, list[list[str]]] = {
    "LYS": [["NZ"]],
    "ARG": [["NE", "NH1", "NH2"]],
}

NEGATIVE_GROUPS: dict[str, list[list[str]]] = {
    "ASP": [["OD1", "OD2"]],
    "GLU": [["OE1", "OE2"]],
}

#: Atoms that can donate a hydrogen bond. Backbone N is a donor in every
#: residue except proline, which has no amide hydrogen.
_BACKBONE_DONOR = ["N"]
HBOND_DONORS: dict[str, list[str]] = {
    "ALA": _BACKBONE_DONOR,
    "ARG": _BACKBONE_DONOR + ["NE", "NH1", "NH2"],
    "ASN": _BACKBONE_DONOR + ["ND2"],
    "ASP": _BACKBONE_DONOR,
    "CYS": _BACKBONE_DONOR,
    "GLN": _BACKBONE_DONOR + ["NE2"],
    "GLU": _BACKBONE_DONOR,
    "GLY": _BACKBONE_DONOR,
    "HIS": _BACKBONE_DONOR + ["ND1", "NE2"],
    "ILE": _BACKBONE_DONOR,
    "LEU": _BACKBONE_DONOR,
    "LYS": _BACKBONE_DONOR + ["NZ"],
    "MET": _BACKBONE_DONOR,
    "PHE": _BACKBONE_DONOR,
    "SER": _BACKBONE_DONOR + ["OG"],
    "THR": _BACKBONE_DONOR + ["OG1"],
    "TRP": _BACKBONE_DONOR + ["NE1"],
    "TYR": _BACKBONE_DONOR + ["OH"],
    "VAL": _BACKBONE_DONOR,
}

_BACKBONE_ACCEPTOR = ["O"]
HBOND_ACCEPTORS: dict[str, list[str]] = {
    "ALA": _BACKBONE_ACCEPTOR,
    "ARG": _BACKBONE_ACCEPTOR,
    "ASN": _BACKBONE_ACCEPTOR + ["OD1"],
    "ASP": _BACKBONE_ACCEPTOR + ["OD1", "OD2"],
    "CYS": _BACKBONE_ACCEPTOR,
    "GLN": _BACKBONE_ACCEPTOR + ["OE1"],
    "GLU": _BACKBONE_ACCEPTOR + ["OE1", "OE2"],
    "GLY": _BACKBONE_ACCEPTOR,
    "HIS": _BACKBONE_ACCEPTOR + ["ND1", "NE2"],
    "ILE": _BACKBONE_ACCEPTOR,
    "LEU": _BACKBONE_ACCEPTOR,
    "LYS": _BACKBONE_ACCEPTOR,
    "MET": _BACKBONE_ACCEPTOR,
    "PHE": _BACKBONE_ACCEPTOR,
    "PRO": _BACKBONE_ACCEPTOR,
    "SER": _BACKBONE_ACCEPTOR + ["OG"],
    "THR": _BACKBONE_ACCEPTOR + ["OG1"],
    "TRP": _BACKBONE_ACCEPTOR,
    "TYR": _BACKBONE_ACCEPTOR + ["OH"],
    "VAL": _BACKBONE_ACCEPTOR,
}

#: Atoms that both donate and accept (a hydroxyl, an imidazole nitrogen).
HBOND_BOTH: dict[str, list[str]] = {
    residue: sorted(set(HBOND_DONORS.get(residue, [])) & set(HBOND_ACCEPTORS.get(residue, [])))
    for residue in HBOND_DONORS
}
HBOND_BOTH = {k: v for k, v in HBOND_BOTH.items() if v}

#: Side-chain carbons that make hydrophobic contacts.
HYDROPHOBIC_ATOMS: dict[str, list[str]] = {
    "ALA": ["CB"],
    "VAL": ["CB", "CG1", "CG2"],
    "LEU": ["CB", "CG", "CD1", "CD2"],
    "ILE": ["CB", "CG1", "CG2", "CD1"],
    "MET": ["CB", "CG", "CE"],
    "PHE": ["CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["CB", "CG", "CD1", "CD2", "CE1", "CE2"],
    "PRO": ["CB", "CG", "CD"],
}

#: Side-chain atoms that can coordinate a metal ion.
METAL_LIGANDS: dict[str, list[str]] = {
    "HIS": ["ND1", "NE2"],
    "CYS": ["SG"],
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "MET": ["SD"],
    "TYR": ["OH"],
    "SER": ["OG"],
    "THR": ["OG1"],
    "ASN": ["OD1"],
    "GLN": ["OE1"],
}

# An atom rule is a dict the viewer evaluates and the compiler applies:
#   {"kind": "narrow", "allowed": {RES: [atoms]}}
#       every picked residue must be of a listed type and have at least one
#       listed atom picked; the role applies to just those atoms.
#   {"kind": "group", "groups": {RES: [[atoms], ...]}}
#       every picked residue must have a complete group picked; the role
#       applies to just that group's atoms (a whole ring, a charged group).
AtomRule = dict[str, Any]


def _narrow(allowed: dict[str, list[str]]) -> AtomRule:
    return {"kind": "narrow", "allowed": allowed}


def _group(groups: dict[str, list[list[str]]]) -> AtomRule:
    return {"kind": "group", "groups": groups}


# --- Roles -----------------------------------------------------------------

GROUP_PRIMARY = "primary"
GROUP_MORE = "more"

VALUE_NONE = "none"
VALUE_ENUM = "enum"  # one of ``variants``; each variant is its own button
VALUE_GROUP = "group"  # a free group label (Float)
VALUE_LAYOUT = "layout"  # the segment layout (Keep)


@dataclass(frozen=True)
class Variant:
    """One member of an enumerated role, offered as its own button."""

    value: str  # the RFD4 enum member name
    label: str
    tooltip: str = ""
    atom_rule: Optional[AtomRule] = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": self.value, "label": self.label, "tooltip": self.tooltip}
        if self.atom_rule is not None:
            out["atom_rule"] = self.atom_rule
        return out


@dataclass(frozen=True)
class Role:
    """A thing a design can be asked to do with a set of atoms."""

    key: str
    label: str
    annotation: str  # the RFD4 condition it compiles to, e.g. "C_HOT"
    tooltip: str
    colour: str  # hex, chosen to stay clear of the slate and green chain colours
    ui_group: str = GROUP_PRIMARY
    #: "residue": atoms are widened to whole residues. "atom": exactly what was
    #: picked. "either": residues stay residues, atoms stay atoms.
    granularity: str = "either"
    value_kind: str = VALUE_NONE
    variants: tuple[Variant, ...] = ()
    atom_rule: Optional[AtomRule] = None
    #: Engine key -> (support level, note).
    engines: dict[str, tuple[str, str]] = field(default_factory=dict)
    #: Whether a Keep layer must contain this role's residues.
    inside_keep: bool = False

    @property
    def is_enum(self) -> bool:
        return self.value_kind == VALUE_ENUM

    def variant(self, value: Any) -> Optional[Variant]:
        text = str(value or "").strip().upper()
        return next((v for v in self.variants if v.value.upper() == text), None)

    def support(self, engine: str) -> tuple[str, str]:
        return self.engines.get(engine, (SUPPORT_NONE, ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "annotation": self.annotation,
            "tooltip": self.tooltip,
            "colour": self.colour,
            "group": self.ui_group,
            "granularity": self.granularity,
            "value_kind": self.value_kind,
            "variants": [v.to_dict() for v in self.variants],
            "atom_rule": self.atom_rule,
            "inside_keep": self.inside_keep,
            "engines": {
                key: {"support": support, "note": note}
                for key, (support, note) in self.engines.items()
            },
        }


_RFD4_ONLY = {
    ENGINE_RFD4: (SUPPORT_NATIVE, ""),
    ENGINE_RFDIFFUSION1: (SUPPORT_NONE, "RFdiffusion has no equivalent; needs RFD4."),
}

ROLES: dict[str, Role] = {
    role.key: role
    for role in (
        Role(
            key="hotspot",
            label="Hotspot",
            annotation="C_HOT",
            tooltip="Hotspot (RFD4 C_HOT): the binder should touch these.",
            colour="#ff8a1f",
            granularity="either",
            inside_keep=True,
            engines={
                ENGINE_RFD4: (SUPPORT_NATIVE, ""),
                ENGINE_RFDIFFUSION1: (
                    SUPPORT_RESIDUE,
                    "RFdiffusion takes whole residues; chosen atoms are widened to their residue.",
                ),
            },
        ),
        Role(
            key="keep",
            label="Keep",
            annotation="contig",
            tooltip=(
                "Keep (RFD4 contig, preserved segment): carry these residues into "
                "the design; everything not kept is dropped."
            ),
            colour="#3b82f6",
            granularity="residue",
            value_kind=VALUE_LAYOUT,
            engines={
                ENGINE_RFD4: (SUPPORT_NATIVE, ""),
                ENGINE_RFDIFFUSION1: (SUPPORT_NATIVE, "A contig segment."),
            },
        ),
        Role(
            key="fix_position",
            label="Fix position",
            annotation="C_CRD",
            tooltip=(
                "Fix position (RFD4 C_CRD): hold these atoms' coordinates in the "
                "design. Choosing side-chain atoms fixes the side chain too."
            ),
            colour="#8b5cf6",
            granularity="either",
            inside_keep=True,
            engines={
                ENGINE_RFD4: (SUPPORT_NATIVE, ""),
                ENGINE_RFDIFFUSION1: (
                    SUPPORT_IMPLIED,
                    "RFdiffusion fixes the backbone of everything it is told to keep.",
                ),
            },
        ),
        Role(
            key="fix_sequence",
            label="Fix sequence",
            annotation="C_SEQ",
            tooltip="Fix sequence (RFD4 C_SEQ): keep these residues' amino-acid identities.",
            colour="#eab308",
            granularity="residue",
            inside_keep=True,
            engines={
                ENGINE_RFD4: (SUPPORT_NATIVE, ""),
                ENGINE_RFDIFFUSION1: (
                    SUPPORT_DOWNSTREAM,
                    "Honoured as fixed positions when ProteinMPNN designs the sequence.",
                ),
            },
        ),
        Role(
            key="float",
            label="Float",
            annotation="C_IDX",
            tooltip=(
                "Float (RFD4 C_IDX false, 'unindex'): keep these residues but let "
                "the design choose where they sit in the sequence. Residues sharing "
                "a group keep their spacing relative to one another."
            ),
            colour="#06b6d4",
            granularity="residue",
            value_kind=VALUE_GROUP,
            inside_keep=True,
            engines=_RFD4_ONLY,
        ),
        Role(
            key="secondary_structure",
            label="Secondary structure",
            annotation="C_SEC",
            tooltip="Secondary structure (RFD4 C_SEC): ask for a helix, sheet or loop here.",
            colour="#f43f5e",
            ui_group=GROUP_MORE,
            granularity="residue",
            value_kind=VALUE_ENUM,
            variants=(
                Variant("HELIX", "Helix", "C_SEC HELIX"),
                Variant("SHEET", "Sheet", "C_SEC SHEET"),
                Variant("LOOP", "Loop", "C_SEC LOOP"),
            ),
            engines=_RFD4_ONLY,
        ),
        Role(
            key="rasa",
            label="Burial",
            annotation="C_RAS",
            tooltip="Burial (RFD4 C_RAS): how exposed to solvent these residues should be.",
            colour="#b45309",
            ui_group=GROUP_MORE,
            granularity="residue",
            value_kind=VALUE_ENUM,
            variants=(
                Variant("BURIED", "Buried", "C_RAS BURIED"),
                Variant("INTERMEDIATE", "Partly buried", "C_RAS INTERMEDIATE"),
                Variant("EXPOSED", "Exposed", "C_RAS EXPOSED"),
            ),
            engines=_RFD4_ONLY,
        ),
        Role(
            key="hbond",
            label="H-bond",
            annotation="C_HB",
            tooltip=(
                "H-bond (RFD4 C_HB): these atoms should donate and/or accept a "
                "hydrogen bond."
            ),
            colour="#ef4444",
            ui_group=GROUP_MORE,
            granularity="atom",
            value_kind=VALUE_ENUM,
            variants=(
                Variant("DONOR", "H-bond donor", "C_HB DONOR", _narrow(HBOND_DONORS)),
                Variant("ACCEPTOR", "H-bond acceptor", "C_HB ACCEPTOR", _narrow(HBOND_ACCEPTORS)),
                Variant("BOTH", "H-bond donor and acceptor", "C_HB BOTH", _narrow(HBOND_BOTH)),
            ),
            engines=_RFD4_ONLY,
        ),
        Role(
            key="pistack",
            label="Pi-stack",
            annotation="C_PISTACK",
            tooltip="Pi-stack (RFD4 C_PISTACK): these aromatic rings should stack.",
            colour="#c026d3",
            ui_group=GROUP_MORE,
            granularity="atom",
            atom_rule=_group(RING_GROUPS),
            engines=_RFD4_ONLY,
        ),
        Role(
            key="pication",
            label="Pi-cation",
            annotation="C_PICATION",
            tooltip="Pi-cation (RFD4 C_PICATION): a ring and a cation should interact.",
            colour="#a21caf",
            ui_group=GROUP_MORE,
            granularity="atom",
            value_kind=VALUE_ENUM,
            variants=(
                Variant("AROMATIC", "Pi-cation ring", "C_PICATION AROMATIC", _group(RING_GROUPS)),
                Variant("CATION", "Pi-cation cation", "C_PICATION CATION", _group(POSITIVE_GROUPS)),
            ),
            engines=_RFD4_ONLY,
        ),
        Role(
            key="salt",
            label="Salt bridge",
            annotation="C_SALT",
            tooltip="Salt bridge (RFD4 C_SALT): a charged group that should pair with an opposite charge.",
            colour="#dc2626",
            ui_group=GROUP_MORE,
            granularity="atom",
            value_kind=VALUE_ENUM,
            variants=(
                Variant("POSITIVE", "Positive charge", "C_SALT POSITIVE", _group(POSITIVE_GROUPS)),
                Variant("NEGATIVE", "Negative charge", "C_SALT NEGATIVE", _group(NEGATIVE_GROUPS)),
            ),
            engines=_RFD4_ONLY,
        ),
        Role(
            key="hydrophobic",
            label="Hydrophobic contact",
            annotation="C_HPHOB",
            tooltip="Hydrophobic contact (RFD4 C_HPHOB): these carbons should pack against a hydrophobe.",
            colour="#92400e",
            ui_group=GROUP_MORE,
            granularity="atom",
            atom_rule=_narrow(HYDROPHOBIC_ATOMS),
            engines=_RFD4_ONLY,
        ),
        Role(
            key="metal",
            label="Metal binding",
            annotation="C_MCOORD",
            tooltip=(
                "Metal binding (RFD4 C_MCOORD COORDINATING): these atoms should "
                "coordinate a metal ion."
            ),
            colour="#78716c",
            ui_group=GROUP_MORE,
            granularity="atom",
            value_kind=VALUE_ENUM,
            variants=(
                Variant("COORDINATING", "Metal-coordinating", "C_MCOORD COORDINATING", _narrow(METAL_LIGANDS)),
            ),
            engines=_RFD4_ONLY,
        ),
    )
}

#: Roles in the order buttons are offered and layers are listed.
ROLE_ORDER: tuple[str, ...] = tuple(ROLES)

#: Annotation (lower-cased) -> role key, so a spec may name either.
_BY_ANNOTATION = {role.annotation.lower(): role.key for role in ROLES.values()}


def get_role(key: Any) -> Optional[Role]:
    """The role named by its key or by its RFD4 annotation, or None."""
    text = str(key or "").strip().lower()
    return ROLES.get(_BY_ANNOTATION.get(text, text))


def role_keys() -> list[str]:
    return list(ROLE_ORDER)


def registry_payload() -> dict[str, Any]:
    """What the viewer and the language model are told the roles are."""
    return {
        "roles": [ROLES[key].to_dict() for key in ROLE_ORDER],
        "engines": [engine.to_dict() for engine in ENGINES.values()],
        "default_engine": DEFAULT_ENGINE,
    }


# --- Applying an atom rule on the server -----------------------------------


def apply_atom_rule(
    rule: Optional[AtomRule], residue_name: str, picked: list[str]
) -> Optional[list[str]]:
    """The atoms a role applies to within one picked residue, or None if invalid.

    Args:
        rule: The role's (or variant's) atom rule; None accepts the pick as is.
        residue_name: Three-letter residue name.
        picked: Names of the atoms picked in the residue.

    Returns:
        The atom names to condition on (the pick, narrowed by the rule), or
        None when the residue cannot play this role at all.
    """
    if rule is None:
        return list(picked)
    name = residue_name.upper()
    have = set(picked)
    if rule["kind"] == "narrow":
        allowed = rule["allowed"].get(name)
        if not allowed:
            return None
        kept = [atom for atom in allowed if atom in have]
        return kept or None
    if rule["kind"] == "group":
        groups = rule["groups"].get(name)
        if not groups:
            return None
        out: list[str] = []
        for group in groups:
            if have.issuperset(group):
                out.extend(atom for atom in group if atom not in out)
        return out or None
    return list(picked)
