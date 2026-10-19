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

"""Checks on a design's conditions, before GPU time is spent on them.

These do not decide whether a design is allowed -- the NIM does that -- but
whether it is likely to be *worth running*, and say so in terms the model can
act on. Every warning names its cause and what to change.

The hotspot thresholds come from how RFD4-Proteina is trained: a training
example's hotspot is a few atoms (their count is geometric with p=0.2, mean 5)
capped at 10% of the eligible atoms. A hand-picked epitope of fifteen residues
is well outside that distribution; the model will not refuse it, it will just
condition less reliably.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from .structure import StructureAtom

#: Mean number of hotspot atoms in a training example (geometric, p = 0.2).
TYPICAL_HOTSPOT_ATOMS = 5
#: Past this the request is noticeably more than the model usually sees.
HIGH_HOTSPOT_ATOMS = 10
#: Past this fewer than 2% of training examples had as many.
VERY_HIGH_HOTSPOT_ATOMS = 20
#: Training capped hotspots at this fraction of the eligible atoms.
TRAINING_HOTSPOT_CAP = 0.10
#: A binder of 60-120 residues spans roughly this far; hotspots further apart
#: than it cannot all be touched by one.
MAX_EPITOPE_SPAN = 35.0

_SELECTOR = re.compile(r"^sel\('(?P<chain>[^/']+)/\*/(?P<seq>-?\d+)(?:/(?P<atoms>[^']+))?'\)$")


def selector_atoms(
    selector: str, target: Sequence[StructureAtom]
) -> Optional[tuple[str, int, int]]:
    """``(label chain, label seq, atoms conditioned)`` for a hotspot selector.

    A selector naming atoms conditions exactly those; one naming only the
    residue conditions all of that residue's atoms in ``target``. None when the
    selector is not a residue selector this module wrote.
    """
    match = _SELECTOR.match(selector)
    if not match:
        return None
    chain, seq = match["chain"], int(match["seq"])
    named = match["atoms"]
    if named:
        names = [n for n in named.strip("[]").split(",") if n]
        return chain, seq, len(names)
    count = sum(1 for a in target if a.label_chain == chain and a.label_seq == seq)
    return chain, seq, count


def hotspot_warnings(
    selectors: Iterable[str], target: Sequence[StructureAtom]
) -> list[str]:
    """Warnings about a set of hotspot selectors on a target.

    Args:
        selectors: The ``sel('chain/*/seq/atoms')`` strings, label numbering.
        target: The target's atoms, as read from the structure being sent.
    """
    warnings: list[str] = []
    residues: list[tuple[str, int]] = []
    atom_total = 0
    for selector in selectors:
        parsed = selector_atoms(selector, target)
        if parsed is None:
            continue
        chain, seq, count = parsed
        residues.append((chain, seq))
        atom_total += count

    if not residues:
        return warnings

    if atom_total > VERY_HIGH_HOTSPOT_ATOMS:
        warnings.append(
            f"The hotspot conditions {atom_total} atoms in {len(residues)} residues. "
            f"RFD4 was trained on hotspots of about {TYPICAL_HOTSPOT_ATOMS} atoms and "
            "very few had more than "
            f"{VERY_HIGH_HOTSPOT_ATOMS}, so this is outside what it learned and it "
            "will follow it less reliably. Narrow it to the core of the epitope, "
            "or pick single atoms instead of whole residues: right-click atoms in "
            "the viewer, or use select_residues with atoms=[...]."
        )
    elif atom_total > HIGH_HOTSPOT_ATOMS:
        warnings.append(
            f"The hotspot conditions {atom_total} atoms in {len(residues)} residues, "
            f"more than the ~{TYPICAL_HOTSPOT_ATOMS} RFD4 usually sees. It will work, "
            "but a tighter epitope is followed more reliably."
        )

    polymer = [a for a in target if a.entity_type == "polymer"]
    eligible = len(polymer)
    if eligible and atom_total > TRAINING_HOTSPOT_CAP * eligible and atom_total > HIGH_HOTSPOT_ATOMS:
        warnings.append(
            f"That is {atom_total / eligible:.0%} of the target's {eligible} atoms; "
            f"training capped hotspots at {TRAINING_HOTSPOT_CAP:.0%}. Consider cropping "
            "to the domain being bound."
        )

    per_chain: dict[str, set[int]] = {}
    for chain, seq in residues:
        per_chain.setdefault(chain, set()).add(seq)
    for chain, seqs in per_chain.items():
        total = len({a.label_seq for a in polymer if a.label_chain == chain})
        if total and len(seqs) / total > 0.5:
            warnings.append(
                f"The hotspot covers {len(seqs)} of chain {chain}'s {total} residues. "
                "That is most of the chain, not an epitope; the binder cannot be "
                "asked to touch all of it."
            )

    span = _span(residues, target)
    if span is not None and span > MAX_EPITOPE_SPAN:
        warnings.append(
            f"The hotspot residues are up to {span:.0f} A apart. A binder of "
            f"typical size spans about {MAX_EPITOPE_SPAN:.0f} A, so it cannot touch "
            "all of them. Pick one contiguous patch, or run separate designs."
        )
    return warnings


def _span(
    residues: Sequence[tuple[str, int]], target: Sequence[StructureAtom]
) -> Optional[float]:
    """Largest distance between any two hotspot residues, by C-alpha."""
    wanted = set(residues)
    points = [
        (a.x, a.y, a.z)
        for a in target
        if a.name == "CA" and (a.label_chain, a.label_seq) in wanted
    ]
    if len(points) < 2:
        return None
    import numpy as np

    arr = np.array(points, dtype=float)
    diff = arr[:, None, :] - arr[None, :, :]
    return float(np.sqrt((diff**2).sum(axis=2)).max())


# --- Checks across roles ---------------------------------------------------


@dataclass
class RoleFindings:
    """What is wrong (``errors``) or doubtful (``warnings``) about a role set."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _named(keys: Iterable[Any], names: Optional[dict[Any, str]]) -> str:
    labels = sorted(str((names or {}).get(key, key)) for key in keys)
    shown = ", ".join(labels[:6])
    return shown + (f" (+{len(labels) - 6} more)" if len(labels) > 6 else "")


def role_findings(
    roles: dict[str, set], names: Optional[dict[Any, str]] = None
) -> RoleFindings:
    """Check that the roles of a design fit together.

    ``roles`` maps a role key to the set of residue identities it covers (any
    hashable, so label and author numbering both work). These are the rules a
    person would state about a motif or an epitope:

    * Hotspot, Fix position, Fix sequence and Float must sit inside Keep when
      there is one -- they describe residues the design carries, and a residue
      that is not kept is not there to be conditioned (error).
    * Float should sit inside Fix position when one is given: a residue whose
      place is free but whose coordinates are not held has nothing to float
      relative to (warning).

    Args:
        roles: Role key -> residue identities.
        names: Optional identity -> display name, for readable messages.
    """
    out = RoleFindings()
    keep = roles.get("keep")
    if keep:
        for role in ("hotspot", "fix_position", "fix_sequence", "float"):
            outside = roles.get(role, set()) - keep
            if outside:
                label = {
                    "hotspot": "Hotspot",
                    "fix_position": "Fix position",
                    "fix_sequence": "Fix sequence",
                    "float": "Float",
                }[role]
                out.errors.append(
                    f"{label} includes residues outside Keep ({_named(outside, names)}). "
                    f"Keep them too, or choose {label.lower()} residues inside what is kept."
                )
    fixed = roles.get("fix_position")
    floating = roles.get("float")
    if fixed and floating:
        loose = floating - fixed
        if loose:
            out.warnings.append(
                f"Float includes residues whose position is not fixed ({_named(loose, names)}); "
                "they have nothing to float relative to. Add them to Fix position."
            )
    return out
