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

"""Read the design conditions carried by an annotated mmCIF.

An *annotated* mmCIF is an ordinary mmCIF whose ``_atom_site`` loop carries
extra columns describing a protein-design problem: which atoms are held at
their coordinates, which residues have a known sequence, which target atoms
the designed binder should engage. RFD4-Proteina consumes and emits this
dialect, and AtomWorks is what writes it.

Columns are named ``{kind}_{name}_{n_body}_{level}`` where kind is one of
``mask``, ``condition`` or ``annotation`` -- so ``mask_hotspot_1_atom`` is a
per-atom boolean marking hotspots. Vector-valued conditions may instead appear
as their own top-level category (``_condition_coordinate_1_atom.val0``).

This module reads that vocabulary without pulling in a full mmCIF library:
Aevom only needs to recognise such a file and say what is conditioned in it.
Writing annotated mmCIF is deliberately not attempted here -- that requires
AtomWorks, and the NIM will build one from a condition spec instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from .mmcif_loop import as_text as _as_text
from .mmcif_loop import atom_site_columns as _atom_site_columns
from .mmcif_loop import iter_atom_site_rows

# The three prefixes AtomWorks uses for the columns it round-trips. A plain
# mmCIF from the PDB has none of them.
CONDITION_PREFIXES: tuple[str, ...] = ("mask_", "condition_", "annotation_")

# Values mmCIF uses for "no value here", plus the sentinels AtomWorks writes
# into integer annotation columns for rows the annotation does not apply to.
# Counting "-1" as a value would report every atom in the file as carrying an
# expandable-segment length.
_NULL_TOKENS = frozenset({".", "?", "nan", "-1"})

# Tokens a boolean mask column uses for "set".
_TRUE_TOKENS = frozenset({"true", "1"})

# Value conditions are inactive at their default: enum conditions such as
# hbond and rasa-binned use 0 for NONE, and the sequence condition writes
# MASKED for residues whose identity is free.
_CONDITION_DEFAULT_TOKENS = _NULL_TOKENS | {"0", "false", "masked"}

# Reading the whole of a large structure to decide whether it is annotated
# wastes time: the _atom_site header block is near the top of the file.
_HEADER_SNIFF_CHARS = 200_000


@dataclass(frozen=True)
class ConditionSummary:
    """One condition column and how much of the structure it covers.

    Attributes:
        column: Full column name as written in the file, e.g.
            ``mask_hotspot_1_atom``.
        name: The condition's own name, e.g. ``hotspot``.
        level: What the condition applies to -- ``atom``, ``residue`` or
            ``token``.
        set_count: Rows whose value is neither null nor false.
    """

    column: str
    name: str
    level: str
    set_count: int

    def describe(self) -> str:
        """One-line, LLM-facing description of this condition."""
        unit = "atoms" if self.level == "atom" else f"{self.level}s"
        return f"{self.name} ({self.set_count} {unit})"


def _is_condition_column(column: str) -> bool:
    return column.startswith(CONDITION_PREFIXES)


def _split_column(column: str) -> tuple[str, str]:
    """Return ``(name, level)`` for a condition column.

    ``mask_hotspot_1_atom`` yields ``("hotspot", "atom")``. Columns that do not
    follow the convention fall back to the whole column name and an unknown
    level, so an unrecognised annotation is still reported rather than dropped.
    """
    body = column.split("_", 1)[1] if "_" in column else column
    parts = body.rsplit("_", 2)
    if len(parts) == 3 and parts[1].isdigit():
        return parts[0], parts[2]
    return body, "unknown"


def condition_columns(text: str) -> list[str]:
    """Return the names of every design-condition column present.

    Covers both the ``_atom_site`` columns and the top-level ``_condition_*``
    categories that vector-valued conditions use.
    """
    found = [c for c in _atom_site_columns(text) if _is_condition_column(c)]
    seen = set(found)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("_condition_"):
            continue
        category = stripped[1:].split(".", 1)[0]
        if category not in seen:
            seen.add(category)
            found.append(category)
    return found


def is_annotated_cif(content: str | bytes) -> bool:
    """Whether this mmCIF carries design-condition columns.

    A plain mmCIF -- from the PDB, from Boltz-2, from any folding model -- has
    none, so this cleanly separates the annotated dialect from the ordinary one.
    """
    text = _as_text(content)
    if not text:
        return False
    return bool(condition_columns(text[:_HEADER_SNIFF_CHARS]))


def summarize_conditions(content: str | bytes) -> list[ConditionSummary]:
    """Report each condition in the file and how many rows it is set on.

    Rows are counted rather than deduplicated: a residue-level column repeats
    its value on every atom of the residue, so the count is in rows of the
    ``_atom_site`` loop. That is enough to tell a conditioned region from an
    empty column, which is what the agent needs in order to describe a file.
    """
    text = _as_text(content)
    if not text:
        return []

    columns = _atom_site_columns(text)
    if not columns:
        return []

    indices = {
        i: column for i, column in enumerate(columns) if _is_condition_column(column)
    }
    if not indices:
        return []

    counts = dict.fromkeys(indices, 0)
    for values in iter_atom_site_rows(text):
        for index, column in indices.items():
            value = values[index].strip().lower()
            if column.startswith("mask_"):
                if value in _TRUE_TOKENS:
                    counts[index] += 1
            elif column.startswith("condition_"):
                if value not in _CONDITION_DEFAULT_TOKENS:
                    counts[index] += 1
            elif value not in _NULL_TOKENS:
                counts[index] += 1

    # AtomWorks derives a condition's active set from its mask column, so when
    # a mask exists it is the authority and the matching value column would
    # only report the same condition a second time.
    masked_names = {
        _split_column(column)[0]
        for column in indices.values()
        if column.startswith("mask_")
    }
    summaries = []
    for index, column in indices.items():
        name, level = _split_column(column)
        if not column.startswith("mask_") and name in masked_names:
            continue
        summaries.append(
            ConditionSummary(
                column=column, name=name, level=level, set_count=counts[index]
            )
        )
    return summaries


def describe(content: str | bytes) -> str:
    """One-line summary of an annotated mmCIF's conditions, for the model.

    Conditions that are present but set on nothing are omitted: an all-false
    column is noise, and AtomWorks writes several of them by default.
    """
    active = [s for s in summarize_conditions(content) if s.set_count]
    if not active:
        return "No design conditions are set in this structure."
    active.sort(key=lambda s: s.set_count, reverse=True)
    return "Design conditions: " + ", ".join(s.describe() for s in active) + "."
