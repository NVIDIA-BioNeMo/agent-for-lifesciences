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

"""Tools that make, combine and show residue selections.

The agent's counterpart to a user picking residues in the viewer: both end in
the same ``selection`` artifact, so a selection made here can be shown to the
user, extended by them, and handed to a design tool without either side
knowing who made it.

Unlike ``structure_query`` these never wait on a browser. They compute from the
stored structure, so they work with the viewer closed, in an eval, and in
training; the viewer is only told to *show* the result.
"""

import json
import logging
from typing import Any, Optional

from api.selections.ops import (
    OPERATIONS,
    SelectionError,
    combine,
    expand,
    parse_residue_spec,
    select_residues,
)
from api.selections.residues import format_residue_listing, residues_from_atoms
from api.selections.service import store_built_selection
from api.selections.structure import parse_structure_atoms
from api.storage import ArtifactNotFoundError, InvalidReferenceError

from .base import BaseTool, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)

# The viewer is sent every residue of a shown selection; this bounds the event.
MAX_SHOWN_RESIDUES = 2000


def _get_redis() -> Optional[Any]:
    """The shared Redis client, or None when this process has none.

    Imported lazily and tolerant of absence: the artifact is the selection, and
    Redis only makes it visible in the next turn's [SELECTION STATE].
    """
    try:
        from api.redis import get_redis

        return get_redis()
    except (RuntimeError, ImportError):
        return None


class _SelectionTool(BaseTool):
    """Shared plumbing for the selection tools."""

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.BOTH

    @property
    def is_async(self) -> bool:
        return False

    async def _retrieve(self, reference: str, what: str) -> bytes:
        if self._storage is None:
            raise SelectionError(f"Storage service not available; cannot read the {what}.")
        try:
            return await self._storage.retrieve(reference)
        except InvalidReferenceError:
            raise SelectionError(f"Invalid {what} reference: '{reference}'.")
        except ArtifactNotFoundError:
            raise SelectionError(
                f"No {what} found for reference '{reference}'. It may have expired."
            )

    async def _load_selection(self, reference: str) -> dict[str, Any]:
        raw = await self._retrieve(reference, "selection")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            data = None
        if not isinstance(data, dict) or "atoms" not in data:
            raise SelectionError(
                f"'{reference}' is not a selection artifact. Pass a selection_* "
                "reference."
            )
        return data

    async def _load_structure(self, reference: str) -> list:
        raw = await self._retrieve(reference, "structure")
        atoms = parse_structure_atoms(raw.decode("utf-8", errors="replace"))
        if not atoms:
            raise SelectionError(
                f"Could not read any atoms from '{reference}'. Pass a PDB or "
                "mmCIF structure reference."
            )
        return atoms

    async def _store(
        self,
        atoms: list[dict[str, Any]],
        *,
        name: Optional[str],
        parent: Optional[str],
        summary: str,
        scope: str = "residue",
    ) -> dict[str, Any]:
        commit = getattr(self._storage, "commit", None)
        return await store_built_selection(
            atoms,
            storage=self._storage,
            redis_client=_get_redis(),
            session_id=self._session_id or "",
            provenance="agent",
            scope=scope,
            name=name,
            parent=parent,
            summary=summary,
            commit=commit,
        )


def show_event(
    reference: str,
    atoms: list[dict[str, Any]],
    *,
    name: Optional[str],
    focus: bool = True,
    style: str = "highlight",
    colour: Optional[str] = None,
    pdb_ref: Optional[str] = None,
) -> dict[str, Any]:
    """The viewer event that displays a selection.

    Residues travel as *label* ids, which is what the viewer's loci are built
    from; no MolQL is generated or parsed on the way. Residues with only some
    of their atoms selected travel as ``atoms`` (chain, seq, atom name) instead
    of in ``residues``.
    """
    seen: set[tuple[str, int]] = set()
    residues: list[dict[str, Any]] = []
    for atom in atoms:
        residue = atom.get("residue") or {}
        chain, number = residue.get("chainId"), residue.get("seqNumber")
        if chain is None or number is None or (chain, number) in seen:
            continue
        seen.add((chain, number))
        residues.append({"chain": chain, "seq": number})

    # A residue with only some of its atoms selected is shown atom by atom, so
    # the viewer does not light up atoms nobody chose. Whole residues stay
    # residues, which is far fewer things to send and to select.
    partial = {
        (r.label_chain, r.label_seq)
        for r in residues_from_atoms(atoms)
        if r.partial and r.label_seq is not None
    }
    shown_atoms: list[dict[str, Any]] = []
    if partial:
        for atom in atoms:
            residue = atom.get("residue") or {}
            key = (residue.get("chainId"), residue.get("seqNumber"))
            if key in partial and atom.get("name"):
                shown_atoms.append({"chain": key[0], "seq": key[1], "name": atom["name"]})
        residues = [r for r in residues if (r["chain"], r["seq"]) not in partial]

    event: dict[str, Any] = {
        "type": "show_selection",
        "selection_ref": reference,
        "name": name or "",
        "focus": focus,
        "style": style,
        "residues": residues[:MAX_SHOWN_RESIDUES],
    }
    if len(residues) > MAX_SHOWN_RESIDUES:
        event["truncated"] = len(residues)
    if shown_atoms:
        event["atoms"] = shown_atoms[:MAX_SHOWN_RESIDUES]
    if colour:
        event["colour"] = colour
    if pdb_ref:
        event["pdb_ref"] = pdb_ref
    return event


def _lineage_metadata(stored: dict[str, Any], parent: Optional[str]) -> dict[str, Any]:
    """Tool-result metadata the registry copies onto the tool's trace span.

    Provenance, size and parent are what let a trace answer "which selection
    did this design run use, who made it, and what was it derived from".
    """
    meta: dict[str, Any] = {
        "selection_ref": stored["reference"],
        "selection_provenance": "agent",
        "selection_atom_count": stored["atom_count"],
        "selection_residue_count": stored["residue_count"],
    }
    if parent:
        meta["selection_parent"] = parent
    return meta


def _summarise(kind: str, stored: dict[str, Any], name: Optional[str]) -> str:
    label = f" '{name}'" if name else ""
    return (
        f"{kind}{label}: {stored['atom_count']} atoms in {stored['residue_count']} "
        f"residues ({stored['residue_listing']}). Reference: {stored['reference']}. "
        f'Pass selection_ref="{stored["reference"]}" to python_execute, or use it as a '
        "hotspot selection for design."
    )


class SelectResiduesTool(_SelectionTool):
    """Select residues of a stored structure by number."""

    @property
    def name(self) -> str:
        return "select_residues"

    @property
    def summary_template(self) -> str:
        return "Selecting residues..."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "select_residues",
                "description": (
                    "Select specific residues of a structure by chain and number, "
                    "without running a MolQL query. Creates the same kind of "
                    "selection_* reference as a selection the user picks in the "
                    "viewer, and shows it to them. Prefer this over structure_query "
                    "whenever you already know the residue numbers (an epitope you "
                    "were told, a hotspot list); use structure_query when the "
                    "residues are defined by geometry or chemistry.\n\n"
                    "Numbering: the default 'author' numbering is what the viewer "
                    "and the PDB entry display, and what a person means by 'residue "
                    "56'. Use 'label' only when you hold numbers from a design tool, "
                    "whose residue ids are label numbers. For some entries the two "
                    "differ by a constant offset."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "structure_ref": {
                            "type": "string",
                            "description": "PDB or mmCIF structure reference.",
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "chain": {
                            "type": "string",
                            "description": "Chain id, in the same numbering system as 'numbering'.",
                        },
                        "residues": {
                            "type": "array",
                            "items": {"type": ["integer", "string"]},
                            "description": (
                                "Residue numbers and ranges, e.g. [56, \"100-120\", "
                                "\"52A\"]. A trailing letter is an insertion code. "
                                "Follow a residue with ':' and atom names to take only "
                                "those atoms of it: \"56:OH\", \"115:SD\", "
                                "\"123:CZ,OH\", \"100-110:sidechain\"."
                            ),
                        },
                        "numbering": {
                            "type": "string",
                            "enum": ["author", "label"],
                            "description": "Numbering system of chain and residues. Default 'author'.",
                        },
                        "atoms": {
                            "type": ["array", "string"],
                            "items": {"type": "string"},
                            "description": (
                                "Keep only these atoms of every residue listed, as names "
                                "(e.g. [\"CA\"]) or one keyword: 'backbone' (N, CA, C, O) "
                                "or 'sidechain'. A residue written 'N:atoms' uses its own "
                                "atoms instead. Omit for whole residues. Whole residues "
                                "send 8 to 14 atoms to a design; a hotspot is better at "
                                "about 5, so name the atoms that touch the binder."
                            ),
                        },
                        "name": {
                            "type": "string",
                            "description": "Short name shown with the selection, e.g. 'epitope'.",
                        },
                        "show": {
                            "type": "boolean",
                            "description": "Show and focus the selection in the viewer. Default true.",
                        },
                    },
                    "required": ["structure_ref", "chain", "residues"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        structure_ref = (arguments.get("structure_ref") or "").strip()
        chain = str(arguments.get("chain") or "").strip()
        items = arguments.get("residues")
        name = (arguments.get("name") or "").strip() or None
        if not structure_ref or not chain or not items:
            return ToolResult.invalid_arguments(
                "structure_ref, chain and residues are all required."
            )
        if isinstance(items, (int, str)):
            items = [items]
        try:
            spans = parse_residue_spec(items)
            structure = await self._load_structure(structure_ref)
            found = select_residues(
                structure,
                chain,
                spans,
                numbering=arguments.get("numbering") or "author",
                atom_names=arguments.get("atoms") or None,
            )
        except SelectionError as exc:
            return ToolResult.error(str(exc))

        if not found.atoms:
            return ToolResult.error(
                "None of those residues exist in chain "
                f"{chain}: {', '.join(found.missing)}. Nothing was selected."
            )

        payload = [a.to_payload() for a in found.atoms]
        stored = await self._store(
            payload,
            name=name,
            parent=None,
            summary=f"Residues {', '.join(str(s) for s in spans)} of chain {chain}",
            scope="atom" if found.atom_level else "residue",
        )
        text = _summarise("Selected", stored, name)
        if found.missing:
            text += f" Not found in chain {chain}: {', '.join(found.missing)}."

        metadata: dict[str, Any] = _lineage_metadata(stored, None)
        if arguments.get("show", True):
            metadata["visualization_events"] = [
                show_event(stored["reference"], payload, name=name, pdb_ref=structure_ref)
            ]
        return ToolResult.text(text, metadata=metadata)


class CombineSelectionsTool(_SelectionTool):
    """Union, intersect, subtract or grow selections."""

    @property
    def name(self) -> str:
        return "combine_selections"

    @property
    def summary_template(self) -> str:
        return "Combining selections..."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "combine_selections",
                "description": (
                    "Make a new selection from existing selection_* references; the "
                    "inputs are left unchanged and the result records the first as "
                    "its parent. It combines a user's hand-picked residues or "
                    "atoms with your own just as well as two of yours. When "
                    "either input holds chosen atoms rather than whole residues "
                    "it compares atoms, so subtracting one atom leaves the rest "
                    "of its residue; otherwise it compares residues.\n\n"
                    "- union: everything in either selection\n"
                    "- intersect: what is in both selections\n"
                    "- subtract: what is in the first and not in the second\n"
                    "- expand: the first plus every residue within 'radius' angstroms "
                    "of it (needs structure_ref)"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": list(OPERATIONS)},
                        "selection_ref": {
                            "type": "string",
                            "description": "The first selection.",
                            "x-aevom-reference": {"data_type": "selection"},
                        },
                        "other_selection_ref": {
                            "type": "string",
                            "description": "The second selection (union, intersect, subtract).",
                            "x-aevom-reference": {"data_type": "selection"},
                        },
                        "structure_ref": {
                            "type": "string",
                            "description": "Structure to measure in (expand).",
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "radius": {
                            "type": "number",
                            "description": "Angstroms, for expand (0-30).",
                        },
                        "name": {"type": "string", "description": "Short name for the result."},
                        "show": {
                            "type": "boolean",
                            "description": "Show and focus the result in the viewer. Default true.",
                        },
                    },
                    "required": ["operation", "selection_ref"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        operation = (arguments.get("operation") or "").strip()
        first_ref = (arguments.get("selection_ref") or "").strip()
        second_ref = (arguments.get("other_selection_ref") or "").strip()
        structure_ref = (arguments.get("structure_ref") or "").strip()
        name = (arguments.get("name") or "").strip() or None

        if operation not in OPERATIONS:
            return ToolResult.invalid_arguments(
                f"operation must be one of: {', '.join(OPERATIONS)}."
            )
        if not first_ref:
            return ToolResult.invalid_arguments("selection_ref is required.")

        try:
            first = await self._load_selection(first_ref)
            atoms = [a for a in first.get("atoms", [])]
            scope = "atom" if first.get("scope", "atom") == "atom" else "residue"
            if operation == "expand":
                if not structure_ref or arguments.get("radius") is None:
                    return ToolResult.invalid_arguments(
                        "expand needs structure_ref and radius."
                    )
                structure = await self._load_structure(structure_ref)
                try:
                    radius = float(arguments["radius"])
                except (TypeError, ValueError):
                    return ToolResult.invalid_arguments("radius must be a number.")
                result = expand(atoms, structure, radius)
                summary = f"{radius:g} A shell around {first_ref}"
            else:
                if not second_ref:
                    return ToolResult.invalid_arguments(
                        f"{operation} needs other_selection_ref."
                    )
                second = await self._load_selection(second_ref)
                atom_level = scope == "atom" or second.get("scope", "atom") == "atom"
                result = combine(
                    operation, atoms, second.get("atoms", []), atom_level=atom_level
                )
                scope = "atom" if atom_level else "residue"
                summary = f"{first_ref} {operation} {second_ref}"
        except SelectionError as exc:
            return ToolResult.error(str(exc))

        if not result:
            return ToolResult.error(
                "The result is empty, so no selection was created. "
                f"({operation} left no residues.)"
            )

        stored = await self._store(
            result, name=name, parent=first_ref, summary=summary, scope=scope
        )
        metadata: dict[str, Any] = _lineage_metadata(stored, first_ref)
        if arguments.get("show", True):
            metadata["visualization_events"] = [
                show_event(stored["reference"], result, name=name, pdb_ref=structure_ref or None)
            ]
        return ToolResult.text(_summarise(operation.capitalize(), stored, name), metadata=metadata)


class ShowSelectionTool(_SelectionTool):
    """Show an existing selection in the viewer."""

    @property
    def name(self) -> str:
        return "show_selection"

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.INPUT_RETRIEVES

    @property
    def summary_template(self) -> str:
        return "Showing selection..."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "show_selection",
                "description": (
                    "Show an existing selection_* reference in the 3D viewer: select "
                    "its residues and optionally focus the camera. Works for "
                    "selections the user made and ones you made. Use it to point at "
                    "what you are about to analyse or design against."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selection_ref": {
                            "type": "string",
                            "x-aevom-reference": {"data_type": "selection"},
                        },
                        "focus": {
                            "type": "boolean",
                            "description": "Move the camera to the selection. Default true.",
                        },
                        "colour": {
                            "type": "string",
                            "description": "Optional highlight colour as #RRGGBB.",
                        },
                    },
                    "required": ["selection_ref"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        reference = (arguments.get("selection_ref") or "").strip()
        if not reference:
            return ToolResult.invalid_arguments("selection_ref is required.")
        colour = (arguments.get("colour") or "").strip() or None
        if colour and not (len(colour) == 7 and colour.startswith("#")):
            return ToolResult.invalid_arguments("colour must look like #RRGGBB.")
        try:
            data = await self._load_selection(reference)
        except SelectionError as exc:
            return ToolResult.error(str(exc))
        atoms = data.get("atoms", [])
        residues = residues_from_atoms(atoms)
        if not residues:
            return ToolResult.error(f"'{reference}' contains no residues to show.")
        event = show_event(
            reference,
            atoms,
            name=data.get("name"),
            focus=bool(arguments.get("focus", True)),
            colour=colour,
        )
        return ToolResult.text(
            f"Showing {reference} in the viewer: {len(residues)} residues "
            f"({format_residue_listing(residues)}).",
            metadata={"visualization_events": [event]},
        )
