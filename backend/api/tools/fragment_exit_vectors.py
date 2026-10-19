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

"""Where can a posed fragment be extended, and what could bridge two of them?

SMDD-Bench hands out fragments with no attachment points marked -- generation
strips the dummy atoms, so `fragment_0.sdf` parses to plain `c1ccccc1` -- which
makes choosing the exit vector part of the problem rather than part of the input.
It is a three-dimensional question: which C-H points into open pocket space, in
the direction of its partner, without threading the protein on the way.

This tool answers it against the fragments' own poses, and hands back the two
things needed to act on the answer: the linker size window that spans the gap
under the rotatable-bond gate, and GenMol templates with the mask already placed
between the fragments. The templates are stored as `safe` references so they
chain straight into genmol_generate_molecules.

The geometry lives in ``services/rdkit/app/fragment_link.py``.
"""

import logging
import os
from typing import Any

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from .base import ContainerTool, DataIOCategory, ToolResult

logger = logging.getLogger(__name__)

MAX_FRAGMENTS = 4

# Site enumeration is milliseconds; the library measurement behind the linker
# suggestions embeds a few dozen conformers on the first call of a process.
REQUEST_TIMEOUT = 120.0

# Templates stored as refs, best pairs first. More than a handful is not useful:
# the model picks one pair to commit to, and the rest are there to fall back on.
MAX_STORED_TEMPLATES = 3


class FragmentExitVectorsTool(ContainerTool):
    """Rank attachment points on posed fragments and size the bridge between them.

    Data Flow:
        Input: fragment SDF refs, optional pocket PDB ref, optional hotspots
        Output: stores GenMol templates as ``safe`` references, returns the
            ranked pairs and their geometry inline
    """

    @property
    def name(self) -> str:
        return "fragment_exit_vectors"

    @property
    def summary_template(self) -> str:
        return "Finding fragment attachment points..."

    def format_summary(self, arguments: dict) -> str:
        count = len(arguments.get("fragment_sdf_refs") or [])
        if count == 1:
            return "Finding growth vectors on the fragment..."
        return f"Finding attachment points across {count} fragments..."

    @property
    def service_url(self) -> str:
        return os.environ.get("RDKIT_SERVICE_URL", "http://rdkit:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.BOTH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "fragment_exit_vectors",
                "description": (
                    "Find where posed fragments can be substituted, and what "
                    "kind of linker could join them.\n\n"
                    "Fragments handed out in a fragment-assembly task carry no "
                    "attachment points -- they are valence-complete molecules "
                    "with a pose. Choosing where to attach is a 3D question, and "
                    "this answers it: for every substitutable hydrogen it "
                    "measures the outward direction, how far a substituent can "
                    "grow there before hitting the protein, and how well it "
                    "points at its partner fragment (or, for a single fragment, "
                    "at hotspots you supply from probe_grid).\n\n"
                    "WHEN TO USE THIS: before generating linkers or growing a "
                    "fragment, whenever the fragment's pose has to be kept. Feed "
                    "it the fragment SDFs exactly as given -- their coordinates "
                    "are the input, not a detail.\n\n"
                    "WHEN NOT TO USE THIS: to check a molecule you already built "
                    "-- that is fragment_link, which embeds and scores it. To "
                    "decorate a scaffold with no pose to keep -- that is "
                    "smiles_to_safe plus genmol_generate_molecules.\n\n"
                    "OUTPUT: ranked site pairs with the anchor distance, the "
                    "angle between exit vectors, the clearance along the straight "
                    "line between them, and the linker atom-count window that "
                    "spans the gap. Each of the best pairs also gets a GenMol "
                    "template stored as a `safe` reference with the mask already "
                    "placed between the two fragments -- pass it to "
                    "genmol_generate_molecules as safe_ref.\n\n"
                    "THE ROTATABLE-BOND BUDGET IS USUALLY THE BINDING "
                    "CONSTRAINT: spanning 6-10 A with an alkyl chain spends most "
                    "of an RB <= 10 allowance. The output says what a saturated "
                    "chain would cost and offers rigid bridges -- aryl, alkyne, "
                    "amide, bicyclopentane -- from a measured library when it "
                    "would not fit."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fragment_sdf_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": MAX_FRAGMENTS,
                            "description": (
                                "References to the fragment SDFs (e.g. "
                                "['sdf_abc12345', 'sdf_def67890']), each carrying "
                                "the pose to keep. Two fragments is a linking "
                                "task; one is a growth task."
                            ),
                            "x-aevom-reference": {"data_type": "sdf"},
                        },
                        "pocket_pdb_ref": {
                            "type": "string",
                            "description": (
                                "Reference to the pocket PDB. Strongly "
                                "recommended: without it there is no free depth "
                                "and no clearance, so sites are ranked on "
                                "fragment geometry alone and a vector pointing "
                                "into the protein wall looks as good as one "
                                "pointing into open space."
                            ),
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "target_points": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3,
                            },
                            "description": (
                                "Hotspot coordinates to aim growth vectors at, "
                                "typically probe_grid minima. Used on a "
                                "one-fragment task; with two fragments each one "
                                "aims at the other."
                            ),
                        },
                        "rb_max": {
                            "type": "integer",
                            "description": (
                                "Rotatable-bond ceiling the assembled molecule "
                                "must respect, default 10 (SMDD-Bench's Table 6 "
                                "limit). The headroom left for the linker is this "
                                "minus what the fragments already spend minus the "
                                "two bonds that attach it."
                            ),
                        },
                        "max_pairs": {
                            "type": "integer",
                            "description": "Site pairs to report, best first (default 5).",
                        },
                        "rigid_linkers_only": {
                            "type": "boolean",
                            "description": (
                                "Only offer rigid library bridges. Worth setting "
                                "when the rotatable-bond headroom is tight."
                            ),
                        },
                        "assemble_candidates": {
                            "type": "boolean",
                            "description": (
                                "Also return complete candidate SMILES built from "
                                "the linker library, needing no generative model. "
                                "This is the offline path when GenMol is "
                                "unavailable, and a sanity check on it when it is."
                            ),
                        },
                    },
                    "required": ["fragment_sdf_refs"],
                },
            },
        }

    async def _fetch(self, reference: str, kind: str) -> str:
        """Retrieve a reference as text, or raise with advice for the model."""
        try:
            content = await self._storage.retrieve(reference)
        except InvalidReferenceError:
            raise ValueError(
                f"Invalid {kind} reference: '{reference}'. Expected something like "
                f"'{kind}_abc12345'."
            )
        except ArtifactNotFoundError:
            raise ValueError(
                f"No {kind} found for '{reference}'. The reference is well formed "
                "but nothing is stored under it."
            )
        return content.decode("utf-8", errors="replace")

    async def execute(self, arguments: dict, context: dict = None) -> ToolResult:
        fragment_refs = arguments.get("fragment_sdf_refs") or []
        pocket_ref = arguments.get("pocket_pdb_ref") or ""

        if not fragment_refs:
            return ToolResult.error(
                "fragment_sdf_refs is required: at least one fragment SDF "
                "reference carrying the pose to keep."
            )
        if len(fragment_refs) > MAX_FRAGMENTS:
            return ToolResult.error(
                f"At most {MAX_FRAGMENTS} fragments; got {len(fragment_refs)}."
            )
        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve the fragments."
            )

        try:
            fragment_sdfs = [
                await self._fetch(reference, "sdf") for reference in fragment_refs
            ]
            pocket_pdb = await self._fetch(pocket_ref, "pdb") if pocket_ref else ""
        except ValueError as exc:
            return ToolResult.error(str(exc))

        payload: dict[str, Any] = {
            "fragment_sdfs": fragment_sdfs,
            "pocket_pdb": pocket_pdb,
            "max_pairs": int(arguments.get("max_pairs") or 5),
            "assemble_candidates": bool(arguments.get("assemble_candidates", False)),
            "rigid_linkers_only": bool(arguments.get("rigid_linkers_only", False)),
        }
        if arguments.get("rb_max") is not None:
            payload["rb_max"] = int(arguments["rb_max"])
        if arguments.get("target_points"):
            payload["target_points"] = [
                [float(v) for v in point] for point in arguments["target_points"]
            ]

        try:
            response = await self._call_service(
                endpoint="/link-geometry",
                method="POST",
                json_data=payload,
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
            logger.exception("Link geometry request failed")
            return ToolResult.error(f"Exit-vector geometry failed: {exc}")

        if not response.get("success"):
            return ToolResult.error(
                f"Exit-vector geometry failed: {response.get('error', 'unknown error')}"
            )

        stored = await self._store_templates(response, fragment_refs)
        metadata = {
            "source": "fragment_exit_vectors",
            "fragment_refs": list(fragment_refs),
            "pocket_ref": pocket_ref or None,
            "rb_headroom": response.get("rb_headroom"),
            "rotatable_bonds_used": response.get("rotatable_bonds_used"),
            "pairs": response.get("pairs") or [],
            "growth": response.get("growth"),
            "candidates": response.get("candidates") or [],
            "fragment_separation": response.get("fragment_separation"),
            "template_refs": stored,
            "outputs": stored,
            "all_references": list(stored.values()),
        }

        message = self._format_message(response, stored)
        if stored:
            first = next(iter(stored.values()))
            return ToolResult.data_reference(
                reference=first,
                data_type="safe",
                message=message,
                metadata=metadata,
            )
        return ToolResult.text(message, metadata=metadata)

    async def _store_templates(
        self, response: dict, fragment_refs: list[str]
    ) -> dict[str, str]:
        """Store the best pairs' SAFE templates, keyed by pair id.

        Only the SAFE form is stored, and only when the service managed to build
        it: that is the encoding genmol_generate_molecules accepts as ``safe_ref``.
        The dummy-marker form travels inline in the message, since it is short and
        the NIM's v2 endpoint takes it as a literal string.
        """
        stored: dict[str, str] = {}
        pairs = response.get("pairs") or []
        for pair in pairs[:MAX_STORED_TEMPLATES]:
            templates = pair.get("templates") or {}
            safe_string = templates.get("safe")
            if not safe_string:
                continue
            reference = await self._storage.store(
                data_type="safe",
                content=safe_string.encode("utf-8"),
                metadata={
                    "source": "fragment_exit_vectors",
                    "pair_id": pair.get("pair_id"),
                    "atom_a": pair.get("atom_a"),
                    "atom_b": pair.get("atom_b"),
                    "anchor_distance": pair.get("anchor_distance"),
                    "mask_min": templates.get("mask_min"),
                    "mask_max": templates.get("mask_max"),
                    "fragment_refs": list(fragment_refs),
                },
                description=(
                    f"GenMol linker template for pair {pair.get('pair_id')} "
                    f"spanning {pair.get('anchor_distance')} A"
                ),
            )
            stored[str(pair.get("pair_id"))] = reference
        return stored

    @staticmethod
    def _format_message(response: dict, stored: dict[str, str]) -> str:
        fragments = response.get("fragments") or []
        pairs = response.get("pairs") or []
        growth = response.get("growth")

        lines = [
            f"{len(fragments)} fragment(s), "
            f"{sum(f.get('num_sites', 0) for f in fragments)} substitutable site(s) "
            f"in total. Rotatable bonds: {response.get('rotatable_bonds_used')} used "
            f"by the fragments, {response.get('rb_headroom')} left for a linker under "
            f"RB <= {response.get('rb_max')}."
        ]
        if response.get("pocket_atoms"):
            lines.append(f"Pocket: {response['pocket_atoms']} heavy atoms.")

        for row in response.get("fragment_separation") or []:
            lines.append(
                f"Fragments {row['fragment_a']} and {row['fragment_b']} are "
                f"{row['closest_heavy_distance']} A apart at their closest heavy atoms."
            )

        if pairs:
            lines.append("\nRanked attachment pairs:")
            for pair in pairs:
                clearance = pair.get("tube_clearance")
                lines.append(
                    f"- {pair['pair_id']}: atom {pair['atom_a']} on fragment "
                    f"{pair['fragment_a']} to atom {pair['atom_b']} on fragment "
                    f"{pair['fragment_b']}, {pair['anchor_distance']} A apart, exit "
                    f"vectors {pair['exit_vector_angle']} deg apart"
                    + (
                        f", {clearance} A clearance along the straight line"
                        if clearance is not None
                        else ""
                    )
                    + f". Linker: {pair['linker_atoms_min']}-{pair['linker_atoms_max']} "
                    f"atoms on the path; a saturated chain would cost "
                    f"{pair['saturated_chain_rb_cost']} rotatable bonds."
                )
                if pair.get("requires_rigid_linker"):
                    lines.append(
                        "  A chain that long breaks the rotatable-bond gate, so the "
                        "bridge has to be rigid."
                    )
                offered = pair.get("linkers") or []
                if offered:
                    lines.append(
                        "  Library bridges that span it: "
                        + ", ".join(
                            f"{entry['name']} ({entry['rb_cost']} RB)"
                            for entry in offered[:4]
                        )
                    )
                marker = (pair.get("templates") or {}).get("marker")
                if marker:
                    lines.append(f"  GenMol v2 marker form: {marker}")
                for warning in pair.get("warnings") or []:
                    lines.append(f"  Note: {warning}")

        if growth:
            lines.append("\nGrowth vectors (single fragment, so nothing to link to):")
            for entry in (growth.get("vectors") or [])[:5]:
                depth = entry.get("free_depth")
                lines.append(
                    f"- atom {entry['atom_index']}: "
                    + (
                        f"{depth} A of open space in front of it"
                        if depth is not None
                        else "depth unknown without a pocket"
                    )
                    + f", mask {entry['mask_min']}-{entry['mask_max']} atoms."
                )

        candidates = response.get("candidates") or []
        if candidates:
            lines.append(
                f"\n{len(candidates)} candidate(s) assembled from the linker library, "
                "needing no generative model:"
            )
            for candidate in candidates[:8]:
                lines.append(
                    f"- {candidate['smiles']} (via {candidate['linker']}, "
                    f"pair {candidate['pair_id']})"
                )
            lines.append(
                "Verify these with fragment_link before believing them: the library "
                "matches on span alone and does not know whether the bridge fits."
            )

        if stored:
            lines.append("")
            for pair_id, reference in stored.items():
                lines.append(
                    format_storage_message(
                        data_type="safe",
                        reference=reference,
                        source_description=f"GenMol linker template for {pair_id}",
                        accepting_tools=["genmol_generate_molecules"],
                    )
                )
            lines.append(
                "Pass one of these as safe_ref to genmol_generate_molecules; the "
                "mask is already sized and placed between the fragments. Then verify "
                "what comes back with fragment_link, which is the step that decides "
                "whether the pose survives."
            )

        for warning in response.get("warnings") or []:
            lines.append(f"\nWarning: {warning}")

        return "\n".join(lines)
