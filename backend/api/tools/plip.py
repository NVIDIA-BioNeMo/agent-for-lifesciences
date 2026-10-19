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

"""Protein-ligand interaction profiling via the plip container service.

Docking and co-folding answer "how tightly", and that is the question the tool set
could already answer. This one answers "how" -- which residues the ligand touches
and by what chemistry. That is what a pharmacophore hypothesis is made of, what
tells you which part of a molecule to keep when hopping scaffolds, and what
distinguishes two poses with the same score.

Two design decisions carry the weight:

**One call takes the whole ensemble.** Boltz-2 stores each diffusion sample as
its own reference and Vina reports nine modes. A single pose's interaction list is
one sample from a stochastic process, and reporting it as "the interactions"
dresses up a sampling artifact as a finding. Passing every sample in one call
makes the consensus -- what the poses agree on -- the headline, and the
disagreement visible underneath it.

**mmCIF is handled, not deflected.** PLIP reads PDB and Boltz-2 writes mmCIF, so
the service converts internally. Telling the model to run ``convert_structure``
first would cost one turn per diffusion sample to answer one question. That is a
different judgment from ``convert_structure`` itself, where the conversion is
explicit because its output is a durable artifact; here nothing downstream ever
sees the intermediate.
"""

import logging
import os
from typing import Any

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
)

from .base import ContainerTool, DataIOCategory, ToolResult

logger = logging.getLogger(__name__)

# Matches the service's ceiling. Checked here so an over-large ensemble fails
# with advice rather than a 422 from FastAPI's validation.
MAX_STRUCTURES = 25

# How many rows of each list to render before summarizing. The consensus is the
# answer; the per-structure detail is supporting evidence.
MAX_CONSERVED_ROWS = 30
MAX_VARIABLE_ROWS = 12
MAX_PER_STRUCTURE_ROWS = 10

# Short labels for the rendered table. PLIP's family names are accurate but too
# long to put in a column, and the legend is printed once beneath the table.
SHORT_LABELS: dict[str, str] = {
    "hydrophobic": "hydrophobic",
    "hbond": "H-bond",
    "waterbridge": "water bridge",
    "saltbridge": "salt bridge",
    "pistacking": "pi-stack",
    "pication": "pi-cation",
    "halogen": "halogen",
    "metal": "metal",
}


class PlipInteractionsTool(ContainerTool):
    """Profile protein-ligand interactions and fold a consensus fingerprint.

    Data Flow:
        Input: one or more structure references (PDB or mmCIF)
        Output: per-structure interactions plus a consensus fingerprint
    """

    @property
    def name(self) -> str:
        return "plip_interactions"

    @property
    def summary_template(self) -> str:
        return "Profiling protein-ligand interactions..."

    def format_summary(self, arguments: dict) -> str:
        refs = arguments.get("structure_refs") or []
        if isinstance(refs, str):
            refs = [refs]
        if len(refs) > 1:
            return f"Profiling interactions across {len(refs)} structures..."
        return "Profiling protein-ligand interactions..."

    @property
    def service_url(self) -> str:
        """URL of the PLIP service container."""
        return os.environ.get("PLIP_SERVICE_URL", "http://plip:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        """Retrieves structure references; the answer itself is text."""
        return DataIOCategory.INPUT_RETRIEVES

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "plip_interactions",
                "description": (
                    "Profile the protein-ligand interactions in a complex: which "
                    "residues the ligand contacts and by what chemistry.\n\n"
                    "Detects hydrogen bonds, hydrophobic contacts, pi-stacking, "
                    "pi-cation, salt bridges, halogen bonds, water bridges and "
                    "metal coordination using PLIP's published geometric "
                    "criteria -- angles and protonation, not just distances.\n\n"
                    "WHEN TO USE THIS: to state a pharmacophore hypothesis from "
                    "a structure, to decide which contacts a new scaffold must "
                    "keep, to explain WHY one pose scores better than another, or "
                    "to check that an analog still makes the key interaction. "
                    "Docking scores tell you how tightly; this tells you how.\n\n"
                    "PASS EVERY POSE IN ONE CALL. boltz2_predict returns each "
                    "diffusion sample as its own reference and docking returns "
                    "several modes. Give all of them here and you get a "
                    "consensus: interactions present in most poses are the "
                    "binding mode, ones in a single pose are sampling noise. One "
                    "pose alone cannot tell you which is which.\n\n"
                    "Accepts both PDB and mmCIF references directly -- do NOT "
                    "call convert_structure first.\n\n"
                    "The structure must contain both the protein and the ligand. "
                    "A receptor-only structure has nothing to profile; use "
                    "vina_dock or boltz2_predict to make a complex first."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "structure_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "References to protein-ligand complexes, e.g. "
                                "['mmcif_abc123', 'mmcif_def456'] from "
                                "boltz2_predict or ['pdb_abc123'] from "
                                f"vina_dock. Up to {MAX_STRUCTURES} per call."
                            ),
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "ligand": {
                            "type": "string",
                            "description": (
                                "Optional. Which ligand to profile, as a residue "
                                "name ('E20') or full identifier "
                                "('E20:A:2001'). Defaults to the ligand with the "
                                "most interactions, skipping glycans and "
                                "crystallization additives. Set this when a "
                                "structure holds several ligands or a cofactor."
                            ),
                        },
                        "consensus_threshold": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": (
                                "Fraction of structures an interaction must "
                                "appear in to count as conserved. Default 0.5. "
                                "Raise it for a stricter pharmacophore."
                            ),
                        },
                    },
                    "required": ["structure_refs"],
                },
            },
        }

    async def _collect_structures(
        self, refs: list[str]
    ) -> tuple[list[dict[str, str]], ToolResult | None]:
        """Retrieve each referenced structure.

        Returns:
            ``(structures, error_result)``. When ``error_result`` is not None the
            caller returns it unchanged.
        """
        structures: list[dict[str, str]] = []
        seen: set[str] = set()

        for ref in refs:
            if not isinstance(ref, str) or not ref.strip():
                continue
            ref = ref.strip()
            if ref in seen:
                # The same structure twice would count twice in the consensus
                # denominator and make a single pose look corroborated.
                continue
            seen.add(ref)

            try:
                content = await self._storage.retrieve(ref)
            except InvalidReferenceError:
                return [], ToolResult.error(
                    f"Invalid structure reference: '{ref}'. A reference looks "
                    f"like 'pdb_abc12345' or 'mmcif_abc12345' and comes from "
                    f"vina_dock, boltz2_predict, openfold3_predict or "
                    f"fetch_and_clean_pdb."
                )
            except ArtifactNotFoundError:
                return [], ToolResult.error(
                    f"No structure found for reference '{ref}'. The data may "
                    f"have expired; re-run the tool that produced it."
                )

            text = content.decode("utf-8", errors="replace")
            if not text.strip():
                return [], ToolResult.error(
                    f"The structure stored at '{ref}' is empty."
                )
            structures.append({"name": ref, "content": text})

        return structures, None

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Profile the referenced complexes and report the consensus.

        Args:
            arguments: Dict with 'structure_refs', plus optional 'ligand' and
                'consensus_threshold'.

        Returns:
            ToolResult with the consensus fingerprint and per-structure detail.
        """
        refs = arguments.get("structure_refs") or []
        ligand = (arguments.get("ligand") or "").strip()
        threshold = arguments.get("consensus_threshold")

        if isinstance(refs, str):
            refs = [refs]

        if not refs:
            return ToolResult.error(
                "The 'structure_refs' parameter is required. Pass one or more "
                "protein-ligand complex references, e.g. ['mmcif_abc12345']. "
                "Pass every diffusion sample or docked pose you have -- the "
                "consensus across them is the useful answer."
            )

        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve structures."
            )

        if len(refs) > MAX_STRUCTURES:
            return ToolResult.error(
                f"{len(refs)} structures exceeds the {MAX_STRUCTURES} per call "
                f"limit. Profile the best-scoring {MAX_STRUCTURES} instead."
            )

        if threshold is not None:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                return ToolResult.error(
                    f"'consensus_threshold' must be a number between 0 and 1, "
                    f"got {threshold!r}."
                )
            if not 0.0 <= threshold <= 1.0:
                return ToolResult.error(
                    f"'consensus_threshold' must be between 0 and 1, got "
                    f"{threshold}."
                )

        structures, failure = await self._collect_structures(refs)
        if failure is not None:
            return failure
        if not structures:
            return ToolResult.error(
                "No usable structures were found in 'structure_refs'."
            )

        payload: dict[str, Any] = {"structures": structures}
        if ligand:
            payload["ligand"] = ligand
        if threshold is not None:
            payload["consensus_threshold"] = threshold

        try:
            response = await self._call_service(
                endpoint="/interactions",
                method="POST",
                json_data=payload,
                # PLIP protonates and analyzes each structure serially, roughly a
                # second each, and a large complex is slower.
                timeout=300.0,
            )
        except Exception as e:
            logger.exception("Interaction profiling failed")
            return ToolResult.error(f"Failed to call the PLIP service: {e}")

        if not response.get("success"):
            return ToolResult.error(
                "Interaction profiling failed: "
                + response.get("error", "Unknown error from the PLIP service")
            )

        return self._render(response)

    def _render(self, response: dict[str, Any]) -> ToolResult:
        """Turn the service response into a readable answer plus metadata."""
        profiled = response.get("structures") or []
        consensus = response.get("consensus") or {}
        residues = response.get("consensus_residues") or []
        failed = response.get("failed") or []

        total = consensus.get("total_structures", len(profiled))
        conserved = consensus.get("conserved") or []
        variable = consensus.get("variable") or []
        threshold = consensus.get("threshold", 0.5)

        lines: list[str] = []
        empty = [entry for entry in profiled if not entry.get("fingerprint")]

        if total == 1:
            lines.extend(self._render_single(profiled[0], conserved))
        else:
            lines.append(
                f"{len(conserved)} interaction(s) are conserved across "
                f"{total} structures (present in at least "
                f"{threshold:.0%} of them)."
            )
            lines.append("")
            lines.extend(self._render_consensus_table(conserved, total))

            if variable:
                lines.append("")
                lines.append(
                    "Below the threshold -- these vary between poses and are "
                    "weaker evidence:"
                )
                for entry in variable[:MAX_VARIABLE_ROWS]:
                    lines.append(
                        f"  {self._residue_label(entry)} "
                        f"{SHORT_LABELS.get(entry['family'], entry['family'])} "
                        f"({entry['count']}/{total})"
                    )
                remaining = consensus.get("num_variable", len(variable)) - min(
                    len(variable), MAX_VARIABLE_ROWS
                )
                if remaining > 0:
                    lines.append(f"  ... and {remaining} more")

            lines.append("")
            lines.append("Per structure:")
            for entry in profiled[:MAX_PER_STRUCTURE_ROWS]:
                designated = entry.get("designated")
                where = f"ligand {designated}" if designated else "no ligand found"
                lines.append(
                    f"  {entry['name']}: {len(entry.get('fingerprint', []))} "
                    f"interaction(s), {where}"
                )
            if len(profiled) > MAX_PER_STRUCTURE_ROWS:
                lines.append(
                    f"  ... and {len(profiled) - MAX_PER_STRUCTURE_ROWS} more"
                )

        # Absence, reported. PLIP has a metal-coordination channel and it
        # can only fire when there is a metal in the file. A predicted
        # complex usually has none, so a zinc metalloenzyme is profiled as
        # its apo site and the bond that actually anchors the ligand comes
        # back as ordinary hydrogen bonds to the coordinating histidines.
        # A design decision that then deletes or moves the anchoring group
        # looks, from here, like giving up one contact among five.
        cofactors = sorted(
            {c for entry in profiled for c in (entry.get("cofactors") or [])}
        )
        lines.append("")
        if cofactors:
            lines.append(f"Cofactors present: {', '.join(cofactors)}.")
        else:
            lines.append(
                "No metal ion or cofactor is present in this structure, so "
                "none of the contacts above is a coordination bond. If this "
                "target is a metalloenzyme or a heme protein, the group that "
                "anchors the ligand is coordinating something that is not "
                "here, and will be reported instead as hydrogen bonds or a "
                "salt bridge to the residues that would hold the metal -- "
                "three histidines, or a His/Cys/Asp set. Weigh those "
                "contacts accordingly before moving or deleting the group "
                "that makes them."
            )

        if residues:
            lines.append("")
            lines.append(
                "Contacted residues"
                + (" (conserved only)" if total > 1 else "")
                + ": "
                + ", ".join(
                    f"{entry['residue_name']}{entry['residue_number']}"
                    for entry in residues
                )
            )

        # Which ligand was profiled is not a footnote: an answer about the wrong
        # het group looks exactly as confident as an answer about the right one.
        designations = {
            entry.get("designated")
            for entry in profiled
            if entry.get("designated")
        }
        if designations:
            lines.append("")
            if len(designations) == 1:
                first = next(
                    entry for entry in profiled if entry.get("designated")
                )
                lines.append(
                    f"Ligand profiled: {first['designated']} "
                    f"({first.get('designation_reason', 'designated')})"
                )
                site = self._designated_site(first)
                if site and site.get("smiles"):
                    lines.append(f"Ligand SMILES (as perceived): {site['smiles']}")
            else:
                lines.append(
                    "Different ligands were profiled across the structures: "
                    + ", ".join(sorted(designations))
                    + ". Pass `ligand` to compare the same one everywhere."
                )

        legend = sorted(
            {entry["family"] for entry in conserved}
            | {entry["family"] for entry in variable}
        )
        if legend and total > 1:
            lines.append("")
            lines.append(
                "Families present: "
                + ", ".join(
                    f"{SHORT_LABELS.get(family, family)} = "
                    f"{(response.get('families') or {}).get(family, family)}"
                    for family in legend
                )
            )

        warnings = [
            warning
            for entry in profiled
            for warning in entry.get("warnings") or []
        ]
        if warnings:
            lines.append("")
            for warning in dict.fromkeys(warnings):
                lines.append(f"Note: {warning}")

        if empty and total > 1:
            lines.append("")
            lines.append(
                f"{len(empty)} structure(s) had no interactions at all "
                f"({', '.join(entry['name'] for entry in empty[:5])}). They "
                f"still count in the denominator above."
            )

        if failed:
            lines.append("")
            lines.append(
                f"{len(failed)} structure(s) could not be profiled and were "
                f"excluded:"
            )
            for entry in failed[:5]:
                lines.append(f"  {entry['name']}: {entry['error']}")

        metadata: dict[str, Any] = {
            "num_structures": total,
            "num_failed": len(failed),
            "consensus_threshold": threshold,
            # The fingerprint a later call can compare against: an analog that
            # keeps these has kept the binding mode.
            "consensus_fingerprint": [entry["key"] for entry in conserved],
            "consensus_residues": [
                f"{entry['residue_name']}{entry['residue_number']}"
                for entry in residues
            ],
            "conserved": conserved,
            "variable_count": consensus.get("num_variable", len(variable)),
            "fingerprints": {
                entry["name"]: entry.get("fingerprint", []) for entry in profiled
            },
            "ligands_profiled": sorted(designations),
        }

        return ToolResult.text("\n".join(lines), metadata=metadata)

    @staticmethod
    def _designated_site(entry: dict[str, Any]) -> dict[str, Any] | None:
        """The ligand record for the structure's designated ligand."""
        for ligand in entry.get("ligands") or []:
            if ligand.get("identifier") == entry.get("designated"):
                return ligand
        return None

    @staticmethod
    def _residue_label(entry: dict[str, Any]) -> str:
        """Render a residue as 'TRP84 (chain A)'."""
        return (
            f"{entry['residue_name']}{entry['residue_number']} "
            f"(chain {entry['chain']})"
        )

    @classmethod
    def _render_single(
        cls, entry: dict[str, Any], conserved: list[dict[str, Any]]
    ) -> list[str]:
        """Render one structure: every interaction, grouped by family."""
        site = cls._designated_site(entry)
        if site is None or not site.get("interactions"):
            return [
                f"No protein-ligand interactions were found in {entry['name']}."
            ]

        lines = [
            (
                f"{site['identifier']} makes {site['num_interactions']} "
                f"interaction(s) with {len(site.get('residues', []))} residue(s)."
            )
        ]

        by_family: dict[str, list[dict[str, Any]]] = {}
        for interaction in site["interactions"]:
            by_family.setdefault(interaction["family"], []).append(interaction)

        for interactions in by_family.values():
            lines.append("")
            lines.append(f"{interactions[0]['label'].title()}:")
            for interaction in interactions:
                detail = cls._describe_detail(interaction)
                lines.append(
                    f"  {cls._residue_label(interaction)}"
                    + (f" -- {detail}" if detail else "")
                )

        # A single structure cannot corroborate anything, so say what the
        # fingerprint is for rather than presenting it as a consensus.
        lines.append("")
        lines.append(
            f"Interaction fingerprint ({len(conserved)} entries): "
            + ", ".join(entry["key"] for entry in conserved)
        )
        lines.append(
            "This is one structure, so every interaction is 'conserved' by "
            "definition. Profile several poses together to tell the binding "
            "mode from sampling noise."
        )
        return lines

    @staticmethod
    def _describe_detail(interaction: dict[str, Any]) -> str:
        """Summarize the geometry of one interaction in a few numbers."""
        detail = interaction.get("detail") or {}
        parts: list[str] = []

        for key, label in (
            ("dist", "distance"),
            ("dist_d-a", "donor-acceptor"),
            ("dist_h-a", "H-acceptor"),
            ("centdist", "ring centers"),
            ("dist_a-w", "acceptor-water"),
        ):
            value = detail.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{label} {value:.2f} A")

        angle = detail.get("don_angle") or detail.get("angle")
        if isinstance(angle, (int, float)):
            parts.append(f"angle {angle:.0f} deg")

        offset = detail.get("offset")
        if isinstance(offset, (int, float)):
            parts.append(f"offset {offset:.2f} A")

        if detail.get("protisdon") is True:
            parts.append("protein donates")
        elif detail.get("protisdon") is False:
            parts.append("ligand donates")

        if detail.get("sidechain") is True:
            parts.append("sidechain")
        elif detail.get("sidechain") is False:
            parts.append("backbone")

        if detail.get("lig_group"):
            parts.append(str(detail["lig_group"]))

        return ", ".join(parts)

    @classmethod
    def _render_consensus_table(
        cls, conserved: list[dict[str, Any]], total: int
    ) -> list[str]:
        """Render the conserved interactions as a table ordered by support."""
        if not conserved:
            return [
                (
                    "No interaction met the threshold. The poses disagree about "
                    "the binding mode, which is itself the finding -- either the "
                    "prediction is unreliable or the ligand binds in more than "
                    "one way. Lower consensus_threshold to see what partial "
                    "support exists."
                )
            ]

        rows = [
            [
                f"{entry['residue_name']}{entry['residue_number']}",
                entry["chain"],
                SHORT_LABELS.get(entry["family"], entry["family"]),
                f"{entry['count']}/{total}",
            ]
            for entry in conserved[:MAX_CONSERVED_ROWS]
        ]
        headers = ["residue", "chain", "interaction", "poses"]
        widths = [len(header) for header in headers]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def line(cells: list[str]) -> str:
            return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

        out = [line(headers), "  ".join("-" * width for width in widths)]
        out.extend(line(row) for row in rows)
        if len(conserved) > MAX_CONSERVED_ROWS:
            out.append(
                f"... and {len(conserved) - MAX_CONSERVED_ROWS} more conserved "
                f"interactions"
            )
        return out
