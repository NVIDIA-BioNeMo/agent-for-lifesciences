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

"""Probe interaction-energy grids over a binding pocket.

Answers "where would a donor, an acceptor, or a hydrophobic group most want to
sit here" from the receptor alone -- no ligand, no search. The grid itself is
computed by the Vina service, which owns the scoring function and the OpenBabel
preparation this needs; see ``services/vina/app/probe_grid.py``.

The grid is far too large for the model's context -- a 20 A box at 0.5 A spacing
is 69,000 points per probe -- so it is stored as an ``array`` reference and only
a short ranked list of minima comes back inline. Python code can pull the full
grid with ``aevom.load_array(ref)`` or ``aevom.probe_grid(...)``.
"""

import base64
import binascii
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

PROBE_TYPES = ("C_H", "C_P", "N_P", "N_A", "N_D", "O_A", "O_DA")
DEFAULT_PROBES = ["C_H", "N_D", "O_A"]

# The grid is seconds of numpy plus an OpenBabel preparation, but a large box at
# fine spacing multiplies both, so allow generously more than the typical case.
REQUEST_TIMEOUT = 300.0


class ProbeGridTool(ContainerTool):
    """Map pharmacophore interaction energy over a pocket.

    Places a probe atom of a chosen type at every point of a box and scores it
    against the receptor with AutoDock Vina's inter-molecular function. Minima
    are positions where a ligand atom of that type would be well satisfied.

    Data Flow:
        Input: receptor_pdb_ref, box centre and size
        Output: stores the grid as an ``array`` reference, returns ranked minima
    """

    @property
    def name(self) -> str:
        return "probe_grid"

    @property
    def summary_template(self) -> str:
        return "Mapping pocket interaction energy..."

    def format_summary(self, arguments: dict) -> str:
        center = arguments.get("center", ["?", "?", "?"])
        try:
            where = ", ".join(f"{float(v):.1f}" for v in center)
        except (TypeError, ValueError):
            where = "?"
        return f"Mapping probe interaction energy at ({where})..."

    @property
    def service_url(self) -> str:
        return os.environ.get("VINA_SERVICE_URL", "http://vina:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.BOTH

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "probe_grid",
                "description": (
                    "Map where each type of chemical group would most want to "
                    "bind in a pocket, from the receptor alone.\n\n"
                    "Places a probe atom at every point of a box and scores it "
                    "with AutoDock Vina's interaction function. A minimum in "
                    "the acceptor-probe grid is a position where a ligand "
                    "hydrogen-bond acceptor would be well satisfied -- because "
                    "a receptor donor is pointing at it.\n\n"
                    "WHEN TO USE THIS: to propose pharmacophore points before "
                    "any molecule exists, to decide where to grow or decorate a "
                    "fragment, to check whether a pocket sub-site wants a polar "
                    "or a greasy group, or to rank candidate positions for a "
                    "specific interaction type.\n\n"
                    "WHEN NOT TO USE THIS: to score a molecule you already have "
                    "-- that is vina_dock. To profile the interactions in an "
                    "existing complex -- that is plip_interactions. This tool "
                    "takes no ligand and searches nothing.\n\n"
                    "PROBE TYPES, named for Vina's internal atom types:\n"
                    "- C_H hydrophobic carbon -- greasy contact\n"
                    "- N_D nitrogen donor -- finds sites for a ligand donor\n"
                    "- O_A oxygen acceptor -- finds sites for a ligand acceptor\n"
                    "- N_A, O_DA, C_P, N_P also available\n"
                    "The default C_H/N_D/O_A covers the three pharmacophore "
                    "families that matter most; ask for fewer only if you know "
                    "you need one.\n\n"
                    "OUTPUT: a ranked list of minima with coordinates, type and "
                    "energy in kcal/mol, plus an `array` reference holding the "
                    "full grid. Load that in python_execute with "
                    "aevom.load_array(ref) for custom analysis -- clustering the "
                    "wells, intersecting families, or scoring your own points.\n\n"
                    "INTERPRETING ENERGIES: these are single-atom interaction "
                    "energies, not binding affinities, so they are small. Around "
                    "-1.0 kcal/mol is a strong single-atom site and -0.3 is "
                    "unremarkable. Compare minima with each other, not against "
                    "docking scores."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "receptor_pdb_ref": {
                            "type": "string",
                            "description": (
                                "Reference to the receptor PDB (e.g. 'pdb_abc12345'). "
                                "An apo or ligand-free structure is fine and is the "
                                "usual case for this tool."
                            ),
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "center": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": (
                                "Box centre [x, y, z] in angstrom. Use the pocket "
                                "centre; compute it from pocket residue coordinates "
                                "with python_execute if you do not have it."
                            ),
                        },
                        "size": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 3,
                            "maxItems": 3,
                            "description": (
                                "Box dimensions [x, y, z] in angstrom. Default "
                                "[20, 20, 20], which covers a typical drug-sized "
                                "pocket. Cost grows with the cube of this."
                            ),
                        },
                        "spacing": {
                            "type": "number",
                            "description": (
                                "Grid spacing in angstrom, default 0.5 (AutoDock's "
                                "default). 0.375 is finer and about 2.4x the cost."
                            ),
                        },
                        "probes": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(PROBE_TYPES)},
                            "description": (
                                "Probe types to map. Default "
                                "['C_H', 'N_D', 'O_A']. Extra probes are cheap: "
                                "the distance calculation is shared."
                            ),
                        },
                        "max_minima": {
                            "type": "integer",
                            "description": (
                                "Ranked minima to report per probe, default 5. "
                                "Minima are kept at least 2 A apart, so each one "
                                "is a distinct site rather than a neighbouring "
                                "point in the same well."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": "Optional description for the stored grid.",
                        },
                    },
                    "required": ["receptor_pdb_ref", "center"],
                },
            },
        }

    async def execute(self, arguments: dict, context: dict = None) -> ToolResult:
        receptor_pdb_ref = arguments.get("receptor_pdb_ref", "")
        center = arguments.get("center")
        size = arguments.get("size") or [20.0, 20.0, 20.0]
        spacing = arguments.get("spacing", 0.5)
        probes = arguments.get("probes") or DEFAULT_PROBES
        max_minima = arguments.get("max_minima", 5)
        description = arguments.get("description")

        if not receptor_pdb_ref:
            return ToolResult.error("receptor_pdb_ref is required")
        if not center or len(center) != 3:
            return ToolResult.error(
                "center is required and must be [x, y, z] in angstrom. This is "
                "the centre of the pocket to map."
            )
        unknown = [p for p in probes if p not in PROBE_TYPES]
        if unknown:
            return ToolResult.error(
                f"Unknown probe type(s): {', '.join(unknown)}. "
                f"Available: {', '.join(PROBE_TYPES)}"
            )
        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve the receptor."
            )

        try:
            receptor_pdb = await self._storage.retrieve(receptor_pdb_ref)
            receptor_content = receptor_pdb.decode("utf-8")
        except InvalidReferenceError:
            return ToolResult.error(
                f"Invalid receptor reference: '{receptor_pdb_ref}'. Expected a "
                "PDB reference like 'pdb_abc12345'; use fetch_and_clean_pdb to "
                "get one."
            )
        except ArtifactNotFoundError:
            return ToolResult.error(
                f"Receptor not found: '{receptor_pdb_ref}'. The reference looks "
                "valid but no such artifact exists."
            )

        try:
            response = await self._call_service(
                endpoint="/probe-grid",
                method="POST",
                json_data={
                    "pdb_content": receptor_content,
                    "center": [float(v) for v in center],
                    "size": [float(v) for v in size],
                    "spacing": float(spacing),
                    "probes": list(probes),
                    "max_minima": int(max_minima),
                },
                timeout=REQUEST_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
            logger.exception("Probe grid request failed")
            return ToolResult.error(f"Probe grid failed: {exc}")

        if not response.get("success"):
            return ToolResult.error(
                f"Probe grid failed: {response.get('error', 'unknown error')}"
            )

        try:
            npz_bytes = base64.b64decode(response["npz_base64"], validate=True)
        except (KeyError, binascii.Error, ValueError) as exc:
            return ToolResult.error(f"Probe grid returned an unreadable grid: {exc}")

        minima = response.get("minima") or []
        shape = response.get("shape") or []
        metadata = {
            "source": "probe_grid",
            "receptor_ref": receptor_pdb_ref,
            "probes": response.get("probes"),
            "origin": response.get("origin"),
            "spacing": response.get("spacing"),
            "shape": shape,
            # Named "grid_shape" as well because the array loader reports the
            # numpy shape, which carries the probe axis in front of (nx, ny, nz).
            "grid_shape": [len(response.get("probes") or []), *shape],
            "center": [float(v) for v in center],
            "size": [float(v) for v in size],
            "minima": minima,
            "num_receptor_atoms": response.get("num_receptor_atoms"),
            "num_template_typed": response.get("num_template_typed"),
        }

        grid_ref = await self._storage.store(
            data_type="array",
            content=npz_bytes,
            metadata=metadata,
            description=description
            or f"Probe energy grid for {receptor_pdb_ref} at {tuple(center)}",
        )

        logger.info(
            "Probe grid stored as %s: shape=%s, %d minima",
            grid_ref,
            metadata["grid_shape"],
            len(minima),
        )

        return ToolResult.data_reference(
            reference=grid_ref,
            data_type="array",
            message=self._format_message(grid_ref, response, minima),
            metadata=metadata,
        )

    @staticmethod
    def _format_message(grid_ref: str, response: dict, minima: list) -> str:
        probes = response.get("probes") or []
        shape = response.get("shape") or []
        origin = response.get("origin") or []
        spacing = response.get("spacing")

        message = format_storage_message(
            data_type="array",
            reference=grid_ref,
            source_description="probe interaction-energy grid",
            accepting_tools=["python_execute"],
        )

        points = 1
        for n in shape:
            points *= n
        message += (
            f"\n\nGrid: {' x '.join(str(n) for n in shape)} = {points:,} points "
            f"at {spacing} A spacing, for {len(probes)} probe(s): "
            f"{', '.join(probes)}.\n"
            f"Array shape is ({len(probes)}, {', '.join(str(n) for n in shape)}); "
            f"point (i, j, k) sits at origin + spacing * (i, j, k) with origin "
            f"{[round(float(v), 3) for v in origin]}.\n"
        )

        if minima:
            message += "\nRanked minima (kcal/mol, most favourable first):\n"
            for m in minima:
                message += (
                    f"- {m['family']:<14s} ({m['probe']}) at "
                    f"({m['x']:.2f}, {m['y']:.2f}, {m['z']:.2f}): "
                    f"{m['energy']:.3f}\n"
                )
            message += (
                "\nA family names the ligand atom that belongs at that point: an "
                "Acceptor site is next to a receptor donor. These are "
                "single-atom energies, so compare them with each other rather "
                "than with a docking score."
            )
        else:
            message += (
                "\nNo favourable minima found. The box may be centred off the "
                "pocket, or entirely inside the protein."
            )

        typed = response.get("num_template_typed")
        atoms = response.get("num_receptor_atoms")
        if typed is not None and atoms:
            message += (
                f"\n\nTyped {typed} of {atoms} receptor atoms in range from "
                "residue templates; the rest from the prepared structure."
            )

        return message
