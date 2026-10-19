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

"""RFDiffusion backbone design via NVIDIA's hosted IPD NIM.

Generates protein backbone coordinates for three design modes -- de novo, motif
scaffolding, and binder design -- all selected through the ``contigs`` DSL rather
than separate parameters. The designed backbone is stored as a ``pdb`` reference,
which chains into ``proteinmpnn_design_sequence`` to put a sequence on it.

The detailed guidance for choosing contigs lives in the upstream
``rfdiffusion-nim`` skill; see ``api.skills.binding``.
"""

import logging
from typing import Any, Optional

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from api.selections.design_spec import SpecError, parse_design_spec
from api.selections.roles import ENGINE_RFDIFFUSION1

from .base import DataIOCategory, NvidiaApiError, NvidiaApiTool, ToolResult
from .spec_plan import plan_spec

logger = logging.getLogger(__name__)

RFDIFFUSION_API_URL = (
    "https://health.api.nvidia.com/v1/biology/ipd/rfdiffusion/generate"
)

# The endpoint rejects any request without an input structure, including a de
# novo design that by definition has nothing to build on. Sending a single
# floating alpha carbon satisfies the check without constraining the result --
# the contigs string carries the whole specification in that mode.
DUMMY_INPUT_PDB = (
    "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1           1\n"
    "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
    "END\n"
)

MIN_DIFFUSION_STEPS = 1
MAX_DIFFUSION_STEPS = 50
DEFAULT_DIFFUSION_STEPS = 50

# RFDiffusion returns in seconds; a minute of gateway polling is generous and
# fails fast when the service is unhealthy rather than hanging the whole turn.
POLL_SECONDS = 120


class RFDiffusionTool(NvidiaApiTool):
    """Design a protein backbone with RFDiffusion.

    Data Flow:
        Input: optional pdb_ref (target or motif structure), contigs DSL string
        Output: Stores the designed backbone PDB, returns a reference
    """

    @property
    def name(self) -> str:
        return "rfdiffusion_design_backbone"

    @property
    def summary_template(self) -> str:
        return "Designing a protein backbone with RFDiffusion..."

    def format_summary(self, arguments: dict) -> str:
        contigs = arguments.get("contigs", "")
        if arguments.get("hotspot_res"):
            return f"Designing a binder backbone ({contigs})..."
        if arguments.get("pdb_ref"):
            return f"Scaffolding a backbone around {arguments['pdb_ref']} ({contigs})..."
        if contigs:
            return f"Designing a de novo backbone ({contigs})..."
        return "Designing a protein backbone with RFDiffusion..."

    @property
    def api_endpoint(self) -> str:
        return RFDIFFUSION_API_URL

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
                "name": "rfdiffusion_design_backbone",
                "description": (
                    "Design a new protein backbone structure with RFDiffusion. "
                    "Generates 3D coordinates only — the result has no amino acid "
                    "sequence yet.\n\n"
                    "READ THE SKILL FIRST: call "
                    'load_skill(skill_id="rfdiffusion-nim") before using this tool '
                    "for anything beyond a plain de novo design. The `contigs` "
                    "argument is a small DSL and is the only thing you must get "
                    "right; the skill documents it in full, along with when to use "
                    "each design mode and how to pick hotspots.\n\n"
                    "THREE DESIGN MODES, all selected through `contigs`:\n"
                    "- De novo: omit pdb_ref, contigs=\"80-120\" (generate a "
                    "protein of 80-120 residues from nothing)\n"
                    "- Motif scaffolding: pass pdb_ref, contigs=\"A25-35/0 50-80\" "
                    "(keep chain A residues 25-35 from the input, then generate "
                    "50-80 new ones)\n"
                    "- Binder design: pass pdb_ref for the target, "
                    "contigs=\"A1-100/0 50-100\", hotspot_res=[\"A50\", \"A51\"] "
                    "(design a binder that contacts those target residues)\n\n"
                    "WORKFLOW: this tool produces a backbone, not a usable "
                    "protein. Follow it with proteinmpnn_design_sequence to design "
                    "sequences for the backbone, then openfold3 to check that the "
                    "designed sequence folds back to the intended shape.\n\n"
                    "Takes about 5-30 seconds depending on diffusion_steps."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "contigs": {
                            "type": "string",
                            "description": (
                                "Design specification. \"100\" generates exactly "
                                "100 residues; \"80-120\" samples a length in that "
                                "range; \"A25-35\" keeps chain A residues 25-35 "
                                "from pdb_ref; \"/0\" inserts a chain break, so "
                                "\"A25-35/0 50-80\" keeps that motif and generates "
                                "50-80 new residues after it. Load the "
                                "rfdiffusion-nim skill for the full DSL."
                            ),
                        },
                        "pdb_ref": {
                            "type": "string",
                            "description": (
                                "Reference to a stored input structure, for motif "
                                "scaffolding or binder design. OMIT THIS for de "
                                "novo design — the tool handles the placeholder "
                                "structure the API requires. Chain IDs used in "
                                "contigs and hotspot_res must exist in this "
                                "structure."
                            ),
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "design_spec_ref": {
                            "type": "string",
                            "description": (
                                "A design_spec_* from create_design_spec, in place of "
                                "contigs, pdb_ref and hotspot_res: the roles the user "
                                "assigned (Hotspot, Keep) are compiled to those, in "
                                "the structure's own residue numbers. Make the spec "
                                "with engine='rfdiffusion1' and a PDB target. Roles "
                                "RFdiffusion cannot take (Float, chemistry, secondary "
                                "structure) are reported as dropped."
                            ),
                            "x-aevom-reference": {"data_type": "design_spec"},
                        },
                        "binder_length_min": {
                            "type": "integer",
                            "description": "With design_spec_ref: shortest binder, in residues.",
                        },
                        "binder_length_max": {
                            "type": "integer",
                            "description": "With design_spec_ref: longest binder, in residues.",
                        },
                        "hotspot_res": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Target residues the designed binder must contact, "
                                "as chain+number strings like [\"A50\", \"A51\"]. "
                                "Binder design only; requires pdb_ref."
                            ),
                        },
                        "diffusion_steps": {
                            "type": "integer",
                            "description": (
                                f"Denoising iterations, {MIN_DIFFUSION_STEPS}-"
                                f"{MAX_DIFFUSION_STEPS} (default "
                                f"{DEFAULT_DIFFUSION_STEPS}). Fewer is faster and "
                                "lower quality; leave at the default unless "
                                "iterating quickly."
                            ),
                        },
                        "random_seed": {
                            "type": "integer",
                            "description": (
                                "Fixed seed for a reproducible design. Omit to get "
                                "a different backbone on each call."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short label for the designed backbone (e.g. "
                                "'de novo 100aa backbone', 'PD-L1 binder')."
                            ),
                        },
                    },
                    # contigs, or design_spec_ref; execute() says which is missing.
                    "required": [],
                },
            },
        }

    async def _resolve_pdb(self, pdb_ref: str) -> str:
        """Retrieve stored structure text by reference."""
        if self._storage is None:
            raise ValueError(
                "Storage service not available. Cannot retrieve structure data."
            )
        try:
            content = await self._storage.retrieve(pdb_ref)
        except InvalidReferenceError:
            raise ValueError(
                f"Invalid structure reference: '{pdb_ref}'. Use fetch_pdb_structure "
                "or another structure tool first, or omit pdb_ref for de novo design."
            )
        except ArtifactNotFoundError:
            raise ValueError(
                f"Structure not found for reference '{pdb_ref}'. The data may have "
                "expired or the reference is invalid."
            )
        return content.decode("utf-8", errors="replace")

    @staticmethod
    def _validate_diffusion_steps(value: Any) -> int:
        if value is None:
            return DEFAULT_DIFFUSION_STEPS
        try:
            steps = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"diffusion_steps must be an integer between "
                f"{MIN_DIFFUSION_STEPS} and {MAX_DIFFUSION_STEPS}."
            )
        if not MIN_DIFFUSION_STEPS <= steps <= MAX_DIFFUSION_STEPS:
            raise ValueError(
                f"diffusion_steps must be between {MIN_DIFFUSION_STEPS} and "
                f"{MAX_DIFFUSION_STEPS}, got {steps}."
            )
        return steps

    @staticmethod
    def _count_residues(pdb_text: str) -> int:
        """Number of residues in a backbone, counted from its CA atoms."""
        return sum(
            1
            for line in pdb_text.splitlines()
            if line.startswith("ATOM") and line[12:16].strip() == "CA"
        )

    async def _compile_spec(self, spec_ref: str, arguments: dict[str, Any]):
        """A design spec as this tool's arguments.

        Returns ``(contigs, hotspot_res, pdb_ref, input_pdb, notes,
        fixed_positions)``, or a :class:`ToolResult` error when the spec cannot
        be run by RFdiffusion.
        """
        if self._storage is None:
            raise ValueError("Storage service not available. Cannot retrieve the design spec.")

        async def retrieve(reference: str, what: str) -> bytes:
            try:
                return await self._storage.retrieve(reference)
            except InvalidReferenceError:
                raise ValueError(f"Invalid {what} reference: '{reference}'.")
            except ArtifactNotFoundError:
                raise ValueError(
                    f"No {what} found for reference '{reference}'. It may have expired."
                )

        spec = parse_design_spec(await retrieve(spec_ref, "design spec"))
        if (spec.engine or ENGINE_RFDIFFUSION1) != ENGINE_RFDIFFUSION1:
            return ToolResult.error(
                f"The spec is for {spec.engine}, not RFdiffusion. Make a spec with "
                "engine='rfdiffusion1', or use design_protein_binder for RFD4."
            )
        minimum = arguments.get("binder_length_min")
        maximum = arguments.get("binder_length_max")
        lengths: dict[str, int] = {}
        if minimum is not None or maximum is not None:
            low = int(minimum if minimum is not None else maximum)
            high = int(maximum if maximum is not None else minimum)
            if low > high:
                raise ValueError(
                    f"binder_length_min ({low}) cannot exceed binder_length_max ({high})."
                )
            lengths = {"binder_min": low, "binder_max": high}
        plan = await plan_spec(spec, retrieve, engine=ENGINE_RFDIFFUSION1, **lengths)
        compiled = plan.rfdiffusion1
        if compiled is None or compiled.errors:
            return ToolResult.error(
                "The design spec cannot be run:\n"
                + "\n".join(f"- {e}" for e in (compiled.errors if compiled else []))
            )
        if plan.target_cif.lstrip().startswith("data_"):
            return ToolResult.error(
                f"'{spec.target_ref}' is mmCIF, and RFdiffusion reads PDB. Convert "
                "the target with convert_structure, make a new spec on the PDB, "
                "and keep its residue numbers (a PDB has author numbering only)."
            )
        notes = list(compiled.notes) + list(compiled.warnings)
        return (
            compiled.contigs,
            list(compiled.hotspot_res),
            spec.target_ref,
            plan.target_cif,
            notes,
            list(compiled.fixed_positions),
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Design a backbone and store the resulting PDB."""
        contigs = (arguments.get("contigs") or "").strip()
        pdb_ref = arguments.get("pdb_ref") or ""
        hotspot_res = arguments.get("hotspot_res") or []
        description = arguments.get("description")
        spec_ref = (arguments.get("design_spec_ref") or "").strip()
        spec_notes: list[str] = []
        fixed_positions: list[str] = []
        spec_input_pdb: Optional[str] = None

        if spec_ref:
            if contigs or hotspot_res or pdb_ref:
                return ToolResult.error(
                    "Pass design_spec_ref alone: it carries the contigs, the "
                    "target and the hotspots. To change them, make a new spec."
                )
            try:
                compiled = await self._compile_spec(spec_ref, arguments)
            except (ValueError, SpecError) as exc:
                return ToolResult.error(str(exc))
            if isinstance(compiled, ToolResult):
                return compiled
            contigs, hotspot_res, pdb_ref, spec_input_pdb, spec_notes, fixed_positions = compiled

        if not contigs:
            return ToolResult.error(
                "contigs is required — it specifies what to generate. Use "
                '"80-120" for a de novo protein of 80-120 residues, or load the '
                "rfdiffusion-nim skill for the full syntax."
            )

        try:
            steps = self._validate_diffusion_steps(arguments.get("diffusion_steps"))

            if hotspot_res and not pdb_ref:
                raise ValueError(
                    "hotspot_res names residues on a target structure, so it "
                    "requires pdb_ref. Omit hotspot_res for de novo design."
                )

            de_novo = not pdb_ref
            input_pdb = (
                spec_input_pdb
                if spec_input_pdb is not None
                else DUMMY_INPUT_PDB if de_novo else await self._resolve_pdb(pdb_ref)
            )

            request_data: dict[str, Any] = {
                "input_pdb": input_pdb,
                "contigs": contigs,
                "diffusion_steps": steps,
            }
            if hotspot_res:
                request_data["hotspot_res"] = list(hotspot_res)
            if arguments.get("random_seed") is not None:
                request_data["random_seed"] = int(arguments["random_seed"])

            logger.info(
                "Calling RFDiffusion: contigs=%s steps=%d de_novo=%s hotspots=%d",
                contigs,
                steps,
                de_novo,
                len(hotspot_res),
            )

            result = await self._call_nvidia_api(
                request_data,
                poll_seconds=POLL_SECONDS,
                service_name="RFDiffusion",
            )

            output_pdb = result.get("output_pdb")
            if not output_pdb:
                return ToolResult.error(
                    "RFDiffusion returned no structure. Check that the chain IDs "
                    "in contigs match the input structure."
                )

            if self._storage is None:
                return ToolResult.error(
                    "Storage service not available. Cannot store the designed backbone."
                )

            num_residues = self._count_residues(output_pdb)
            mode = (
                "de_novo"
                if de_novo
                else ("binder_design" if hotspot_res else "motif_scaffolding")
            )
            storage_metadata: dict[str, Any] = {
                "source": "rfdiffusion",
                "design_mode": mode,
                "contigs": contigs,
                "diffusion_steps": steps,
                "num_residues": num_residues,
                "sequence_designed": False,
            }
            if hotspot_res:
                storage_metadata["hotspot_res"] = list(hotspot_res)
            if not de_novo:
                storage_metadata["input_pdb_ref"] = pdb_ref
            if spec_ref:
                storage_metadata["design_spec_ref"] = spec_ref
            if fixed_positions:
                storage_metadata["fixed_positions"] = list(fixed_positions)
            if result.get("elapsed_ms") is not None:
                storage_metadata["elapsed_ms"] = result["elapsed_ms"]

            if not description:
                description = f"RFDiffusion {mode.replace('_', ' ')} backbone"

            reference = await self._storage.store(
                data_type="pdb",
                content=output_pdb.encode("utf-8"),
                metadata=storage_metadata,
                description=description,
            )
            logger.info("Stored designed backbone as %s (%d residues)", reference, num_residues)

            message = format_storage_message(
                data_type="pdb",
                reference=reference,
                source_description=description,
                accepting_tools=["proteinmpnn_design_sequence", "visualize_pdb"],
            )
            message += (
                f"\n\nDesign summary:\n"
                f"- Mode: {mode.replace('_', ' ')}\n"
                f"- Residues generated: {num_residues}\n"
                f"- Contigs: {contigs}\n"
                f"- Diffusion steps: {steps}\n\n"
                "This backbone has coordinates but no sequence — every residue is "
                "glycine. Run proteinmpnn_design_sequence on it to design real "
                "sequences, then openfold3 to verify one folds back to this shape."
            )
            if fixed_positions:
                message += (
                    "\n\nThe spec asks for these positions to keep their amino "
                    f"acids: {', '.join(fixed_positions)} (author numbering). "
                    "proteinmpnn_design_sequence has no fixed-positions option, so "
                    "check them in the designed sequences and say so if they "
                    "differ."
                )
            if spec_notes:
                message += "\n\nFrom the design spec:\n" + "\n".join(
                    f"- {note}" for note in spec_notes
                )

            return ToolResult.data_reference(
                reference=reference,
                data_type="pdb",
                message=message,
                metadata=storage_metadata,
            )

        except ValueError as exc:
            logger.error("RFDiffusion validation error: %s", exc)
            return ToolResult.error(str(exc))
        except NvidiaApiError as exc:
            logger.error("RFDiffusion API error: %s", exc)
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool must not break the turn
            logger.exception("Failed to design backbone with RFDiffusion")
            return ToolResult.error(f"Failed to design backbone: {exc}")
