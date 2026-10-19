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

"""SAFE notation conversion tools backed by the RDKit container service.

SAFE (Sequential Attachment-based Fragment Embedding) rewrites a SMILES so that
each fragment of a molecular decomposition is a contiguous substring, separated
by ``.``. The result is still valid SMILES, but generative models trained on SAFE
-- GenMol, SAFE-GPT -- depend on the fragment ordering, so a plain SMILES fed to
them produces worse output without raising an error anywhere. That silent failure
mode is why SAFE is a distinct storage type rather than a flavour of ``smiles``.

These two tools let the model convert without writing Python, so SAFE fits into a
normal tool chain: ``fetch/store SMILES -> smiles_to_safe -> GenMol skill or MCP
tool -> safe_to_smiles -> generate_3d_coordinates``.
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

SUPPORTED_SLICERS = ("brics", "hr", "rotatable", "recap", "mmpa", "attach")


class _SafeServiceTool(ContainerTool):
    """Shared plumbing for the SAFE conversion tools."""

    @property
    def service_url(self) -> str:
        """URL of the RDKit service container, which hosts safe-mol."""
        return os.environ.get("RDKIT_SERVICE_URL", "http://rdkit:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        """Both tools take a reference and store their output."""
        return DataIOCategory.BOTH

    @property
    def is_async(self) -> bool:
        return False

    async def _retrieve(
        self, reference: str, data_type: str
    ) -> tuple[str, dict, ToolResult | None]:
        """Retrieve and decode a stored text artifact.

        Returns:
            ``(content, metadata, error_result)``. When ``error_result`` is not
            None the caller should return it unchanged.
        """
        try:
            content_bytes, metadata = await self._storage.retrieve_with_metadata(
                reference
            )
        except InvalidReferenceError:
            return (
                "",
                {},
                ToolResult.error(
                    f"Invalid {data_type} reference: '{reference}'. A reference looks "
                    f"like '{data_type}_abc12345' and comes from a previous tool call "
                    f"or from aevom.store_{data_type}() in python_execute."
                ),
            )
        except ArtifactNotFoundError:
            return (
                "",
                {},
                ToolResult.error(
                    f"No {data_type} found for reference '{reference}'. Store the "
                    f"{data_type} first, then pass the reference it returns."
                ),
            )

        content = content_bytes.decode("utf-8").strip()
        if not content:
            return (
                "",
                {},
                ToolResult.error(f"Retrieved {data_type} from '{reference}' is empty"),
            )

        return content, (metadata or {}), None


class SmilesToSafeTool(_SafeServiceTool):
    """Encode a stored SMILES as SAFE fragment notation.

    Data Flow:
        Input: SMILES reference (e.g. "smiles_abc123")
        Output: stores a SAFE artifact, returns "safe_xyz789"
    """

    @property
    def name(self) -> str:
        return "smiles_to_safe"

    @property
    def summary_template(self) -> str:
        return "Encoding {smiles_ref} as SAFE notation..."

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "smiles_to_safe",
                "description": (
                    "Convert a stored SMILES string into SAFE (Sequential "
                    "Attachment-based Fragment Embedding) notation and store the "
                    "result.\n\n"
                    "SAFE writes a molecule as a dot-separated sequence of "
                    "fragments. Use it whenever a generative model expects SAFE "
                    "input -- GenMol and SAFE-GPT both do, even when their API "
                    "field is named 'smiles'. Passing plain SMILES to those models "
                    "produces poor results without reporting an error.\n\n"
                    "Set mask_min and mask_max to set up scaffold decoration or "
                    "lead optimization. The molecule's terminal fragment is "
                    "replaced by a generation mask '[*{min-max}]', asking the model "
                    "to invent a new fragment of that many tokens bonded to the "
                    "remaining scaffold. Wider ranges give more diversity; tighter "
                    "ranges keep analog size predictable.\n\n"
                    "The mask must replace a fragment, not sit alongside a whole "
                    "molecule, so masking only works on a molecule that can be "
                    "fragmented. Requesting a mask for one that cannot is an error "
                    "rather than a silent no-op. To build molecules from nothing "
                    "instead, skip this tool and pass a bare mask such as "
                    "'[*{20-30}]' to the generative model.\n\n"
                    "Returns a 'safe_*' reference to pass to the generative tool."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "smiles_ref": {
                            "type": "string",
                            "description": (
                                "Reference to a stored SMILES string (e.g. "
                                "'smiles_abc123')."
                            ),
                            "x-aevom-reference": {"data_type": "smiles"},
                        },
                        "slicer": {
                            "type": "string",
                            "enum": list(SUPPORTED_SLICERS),
                            "description": (
                                "Fragmentation algorithm. 'brics' (the default) is "
                                "what GenMol and SAFE-GPT were trained with. If it "
                                "finds no cut points another slicer is tried "
                                "automatically, so you rarely need to set this."
                            ),
                        },
                        "mask_min": {
                            "type": "integer",
                            "minimum": 1,
                            "description": (
                                "Minimum size of the generation mask. Must be given "
                                "together with mask_max."
                            ),
                        },
                        "mask_max": {
                            "type": "integer",
                            "minimum": 1,
                            "description": (
                                "Maximum size of the generation mask. Must be given "
                                "together with mask_min."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Optional human-readable label for the stored SAFE "
                                "artifact."
                            ),
                        },
                    },
                    "required": ["smiles_ref"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Encode a stored SMILES as SAFE and store the result.

        Args:
            arguments: Dict with 'smiles_ref' and optional 'slicer', 'mask_min',
                'mask_max', 'description'.

        Returns:
            ToolResult with a 'safe_*' data reference, or an error.
        """
        smiles_ref = arguments.get("smiles_ref", "")
        slicer = arguments.get("slicer") or "brics"
        mask_min = arguments.get("mask_min")
        mask_max = arguments.get("mask_max")
        description = arguments.get("description")

        if not smiles_ref:
            return ToolResult.error(
                "smiles_ref is required. Store a SMILES string first (for example "
                "with aevom.store_smiles() in python_execute) and pass the "
                "reference it returns."
            )

        if slicer not in SUPPORTED_SLICERS:
            return ToolResult.error(
                f"Unknown slicer '{slicer}'. Choose one of: "
                f"{', '.join(SUPPORTED_SLICERS)}."
            )

        if (mask_min is None) != (mask_max is None):
            return ToolResult.error(
                "mask_min and mask_max must be provided together. Omit both for a "
                "plain SAFE encoding."
            )

        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve SMILES or store SAFE data."
            )

        try:
            smiles, smiles_metadata, failure = await self._retrieve(
                smiles_ref, "smiles"
            )
            if failure is not None:
                return failure

            payload: dict[str, Any] = {"smiles": smiles, "slicer": slicer}
            if mask_min is not None:
                payload["mask_min"] = mask_min
                payload["mask_max"] = mask_max

            response = await self._call_service(
                endpoint="/smiles-to-safe",
                method="POST",
                json_data=payload,
                timeout=60.0,
            )

            if not response.get("success"):
                error_msg = response.get("error", "Unknown error from RDKit service")
                return ToolResult.error(f"Failed to encode SAFE: {error_msg}")

            safe_string = response.get("safe", "")
            if not safe_string:
                return ToolResult.error("No SAFE string returned from RDKit service")

            num_fragments = response.get("num_fragments")
            fragmented = response.get("fragmented", True)
            masked = response.get("masked", False)
            slicer_used = response.get("slicer_used") or slicer
            replaced_fragment = response.get("replaced_fragment")

            safe_metadata: dict[str, Any] = {
                "smiles": smiles,
                "smiles_ref": smiles_ref,
                "slicer": slicer_used,
                "num_fragments": num_fragments,
                "masked": masked,
            }
            if replaced_fragment:
                safe_metadata["replaced_fragment"] = replaced_fragment
            if masked:
                safe_metadata["mask"] = f"[*{{{mask_min}-{mask_max}}}]"
            for key in ("name", "source", "cid"):
                if key in smiles_metadata:
                    safe_metadata[key] = smiles_metadata[key]

            reference = await self._storage.store(
                data_type="safe",
                content=safe_string.encode("utf-8"),
                metadata=safe_metadata,
                description=description,
            )

            logger.info(
                f"Encoded {smiles_ref} as SAFE {reference} "
                f"({num_fragments} fragments, slicer={slicer_used}, masked={masked})"
            )

            message = format_storage_message(
                data_type="safe",
                reference=reference,
                source_description=f"SAFE encoding of {smiles_ref}",
                accepting_tools=["safe_to_smiles"],
            )
            message += f"\nSAFE string: {safe_string}"
            if slicer_used != slicer:
                message += (
                    f"\nNote: the '{slicer}' slicer found no cut points in this "
                    f"molecule, so '{slicer_used}' was used instead."
                )
            if masked:
                message += (
                    f"\nThe generation mask replaced the terminal fragment "
                    f"'{replaced_fragment}', leaving an open attachment point for "
                    f"the model to build onto. This SAFE string is ready for "
                    f"scaffold decoration."
                )
            elif not fragmented:
                # The model must not assume a mask would decorate this scaffold:
                # masking a one-fragment string makes GenMol echo the input back.
                message += (
                    f"\nNote: no slicer could fragment this molecule, so the SAFE "
                    f"string is a single fragment (the canonical SMILES). It is "
                    f"valid for round-tripping, but it CANNOT be used for scaffold "
                    f"decoration -- there is no attachment point for a generation "
                    f"mask to bond to, and adding one makes GenMol return this same "
                    f"molecule unchanged. For generative work, either start from a "
                    f"larger scaffold or generate de novo from a bare mask."
                )

            return ToolResult.data_reference(
                reference=reference,
                data_type="safe",
                message=message,
                metadata=safe_metadata,
            )

        except Exception as e:
            logger.exception("Failed to encode SAFE")
            return ToolResult.error(f"Failed to call RDKit service: {str(e)}")


class SafeToSmilesTool(_SafeServiceTool):
    """Decode a stored SAFE string back to canonical SMILES.

    Data Flow:
        Input: SAFE reference (e.g. "safe_abc123")
        Output: stores a SMILES artifact, returns "smiles_xyz789"
    """

    @property
    def name(self) -> str:
        return "safe_to_smiles"

    @property
    def summary_template(self) -> str:
        return "Decoding {safe_ref} to SMILES..."

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "safe_to_smiles",
                "description": (
                    "Convert a stored SAFE (fragment notation) string back to "
                    "canonical SMILES and store the result.\n\n"
                    "Use this on SAFE output from a generative model before passing "
                    "the molecule to tools that expect ordinary SMILES, such as "
                    "molecule_info or generate_3d_coordinates.\n\n"
                    "Returns a 'smiles_*' reference."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "safe_ref": {
                            "type": "string",
                            "description": (
                                "Reference to a stored SAFE string (e.g. "
                                "'safe_abc123')."
                            ),
                            "x-aevom-reference": {"data_type": "safe"},
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Optional human-readable label for the stored "
                                "SMILES artifact."
                            ),
                        },
                    },
                    "required": ["safe_ref"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Decode a stored SAFE string to SMILES and store the result.

        Args:
            arguments: Dict with 'safe_ref' and optional 'description'.

        Returns:
            ToolResult with a 'smiles_*' data reference, or an error.
        """
        safe_ref = arguments.get("safe_ref", "")
        description = arguments.get("description")

        if not safe_ref:
            return ToolResult.error(
                "safe_ref is required. Use smiles_to_safe or a generative tool to "
                "produce a SAFE reference first."
            )

        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve SAFE or store SMILES data."
            )

        try:
            safe_string, safe_metadata, failure = await self._retrieve(safe_ref, "safe")
            if failure is not None:
                return failure

            response = await self._call_service(
                endpoint="/safe-to-smiles",
                method="POST",
                json_data={"safe": safe_string},
                timeout=60.0,
            )

            if not response.get("success"):
                error_msg = response.get("error", "Unknown error from RDKit service")
                return ToolResult.error(f"Failed to decode SAFE: {error_msg}")

            smiles = response.get("smiles", "")
            if not smiles:
                return ToolResult.error("No SMILES returned from RDKit service")

            smiles_metadata: dict[str, Any] = {"safe": safe_string, "safe_ref": safe_ref}
            for key in ("name", "source", "cid"):
                if key in safe_metadata:
                    smiles_metadata[key] = safe_metadata[key]

            reference = await self._storage.store(
                data_type="smiles",
                content=smiles.encode("utf-8"),
                metadata=smiles_metadata,
                description=description,
            )

            logger.info(f"Decoded SAFE {safe_ref} to SMILES {reference}")

            message = format_storage_message(
                data_type="smiles",
                reference=reference,
                source_description=f"SMILES decoded from {safe_ref}",
                accepting_tools=["molecule_info", "generate_3d_coordinates"],
            )
            message += f"\nSMILES: {smiles}"

            return ToolResult.data_reference(
                reference=reference,
                data_type="smiles",
                message=message,
                metadata=smiles_metadata,
            )

        except Exception as e:
            logger.exception("Failed to decode SAFE")
            return ToolResult.error(f"Failed to call RDKit service: {str(e)}")
