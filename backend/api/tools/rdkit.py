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

"""RDKit tool for molecular operations via container service.

This tool communicates with the RDKit container service to perform
molecular operations like getting molecule information.
"""

import logging
import os
from typing import Any

from api.storage import ArtifactNotFoundError, InvalidReferenceError

from .base import ContainerTool, DataIOCategory, ToolResult

logger = logging.getLogger(__name__)


class RDKitMoleculeInfoTool(ContainerTool):
    """Get molecular properties and information from a stored SMILES reference.

    This tool retrieves a SMILES string from the data store and returns
    properties like molecular weight, formula, LogP, etc.

    Data Flow:
        Input: SMILES reference (e.g., "smiles_abc123") from aevom.store_smiles()
        Output: Text with molecular properties (no data stored)
    """

    @property
    def name(self) -> str:
        return "molecule_info"

    @property
    def summary_template(self) -> str:
        return "Analyzing molecular properties of {smiles_ref}..."

    @property
    def service_url(self) -> str:
        """URL of the RDKit service container."""
        return os.environ.get("RDKIT_SERVICE_URL", "http://rdkit:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        """This tool retrieves input data from references."""
        return DataIOCategory.INPUT_RETRIEVES

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "molecule_info",
                "description": (
                    "Get detailed molecular properties and information from a stored SMILES reference.\n\n"
                    "Returns properties like molecular weight, formula, LogP, TPSA, "
                    "number of hydrogen bond donors/acceptors, rotatable bonds, etc.\n\n"
                    "WORKFLOW:\n"
                    "1. First, use python_execute to look up a molecule (e.g., from PubChem) and "
                    "store its SMILES using aevom.store_smiles()\n"
                    "2. Then call this tool with the returned smiles_ref\n\n"
                    "EXAMPLE PYTHON CODE (for python_execute):\n"
                    "  import pubchempy as pcp\n"
                    "  compound = pcp.get_compounds('caffeine', 'name')[0]\n"
                    "  smiles_ref = aevom.store_smiles(compound.canonical_smiles, description='Caffeine')\n"
                    "  print(f'Stored SMILES as: {smiles_ref}')\n\n"
                    "Then call: molecule_info(smiles_ref=smiles_ref)"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "smiles_ref": {
                            "type": "string",
                            "description": (
                                "Reference to a stored SMILES string (e.g., 'smiles_abc123'). "
                                "Obtain this by calling aevom.store_smiles() in python_execute."
                            ),
                            "x-aevom-reference": {"data_type": "smiles"},
                        },
                    },
                    "required": ["smiles_ref"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Get molecule information from a stored SMILES reference.

        Args:
            arguments: Dict containing 'smiles_ref'.

        Returns:
            ToolResult with molecule info or error.
        """
        smiles_ref = arguments.get("smiles_ref", "")

        if not smiles_ref:
            return ToolResult.error(
                "smiles_ref is required. Use python_execute with aevom.store_smiles() "
                "to store a SMILES string first, then pass the returned reference here."
            )

        # Validate storage service is available
        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve SMILES data."
            )

        try:
            # 1. Retrieve the SMILES from storage
            logger.info(f"Retrieving SMILES from {smiles_ref}")
            try:
                smiles_content, metadata = await self._storage.retrieve_with_metadata(
                    smiles_ref
                )
                smiles = smiles_content.decode("utf-8").strip()
            except InvalidReferenceError:
                return ToolResult.error(
                    f"Invalid SMILES reference: '{smiles_ref}'. "
                    "Use python_execute with aevom.store_smiles() to store a SMILES first, "
                    "which returns a reference like 'smiles_abc12345'."
                )
            except ArtifactNotFoundError:
                return ToolResult.error(
                    f"SMILES not found for reference '{smiles_ref}'. "
                    "Use python_execute with aevom.store_smiles() to store a SMILES first."
                )

            if not smiles:
                return ToolResult.error(
                    f"Retrieved SMILES from '{smiles_ref}' is empty"
                )

            logger.info(f"Retrieved SMILES: {smiles[:50]}...")

            # 2. Call RDKit service to get molecule info
            response = await self._call_service(
                endpoint="/molecule-info",
                method="POST",
                json_data={"smiles": smiles},
                timeout=30.0,
            )

            if response.get("success"):
                info = response.get("info", {})

                # Include description from metadata if available
                description = metadata.get("description", "")
                header = (
                    f"Molecular Information for {description}"
                    if description
                    else "Molecular Information"
                )
                header += f" ({info.get('canonical_smiles', smiles)}):"

                # Format as readable text for LLM
                text_parts = [
                    header,
                    f"- Formula: {info.get('molecular_formula', 'N/A')}",
                    f"- Molecular Weight: {info.get('molecular_weight', 'N/A')} g/mol",
                    f"- Atoms: {info.get('num_atoms', 'N/A')}",
                    f"- Bonds: {info.get('num_bonds', 'N/A')}",
                    f"- Rings: {info.get('num_rings', 'N/A')}",
                    f"- Rotatable Bonds: {info.get('num_rotatable_bonds', 'N/A')}",
                    f"- H-Bond Donors: {info.get('num_hbd', 'N/A')}",
                    f"- H-Bond Acceptors: {info.get('num_hba', 'N/A')}",
                    f"- LogP: {info.get('logp', 'N/A')}",
                    f"- TPSA: {info.get('tpsa', 'N/A')} Å²",
                ]

                # Carry the structured properties alongside the prose so the
                # findings ledger can record MW / LogP / TPSA without having
                # to re-parse the text it just rendered.
                return ToolResult.text("\n".join(text_parts), metadata=info)
            else:
                return ToolResult.error(
                    response.get("error", "Unknown error from RDKit service")
                )

        except (InvalidReferenceError, ArtifactNotFoundError):
            return ToolResult.error(
                f"SMILES reference '{smiles_ref}' not found or invalid"
            )
        except Exception as e:
            logger.exception("Failed to get molecule info")
            return ToolResult.error(f"Failed to call RDKit service: {str(e)}")
