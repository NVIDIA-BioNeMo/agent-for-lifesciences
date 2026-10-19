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

"""3D visualization tools for molecular structures.

These tools enable the LLM to generate 3D coordinates and display
molecules in the Molstar 3D viewer.

This module demonstrates the data storage pattern:
- Generate3DCoordinatesTool: BOTH - retrieves SMILES ref, stores SDF, returns reference
- Visualize3DTool: INPUT_RETRIEVES - accepts SDF reference, retrieves and displays
- VisualizePdbTool: INPUT_RETRIEVES - accepts PDB reference, displays protein structure
"""

import logging
import os
from typing import Any, Optional

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from .base import BaseTool, ContainerTool, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)


class Generate3DCoordinatesTool(ContainerTool):
    """Generate 3D atomic coordinates for a molecule from a stored SMILES reference.

    This tool retrieves a SMILES string from the data store, converts it to SDF
    format with optimized 3D geometry, and stores the result.

    Data Flow:
        Input: SMILES reference (e.g., "smiles_abc123") from store_smiles()
        Output: Stores SDF data, returns reference (e.g., "sdf_xyz789")

    Workflow:
        1. Python code stores SMILES via aevom.store_smiles() -> "smiles_abc123"
        2. LLM calls generate_3d_coordinates(smiles_ref="smiles_abc123")
        3. Tool retrieves SMILES, generates 3D coords, stores SDF
        4. Returns "sdf_xyz789" for use with visualize_3d or vina_dock
    """

    @property
    def name(self) -> str:
        return "generate_3d_coordinates"

    @property
    def summary_template(self) -> str:
        return "Generating 3D coordinates for {smiles_ref}..."

    def format_summary(self, arguments: dict) -> str:
        """Custom summary that shows the reference being processed."""
        smiles_ref = arguments.get("smiles_ref", "")
        if smiles_ref:
            return f"Generating 3D coordinates for {smiles_ref}..."
        return "Generating 3D coordinates..."

    @property
    def service_url(self) -> str:
        """URL of the RDKit service container."""
        return os.environ.get("RDKIT_SERVICE_URL", "http://rdkit:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        """This tool retrieves SMILES input and stores SDF output."""
        return DataIOCategory.BOTH

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "generate_3d_coordinates",
                "description": (
                    "Generate 3D atomic coordinates for a molecule from a stored SMILES reference.\n\n"
                    "This tool takes a SMILES reference (from python_execute using aevom.store_smiles()) "
                    "and generates an optimized 3D molecular structure in SDF format.\n\n"
                    "WORKFLOW:\n"
                    "1. First, use python_execute to look up a molecule (e.g., from PubChem) and "
                    "store its SMILES using aevom.store_smiles()\n"
                    "2. Then call this tool with the returned smiles_ref\n"
                    "3. Use the returned sdf_ref with visualize_3d or vina_dock\n\n"
                    "EXAMPLE PYTHON CODE (for python_execute):\n"
                    "  import pubchempy as pcp\n"
                    "  compound = pcp.get_compounds('aspirin', 'name')[0]\n"
                    "  smiles_ref = aevom.store_smiles(compound.canonical_smiles, description='Aspirin')\n"
                    "  print(f'Stored SMILES as: {smiles_ref}')\n\n"
                    "Then call: generate_3d_coordinates(smiles_ref=smiles_ref)"
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
                        "description": {
                            "type": "string",
                            "description": (
                                "Short description of the molecule for reference tracking "
                                "(e.g., 'Aspirin 3D structure', 'Ibuprofen conformer'). "
                                "This helps identify the stored SDF data later."
                            ),
                        },
                    },
                    "required": ["smiles_ref"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Generate 3D coordinates for a molecule and store the result.

        Args:
            arguments: Dict containing 'smiles_ref' and optional 'description'.

        Returns:
            ToolResult with reference to stored SDF data.
        """
        smiles_ref = arguments.get("smiles_ref", "")
        description = arguments.get("description")

        if not smiles_ref:
            return ToolResult.error(
                "smiles_ref is required. Use python_execute with aevom.store_smiles() "
                "to store a SMILES string first, then pass the returned reference here."
            )

        # Validate storage service is available
        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve SMILES or store SDF data."
            )

        try:
            # 1. Retrieve the SMILES from storage
            logger.info(f"Retrieving SMILES from {smiles_ref}")
            try:
                (
                    smiles_content,
                    smiles_metadata,
                ) = await self._storage.retrieve_with_metadata(smiles_ref)
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

            # 2. Call RDKit service to generate 3D coordinates
            response = await self._call_service(
                endpoint="/smiles-to-sdf",
                method="POST",
                json_data={"smiles": smiles, "optimize": True},
                timeout=60.0,
            )

            if not response.get("success"):
                error_msg = response.get("error", "Unknown error from RDKit service")
                return ToolResult.error(f"Failed to generate 3D structure: {error_msg}")

            sdf = response.get("sdf", "")
            if not sdf:
                return ToolResult.error("No SDF data returned from RDKit service")

            # 3. Store the SDF data
            # Preserve original SMILES metadata and add source reference
            sdf_metadata = {
                "smiles": smiles,
                "smiles_ref": smiles_ref,
                "optimized": True,
            }
            # Copy over any useful metadata from the SMILES
            if smiles_metadata:
                for key in ("name", "source", "cid"):
                    if key in smiles_metadata:
                        sdf_metadata[key] = smiles_metadata[key]

            reference = await self._storage.store(
                data_type="sdf",
                content=sdf.encode("utf-8"),
                metadata=sdf_metadata,
                description=description,
            )

            logger.info(f"Stored SDF for {smiles_ref} as {reference}")

            # 4. Return a helpful message guiding the LLM
            message = format_storage_message(
                data_type="sdf",
                reference=reference,
                source_description=f"3D structure from {smiles_ref}",
                accepting_tools=["visualize_3d", "vina_dock"],
            )

            return ToolResult.data_reference(
                reference=reference,
                data_type="sdf",
                message=message,
                metadata={"smiles": smiles, "smiles_ref": smiles_ref},
            )

        except (InvalidReferenceError, ArtifactNotFoundError):
            return ToolResult.error(
                f"SMILES reference '{smiles_ref}' not found or invalid"
            )
        except Exception as e:
            logger.exception("Failed to generate 3D coordinates")
            return ToolResult.error(f"Failed to generate 3D coordinates: {str(e)}")


class Visualize3DTool(ContainerTool):
    """Display a molecule in the interactive 3D viewer (Molstar).

    This tool renders molecules in the 3D viewer panel. It can accept:
    - A reference to stored SDF data (from generate_3d_coordinates)
    - A SMILES string (for convenience, will generate 3D internally)

    Data Flow:
        Input: SDF reference or SMILES string
        Output: Triggers visualization (no data stored)
    """

    @property
    def name(self) -> str:
        return "visualize_3d"

    @property
    def summary_template(self) -> str:
        return "Displaying 3D structure in viewer..."

    def format_summary(self, arguments: dict) -> str:
        """Custom summary that handles both sdf_ref and smiles."""
        sdf_ref = arguments.get("sdf_ref", "")
        smiles = arguments.get("smiles", "")
        label = arguments.get("label", "")

        if label:
            return f"Displaying {label} in 3D viewer..."
        elif sdf_ref:
            return f"Displaying {sdf_ref} in 3D viewer..."
        elif smiles:
            return f"Displaying {smiles} in 3D viewer..."
        return "Displaying 3D structure in viewer..."

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
                "name": "visualize_3d",
                "description": (
                    "Display a molecule in the interactive 3D viewer (Molstar). Use this "
                    "tool when the user asks to 'show', 'display', 'visualize', 'render', "
                    "or 'view' a molecule in 3D.\n\n"
                    "You can provide:\n"
                    "- sdf_ref: A reference from generate_3d_coordinates (e.g., 'sdf_abc123')\n"
                    "- smiles: A SMILES string (will generate 3D automatically)\n\n"
                    "If you have already called generate_3d_coordinates, use the sdf_ref "
                    "from that call. Otherwise, provide the SMILES string directly.\n\n"
                    "Common molecules:\n"
                    "- Water: O\n"
                    "- Ethanol: CCO\n"
                    "- Benzene: c1ccccc1\n"
                    "- Caffeine: Cn1cnc2c1c(=O)n(c(=O)n2C)C\n"
                    "- Aspirin: CC(=O)Oc1ccccc1C(=O)O\n"
                    "- Glucose: OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sdf_ref": {
                            "type": "string",
                            "description": (
                                "Reference to stored SDF data from generate_3d_coordinates "
                                "(e.g., 'sdf_abc123'). Use this if you have already generated "
                                "3D coordinates."
                            ),
                            "x-aevom-reference": {"data_type": "sdf"},
                        },
                        "smiles": {
                            "type": "string",
                            "description": (
                                "SMILES string of the molecule. Use this for direct "
                                "visualization without pre-generating coordinates."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "Display label for the molecule (e.g., 'Glucose').",
                        },
                    },
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Display a molecule in the 3D viewer.

        Args:
            arguments: Dict containing 'sdf_ref' or 'smiles', and optional 'label'.

        Returns:
            ToolResult with SDF_3D content type for frontend visualization.
        """
        sdf_ref = arguments.get("sdf_ref", "")
        smiles = arguments.get("smiles", "")
        label = arguments.get("label", "")

        sdf = ""
        source_smiles = smiles

        # Try to retrieve SDF from reference first
        if sdf_ref:
            if self._storage is None:
                return ToolResult.error(
                    "Storage service not available. Cannot retrieve SDF data."
                )

            try:
                content, metadata = await self._storage.retrieve_with_metadata(sdf_ref)
                sdf = content.decode("utf-8")
                source_smiles = metadata.get("smiles", source_smiles)
                logger.info(f"Retrieved SDF from {sdf_ref}")
            except InvalidReferenceError:
                return ToolResult.error(
                    f"Invalid SDF reference: '{sdf_ref}'. "
                    "Use generate_3d_coordinates or download_pubchem_sdf to create an SDF, "
                    "which returns a reference like 'sdf_abc12345'."
                )
            except ArtifactNotFoundError:
                return ToolResult.error(
                    f"SDF data not found for reference '{sdf_ref}'. "
                    "The data may have expired or the reference is invalid. "
                    "Try calling generate_3d_coordinates first to create new SDF data."
                )

        # If no SDF from reference, generate from SMILES
        if not sdf:
            if not smiles:
                return ToolResult.error(
                    "Either 'sdf_ref' (reference to stored SDF) or 'smiles' is required. "
                    "Provide a reference from generate_3d_coordinates, or a SMILES string."
                )

            try:
                response = await self._call_service(
                    endpoint="/smiles-to-sdf",
                    method="POST",
                    json_data={"smiles": smiles, "optimize": True},
                    timeout=60.0,
                )

                if response.get("success"):
                    sdf = response.get("sdf", "")
                else:
                    error_msg = response.get(
                        "error", "Unknown error from RDKit service"
                    )
                    return ToolResult.error(
                        f"Failed to generate 3D structure: {error_msg}"
                    )

            except Exception as e:
                logger.exception("Failed to generate 3D structure for visualization")
                return ToolResult.error(f"Failed to generate 3D structure: {str(e)}")

        if not sdf:
            return ToolResult.error("Failed to obtain SDF data for visualization")

        # Use label or smiles as the molecule identifier
        display_label = label or source_smiles or "Molecule"

        logger.info(f"Visualizing molecule: {display_label}")

        # Return SDF_3D result - the frontend will receive this via SSE
        return ToolResult.sdf_3d(
            sdf_data=sdf,
            label=display_label,
            metadata={"smiles": source_smiles} if source_smiles else {},
        )


class VisualizePdbTool(BaseTool):
    """Display a protein structure in the interactive 3D viewer (Molstar).

    Accepts either a PDB or an mmCIF reference. Molstar parses both natively, so
    the format is carried through to the viewer rather than converted here --
    ``boltz2`` only produces mmCIF, and round-tripping it to PDB would lose the
    long chain ids and large atom counts that mmCIF exists to represent.

    Data Flow:
        Input: PDB or mmCIF reference (from predict_structure, boltz2, and others)
        Output: Triggers visualization (no data stored)
    """

    # Reference prefixes that hold mmCIF rather than PDB text.
    # Mol* ignores the extra design-condition columns of an annotated mmCIF
    # and renders it like any other mmCIF.
    _MMCIF_TYPES = frozenset({"mmcif", "cif", "pdbx", "annotated_cif"})

    @property
    def name(self) -> str:
        return "visualize_pdb"

    @property
    def summary_template(self) -> str:
        return "Displaying protein structure {pdb_ref}..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

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
                "name": "visualize_pdb",
                "description": (
                    "Display a protein structure in the interactive 3D viewer (Molstar). "
                    "Use this tool when the user asks to 'show', 'display', 'visualize', "
                    "'render', or 'view' a protein structure in 3D.\n\n"
                    "Accepts either a PDB reference (e.g., 'pdb_abc123') or an mmCIF "
                    "reference (e.g., 'mmcif_abc123'); the viewer reads both, so there "
                    "is no need to convert an mmCIF structure before displaying it.\n\n"
                    "If you have predicted a protein structure using predict_structure "
                    "or boltz2, use the reference from that call to display the result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_ref": {
                            "type": "string",
                            "description": (
                                "Reference to a stored structure, either PDB "
                                "('pdb_abc123') or mmCIF ('mmcif_abc123'). Required."
                            ),
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "label": {
                            "type": "string",
                            "description": (
                                "Display label for the protein structure "
                                "(e.g., 'PGK1 predicted structure'). Optional."
                            ),
                        },
                    },
                    "required": ["pdb_ref"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Display a protein structure in the 3D viewer.

        Args:
            arguments: Dict containing 'pdb_ref' and optional 'label'.

        Returns:
            ToolResult with PDB_3D content type for frontend visualization.
        """
        pdb_ref = arguments.get("pdb_ref", "")
        label = arguments.get("label", "")

        if not pdb_ref:
            return ToolResult.error(
                "The 'pdb_ref' parameter is required. "
                "Provide a reference from predict_structure (e.g., 'pdb_abc123') "
                "or boltz2 (e.g., 'mmcif_abc123')."
            )

        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve PDB data."
            )

        try:
            content, metadata = await self._storage.retrieve_with_metadata(pdb_ref)
            pdb_data = content.decode("utf-8")
            logger.info(f"Retrieved structure from {pdb_ref}")
        except InvalidReferenceError:
            return ToolResult.error(
                f"Invalid structure reference: '{pdb_ref}'. "
                "Use fetch_and_clean_pdb or openfold3 to obtain a PDB "
                "('pdb_abc12345'), or boltz2 for an mmCIF ('mmcif_abc12345')."
            )
        except ArtifactNotFoundError:
            return ToolResult.error(
                f"Structure data not found for reference '{pdb_ref}'. "
                "The data may have expired or the reference is invalid. "
                "Try calling predict_structure first to generate a new structure."
            )

        if not pdb_data:
            return ToolResult.error("Retrieved structure data is empty")

        data_type = (self._storage.get_data_type(pdb_ref) or "pdb").lower()
        structure_format = "mmcif" if data_type in self._MMCIF_TYPES else "pdb"

        # Build display label from metadata if not provided
        if not label:
            protein_id = metadata.get("protein_id", "")
            gene_name = metadata.get("gene_name", "")
            label = protein_id or gene_name or "Protein Structure"

        # Include confidence info in metadata for display
        display_metadata = {
            "protein_id": metadata.get("protein_id", ""),
            "confidence_score": metadata.get("confidence_score", 0.0),
            "plddt_score": metadata.get("plddt_score", 0.0),
            "format": structure_format,
        }
        # A design says which of its chains were designed and which are the
        # target, so the viewer can colour by role instead of by chain letter.
        roles = _chain_roles(metadata)
        if roles:
            display_metadata["roles"] = roles

        logger.info(f"Visualizing {structure_format} structure: {label}")

        # Return PDB_3D result - the frontend will receive this via SSE
        return ToolResult.pdb_3d(
            pdb_data=pdb_data,
            label=label,
            metadata=display_metadata,
        )


def _chain_roles(metadata: dict) -> Optional[dict]:
    """The target and binder label chains an RFD4 design recorded, if any."""
    binder = metadata.get("binder_chains")
    target = metadata.get("target_chains")
    if not isinstance(binder, list) or not isinstance(target, list):
        return None
    binder = [str(c) for c in binder if c]
    target = [str(c) for c in target if c]
    if not binder and not target:
        return None
    return {"target": target, "binder": binder}
