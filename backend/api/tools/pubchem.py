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

"""PubChem and CCD lookup tools for small molecule operations.

This module provides first-class LangGraph tools for operations that were
previously routed through python_execute via the aevom sandbox client:

1. search_pubchem - Search PubChem by molecule name
2. get_pubchem_properties - Get molecular properties by CID
3. download_pubchem_sdf - Download 3D SDF and store it
4. lookup_ccd_code - Look up PDB CCD code via PubChem + RCSB
5. visualize_sdf - Display stored SDF in the Molstar 3D viewer
"""

import json
import logging
from typing import Any

import httpx
import pubchempy as pcp

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from .base import BaseTool, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)


class SearchPubChemTool(BaseTool):
    """Search PubChem for a molecule by common or IUPAC name.

    Returns identifiers (CID, SMILES, InChI, InChIKey) that downstream tools
    such as get_pubchem_properties and download_pubchem_sdf accept.

    Data Flow:
        Input: molecule name (text)
        Output: JSON with CID, SMILES, InChI, InChIKey (text, no storage)
    """

    @property
    def name(self) -> str:
        return "search_pubchem"

    @property
    def summary_template(self) -> str:
        return "Searching PubChem for {name}..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_pubchem",
                "description": (
                    "Search PubChem for a molecule by common or IUPAC name. "
                    "Returns identifiers including CID, SMILES, InChI, and InChIKey.\n\n"
                    "Use this as the first step when working with a named molecule. "
                    "The returned CID can be passed to get_pubchem_properties or "
                    "download_pubchem_sdf."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Common or IUPAC name of the molecule "
                                "(e.g., 'aspirin', 'Tovorafenib', 'caffeine')."
                            ),
                        },
                    },
                    "required": ["name"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        name = arguments.get("name", "").strip()
        if not name:
            return ToolResult.error("Molecule name is required")

        try:
            compounds = pcp.get_compounds(name, "name")
            if not compounds:
                return ToolResult.text(f"No compounds found in PubChem for '{name}'.")

            compound = compounds[0]
            smiles_value = getattr(compound, "smiles", None) or compound.isomeric_smiles

            result = {
                "cid": compound.cid,
                "name": compound.iupac_name or name,
                "isomeric_smiles": smiles_value,
                "inchi": compound.inchi,
                "inchi_key": compound.inchikey,
            }

            return ToolResult.text(json.dumps(result, indent=2))

        except Exception as e:
            logger.exception(f"PubChem search failed for '{name}'")
            return ToolResult.error(f"Error searching PubChem for '{name}': {e}")


class GetPubChemPropertiesTool(BaseTool):
    """Retrieve molecular properties from PubChem by Compound ID.

    Returns MW, LogP, TPSA, H-bond counts, rotatable bonds, etc.
    Useful for Lipinski rule-of-five evaluation and ADME screening.

    Data Flow:
        Input: CID (integer)
        Output: JSON with properties (text, no storage)
    """

    @property
    def name(self) -> str:
        return "get_pubchem_properties"

    @property
    def summary_template(self) -> str:
        return "Fetching PubChem properties for CID {cid}..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_pubchem_properties",
                "description": (
                    "Get molecular properties from PubChem by Compound ID (CID). "
                    "Returns molecular weight, LogP, TPSA, H-bond donor/acceptor counts, "
                    "rotatable bonds, and other calculated properties.\n\n"
                    "Use after search_pubchem to evaluate drug-likeness (Lipinski rules)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cid": {
                            "type": "integer",
                            "description": "PubChem Compound ID (from search_pubchem result).",
                        },
                    },
                    "required": ["cid"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cid = arguments.get("cid")
        if not isinstance(cid, int) or cid <= 0:
            return ToolResult.error(f"CID must be a positive integer, got: {cid}")

        try:
            compound = pcp.Compound.from_cid(cid)
            if compound is None:
                return ToolResult.text(f"No compound found for CID {cid}.")

            result = {
                "molecular_weight": compound.molecular_weight,
                "molecular_formula": compound.molecular_formula,
                "xlogp": compound.xlogp,
                "tpsa": compound.tpsa,
                "charge": compound.charge or 0,
                "exact_mass": compound.exact_mass,
                "complexity": compound.complexity,
                "h_bond_donor_count": compound.h_bond_donor_count,
                "h_bond_acceptor_count": compound.h_bond_acceptor_count,
                "rotatable_bond_count": compound.rotatable_bond_count,
                "heavy_atom_count": compound.heavy_atom_count,
            }

            return ToolResult.text(json.dumps(result, indent=2))

        except Exception as e:
            logger.exception(f"Failed to get PubChem properties for CID {cid}")
            return ToolResult.error(
                f"Error getting PubChem properties for CID {cid}: {e}"
            )


class DownloadPubChemSdfTool(BaseTool):
    """Download a 3D SDF structure from PubChem and store it.

    Attempts 3D conformer first; falls back to 2D if unavailable.
    The SDF is stored via DataStorageService and a reference is returned
    that can be passed to visualize_sdf or vina_dock.

    Data Flow:
        Input: CID, optional name (text)
        Output: Stores SDF, returns reference (e.g. "sdf_abc12345")
    """

    @property
    def name(self) -> str:
        return "download_pubchem_sdf"

    @property
    def summary_template(self) -> str:
        return "Downloading SDF for CID {cid} from PubChem..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.OUTPUT_STORES

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "download_pubchem_sdf",
                "description": (
                    "Download a 3D molecular structure (SDF) from PubChem and store it. "
                    "Tries the 3D conformer first; falls back to 2D if 3D is unavailable.\n\n"
                    "Returns a reference (e.g. 'sdf_abc12345') that can be passed to:\n"
                    "- visualize_sdf to display in the 3D viewer\n"
                    "- vina_dock as the ligand_sdf_ref\n\n"
                    "Use after search_pubchem to obtain the CID."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cid": {
                            "type": "integer",
                            "description": "PubChem Compound ID (from search_pubchem result).",
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Optional molecule name for metadata "
                                "(e.g. 'aspirin'). Helps identify the stored SDF later."
                            ),
                        },
                    },
                    "required": ["cid"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        cid = arguments.get("cid")
        mol_name = arguments.get("name")

        if not isinstance(cid, int) or cid <= 0:
            return ToolResult.error(f"CID must be a positive integer, got: {cid}")

        if self._storage is None:
            return ToolResult.error("Storage service not available")

        sdf_content = None

        url_3d = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
            f"/record/SDF?record_type=3d"
        )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url_3d)
                if resp.status_code == 200:
                    sdf_content = resp.text
                elif resp.status_code != 404:
                    return ToolResult.error(
                        f"Failed to download 3D SDF for CID {cid}: HTTP {resp.status_code}"
                    )
        except httpx.HTTPError as e:
            logger.warning(f"3D SDF request failed for CID {cid}: {e}")

        if sdf_content is None:
            url_2d = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
                f"/record/SDF"
            )
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.get(url_2d)
                    resp.raise_for_status()
                    sdf_content = resp.text
            except httpx.HTTPStatusError as e:
                return ToolResult.error(
                    f"Failed to download SDF for CID {cid}: HTTP {e.response.status_code}. "
                    "Structure may not be available."
                )
            except httpx.HTTPError as e:
                return ToolResult.error(
                    f"Network error downloading SDF for CID {cid}: {e}"
                )

        metadata: dict[str, Any] = {"pubchem_cid": cid, "source": "pubchem"}
        if mol_name:
            metadata["name"] = mol_name

        try:
            reference = await self._storage.store(
                data_type="sdf",
                content=sdf_content.encode("utf-8"),
                metadata=metadata,
                description=f"PubChem CID {cid}"
                + (f" ({mol_name})" if mol_name else ""),
            )

            logger.info(f"Stored PubChem SDF for CID {cid} as {reference}")

            message = format_storage_message(
                data_type="sdf",
                reference=reference,
                source_description=f"PubChem CID {cid}"
                + (f" ({mol_name})" if mol_name else ""),
                accepting_tools=["visualize_sdf", "vina_dock"],
            )

            return ToolResult.data_reference(
                reference=reference,
                data_type="sdf",
                message=message,
                metadata=metadata,
            )

        except Exception as e:
            logger.exception(f"Failed to store SDF for CID {cid}")
            return ToolResult.error(f"Failed to store SDF: {e}")


class LookupCcdCodeTool(BaseTool):
    """Look up the PDB Chemical Component Dictionary (CCD) code for a molecule.

    Two-step process: PubChem name -> InChIKey -> RCSB CCD search.
    CCD codes are required by OpenFold3's ligand input format.

    Data Flow:
        Input: molecule name (text)
        Output: CCD code string (text, no storage)
    """

    @property
    def name(self) -> str:
        return "lookup_ccd_code"

    @property
    def summary_template(self) -> str:
        return "Looking up CCD code for {molecule_name}..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "lookup_ccd_code",
                "description": (
                    "Look up the PDB Chemical Component Dictionary (CCD) code for a "
                    "molecule by its common name. Searches PubChem to obtain the InChIKey, "
                    "then queries RCSB to find the corresponding CCD code.\n\n"
                    "CCD codes are needed for OpenFold3 ligand input:\n"
                    '  openfold3(molecules=[..., {"type": "ligand", "ccd_codes": "<code>"}])\n\n'
                    "Example: lookup_ccd_code('imatinib') -> 'STI'"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "molecule_name": {
                            "type": "string",
                            "description": (
                                "Common name of the molecule "
                                "(e.g. 'imatinib', 'aspirin', 'ATP')."
                            ),
                        },
                    },
                    "required": ["molecule_name"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        molecule_name = arguments.get("molecule_name", "").strip()
        if not molecule_name:
            return ToolResult.error("Molecule name is required")

        # Step 1: PubChem -> InChIKey
        try:
            compounds = pcp.get_compounds(molecule_name, "name")
            if not compounds:
                return ToolResult.text(
                    f"No PubChem compounds found for '{molecule_name}'. "
                    "Cannot determine CCD code without an InChIKey."
                )

            inchi_key = compounds[0].inchikey
            if not inchi_key:
                return ToolResult.text(
                    f"PubChem compound for '{molecule_name}' has no InChIKey. "
                    "Cannot look up CCD code."
                )
        except Exception as e:
            logger.exception(f"PubChem lookup failed for '{molecule_name}'")
            return ToolResult.error(
                f"Failed to search PubChem for '{molecule_name}': {e}"
            )

        # Step 2: RCSB CCD search via InChIKey
        rcsb_search_url = "https://search.rcsb.org/rcsbsearch/v2/query"
        search_payload = {
            "query": {
                "type": "terminal",
                "service": "text_chem",
                "parameters": {
                    "attribute": "rcsb_chem_comp_descriptor.InChIKey",
                    "operator": "exact_match",
                    "value": inchi_key,
                },
            },
            "return_type": "mol_definition",
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    rcsb_search_url,
                    json=search_payload,
                    headers={"Content-Type": "application/json"},
                )
                resp.raise_for_status()
                result = resp.json()

            result_set = result.get("result_set", [])
            if result_set:
                ccd_code = result_set[0].get("identifier")
                if ccd_code:
                    return ToolResult.text(
                        f"CCD code for '{molecule_name}': {ccd_code}"
                    )

            return ToolResult.text(
                f"No CCD code found in RCSB for '{molecule_name}' "
                f"(InChIKey: {inchi_key}). The molecule may not be in the PDB "
                "Chemical Component Dictionary. Consider using SMILES instead."
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"RCSB CCD search failed: HTTP {e.response.status_code}")
            return ToolResult.error(
                f"Failed to query RCSB for CCD code: HTTP {e.response.status_code}"
            )
        except Exception as e:
            logger.exception("RCSB CCD search failed")
            return ToolResult.error(f"Network error querying RCSB: {e}")


class VisualizeSdfTool(BaseTool):
    """Display a stored SDF molecule in the Molstar 3D viewer.

    Retrieves SDF content from DataStorageService by reference and sends
    it to the frontend for 3D rendering, following the same pattern as
    VisualizePdbTool.

    Data Flow:
        Input: SDF reference (from download_pubchem_sdf or python_execute store)
        Output: Triggers Molstar visualization (no data stored)
    """

    @property
    def name(self) -> str:
        return "visualize_sdf"

    @property
    def summary_template(self) -> str:
        return "Displaying SDF structure in 3D viewer..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.INPUT_RETRIEVES

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "visualize_sdf",
                "description": (
                    "Display a stored SDF molecule in the interactive 3D viewer (Molstar). "
                    "Accepts an SDF reference from download_pubchem_sdf or from "
                    "python_execute (aevom.store_sdf).\n\n"
                    "Use this after download_pubchem_sdf to visualize the molecule."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sdf_ref": {
                            "type": "string",
                            "description": (
                                "Reference to stored SDF data "
                                "(e.g. 'sdf_abc12345' from download_pubchem_sdf)."
                            ),
                            "x-aevom-reference": {"data_type": "sdf"},
                        },
                        "label": {
                            "type": "string",
                            "description": "Display label for the molecule (e.g. 'Aspirin').",
                        },
                    },
                    "required": ["sdf_ref"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        sdf_ref = arguments.get("sdf_ref", "").strip()
        label = arguments.get("label", "")

        if not sdf_ref:
            return ToolResult.error(
                "sdf_ref is required. Use download_pubchem_sdf to obtain one."
            )

        if self._storage is None:
            return ToolResult.error("Storage service not available")

        try:
            content, metadata = await self._storage.retrieve_with_metadata(sdf_ref)
            sdf_data = content.decode("utf-8")
        except InvalidReferenceError:
            return ToolResult.error(
                f"Invalid SDF reference: '{sdf_ref}'. "
                "Use download_pubchem_sdf to download an SDF first, "
                "which returns a reference like 'sdf_abc12345'."
            )
        except ArtifactNotFoundError:
            return ToolResult.error(
                f"SDF data not found for reference '{sdf_ref}'. "
                "The data may have expired or the reference is invalid."
            )

        display_label = label or metadata.get("name", sdf_ref)

        logger.info(f"Visualizing SDF {sdf_ref} as '{display_label}'")

        return ToolResult.sdf_3d(
            sdf_data=sdf_data,
            label=display_label,
            metadata={"sdf_ref": sdf_ref},
        )
