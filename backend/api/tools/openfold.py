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

"""OpenFold3 API tool for biomolecular structure prediction.

This tool uses NVIDIA's hosted OpenFold3 API to predict 3D structures
of biomolecular complexes including proteins, DNA, RNA, and ligands.
It mirrors the OpenFold3 API structure and integrates with the data
storage pattern for sequence references and PDB output.
"""

import json
import logging
import os
import re
import tempfile
from typing import Any

import httpx

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from .base import DataIOCategory, NvidiaApiTool, ToolResult

logger = logging.getLogger(__name__)

# OpenFold3 API endpoint
OPENFOLD3_API_URL = (
    "https://health.api.nvidia.com/v1/biology/openfold/openfold3/predict"
)

# Valid characters for different molecule types
VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
VALID_DNA_NUCLEOTIDES = set("ATCG")
VALID_RNA_NUCLEOTIDES = set("AUCG")


class OpenFold3Tool(NvidiaApiTool):
    """Predict 3D structures of biomolecular complexes using OpenFold3.

    This tool calls the NVIDIA-hosted OpenFold3 API to predict the 3D structure
    of proteins, protein-ligand complexes, protein-DNA/RNA complexes, and
    multi-chain protein assemblies.

    The tool mirrors the OpenFold3 API structure, accepting a `molecules` array
    where each molecule can be a protein, DNA, RNA, or ligand.

    Data Flow:
        Input: molecules array with sequence/SMILES data, or simple sequence_ref
        Output: Stores PDB/CIF data, returns reference (e.g., "pdb_abc123")

    Examples:
        # Single protein (backward compatible)
        openfold3(sequence_ref="sequence_abc123")

        # Protein-ligand docking
        openfold3(molecules=[
            {"type": "protein", "sequence_ref": "sequence_abc123"},
            {"type": "ligand", "smiles": "CC(=O)Oc1ccccc1C(=O)O"}
        ])

        # Protein with ATP cofactor
        openfold3(molecules=[
            {"type": "protein", "sequence": "MGREEPLNHVEAERQRREKL..."},
            {"type": "ligand", "ccd_codes": "ATP"}
        ])
    """

    @property
    def name(self) -> str:
        return "openfold3"

    @property
    def summary_template(self) -> str:
        return "Predicting structure with OpenFold3..."

    def format_summary(self, arguments: dict) -> str:
        """Custom summary based on input type."""
        molecules = arguments.get("molecules", [])
        sequence_ref = arguments.get("sequence_ref", "")

        if molecules and isinstance(molecules, list):
            mol_types = [
                m.get("type", "unknown") if isinstance(m, dict) else "unknown"
                for m in molecules
            ]
            if "ligand" in mol_types and "protein" in mol_types:
                return "Predicting protein-ligand docked structure..."
            elif len([t for t in mol_types if t == "protein"]) > 1:
                return "Predicting multi-chain protein complex..."
            return f"Predicting {'/'.join(set(mol_types))} structure..."
        elif sequence_ref:
            return f"Predicting structure for {sequence_ref}..."
        return "Predicting structure with OpenFold3..."

    @property
    def api_endpoint(self) -> str:
        return OPENFOLD3_API_URL

    @property
    def data_io_category(self) -> DataIOCategory:
        """This tool retrieves sequence data and stores PDB output."""
        return DataIOCategory.BOTH

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema mirroring OpenFold3 API."""
        return {
            "type": "function",
            "function": {
                "name": "openfold3",
                "description": (
                    "Predict 3D structure of biomolecular complexes using OpenFold3. "
                    "Supports single proteins, protein-ligand docking, protein-DNA/RNA "
                    "complexes, and multi-chain protein assemblies.\n\n"
                    "USE CASES:\n"
                    "- Single protein structure prediction\n"
                    "- Protein-ligand docking (predict how a small molecule binds)\n"
                    "- Protein-cofactor complexes (ATP, NAD, heme, etc.)\n"
                    "- Multi-chain protein complexes\n\n"
                    "INPUT OPTIONS:\n"
                    "1. Simple mode: Use 'sequence_ref' or 'sequence' for single proteins\n"
                    "2. Complex mode: Use 'molecules' array for multi-molecule predictions\n\n"
                    "MSA SUPPORT:\n"
                    "Optionally provide 'msa_ref' from msa_search to use a real multiple "
                    "sequence alignment instead of a single-sequence placeholder. This "
                    "significantly improves prediction quality.\n\n"
                    "MOLECULES ARRAY:\n"
                    "Each molecule object requires a 'type' field:\n"
                    "- protein: Requires 'sequence' or 'sequence_ref'\n"
                    "- ligand: Requires 'smiles' OR 'ccd_codes' (not both)\n"
                    "- dna: Requires 'sequence' (A, T, C, G)\n"
                    "- rna: Requires 'sequence' (A, U, C, G)\n\n"
                    "LIGAND SPECIFICATION:\n"
                    "- smiles: SMILES string (e.g., 'CC(=O)Oc1ccccc1C(=O)O' for aspirin)\n"
                    "- ccd_codes: a single (not multiple) PDB Chemical Component Dictionary code (e.g., 'ATP')\n\n"
                    "OUTPUT: Stores predicted PDB structure, returns reference for visualize_pdb.\n\n"
                    "Note: Structure prediction may take 30-120 seconds depending on complexity."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "molecules": {
                            "type": "array",
                            "description": (
                                "Array of molecules to predict as a complex. Use this for "
                                "protein-ligand docking, multi-chain complexes, or when you "
                                "need to specify multiple molecules."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "enum": ["protein", "dna", "rna", "ligand"],
                                        "description": "Type of molecule.",
                                    },
                                    "sequence": {
                                        "type": "string",
                                        "description": (
                                            "Amino acid sequence (for protein) or nucleotide "
                                            "sequence (for DNA: ATCG, for RNA: AUCG)."
                                        ),
                                    },
                                    "sequence_ref": {
                                        "type": "string",
                                        "description": (
                                            "Reference to stored sequence from fetch_protein_sequence. "
                                            "Alternative to providing sequence directly."
                                        ),
                                        "x-aevom-reference": {"data_type": "sequence"},
                                    },
                                    "smiles": {
                                        "type": "string",
                                        "description": (
                                            "SMILES string for ligand molecules. "
                                            "Example: 'CC(=O)Oc1ccccc1C(=O)O' for aspirin."
                                        ),
                                    },
                                    "ccd_codes": {
                                        "type": "string",
                                        "description": (
                                            "PDB Chemical Component Dictionary code for ligand. "
                                            "Examples: 'ATP', 'NAD', 'HEM', 'FAD', 'GDP'."
                                        ),
                                    },
                                    "id": {
                                        "type": "string",
                                        "description": (
                                            "Chain identifier (e.g., 'A', 'B'). Auto-assigned if not provided."
                                        ),
                                    },
                                },
                                "required": ["type"],
                            },
                        },
                        "sequence_ref": {
                            "type": "string",
                            "description": (
                                "Reference to stored sequence from fetch_protein_sequence. "
                                "Use for simple single-protein predictions. "
                                "Alternative to using the molecules array."
                            ),
                            "x-aevom-reference": {"data_type": "sequence"},
                        },
                        "msa_ref": {
                            "type": "string",
                            "description": (
                                "Reference to stored MSA from msa_search. "
                                "When provided, uses the real MSA instead of a "
                                "single-sequence placeholder. "
                                "Improves structure prediction quality significantly."
                            ),
                            "x-aevom-reference": {"data_type": "msa"},
                        },
                        "sequence": {
                            "type": "string",
                            "description": (
                                "Raw amino acid sequence for single-protein predictions. "
                                "Alternative to using the molecules array."
                            ),
                        },
                        "output_format": {
                            "type": "string",
                            "enum": ["pdb", "cif"],
                            "description": "Output structure format. Default: 'pdb'.",
                        },
                        "diffusion_samples": {
                            "type": "integer",
                            "description": (
                                "Number of independent structures to generate (1-5). "
                                "More samples increase diversity but take longer. Default: 1."
                            ),
                        },
                        "prediction_id": {
                            "type": "string",
                            "description": (
                                "Identifier for this prediction (e.g., protein name). "
                                "Used for labeling output. Optional."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short description of the predicted structure for reference tracking "
                                "(e.g., 'PGK1 kinase domain', 'GLP-1 with semaglutide'). "
                                "This helps identify the stored structure later."
                            ),
                        },
                    },
                },
            },
        }

    async def _resolve_sequence(
        self, molecule: dict[str, Any]
    ) -> tuple[str, dict[str, Any]]:
        """Resolve sequence from reference or direct value.

        Args:
            molecule: Molecule dict with 'sequence' or 'sequence_ref'.

        Returns:
            Tuple of (sequence_string, metadata_dict).

        Raises:
            ValueError: If sequence cannot be resolved.
        """
        sequence_ref = molecule.get("sequence_ref", "")
        raw_sequence = molecule.get("sequence", "")

        if sequence_ref:
            if self._storage is None:
                raise ValueError(
                    "Storage service not available. Cannot retrieve sequence data."
                )

            try:
                content, metadata = await self._storage.retrieve_with_metadata(
                    sequence_ref
                )
                sequence = content.decode("utf-8")
                logger.info(
                    f"Retrieved sequence from {sequence_ref} ({len(sequence)} residues)"
                )
                return sequence, metadata
            except InvalidReferenceError:
                raise ValueError(
                    f"Invalid sequence reference: '{sequence_ref}'. "
                    "Use fetch_protein_sequence first, which returns a reference "
                    "like 'sequence_abc12345'."
                )
            except ArtifactNotFoundError:
                raise ValueError(
                    f"Sequence data not found for reference '{sequence_ref}'. "
                    "The data may have expired or the reference is invalid."
                )

        if raw_sequence:
            return raw_sequence.strip().upper(), {}

        raise ValueError(
            "Either 'sequence' or 'sequence_ref' is required for this molecule."
        )

    def _validate_protein_sequence(self, sequence: str) -> None:
        """Validate protein amino acid sequence."""
        invalid_chars = set(sequence) - VALID_AMINO_ACIDS
        if invalid_chars:
            raise ValueError(
                f"Invalid characters in protein sequence: {invalid_chars}. "
                "Use standard amino acid letters (A-Z excluding B, J, O, U, X, Z)."
            )

    def _validate_dna_sequence(self, sequence: str) -> None:
        """Validate DNA nucleotide sequence."""
        invalid_chars = set(sequence) - VALID_DNA_NUCLEOTIDES
        if invalid_chars:
            raise ValueError(
                f"Invalid characters in DNA sequence: {invalid_chars}. "
                "Use A, T, C, G only."
            )

    def _validate_rna_sequence(self, sequence: str) -> None:
        """Validate RNA nucleotide sequence."""
        invalid_chars = set(sequence) - VALID_RNA_NUCLEOTIDES
        if invalid_chars:
            raise ValueError(
                f"Invalid characters in RNA sequence: {invalid_chars}. "
                "Use A, U, C, G only."
            )

    def _validate_smiles(self, smiles: str) -> None:
        """Basic SMILES validation."""
        if not smiles or len(smiles) < 1:
            raise ValueError("SMILES string cannot be empty.")
        # Basic check for valid SMILES characters
        # SMILES can contain: letters, numbers, and special chars like ()[]=#@+-.\/%
        if not re.match(r"^[A-Za-z0-9\(\)\[\]=#@+\-\.\\/\%\*:]+$", smiles):
            raise ValueError(
                f"Invalid SMILES string: '{smiles}'. Check for invalid characters."
            )

    def _validate_ccd_code(self, ccd_code: str) -> None:
        """Validate CCD code format."""
        if not re.match(r"^[A-Z0-9]{1,5}$", ccd_code):
            raise ValueError(
                f"Invalid CCD code: '{ccd_code}'. "
                "CCD codes must be 1-5 uppercase letters/numbers (e.g., 'ATP', 'NAD')."
            )

    def _build_protein_msa(self, sequence: str) -> dict[str, Any]:
        """Build minimal MSA for a protein sequence."""
        msa_csv = f"key,sequence\n-1,{sequence}"
        return {
            "main_db": {
                "csv": {
                    "alignment": msa_csv,
                    "format": "csv",
                }
            }
        }

    async def _resolve_msa(self, msa_ref: str) -> dict[str, Any]:
        """Retrieve a3m alignment from storage and wrap in OpenFold3 format."""
        if self._storage is None:
            raise ValueError("Storage service not available. Cannot retrieve MSA data.")
        try:
            content, _ = await self._storage.retrieve_with_metadata(msa_ref)
            a3m_text = content.decode("utf-8")
            logger.info(f"Retrieved MSA from {msa_ref} ({len(a3m_text)} chars)")
            return {
                "main_db": {
                    "a3m": {
                        "alignment": a3m_text,
                        "format": "a3m",
                    }
                }
            }
        except InvalidReferenceError:
            raise ValueError(
                f"Invalid MSA reference: '{msa_ref}'. "
                "Use msa_search first to generate an MSA."
            )
        except ArtifactNotFoundError:
            raise ValueError(
                f"MSA data not found for reference '{msa_ref}'. "
                "The data may have expired or the reference is invalid."
            )

    async def _process_molecules(
        self,
        molecules: list[dict[str, Any]],
        msa_ref: str = "",
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Process molecules array into API format.

        Args:
            molecules: List of molecule specifications from user.
            msa_ref: Optional MSA reference applied to the first protein molecule.

        Returns:
            Tuple of (api_molecules_list, combined_metadata).
        """
        api_molecules = []
        combined_metadata: dict[str, Any] = {
            "molecule_types": [],
            "has_ligand": False,
            "protein_count": 0,
            "ligand_count": 0,
        }

        chain_id_counter = ord("A")

        for i, mol in enumerate(molecules):
            mol_type = mol.get("type", "").lower()
            chain_id = mol.get("id", chr(chain_id_counter))
            chain_id_counter += 1

            combined_metadata["molecule_types"].append(mol_type)

            if mol_type == "protein":
                sequence, seq_metadata = await self._resolve_sequence(mol)
                self._validate_protein_sequence(sequence)

                # Use real MSA if provided (for first protein, or per-molecule msa_ref)
                mol_msa_ref = mol.get("msa_ref", "")
                effective_msa_ref = mol_msa_ref or msa_ref
                if effective_msa_ref and combined_metadata["protein_count"] == 0:
                    msa_data = await self._resolve_msa(effective_msa_ref)
                    logger.info(f"Using real MSA from {effective_msa_ref}")
                    msa_ref = ""  # consume the top-level ref for first protein only
                else:
                    msa_data = self._build_protein_msa(sequence)

                api_mol = {
                    "type": "protein",
                    "id": chain_id,
                    "sequence": sequence,
                    "msa": msa_data,
                }
                api_molecules.append(api_mol)
                combined_metadata["protein_count"] += 1
                combined_metadata[f"protein_{chain_id}_length"] = len(sequence)

                # Preserve metadata from sequence reference
                if seq_metadata.get("gene_name"):
                    combined_metadata["gene_name"] = seq_metadata.get("gene_name")
                if seq_metadata.get("uniprot_id"):
                    combined_metadata["uniprot_id"] = seq_metadata.get("uniprot_id")

            elif mol_type == "ligand":
                smiles = mol.get("smiles", "")
                ccd_codes = mol.get("ccd_codes", "")

                if smiles and ccd_codes:
                    raise ValueError(
                        f"Molecule {i + 1}: Provide either 'smiles' OR 'ccd_codes', not both."
                    )

                if not smiles and not ccd_codes:
                    raise ValueError(
                        f"Molecule {i + 1}: Ligand requires either 'smiles' or 'ccd_codes'."
                    )

                api_mol = {
                    "type": "ligand",
                    "id": chain_id,
                }

                if smiles:
                    self._validate_smiles(smiles)
                    api_mol["smiles"] = smiles
                    combined_metadata[f"ligand_{chain_id}_smiles"] = smiles
                else:
                    self._validate_ccd_code(ccd_codes)
                    api_mol["ccd_codes"] = ccd_codes
                    combined_metadata[f"ligand_{chain_id}_ccd"] = ccd_codes

                api_molecules.append(api_mol)
                combined_metadata["has_ligand"] = True
                combined_metadata["ligand_count"] += 1

            elif mol_type == "dna":
                sequence, _ = await self._resolve_sequence(mol)
                sequence = sequence.upper()
                self._validate_dna_sequence(sequence)

                api_mol = {
                    "type": "dna",
                    "id": chain_id,
                    "sequence": sequence,
                }
                api_molecules.append(api_mol)
                combined_metadata[f"dna_{chain_id}_length"] = len(sequence)

            elif mol_type == "rna":
                sequence, _ = await self._resolve_sequence(mol)
                sequence = sequence.upper()
                self._validate_rna_sequence(sequence)

                # RNA requires MSA like proteins
                msa_csv = f"key,sequence\n-1,{sequence}"
                api_mol = {
                    "type": "rna",
                    "id": chain_id,
                    "sequence": sequence,
                    "msa": {
                        "main_db": {
                            "csv": {
                                "alignment": msa_csv,
                                "format": "csv",
                            }
                        }
                    },
                }
                api_molecules.append(api_mol)
                combined_metadata[f"rna_{chain_id}_length"] = len(sequence)

            else:
                raise ValueError(
                    f"Molecule {i + 1}: Invalid type '{mol_type}'. "
                    "Must be 'protein', 'ligand', 'dna', or 'rna'."
                )

        return api_molecules, combined_metadata

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute OpenFold3 structure prediction.

        Supports both simple mode (sequence/sequence_ref) and complex mode (molecules array).

        Args:
            arguments: Dict containing molecules array OR sequence/sequence_ref.

        Returns:
            ToolResult with reference to stored PDB/CIF data.
        """
        molecules_input = arguments.get("molecules", [])
        sequence_ref = arguments.get("sequence_ref", "")
        msa_ref = arguments.get("msa_ref", "")
        raw_sequence = arguments.get("sequence", "")
        output_format = arguments.get("output_format", "pdb").lower()
        diffusion_samples = arguments.get("diffusion_samples", 1)
        prediction_id = arguments.get("prediction_id", "")
        description = arguments.get("description")

        # Coerce double-encoded JSON strings (LLMs sometimes stringify arrays)
        if isinstance(molecules_input, str):
            try:
                molecules_input = json.loads(molecules_input)
            except (json.JSONDecodeError, TypeError):
                return ToolResult.error(
                    "molecules must be a JSON array of molecule objects, not a string."
                )

        # Validate output format
        if output_format not in ("pdb", "cif"):
            return ToolResult.error(
                f"Invalid output_format: '{output_format}'. Must be 'pdb' or 'cif'."
            )

        # Validate diffusion samples
        if (
            not isinstance(diffusion_samples, int)
            or diffusion_samples < 1
            or diffusion_samples > 5
        ):
            return ToolResult.error(
                f"diffusion_samples must be an integer between 1 and 5, got: {diffusion_samples}"
            )

        try:
            # Determine mode: complex (molecules array) vs simple (sequence/sequence_ref)
            if molecules_input:
                # Complex mode: process molecules array
                if not isinstance(molecules_input, list):
                    return ToolResult.error(
                        "molecules must be an array of molecule objects."
                    )
                if len(molecules_input) == 0:
                    return ToolResult.error("molecules array cannot be empty.")
                if len(molecules_input) > 32:
                    return ToolResult.error(
                        "molecules array cannot contain more than 32 molecules."
                    )

                api_molecules, combined_metadata = await self._process_molecules(
                    molecules_input, msa_ref=msa_ref
                )

            elif sequence_ref or raw_sequence:
                # Simple mode: single protein backward compatibility
                simple_mol = {
                    "type": "protein",
                    "sequence_ref": sequence_ref,
                    "sequence": raw_sequence,
                }
                api_molecules, combined_metadata = await self._process_molecules(
                    [simple_mol], msa_ref=msa_ref
                )

            else:
                return ToolResult.error(
                    "Provide either 'molecules' array for complex predictions, "
                    "or 'sequence_ref'/'sequence' for simple protein predictions."
                )

            # Build request ID from prediction_id or metadata
            request_id = prediction_id or combined_metadata.get(
                "gene_name", "prediction"
            )
            input_id = prediction_id or "complex_1"

            # Build API request
            request_data = {
                "request_id": request_id,
                "inputs": [
                    {
                        "input_id": input_id,
                        "molecules": api_molecules,
                        "output_format": output_format,
                        "diffusion_samples": diffusion_samples,
                    }
                ],
            }

            # Log the prediction
            mol_summary = ", ".join(combined_metadata["molecule_types"])
            logger.info(
                f"Calling OpenFold3 API: {mol_summary} ({len(api_molecules)} molecules)"
            )

            # Make the API call
            headers = {
                "Authorization": f"Bearer {self._get_api_key()}",
                "Content-Type": "application/json",
                "NVCF-POLL-SECONDS": "300",  # Allow up to 5 minutes for prediction
            }

            async with httpx.AsyncClient(timeout=350.0) as client:
                response = await client.post(
                    self.api_endpoint,
                    headers=headers,
                    json=request_data,
                )
                response.raise_for_status()
                result = response.json()

            # Extract structure from response
            outputs = result.get("outputs", [])
            if not outputs:
                return ToolResult.error(
                    "OpenFold3 returned no outputs. The prediction may have failed."
                )

            structures = outputs[0].get("structures_with_scores", [])
            if not structures:
                return ToolResult.error("OpenFold3 returned no predicted structures.")

            # Get the best structure (first one, ranked by confidence)
            structure_data = structures[0]
            structure_content = structure_data.get("structure", "")

            if not structure_content:
                return ToolResult.error("OpenFold3 returned empty structure data.")

            # Handle escaped newlines
            if "\\n" in structure_content:
                structure_content = structure_content.replace("\\n", "\n")

            # Normalize PDB: adds TER records between chains for Molstar compatibility
            if output_format == "pdb":
                structure_content = self._normalize_pdb(structure_content)

            # Extract confidence scores
            confidence_score = structure_data.get("confidence_score", 0.0)
            plddt_score = structure_data.get("complex_plddt_score", 0.0)
            pde_score = structure_data.get("complex_pde_score", 0.0)
            ptm_score = structure_data.get("ptm_score", 0.0)
            iptm_score = structure_data.get("iptm_score", 0.0)

            logger.info(
                f"OpenFold3 prediction complete: "
                f"confidence={confidence_score:.3f}, pLDDT={plddt_score:.1f}, "
                f"ipTM={iptm_score:.3f}"
            )

            # Store the structure
            if self._storage is None:
                return ToolResult.error(
                    "Storage service not available. Cannot store structure data."
                )

            # Build metadata for storage
            storage_metadata = {
                "prediction_id": prediction_id,
                "output_format": output_format,
                "confidence_score": confidence_score,
                "plddt_score": plddt_score,
                "pde_score": pde_score,
                "ptm_score": ptm_score,
                "iptm_score": iptm_score,
                **combined_metadata,
            }

            reference = await self._storage.store(
                data_type="pdb",  # Use pdb type for both formats
                content=structure_content.encode("utf-8"),
                metadata=storage_metadata,
                description=description,
            )

            logger.info(f"Stored structure as {reference}")

            # Offload the per-residue pLDDT profile as an array artifact so the
            # model never has to recover it from the structure's B-factors.
            plddt_ref = None
            if output_format == "pdb":
                plddt_ref = await self._store_per_residue_plddt(
                    structure_content, prediction_id or "", reference, description
                )
                if plddt_ref:
                    storage_metadata["per_residue_plddt_ref"] = plddt_ref

            # Build descriptive message
            if combined_metadata["has_ligand"]:
                source_desc = "protein-ligand complex"
            elif combined_metadata["protein_count"] > 1:
                source_desc = (
                    f"multi-chain complex ({combined_metadata['protein_count']} chains)"
                )
            else:
                source_desc = prediction_id or combined_metadata.get(
                    "gene_name", "protein"
                )

            message = format_storage_message(
                data_type="pdb",
                reference=reference,
                source_description=source_desc,
                accepting_tools=["visualize_pdb"],
            )

            # Add confidence scores with explanations
            message += (
                f"\n\nPrediction confidence scores:\n"
                f"- Overall confidence: {confidence_score:.3f}\n"
                f"- pLDDT score: {plddt_score:.1f} (local confidence, higher is better, max 100)\n"
                f"- pTM score: {ptm_score:.3f} (predicted TM-score for overall fold)\n"
                f"- ipTM score: {iptm_score:.3f} (interface TM-score"
            )

            # Add ipTM interpretation for docking
            if combined_metadata["has_ligand"]:
                message += ", key metric for docking quality"
            message += ")"

            if plddt_ref:
                message += (
                    f"\n- Per-residue pLDDT: stored as '{plddt_ref}' (one value "
                    f"per residue, 0-100). Load it with "
                    f"aevom.load_array('{plddt_ref}') to find low-confidence "
                    f"regions -- do not re-parse the structure's B-factors."
                )

            return ToolResult.data_reference(
                reference=reference,
                data_type="pdb",
                message=message,
                metadata=storage_metadata,
            )

        except ValueError as e:
            logger.error(f"Validation error: {e}")
            return ToolResult.error(str(e))
        except httpx.HTTPStatusError as e:
            logger.error(f"OpenFold3 API HTTP error: {e}")
            error_detail = ""
            try:
                error_json = e.response.json()
                if "detail" in error_json:
                    error_detail = f" Details: {error_json['detail']}"
            except Exception:
                if e.response.text:
                    error_detail = f" Response: {e.response.text[:300]}"
            return ToolResult.error(
                f"OpenFold3 API request failed with status {e.response.status_code}.{error_detail}"
            )
        except httpx.TimeoutException:
            logger.error("OpenFold3 API timeout")
            return ToolResult.error(
                "OpenFold3 prediction timed out. Complex predictions may take longer. "
                "Please try again later."
            )
        except httpx.RequestError as e:
            logger.error(f"OpenFold3 API request error: {e}")
            return ToolResult.error(
                f"Failed to connect to OpenFold3 API: {str(e)}. "
                "Please check your internet connection."
            )
        except Exception as e:
            logger.exception("Failed to predict structure")
            return ToolResult.error(f"Failed to predict structure: {str(e)}")

    def _normalize_pdb(self, pdb_content: str) -> str:
        """Round-trip PDB through Biopython to add TER/END records.

        OpenFold3 returns PDB files with no TER records between chains,
        which prevents Molstar from classifying HETATM chains as
        non-polymer entities. Biopython's PDBIO.save() automatically
        inserts TER and END records.
        """
        from Bio.PDB import PDBParser, PDBIO

        parser = PDBParser(QUIET=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(pdb_content)
            in_path = f.name

        try:
            struct = parser.get_structure("pred", in_path)
        finally:
            os.unlink(in_path)

        pdb_io = PDBIO()
        pdb_io.set_structure(struct)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            out_path = f.name

        try:
            pdb_io.save(out_path)
            with open(out_path, "r") as f:
                return f.read()
        finally:
            os.unlink(out_path)

    async def _store_per_residue_plddt(
        self,
        pdb_content: str,
        prediction_id: str,
        structure_ref: str,
        description: str | None,
    ) -> str | None:
        """Offload per-residue pLDDT (from the CA B-factor column) as an array.

        OpenFold3 reports only a *mean* pLDDT in its scores; the per-residue
        profile arrives inside the predicted PDB, in the CA atoms' B-factor
        column. Models that need "which regions are uncertain" otherwise reach
        into the structure to recover it and reliably trip over biotite's
        default parse, which drops B-factors unless asked for them. Storing the
        profile as a ready ``array`` artifact (the same pattern boltz2 and
        proteinmpnn use) removes that whole failure path -- the model just calls
        ``aevom.load_array(ref)``.

        Returns the array reference, or None if no CA B-factors were found or
        storage is unavailable. Never raises: a failure here must not sink an
        otherwise-successful prediction.
        """
        if self._storage is None:
            return None
        try:
            plddt = [
                float(line[60:66])
                for line in pdb_content.splitlines()
                if line.startswith("ATOM") and line[12:16].strip() == "CA"
            ]
            if not plddt:
                return None

            import io as _io

            import numpy as _np

            buffer = _io.BytesIO()
            _np.savez_compressed(buffer, data=_np.asarray(plddt, dtype=_np.float32))
            return await self._storage.store(
                data_type="array",
                content=buffer.getvalue(),
                metadata={
                    "kind": "per_residue_plddt",
                    "prediction_id": prediction_id,
                    "structure_ref": structure_ref,
                    "n_residues": len(plddt),
                },
                description=f"Per-residue pLDDT for {description or structure_ref}",
            )
        except Exception as e:  # noqa: BLE001 - offload is best-effort
            logger.warning(f"Could not offload per-residue pLDDT: {e}")
            return None
