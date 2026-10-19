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

"""Autodock Vina tool for molecular docking.

This tool provides physics-based molecular docking using the Vina
containerized service to predict binding poses and affinities.
"""

import logging
import os
import tempfile
from typing import Any

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)
from api.storage.envelope import ArtifactProvenance

from .base import ContainerTool, DataIOCategory, ToolResult

logger = logging.getLogger(__name__)


class VinaDockTool(ContainerTool):
    """Dock a ligand to a protein receptor using Autodock Vina.

    This tool performs physics-based molecular docking to predict
    binding poses and calculate binding affinities (kcal/mol).

    The tool:
    1. Retrieves receptor PDB and ligand SDF from storage
    2. Prepares both for docking (PDBQT format)
    3. Executes Vina docking with specified bounding box
    4. Stores the best docked pose and returns binding affinity

    Data Flow:
        Input: receptor_pdb_ref, ligand_sdf_ref, bounding box coordinates
        Output: Stores docked pose PDB, returns affinity score

    Example:
        vina_dock(
            receptor_pdb_ref="pdb_abc123",
            ligand_sdf_ref="sdf_xyz789",
            center_x=12.5,
            center_y=34.2,
            center_z=56.8,
            size_x=25.0,
            size_y=25.0,
            size_z=25.0
        )
    """

    @property
    def name(self) -> str:
        return "vina_dock"

    @property
    def summary_template(self) -> str:
        return "Docking ligand to receptor with Autodock Vina..."

    def format_summary(self, arguments: dict) -> str:
        """Custom summary with bounding box info."""
        center_x = arguments.get("center_x", "?")
        center_y = arguments.get("center_y", "?")
        center_z = arguments.get("center_z", "?")
        return f"Docking at ({center_x}, {center_y}, {center_z}) with Autodock Vina..."

    @property
    def service_url(self) -> str:
        """URL of the Vina service container."""
        return os.environ.get("VINA_SERVICE_URL", "http://vina:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        """This tool retrieves refs and stores results."""
        return DataIOCategory.BOTH

    @property
    def is_async(self) -> bool:
        """Docking can take a while; we still wait for it, but opt into
        JobManager tracking (when a JobManager is present -- see ADR 0001 and
        ToolRegistry._execute_impl) so this call's status is recorded as a
        real Job row instead of only appearing as an OTel span. This is the
        P0 reference implementation for durable job semantics; behavior is
        unchanged for callers that don't pass a JobManager, which today is
        every path except the production chat endpoint."""
        return True

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "vina_dock",
                "description": (
                    "Dock a small molecule ligand to a protein receptor using "
                    "Autodock Vina physics-based docking. Returns binding affinity "
                    "in kcal/mol (more negative = stronger binding).\n\n"
                    "REQUIRED INPUTS:\n"
                    "- receptor_pdb_ref: Reference to protein structure (from openfold3 or other source)\n"
                    "- ligand_sdf_ref: Reference to ligand structure (from aevom.download_pubchem_sdf or generate_3d)\n"
                    "- Bounding box: center coordinates (x,y,z) and size in Angstroms\n\n"
                    "BOUNDING BOX:\n"
                    "The search box defines where Vina looks for binding poses. Center it on:\n"
                    "- The known binding site (if ligand position is known from OpenFold3 complex)\n"
                    "- A specific residue or active site\n"
                    "- Use python_execute to calculate center from atom coordinates if needed\n\n"
                    "OUTPUT: Binding affinity score in kcal/mol and docked pose reference.\n\n"
                    "INTERPRETING AFFINITIES:\n"
                    "- > -6 kcal/mol: Weak binding\n"
                    "- -6 to -9 kcal/mol: Moderate binding (typical drug range)\n"
                    "- < -9 kcal/mol: Strong binding\n"
                    "- Typical drug affinities: -6 to -12 kcal/mol"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "receptor_pdb_ref": {
                            "type": "string",
                            "description": (
                                "Reference to receptor PDB structure (e.g., 'pdb_abc123'). "
                                "This should be the protein-only structure, typically from openfold3 "
                                "or a prepared receptor file."
                            ),
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "ligand_sdf_ref": {
                            "type": "string",
                            "description": (
                                "Reference to ligand SDF structure (e.g., 'sdf_xyz789'). "
                                "Get this from aevom.download_pubchem_sdf() or generate_3d_coordinates."
                            ),
                            "x-aevom-reference": {"data_type": "sdf"},
                        },
                        "center_x": {
                            "type": "number",
                            "description": (
                                "X coordinate of search box center (Angstroms). "
                                "This should be at the binding site location."
                            ),
                        },
                        "center_y": {
                            "type": "number",
                            "description": "Y coordinate of search box center (Angstroms).",
                        },
                        "center_z": {
                            "type": "number",
                            "description": "Z coordinate of search box center (Angstroms).",
                        },
                        "size_x": {
                            "type": "number",
                            "description": (
                                "Search box size in X dimension (Angstroms). "
                                "Default: 20. Larger boxes = more search area but slower."
                            ),
                        },
                        "size_y": {
                            "type": "number",
                            "description": "Search box size in Y dimension (Angstroms). Default: 20.",
                        },
                        "size_z": {
                            "type": "number",
                            "description": "Search box size in Z dimension (Angstroms). Default: 20.",
                        },
                        "exhaustiveness": {
                            "type": "integer",
                            "description": (
                                "Search thoroughness from 1 to 64. Default: 16. "
                                "Higher = more accurate but slower. "
                                "Use 8-16 for quick screening, 32-64 for production."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short description of the docking result for reference tracking "
                                "(e.g., 'Aspirin docked to COX-2 active site'). "
                                "This helps identify the stored complex later."
                            ),
                        },
                    },
                    "required": [
                        "receptor_pdb_ref",
                        "ligand_sdf_ref",
                        "center_x",
                        "center_y",
                        "center_z",
                    ],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute Vina docking via the auto-routing pipeline.

        Sends receptor content (PDB or pre-prepared PDBQT) and ligand
        SDF to the ``/dock-auto`` endpoint which auto-detects the
        receptor format, skips preparation for PDBQT, and selects the
        optimal docking strategy (flexible vs rigid ensemble).
        """
        receptor_pdb_ref = arguments.get("receptor_pdb_ref", "")
        ligand_sdf_ref = arguments.get("ligand_sdf_ref", "")
        center_x = arguments.get("center_x")
        center_y = arguments.get("center_y")
        center_z = arguments.get("center_z")
        size_x = arguments.get("size_x", 20.0)
        size_y = arguments.get("size_y", 20.0)
        size_z = arguments.get("size_z", 20.0)
        exhaustiveness = arguments.get("exhaustiveness", 16)
        description = arguments.get("description")

        if not receptor_pdb_ref:
            return ToolResult.error("receptor_pdb_ref is required")
        if not ligand_sdf_ref:
            return ToolResult.error("ligand_sdf_ref is required")
        if center_x is None or center_y is None or center_z is None:
            return ToolResult.error(
                "center_x, center_y, and center_z are required. "
                "These define the center of the docking search box."
            )

        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve molecular data."
            )

        try:
            # 1. Retrieve receptor PDB from storage
            logger.info(f"Retrieving receptor from {receptor_pdb_ref}")
            try:
                receptor_pdb = await self._storage.retrieve(receptor_pdb_ref)
                receptor_pdb_content = receptor_pdb.decode("utf-8")
            except InvalidReferenceError:
                return ToolResult.error(
                    f"Invalid receptor reference: '{receptor_pdb_ref}'. "
                    "Use fetch_and_clean_pdb to download a PDB first, which returns "
                    "a reference like 'pdb_abc12345'."
                )
            except ArtifactNotFoundError:
                return ToolResult.error(
                    f"Receptor not found: '{receptor_pdb_ref}'. "
                    "Use openfold3 to predict a protein structure first."
                )

            # 2. Retrieve ligand SDF from storage
            logger.info(f"Retrieving ligand from {ligand_sdf_ref}")

            if not ligand_sdf_ref.startswith("sdf_") or len(ligand_sdf_ref) < 12:
                if ligand_sdf_ref.isdigit():
                    return ToolResult.error(
                        f"Invalid ligand reference: '{ligand_sdf_ref}' appears to be a PubChem CID, not an SDF reference. "
                        f"Use aevom.download_pubchem_sdf({ligand_sdf_ref}) to download and store the SDF, "
                        "which returns a proper reference like 'sdf_abc12345'."
                    )
                return ToolResult.error(
                    f"Invalid ligand reference format: '{ligand_sdf_ref}'. "
                    "Expected format: 'sdf_xxxxxxxx' (e.g., 'sdf_abc12345'). "
                    "Use aevom.download_pubchem_sdf() or generate_3d_coordinates to get a valid SDF reference."
                )

            try:
                ligand_sdf = await self._storage.retrieve(ligand_sdf_ref)
                ligand_sdf_content = ligand_sdf.decode("utf-8")
            except InvalidReferenceError:
                return ToolResult.error(
                    f"Invalid ligand reference: '{ligand_sdf_ref}'. "
                    "Use download_pubchem_sdf or generate_3d_coordinates to create an SDF, "
                    "which returns a reference like 'sdf_abc12345'."
                )
            except ArtifactNotFoundError:
                return ToolResult.error(
                    f"Ligand not found: '{ligand_sdf_ref}'. "
                    "The reference looks valid but the data doesn't exist in storage. "
                    "Use aevom.download_pubchem_sdf() or generate_3d_coordinates to create a new SDF reference."
                )

            # 3. Call /dock-auto — single request handles preparation,
            #    strategy selection, conformer generation, and docking.
            num_conformers = int(os.environ.get("CONFORMER_COUNT", "50"))
            rot_bond_threshold = int(os.environ.get("ROT_BOND_THRESHOLD", "15"))

            logger.info(
                f"Calling /dock-auto: center=({center_x}, {center_y}, {center_z}), "
                f"exhaustiveness={exhaustiveness}"
            )
            try:
                dock_response = await self._call_service(
                    endpoint="/dock-auto",
                    method="POST",
                    json_data={
                        "receptor_pdb": receptor_pdb_content,
                        "ligand_sdf": ligand_sdf_content,
                        "center_x": float(center_x),
                        "center_y": float(center_y),
                        "center_z": float(center_z),
                        "size_x": float(size_x),
                        "size_y": float(size_y),
                        "size_z": float(size_z),
                        "exhaustiveness": int(exhaustiveness),
                        "num_modes": 9,
                        "num_conformers": num_conformers,
                        "rot_bond_threshold": rot_bond_threshold,
                    },
                    timeout=600.0,
                )

                if not dock_response.get("success"):
                    error = dock_response.get("error", "Unknown docking error")
                    return ToolResult.error(f"Docking failed: {error}")

            except Exception as e:
                logger.error(f"Docking failed: {e}")
                return ToolResult.error(f"Docking execution failed: {str(e)}")

            # 4. Extract results
            poses = dock_response.get("poses", [])
            if not poses:
                return ToolResult.error("Docking completed but no poses were found")

            best_affinity = dock_response.get("best_affinity")
            gpu_used = dock_response.get("gpu_accelerated", False)
            exec_time = dock_response.get("execution_time_ms", 0)
            docking_mode = dock_response.get("docking_mode", "flexible")
            conformers_generated = dock_response.get("conformers_generated", 0)
            rotatable_bonds = dock_response.get("rotatable_bonds")

            best_pose_pdbqt = poses[0].get("pdbqt", "")
            if not best_pose_pdbqt:
                return ToolResult.error("Best pose has no coordinate data")

            # 5. Parse best pose PDBQT into a Biopython Residue, then
            #    combine with the receptor into a single complex PDB.
            ligand_residue = self._parse_pdbqt_ligand(best_pose_pdbqt)

            logger.info(
                "Combining receptor and docked ligand into complex using Biopython..."
            )
            complex_pdb = self._combine_receptor_ligand_pdb(
                receptor_pdb=receptor_pdb_content,
                ligand_residue=ligand_residue,
                ligand_chain_id="Z",
            )

            # 6. Store the docked complex
            pose_metadata = {
                "source": "vina_dock",
                "affinity_kcal_mol": best_affinity,
                "num_poses": len(poses),
                "gpu_accelerated": gpu_used,
                "execution_time_ms": exec_time,
                "receptor_ref": receptor_pdb_ref,
                "ligand_ref": ligand_sdf_ref,
                "ligand_chain": "Z",
                "center": [center_x, center_y, center_z],
                "size": [size_x, size_y, size_z],
                "exhaustiveness": exhaustiveness,
                "is_complex": True,
                "docking_mode": docking_mode,
                "conformers_generated": conformers_generated,
                "rotatable_bonds": rotatable_bonds,
                # Scientific artifact envelope (ADR 0001): units and lineage,
                # lifted into dedicated DataArtifact columns by the storage
                # repository. Additive -- metadata_json keeps every field
                # above unchanged for existing readers.
                "units": {"affinity": "kcal/mol"},
                "provenance": ArtifactProvenance(
                    source_tool=self.name,
                    input_refs=[receptor_pdb_ref, ligand_sdf_ref],
                    parameters={
                        "center": [center_x, center_y, center_z],
                        "size": [size_x, size_y, size_z],
                        "exhaustiveness": exhaustiveness,
                    },
                ).to_dict(),
            }

            pose_ref = await self._storage.store(
                data_type="pdb",
                content=complex_pdb.encode("utf-8"),
                metadata=pose_metadata,
                description=description,
            )

            logger.info(
                f"Docking complete: affinity={best_affinity:.2f} kcal/mol, "
                f"mode={docking_mode}, stored complex as {pose_ref}"
            )

            # 7. Build result message
            message = format_storage_message(
                data_type="pdb",
                reference=pose_ref,
                source_description="receptor-ligand docked complex",
                accepting_tools=["visualize_pdb"],
            )

            lig_resname = ligand_residue.resname
            message += (
                f"\n\nDocking Results:\n"
                f"- Best binding affinity: {best_affinity:.2f} kcal/mol\n"
                f"- Number of poses found: {len(poses)}\n"
                f"- Execution time: {exec_time}ms\n"
                f"- Complex structure: Receptor + Ligand (chain Z)\n"
                f"- Ligand residue name in complex: {lig_resname} (chain Z, residue 1)\n"
                f"- Docking mode: {docking_mode}\n"
            )

            if docking_mode == "rigid_ensemble":
                message += f"- Conformers generated: {conformers_generated}\n"
            if rotatable_bonds is not None:
                message += f"- Rotatable bonds: {rotatable_bonds}\n"
            if gpu_used:
                message += "- GPU acceleration: Yes\n"

            if best_affinity > -6:
                message += "\nInterpretation: Weak binding (> -6 kcal/mol)"
            elif best_affinity > -9:
                message += "\nInterpretation: Moderate binding (-6 to -9 kcal/mol, typical drug range)"
            else:
                message += "\nInterpretation: Strong binding (< -9 kcal/mol)"

            return ToolResult.data_reference(
                reference=pose_ref,
                data_type="pdb",
                message=message,
                metadata=pose_metadata,
            )

        except Exception as e:
            logger.exception("Vina docking failed")
            return ToolResult.error(f"Docking failed: {str(e)}")

    def _parse_pdbqt_ligand(self, pdbqt_content: str, resname: str = "UNL"):
        """Parse PDBQT and build a Biopython Residue with all ligand atoms.

        Handles Vina-specific atom types (A, OA, HD, NA, etc.) and generates
        unique atom names. Returns a single HETATM Residue (resseq=1) that
        can be passed directly to ``_combine_receptor_ligand_pdb``.

        This replaces the old ``_pdbqt_to_pdb`` approach which used brittle
        manual PDB column formatting, causing per-atom residue numbering that
        broke Molstar entity classification.

        Args:
            pdbqt_content: PDBQT format string from Vina.
            resname: Residue name for the ligand (default ``UNL``).

        Returns:
            A ``Bio.PDB.Residue.Residue`` containing all ligand atoms.
        """
        import numpy
        from Bio.PDB.Atom import Atom
        from Bio.PDB.Residue import Residue

        residue = Residue(("H_" + resname, 1, " "), resname, " ")
        element_counts: dict[str, int] = {}
        skip_keywords = {"ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF"}
        serial = 0

        for line in pdbqt_content.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if not parts:
                continue

            keyword = parts[0]
            if keyword in skip_keywords:
                continue

            if keyword in ("ATOM", "HETATM"):
                vina_type = ""
                if len(line) > 77:
                    vina_type = line[77:].split()[0] if line[77:].strip() else ""

                element = self._vina_type_to_element(vina_type)

                element_counts[element] = element_counts.get(element, 0) + 1
                count = element_counts[element]

                if len(element) == 1:
                    atom_name = f" {element}{count:<2d}"
                else:
                    atom_name = f"{element}{count:<2d}"
                atom_name = atom_name[:4]

                try:
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except (ValueError, IndexError):
                    continue

                serial += 1
                atom = Atom(
                    name=atom_name.strip(),
                    coord=numpy.array([x, y, z], dtype="f"),
                    bfactor=0.0,
                    occupancy=1.0,
                    altloc=" ",
                    fullname=atom_name,
                    serial_number=serial,
                    element=element,
                )
                residue.add(atom)

        return residue

    def _vina_type_to_element(self, vina_type: str) -> str:
        """Convert Vina atom type to element symbol.

        Vina/AutoDock atom types include:
        - A: Aromatic carbon
        - C: Aliphatic carbon
        - NA: Nitrogen H-bond acceptor
        - N: Nitrogen (other)
        - OA: Oxygen H-bond acceptor
        - O: Oxygen (other)
        - SA: Sulfur H-bond acceptor
        - S: Sulfur (other)
        - HD: Hydrogen H-bond donor (polar H)
        - H: Hydrogen (other)
        - F, Cl, Br, I: Halogens
        - P: Phosphorus
        - Fe, Zn, Mg, Ca, Mn: Metals

        Args:
            vina_type: Vina atom type string.

        Returns:
            Element symbol (1-2 characters).
        """
        vina_type = vina_type.strip().upper()

        # Map Vina types to elements
        type_map = {
            "A": "C",  # Aromatic carbon
            "C": "C",  # Carbon
            "NA": "N",  # Nitrogen acceptor
            "N": "N",  # Nitrogen
            "OA": "O",  # Oxygen acceptor
            "O": "O",  # Oxygen
            "SA": "S",  # Sulfur acceptor
            "S": "S",  # Sulfur
            "HD": "H",  # Hydrogen donor
            "H": "H",  # Hydrogen
            "F": "F",  # Fluorine
            "CL": "CL",  # Chlorine
            "BR": "BR",  # Bromine
            "I": "I",  # Iodine
            "P": "P",  # Phosphorus
            "FE": "FE",  # Iron
            "ZN": "ZN",  # Zinc
            "MG": "MG",  # Magnesium
            "CA": "CA",  # Calcium (element, not alpha carbon)
            "MN": "MN",  # Manganese
        }

        if vina_type in type_map:
            return type_map[vina_type]

        # Fallback: use first 1-2 characters
        if vina_type:
            return (
                vina_type[:2]
                if len(vina_type) >= 2 and vina_type[1].isalpha()
                else vina_type[0]
            )

        return "C"  # Default to carbon

    def _combine_receptor_ligand_pdb(
        self,
        receptor_pdb: str,
        ligand_residue,
        ligand_chain_id: str = "Z",
    ) -> str:
        """Combine receptor and docked ligand into a single complex PDB.

        Accepts a pre-built Biopython ``Residue`` for the ligand (from
        ``_parse_pdbqt_ligand``) and the receptor as a PDB string.
        ``PDBIO.save()`` handles TER records, serial renumbering, and
        column alignment automatically.

        Args:
            receptor_pdb: PDB content of the receptor (protein).
            ligand_residue: Biopython ``Residue`` object for the ligand.
            ligand_chain_id: Chain ID to assign to the ligand (default ``Z``).

        Returns:
            Combined PDB content with receptor and ligand.
        """
        from Bio.PDB import PDBParser, PDBIO
        from Bio.PDB.Chain import Chain
        from Bio.PDB.Residue import Residue

        parser = PDBParser(QUIET=True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(receptor_pdb)
            receptor_path = f.name

        try:
            receptor_struct = parser.get_structure("receptor", receptor_path)
        finally:
            os.unlink(receptor_path)

        receptor_model = receptor_struct[0]

        receptor_chain_ids = {chain.id for chain in receptor_model}
        if ligand_chain_id in receptor_chain_ids:
            for candidate in "ZYXWVUTSRQPONMLKJIHGFEDCB":
                if candidate not in receptor_chain_ids:
                    ligand_chain_id = candidate
                    break

        new_chain = Chain(ligand_chain_id)

        new_residue = Residue(
            ligand_residue.id, ligand_residue.resname, ligand_residue.segid
        )
        for atom in ligand_residue:
            new_residue.add(atom.copy())
        new_chain.add(new_residue)

        receptor_model.add(new_chain)

        pdb_io = PDBIO()
        pdb_io.set_structure(receptor_struct)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            output_path = f.name

        try:
            pdb_io.save(output_path)
            with open(output_path, "r") as f:
                combined_pdb = f.read()
        finally:
            os.unlink(output_path)

        return combined_pdb
