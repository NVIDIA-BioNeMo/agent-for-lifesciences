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

"""Pydantic models for Vina service request/response types."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from .probe_grid import DEFAULT_PROBES


# =============================================================================
# Receptor Preparation Models
# =============================================================================


class PrepareReceptorRequest(BaseModel):
    """Request model for receptor preparation (PDB to PDBQT)."""

    pdb_content: str = Field(
        ..., description="PDB format string of the receptor protein"
    )
    remove_water: bool = Field(
        default=True, description="Remove water molecules from the structure"
    )
    remove_heteroatoms: bool = Field(
        default=False,
        description="Remove heteroatoms (ligands, cofactors). Keep False to preserve cofactors.",
    )
    ph: float = Field(
        default=7.4,
        ge=0.0,
        le=14.0,
        description="pH for protonation state determination",
    )


class PrepareReceptorResponse(BaseModel):
    """Response model for receptor preparation."""

    success: bool
    pdbqt: Optional[str] = Field(
        default=None, description="Prepared receptor in PDBQT format"
    )
    num_atoms: Optional[int] = Field(
        default=None, description="Number of atoms in prepared receptor"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if preparation failed"
    )


# =============================================================================
# Ligand Preparation Models
# =============================================================================


class PrepareLigandRequest(BaseModel):
    """Request model for ligand preparation (SDF/MOL to PDBQT)."""

    sdf_content: Optional[str] = Field(
        default=None, description="SDF/MOL format string of the ligand"
    )
    smiles: Optional[str] = Field(
        default=None,
        description="SMILES string (alternative to SDF, will generate 3D coords)",
    )
    add_hydrogens: bool = Field(
        default=True, description="Add hydrogens to the molecule"
    )
    ph: float = Field(
        default=7.4, ge=0.0, le=14.0, description="pH for protonation state"
    )


class PrepareLigandResponse(BaseModel):
    """Response model for ligand preparation."""

    success: bool
    pdbqt: Optional[str] = Field(
        default=None, description="Prepared ligand in PDBQT format"
    )
    num_atoms: Optional[int] = Field(
        default=None, description="Number of atoms in prepared ligand"
    )
    num_rotatable_bonds: Optional[int] = Field(
        default=None, description="Number of rotatable bonds (torsions)"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if preparation failed"
    )


# =============================================================================
# Docking Models
# =============================================================================


class DockRequest(BaseModel):
    """Request model for molecular docking."""

    receptor_pdbqt: str = Field(..., description="Prepared receptor in PDBQT format")
    ligand_pdbqt: str = Field(..., description="Prepared ligand in PDBQT format")

    # Bounding box center coordinates (Angstroms)
    center_x: float = Field(
        ..., description="X coordinate of search box center (Angstroms)"
    )
    center_y: float = Field(
        ..., description="Y coordinate of search box center (Angstroms)"
    )
    center_z: float = Field(
        ..., description="Z coordinate of search box center (Angstroms)"
    )

    # Bounding box dimensions (Angstroms)
    size_x: float = Field(
        default=20.0,
        gt=0,
        le=126.0,
        description="Search box size in X dimension (Angstroms)",
    )
    size_y: float = Field(
        default=20.0,
        gt=0,
        le=126.0,
        description="Search box size in Y dimension (Angstroms)",
    )
    size_z: float = Field(
        default=20.0,
        gt=0,
        le=126.0,
        description="Search box size in Z dimension (Angstroms)",
    )

    # Docking parameters
    exhaustiveness: int = Field(
        default=32,
        ge=1,
        le=64,
        description="Search thoroughness (higher = more accurate but slower)",
    )
    num_modes: int = Field(
        default=9,
        ge=1,
        le=20,
        description="Maximum number of binding poses to generate",
    )
    energy_range: float = Field(
        default=3.0,
        gt=0,
        description="Maximum energy difference from best pose (kcal/mol)",
    )
    score_mode: Literal["search", "local", "score"] = Field(
        default="search",
        description=(
            "What to do with the pose in the ligand PDBQT. 'search' ignores it and "
            "docks globally. 'local' minimises from it and keeps it, for a pose that "
            "already means something -- a constrained embedding, a crystal ligand, a "
            "scaffold hop off a known binder. 'score' evaluates it untouched. The "
            "pose-preserving modes run on CPU Vina and ignore exhaustiveness."
        ),
    )


class DockingPose(BaseModel):
    """A single docking pose result."""

    rank: int = Field(..., description="Pose rank (1 = best)")
    affinity: float = Field(
        ..., description="Binding affinity in kcal/mol (more negative = stronger)"
    )
    rmsd_lb: float = Field(default=0.0, description="RMSD lower bound from best pose")
    rmsd_ub: float = Field(default=0.0, description="RMSD upper bound from best pose")
    pdbqt: str = Field(..., description="Pose coordinates in PDBQT format")


class DockResponse(BaseModel):
    """Response model for docking results."""

    success: bool
    gpu_accelerated: bool = Field(
        default=False, description="Whether GPU acceleration was used"
    )
    execution_time_ms: int = Field(
        default=0, description="Docking execution time in milliseconds"
    )
    poses: list[DockingPose] = Field(
        default_factory=list, description="List of docking poses, sorted by affinity"
    )
    best_affinity: Optional[float] = Field(
        default=None, description="Best (most negative) binding affinity in kcal/mol"
    )
    error: Optional[str] = Field(
        default=None, description="Error message if docking failed"
    )
    docking_mode: str = Field(
        default="flexible",
        description=(
            "Strategy used: 'flexible', 'rigid_ensemble', or -- when the input pose "
            "was preserved -- 'local' or 'score'"
        ),
    )
    conformers_generated: int = Field(
        default=0,
        description="Number of conformers generated (0 for flexible docking)",
    )
    best_conformer_idx: int = Field(
        default=0,
        description="Index of the best-scoring conformer in the ensemble",
    )
    rotatable_bonds: Optional[int] = Field(
        default=None,
        description="Number of rotatable bonds in the ligand",
    )


# =============================================================================
# Auto-Dock Models (combined preparation + docking)
# =============================================================================


class DockAutoRequest(BaseModel):
    """Request model for auto-routing docking pipeline.

    Accepts raw PDB **or** pre-prepared PDBQT for the receptor, plus raw
    SDF for the ligand.  The pipeline auto-detects PDBQT input and skips
    preparation when the receptor is already prepared, avoiding
    double-conversion and false size-limit rejections.
    """

    receptor_pdb: str = Field(
        ...,
        description=(
            "Receptor structure content in PDB or PDBQT format. "
            "If PDBQT is detected, preparation is skipped automatically."
        ),
    )
    ligand_sdf: str = Field(..., description="Raw SDF/MOL content for the ligand")

    center_x: float = Field(
        ..., description="X coordinate of search box center (Angstroms)"
    )
    center_y: float = Field(
        ..., description="Y coordinate of search box center (Angstroms)"
    )
    center_z: float = Field(
        ..., description="Z coordinate of search box center (Angstroms)"
    )

    size_x: float = Field(
        default=20.0,
        gt=0,
        le=35.0,
        description="Search box size in X dimension (Angstroms). "
        "Uni-Dock crashes with dimensions > 35 A.",
    )
    size_y: float = Field(
        default=20.0,
        gt=0,
        le=35.0,
        description="Search box size in Y dimension (Angstroms). "
        "Uni-Dock crashes with dimensions > 35 A.",
    )
    size_z: float = Field(
        default=20.0,
        gt=0,
        le=35.0,
        description="Search box size in Z dimension (Angstroms). "
        "Uni-Dock crashes with dimensions > 35 A.",
    )

    exhaustiveness: int = Field(
        default=32,
        ge=1,
        le=64,
        description="Search thoroughness (higher = more accurate but slower)",
    )
    num_modes: int = Field(
        default=9,
        ge=1,
        le=20,
        description="Maximum number of binding poses to generate",
    )
    energy_range: float = Field(
        default=3.0,
        gt=0,
        description="Maximum energy difference from best pose (kcal/mol)",
    )
    num_conformers: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Number of conformers to generate for ensemble rigid docking",
    )
    rot_bond_threshold: int = Field(
        default=15,
        ge=0,
        le=100,
        description="Rotatable-bond count above which ensemble rigid docking is used",
    )
    score_mode: Literal["search", "local", "score"] = Field(
        default="search",
        description=(
            "What to do with the pose in the ligand SDF. 'search' ignores it and docks "
            "globally. 'local' minimises from it and keeps it. 'score' evaluates it "
            "untouched. The pose-preserving modes skip conformer generation entirely, "
            "since regenerating conformers would discard the pose being asked about."
        ),
    )


# =============================================================================
# Probe Grid Models
# =============================================================================


class ProbeGridRequest(BaseModel):
    """Request model for probe interaction-energy grids."""

    pdb_content: str = Field(
        ..., description="PDB format string of the receptor protein"
    )
    center: list[float] = Field(
        ..., min_length=3, max_length=3, description="Box centre [x, y, z] in angstrom"
    )
    size: list[float] = Field(
        default=[20.0, 20.0, 20.0],
        min_length=3,
        max_length=3,
        description="Box dimensions [x, y, z] in angstrom",
    )
    spacing: float = Field(
        default=0.5,
        ge=0.2,
        le=1.0,
        description="Grid spacing in angstrom. 0.5 matches AutoDock's default.",
    )
    probes: list[str] = Field(
        default=list(DEFAULT_PROBES),
        min_length=1,
        description=(
            "Probe atom types, named for Vina's XS types: C_H hydrophobic carbon, "
            "C_P polar carbon, N_P/N_A/N_D nitrogen, O_A/O_DA oxygen."
        ),
    )
    atom_typing: Literal["template", "pdbqt"] = Field(
        default="template",
        description=(
            "'template' types the 20 standard amino acids from known chemistry "
            "and falls back to the PDBQT for everything else. 'pdbqt' uses "
            "OpenBabel's perception throughout, reproducing Vina's own typing."
        ),
    )
    max_minima: int = Field(
        default=5, ge=1, le=50, description="Ranked minima to return per probe"
    )
    min_separation: float = Field(
        default=2.0,
        gt=0.0,
        description="Minimum distance in angstrom between two reported minima",
    )
    remove_water: bool = Field(
        default=True, description="Remove water molecules before typing"
    )
    remove_heteroatoms: bool = Field(
        default=False,
        description="Remove ligands and cofactors. Keep False to include cofactors.",
    )
    ph: float = Field(
        default=7.4,
        ge=0.0,
        le=14.0,
        description="pH for protonation state determination",
    )


class ProbeGridMinimum(BaseModel):
    """One ranked interaction hotspot."""

    probe: str = Field(..., description="Probe type that found this minimum")
    family: str = Field(
        ...,
        description=(
            "Pharmacophore family of the ligand atom belonging here. An acceptor "
            "probe sits next to a receptor donor, so its minima are Acceptor sites."
        ),
    )
    x: float
    y: float
    z: float
    energy: float = Field(..., description="Vina interaction energy in kcal/mol")


class ProbeGridResponse(BaseModel):
    """Response model for probe interaction-energy grids."""

    success: bool
    npz_base64: Optional[str] = Field(
        default=None,
        description=(
            "Base64 of a compressed .npz holding one array under the key 'data', "
            "shaped (n_probes, nx, ny, nz) in the order given by 'probes'."
        ),
    )
    probes: Optional[list[str]] = Field(default=None)
    origin: Optional[list[float]] = Field(
        default=None,
        description="Coordinate of grid index (0, 0, 0); point (i, j, k) is origin + spacing * (i, j, k)",
    )
    spacing: Optional[float] = Field(default=None)
    shape: Optional[list[int]] = Field(default=None, description="(nx, ny, nz)")
    minima: Optional[list[ProbeGridMinimum]] = Field(
        default=None, description="Ranked hotspots, most negative energy first"
    )
    num_receptor_atoms: Optional[int] = Field(
        default=None, description="Heavy atoms within reach of the box"
    )
    num_template_typed: Optional[int] = Field(
        default=None,
        description="How many of those were typed from residue templates rather than perceived",
    )
    error: Optional[str] = Field(default=None)


# =============================================================================
# Info/Health Models
# =============================================================================


class VinaInfoResponse(BaseModel):
    """Response model for service info."""

    vina_version: str = Field(..., description="Autodock Vina / Uni-Dock version")
    gpu_available: bool = Field(
        ..., description="Whether GPU docking will work (nvidia-smi detects GPU)"
    )
    gpu_enabled: bool = Field(
        ..., description="Whether GPU mode is enabled via VINA_GPU env var"
    )
    gpu_backend: str = Field(
        default="none",
        description="GPU docking backend: 'unidock' if GPU enabled and available, 'cpu_vina' if CPU-only, 'none' if GPU enabled but unavailable",
    )


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str = Field(default="healthy")
    service: str = Field(default="vina")
