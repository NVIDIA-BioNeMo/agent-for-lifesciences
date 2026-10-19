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

"""Dataset adapter for converting aevom-data-gen output to NeMo-Gym format.

This module converts the CDD (Computational Drug Discovery) dataset format
produced by aevom-data-gen into JSONL format compatible with NeMo-Gym's
rollout collection infrastructure.
"""

import json
import logging
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    from api.storage import generate_reference_id
except ImportError:

    def generate_reference_id(data_type: str, length: int = 8) -> str:
        return f"{data_type}_{secrets.token_hex(length // 2)}"

logger = logging.getLogger(__name__)


@dataclass
class MoleculePair:
    """A winner/loser molecule pair from the CDD dataset."""

    complex_id: str
    pdb_id: str

    # Winner molecule info
    winner_sdf_path: Path
    winner_smiles: str
    winner_vina: float
    winner_metrics: dict[str, Any]

    # Loser molecule info
    loser_sdf_path: Path
    loser_smiles: Optional[str]
    loser_vina: float
    loser_metrics: dict[str, Any]

    # Sabotage info (ground truth)
    sabotage_type: str
    pattern_name: str
    modification_desc: str
    focus_atom_idx: Optional[int]
    smarts_pattern: Optional[str]
    tanimoto_similarity: float
    delta_vina: float

    # Receptor
    receptor_pdbqt_path: Path
    receptor_pdb_path: Optional[Path] = None

    # Binding site bounding box (from metadata)
    box_center: Optional[list[float]] = None
    box_size: Optional[list[float]] = None


@dataclass
class TrainingTask:
    """A training task in NeMo-Gym format."""

    task_id: str
    responses_create_params: dict[str, Any]
    ground_truth: dict[str, Any]
    preload_refs: dict[str, str]

    def to_jsonl_dict(self) -> dict:
        """Convert to dictionary for JSONL serialization."""
        return {
            "task_id": self.task_id,
            "responses_create_params": self.responses_create_params,
            "ground_truth": self.ground_truth,
            "preload_refs": self.preload_refs,
            "agent_ref": {
                "type": "responses_api_agents",
                "name": "aevom_simple_agent",
            },
        }


# Default system prompt for training tasks
DEFAULT_SYSTEM_PROMPT = """You are **Aevom**, an expert AI Computational Chemist. Use your tools to analyze molecules -- do NOT just describe what you would do, actually call the tools.

## Your Tools (call these directly)
- `vina_dock`: Dock a ligand to a receptor. Pass `receptor_pdb_ref`, `ligand_sdf_ref`, and binding site coordinates `center_x`, `center_y`, `center_z`. Returns binding affinity (kcal/mol, more negative = stronger).
- `python_execute`: Run Python code with RDKit/NumPy in a sandbox. Each call is independent -- always re-import modules and re-load data.
- `molecule_info`: Get molecular properties (MW, LogP, TPSA, QED, HBD, HBA) for a SMILES reference. Pass `smiles_ref`.
- `structure_query`: Execute a MolQL selection against a specific PDB structure identified by `pdb_ref`. Returns atom-level data that can be passed to python_execute for geometric analysis. Pass the docked-complex `pdb_ref` returned by `vina_dock` to query the bound pose, or the apo receptor ref to query the empty pocket.
- `visualize_pdb`: Load a PDB structure into the viewer (display only).

## python_execute: Loading Data and Common Patterns
In python_execute, use the pre-loaded `aevom` client to access stored molecules:
```python
# Load SDF as RDKit Mol object
mol = aevom.load_rdkit_mol("sdf_abc12345")

# Load raw file content as string
data = aevom.load("pdb_abc12345")

# Get canonical SMILES
smiles = aevom.canonical_smiles(mol)
```

To store a molecule you have edited (returns a new ref you can pass to vina_dock):
```python
new_ref = aevom.store_sdf(Chem.MolToMolBlock(edited_mol))    # RDKit Mol -> sdf_*
new_ref = aevom.store_smiles("CCO")                          # SMILES -> smiles_*
```

IMPORTANT: Each python_execute call starts fresh. Always include ALL imports:
```python
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
import numpy as np
```

## MolQL Selection Queries
Use `structure_query(pdb_ref=..., molql=...)` to select atoms in a specific PDB structure. `pdb_ref` is REQUIRED -- pass the docked-complex ref returned by `vina_dock` (to access both protein and ligand atoms in the bound pose), or the apo receptor ref. The selection data is stored and can be passed to `python_execute` for geometric analysis.

**MolQL Cheatsheet:**
| Query | MolQL |
| --- | --- |
| Ligand (non-polymer) | `(sel.atom.res (= atom.entity-type non-polymer))` |
| Binding pocket (5Å) | `(sel.atom.include-surroundings (sel.atom.res (= atom.entity-type non-polymer)) :radius 5 :as-whole-residues true)` |
| Specific chain | `(sel.atom.chains (= atom.chain A))` |
| Residue range | `(sel.atom.res (in-range atom.resno 50 100))` |
| Alpha carbons | `(sel.atom.atoms (set.has (set CA C N O) atom.name))` |
| Aromatic residues | `(sel.atom.res (set.has (set PHE TRP TYR HIS) atom.resname))` |
| Charged residues | `(sel.atom.res (set.has (set ASP GLU LYS ARG) atom.resname))` |

**Workflow: structure_query -> python_execute**
1. `structure_query(pdb_ref="pdb_<docked_complex_or_receptor>", molql="(sel.atom.include-surroundings (sel.atom.res (= atom.entity-type non-polymer)) :radius 5 :as-whole-residues true)")`
   -> Returns: "Selection active: 214 atoms in 27 residues. Reference: selection_abc123."
2. `python_execute(code="...", selection_ref="selection_abc123")`
   -> The `selection` variable contains atom coordinates, residue info, etc.

**Selection data structure** (available as `selection` in python_execute):
```python
selection = {
    "atoms": [{"name": "CA", "element": "C", "id": 42, "coords": [x, y, z],
               "residue": {"name": "ALA", "seqNumber": 45, "chainId": "A", "entityType": "polymer"}}, ...],
    "atom_count": 214, "residue_count": 27
}
```

## Workflow
1. Dock BOTH molecules with `vina_dock` using the coordinates from the task prompt -- each call returns a docked-complex `pdb_ref`
2. Use `structure_query(pdb_ref=<docked_complex_ref>, molql=...)` against the SPECIFIC docked complex you want to analyze (the loser pose, the winner pose, etc.) -- always pass the explicit `pdb_ref` so the selection comes from the structure you intend
3. Optionally `visualize_pdb(pdb_ref=<docked_complex_ref>)` if you want the user to see it
4. Use `molecule_info` on both SMILES references to compare properties
5. Use `python_execute` for structural comparison (fingerprints, substructure analysis, binding pocket geometry)
6. Provide a clear diagnosis explaining the molecular basis of the binding difference

Reference IDs (e.g. `sdf_abc12345`, `pdbqt_xyz789`) are passed directly as tool arguments.
"""


_cached_tool_schemas: Optional[list[dict]] = None


def get_training_tool_schemas() -> list[dict]:
    """Pull tool schemas from the real production tool classes.

    Converts from the OpenAI Chat Completion format (used in production)
    to the Responses API FunctionToolParam format (used by NeMo-Gym),
    and adds ``strict: True`` + ``additionalProperties: False`` required
    by NeMo-Gym's schema validation.

    If real tool classes cannot be imported (e.g. running outside the
    backend package), falls back to minimal hardcoded schemas so data
    generation can still work in standalone mode.

    Results are cached after the first call.
    """
    global _cached_tool_schemas
    if _cached_tool_schemas is not None:
        return _cached_tool_schemas

    try:
        from api.tools.python_exec import PythonExecuteTool
        from api.tools.vina import VinaDockTool
        from api.tools.rdkit import RDKitMoleculeInfoTool
        from api.tools.molstar_queries import MolstarQueryTool
        from api.tools.visualization import VisualizePdbTool

        tools = [PythonExecuteTool(), VinaDockTool(), RDKitMoleculeInfoTool(), MolstarQueryTool(), VisualizePdbTool()]
        schemas = []
        for tool in tools:
            raw = tool.get_schema()
            func = raw.get("function", raw)
            params = func.get("parameters", {})
            clean_props = {}
            for pname, pdef in params.get("properties", {}).items():
                clean_props[pname] = {
                    k: v for k, v in pdef.items() if not k.startswith("x-")
                }
            clean_params = {
                "type": params.get("type", "object"),
                "properties": clean_props,
                "required": params.get("required", []),
                "additionalProperties": False,
            }
            schemas.append(
                {
                    "type": "function",
                    "name": func["name"],
                    "strict": True,
                    "description": func.get("description", ""),
                    "parameters": clean_params,
                }
            )
        _cached_tool_schemas = schemas
        return schemas
    except ImportError:
        logger.warning(
            "Could not import real tool classes -- using minimal fallback schemas. "
            "Run from within the backend package for full schemas."
        )
        _cached_tool_schemas = _FALLBACK_TOOL_SCHEMAS
        return _cached_tool_schemas


# Fallback schemas that mirror the real production tool interfaces exactly.
# These MUST stay in sync with the tool classes in api/tools/.
_FALLBACK_TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": "vina_dock",
        "strict": True,
        "description": (
            "Dock a small molecule ligand to a protein receptor using Autodock Vina "
            "physics-based docking. Returns binding affinity in kcal/mol "
            "(more negative = stronger binding)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "receptor_pdb_ref": {
                    "type": "string",
                    "description": (
                        "Reference to receptor PDB structure (e.g., 'pdb_abc123'). "
                        "This should be the protein-only structure."
                    ),
                },
                "ligand_sdf_ref": {
                    "type": "string",
                    "description": (
                        "Reference to ligand SDF structure (e.g., 'sdf_xyz789')."
                    ),
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
                        "Higher = more accurate but slower."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": (
                        "Short description of the docking result for reference tracking "
                        "(e.g., 'Aspirin docked to COX-2 active site')."
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
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "python_execute",
        "strict": True,
        "description": "Execute Python code in a secure sandbox to analyze molecular data.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "molecule_info",
        "strict": True,
        "description": (
            "Get detailed molecular properties and information from a stored SMILES reference. "
            "Returns properties like molecular weight, formula, LogP, TPSA, "
            "number of hydrogen bond donors/acceptors, rotatable bonds, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles_ref": {
                    "type": "string",
                    "description": (
                        "Reference to a stored SMILES string (e.g., 'smiles_abc123')."
                    ),
                },
            },
            "required": ["smiles_ref"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "structure_query",
        "strict": True,
        "description": (
            "Execute a MolQL selection query against the PDB structure identified "
            "by pdb_ref. Returns selection statistics and a reference for "
            "python_execute analysis. Use MolQL S-expression syntax, e.g. "
            "(sel.atom.res (= atom.entity-type non-polymer))."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_ref": {
                    "type": "string",
                    "description": (
                        "Reference to the PDB structure to query (e.g., 'pdb_abc12345'). "
                        "Use the docked-complex reference returned by vina_dock to "
                        "analyze a bound pose, or the original receptor reference "
                        "for apo analysis."
                    ),
                },
                "molql": {
                    "type": "string",
                    "description": (
                        "MolQL S-expression query string. Must start with '('. "
                        "Example: (sel.atom.res (= atom.entity-type non-polymer))"
                    ),
                },
            },
            "required": ["pdb_ref", "molql"],
            "additionalProperties": False,
        },
    },
    {
        "type": "function",
        "name": "visualize_pdb",
        "strict": True,
        "description": (
            "Display a protein structure in the interactive 3D viewer (Molstar) "
            "for the user. Optional. structure_query no longer requires the viewer "
            "to be loaded -- it accepts pdb_ref directly."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_ref": {
                    "type": "string",
                    "description": (
                        "Reference to stored PDB data (e.g., 'pdb_abc123'). "
                        "Use the complex PDB reference returned by vina_dock."
                    ),
                },
                "label": {
                    "type": "string",
                    "description": (
                        "Display label for the protein structure "
                        "(e.g., 'Docked complex'). Optional."
                    ),
                },
            },
            "required": ["pdb_ref"],
            "additionalProperties": False,
        },
    },
]

# Task prompt template
TASK_PROMPT_TEMPLATE = """Here are two molecules: @ref:{winner_ref} and @ref:{loser_ref}

One has high binding activity and one has poor activity against the target protein (receptor: @ref:{receptor_ref}, PDB structure: @ref:{receptor_pdb_ref}).

The binding site center is at coordinates ({center_x}, {center_y}, {center_z}). Use these as the docking box center when calling vina_dock.

Diagnose why these molecules differ in binding affinity. Use docking, structure queries on the receptor, and analysis programs you write in python to support your conclusion.

When you plan, be decisive rather than going around and around in circles. Be scientific, analytical and precise. 
"""

# Advanced mode task prompt (metric-grounded hypothesis -> edit -> redock)
ADVANCED_TASK_PROMPT_TEMPLATE = """Here are two molecules: @ref:{winner_ref} and @ref:{loser_ref}

One has high binding activity and one has poor activity against the target protein (receptor: @ref:{receptor_ref}, PDB structure: @ref:{receptor_pdb_ref}).

The binding site center is at coordinates ({center_x}, {center_y}, {center_z}). Use these as the docking box center when calling vina_dock.

You will be evaluated on the full advanced-mode loop. Complete every step:

**Step 1 -- Quantify the difference.**
Use `python_execute` with RDKit to compute at least one numerical metric (e.g. LogP, TPSA, MW, HBD, HBA, formal charge) for BOTH molecules and print them side by side. Choose a metric whose winner-vs-loser difference is large enough to plausibly explain the binding gap. Print the comparison in a clearly paired form, for example:
    `LogP winner: 3.4, loser: 6.1`
or
    `print(f"winner_tpsa = {tpsa_w}, loser_tpsa = {tpsa_l}")`

**Step 2 -- State a metric-grounded hypothesis.**
After you see the paired metric, state your hypothesis explicitly using the phrase "My hypothesis is:" and CITE the specific metric value that motivates it. Example:
    `My hypothesis is: the loser binds poorly because its LogP of 6.1 (vs 3.4 for the winner) makes it too hydrophobic for the polar binding pocket.`

**Step 3 -- Edit the loser to test the hypothesis.**
Use `python_execute` with RDKit to MODIFY the loser molecule in a way that should reverse the defect you identified. The edit must be a real chemical change designed to address the hypothesis -- do NOT simply regenerate or copy the winner molecule. Store the new molecule with `aevom.store_sdf(...)` (or `aevom.store_smiles(...)`) so you receive a new sdf reference.

**Step 4 -- Re-dock and report.**
Call `vina_dock` with the new sdf reference (and the same binding site coordinates) and report whether the binding affinity recovered toward the winner's score.

Be decisive and scientific -- avoid going around in circles.
"""


class CDDDatasetAdapter:
    """Adapter for converting CDD dataset to NeMo-Gym training format.

    The CDD (Computational Drug Discovery) dataset contains winner/loser
    molecule pairs with labeled sabotage types. This adapter converts
    them to JSONL tasks compatible with NeMo-Gym's rollout collection.

    Example usage:
        adapter = CDDDatasetAdapter(
            dataset_path="/path/to/dataset_cdd_v1",
            system_prompt=custom_prompt,  # Optional
        )

        # Write training tasks to JSONL
        adapter.write_training_jsonl("training_tasks.jsonl")

        # Or iterate over tasks
        for task in adapter.iter_tasks():
            print(task.task_id, task.ground_truth["sabotage_type"])
    """

    def __init__(
        self,
        dataset_path: str | Path,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        advanced_mode: bool = False,
        include_receptor: bool = True,
        relative_paths: bool = False,
    ):
        """Initialize the dataset adapter.

        Args:
            dataset_path: Path to the CDD dataset directory.
            system_prompt: System prompt for training tasks.
            advanced_mode: Whether to use advanced mode prompts (hypothesis + fix).
            include_receptor: Whether to include receptor in preload refs.
            relative_paths: If True, preload_refs use paths relative to dataset_path.
                This is needed for cluster deployment where the dataset is mounted
                at a different path than where it was generated.
        """
        self.dataset_path = Path(dataset_path)
        self.system_prompt = system_prompt
        self.advanced_mode = advanced_mode
        self.include_receptor = include_receptor
        self.relative_paths = relative_paths

        if not self.dataset_path.exists():
            raise ValueError(f"Dataset path does not exist: {self.dataset_path}")

        self._summary = self._load_summary()

    def _load_summary(self) -> dict:
        """Load dataset summary if available."""
        summary_path = self.dataset_path / "dataset_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                return json.load(f)
        return {}

    @staticmethod
    def _stratification_key(pair: MoleculePair) -> str:
        """Derive a fine-grained key for stratified splitting.

        Combines sabotage_type with pattern_name so that train/val splits
        contain proportional representation of every sub-strategy.
        """
        return f"{pair.sabotage_type}/{pair.pattern_name}"

    def iter_complexes(self) -> Iterator[Path]:
        """Iterate over complex directories in the dataset."""
        for path in sorted(self.dataset_path.iterdir()):
            if path.is_dir() and path.name.startswith("complex_"):
                yield path

    def load_complex_metadata(self, complex_dir: Path) -> dict:
        """Load metadata.json for a complex."""
        metadata_path = complex_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"No metadata.json in {complex_dir}")

        with open(metadata_path) as f:
            return json.load(f)

    def iter_pairs(self) -> Iterator[MoleculePair]:
        """Iterate over all winner/loser pairs in the dataset."""
        for complex_dir in self.iter_complexes():
            try:
                metadata = self.load_complex_metadata(complex_dir)
            except ValueError as e:
                logger.warning(f"Skipping {complex_dir}: {e}")
                continue

            # Check for required files
            winner_sdf = complex_dir / "winner.sdf"
            receptor_pdbqt = complex_dir / "receptor.pdbqt"
            receptor_pdb = complex_dir / "receptor.pdb"

            if not winner_sdf.exists():
                logger.warning(f"No winner.sdf in {complex_dir}")
                continue

            if self.include_receptor and not receptor_pdbqt.exists():
                logger.warning(f"No receptor.pdbqt in {complex_dir}")
                continue

            # Extract winner info
            winner_metrics = metadata.get("winner_metrics", {})

            # Iterate over pairs
            for pair_info in metadata.get("pairs", []):
                loser_filename = pair_info.get("loser_filename", "")
                loser_sdf = complex_dir / loser_filename

                if not loser_sdf.exists():
                    logger.warning(f"Loser SDF not found: {loser_sdf}")
                    continue

                # Extract loser metrics
                loser_metrics = pair_info.get("metrics", {})

                yield MoleculePair(
                    complex_id=complex_dir.name,
                    pdb_id=metadata.get("pdb_id", ""),
                    winner_sdf_path=winner_sdf,
                    winner_smiles=winner_metrics.get("smiles", ""),
                    winner_vina=winner_metrics.get("vina_score", 0.0),
                    winner_metrics=winner_metrics,
                    loser_sdf_path=loser_sdf,
                    loser_smiles=loser_metrics.get("smiles"),
                    loser_vina=loser_metrics.get("vina_score", 0.0),
                    loser_metrics=loser_metrics,
                    sabotage_type=pair_info.get("sabotage_type", "unknown"),
                    pattern_name=pair_info.get(
                        "pattern_name",
                        pair_info.get("sabotage_type", "unknown"),
                    ),
                    modification_desc=pair_info.get("modification_desc", ""),
                    focus_atom_idx=pair_info.get("focus_atom_idx"),
                    smarts_pattern=pair_info.get("smarts_pattern"),
                    tanimoto_similarity=pair_info.get("tanimoto_similarity", 0.0),
                    delta_vina=loser_metrics.get("delta_score", 0.0),
                    receptor_pdbqt_path=receptor_pdbqt,
                    receptor_pdb_path=receptor_pdb if receptor_pdb.exists() else None,
                    box_center=metadata.get("box_center"),
                    box_size=metadata.get("box_size"),
                )

    def pair_to_task(self, pair: MoleculePair, task_idx: int) -> TrainingTask:
        """Convert a molecule pair to a training task.

        Args:
            pair: The molecule pair to convert.
            task_idx: Index for generating unique task ID.

        Returns:
            TrainingTask ready for NeMo-Gym.
        """
        # Generate reference IDs
        winner_ref = generate_reference_id("sdf")
        loser_ref = generate_reference_id("sdf")
        receptor_ref = generate_reference_id("pdbqt")
        receptor_pdb_ref = generate_reference_id("pdb")

        # Choose prompt template based on mode
        template = (
            ADVANCED_TASK_PROMPT_TEMPLATE
            if self.advanced_mode
            else TASK_PROMPT_TEMPLATE
        )

        # Extract binding site coordinates (default to origin if not available)
        if pair.box_center and len(pair.box_center) == 3:
            center_x, center_y, center_z = pair.box_center
        else:
            logger.warning(
                f"No box_center in metadata for {pair.complex_id}, "
                "using (0, 0, 0) -- docking will search wrong region"
            )
            center_x, center_y, center_z = 0.0, 0.0, 0.0

        # Build user prompt
        user_content = template.format(
            winner_ref=winner_ref,
            loser_ref=loser_ref,
            receptor_ref=receptor_ref,
            receptor_pdb_ref=receptor_pdb_ref,
            center_x=round(center_x, 2),
            center_y=round(center_y, 2),
            center_z=round(center_z, 2),
        )

        # Build responses_create_params (NeMo-Gym Responses API format)
        # The 'tools' field is critical: without it, vLLM won't enable
        # function calling and the model will only generate plain text.
        responses_create_params = {
            "input": [
                {"role": "developer", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            "tools": get_training_tool_schemas(),
            "max_output_tokens": 4096,
            "temperature": 0.7,
        }

        # Build ground truth for reward computation
        ground_truth = {
            "sabotage_type": pair.sabotage_type,
            "pattern_name": pair.pattern_name,
            "winner_vina": pair.winner_vina,
            "loser_vina": pair.loser_vina,
            "delta_vina": pair.delta_vina,
            "modification_desc": pair.modification_desc,
            "tanimoto_similarity": pair.tanimoto_similarity,
            "focus_atom_idx": pair.focus_atom_idx,
            "smarts_pattern": pair.smarts_pattern,
            "pdb_id": pair.pdb_id,
            "winner_metrics": pair.winner_metrics,
            "loser_metrics": pair.loser_metrics,
        }

        # Build preload refs
        # When relative_paths=True, paths are relative to dataset root so the
        # resource server can resolve them against its own dataset_base_path.
        if self.relative_paths:
            try:
                winner_path = str(pair.winner_sdf_path.relative_to(self.dataset_path))
            except ValueError:
                winner_path = str(pair.winner_sdf_path)
            try:
                loser_path = str(pair.loser_sdf_path.relative_to(self.dataset_path))
            except ValueError:
                loser_path = str(pair.loser_sdf_path)
        else:
            winner_path = str(pair.winner_sdf_path)
            loser_path = str(pair.loser_sdf_path)

        preload_refs = {
            winner_ref: winner_path,
            loser_ref: loser_path,
        }

        if self.include_receptor:
            if self.relative_paths:
                try:
                    receptor_path = str(
                        pair.receptor_pdbqt_path.relative_to(self.dataset_path)
                    )
                except ValueError:
                    receptor_path = str(pair.receptor_pdbqt_path)
            else:
                receptor_path = str(pair.receptor_pdbqt_path)
            preload_refs[receptor_ref] = receptor_path

            # Also preload the receptor PDB for structure_query
            if pair.receptor_pdb_path and pair.receptor_pdb_path.exists():
                if self.relative_paths:
                    try:
                        pdb_path = str(
                            pair.receptor_pdb_path.relative_to(self.dataset_path)
                        )
                    except ValueError:
                        pdb_path = str(pair.receptor_pdb_path)
                else:
                    pdb_path = str(pair.receptor_pdb_path)
                preload_refs[receptor_pdb_ref] = pdb_path

        return TrainingTask(
            task_id=f"task_{task_idx:06d}_{pair.complex_id}_{pair.sabotage_type}_{pair.pattern_name}",
            responses_create_params=responses_create_params,
            ground_truth=ground_truth,
            preload_refs=preload_refs,
        )

    def iter_tasks(self) -> Iterator[TrainingTask]:
        """Iterate over all training tasks."""
        for idx, pair in enumerate(self.iter_pairs()):
            yield self.pair_to_task(pair, idx)

    def collect_tasks(
        self,
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> list[dict]:
        """Collect all tasks as serializable dicts.

        Args:
            limit: Optional cap on number of tasks.
            shuffle: Whether to shuffle the collected tasks.
            seed: Random seed for reproducible shuffling.

        Returns:
            List of task dicts ready for JSONL serialization.
        """
        import random

        tasks = []
        for task in self.iter_tasks():
            if limit is not None and len(tasks) >= limit:
                break
            tasks.append(task.to_jsonl_dict())

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(tasks)

        return tasks

    def write_training_jsonl(
        self,
        output_path: str | Path,
        limit: Optional[int] = None,
    ) -> int:
        """Write training tasks to a single JSONL file.

        Args:
            output_path: Path to output JSONL file.
            limit: Optional limit on number of tasks.

        Returns:
            Number of tasks written.
        """
        tasks = self.collect_tasks(limit=limit)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for t in tasks:
                f.write(json.dumps(t) + "\n")

        logger.info(f"Wrote {len(tasks)} training tasks to {output_path}")
        return len(tasks)

    def write_training_jsonl_split(
        self,
        output_path: str | Path,
        split_ratio: float = 0.85,
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Write training tasks split into train and validation JSONL files.

        Uses **stratified splitting** so that every sabotage type / pattern_name
        combination is proportionally represented in both train and val sets.

        Output files are placed next to ``output_path`` using its stem:
        ``<stem>_train.jsonl`` and ``<stem>_val.jsonl``.

        Args:
            output_path: Base output path (used to derive train/val filenames).
            split_ratio: Fraction of tasks for the training split.
            limit: Optional cap on total tasks before splitting.
            shuffle: Shuffle tasks before splitting.
            seed: Random seed for reproducible shuffling.

        Returns:
            Dict with ``train``, ``val``, and ``total`` counts, paths, and
            per-stratification-group distribution for each split.
        """
        import random
        from collections import defaultdict

        rng = random.Random(seed)

        # Collect (task_dict, stratification_key) tuples directly from pairs
        # so we retain the stratification key for grouping.
        tagged_tasks: list[tuple[dict, str]] = []
        for idx, pair in enumerate(self.iter_pairs()):
            if limit is not None and len(tagged_tasks) >= limit:
                break
            task = self.pair_to_task(pair, idx)
            key = self._stratification_key(pair)
            tagged_tasks.append((task.to_jsonl_dict(), key))

        # Group by stratification key
        groups: dict[str, list[dict]] = defaultdict(list)
        for task_dict, key in tagged_tasks:
            groups[key].append(task_dict)

        # Stratified split: proportional allocation within each group
        train_tasks: list[dict] = []
        val_tasks: list[dict] = []
        train_dist: dict[str, int] = {}
        val_dist: dict[str, int] = {}

        for key in sorted(groups):
            group = groups[key]
            if shuffle:
                rng.shuffle(group)

            if len(group) == 1:
                train_tasks.append(group[0])
                train_dist[key] = 1
                val_dist[key] = 0
                logger.warning(
                    f"Stratification group '{key}' has only 1 example -- "
                    "assigned to train only, no val coverage"
                )
            else:
                n_train = max(1, round(len(group) * split_ratio))
                n_train = min(n_train, len(group) - 1)
                train_tasks.extend(group[:n_train])
                val_tasks.extend(group[n_train:])
                train_dist[key] = n_train
                val_dist[key] = len(group) - n_train

        # Final shuffle of the concatenated splits
        if shuffle:
            rng.shuffle(train_tasks)
            rng.shuffle(val_tasks)

        # Write files
        output_path = Path(output_path)
        parent = output_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        stem = output_path.stem.removesuffix("_train").removesuffix("_val")

        train_path = parent / f"{stem}_train.jsonl"
        val_path = parent / f"{stem}_val.jsonl"

        for path, subset in [(train_path, train_tasks), (val_path, val_tasks)]:
            with open(path, "w") as f:
                for t in subset:
                    f.write(json.dumps(t) + "\n")

        total = len(train_tasks) + len(val_tasks)
        logger.info(
            f"Wrote {len(train_tasks)} train + {len(val_tasks)} val tasks "
            f"({total} total, stratified by {len(groups)} groups)"
        )

        return {
            "train": len(train_tasks),
            "val": len(val_tasks),
            "total": total,
            "train_path": str(train_path),
            "val_path": str(val_path),
            "train_distribution": train_dist,
            "val_distribution": val_dist,
        }

    def get_statistics(self) -> dict:
        """Get statistics about the dataset.

        Returns:
            Dictionary with dataset statistics including per-pattern counts.
        """
        stats: dict[str, Any] = {
            "total_pairs": 0,
            "sabotage_distribution": {},
            "pattern_distribution": {},
            "complexes": 0,
        }

        for pair in self.iter_pairs():
            stats["total_pairs"] += 1
            stype = pair.sabotage_type
            stats["sabotage_distribution"][stype] = (
                stats["sabotage_distribution"].get(stype, 0) + 1
            )
            pkey = self._stratification_key(pair)
            stats["pattern_distribution"][pkey] = (
                stats["pattern_distribution"].get(pkey, 0) + 1
            )

        stats["complexes"] = len(list(self.iter_complexes()))

        return stats


def convert_dataset(
    dataset_path: str,
    output_path: str,
    system_prompt: Optional[str] = None,
    advanced_mode: bool = False,
    limit: Optional[int] = None,
    relative_paths: bool = False,
) -> dict:
    """Convenience function to convert a CDD dataset to NeMo-Gym format.

    Args:
        dataset_path: Path to CDD dataset directory.
        output_path: Path for output JSONL file.
        system_prompt: Optional custom system prompt.
        advanced_mode: Whether to use advanced mode prompts.
        limit: Optional limit on number of tasks.
        relative_paths: If True, use paths relative to dataset_path.

    Returns:
        Statistics dictionary.
    """
    adapter = CDDDatasetAdapter(
        dataset_path=dataset_path,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        advanced_mode=advanced_mode,
        relative_paths=relative_paths,
    )

    num_tasks = adapter.write_training_jsonl(output_path, limit=limit)
    stats = adapter.get_statistics()
    stats["tasks_written"] = num_tasks

    return stats


if __name__ == "__main__":
    import argparse
    import sys

    _backend_dir = str(Path(__file__).resolve().parents[2])
    if _backend_dir not in sys.path:
        sys.path.insert(0, _backend_dir)

    parser = argparse.ArgumentParser(
        description="Convert CDD dataset to NeMo-Gym training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Quick stats check (no files written)
  python %(prog)s -d ../aevom-data-gen/data/output/pdb_v10/ --stats

  # Generate a single JSONL file
  python %(prog)s -d /path/to/dataset -o output.jsonl --relative-paths

  # Generate with train/val split (85/15), shuffled
  python %(prog)s -d /path/to/dataset -o training/data/aevom.jsonl \\
      --relative-paths --split 0.85 --shuffle --seed 42 -v
""",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help="Path to CDD dataset directory",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output JSONL file path (required unless --stats is used)",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print dataset statistics and exit (no output written)",
    )
    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Use advanced mode prompts (hypothesis + fix)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tasks",
    )
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="Use paths relative to dataset root (for cluster deployment)",
    )
    parser.add_argument(
        "--split",
        type=float,
        default=None,
        metavar="RATIO",
        help="Split output into train/val files (e.g. --split 0.85 for 85/15)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle tasks before writing (and before splitting)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible shuffling",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    adapter = CDDDatasetAdapter(
        dataset_path=args.dataset,
        advanced_mode=args.advanced,
        relative_paths=args.relative_paths,
    )

    # --stats mode: print statistics and exit
    if args.stats:
        stats = adapter.get_statistics()
        print(f"\nDataset: {args.dataset}")
        print(f"  Complexes:    {stats['complexes']}")
        print(f"  Total pairs:  {stats['total_pairs']}")
        print("\n  Sabotage distribution:")
        for stype, count in sorted(stats["sabotage_distribution"].items()):
            print(f"    {stype:30s} {count:4d}")
        print("\n  Pattern distribution (stratification groups):")
        for pkey, count in sorted(stats["pattern_distribution"].items()):
            print(f"    {pkey:45s} {count:4d}")
        raise SystemExit(0)

    if args.output is None:
        parser.error("--output / -o is required when not using --stats")

    # --split mode: write train + val files
    if args.split is not None:
        if not 0.0 < args.split < 1.0:
            parser.error("--split must be between 0 and 1 (exclusive)")

        result = adapter.write_training_jsonl_split(
            output_path=args.output,
            split_ratio=args.split,
            limit=args.limit,
            shuffle=args.shuffle,
            seed=args.seed,
        )

        stats = adapter.get_statistics()
        print("\nDataset conversion complete (stratified split):")
        print(f"  Complexes:       {stats['complexes']}")
        print(f"  Total pairs:     {stats['total_pairs']}")
        print(f"  Train tasks:     {result['train']}  -> {result['train_path']}")
        print(f"  Val tasks:       {result['val']}  -> {result['val_path']}")
        print("\n  Sabotage distribution:")
        for stype, count in sorted(stats["sabotage_distribution"].items()):
            print(f"    {stype:30s} {count:4d}")
        train_dist = result.get("train_distribution", {})
        val_dist = result.get("val_distribution", {})
        all_keys = sorted(set(train_dist) | set(val_dist))
        print("\n  Stratified pattern distribution (train / val):")
        for pkey in all_keys:
            t = train_dist.get(pkey, 0)
            v = val_dist.get(pkey, 0)
            print(f"    {pkey:45s} {t:5d} / {v:4d}")
    else:
        # Single file mode (original behavior, with optional shuffle)
        if args.shuffle:
            tasks = adapter.collect_tasks(
                limit=args.limit, shuffle=True, seed=args.seed
            )
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for t in tasks:
                    f.write(json.dumps(t) + "\n")
            num_written = len(tasks)
        else:
            num_written = adapter.write_training_jsonl(
                args.output, limit=args.limit
            )

        stats = adapter.get_statistics()
        print("\nDataset conversion complete:")
        print(f"  Complexes:       {stats['complexes']}")
        print(f"  Total pairs:     {stats['total_pairs']}")
        print(f"  Tasks written:   {num_written}  -> {args.output}")
        print("\n  Sabotage distribution:")
        for stype, count in sorted(stats["sabotage_distribution"].items()):
            print(f"    {stype:30s} {count:4d}")
        print("\n  Pattern distribution:")
        for pkey, count in sorted(stats["pattern_distribution"].items()):
            print(f"    {pkey:45s} {count:4d}")
