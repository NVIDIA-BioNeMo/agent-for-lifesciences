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

"""MolQL structure query tool for Molstar visualization.

This tool enables the LLM to execute MolQL queries on molecules
displayed in the Molstar 3D viewer using native MolQL S-expression syntax.

MolQL Reference: https://molql.org/
"""

import logging
from typing import Any

from .base import BaseTool, ContentType, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)


class MolstarQueryTool(BaseTool):
    """Execute MolQL structure queries in the Molstar 3D viewer.

    This tool accepts a MolQL S-expression string and automatically:
    1. Creates a persistent selection in the viewer
    2. Focuses the camera on the selection
    3. Returns selection statistics (atom/residue counts) for analysis

    The selection data is stored and can be passed to python_execute for
    geometric and chemical analysis.
    """

    @property
    def name(self) -> str:
        return "structure_query"

    @property
    def summary_template(self) -> str:
        return "Running MolQL query..."

    def format_summary(self, arguments: dict) -> str:
        """Generate summary based on the MolQL query."""
        molql = arguments.get("molql", "")

        # Extract a brief description from the MolQL query
        if "entity-type non-polymer" in molql or "entity-type ligand" in molql:
            return "Selecting ligand molecules..."
        elif "entity-type polymer" in molql:
            return "Selecting protein/polymer chains..."
        elif "atom.chain" in molql:
            return "Selecting chain..."
        elif "atom.resname" in molql or "atom.resno" in molql:
            return "Selecting residues..."
        elif "include-surroundings" in molql:
            return "Selecting surrounding residues..."
        elif "sec-struct" in molql:
            return "Selecting secondary structure..."
        elif "atom.el" in molql:
            return "Selecting atoms..."

        return "Selecting structure..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "structure_query",
                "description": """Execute a MolQL selection query against a specific PDB structure.

The structure to query is identified explicitly by `pdb_ref`. After
`vina_dock`, pass the docked-complex reference returned by the dock to
analyze the bound ligand or its surroundings; to analyze the apo
receptor, pass the original receptor reference.

This tool automatically:
1. Creates a persistent selection in the viewer (visible to the user)
2. Creates a selection reference that can be passed to python_execute for analysis
3. Returns selection statistics (atom count, residue count)

Use this tool to:
- Select, highlight, or focus on specific parts of a structure, helpful for explaining the structure to the user
- Find residues near a ligand or specific atoms, helpful for analyzing interactions
- Prepare selections for analysis in python_execute

MOLQL SYNTAX:
MolQL uses S-expression (Lisp-like) syntax: (function-name arg1 arg2 :keyword value)

SELECTION GENERATORS:
- (sel.atom.all) - All atoms
- (sel.atom.atoms TEST) - Singleton atoms matching test
- (sel.atom.res TEST) - Atoms grouped by residue
- (sel.atom.chains TEST) - Atoms grouped by chain
- (sel.atom.rings) - Ring structures

ATOM PROPERTIES:
- atom.el - Element symbol (use _C, _N, _Fe, etc.)
- atom.chain - Chain ID
- atom.resname - Residue name (ALA, HEM, etc.)
- atom.resno - Residue sequence number
- atom.name - Atom name (CA, CB, N, O)
- atom.entity-type - polymer, non-polymer, water
- atom.bfactor - B-factor/temperature factor

OPERATORS:
- (= prop value) - Equals
- (!= prop value) - Not equals
- (< prop value), (> prop value) - Comparison
- (in-range prop min max) - Range check
- (and expr1 expr2), (or expr1 expr2), (not expr) - Logic
- (set.has (set val1 val2) prop) - Set membership; the way to match a list of
  discrete values such as residue numbers

There is NO `in` function and NO [a, b, c] list syntax. `(in atom.resno [1, 2])`
is rejected as an error. Use set.has for discrete values, in-range only for a
continuous range.

SELECTION MODIFIERS:
- (sel.atom.include-surroundings SEL :radius N :as-whole-residues true) - Within distance
- (sel.atom.include-connected SEL :bond-test BOND :layer-count N) - Connected atoms
- (sel.atom.except-by SEL :by EXCLUDE) - Exclude atoms
- (sel.atom.union SEL) - Merge into single set

COMMON EXAMPLES:

1. Select ligand:
   (sel.atom.res (= atom.entity-type non-polymer))

2. Binding site (residues within 5Å of ligand):
   (sel.atom.include-surroundings (sel.atom.res (= atom.entity-type non-polymer)) :radius 5 :as-whole-residues true)

3. Specific chain:
   (sel.atom.chains (= atom.chain A))

4. Residue range:
   (sel.atom.res (in-range atom.resno 50 100))

5. Backbone atoms:
   (sel.atom.atoms (set.has (set CA C N O) atom.name))

6. Alpha helices:
   (sel.atom.atom-groups :residue-test (atom.sec-struct.is helix) :group-by atom.key.sec-struct)

7. Beta sheets:
   (sel.atom.atom-groups :residue-test (atom.sec-struct.is sheet) :group-by atom.key.sec-struct)

8. Specific residue:
   (sel.atom.res (and (= atom.chain A) (= atom.resname HEM)))

9. All metals:
   (sel.atom.atoms (set.has (set _Fe _Zn _Mg _Ca _Cu) atom.el))

10. High B-factor residues:
    (sel.atom.res (> atom.bfactor 60))

11. Specific residues by number (e.g. a hotspot list) in one chain:
    (sel.atom.res (and (= atom.chain A) (set.has (set 29 30 31 35 43 128) atom.resno)))

SELECTION DATA:
After executing the query, selection data (atoms, coordinates) is stored and
a reference ID is returned. You can pass this reference to python_execute to
analyze the selection:

Example workflow:
1. structure_query(pdb_ref="pdb_abc12345", molql="(sel.atom.res (= atom.entity-type non-polymer))")
   -> Returns: "Selected 47 atoms across 1 residue. Reference: selection_abc123"
2. python_execute(code="...", selection_ref="selection_abc123")
   -> Analyze atoms, compute distances, etc.

For neighbor-shell / binding-site selections and how to list the residues in
python (and the correct fallback when no viewer is available), load the
'Structure Selection & Neighbor Analysis' skill.""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_ref": {
                            "type": "string",
                            "description": (
                                "Reference to the PDB structure to query (e.g., "
                                "'pdb_abc12345'). Use the receptor reference for apo "
                                "analysis, or a docked-complex reference returned by "
                                "vina_dock to analyze a specific bound pose."
                            ),
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "molql": {
                            "type": "string",
                            "description": "MolQL S-expression query string. Must be a valid MolQL expression starting with '('.",
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Optional short name for the resulting selection "
                                "(e.g. 'pocket'), shown next to its reference so "
                                "you and the user can refer to it later."
                            ),
                        },
                    },
                    "required": ["pdb_ref", "molql"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the MolQL structure query.

        This tool validates the MolQL syntax and returns a specification
        that downstream wrappers (production graph tools, training resource
        server) will parse and execute against the requested PDB structure.

        The frontend / training wrapper will:
        1. Resolve ``pdb_ref`` to PDB content
        2. Create a persistent selection
        3. Focus the camera on the selection (production)
        4. Return selection statistics
        """
        pdb_ref = arguments.get("pdb_ref", "")
        molql = arguments.get("molql", "")

        if not pdb_ref or not pdb_ref.strip():
            return ToolResult.error(
                "pdb_ref is required. Pass the reference to the PDB structure "
                "you want to query (e.g., 'pdb_abc12345'). After vina_dock, "
                "use the docked-complex reference; for apo analysis use the "
                "original receptor reference."
            )

        pdb_ref = pdb_ref.strip()

        if not molql or not molql.strip():
            return ToolResult.error("MolQL expression is required")

        molql = molql.strip()

        # Basic syntax validation
        if not molql.startswith("("):
            return ToolResult.error(
                "MolQL expression must start with '('. "
                "Example: (sel.atom.res (= atom.entity-type non-polymer))"
            )

        if molql.count("(") != molql.count(")"):
            return ToolResult.error(
                f"Unbalanced parentheses in MolQL expression. "
                f"Found {molql.count('(')} opening and {molql.count(')')} closing parentheses."
            )

        name = (arguments.get("name") or "").strip()[:80]

        logger.info(
            f"MolQL query against {pdb_ref}: {molql[:100]}"
            f"{'...' if len(molql) > 100 else ''}"
        )

        # Return as a structure query result - downstream wrapper will resolve
        # pdb_ref and execute. Always use action="select" - frontend will also
        # focus and return info. Note: The graph tool wrapper waits for the
        # selection POST and returns outcome-based text (atom/residue counts
        # or error); this message is unused.
        return ToolResult(
            content_type=ContentType.STRUCTURE_QUERY,
            data={
                "pdb_ref": pdb_ref,
                "molql": molql,
                "action": "select",
                **({"name": name} if name else {}),
            },
            metadata={"message": "Selection created in the 3D viewer."},
        )
