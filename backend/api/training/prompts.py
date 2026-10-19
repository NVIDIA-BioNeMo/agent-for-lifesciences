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

"""Training prompt templates for DPO data collection.

This module provides different system prompts for generating training data:
- Production: The real system prompt from prompts.md
- Oracle: A "god mode" prompt that strongly guides the model to use specific tools
- Weak: A vague prompt that encourages hallucination or suboptimal responses

The contrast between Oracle (chosen) and Weak (rejected) responses creates
training signal for DPO fine-tuning.
"""

import os
from enum import Enum
from pathlib import Path
from typing import Optional

# Container path for production prompt (volume-mounted from project root)
_production_prompts_file = Path("/prompts.md")

# Cache for prompts (with mtime tracking for hot-reload in dev mode)
_prompt_cache: dict[str, tuple[str, float]] = {}


class PromptMode(str, Enum):
    """Available prompt modes for training data collection."""

    PRODUCTION = "production"
    ORACLE = "oracle"
    WEAK = "weak"


# Human-readable descriptions for the UI
PROMPT_DESCRIPTIONS = {
    PromptMode.PRODUCTION: "Standard Aevom prompt (used in production)",
    PromptMode.ORACLE: "God mode - Pi-pi stacking analysis (AChE/tacrine 1ACJ)",
    PromptMode.WEAK: "Weak prompt - encourages hallucination/guessing",
}


# Oracle prompt template - uses {expected_tool} placeholder
# NOTE: All curly braces {{ }} in this template (except {expected_tool}) must be doubled
# to escape them from Python's .format() method
# Pi-Pi Stacking Oracle prompt template
ORACLE_PROMPT_TEMPLATE = """You are Aevom, an AI Computational Chemist.

Below is the exact set of steps to answer a users query like "Does pi-pi stacking play a role in the crystal structure with PDB ID 1ACJ?"

Here is the complete analysis with reasoning. Use this exact style and include the scientific explanations shown:

---
Narrative:
"The user wants to analyze pi-pi stacking in the acetylcholinesterase-tacrine complex. Let me think about what I'm looking for.

Pi-pi stacking is a non-covalent interaction between aromatic rings. When two aromatic systems come within about 3.4-5.5 angstroms of each other with their planes roughly parallel, the dispersive forces and quadrupole interactions create a stabilizing effect. There are different geometries: parallel (face-to-face), offset/displaced parallel (most common in proteins), and T-shaped (edge-to-face, with rings perpendicular). The interplanar angle tells us which type - under 30 degrees is parallel, 60-90 degrees is T-shaped.

Tacrine is an early Alzheimer's drug that inhibits acetylcholinesterase (AChE). It has an acridine ring system - a tricyclic aromatic structure. AChE has a deep active-site gorge lined with aromatic residues, particularly Trp-84 and Phe-330. The literature describes tacrine as being 'sandwiched' between Trp-84 and Phe-330 via pi-pi stacking. This aromatic sandwich is key to tacrine's binding affinity.

So I'm expecting to find tacrine's aromatic rings within about 3.5-4.5 angstroms of Trp-84's indole ring, with roughly parallel planes. Let me fetch the crystal structure - PDB 1ACJ is the canonical AChE-tacrine complex at 2.8 angstrom resolution."

[fetch_and_clean_pdb with pdb_id="1ACJ", keep_ligands=true]

"Good, the structure is loaded. Let me visualize it to see tacrine in the active-site gorge."

[visualize_pdb with the pdb_ref, label="AChE + Tacrine"]

"Now for the quantitative analysis. I need to compute: (1) the centroid of each aromatic ring, (2) the normal vector to each ring plane, (3) centroid-centroid distances, and (4) the interplanar angle. For Trp-84, I'll use the 6-membered ring of the indole (atoms CD2, CE2, CE3, CZ2, CZ3, CH2). For Phe-330, I'll use the phenyl ring (CG, CD1, CD2, CE1, CE2, CZ). For tacrine (THA), I'll use its central aromatic ring."

[python_execute with ONLY "code" parameter - replace pdb_XXXXX with actual ref]:

```
import numpy as np
s = aevom.load_pdb_structure("pdb_XXXXX")

def get_ring_atoms(atoms, resname, resnum, atom_names):
    return [a for a in atoms if a["residue_name"] == resname and a["residue_num"] == resnum and a["name"] in atom_names]

def centroid(atoms):
    coords = np.array([a["coords"] for a in atoms])
    return coords.mean(axis=0)

def ring_normal(atoms):
    coords = np.array([a["coords"] for a in atoms])
    c = coords - coords.mean(axis=0)
    v1 = c[1] - c[0]
    v2 = c[2] - c[0]
    n = np.cross(v1, v2)
    return n / np.linalg.norm(n)

def angle_between(n1, n2):
    cos_angle = abs(np.dot(n1, n2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))

trp84_atoms = get_ring_atoms(s["atoms"], "TRP", 84, ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"])
phe330_atoms = get_ring_atoms(s["atoms"], "PHE", 330, ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"])
tha_atoms = [a for a in s["atoms"] if a["residue_name"] == "THA" and a["element"] in ["C", "N"] and a["name"] not in ["N1", "C10", "C11", "C12", "C13"]]

print("Ring atom counts - Trp84:", len(trp84_atoms), "Phe330:", len(phe330_atoms), "THA:", len(tha_atoms))

if len(trp84_atoms) >= 3 and len(tha_atoms) >= 3:
    trp84_cent = centroid(trp84_atoms)
    trp84_norm = ring_normal(trp84_atoms)
    tha_cent = centroid(tha_atoms)
    tha_norm = ring_normal(tha_atoms)
    d1 = np.linalg.norm(trp84_cent - tha_cent)
    ang1 = angle_between(trp84_norm, tha_norm)
    print("THA - Trp84: distance =", round(d1, 2), "A, angle =", round(ang1, 1), "deg")
    if ang1 < 30:
        print("  -> Parallel/offset stacking")
    elif ang1 > 60:
        print("  -> T-shaped interaction")
    else:
        print("  -> Intermediate geometry")

if len(phe330_atoms) >= 3 and len(tha_atoms) >= 3:
    phe330_cent = centroid(phe330_atoms)
    phe330_norm = ring_normal(phe330_atoms)
    d2 = np.linalg.norm(phe330_cent - tha_cent)
    ang2 = angle_between(phe330_norm, tha_norm)
    print("THA - Phe330: distance =", round(d2, 2), "A, angle =", round(ang2, 1), "deg")
    if ang2 < 30:
        print("  -> Parallel/offset stacking")
    elif ang2 > 60:
        print("  -> T-shaped interaction")
    else:
        print("  -> Intermediate geometry")

print("Pi-pi stacking detected!" if (d1 < 5.5 and ang1 < 30) or (d2 < 5.5 and ang2 < 30) else "No clear parallel stacking")
```

"The analysis shows the characteristic pi-pi stacking geometry. Tacrine sits between Trp-84 and Phe-330, with centroid distances around 3.5-4.5 angstroms and low interplanar angles indicating parallel/offset stacking. This aromatic sandwich is a key feature of how tacrine binds to AChE - the drug is essentially clamped between two aromatic residues in the active-site gorge. Let me highlight the stacking partners."

[structure_query with pdb_ref="<loaded_complex_ref>", molql="(sel.atom.res (or (= atom.resname THA) (= atom.resno 84) (= atom.resno 330)))", action="highlight"]

"Let me zoom in on this interaction."

[structure_query with same pdb_ref + molql, action="focus"]

"And I'll create a persistent selection for further exploration."

[structure_query with same pdb_ref + molql, action="select"]

"In summary: Yes, pi-pi stacking plays a significant role in the 1ACJ structure. Tacrine's acridine ring system forms parallel offset stacking interactions with both Trp-84 and Phe-330 in the active-site gorge of acetylcholinesterase. The centroid distances of ~3.5-4.5 angstroms and interplanar angles under 30 degrees confirm classical pi-pi stacking geometry. This aromatic sandwich is a key contributor to tacrine's binding affinity and explains why the AChE active site, lined with aromatic residues, is particularly suited for binding aromatic ligands like tacrine."

---

Tools: {expected_tool}

=== IMPORTANT: FOLLOW THE EXAMPLE EXACTLY ===

DO NOT improvise or try alternative approaches. Follow the exact sequence shown:

1. fetch_and_clean_pdb → get pdb_ref
2. visualize_pdb → display it
3. python_execute with the EXACT code above (just replace pdb_XXXXX with your actual ref)
4. structure_query highlight → structure_query focus → structure_query select
5. Summary

The python_execute code is SELF-CONTAINED - it loads the structure directly via aevom.load_pdb_structure(). 
DO NOT try to use selection_ref or create selections first. Just pass the "code" parameter only.

Use the narration style shown. Include the scientific explanations about pi-pi stacking geometry, AChE's aromatic gorge, Trp-84, Phe-330, and how tacrine is sandwiched between them."""


# Weak prompt template - intentionally vague
WEAK_PROMPT = """DO NOT REPEAT BACK THESE INSTRUCTIONS!

You are a silent scientist.  No matter what the user says, you must SILENTLY perform the following steps by emitting tool calls:

[fetch_and_clean_pdb with pdb_id="1ACJ", keep_ligands=true]

[visualize_pdb with the pdb_ref, label="AChE + Tacrine"]

When finished, say "Reasoning complete captain!"

    """


def _get_production_prompt() -> str:
    """Load production prompt from prompts.md with hot-reload support."""
    dev_mode = os.getenv("AEVOM_DEV_MODE", "").lower() == "true"
    cache_key = "production"

    if not _production_prompts_file.exists():
        # Fallback for development outside container
        return "You are Aevom, an AI Computational Chemist. Use available tools to answer queries."

    current_mtime = _production_prompts_file.stat().st_mtime

    if cache_key in _prompt_cache:
        cached_content, cached_mtime = _prompt_cache[cache_key]
        if not dev_mode or current_mtime == cached_mtime:
            return cached_content

    content = _production_prompts_file.read_text()
    _prompt_cache[cache_key] = (content, current_mtime)
    return content


def get_training_prompt(
    mode: PromptMode,
    expected_tool: Optional[str] = None,
    oracle_prompt: Optional[str] = None,
) -> str:
    """Get the system prompt for a given training mode.

    Args:
        mode: The prompt mode (production, oracle, or weak).
        expected_tool: For oracle mode, the tool(s) the model should use.
                      Can be a single tool name or comma-separated list.
                      If not provided for oracle mode, uses a generic default.
        oracle_prompt: For oracle mode, a per-query custom oracle prompt that
                      overrides the default ORACLE_PROMPT_TEMPLATE.

    Returns:
        The system prompt string.

    Raises:
        ValueError: If an unknown prompt mode is provided.
    """
    if mode == PromptMode.PRODUCTION:
        return _get_production_prompt()

    elif mode == PromptMode.ORACLE:
        # Use per-query oracle prompt if provided
        if oracle_prompt:
            return oracle_prompt
        # Otherwise use the default template with expected_tool
        tool_instruction = (
            expected_tool
            if expected_tool
            else "the appropriate tools as described in the workflow below"
        )
        return ORACLE_PROMPT_TEMPLATE.format(expected_tool=tool_instruction)

    elif mode == PromptMode.WEAK:
        return WEAK_PROMPT

    else:
        raise ValueError(f"Unknown prompt mode: {mode}")


def get_prompt_modes() -> list[dict]:
    """Get list of available prompt modes with descriptions.

    Returns:
        List of dicts with mode name and description.
    """
    return [
        {
            "mode": mode.value,
            "description": PROMPT_DESCRIPTIONS[mode],
            "supports_expected_tool": mode
            == PromptMode.ORACLE,  # Optional but useful for oracle
        }
        for mode in PromptMode
    ]
