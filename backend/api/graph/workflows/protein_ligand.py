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

"""Protein-Ligand Interaction Analysis workflow subgraph.

Workflow 1 (Primary) from prompts.md:
  Phase1: Identify ligand
  Phase2: Obtain 3D structure (crystal / AI / docking)
  Phase3: Analyze interactions (structure_query + python_execute)

Trigger: "How does X interact with Y?", "binding interactions", "analyze binding"
"""

import logging
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)

WORKFLOW_DESCRIPTION = (
    "Run a comprehensive protein-ligand interaction analysis. Identifies the ligand "
    "via PubChem, obtains the 3D structure (from crystal structure, AI prediction, "
    "or docking), visualizes the complex, selects binding site residues, computes "
    "interaction distances and types (H-bonds, salt bridges, pi-stacking, hydrophobic), "
    "and provides a detailed interaction report. Use this when the user asks about "
    "binding interactions, contact residues, or how a molecule interacts with a protein."
)


class ProteinLigandState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    viewer_state: dict
    storage_state: dict
    selections: list[dict]
    findings: list[dict]
    tool_budgets: list[dict]
    candidates: list[dict]
    exclusions: list[dict]
    request_id: str
    session_id: str


PROTEIN_LIGAND_SYSTEM = """You are executing the Protein-Ligand Interaction Analysis workflow.
This is the most comprehensive workflow. Complete ALL three phases.

## PHASE 1: Identify the Ligand
- search_pubchem(name="molecule_name") to get CID and SMILES
- lookup_ccd_code(molecule_name="molecule_name") to get CCD code for OpenFold3

## PHASE 2: Obtain 3D Structure (choose the best option)

Option A - Experimental Structure (PREFERRED when available):
1. search_rcsb with a gene query and NO experimental-method filter, sorted by
   rcsb_entry_info.resolution_combined ascending -> pick a hit with is_holo: true.
   Cryo-EM structures are often the only ones available, and the method breakdown
   in the result shows what any filter you add would exclude.
2. fetch_and_clean_pdb(pdb_id, keep_ligands=true) -> get structure WITH ligand
3. visualize_pdb(pdb_ref)

Option B - AI Prediction with OpenFold3:
1. fetch_protein_sequence(gene_name=...) -> get sequence_ref
2. msa_search(sequence_ref=...) -> get msa_ref (improves prediction quality)
3. lookup_ccd_code(molecule_name="ligand")
4. openfold3(molecules=[{type: "protein", sequence_ref}, {type: "ligand", ccd_codes}], msa_ref=...)
5. visualize_pdb(pdb_ref)

Option C - Physics-based Docking (for novel ligands, receptor structure available):
1. search_rcsb (no method filter) -> fetch_and_clean_pdb(keep_ligands=false)
2. search_pubchem(name="...") -> download_pubchem_sdf(cid=..., name="...")
3. vina_dock(receptor_ref, ligand_ref, center_x, center_y, center_z)
4. visualize_pdb(docked_complex_ref)

Option D - AI Prediction with Boltz-2 (novel ligand and no receptor structure, or
affinity wanted alongside the complex):
1. fetch_protein_sequence(gene_name=...) -> get sequence_ref
2. msa_search(sequence_ref=...) -> get msa_ref
3. boltz2(polymers=[{molecule_type: "protein", sequence_ref}], ligands=[{smiles: "...", predict_affinity: true}], msa_ref=...)
   - Takes SMILES, so it does not need a CCD code. One ligand per call may set predict_affinity.
4. visualize_pdb(mmcif_ref) — the viewer reads mmCIF directly
   - The returned pIC50 is a model estimate for ranking, not a measured affinity.
   - To dock the predicted complex afterwards, run convert_structure(structure_ref=..., target_format="pdb") first.

## PHASE 3: Analyze Interactions

### Preferred: plip_interactions
```
plip_interactions(structure_refs=["<complex_ref>"])
```
This is the right tool for the question. It applies PLIP's published geometric
criteria -- donor angles, ring-plane offsets, protonation-aware charges -- and
names every hydrogen bond, hydrophobic contact, pi-stacking, pi-cation, salt
bridge, halogen bond, water bridge and metal coordination with its residue and
geometry. It accepts both PDB and mmCIF refs, so a Boltz-2 complex needs no
conversion.

If the structure came from boltz2 with several diffusion samples, pass EVERY
sample reference in the one call. Interactions conserved across poses are the
binding mode; ones in a single pose are sampling noise, and one pose cannot tell
you which is which.

Report the conserved interactions with their residues, and say how many poses
supported each one.

### Fallback: manual geometry in python_execute
Use this only when plip_interactions is unavailable, or when you need a
measurement it does not report (a specific atom-to-atom distance, a custom
cutoff, a buried-surface calculation). Distance alone does not establish a
hydrogen bond -- if you go this route, say that the contacts are distance-based
rather than calling them interactions.

#### Step 1: Select binding pocket residues
Call structure_query with the ref of the loaded complex (the reference returned by openfold3, boltz2 or vina_dock) and a MolQL expression that selects all atoms within 5A of the ligand:
```
structure_query(pdb_ref="<docked_complex_ref>", molql="(sel.atom.include-surroundings (sel.atom.res (= atom.entity-type non-polymer)) :radius 5 :as-whole-residues true)")
```

After structure_query completes, check [SELECTION STATE] for the selection reference.

#### Step 2: Compute interactions with python_execute
**CRITICAL**: Pass `selection_ref` from [SELECTION STATE] to python_execute. Example:
```
python_execute(selection_ref="sel_abc12345", code="...")
```

The `selection` variable has this structure — note that `coords` is a **list** `[x, y, z]` (not hashable):
```python
selection = {
    "atoms": [
        {
            "name": "CA",            # atom name (str)
            "element": "C",          # element symbol (str)
            "id": 123,               # atom ID (int)
            "coords": [x, y, z],     # 3D coordinates — THIS IS A LIST
            "residue": {
                "name": "ALA",       # residue name (str)
                "seqNumber": 45,     # sequence number (int)
                "chainId": "A",      # chain ID (str)
                "entityType": "polymer"  # "polymer" or "non-polymer"
            }
        }, ...
    ]
}
```

**Use this code template for interaction analysis:**
```python
import numpy as np

ligand_atoms = [a for a in selection['atoms'] if a['residue']['entityType'] == 'non-polymer']
protein_atoms = [a for a in selection['atoms'] if a['residue']['entityType'] == 'polymer']

# Group protein atoms by residue — use string key, NOT tuple
residues = {}
for atom in protein_atoms:
    res_key = f"{atom['residue']['name']}{atom['residue']['seqNumber']}"
    if res_key not in residues:
        residues[res_key] = {'atoms': [], 'coords': []}
    residues[res_key]['atoms'].append(atom)
    residues[res_key]['coords'].append(atom['coords'])

# Find closest contacts per residue
lig_coords = np.array([a['coords'] for a in ligand_atoms])
for res_name, res_data in residues.items():
    res_coords = np.array(res_data['coords'])
    # Pairwise distance matrix: (n_res_atoms, n_lig_atoms)
    diffs = res_coords[:, None, :] - lig_coords[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    min_dist = float(dists.min())
    if min_dist < 4.0:
        print(f"{res_name}: {min_dist:.2f} A - potential interaction")
```

#### Step 3: Classify contacts (approximate -- these are distance cutoffs only)
- **H-bonds:** Donor-acceptor pairs (N, O atoms) < 3.5 A
- **Salt bridges:** ASP/GLU (negative) with LYS/ARG (positive) < 4 A
- **Pi-stacking:** PHE/TYR/TRP aromatic rings < 5 A
- **Hydrophobic:** LEU/ILE/VAL/ALA/MET contacts < 4 A

#### Step 4: Summarize with specific residues and distances

IMPORTANT: Do NOT skip Phase 3. The user needs interaction analysis, not just visualization.
Always prefer Option A (crystal structure) when available.
"""


def build_protein_ligand_graph(
    llm: Any,
    tools: list[Any],
) -> Any:
    """Build the protein-ligand interaction analysis subgraph."""
    from api.graph.state_modifier import inject_state_into_prompt

    relevant_names = {
        "search_rcsb",
        "fetch_and_clean_pdb",
        "fetch_protein_sequence",
        "msa_search",
        "openfold3",
        "boltz2",
        "convert_structure",
        "vina_dock",
        "visualize_pdb",
        "search_pubchem",
        "download_pubchem_sdf",
        "lookup_ccd_code",
        "visualize_sdf",
        "python_execute",
        "structure_query",
        "get_viewer_state",
        "plip_interactions",
    }
    workflow_tools = [t for t in tools if t.name in relevant_names]
    if not workflow_tools:
        workflow_tools = tools

    llm_with_tools = llm.bind_tools(workflow_tools)

    async def agent_node(state: ProteinLigandState) -> dict:
        viewer = state.get("viewer_state") or {}
        storage = state.get("storage_state") or {}
        selections = state.get("selections") or []
        findings = state.get("findings") or []

        augmented = inject_state_into_prompt(
            PROTEIN_LIGAND_SYSTEM,
            viewer,
            storage,
            selections,
            findings=findings,
            tool_budgets=state.get("tool_budgets") or [],
            candidates=state.get("candidates") or [],
            exclusions=state.get("exclusions") or [],
        )

        messages = list(state["messages"])
        result_messages = []
        system_replaced = False
        for msg in messages:
            if isinstance(msg, SystemMessage) and not system_replaced:
                result_messages.append(SystemMessage(content=augmented))
                system_replaced = True
            else:
                result_messages.append(msg)
        if not system_replaced:
            result_messages.insert(0, SystemMessage(content=augmented))

        response = await llm_with_tools.ainvoke(result_messages)
        return {"messages": [response]}

    def should_continue(state: ProteinLigandState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        if not messages:
            return "__end__"
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    async def refresh_node(
        state: ProteinLigandState, config: Optional[RunnableConfig] = None
    ) -> dict:
        from api.graph.state import refresh_state_from_backends

        return await refresh_state_from_backends(state, config)

    async def await_pending_ops_node(state: ProteinLigandState) -> dict:
        from api.graph.tools import drain_pending_ops, get_and_clear_pending_structure_queries
        from api.selections.format import (
            format_selection_query_error,
            is_molql_error_summary,
        )
        from api.selections.registry import get_selection_by_query_id
        from api.redis import get_redis
        from langchain_core.messages import ToolMessage

        unacked = await drain_pending_ops(timeout=10.0)
        if unacked:
            logger.warning(f"await_pending_ops: unacknowledged ops: {unacked}")

        pending_sqs = get_and_clear_pending_structure_queries()
        if not pending_sqs:
            return {}

        redis_client = get_redis()
        replacements: list[ToolMessage] = []

        for query_id, session_id, deferred_span in pending_sqs:
            sel_info = await get_selection_by_query_id(redis_client, session_id, query_id)

            if sel_info and sel_info.atom_count > 0:
                new_content = (
                    f"Selection active: {sel_info.atom_count} atoms in "
                    f"{sel_info.residue_count} residues. "
                    f"Reference: {sel_info.reference}. "
                    f'Pass selection_ref="{sel_info.reference}" to python_execute '
                    f"for geometric/chemical analysis."
                )
            elif sel_info and is_molql_error_summary(sel_info.summary):
                new_content = format_selection_query_error(sel_info.summary)
            elif sel_info and sel_info.atom_count == 0:
                new_content = (
                    "Error: Selection query matched 0 atoms. The MolQL expression "
                    "did not select any atoms in the loaded structure."
                )
            elif query_id in unacked:
                new_content = (
                    "Error: Selection query timed out waiting for the viewer."
                )
            else:
                new_content = "Selection query acknowledged but no data returned."

            if deferred_span:
                try:
                    deferred_span.set_attribute("tool.raw.output", new_content)
                    deferred_span.end()
                except Exception:
                    logger.debug("Failed to finalize deferred span", exc_info=True)

            target_msg = None
            for msg in reversed(state["messages"]):
                if (
                    isinstance(msg, ToolMessage)
                    and f"[PENDING:{query_id}]" in (msg.content or "")
                ):
                    target_msg = msg
                    break

            if target_msg:
                replacements.append(ToolMessage(
                    content=new_content,
                    tool_call_id=target_msg.tool_call_id,
                    id=target_msg.id,
                ))

        if replacements:
            return {"messages": replacements}
        return {}

    tool_node = ToolNode(workflow_tools)

    graph = StateGraph(ProteinLigandState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("await_pending_ops", await_pending_ops_node)
    graph.add_node("refresh_state", refresh_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tools", "await_pending_ops")
    graph.add_edge("await_pending_ops", "refresh_state")
    graph.add_edge("refresh_state", "agent")

    return graph.compile()
