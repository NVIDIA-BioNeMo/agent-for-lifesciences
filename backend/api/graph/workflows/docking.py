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

"""Docking & Binding Affinity workflow subgraph.

Workflow 2 from prompts.md:
  Get Receptor -> Get Ligand -> Run Vina -> Interpret

Trigger: "Dock", "binding affinity", "Vina", "how strongly does X bind?"
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
    "Run a physics-based molecular docking workflow using AutoDock Vina. "
    "Obtains the receptor protein structure (from RCSB or OpenFold3), prepares "
    "the ligand (from PubChem or SMILES), runs Vina docking to compute binding "
    "affinity, visualizes the docked complex, and interprets the results. "
    "Use this when the user asks about docking, binding affinity, or Vina calculations."
)


class DockingState(TypedDict):
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


DOCKING_SYSTEM = """You are executing the Docking & Binding Affinity workflow.
Your task is to determine how tightly a ligand binds a protein, and interpret the
result.

Two routes give an affinity, and they answer different questions:
- vina_dock: a physics-based binding energy in kcal/mol against a real receptor
  structure. Prefer this whenever a holo structure exists or can be fetched.
- boltz2 with predict_affinity: a co-folded complex and an estimated pIC50 from
  sequence plus SMILES. Use it when there is no structure to dock into, or when
  triaging novel molecules that have no crystal structure. It is a ranking
  signal, not a measured IC50 — say so when reporting it.

Vina route:
1. Get Receptor:
   - If gene name given: search_rcsb with a gene query and NO experimental-method
     filter (cryo-EM is the only method for many targets) -> select a hit with
     is_holo: true -> fetch_and_clean_pdb(pdb_id, keep_ligands=false)
   - If prediction needed: fetch_protein_sequence -> msa_search -> openfold3 (pass msa_ref for better quality)
2. Get Ligand:
   - If common name: search_pubchem(name="...") -> download_pubchem_sdf(cid=..., name="...")
   - If SMILES: python_execute with aevom.mol_from_smiles(smiles), aevom.embed_mol_3d(mol)
3. Run Vina: vina_dock(receptor_pdb_ref, ligand_sdf_ref, center_x, center_y, center_z)
   - Use binding site coordinates from the holo structure
4. Visualize: visualize_pdb(docked_complex_ref)
5. Interpret affinity: > -6 kcal/mol (weak), -6 to -9 (moderate), < -9 (strong)
6. If the user asks WHY it binds, or which residues matter, call
   plip_interactions(structure_refs=["<docked_complex_ref>"]). The score says how
   tightly; that says how

Boltz-2 route:
1. fetch_protein_sequence(gene_name=...) -> optionally msa_search(sequence_ref=...)
2. boltz2(polymers=[{molecule_type: "protein", sequence_ref: "..."}], ligands=[{smiles: "...", predict_affinity: true}], msa_ref="...")
   - Exactly one ligand may set predict_affinity. To rank several, make one call each.
   - diffusion_samples defaults to 1, which is one draw from a stochastic
     process. If you are going to reason about the pose rather than just look at
     it, pass diffusion_samples=5 in this call. Deciding afterwards means paying
     for the whole prediction twice.
3. Visualize: visualize_pdb(mmcif_ref) — it reads mmCIF directly
4. Interpret pIC50: higher binds tighter. To then dock the predicted complex, run
   convert_structure(structure_ref=..., target_format="pdb") first.
5. To explain the binding mode, call plip_interactions with EVERY diffusion
   sample reference in one call -- it reads mmCIF directly, needs no conversion,
   and the interactions the samples agree on are the ones worth reporting. A
   contact present in one sample of one is not evidence of anything.

IMPORTANT: For Vina you MUST select a HOLO structure (it has binding site coordinates).
Always complete through visualization and interpretation.
"""


def build_docking_graph(
    llm: Any,
    tools: list[Any],
) -> Any:
    """Build the docking workflow subgraph."""
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
        "plip_interactions",
        "visualize_pdb",
        "search_pubchem",
        "download_pubchem_sdf",
        "python_execute",
    }
    workflow_tools = [t for t in tools if t.name in relevant_names]
    if not workflow_tools:
        workflow_tools = tools

    llm_with_tools = llm.bind_tools(workflow_tools)

    async def agent_node(state: DockingState) -> dict:
        viewer = state.get("viewer_state") or {}
        storage = state.get("storage_state") or {}
        selections = state.get("selections") or []
        findings = state.get("findings") or []

        augmented = inject_state_into_prompt(
            DOCKING_SYSTEM,
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

    def should_continue(state: DockingState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        if not messages:
            return "__end__"
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    async def refresh_node(
        state: DockingState, config: Optional[RunnableConfig] = None
    ) -> dict:
        from api.graph.state import refresh_state_from_backends

        return await refresh_state_from_backends(state, config)

    async def await_pending_ops_node(state: DockingState) -> dict:
        from api.graph.tools import drain_pending_ops

        unacked = await drain_pending_ops(timeout=10.0)
        if unacked:
            logger.warning(f"await_pending_ops: unacknowledged ops: {unacked}")
        return {}

    tool_node = ToolNode(workflow_tools)

    graph = StateGraph(DockingState)
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
