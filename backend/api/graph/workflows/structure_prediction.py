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

"""Structure Prediction workflow subgraph.

Workflow 3 from prompts.md:
  Fetch sequence -> (optional) MSA Search -> OpenFold3 -> Visualize

Trigger: "Predict structure", "fold this sequence", "show complex of X and Y"
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
    "Run an AI structure prediction workflow using OpenFold3. Fetches the protein "
    "sequence from UniProt, optionally runs MSA search for better predictions, "
    "optionally looks up ligand CCD codes, runs OpenFold3 to predict the 3D "
    "structure, and visualizes the result. Use this when the user asks to predict, "
    "fold, or model a protein structure or protein-ligand complex."
)


class StructurePredictionState(TypedDict):
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


STRUCTURE_PREDICTION_SYSTEM = """You are executing the Structure Prediction workflow.
Your task is to predict a 3D structure and visualize it.

Choose the predictor first:
- openfold3: structure only, ligands identified by CCD code. The default.
- boltz2: structure plus an optional binding affinity estimate, ligands given as
  SMILES or CCD. Use it when the ligand has no CCD code, or when the request also
  asks how tightly the ligand binds.

Steps:
1. Use fetch_protein_sequence(gene_name=...) to get the protein sequence reference
2. Use msa_search(sequence_ref=...) to generate a multiple sequence alignment — this significantly improves prediction quality
3. If a ligand is involved, use lookup_ccd_code(molecule_name="...") to get the CCD code
4. Call the predictor with the msa_ref:
   - Single protein: openfold3(sequence_ref="...", msa_ref="...")
   - Protein-ligand: openfold3(molecules=[{type: "protein", sequence_ref: "..."}, {type: "ligand", ccd_codes: "..."}], msa_ref="...")
   - Novel ligand or affinity wanted: boltz2(polymers=[{molecule_type: "protein", sequence_ref: "..."}], ligands=[{smiles: "...", predict_affinity: true}], msa_ref="...")
5. Call visualize_pdb with the resulting reference
6. Summarize the prediction result

IMPORTANT: Always complete visualization. Do not stop after prediction.
IMPORTANT: Always run msa_search before predicting, for higher quality predictions.
OpenFold3 prefers CCD codes over SMILES for ligands.
Boltz-2 returns an mmcif reference. visualize_pdb accepts it directly; use
convert_structure only when a downstream tool needs PDB.
A Boltz-2 pIC50 is a model estimate for ranking, not a measured affinity.
"""


def build_structure_prediction_graph(
    llm: Any,
    tools: list[Any],
) -> Any:
    """Build the structure prediction subgraph."""
    from api.graph.state_modifier import inject_state_into_prompt

    relevant_names = {
        "fetch_protein_sequence",
        "msa_search",
        "openfold3",
        "boltz2",
        "convert_structure",
        "visualize_pdb",
        "lookup_ccd_code",
    }
    workflow_tools = [t for t in tools if t.name in relevant_names]
    if not workflow_tools:
        workflow_tools = tools

    llm_with_tools = llm.bind_tools(workflow_tools)

    async def agent_node(state: StructurePredictionState) -> dict:
        viewer = state.get("viewer_state") or {}
        storage = state.get("storage_state") or {}
        selections = state.get("selections") or []
        findings = state.get("findings") or []

        augmented = inject_state_into_prompt(
            STRUCTURE_PREDICTION_SYSTEM,
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

    def should_continue(state: StructurePredictionState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        if not messages:
            return "__end__"
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    async def refresh_node(
        state: StructurePredictionState, config: Optional[RunnableConfig] = None
    ) -> dict:
        from api.graph.state import refresh_state_from_backends

        return await refresh_state_from_backends(state, config)

    async def await_pending_ops_node(state: StructurePredictionState) -> dict:
        from api.graph.tools import drain_pending_ops

        unacked = await drain_pending_ops(timeout=10.0)
        if unacked:
            logger.warning(f"await_pending_ops: unacknowledged ops: {unacked}")
        return {}

    tool_node = ToolNode(workflow_tools)

    graph = StateGraph(StructurePredictionState)
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
