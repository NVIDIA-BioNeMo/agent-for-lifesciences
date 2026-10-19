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

"""Small Molecule Informatics workflow subgraph.

Workflow 4 from prompts.md:
  Search PubChem -> Get properties -> Visualize

Trigger: "Show me aspirin", "Properties of caffeine", "Lipinski rules"
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
    "Run a small molecule informatics workflow: search PubChem for the molecule, "
    "retrieve its properties (MW, LogP, H-bond donors/acceptors), download 3D SDF, "
    "visualize it in the viewer, and optionally perform RDKit fragment analysis. "
    "Use this when the user asks about a specific small molecule's structure, "
    "properties, or Lipinski rule compliance."
)


class SmallMoleculeState(TypedDict):
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


SMALL_MOLECULE_SYSTEM = """You are executing the Small Molecule Informatics workflow.
Your task is to search for the molecule, get its properties, and visualize it.

Steps:
1. search_pubchem(name="...") to find the CID and SMILES
2. download_pubchem_sdf(cid=..., name="...") to get the 3D structure
3. visualize_sdf(sdf_ref="...", label="...") to display it in the viewer
4. get_pubchem_properties(cid=...) for MW, LogP, H-bond counts, TPSA
5. If the question is about safety, toxicity, permeability, solubility or
   whether the molecule is a plausible drug candidate, call
   admet_predict(smiles=["..."]). Report those values as predictions, and note
   that blood-brain barrier penetration and plasma protein binding are good or
   bad depending on the target
6. If the question is whether the molecule is drug-like or synthesizable, call
   molecule_gates(smiles=["..."]) for the full pass/fail panel -- descriptors,
   synthetic accessibility, PAINS/Brenk alerts and QED in one call
7. If the user asks for detailed analysis, use python_execute with RDKit for fragment analysis
8. Summarize all findings

IMPORTANT: You MUST complete all steps. Do not stop after search.
"""


def build_small_molecule_graph(
    llm: Any,
    tools: list[Any],
) -> Any:
    """Build the small molecule informatics subgraph."""
    from api.graph.state_modifier import inject_state_into_prompt

    # Filter to just the tools needed for this workflow
    relevant_names = {
        "search_pubchem",
        "get_pubchem_properties",
        "download_pubchem_sdf",
        "visualize_sdf",
        "python_execute",
        "rdkit_molecule_info",
        "admet_predict",
        "molecule_gates",
        "record_candidates",
        "query_candidates",
        "best_candidate",
    }
    workflow_tools = [t for t in tools if t.name in relevant_names]
    if not workflow_tools:
        workflow_tools = tools

    llm_with_tools = llm.bind_tools(workflow_tools)

    async def agent_node(state: SmallMoleculeState) -> dict:
        viewer = state.get("viewer_state") or {}
        storage = state.get("storage_state") or {}
        selections = state.get("selections") or []
        findings = state.get("findings") or []

        augmented_prompt = inject_state_into_prompt(
            SMALL_MOLECULE_SYSTEM,
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
                result_messages.append(SystemMessage(content=augmented_prompt))
                system_replaced = True
            else:
                result_messages.append(msg)
        if not system_replaced:
            result_messages.insert(0, SystemMessage(content=augmented_prompt))

        response = await llm_with_tools.ainvoke(result_messages)
        return {"messages": [response]}

    def should_continue(state: SmallMoleculeState) -> Literal["tools", "__end__"]:
        messages = state["messages"]
        if not messages:
            return "__end__"
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "__end__"

    async def refresh_node(
        state: SmallMoleculeState, config: Optional[RunnableConfig] = None
    ) -> dict:
        from api.graph.state import refresh_state_from_backends

        return await refresh_state_from_backends(state, config)

    async def await_pending_ops_node(state: SmallMoleculeState) -> dict:
        from api.graph.tools import drain_pending_ops

        unacked = await drain_pending_ops(timeout=10.0)
        if unacked:
            logger.warning(f"await_pending_ops: unacknowledged ops: {unacked}")
        return {}

    tool_node = ToolNode(workflow_tools)

    graph = StateGraph(SmallMoleculeState)
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
