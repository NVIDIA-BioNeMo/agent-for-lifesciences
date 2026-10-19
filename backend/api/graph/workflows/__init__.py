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

"""Workflow subgraphs for Aevom.

Each module defines a compiled subgraph that encodes one of the four
primary workflows from prompts.md as a deterministic LangGraph graph.
These subgraphs are exposed to the main agent as callable tools.
"""

import logging
from typing import Any

from langchain_core.runnables.config import var_child_runnable_config
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)

# How many agent turns a workflow subgraph gets. Unlike the main graph, none of
# the subgraphs routes to a `force_respond` of its own, so LangGraph's super-step
# budget is the only stop they have -- which is why `_run_subgraph` treats
# exhausting it as a result to report rather than an error to raise.
WORKFLOW_MAX_ITERATIONS = 25


def _render_subgraph_result(result: dict[str, Any]) -> str:
    """Render a workflow's structured return as the tool response.

    The measurements are stated as the child's tools produced them rather than
    left to the child's closing paragraph to paraphrase. Artifact references
    are listed explicitly because the parent cannot see the child's tool
    output and would otherwise have no way to name what was created.
    """
    parts = [result.get("summary") or "Workflow completed but produced no output."]

    artifacts = result.get("artifacts_created") or []
    if artifacts:
        parts.append(f"Artifacts created: {', '.join(artifacts)}")

    findings = result.get("findings") or []
    if findings:
        measured = "\n".join(
            f"- {finding.get('summary', '')}"
            for finding in findings
            if finding.get("summary")
        )
        if measured:
            parts.append(f"Measured during this workflow:\n{measured}")

    return "\n\n".join(parts)


def build_workflow_tools(llm: Any, tools: list[Any]) -> list[StructuredTool]:
    """Build LangChain tools that invoke workflow subgraphs.

    Each workflow subgraph is compiled and wrapped as a StructuredTool so
    the main agent's LLM can call it like any other tool. When invoked,
    the subgraph takes over, executes its multi-step sequence, and returns
    the final result to the main agent.

    Args:
        llm: ChatAevomLLM instance.
        tools: All LangChain tool wrappers (subgraphs pick the subset they need).

    Returns:
        List of StructuredTool wrappers for each workflow subgraph.
    """
    from api.graph.workflows.small_molecule import (
        build_small_molecule_graph,
        WORKFLOW_DESCRIPTION as SM_DESC,
    )
    from api.graph.workflows.structure_prediction import (
        build_structure_prediction_graph,
        WORKFLOW_DESCRIPTION as SP_DESC,
    )
    from api.graph.workflows.docking import (
        build_docking_graph,
        WORKFLOW_DESCRIPTION as DK_DESC,
    )
    from api.graph.workflows.protein_ligand import (
        build_protein_ligand_graph,
        WORKFLOW_DESCRIPTION as PL_DESC,
    )
    from api.graph.agent import NODES_PER_ITERATION
    from langchain_core.messages import HumanMessage
    from langgraph.errors import GraphRecursionError

    # Derived rather than fixed, for the reason spelled out on
    # api.graph.agent.GRAPH_RECURSION_LIMIT: a literal budget silently stops
    # meaning what it says the moment the node cycle changes shape.
    workflow_recursion_limit = WORKFLOW_MAX_ITERATIONS * NODES_PER_ITERATION

    sm_graph = build_small_molecule_graph(llm, tools)
    sp_graph = build_structure_prediction_graph(llm, tools)
    dk_graph = build_docking_graph(llm, tools)
    pl_graph = build_protein_ligand_graph(llm, tools)

    async def _run_subgraph(graph: Any, query: str) -> dict[str, Any]:
        """Stream a workflow subgraph, forwarding inner tool events to the
        shared workflow event queue so they appear on the frontend timeline.

        Reads storage, session_id, and request_id from the parent graph's
        ContextVars so the subgraph can refresh state and look up selections
        under the correct session.

        Returns:
            ``{summary, artifacts_created, findings}``. Structured rather than
            a bare string because the child's trajectory never touches the
            parent -- that firewall is the point -- so whatever the child
            measured has to cross the boundary as data. Flattened into prose it
            would be the parent's word for a number rather than the number.
        """
        from api.conversations import get_recorder
        from api.graph.state import refresh_state_from_backends
        from api.graph.tools import (
            _graph_storage_ctx,
            _graph_session_id_ctx,
            _graph_request_id_ctx,
            get_workflow_event_queue,
        )
        from api.tools.registry import get_registry
        from langchain_core.messages import AIMessage, ToolMessage

        storage = _graph_storage_ctx.get()
        session_id = _graph_session_id_ctx.get()
        request_id = _graph_request_id_ctx.get()
        wf_queue = get_workflow_event_queue()
        registry = get_registry()
        recorder = get_recorder()

        subgraph_config: dict[str, Any] = {
            "recursion_limit": workflow_recursion_limit,
            "configurable": {"storage": storage},
        }

        # The child starts from a fresh [HumanMessage] -- that firewall is the
        # point of a subgraph. Its *state* is the parent's, though: this used to
        # be seeded with a literal empty storage state, so the first LLM call
        # inside a workflow was told the session held no artifacts while the
        # parent was looking at a docked complex. Seeded through the refresh
        # node's own function, so the child's first call sees what the parent's
        # last one did -- the session log included, because a workflow that
        # re-docks something the parent already measured spends a GPU-minute to
        # learn a number the ledger already holds.
        initial_state: dict[str, Any] = {
            "messages": [HumanMessage(content=query)],
            "viewer_state": {"is_empty": True, "structures": []},
            "storage_state": {"is_empty": True, "artifacts": []},
            "selections": [],
            "findings": [],
            "tool_budgets": [],
            "candidates": [],
            "exclusions": [],
            "nim_notices": [],
            "request_id": request_id,
            "session_id": session_id,
        }
        try:
            initial_state.update(
                await refresh_state_from_backends(initial_state, subgraph_config)
            )
        except Exception:
            # Degrades to the empty-shaped defaults above rather than ending the
            # user's turn: a workflow that runs with a stale envelope is worse
            # than one that runs with a fresh one, and better than one that
            # never runs.
            logger.warning(
                "Could not seed the workflow with the session's state", exc_info=True
            )
        findings_before = len(initial_state["findings"])

        # Prevent inner graph events from leaking into the outer
        # astream_events stream. The inner graph uses identical node
        # names ("agent", "tools", etc.), which confuses the streaming
        # layer's event matching.
        token = var_child_runnable_config.set(None)
        final_content = ""
        try:
            async for event in graph.astream_events(
                initial_state,
                version="v2",
                config=subgraph_config,
            ):
                kind = event.get("event")
                data = event.get("data", {})
                name = event.get("name", "")
                node = event.get("metadata", {}).get("langgraph_node", "")

                if kind == "on_chat_model_stream" and node == "agent":
                    chunk = data.get("chunk")
                    if chunk and wf_queue is not None:
                        reasoning = getattr(chunk, "additional_kwargs", {}).get("reasoning")
                        if reasoning:
                            wf_queue.put_nowait({"workflow_reasoning": reasoning})

                elif kind == "on_chat_model_end" and node == "agent":
                    output = data.get("output")
                    if not (output and isinstance(output, AIMessage)):
                        continue
                    if output.tool_calls:
                        if wf_queue is not None:
                            for tc in output.tool_calls:
                                tc_id = tc.get("id", "")
                                tc_name = tc.get("name", "")
                                tc_args = tc.get("args", {})
                                tool_def = registry.get(tc_name)
                                summary = (
                                    tool_def.format_summary(tc_args)
                                    if tool_def
                                    else f"{tc_name} called"
                                )
                                wf_queue.put_nowait({
                                    "workflow_tool_call_start": {
                                        "id": tc_id,
                                        "name": tc_name,
                                        "args": tc_args,
                                        "summary": summary,
                                    }
                                })
                    elif output.content:
                        final_content = output.content

                elif kind == "on_tool_end" and node == "tools":
                    if wf_queue is not None:
                        output = data.get("output")
                        tool_call_id = ""
                        success = True
                        output_preview = ""
                        if isinstance(output, ToolMessage):
                            tool_call_id = output.tool_call_id or ""
                            raw = str(output.content)
                            success = "Error:" not in raw[:50]
                            output_preview = raw
                        wf_queue.put_nowait({
                            "workflow_tool_call_end": {
                                "id": tool_call_id,
                                "name": name,
                                "success": success,
                                "output_preview": output_preview,
                            }
                        })

                elif kind == "on_chain_end" and name == "refresh_state":
                    output = data.get("output", {})
                    storage_state = output.get("storage_state", {})
                    if storage_state and not storage_state.get("is_empty", True):
                        artifacts = storage_state.get("artifacts", [])
                        if wf_queue is not None:
                            wf_queue.put_nowait({
                                "workflow_storage_update": {
                                    "artifacts": artifacts,
                                }
                            })

        except GraphRecursionError:
            # Running out of budget is an ordinary outcome for a subgraph that
            # has no other stop, so it crosses back as a result. Raised, it
            # would leave the tool through LangGraph's error handler and end the
            # user's whole turn -- discarding minutes of fetching and measuring
            # that the findings below still describe.
            logger.warning(
                f"Workflow exhausted its {WORKFLOW_MAX_ITERATIONS}-iteration budget; "
                "returning partial results"
            )
            note = (
                f"This workflow used all {WORKFLOW_MAX_ITERATIONS} of its steps "
                "without reaching a conclusion. Everything it established is "
                "below; continue from there rather than starting it again."
            )
            final_content = f"{final_content}\n\n{note}" if final_content else note
        finally:
            var_child_runnable_config.reset(token)

        # Everything the child's tools recorded while it ran. They already
        # sit in the session ledger -- the recorder is request-scoped, so
        # subgraph tools file findings like any other -- but the parent
        # needs to know which of them are the answer to the call it just
        # made.
        child_findings: list[dict] = []
        if recorder is not None:
            child_findings = (await recorder.get_findings())[findings_before:]

        return {
            "summary": final_content,
            "artifacts_created": sorted(
                {
                    ref
                    for finding in child_findings
                    for ref in finding.get("output_refs") or []
                }
            ),
            "findings": child_findings,
        }

    async def run_small_molecule_analysis(query: str) -> str:
        return _render_subgraph_result(await _run_subgraph(sm_graph, query))

    async def run_structure_prediction(query: str) -> str:
        return _render_subgraph_result(await _run_subgraph(sp_graph, query))

    async def run_docking_workflow(query: str) -> str:
        return _render_subgraph_result(await _run_subgraph(dk_graph, query))

    async def run_protein_ligand_analysis(query: str) -> str:
        return _render_subgraph_result(await _run_subgraph(pl_graph, query))

    workflow_tools = [
        StructuredTool.from_function(
            coroutine=run_small_molecule_analysis,
            name="run_small_molecule_analysis",
            description=SM_DESC,
        ),
        StructuredTool.from_function(
            coroutine=run_structure_prediction,
            name="run_structure_prediction",
            description=SP_DESC,
        ),
        StructuredTool.from_function(
            coroutine=run_docking_workflow,
            name="run_docking_workflow",
            description=DK_DESC,
        ),
        StructuredTool.from_function(
            coroutine=run_protein_ligand_analysis,
            name="run_protein_ligand_analysis",
            description=PL_DESC,
        ),
    ]

    # Structure prediction is OpenFold3 end to end; without the NVIDIA key that
    # serves it, the subgraph would fall back to every tool and wander.
    from api.tools.registry import nim_tools_unavailable

    if nim_tools_unavailable():
        workflow_tools = [t for t in workflow_tools if t.name != "run_structure_prediction"]

    logger.info(
        f"Built {len(workflow_tools)} workflow tools: "
        f"{[t.name for t in workflow_tools]}"
    )
    return workflow_tools
