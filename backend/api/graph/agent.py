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

"""Main Aevom agent graph built with LangGraph.

Implements a ReAct-style agent using LangGraph's ``StateGraph``.
The graph has three node types:

1. **agent** — Calls the LLM with tools bound. The LLM decides whether
   to call a tool or produce a final response.
2. **tools** — Executes tool calls returned by the LLM, appending
   ``ToolMessage`` results to state.
3. **refresh_state** — After tool execution, refreshes viewer_state,
   storage_state, and selections from Redis/DB so the next LLM turn
   sees up-to-date context.

The graph loops: agent -> tools -> refresh_state -> agent until the
LLM produces a response with no tool calls.
"""

import logging
import os
from typing import Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from api.graph.state import AevomState, refresh_state_from_backends
from api.telemetry import IterationSpanContext, set_current_iteration_span

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 50

# One iteration walks agent -> tools -> await_pending_ops -> refresh_state, so
# LangGraph's super-step budget has to scale with the iteration cap. Left fixed,
# it becomes the binding constraint and the run dies with GraphRecursionError
# partway to the cap instead of stopping cleanly through force_respond.
NODES_PER_ITERATION = 4
GRAPH_RECURSION_LIMIT = MAX_ITERATIONS * NODES_PER_ITERATION + 20

# LangGraph isolates ContextVars per node (each node runs in a copied context).
# ContextVar changes in agent_node don't propagate to tool_node. We use a plain
# dict keyed by request_id to share the active iteration span across nodes.
_active_iter_spans: dict[str, tuple[IterationSpanContext, int]] = {}


def _rotate_iteration_span(request_id: str) -> None:
    """Close the previous iteration span and open a new one.

    Called at the start of each ``agent_node`` invocation so the LLM span
    and subsequent tool spans nest under the correct iteration.
    """
    prev = _active_iter_spans.get(request_id)
    if prev:
        prev[0].__exit__(None, None, None)

    counter = (prev[1] if prev else 0) + 1
    ctx = IterationSpanContext(counter)
    ctx.__enter__()
    _active_iter_spans[request_id] = (ctx, counter)


def restore_iteration_span(request_id: str) -> None:
    """Restore ``_current_iteration_span`` ContextVar in the current node.

    LangGraph copies ContextVars per node, so changes from ``agent_node``
    don't persist to ``tool_node``.  Call this at the start of any node
    that needs the iteration span to be active.
    """
    entry = _active_iter_spans.get(request_id)
    if entry:
        set_current_iteration_span(entry[0].span)


def reset_iteration_tracking(request_id: str) -> None:
    """Close the active iteration span and clean up. Call at end of request."""
    entry = _active_iter_spans.pop(request_id, None)
    if entry:
        entry[0].__exit__(None, None, None)
    set_current_iteration_span(None)


def _should_continue(state: AevomState) -> Literal["tools", "force_respond", "__end__"]:
    """Routing function: continue to tools, force a final response, or end."""
    messages = state["messages"]
    if not messages:
        return "__end__"
    last = messages[-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        if state.get("iteration_count", 0) >= MAX_ITERATIONS:
            return "force_respond"
        return "tools"
    return "__end__"


async def _await_pending_ops_node(state: AevomState) -> dict:
    """Graph node that waits for pending frontend operations (e.g. selection POSTs).

    Runs after the tools node so that the SSE events have been delivered
    to the frontend before we block.  The streaming layer drains the
    viz_event_queue on ``on_tool_end``, and this node runs afterward,
    giving the frontend time to execute MolQL queries and POST results.

    After draining, any structure_query ToolMessages that contain a
    ``[PENDING:<query_id>]`` marker are replaced with rich content
    (atom/residue counts, reference, or error) looked up from Redis.
    LangGraph's ``add_messages`` reducer handles the replacement by
    matching on the ToolMessage ``id``.
    """
    from api.graph.tools import drain_pending_ops, get_and_clear_pending_structure_queries
    from api.selections.registry import get_selection_by_query_id
    from api.selections.format import (
        format_selection_success,
        format_selection_empty_with_hints,
        format_selection_timeout,
        format_selection_no_data,
        format_selection_query_error,
        is_molql_error_summary,
    )
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
            new_content = format_selection_success(
                sel_info.atom_count, sel_info.residue_count, sel_info.reference
            )
        elif sel_info and is_molql_error_summary(sel_info.summary):
            # The expression never ran. Say so, rather than the "matched 0
            # atoms" below, which reads as a fact about the structure.
            new_content = format_selection_query_error(sel_info.summary)
        elif sel_info and sel_info.atom_count == 0:
            vs = state.get("viewer_state", {})
            structs = vs.get("structures", [])
            new_content = format_selection_empty_with_hints(structs)
        elif query_id in unacked:
            new_content = format_selection_timeout()
        else:
            new_content = format_selection_no_data()

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

        if not target_msg:
            logger.warning(
                f"await_pending_ops: no ToolMessage found for query_id={query_id} "
                f"(handled by workflow subgraph?)"
            )
            continue

        replacement = ToolMessage(
            content=new_content,
            tool_call_id=target_msg.tool_call_id,
            id=target_msg.id,
        )
        replacements.append(replacement)
        logger.info(
            f"await_pending_ops: patched ToolMessage for query_id={query_id} "
            f"-> {new_content[:80]}..."
        )

    if replacements:
        return {"messages": replacements}
    return {}


async def _refresh_node(
    state: AevomState, config: Optional[RunnableConfig] = None
) -> dict:
    """Graph node that refreshes state from backends after tool execution.

    ``config`` must be annotated as ``Optional[RunnableConfig]``: LangGraph
    only injects the runnable config into nodes whose annotation it
    recognises, and silently passes nothing otherwise. Annotated as ``dict``
    this node ran with ``config=None``, so it could not reach the request's
    storage service and reported the session as having no artifacts.
    """
    return await refresh_state_from_backends(state, config)


_FORCE_RESPOND_NUDGE = (
    "You have reached the maximum number of tool-calling iterations. "
    "Do not call any more tools. Provide your best response to the user "
    "based on all the work and results you have gathered so far."
)


def workflows_enabled() -> bool:
    """Whether the workflow subgraph meta-tools are exposed.

    On in production. A benchmark run that has to match another agent's tool
    set turns them off with ``AEVOM_INCLUDE_WORKFLOWS=false``, since they are
    the one part of the toolset that is not a tool the model could have been
    given anywhere else.
    """
    return os.getenv("AEVOM_INCLUDE_WORKFLOWS", "true").strip().lower() not in (
        "false",
        "0",
        "no",
    )


def build_agent_graph(
    llm: Any,
    tools: list[Any],
    system_prompt: str,
    max_iterations: int = MAX_ITERATIONS,
    include_workflows: Optional[bool] = None,
) -> Any:
    """Compile the Aevom agent graph.

    Args:
        llm: ChatAevomLLM instance (or any LangChain ChatModel).
        tools: List of LangChain StructuredTool instances (individual tools).
        system_prompt: Base system prompt (before state injection).
        max_iterations: Max tool-calling loops before forced stop.
        include_workflows: Whether to add workflow subgraph tools. Defaults to
            what ``AEVOM_INCLUDE_WORKFLOWS`` says, which is yes.

    Returns:
        Compiled LangGraph graph ready for ``.ainvoke()`` or ``.astream_events()``.
    """
    from api.context import build_context_assembler

    if include_workflows is None:
        include_workflows = workflows_enabled()

    # Optionally add workflow subgraph tools alongside individual tools
    all_tools = list(tools)
    if include_workflows:
        from api.graph.workflows import build_workflow_tools
        workflow_tools = build_workflow_tools(llm, tools)
        all_tools.extend(workflow_tools)

    # Bind all tools to the LLM so they're included in every call
    llm_with_tools = llm.bind_tools(all_tools)

    # The bound schemas go out with every call, so the assembler has to charge
    # the budget for them the same way it charges for the system prompt.
    bound_schemas = (getattr(llm_with_tools, "kwargs", None) or {}).get("tools")
    state_modifier = build_context_assembler(
        system_prompt,
        config=getattr(llm, "client", None) and llm.client.config,
        tool_schemas=bound_schemas,
    )

    async def agent_node(state: AevomState) -> dict:
        """Call the LLM with state-modified messages and bound tools."""
        from api.telemetry import get_token_accumulator

        _rotate_iteration_span(state["request_id"])
        modified_messages = state_modifier(state)

        acc = get_token_accumulator()
        before_in = acc.input_tokens if acc else 0
        before_out = acc.output_tokens if acc else 0

        response = await llm_with_tools.ainvoke(modified_messages)

        update: dict = {"messages": [response], "iteration_count": 1}
        if acc:
            delta_in = acc.input_tokens - before_in
            delta_out = acc.output_tokens - before_out
            if delta_in or delta_out:
                update["total_episode_input_tokens"] = delta_in
                update["total_episode_output_tokens"] = delta_out
        return update

    _raw_tool_node = ToolNode(all_tools)

    async def tool_node(
        state: AevomState, config: Optional[RunnableConfig] = None
    ) -> dict:
        """Execute tools with the iteration span restored."""
        restore_iteration_span(state["request_id"])
        return await _raw_tool_node.ainvoke(state, config=config)

    async def force_respond_node(state: AevomState) -> dict:
        """Gracefully end the loop when the iteration limit is reached.

        Stubs out pending tool calls, instructs the LLM to summarise
        progress, and calls it without tools so it can only produce text.
        """
        from api.telemetry import get_current_chat_span

        last_msg = state["messages"][-1]
        stub_messages = [
            ToolMessage(
                content="[Skipped \u2014 iteration limit reached]",
                tool_call_id=tc["id"],
            )
            for tc in last_msg.tool_calls
        ]
        nudge = HumanMessage(content=_FORCE_RESPOND_NUDGE)

        modified = state_modifier(state) + stub_messages + [nudge]
        response = await llm.ainvoke(modified)

        chat_span = get_current_chat_span()
        if chat_span:
            chat_span.set_attribute("chat.hit_iteration_limit", True)

        return {"messages": stub_messages + [nudge, response]}

    # Build the graph
    graph = StateGraph(AevomState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("await_pending_ops", _await_pending_ops_node)
    graph.add_node("refresh_state", _refresh_node)
    graph.add_node("force_respond", force_respond_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", _should_continue)
    graph.add_edge("tools", "await_pending_ops")
    graph.add_edge("await_pending_ops", "refresh_state")
    graph.add_edge("refresh_state", "agent")
    graph.add_edge("force_respond", END)

    compiled = graph.compile()

    logger.info(
        f"Agent graph compiled with {len(all_tools)} tools "
        f"({len(tools)} individual + "
        f"{len(all_tools) - len(tools)} workflow subgraphs), "
        f"max_iterations={max_iterations}"
    )
    return compiled
