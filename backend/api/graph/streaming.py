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

"""LangGraph streaming runner — maps ``astream_events()`` to Aevom SSE.

Replaces the bespoke ``StreamingRunner`` with a thin adapter that
translates LangGraph's event stream into the SSE event format that
the frontend expects (reasoning, tool_call_start, tool_call_end,
tool_event, storage_update, content).
"""

import asyncio
import logging
import uuid
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import AIMessage, ToolMessage
from api.context import rehydrate_messages
from api.conversations import reset_recorder, set_recorder
from api.graph.state import AevomState, refresh_state_from_backends
from api.graph.tools import set_tool_context, reset_tool_context
from api.graph.agent import GRAPH_RECURSION_LIMIT, reset_iteration_tracking
from api.telemetry import (
    ChatSpanContext,
    get_current_span_id,
    TokenAccumulator,
    set_token_accumulator,
    get_current_chat_span,
)

logger = logging.getLogger(__name__)

_STREAM_EXHAUSTED = object()


async def _safe_anext(aiter):
    """Advance an async iterator, returning ``_STREAM_EXHAUSTED`` on end.

    Wrapping ``__anext__`` avoids the PEP-479 ``RuntimeError`` that occurs
    when ``StopAsyncIteration`` propagates out of an ``asyncio.Task``.
    """
    try:
        return await aiter.__anext__()
    except StopAsyncIteration:
        return _STREAM_EXHAUSTED


def _usage_event(acc: TokenAccumulator) -> dict:
    """The turn's token totals as an SSE frame.

    Sent once, when the stream ends, because the count is only complete after
    the last LLM call of the turn -- a turn spans as many calls as the agent
    takes iterations. The eval harness reads this to fill Harbor's token
    column, and the frontend ignores frames it has no branch for.
    """
    return {
        "usage": {
            "input_tokens": acc.input_tokens,
            "output_tokens": acc.output_tokens,
            "total_tokens": acc.input_tokens + acc.output_tokens,
        }
    }


def _invocation_fields(tool_name: str) -> dict:
    """Canonical-event fields to merge into a ``tool_call_end`` payload.

    Closes the ADR 0001 gap that left ``ToolInvocationEvent`` with no readers:
    the registry builds the event, the tool wrapper parks it on a ContextVar,
    and this is where it reaches the wire so the eval trajectory and the
    frontend can see policy and job outcomes rather than only a rendered string.

    Returns an empty dict when the tool ran outside a bound context (workflow
    subgraphs, tests) or already had its event consumed. Absent keys mean "not
    known for this call" -- never "no job" or "no policy" -- so consumers must
    treat absence as unknown.
    """
    from api.graph.tools import take_invocation_event

    event = take_invocation_event(tool_name)
    if event is None:
        return {}
    fields: dict[str, Any] = {
        "job_id": event.job_id,
        "job_status": event.job_status,
        "policy_decision": event.policy_decision,
        "artifact_refs": event.artifact_refs,
    }
    # result_ref has been on this payload since it was written but was hardcoded
    # None; the event now carries the artifact the call produced, which is what
    # the field was for.
    if event.artifact_refs:
        fields["result_ref"] = event.artifact_refs[0]
    return fields


async def stream_graph(
    graph: Any,
    *,
    system_prompt: str,
    conversation_history: list[dict],
    viewer_state: Optional[dict],
    session_id: str,
    storage: Any,
    request_id: str,
    db_session_factory: Any,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    model_name: Optional[str] = None,
    recorder: Optional[Any] = None,
    job_manager: Optional[Any] = None,
) -> AsyncIterator[dict]:
    """Run the LangGraph agent and yield SSE events for the frontend.

    This is the main entry point that replaces ``StreamingRunner.run_streaming()``.

    Args:
        graph: Compiled LangGraph agent.
        system_prompt: Rendered system prompt (before state injection).
        conversation_history: List of message dicts from the request.
        viewer_state: Current Molstar viewer state from the frontend.
        session_id: Client session identifier.
        storage: DataStorageService instance.
        request_id: Unique ID for this request (Redis key).
        db_session_factory: DB session factory for storage init.
        temperature: LLM temperature.
        max_tokens: Optional max tokens for LLM.
        model_name: Model name, for the root span.
        recorder: Optional TurnRecorder. When given, the trajectory is written
            to the durable record as it happens rather than being reconstructed
            from what the browser happens to have kept.
        job_manager: Optional ``JobManager`` (``api.jobs.manager``). When
            given, tools that opted into job tracking (``is_async == True``)
            run through it so their execution is recorded as a durable Job
            row -- see ADR 0001. Defaults to None (today's behavior).

    Yields:
        SSE event dicts matching the existing frontend contract.
    """
    from api.tools.viewer_state import set_viewer_state

    # Before the refresh below, so it reads back what the frontend just sent.
    await set_viewer_state(request_id, viewer_state)

    graph_config: dict[str, Any] = {
        "configurable": {
            "temperature": temperature,
            # Services, not state: they never change during a run and cannot
            # be serialised.
            "storage": storage,
            "db_session_factory": db_session_factory,
        },
        # Safety net only; the actual iteration limit is MAX_ITERATIONS in
        # agent.py. Derived from it so the two cannot drift -- a fixed value
        # here silently becomes the binding constraint as soon as the cap is
        # raised.
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }

    # Expose the recorder to the tool registry, several layers down, so it can
    # file a finding per execution without threading it through every caller.
    # Set before the seed below, not after: the ledger loaders fall back to the
    # recorder when there is no session factory to read from.
    recorder_token = set_recorder(recorder)

    # Rebuild the trajectory from the record, tool calls and reasoning included.
    # When the caller is a legacy client sending flat role/content pairs this
    # degenerates to the old behaviour, since there is nothing else to restore.
    lc_messages = rehydrate_messages(conversation_history)

    # Seeded through the refresh node's own function rather than a hand-rolled
    # copy of it. The graph runs START -> agent, so the turn's *planning* call
    # reads whatever is seeded here; assembling it any other way is how this
    # seed came to omit candidates, exclusions and tool budgets while every
    # later call in the same turn had them.
    initial_state: AevomState = {
        "messages": lc_messages,
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
        "total_episode_input_tokens": 0,
        "total_episode_output_tokens": 0,
        "iteration_count": 0,
    }
    # Empty-shaped defaults above, updated rather than replaced: the refresh
    # omits findings and candidates when they cannot be read, since None there
    # means "unknown", and no key of this TypedDict may go missing.
    initial_state.update(
        await refresh_state_from_backends(initial_state, graph_config)
    )

    # Set tool context for ContextVar-based injection
    tokens, viz_queue, retry_queue, wf_queue = set_tool_context(
        storage, session_id, request_id, job_manager=job_manager
    )

    # Create per-request token accumulator (shared across graph nodes via ContextVar)
    token_acc = TokenAccumulator()
    set_token_accumulator(token_acc)

    # Extract user message for the root span attribute
    user_message = ""
    for msg in reversed(conversation_history):
        if msg.get("role") == "user":
            user_message = msg.get("content", "")
            break

    iteration = 0
    active_tool_calls: dict[str, dict] = {}
    pending_tool_ends: list[dict] = []
    outer_tool_ids: set[str] = set()
    next_event_task: asyncio.Task | None = None

    try:
        async with ChatSpanContext(
            user_message, model_name or "aevom-llm", session_id=session_id
        ):
            span_id = get_current_span_id()
            if span_id:
                yield {"span_id": span_id}

            graph_aiter = graph.astream_events(
                initial_state,
                version="v2",
                # The same config the seed above was built from, so the two
                # cannot disagree about which storage service they mean.
                config=graph_config,
            ).__aiter__()

            while True:
                # Drain retry status queue on every poll cycle so events
                # reach the frontend even while the graph is blocked
                # (e.g. during synchronous rate-limit retry sleeps).
                while not retry_queue.empty():
                    try:
                        status = retry_queue.get_nowait()
                        yield {"rate_limit_status": status}
                    except asyncio.QueueEmpty:
                        break

                # Drain viz event queue on every poll cycle so events from
                # workflow subgraphs reach the frontend while the inner
                # graph is blocked in drain_pending_ops. Without this,
                # structure_query events inside workflows never reach the
                # frontend, causing a deadlock (the frontend can't ACK
                # a selection it never received).
                while not viz_queue.empty():
                    try:
                        yield viz_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                # Drain workflow event queue so inner subgraph tool calls
                # appear on the frontend timeline in real time.
                while not wf_queue.empty():
                    try:
                        wf_event = wf_queue.get_nowait()
                        if "workflow_tool_call_start" in wf_event:
                            yield {
                                "tool_call_start": wf_event["workflow_tool_call_start"],
                                "iteration": iteration,
                            }
                        elif "workflow_tool_call_end" in wf_event:
                            yield {
                                "tool_call_end": wf_event["workflow_tool_call_end"],
                                "iteration": iteration,
                            }
                        elif "workflow_storage_update" in wf_event:
                            yield {
                                "storage_update": wf_event["workflow_storage_update"],
                            }
                        elif "workflow_reasoning" in wf_event:
                            yield {
                                "reasoning": wf_event["workflow_reasoning"],
                                "iteration": iteration,
                            }
                    except asyncio.QueueEmpty:
                        break

                if next_event_task is None:
                    next_event_task = asyncio.ensure_future(
                        _safe_anext(graph_aiter)
                    )

                done, _ = await asyncio.wait(
                    {next_event_task}, timeout=0.5
                )

                if not done:
                    continue

                event = next_event_task.result()
                next_event_task = None

                if event is _STREAM_EXHAUSTED:
                    chat_span = get_current_chat_span()
                    if chat_span and token_acc:
                        chat_span.set_attribute(
                            "episode.total_input_tokens", token_acc.input_tokens
                        )
                        chat_span.set_attribute(
                            "episode.total_output_tokens", token_acc.output_tokens
                        )
                        chat_span.set_attribute(
                            "episode.total_tokens",
                            token_acc.input_tokens + token_acc.output_tokens,
                        )
                    # Outside the span guard above: the totals are worth
                    # reporting whether or not tracing is on. Only the clean
                    # end of the stream sends this, so a turn that raised
                    # reports no usage rather than a total it never reached.
                    yield _usage_event(token_acc)
                    break

                kind = event.get("event")
                data = event.get("data", {})
                name = event.get("name", "")
                metadata = event.get("metadata", {})
                node = metadata.get("langgraph_node", "")

                # --- Agent / force_respond node: LLM reasoning + content streaming ---
                if kind == "on_chat_model_stream" and node in ("agent", "force_respond"):
                    chunk = data.get("chunk")
                    if chunk:
                        reasoning = getattr(chunk, "additional_kwargs", {}).get("reasoning")
                        if reasoning:
                            yield {"reasoning": reasoning, "iteration": iteration}

                        if chunk.content and not outer_tool_ids:
                            yield {"content": chunk.content}

                # --- Force respond node complete: notify frontend ---
                elif kind == "on_chat_model_end" and node == "force_respond":
                    output = data.get("output")
                    if recorder is not None and isinstance(output, AIMessage):
                        await recorder.record_assistant(
                            content=str(output.content or ""),
                            reasoning=output.additional_kwargs.get("reasoning"),
                        )
                    yield {"hit_iteration_limit": True}

                # --- Agent node complete: check for tool calls ---
                elif kind == "on_chat_model_end" and node == "agent":
                    output = data.get("output")
                    if output and isinstance(output, AIMessage):
                        if recorder is not None:
                            await recorder.record_assistant(
                                content=str(output.content or ""),
                                reasoning=output.additional_kwargs.get("reasoning"),
                                tool_calls=output.tool_calls or None,
                            )
                        if output.tool_calls:
                            is_outer = not outer_tool_ids
                            if is_outer:
                                iteration += 1
                                if recorder is not None:
                                    recorder.iteration = iteration

                            for tc in output.tool_calls:
                                tc_id = tc.get("id", str(uuid.uuid4()))
                                tc_name = tc.get("name", "")
                                tc_args = tc.get("args", {})

                                from api.tools.registry import get_registry
                                registry = get_registry()
                                tool = registry.get(tc_name)
                                try:
                                    summary = (
                                        tool.format_summary(tc_args)
                                        if tool else f"{tc_name} called"
                                    )
                                except Exception:
                                    logger.debug(f"format_summary failed for {tc_name}", exc_info=True)
                                    summary = f"{tc_name} called"

                                active_tool_calls[tc_id] = {
                                    "name": tc_name,
                                    "args": tc_args,
                                }
                                if is_outer:
                                    outer_tool_ids.add(tc_id)

                                yield {
                                    "tool_call_start": {
                                        "id": tc_id,
                                        "name": tc_name,
                                        "args": tc_args,
                                        "summary": summary,
                                    },
                                    "iteration": iteration,
                                }

                # --- Tool node: individual tool execution complete ---
                elif kind == "on_tool_end" and node == "tools":
                    output = data.get("output")
                    tool_name = name

                    if recorder is not None and isinstance(output, ToolMessage):
                        await recorder.record_tool(
                            content=str(output.content),
                            tool_call_id=output.tool_call_id,
                            name=tool_name,
                        )

                    matched_id = None
                    for tc_id, tc_info in active_tool_calls.items():
                        if tc_info["name"] == tool_name:
                            matched_id = tc_id
                            break

                    if matched_id:
                        success = True
                        output_preview = ""
                        if isinstance(output, ToolMessage):
                            raw = str(output.content)
                            success = "Error:" not in raw[:50]
                            output_preview = raw

                        is_outer_tool = matched_id in outer_tool_ids
                        del active_tool_calls[matched_id]

                        # Taken once per tool end, before the branch: both
                        # branches below are exclusive emit sites for the same
                        # call, and the event is consumed on read.
                        invocation_fields = _invocation_fields(tool_name)

                        if is_outer_tool:
                            pending_tool_ends.append({
                                "tool_call_end": {
                                    "id": matched_id,
                                    "name": tool_name,
                                    "result_ref": None,
                                    "success": success,
                                    "output_preview": output_preview,
                                    **invocation_fields,
                                },
                                "iteration": iteration,
                            })
                            outer_tool_ids.discard(matched_id)

                            if not outer_tool_ids and pending_tool_ends:
                                viz_events: list[dict] = []
                                while not viz_queue.empty():
                                    try:
                                        viz_events.append(viz_queue.get_nowait())
                                    except asyncio.QueueEmpty:
                                        break

                                if viz_events:
                                    _VIZ_ORDER = {
                                        "visualize_3d": 0,
                                        "visualize_pdb": 1,
                                        "structure_query": 2,
                                    }
                                    viz_events.sort(
                                        key=lambda e: _VIZ_ORDER.get(
                                            e.get("tool_event", {}).get("type", ""), 1
                                        )
                                    )
                                    for ve in viz_events:
                                        evt_type = ve.get("tool_event", {}).get("type", "?")
                                        logger.info(f"[streaming] Yielding viz event: type={evt_type}")
                                        yield ve

                                for te in pending_tool_ends:
                                    yield te
                                pending_tool_ends.clear()
                        else:
                            # Inner (subgraph) tool — yield immediately
                            viz_events: list[dict] = []
                            while not viz_queue.empty():
                                try:
                                    viz_events.append(viz_queue.get_nowait())
                                except asyncio.QueueEmpty:
                                    break

                            for ve in viz_events:
                                evt_type = ve.get("tool_event", {}).get("type", "?")
                                logger.info(f"[streaming] Yielding inner viz event: type={evt_type}")
                                yield ve

                            yield {
                                "tool_call_end": {
                                    "id": matched_id,
                                    "name": tool_name,
                                    "result_ref": None,
                                    "success": success,
                                    "output_preview": output_preview,
                                    **invocation_fields,
                                },
                                "iteration": iteration,
                            }
                    else:
                        logger.warning(
                            f"[streaming] Unmatched on_tool_end: name={tool_name}, "
                            f"active_tool_calls={list(active_tool_calls.keys())}, "
                            f"outer_tool_ids={outer_tool_ids}"
                        )

                # --- Tool node: tool execution error ---
                elif kind == "on_tool_error" and node == "tools":
                    tool_name = name
                    matched_id = None
                    for tc_id, tc_info in active_tool_calls.items():
                        if tc_info["name"] == tool_name:
                            matched_id = tc_id
                            break
                    if matched_id:
                        is_outer_tool = matched_id in outer_tool_ids
                        del active_tool_calls[matched_id]
                        error_msg = str(data.get("error", "Tool execution failed"))
                        # Drained even here: the registry catches its own
                        # failures and still emits an event, so this path
                        # usually has one. Draining also keeps the per-name
                        # FIFO aligned for the next call to the same tool.
                        end_event = {
                            "tool_call_end": {
                                "id": matched_id,
                                "name": tool_name,
                                "result_ref": None,
                                "success": False,
                                "output_preview": error_msg,
                                **_invocation_fields(tool_name),
                            },
                            "iteration": iteration,
                        }
                        if is_outer_tool:
                            pending_tool_ends.append(end_event)
                            outer_tool_ids.discard(matched_id)
                            if not outer_tool_ids and pending_tool_ends:
                                for te in pending_tool_ends:
                                    yield te
                                pending_tool_ends.clear()
                        else:
                            yield end_event
                    else:
                        logger.warning(
                            f"[streaming] Unmatched on_tool_error: name={tool_name}, "
                            f"active_tool_calls={list(active_tool_calls.keys())}, "
                            f"outer_tool_ids={outer_tool_ids}"
                        )

                # --- Await pending ops complete: emit resolved output for deferred tools ---
                elif kind == "on_chain_end" and name == "await_pending_ops":
                    output = data.get("output", {})
                    resolved_msgs = output.get("messages", [])
                    for msg in resolved_msgs:
                        if isinstance(msg, ToolMessage):
                            if recorder is not None:
                                await recorder.update_tool(
                                    msg.tool_call_id, str(msg.content)
                                )
                            yield {
                                "tool_call_update": {
                                    "id": msg.tool_call_id,
                                    "output_preview": str(msg.content),
                                },
                            }

                # --- Refresh state node complete: emit storage_update ---
                elif kind == "on_chain_end" and name == "refresh_state":
                    output = data.get("output", {})
                    storage_state = output.get("storage_state", {})
                    if storage_state and not storage_state.get("is_empty", True):
                        artifacts = storage_state.get("artifacts", [])
                        yield {
                            "storage_update": {
                                "artifacts": artifacts,
                            }
                        }

    except Exception as e:
        from api.graph.llm import ProviderUnavailableError

        if isinstance(e, ProviderUnavailableError):
            while not retry_queue.empty():
                try:
                    status = retry_queue.get_nowait()
                    yield {"rate_limit_status": status}
                except asyncio.QueueEmpty:
                    break
            logger.warning(f"{type(e).__name__} for {e.model_id}: {e}")
            yield {"content": str(e)}
        else:
            logger.exception(f"Error in LangGraph streaming: {e}")
            yield {"content": f"Error: {str(e)}"}
    finally:
        for te in pending_tool_ends:
            yield te
        pending_tool_ends.clear()

        if next_event_task is not None and not next_event_task.done():
            next_event_task.cancel()
        reset_iteration_tracking(request_id)
        reset_tool_context(tokens)
        reset_recorder(recorder_token)
        set_token_accumulator(None)
