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

"""LangChain tool wrappers for Aevom's existing ToolRegistry.

Each wrapper delegates to ``ToolRegistry.execute()`` so all existing
tool logic, storage patterns, telemetry, and ContextVar injection are
preserved without duplication.

The wrappers use ``StructuredTool.from_function()`` with the JSON schema
extracted from each tool's ``get_schema()`` to guarantee the LLM sees
identical parameter names, types, and descriptions.

Visualization side-effects (SDF_3D, PDB_3D, STRUCTURE_QUERY, and
python_execute viz events) are captured in a shared queue that the
streaming layer drains and emits as SSE ``tool_event`` messages.
"""

import asyncio
import json
import logging
from contextvars import ContextVar
from typing import Annotated, Any, Optional, Union

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, BeforeValidator, Field, create_model

logger = logging.getLogger(__name__)

# Per-request context injected by the streaming runner before graph invocation.
# Tool wrappers read these to pass storage/session/request context to the registry.
_graph_storage_ctx: ContextVar[Any] = ContextVar("graph_storage", default=None)
_graph_session_id_ctx: ContextVar[str] = ContextVar("graph_session_id", default="default")
_graph_request_id_ctx: ContextVar[str] = ContextVar("graph_request_id", default="default")
# Optional JobManager (api.jobs.manager) for the current request. None unless
# the caller wires one up (see main.py's chat endpoint) -- ADR 0001 P0 job
# semantics reference implementation. Absence is the default, unchanged path.
_graph_job_manager_ctx: ContextVar[Any] = ContextVar("graph_job_manager", default=None)

# Visualization event queue — populated by tool wrappers, drained by streaming.py.
_viz_event_queue: ContextVar[Optional[asyncio.Queue]] = ContextVar("viz_event_queue", default=None)

# Canonical ToolInvocationEvents accumulated during a tool step, keyed by tool
# name (ADR 0001). LangGraph's ToolNode requires the wrapper to return a string,
# so `result.to_tool_response()` below discards the ToolResult and with it the
# structured event -- which is exactly why that event had no readers when it was
# introduced. It rides this side channel instead, the same shape as
# _viz_event_queue above.
#
# Keyed by name and consumed FIFO because the wrapper never receives the
# tool_call id; that is the same correlation streaming.py already uses to match
# a tool_call_start to its tool_call_end, so this is no less precise than the
# event stream it feeds.
_invocation_events_ctx: ContextVar[Optional[dict[str, list]]] = ContextVar(
    "invocation_events", default=None
)

# Retry status queue — populated by ChatAevomLLM during 429 back-off, drained by streaming.py.
_retry_status_queue: ContextVar[Optional[asyncio.Queue]] = ContextVar("retry_status_queue", default=None)

# Workflow tool event queue — populated by _run_subgraph with inner tool_call_start/end
# events, drained by streaming.py so inner workflow tool calls appear in the timeline.
_workflow_event_queue: ContextVar[Optional[asyncio.Queue]] = ContextVar("workflow_event_queue", default=None)

# Pending ops accumulated during a single tool execution step.
# The await_pending_ops graph node drains this after tools complete.
_pending_ops_ctx: ContextVar[Optional[list]] = ContextVar("pending_ops", default=None)

# Tracks (query_id, session_id) tuples for structure_query calls in a tool step.
# Used by _await_pending_ops_node to patch ToolMessages with rich selection data.
# Initialized as a mutable list in set_tool_context() so mutations propagate
# across LangGraph nodes (which copy ContextVar tokens but share mutable objects).
_pending_structure_query_ids_ctx: ContextVar[Optional[list[tuple[str, str, Any]]]] = ContextVar(
    "pending_structure_query_ids", default=None
)


def _coerce_json_str(v: Any) -> Any:
    """Auto-parse JSON strings that LLMs double-encode as string literals.

    LLMs sometimes serialize array/object arguments as JSON strings instead of
    native JSON structures (e.g. ``"[{...}]"`` instead of ``[{...}]``).  This
    validator transparently unwraps the extra encoding layer so Pydantic
    receives the intended ``list`` or ``dict``.
    """
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, (list, dict)):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return v


def _coerce_scalar_str(v: Any) -> Any:
    """Accept a bare number where the schema declares a string.

    The mirror image of :func:`_coerce_json_str`. Models emit numeric-looking
    identifiers as JSON numbers -- ``organism_id: 9606`` rather than ``"9606"`` --
    and the resulting validation error names Pydantic rather than the fix, so the
    model has to guess. An eval baseline burned three consecutive
    ``fetch_protein_sequence`` calls on exactly that before quoting the taxonomy
    id.

    Only scalars are converted. A list or dict passed where a string belongs is a
    real mistake and still fails, rather than arriving as ``"['A']"``.
    """
    if isinstance(v, bool):
        # Deliberately before the numeric case: bool is an int subclass, and
        # "True" is not what a string field wants from a boolean.
        return v
    if isinstance(v, (int, float)):
        return str(v)
    return v


def _json_schema_to_pydantic(
    name: str, properties: dict, required: list[str]
) -> type[BaseModel]:
    """Dynamically create a Pydantic model from a JSON Schema ``properties`` block.

    This bridges our hand-crafted OpenAI-format schemas to LangChain's
    Pydantic-based tool signature system. A list of types, such as
    ``["array", "string"]``, becomes a union tried in the order listed.
    """
    field_defs: dict[str, Any] = {}
    type_map = {
        "string": Annotated[str, BeforeValidator(_coerce_scalar_str)],
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": Annotated[list, BeforeValidator(_coerce_json_str)],
        "object": Annotated[dict, BeforeValidator(_coerce_json_str)],
    }

    for prop_name, prop_def in properties.items():
        json_type = prop_def.get("type", "string")
        extra: dict[str, Any] = {}
        if isinstance(json_type, list):
            # "null" is dropped because optionality comes from `required` below.
            parts = [type_map.get(t, Any) for t in json_type if t != "null"]
            if not parts or Any in parts:
                py_type = Any
            elif len(parts) == 1:
                py_type = parts[0]
            else:
                py_type = Union[tuple(parts)]
                # '["CA"]' is a valid string too; the schema's order decides.
                extra["union_mode"] = "left_to_right"
        else:
            py_type = type_map.get(json_type, Any)
        description = prop_def.get("description", "")
        default_val = prop_def.get("default")
        is_required = prop_name in required

        if is_required:
            field_defs[prop_name] = (py_type, Field(description=description, **extra))
        elif default_val is not None:
            field_defs[prop_name] = (
                Optional[py_type],
                Field(default=default_val, description=description, **extra),
            )
        else:
            field_defs[prop_name] = (
                Optional[py_type],
                Field(default=None, description=description, **extra),
            )

    model = create_model(f"{name}_Args", **field_defs)
    return model


def _record_invocation_event(tool_name: str, event: Any) -> None:
    """Stash a tool call's canonical event for the streaming layer to pick up.

    A no-op when no context is bound (workflow subgraphs, tests, the training
    resource server), so nothing depends on the side channel existing.
    """
    if event is None:
        return
    events = _invocation_events_ctx.get()
    if events is None:
        return
    events.setdefault(tool_name, []).append(event)


def take_invocation_event(tool_name: str) -> Optional[Any]:
    """Pop the oldest unconsumed invocation event for ``tool_name``.

    Called by ``streaming.py`` when it emits ``tool_call_end``. Returns None
    when the tool ran outside a bound context or its event was already taken;
    callers must treat absence as "unknown", not as a failure.
    """
    events = _invocation_events_ctx.get()
    if not events:
        return None
    queued = events.get(tool_name)
    if not queued:
        return None
    return queued.pop(0)


def _make_tool_wrapper(tool_name: str, schema: dict) -> StructuredTool:
    """Create a LangChain StructuredTool that delegates to ToolRegistry.execute().

    The wrapper is async so it can call the async ``registry.execute()`` directly.
    """
    from api.tools.registry import get_registry

    func_schema = schema["function"]
    params = func_schema.get("parameters", {})
    properties = params.get("properties", {})
    required = params.get("required", [])

    args_model = _json_schema_to_pydantic(tool_name, properties, required)

    async def _execute(**kwargs: Any) -> str:
        from api.tools.base import ContentType, ToolResult
        from api.tools.viewer_state import add_visualization_to_state

        registry = get_registry()
        storage = _graph_storage_ctx.get()
        session_id = _graph_session_id_ctx.get()
        request_id = _graph_request_id_ctx.get()
        job_manager = _graph_job_manager_ctx.get()
        viz_queue = _viz_event_queue.get()

        # Strip None values — registry doesn't expect them
        clean_args = {k: v for k, v in kwargs.items() if v is not None}

        result = await registry.execute(
            tool_name,
            clean_args,
            storage=storage,
            session_id=session_id,
            request_id=request_id,
            job_manager=job_manager,
        )

        # Hand the structured event to the streaming layer before the ToolResult
        # goes out of scope at the `to_tool_response()` return below.
        _record_invocation_event(tool_name, getattr(result, "invocation_event", None))

        # Emit visualization side-effects that the frontend needs
        if viz_queue is not None:
            if result.content_type == ContentType.SDF_3D:
                label = result.metadata.get("label", "molecule")
                viz_queue.put_nowait({
                    "tool_event": {
                        "type": "visualize_3d",
                        "sdf": result.data,
                        "label": label,
                    }
                })
                await add_visualization_to_state(
                    request_id,
                    label=label,
                    structure_type="small_molecule",
                    format="sdf",
                    metadata=result.metadata,
                )

            elif result.content_type == ContentType.PDB_3D:
                label = result.metadata.get("label", "protein")
                # Capture the source pdb_ref (when known) so structure_query
                # can later validate that its target is loaded in the viewer.
                pdb_ref_input = clean_args.get("pdb_ref", "") if isinstance(clean_args, dict) else ""
                # Molstar needs to be told which parser to use; boltz2 produces
                # mmCIF, everything else here produces PDB.
                structure_format = result.metadata.get("format") or "pdb"
                pdb_event = {
                    "type": "visualize_pdb",
                    "pdb": result.data,
                    "label": label,
                    "format": structure_format,
                }
                if result.metadata.get("roles"):
                    pdb_event["roles"] = result.metadata["roles"]
                viz_queue.put_nowait({"tool_event": pdb_event})
                await add_visualization_to_state(
                    request_id,
                    label=label,
                    structure_type="protein",
                    format=structure_format,
                    metadata=result.metadata,
                    pdb_ref=pdb_ref_input,
                )

            elif result.content_type == ContentType.STRUCTURE_QUERY:
                from api.pending_ops import RedisPendingOps
                from api.redis import get_redis
                from api.tools.viewer_state import get_current_viewer_state

                # Validate that the requested pdb_ref is loaded in the viewer.
                # We only run this check when the result carries a pdb_ref
                # (added by the new structure_query schema) and the viewer
                # state has structures with known refs. This is a soft guard
                # that keeps prod aligned without requiring a Molstar change.
                requested_pdb_ref = (
                    result.data.get("pdb_ref", "")
                    if isinstance(result.data, dict)
                    else ""
                )
                if requested_pdb_ref:
                    viewer_state = await get_current_viewer_state(request_id)
                    loaded_refs = {
                        s.pdb_ref for s in viewer_state.structures if s.pdb_ref
                    }
                    if loaded_refs and requested_pdb_ref not in loaded_refs:
                        loaded_summary = ", ".join(sorted(loaded_refs)) or "none"
                        return ToolResult.error(
                            f"structure_query target pdb_ref='{requested_pdb_ref}' is "
                            f"not loaded in the viewer (loaded: {loaded_summary}). "
                            f"Call visualize_pdb(pdb_ref='{requested_pdb_ref}') first, "
                            "then re-run structure_query."
                        ).to_tool_response()

                import uuid as _uuid
                query_id = str(_uuid.uuid4())

                redis_client = get_redis()
                pending = RedisPendingOps(redis_client)
                await pending.register(query_id, "selection")

                event_payload = {
                    "tool_event": {
                        "type": "structure_query",
                        "query_id": query_id,
                        "session_id": session_id,
                        "query": result.data,
                    }
                }
                viz_queue.put_nowait(event_payload)
                logger.info(
                    f"[structure_query] Pushed viz event to queue "
                    f"(query_id={query_id}, queue_size={viz_queue.qsize()}, "
                    f"pdb_ref={requested_pdb_ref or '?'}, "
                    f"molql={result.data.get('molql', '?')[:80]})"
                )

                result.metadata["message"] = (
                    f"[PENDING:{query_id}] Selection query sent to viewer. Awaiting results."
                )

                deferred_span = getattr(result, "_deferred_span", None)
                pending_sq_list = _pending_structure_query_ids_ctx.get()
                if pending_sq_list is not None:
                    pending_sq_list.append((query_id, session_id, deferred_span))

                pending_list = _pending_ops_ctx.get()
                if pending_list is not None:
                    pending_list.append(pending)

            # python_execute visualization events stored in metadata
            viz_events = result.metadata.get("visualization_events", [])
            for viz in viz_events:
                viz_type = viz.get("type", "visualize_3d")
                if viz_type == "visualize_3d":
                    label = viz.get("label", "molecule")
                    viz_queue.put_nowait({
                        "tool_event": {
                            "type": "visualize_3d",
                            "sdf": viz.get("sdf", ""),
                            "label": label,
                        }
                    })
                    await add_visualization_to_state(
                        request_id,
                        label=label,
                        structure_type="small_molecule",
                        format="sdf",
                        metadata={"label": label},
                    )
                elif viz_type == "visualize_pdb":
                    label = viz.get("label", "molecule")
                    structure_format = viz.get("format") or "pdb"
                    viz_queue.put_nowait({
                        "tool_event": {
                            "type": "visualize_pdb",
                            "pdb": viz.get("pdb", ""),
                            "label": label,
                            "format": structure_format,
                        }
                    })
                    await add_visualization_to_state(
                        request_id,
                        label=label,
                        structure_type="complex",
                        format=structure_format,
                        metadata={"label": label},
                    )
                elif viz_type == "show_selection":
                    # Selects residues of what is already loaded; adds no
                    # structure, so there is no viewer state to record.
                    viz_queue.put_nowait({"tool_event": dict(viz)})

        return result.to_tool_response()

    return StructuredTool.from_function(
        coroutine=_execute,
        name=tool_name,
        description=func_schema.get("description", ""),
        args_schema=args_model,
        return_direct=False,
    )


async def build_langchain_tools(
    session_id: Optional[str] = None,
) -> list[StructuredTool]:
    """Build LangChain tool wrappers for all registered Aevom tools.

    Reads schemas from the existing ToolRegistry and creates one
    StructuredTool per registered tool, preserving identical schemas.

    Args:
        session_id: Optional session ID to include MCP tools.

    Returns:
        List of LangChain StructuredTool instances.
    """
    from api.tools.registry import get_registry

    registry = get_registry()
    schemas = await registry.get_all_schemas(session_id=session_id)

    tools: list[StructuredTool] = []
    for schema in schemas:
        func_schema = schema.get("function", {})
        name = func_schema.get("name")
        if not name:
            logger.warning(f"Skipping tool with no name: {schema}")
            continue

        try:
            wrapper = _make_tool_wrapper(name, schema)
            tools.append(wrapper)
            logger.debug(f"Created LangChain wrapper for tool: {name}")
        except Exception:
            logger.exception(f"Failed to create LangChain wrapper for tool: {name}")

    logger.info(
        f"Built {len(tools)} LangChain tool wrappers: "
        f"{[t.name for t in tools]}"
    )
    return tools


def set_tool_context(
    storage: Any,
    session_id: str,
    request_id: str,
    job_manager: Any = None,
) -> tuple[list, asyncio.Queue, asyncio.Queue, asyncio.Queue]:
    """Set per-request context for tool wrappers.

    Args:
        job_manager: Optional ``JobManager`` (ADR 0001 P0 job semantics).
            Defaults to None, which preserves today's behavior exactly.

    Returns (reset_tokens, viz_event_queue, retry_status_queue, workflow_event_queue).
    The caller should drain all queues during streaming.
    """
    viz_queue: asyncio.Queue = asyncio.Queue()
    retry_queue: asyncio.Queue = asyncio.Queue()
    wf_queue: asyncio.Queue = asyncio.Queue()
    pending_list: list = []
    pending_sq_list: list[tuple[str, str, Any]] = []
    invocation_events: dict[str, list] = {}
    tokens = [
        _graph_storage_ctx.set(storage),
        _graph_session_id_ctx.set(session_id),
        _graph_request_id_ctx.set(request_id),
        _graph_job_manager_ctx.set(job_manager),
        _invocation_events_ctx.set(invocation_events),
        _viz_event_queue.set(viz_queue),
        _retry_status_queue.set(retry_queue),
        _workflow_event_queue.set(wf_queue),
        _pending_ops_ctx.set(pending_list),
        _pending_structure_query_ids_ctx.set(pending_sq_list),
    ]
    return tokens, viz_queue, retry_queue, wf_queue


def push_retry_status(status: Optional[dict]) -> None:
    """Push a retry status event onto the queue. None = clear."""
    q = _retry_status_queue.get()
    if q is not None:
        q.put_nowait(status)


def get_workflow_event_queue() -> Optional[asyncio.Queue]:
    """Return the workflow event queue for the current request, if set."""
    return _workflow_event_queue.get()


async def drain_pending_ops(timeout: float = 10.0) -> list[str]:
    """Wait for all pending ops registered during the last tool step.

    Called by the ``await_pending_ops`` graph node after the streaming
    layer has delivered the SSE events to the frontend.

    Returns list of unacknowledged query_ids (empty on success).
    """
    pending_list = _pending_ops_ctx.get()
    if not pending_list:
        return []

    all_unacked: list[str] = []
    for pending in pending_list:
        unacked = await pending.await_all(timeout=timeout)
        all_unacked.extend(unacked)

    pending_list.clear()

    if all_unacked:
        logger.warning(f"Pending ops timed out: {all_unacked}")
    return all_unacked


def get_and_clear_pending_structure_queries() -> list[tuple[str, str, Any]]:
    """Return and clear the list of (query_id, session_id, deferred_span) tuples
    accumulated during the last tool step.

    Called by ``_await_pending_ops_node`` to know which ToolMessages to patch
    with rich selection data from Redis, and to finalise the deferred OTel spans.
    """
    pending_sq_list = _pending_structure_query_ids_ctx.get()
    if not pending_sq_list:
        return []
    result = list(pending_sq_list)
    pending_sq_list.clear()
    return result


def reset_tool_context(tokens: list) -> None:
    """Reset per-request context after graph execution.

    Also ends any deferred OTel spans that were never closed
    (e.g. if the graph errored before ``_await_pending_ops_node`` ran).
    """
    pending_sq_list = _pending_structure_query_ids_ctx.get()
    if pending_sq_list:
        for entry in pending_sq_list:
            deferred_span = entry[2] if len(entry) > 2 else None
            if deferred_span:
                try:
                    deferred_span.end()
                except Exception:
                    pass
        pending_sq_list.clear()

    for token in tokens:
        token.var.reset(token)
