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

"""Tool registry for discovering and routing tool calls.

The registry collects all available tools, provides their schemas
to the LLM, and routes execution to the appropriate handler.
"""

import json
import logging
import os
from typing import TYPE_CHECKING, Any, Optional

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from api.telemetry import (
    get_tracer,
    get_current_iteration_span,
    get_current_chat_span,
    set_io,
    set_last_tool_span_id,
    set_span_kind,
    should_skip_tool_spans,
)
from api.storage import InvalidReferenceError

from . import budget
from .base import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_RESPONSE_TOKENS,
    INVALID_ARGUMENTS,
    BaseTool,
    ContentType,
    DataIOCategory,
    NvidiaApiTool,
    ToolResult,
    ToolType,
    build_capped_response,
    _storage_ctx,
    _request_id_ctx,
    _session_id_ctx,
)

if TYPE_CHECKING:
    from api.storage import DataStorageService
    from .policy import PolicyEngine

logger = logging.getLogger(__name__)

# Tool-result metadata keys copied onto the tool's trace span. The first
# underscore becomes a dot: ``selection_provenance`` is ``tool.selection.provenance``.
_SPAN_METADATA_KEYS = (
    "selection_ref",
    "selection_provenance",
    "selection_atom_count",
    "selection_residue_count",
    "selection_parent",
    "design_spec_ref",
    "hotspot_selector_count",
)


def nim_tools_unavailable() -> bool:
    """Whether NIM-backed tools should be withheld for want of a key.

    True only in an Anthropic-only deployment: some LLM key is configured but
    the NVIDIA one is not. With no key at all (unit tests, a bare checkout)
    nothing is withheld, and a tool-cache replay needs no live credential.
    """
    from api.llm.config import NVIDIA_KEY_FILE, configured_key_files, key_file_present

    if _tool_cache_mode() in ("replay", "mixed"):
        return False
    nvidia_key_file = os.environ.get("NVIDIA_API_KEY_FILE", NVIDIA_KEY_FILE)
    return not key_file_present(nvidia_key_file) and bool(configured_key_files())

# Tools that read the ledger rather than adding to it. Recording a search as a
# finding makes it findable by the next search, so the ledger fills with echoes
# of itself and the session log fills with them too.
_NOT_FINDINGS = frozenset(
    {"search_session_history", "query_candidates", "best_candidate"}
)


def _names_from_env(variable: str) -> frozenset[str]:
    """Read a comma-separated tool name list from the environment."""
    return frozenset(
        name.strip() for name in os.getenv(variable, "").split(",") if name.strip()
    )


class ToolFilter:
    """Which tools a deployment is willing to expose.

    Both lists empty is production: every tool that registers stays. A harness
    comparing Aevom against another agent's fixed tool set, or one that must
    not be able to look up an anonymised target, narrows the surface here
    instead of unregistering tools by hand after startup.

    The allowlist is exclusive -- naming any tool withholds all the others --
    and the denylist is applied on top, so a denied name stays denied even if
    someone later widens the allowlist.
    """

    def __init__(
        self,
        allowlist: Optional[frozenset[str]] = None,
        denylist: Optional[frozenset[str]] = None,
    ):
        self.allowlist = (
            _names_from_env("AEVOM_TOOL_ALLOWLIST") if allowlist is None else frozenset(allowlist)
        )
        self.denylist = (
            _names_from_env("AEVOM_TOOL_DENYLIST") if denylist is None else frozenset(denylist)
        )

    @property
    def active(self) -> bool:
        return bool(self.allowlist or self.denylist)

    def permits(self, tool_name: str) -> bool:
        if tool_name in self.denylist:
            return False
        return not self.allowlist or tool_name in self.allowlist


class ToolRegistry:
    """Central registry for all available tools.

    The registry manages tool registration, schema collection,
    and execution routing.

    Example:
        registry = ToolRegistry()
        registry.register(WeatherTool())
        registry.register(RDKitTool())

        # Get schemas for LLM
        schemas = registry.get_all_schemas()

        # Execute a tool
        result = await registry.execute("get_weather", {"location": "Miami"})
    """

    def __init__(
        self,
        tool_filter: Optional[ToolFilter] = None,
        policy_engine: Optional["PolicyEngine"] = None,
    ):
        """Initialize empty registry.

        Args:
            tool_filter: Which tool names may be exposed. Defaults to the one
                the environment describes, which is "all of them" unless
                AEVOM_TOOL_ALLOWLIST or AEVOM_TOOL_DENYLIST says otherwise.
            policy_engine: Model/tool/run-level policy hook (ADR 0001). Defaults
                to the standard engine (env-configured approval list plus the
                http_request allowlist check), which changes nothing for a
                deployment that hasn't set AEVOM_REQUIRE_APPROVAL_FOR.
        """
        from .policy import PolicyEngine

        self._tools: dict[str, BaseTool] = {}
        self._filter = tool_filter or ToolFilter()
        self._policy_engine = policy_engine or PolicyEngine()
        self.withheld: list[str] = []
        # Withheld for want of a credential rather than by the filter; the
        # system prompt names these so the model does not plan around them.
        self.unavailable: list[str] = []

    def register(self, tool: BaseTool) -> None:
        """Register a tool with the registry.

        A tool the filter does not permit is dropped here rather than at the
        call sites, so the schemas the model sees, the LangChain tools the
        graph binds, and the execution path cannot disagree about which tools
        exist.

        Args:
            tool: Tool instance to register.

        Raises:
            ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        if not self._filter.permits(tool.name):
            self.withheld.append(tool.name)
            logger.info(f"Withheld tool: {tool.name} (AEVOM_TOOL_ALLOWLIST/AEVOM_TOOL_DENYLIST)")
            return
        if isinstance(tool, NvidiaApiTool) and nim_tools_unavailable():
            self.withheld.append(tool.name)
            self.unavailable.append(tool.name)
            logger.info(f"Withheld tool: {tool.name} (no NVIDIA API key)")
            return
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} (type: {tool.tool_type.value})")

    def unregister(self, tool_name: str) -> None:
        """Remove a tool from the registry.

        Args:
            tool_name: Name of the tool to remove.
        """
        self._tools.pop(tool_name, None)

    def get(self, tool_name: str) -> Optional[BaseTool]:
        """Get a tool by name.

        Args:
            tool_name: Name of the tool.

        Returns:
            Tool instance or None if not found.
        """
        return self._tools.get(tool_name)

    async def get_all_schemas(
        self,
        session_id: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """Get OpenAI-compatible schemas for all registered tools.

        Args:
            session_id: Optional session ID to include MCP tools for that session.

        Returns:
            List of tool schemas for LLM API calls.
        """
        # Get builtin tool schemas
        schemas = [tool.get_schema() for tool in self._tools.values()]

        # Add MCP tool schemas if session_id provided
        if session_id:
            from api.mcp import get_mcp_session_store

            mcp_store = get_mcp_session_store()
            mcp_schemas = await mcp_store.get_tool_schemas(session_id)
            schemas.extend(self._permitted(mcp_schemas))

            # Tools contributed by a self-hosted NIM the user registered.
            # Like MCP's, these are withheld until the NIM is confirmed, and
            # a NIM that is unreachable must cost the model only its own
            # tools rather than the whole schema list.
            from api.nim import get_nim_session_store

            try:
                nim_schemas = await get_nim_session_store().get_tool_schemas(
                    session_id
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to build NIM tool schemas: %s", exc)
            else:
                schemas.extend(self._permitted(nim_schemas))

        # Add tools synthesized from skill data-type contracts. Builtin skills
        # contribute these too, so this is not gated on session_id.
        from api.skills import get_skill_session_store

        try:
            skill_schemas = await get_skill_session_store().get_tool_schemas(session_id)
        except Exception as exc:  # noqa: BLE001
            # A malformed contract must not cost the model its whole toolset.
            logger.warning("Failed to build skill tool schemas: %s", exc)
        else:
            schemas.extend(self._permitted(skill_schemas))

        return schemas

    def _permitted(self, schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Drop schemas for tools the filter withholds.

        MCP tools and skill-contract tools are resolved per session rather than
        registered, so they do not pass through ``register``. Without this they
        would be the one way past an allowlist.
        """
        if not self._filter.active:
            return schemas
        return [
            schema
            for schema in schemas
            if self._filter.permits((schema.get("function") or {}).get("name", ""))
        ]

    def get_tools_by_type(self, tool_type: ToolType) -> list[BaseTool]:
        """Get all tools of a specific type.

        Args:
            tool_type: Type of tools to retrieve.

        Returns:
            List of matching tools.
        """
        return [tool for tool in self._tools.values() if tool.tool_type == tool_type]

    def get_tools_by_data_io_category(self, category: DataIOCategory) -> list[BaseTool]:
        """Get all tools with a specific data I/O category.

        Args:
            category: Data I/O category to filter by.

        Returns:
            List of matching tools.
        """
        return [
            tool for tool in self._tools.values() if tool.data_io_category == category
        ]

    def get_tools_requiring_storage(self) -> list[BaseTool]:
        """Get all tools that require storage for input or output.

        Returns:
            List of tools with data_io_category != TEXT_ONLY.
        """
        return [
            tool
            for tool in self._tools.values()
            if tool.data_io_category != DataIOCategory.TEXT_ONLY
        ]

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        storage: Optional["DataStorageService"] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        job_manager: Optional[Any] = None,
    ) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            storage: Optional storage service for tools that need data storage.
                     Required for tools with data_io_category != TEXT_ONLY.
            session_id: Optional session ID for MCP tool resolution.
            request_id: Optional request ID for Redis-keyed per-request state.
            job_manager: Optional ``JobManager`` (``api.jobs.manager``). When
                given and the resolved tool opted into job tracking
                (``tool.is_async == True``), execution is routed through
                ``JobManager.execute_tool_with_tracking`` instead of calling
                ``tool.execute()`` directly, so the call is recorded as a
                durable ``Job`` row (see ADR 0001). Defaults to None, which
                preserves today's behavior exactly -- this is opt-in per
                caller, not a change to any existing call site.

        Returns:
            ToolResult from tool execution.
        """
        logger.info(f"Executing tool: {tool_name} with arguments: {arguments}")

        if should_skip_tool_spans():
            return await self._execute_impl(
                tool_name,
                arguments,
                storage,
                session_id,
                request_id,
                span=None,
                job_manager=job_manager,
            )

        tracer = get_tracer()
        parent = get_current_iteration_span() or get_current_chat_span()
        context = trace.set_span_in_context(parent) if parent else None

        span = tracer.start_span(f"tool.{tool_name}", context=context)
        # Record the live span_id in a ContextVar so the training resource
        # server (training/nemo_gym_wrapper/app.py:execute_tool) can recover
        # it after this method returns and target per-tool reward annotations
        # at this exact span. A no-op tracer / invalid context yields a
        # zero span_id, which we treat as "unknown" (None) so downstream
        # lookups fall back cleanly to the root chat_request.
        try:
            sc = span.get_span_context()
            if sc is not None and sc.is_valid and sc.span_id != 0:
                set_last_tool_span_id(format(sc.span_id, "016x"))
            else:
                set_last_tool_span_id(None)
        except Exception:
            set_last_tool_span_id(None)
        try:
            with trace.use_span(span, end_on_exit=False):
                result = await self._execute_impl(
                    tool_name,
                    arguments,
                    storage,
                    session_id,
                    request_id,
                    span=span,
                    job_manager=job_manager,
                )
            if result.content_type == ContentType.STRUCTURE_QUERY:
                result._deferred_span = span  # type: ignore[attr-defined]
            else:
                span.end()
            return result
        except Exception:
            span.end()
            raise

    async def _apply_response_budget(
        self,
        tool_name: str,
        tool: BaseTool,
        result: ToolResult,
        raw_text: str,
        storage: Optional["DataStorageService"],
    ) -> Optional[dict[str, Any]]:
        """Cap an over-budget tool response, storing the overflow as an artifact.

        This is the one place tool output is bounded before it reaches the
        model. It runs in the registry rather than in ``to_tool_response()``
        because storing the overflow is async and needs the request's storage
        service -- and because the registry is the single path shared by the
        production graph and the NeMo-Gym resource server, so production and
        training see the same observation distribution.

        Args:
            tool_name: Name of the executed tool, for logging.
            tool: The tool instance, consulted for ``max_response_tokens``.
            result: Mutated in place via ``response_override`` when over budget.
            raw_text: The full rendered response.
            storage: Request storage, or None when unavailable.

        Returns:
            A dict describing the capping, or None when the response fit.
        """
        max_tokens = getattr(tool, "max_response_tokens", DEFAULT_MAX_RESPONSE_TOKENS)
        max_chars = max(1, int(int(max_tokens) * CHARS_PER_TOKEN))
        if len(raw_text) <= max_chars:
            return None

        overflow_ref: Optional[str] = None
        if storage is not None:
            try:
                overflow_ref = await storage.store(
                    "text",
                    raw_text.encode("utf-8"),
                    metadata={"tool": tool_name, "reason": "response_budget"},
                    description=f"Full {tool_name} output ({len(raw_text)} chars)",
                )
            except Exception:
                # Never fail a tool call because the overflow could not be
                # persisted; the capped response degrades to "not recoverable".
                logger.warning(
                    f"Failed to store overflow for {tool_name}", exc_info=True
                )

        result.response_override = build_capped_response(
            raw_text, max_chars, overflow_ref
        )
        logger.info(
            f"Tool {tool_name} response capped: {len(raw_text)} -> "
            f"{len(result.response_override)} chars "
            f"(budget {max_tokens} tokens, overflow_ref={overflow_ref})"
        )
        return {
            "raw_chars": len(raw_text),
            "kept_chars": len(result.response_override),
            "overflow_ref": overflow_ref,
        }

    async def _record_finding(
        self, tool_name: str, arguments: dict[str, Any], result: ToolResult
    ) -> None:
        """File what this call established in the session's findings ledger.

        Here rather than in the graph because this is where the ``ToolResult``
        still exists. By the time a result reaches the streaming layer it has
        been rendered to a string, and the metadata carrying the affinity or
        the pLDDT -- the whole point of the ledger -- is gone.

        A no-op when no recorder is bound, which is the case for the training
        resource server and for tests.
        """
        from api.conversations import get_recorder

        if tool_name in _NOT_FINDINGS:
            return

        recorder = get_recorder()
        if recorder is None:
            return
        # Structure queries return a [PENDING:] marker that only resolves once
        # the frontend answers, so there is no fact to record yet.
        if result.content_type == ContentType.STRUCTURE_QUERY:
            return
        await recorder.record_finding(tool_name, arguments, result)
        # Molecule-level results also go to the candidate ledger, which is keyed
        # by molecule rather than by call. Same result object, different grain.
        await recorder.capture_candidates(tool_name, result)

    def _attach_invocation_event(
        self,
        result: ToolResult,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: Optional[str],
        request_id: Optional[str],
        job_id: Optional[str] = None,
        job_status: Optional[str] = None,
        policy_decision: Optional[str] = None,
        policy_reason: Optional[str] = None,
        budget_state: Optional[dict[str, Any]] = None,
        raw_output_ref: Optional[str] = None,
    ) -> ToolResult:
        """Attach the canonical ``ToolInvocationEvent`` to a finished result.

        Called from *every* terminal branch of :meth:`_execute_impl`, including
        the failure ones. A contract that covers only the happy path is not a
        contract: a consumer (the SSE layer, the eval trajectory, the training
        wrapper) cannot distinguish "this call produced no event" from "this
        branch forgot to emit one", so an absent event would have to be read as
        "unknown" everywhere and the field would carry no information.

        ``policy_decision`` is None only where policy genuinely never ran -- an
        unknown tool name is rejected before the engine is consulted.
        """
        from api.telemetry_contract import build_invocation_event

        result.invocation_event = build_invocation_event(
            tool_name=tool_name,
            arguments=arguments,
            session_id=session_id,
            request_id=request_id,
            result=result,
            job_id=job_id,
            job_status=job_status,
            policy_decision=policy_decision,
            policy_reason=policy_reason,
            budget_state=budget_state,
            raw_output_ref=raw_output_ref,
        )
        return result

    @staticmethod
    def _budget_state(verdict: Any) -> Optional[dict[str, Any]]:
        """Render a budget verdict into the event's ``budget_state`` shape."""
        if verdict is None or verdict.state is None:
            return None
        return {
            "limit": verdict.state.limit,
            "used": verdict.state.used,
            "remaining": verdict.state.remaining,
        }

    async def _execute_impl(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        storage: Optional["DataStorageService"],
        session_id: Optional[str],
        request_id: Optional[str],
        span: Optional[trace.Span],
        job_manager: Optional[Any] = None,
    ) -> ToolResult:
        """Core execution logic shared by production (span!=None) and training (span=None)."""
        if span:
            span.set_attribute("tool.name", tool_name)
            # OpenInference: render as a TOOL span with structured I/O in
            # Phoenix. Additive -- tool.name / tool.arg.* below are preserved.
            set_span_kind(span, "TOOL")
            try:
                span.set_attribute("tool.parameters", json.dumps(arguments, default=str))
            except Exception:
                pass
            set_io(span, input_value=arguments)
            # Belt-and-braces: tag every tool span with the session/request
            # so that orphan tool spans (e.g. when traceparent propagation
            # fails between simple_agent and the resource server) can still
            # be re-attributed to a chat_request post-hoc by joining on
            # session_id. v0527a-3n probe surfaced exactly this orphan
            # mode; see internal_docs/docs/training_scaling_curve.md.
            if session_id:
                span.set_attribute("tool.session_id", session_id)
            if request_id:
                span.set_attribute("tool.request_id", request_id)
            for key, value in arguments.items():
                str_value = str(value)
                if len(str_value) > 500:
                    str_value = str_value[:500] + "...[truncated]"
                span.set_attribute(f"tool.arg.{key}", str_value)

        # First check builtin tools
        tool = self.get(tool_name)

        # If not found and session_id provided, check MCP tools
        if tool is None and session_id:
            from api.mcp import get_mcp_session_store

            mcp_store = get_mcp_session_store()
            tool = await mcp_store.get_tool_by_name(session_id, tool_name)
            if tool:
                if span:
                    span.set_attribute("tool.source", "mcp")
                logger.info(f"Found MCP tool: {tool_name}")

        # Then tools contributed by a registered NIM.
        if tool is None and session_id:
            from api.nim import get_nim_session_store

            tool = await get_nim_session_store().get_tool_by_name(
                session_id, tool_name
            )
            if tool:
                if span:
                    span.set_attribute("tool.source", "nim")
                logger.info(f"Found NIM tool: {tool_name}")

        # Then tools synthesized from a skill contract.
        if tool is None:
            from api.skills import get_skill_session_store

            tool = await get_skill_session_store().get_tool_by_name(
                session_id, tool_name
            )
            if tool:
                if span:
                    span.set_attribute("tool.source", "skill")
                    span.set_attribute("tool.skill_id", tool.skill_id)
                logger.info(f"Found synthesized skill tool: {tool_name}")

        if tool is None:
            logger.error(
                f"Unknown tool requested: {tool_name}. Available tools: {self.tool_names}"
            )
            if span:
                span.set_attribute("tool.error", f"Unknown tool: {tool_name}")
                span.set_status(Status(StatusCode.ERROR, f"Unknown tool: {tool_name}"))
            # policy_decision stays None: the engine is consulted below, after
            # the tool resolves, so for an unknown name no decision was made.
            return self._attach_invocation_event(
                ToolResult.error(f"Unknown tool: {tool_name}"),
                tool_name=tool_name,
                arguments=arguments,
                session_id=session_id,
                request_id=request_id,
            )

        if span:
            span.set_attribute("tool.type", tool.tool_type.value)
            span.set_attribute("tool.data_io_category", tool.data_io_category.value)

            # Record adaptation info for adapted MCP tools
            if hasattr(tool, "adapted") and tool.adapted:
                span.set_attribute("tool.adapted", True)
                if hasattr(tool, "adaptation_info"):
                    info = tool.adaptation_info
                    input_types = [
                        v["detected_type"]
                        for v in info.get("input_mappings", {}).values()
                    ]
                    output_types = [
                        v["detected_type"] for v in info.get("output_mappings", [])
                    ]
                    if input_types:
                        span.set_attribute(
                            "tool.adapted.input_types", ",".join(input_types)
                        )
                    if output_types:
                        span.set_attribute(
                            "tool.adapted.output_types", ",".join(output_types)
                        )

        # Policy runs before budget spend: a denied call should not cost the
        # session a budgeted slot (ADR 0001). Default engine is a no-op for a
        # deployment that hasn't configured AEVOM_REQUIRE_APPROVAL_FOR.
        policy_result = await self._policy_engine.evaluate(
            tool_name, arguments, session_id, request_id
        )
        if span:
            span.set_attribute("tool.policy.decision", "allow" if policy_result.allowed else "deny")
            if policy_result.reason:
                span.set_attribute("tool.policy.reason", policy_result.reason)
        if not policy_result.allowed:
            logger.info("Refused %s: policy denied (%s)", tool_name, policy_result.reason)
            if span:
                span.set_status(Status(StatusCode.ERROR, "policy denied"))
            result = self._attach_invocation_event(
                ToolResult.error(
                    policy_result.reason or f"'{tool_name}' was denied by policy."
                ),
                tool_name=tool_name,
                arguments=arguments,
                session_id=session_id,
                request_id=request_id,
                policy_decision="deny",
                policy_reason=policy_result.reason,
            )
            await self._record_finding(tool_name, arguments, result)
            return result

        # Budgeted tools are charged before they run, so a call that fails inside
        # the expensive service still counts against the budget. The alternative
        # -- refunding failures -- would let a persistently failing endpoint be
        # called without limit, which is exactly the spend a budget exists to
        # bound. The exceptions are below: InvalidReferenceError and
        # InvalidArgumentsError both mean the call never left this process, so
        # nothing was spent.
        verdict = await budget.spend(session_id, tool_name)
        if not verdict.allowed and verdict.state is not None:
            logger.info(
                "Refused %s: call budget exhausted for session %s",
                tool_name,
                session_id,
            )
            if span:
                span.set_attribute("tool.budget.refused", True)
                span.set_attribute("tool.budget.limit", verdict.state.limit)
                span.set_status(Status(StatusCode.ERROR, "call budget exhausted"))
            # policy_decision is "allow": policy permitted this call and the
            # budget is what refused it. Collapsing the two would make a spend
            # limit look like a safety refusal in the trajectory.
            result = self._attach_invocation_event(
                ToolResult.error(budget.exhausted_message(verdict.state)),
                tool_name=tool_name,
                arguments=arguments,
                session_id=session_id,
                request_id=request_id,
                policy_decision="allow",
                budget_state=self._budget_state(verdict),
            )
            await self._record_finding(tool_name, arguments, result)
            return result

        # Inject per-request context via ContextVars (async-safe).
        # Token-based set/reset ensures clean context after execution,
        # even when multiple tools run sequentially in the same task.
        tokens = []
        if request_id:
            tokens.append(_request_id_ctx.set(request_id))
        if session_id:
            tokens.append(_session_id_ctx.set(session_id))

        if tool.data_io_category != DataIOCategory.TEXT_ONLY:
            if storage is None:
                logger.warning(
                    f"Tool {tool_name} requires storage (category: {tool.data_io_category.value}) "
                    f"but no storage service was provided"
                )
            else:
                tokens.append(_storage_ctx.set(storage))
                logger.debug(f"Injected storage service into tool {tool_name}")

        try:
            job_id: Optional[str] = None
            job_status: Optional[str] = None
            if job_manager is not None and tool.is_async:
                # ADR 0001: route job-tracking-opted-in tools through
                # JobManager so the call is recorded as a durable Job row.
                # Default (job_manager=None) path below is unchanged.
                result, job_id, job_status = await job_manager.execute_tool_with_tracking(
                    tool, arguments, session_id=session_id, request_id=request_id
                )
            else:
                result = await tool.execute(arguments)
            logger.info(
                f"Tool {tool_name} completed with content_type: {result.content_type.value}"
            )
            if (result.metadata or {}).get(INVALID_ARGUMENTS):
                # The tool refused its arguments without doing its work, so the
                # resource the budget protects was not touched. Refunding here
                # rather than for every failure keeps the guard that matters: a
                # call that reached a failing service still costs one.
                await budget.refund(session_id, tool_name)
                # None when the tool has no budget at all, which is every
                # tool under a profile that runs without them. _budget_state
                # already guards for it; the two readers below did not, so a
                # tool refusing its own arguments crashed on the refund path
                # instead of returning the guidance it had written.
                verdict = await budget.current_state(session_id, tool_name)
                if span:
                    span.set_attribute("tool.budget.refunded", True)
            if span and job_id:
                span.set_attribute("tool.job_id", job_id)
            if span and job_status:
                span.set_attribute("tool.job_status", job_status)

            # STRUCTURE_QUERY output is determined asynchronously: the wrapper
            # embeds a [PENDING:] placeholder which _await_pending_ops_node
            # later patches with rich content. Recording the pre-patch value
            # here would be misleading, and capping it would corrupt the marker.
            raw_text: Optional[str] = None
            response_budget: Optional[dict[str, Any]] = None
            if result.content_type != ContentType.STRUCTURE_QUERY:
                raw_text = result.to_tool_response()
                response_budget = await self._apply_response_budget(
                    tool_name, tool, result, raw_text, storage
                )
                # After capping, so the note survives truncation: knowing one
                # call is left is worth more than the line it displaces.
                if verdict is not None and verdict.state is not None:
                    result.response_override = (
                        f"{result.to_tool_response()}\n\n"
                        f"{budget.remaining_note(verdict.state)}"
                    )

            if span:
                if raw_text is not None:
                    # tool.raw.output stays the untruncated source of truth for
                    # training and profiling; set_io carries what the model
                    # actually received, which may have been capped.
                    span.set_attribute("tool.raw.output", raw_text)
                    set_io(span, output_value=result.to_tool_response())
                if response_budget:
                    span.set_attribute("tool.response.capped", True)
                    span.set_attribute(
                        "tool.response.raw_chars", response_budget["raw_chars"]
                    )
                    span.set_attribute(
                        "tool.response.kept_chars", response_budget["kept_chars"]
                    )
                    if response_budget.get("overflow_ref"):
                        span.set_attribute(
                            "tool.response.overflow_ref",
                            response_budget["overflow_ref"],
                        )
                if verdict is not None and verdict.state is not None:
                    span.set_attribute("tool.budget.limit", verdict.state.limit)
                    span.set_attribute("tool.budget.used", verdict.state.used)
                    span.set_attribute(
                        "tool.budget.remaining", verdict.state.remaining
                    )
                span.set_attribute("tool.raw.arguments", json.dumps(arguments))

                # Which selection or spec this call produced or consumed, who
                # made it and what it came from, for tracing a design back to
                # the residues a person (or the agent) chose.
                for key in _SPAN_METADATA_KEYS:
                    value = (result.metadata or {}).get(key)
                    if value is not None:
                        span.set_attribute(f"tool.{key.replace('_', '.', 1)}", value)

                # Record result metadata in span
                span.set_attribute(
                    "tool.result.content_type", result.content_type.value
                )
                span.set_attribute(
                    "tool.result.success", result.content_type.value != "error"
                )

                # For errors, also record as error attribute for visibility
                if result.content_type.value == "error":
                    error_text = raw_text or str(result.data)
                    span.set_attribute("tool.error", error_text)
                    span.set_status(Status(StatusCode.ERROR, error_text[:200]))
                else:
                    span.set_status(Status(StatusCode.OK))

            # Canonical invocation event (ADR 0001): the same shape for
            # production, training, and eval, since all three call this
            # method. Additive -- attached alongside the span attributes
            # above, not instead of them.
            result.job_id = result.job_id or job_id
            self._attach_invocation_event(
                result,
                tool_name=tool_name,
                arguments=arguments,
                session_id=session_id,
                request_id=request_id,
                job_id=job_id,
                job_status=job_status,
                policy_decision="allow",
                budget_state=self._budget_state(verdict),
                raw_output_ref=(response_budget or {}).get("overflow_ref"),
            )

            await self._record_finding(tool_name, arguments, result)
            return result
        except InvalidReferenceError as e:
            msg = (
                f"{e} Use the appropriate tool to create a valid reference first "
                "(e.g., fetch_and_clean_pdb, fetch_protein_sequence, "
                "download_pubchem_sdf, generate_3d_coordinates)."
            )
            logger.warning(f"Tool {tool_name}: invalid reference - {e}")
            # The arguments never resolved to data, so the expensive work the
            # budget protects did not happen. Charging for it would make a
            # mistyped reference cost a docking run.
            await budget.refund(session_id, tool_name)
            if span:
                span.set_attribute("tool.error", msg)
                span.set_status(Status(StatusCode.ERROR, msg[:200]))
            # Re-read rather than reuse the spend verdict: that verdict holds
            # the count spend incremented, so after a refund it would report one
            # call fewer than the session actually has left.
            result = self._attach_invocation_event(
                ToolResult.error(msg),
                tool_name=tool_name,
                arguments=arguments,
                session_id=session_id,
                request_id=request_id,
                policy_decision="allow",
                budget_state=self._budget_state(
                    await budget.current_state(session_id, tool_name)
                ),
            )
            await self._record_finding(tool_name, arguments, result)
            return result
        except Exception as e:
            logger.exception(f"Tool {tool_name} execution failed")
            if span:
                span.set_attribute("tool.error", str(e))
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            result = self._attach_invocation_event(
                ToolResult.error(f"Tool execution failed: {str(e)}"),
                tool_name=tool_name,
                arguments=arguments,
                session_id=session_id,
                request_id=request_id,
                policy_decision="allow",
                budget_state=self._budget_state(verdict),
            )
            await self._record_finding(tool_name, arguments, result)
            return result
        finally:
            for token in tokens:
                token.var.reset(token)

    @property
    def tool_names(self) -> list[str]:
        """Get list of all registered tool names."""
        return list(self._tools.keys())

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._tools


# Global registry instance
_registry: Optional[ToolRegistry] = None

# Cache modes accepted by AEVOM_TOOL_CACHE_MODE. "off" is production.
_TOOL_CACHE_MODES = ("off", "record", "replay", "mixed")


def _tool_cache_mode() -> str:
    """Read AEVOM_TOOL_CACHE_MODE, falling back to production behaviour.

    An unrecognized value is a misconfiguration that would otherwise silently
    hand the eval suite live tools, so it fails loudly instead.
    """
    mode = os.getenv("AEVOM_TOOL_CACHE_MODE", "off").strip().lower() or "off"
    if mode not in _TOOL_CACHE_MODES:
        raise ValueError(
            f"AEVOM_TOOL_CACHE_MODE={mode!r} is not one of {_TOOL_CACHE_MODES}"
        )
    return mode


def get_registry() -> ToolRegistry:
    """Get the global tool registry instance.

    Creates the registry on first call and registers default tools.

    When ``AEVOM_TOOL_CACHE_MODE`` is set to something other than ``off``, the
    registry is wrapped in a ``CachingRegistry`` so the eval suite can replay
    recorded tool responses instead of calling live external APIs. The wrapper
    is a ``ToolRegistry`` subclass that delegates schema listing and execution,
    so the code path above it is unchanged. Default ``off`` means production
    behaves exactly as it did before this knob existed.

    Returns:
        Global ToolRegistry instance.
    """
    global _registry
    if _registry is None:
        logger.info("Initializing tool registry...")
        registry: ToolRegistry = ToolRegistry()
        _register_default_tools(registry)
        logger.info(
            f"Tool registry initialized with {len(registry)} tools: {registry.tool_names}"
        )
        if registry.withheld:
            # The exposed set is what a benchmark run has to report, and this
            # line is where an external harness reads it back.
            logger.warning(
                f"Tool filter withheld {len(registry.withheld)} tools: {registry.withheld}"
            )

        mode = _tool_cache_mode()
        if mode != "off":
            # Imported lazily: this module is the base registry's home, and
            # mock_registry imports from it.
            from api.tools.mock_registry import CachingRegistry

            cache_dir = os.getenv("AEVOM_TOOL_CACHE_DIR", ".tool_cache")
            logger.warning(
                f"Tool cache active: mode={mode} dir={cache_dir}. "
                "Tool responses are not live."
            )
            registry = CachingRegistry(
                base_registry=registry, cache_dir=cache_dir, mode=mode
            )

        _registry = registry
    return _registry


def _register_default_tools(registry: ToolRegistry) -> None:
    """Register default tools with the registry.

    This function is called once when the registry is first accessed.
    Add new tools here to make them available to the system.
    Each tool is registered independently so one failure doesn't block others.
    """

    # Register RDKit tools (container-based, may not be available in dev)
    try:
        from .rdkit import RDKitMoleculeInfoTool

        registry.register(RDKitMoleculeInfoTool())
    except Exception as e:
        logger.warning(
            f"Failed to register RDKit tools (this is OK if RDKit service is not running): {e}"
        )

    # Register drug-likeness gates (RDKit container hosts the descriptor panel)
    try:
        from .molecule_gates import MoleculeGatesTool

        registry.register(MoleculeGatesTool())
    except Exception as e:
        logger.warning(f"Failed to register molecule gates tool: {e}")

    # Register SAFE notation tools (RDKit container hosts safe-mol)
    try:
        from .safe_notation import SafeToSmilesTool, SmilesToSafeTool

        registry.register(SmilesToSafeTool())
        registry.register(SafeToSmilesTool())
    except Exception as e:
        logger.warning(
            f"Failed to register SAFE notation tools (this is OK if RDKit service is not running): {e}"
        )

    # Register PDB visualization tool (local)
    # Note: Generate3DCoordinatesTool and Visualize3DTool are intentionally NOT registered.
    # SDF visualization is handled by VisualizeSdfTool (registered below from pubchem.py).
    try:
        from .visualization import VisualizePdbTool

        registry.register(VisualizePdbTool())
    except Exception as e:
        logger.warning(f"Failed to register PDB visualization tool: {e}")

    # Register UniProt tools (local, external API)
    try:
        from .uniprot import UniProtFetchSequenceTool

        registry.register(UniProtFetchSequenceTool())
    except Exception as e:
        logger.warning(f"Failed to register UniProt tools: {e}")

    # Register OpenFold3 structure prediction tool (NVIDIA API)
    try:
        from .openfold import OpenFold3Tool

        registry.register(OpenFold3Tool())
    except Exception as e:
        logger.warning(f"Failed to register OpenFold3 tools: {e}")

    # Register MSA Search tool (NVIDIA ColabFold API)
    try:
        from .msa_search import MSASearchTool

        registry.register(MSASearchTool())
    except Exception as e:
        logger.warning(f"Failed to register MSA Search tool: {e}")

    # Register Boltz-2 (NVIDIA MIT NIM): joint structure and affinity prediction.
    # Its usage guidance lives in the bound boltz2-nim skill; see
    # api.skills.binding.
    try:
        from .boltz2 import Boltz2Tool

        registry.register(Boltz2Tool())
    except Exception as e:
        logger.warning(f"Failed to register Boltz-2 tool: {e}")

    # Register structure format conversion. Registered alongside Boltz-2 because
    # its only reason to exist is bridging Boltz-2's mmCIF output into the
    # PDB-only tools.
    try:
        from .structure_convert import ConvertStructureTool

        registry.register(ConvertStructureTool())
    except Exception as e:
        logger.warning(f"Failed to register structure conversion tool: {e}")

    # Register the protein design NIMs (NVIDIA IPD APIs). These two chain:
    # RFDiffusion generates a backbone, ProteinMPNN puts a sequence on it. Their
    # usage guidance lives in the bound skills; see api.skills.binding.
    try:
        from .rfdiffusion import RFDiffusionTool

        registry.register(RFDiffusionTool())
    except Exception as e:
        logger.warning(f"Failed to register RFDiffusion tool: {e}")

    try:
        from .proteinmpnn import ProteinMPNNTool

        registry.register(ProteinMPNNTool())
    except Exception as e:
        logger.warning(f"Failed to register ProteinMPNN tool: {e}")

    # Register GenMol molecule generation (NVIDIA API). Completes the SAFE chain
    # that smiles_to_safe / safe_to_smiles were written for.
    try:
        from .genmol import GenMolTool

        registry.register(GenMolTool())
    except Exception as e:
        logger.warning(f"Failed to register GenMol tool: {e}")

    # Register Molstar structure query tool
    try:
        from .molstar_queries import MolstarQueryTool

        registry.register(MolstarQueryTool())
    except Exception as e:
        logger.warning(f"Failed to register Molstar query tools: {e}")

    # Register residue selection tools (make, combine and show selections)
    try:
        from .selection_tools import (
            CombineSelectionsTool,
            SelectResiduesTool,
            ShowSelectionTool,
        )

        registry.register(SelectResiduesTool())
        registry.register(CombineSelectionsTool())
        registry.register(ShowSelectionTool())
    except Exception as e:
        logger.warning(f"Failed to register selection tools: {e}")

    # Register design spec tools (bind selections to design conditions)
    try:
        from .design_spec_tools import CreateDesignSpecTool, ValidateDesignSpecTool

        registry.register(CreateDesignSpecTool())
        registry.register(ValidateDesignSpecTool())
    except Exception as e:
        logger.warning(f"Failed to register design spec tools: {e}")

    # Register viewer state tool
    try:
        from .viewer_state import GetViewerStateTool

        registry.register(GetViewerStateTool())
    except Exception as e:
        logger.warning(f"Failed to register viewer state tool: {e}")

    # Register session history search (findings ledger + artifact store)
    try:
        from .session_history import SearchSessionHistoryTool

        registry.register(SearchSessionHistoryTool())
    except Exception as e:
        logger.warning(f"Failed to register session history tool: {e}")

    # Register the candidate ledger tools (no service dependency)
    try:
        from .candidates import (
            BestCandidateTool,
            QueryCandidatesTool,
            RecordCandidatesTool,
        )

        registry.register(RecordCandidatesTool())
        registry.register(QueryCandidatesTool())
        registry.register(BestCandidateTool())
    except Exception as e:
        logger.warning(f"Failed to register candidate ledger tools: {e}")

    # Register Python execution tool (requires pyexec container service)
    try:
        from .python_exec import PythonExecuteTool

        registry.register(PythonExecuteTool())
    except Exception as e:
        logger.warning(f"Failed to register Python execution tool: {e}")

    # Register ADMET prediction tool (requires admet container service)
    try:
        from .admet import AdmetPredictTool

        registry.register(AdmetPredictTool())
    except Exception as e:
        logger.warning(f"Failed to register ADMET prediction tool: {e}")

    # Register interaction profiling tool (requires plip container service)
    try:
        from .plip import PlipInteractionsTool

        registry.register(PlipInteractionsTool())
    except Exception as e:
        logger.warning(f"Failed to register PLIP interactions tool: {e}")

    # Register Vina docking tool (requires vina container service)
    try:
        from .vina import VinaDockTool

        registry.register(VinaDockTool())
    except Exception as e:
        logger.warning(f"Failed to register Vina docking tool: {e}")

    # Register probe interaction-energy grids (same vina container service)
    try:
        from .probe_grid import ProbeGridTool

        registry.register(ProbeGridTool())
    except Exception as e:
        logger.warning(f"Failed to register probe grid tool: {e}")

    # Register pose-preserving fragment assembly (rdkit container service)
    try:
        from .fragment_exit_vectors import FragmentExitVectorsTool
        from .fragment_link import FragmentLinkTool

        registry.register(FragmentExitVectorsTool())
        registry.register(FragmentLinkTool())
    except Exception as e:
        logger.warning(f"Failed to register fragment linking tools: {e}")

    # Register RCSB PDB tools for structure retrieval
    try:
        from .rcsb import SearchRcsbTool, FetchAndCleanPdbTool

        registry.register(SearchRcsbTool())
        registry.register(FetchAndCleanPdbTool())
    except Exception as e:
        logger.warning(f"Failed to register RCSB PDB tools: {e}")

    # Register PubChem / CCD / SDF visualization tools (local, external API)
    try:
        from .pubchem import (
            SearchPubChemTool,
            GetPubChemPropertiesTool,
            DownloadPubChemSdfTool,
            LookupCcdCodeTool,
            VisualizeSdfTool,
        )

        registry.register(SearchPubChemTool())
        registry.register(GetPubChemPropertiesTool())
        registry.register(DownloadPubChemSdfTool())
        registry.register(LookupCcdCodeTool())
        registry.register(VisualizeSdfTool())
    except Exception as e:
        logger.warning(f"Failed to register PubChem/CCD tools: {e}")

    # Register skill disclosure tools (progressive disclosure: Level 2 / 3)
    try:
        from api.skills.tools import LoadSkillResourceTool, LoadSkillTool

        registry.register(LoadSkillTool())
        registry.register(LoadSkillResourceTool())
    except Exception as e:
        logger.warning(f"Failed to register skill tools: {e}")

    # Register the deliverable submission tool (local, storage-backed)
    try:
        from .submit_artifact import SubmitArtifactTool

        registry.register(SubmitArtifactTool())
    except Exception as e:
        logger.warning(f"Failed to register submit_artifact tool: {e}")

    # Register the HTTP request tool (authenticated external API calls)
    from .http_common import http_request_tool_enabled

    if http_request_tool_enabled():
        try:
            from .http_request import HttpRequestTool

            registry.register(HttpRequestTool())
        except Exception as e:
            logger.warning(f"Failed to register HTTP request tool: {e}")
    else:
        registry.withheld.append("http_request")
        logger.info("Withheld tool: http_request (AEVOM_FEAT_HTTP_REQUEST_TOOL off)")