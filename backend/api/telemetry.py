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

"""Telemetry module for Aevom using OpenTelemetry and Arize Phoenix.

This module initializes OpenTelemetry tracing with Phoenix as the collector,
and provides helpers for creating spans throughout the application.
"""

import json
import logging
import os
import threading
from contextvars import ContextVar
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry import metrics as otel_metrics
from opentelemetry.trace import Span, Status, StatusCode, Tracer

logger = logging.getLogger(__name__)

# Context variable to hold the current chat span for nested instrumentation
_current_chat_span: ContextVar[Optional[Span]] = ContextVar(
    "current_chat_span", default=None
)
_current_iteration_span: ContextVar[Optional[Span]] = ContextVar(
    "current_iteration_span", default=None
)

# When True, ToolRegistry.execute() skips span creation.
# Used by training to suppress real-time tool spans (reconstructed at verify time).
_skip_tool_spans: ContextVar[bool] = ContextVar("skip_tool_spans", default=False)

# Hex span_id (16-char lower-hex) of the most recently opened tool.{name} span
# in this task. Written by ``trace_tool_execution`` so the training resource
# server (training/nemo_gym_wrapper/app.py:execute_tool) can recover the live
# tool span_id after ToolRegistry.execute() returns and store it on the
# per-call ``session.tool_calls[i]`` record. EpisodeSpan.finalize_trace()
# then uses these IDs to target per-tool reward annotations at the actual
# ``tool.{name}`` spans instead of falling back to the root ``chat_request``.
# Lives in a ContextVar so concurrent FastAPI requests don't clobber each
# other's IDs.
_last_tool_span_id: ContextVar[Optional[str]] = ContextVar(
    "last_tool_span_id", default=None
)

# Tracer instance (initialized lazily)
_tracer: Optional[Tracer] = None
_tracer_provider: Any = None
_initialized: bool = False

# Metrics
_meter: Optional[otel_metrics.Meter] = None
_token_counter: Optional[otel_metrics.Counter] = None


class TokenAccumulator:
    """Thread-safe token counter shared across graph nodes via ContextVar.

    Because the ContextVar holds a reference to a mutable object, LangGraph's
    per-node ContextVar copy still points to the *same* accumulator instance,
    so writes from inside graph nodes are visible outside.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens


_token_accumulator: ContextVar[Optional[TokenAccumulator]] = ContextVar(
    "token_accumulator", default=None
)


def set_token_accumulator(acc: Optional[TokenAccumulator]) -> None:
    _token_accumulator.set(acc)


def get_token_accumulator() -> Optional[TokenAccumulator]:
    return _token_accumulator.get()


def init_telemetry() -> Tracer:
    """Initialize OpenTelemetry with Phoenix collector.

    Uses arize-phoenix-otel's register() function which handles:
    - TracerProvider setup
    - Span exporter configuration (gRPC to Phoenix)
    - Environment variable reading (PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_PROJECT_NAME)

    Returns:
        The configured Tracer instance.
    """
    global _tracer, _tracer_provider, _initialized, _meter, _token_counter

    if _initialized:
        return _tracer

    endpoint = os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:4317")

    try:
        from phoenix.otel import register

        project_name = os.environ.get("PHOENIX_PROJECT_NAME", "aevom")

        logger.info(f"Initializing OpenTelemetry with Phoenix endpoint: {endpoint}")

        # Register with Phoenix - this sets up the global TracerProvider
        tracer_provider = register(
            project_name=project_name,
            endpoint=endpoint,
        )
        _tracer_provider = tracer_provider

        # Get our tracer from the provider
        _tracer = trace.get_tracer(
            "aevom",
            tracer_provider=tracer_provider,
        )
        _initialized = True

        logger.info(
            f"OpenTelemetry initialized successfully for project: {project_name}"
        )

    except ImportError as e:
        logger.warning(
            f"Failed to import phoenix.otel: {e}. "
            "Telemetry will be disabled. Install arize-phoenix-otel to enable."
        )
        _tracer = trace.get_tracer("aevom")  # NoOp tracer
        _initialized = True
    except Exception as e:
        logger.exception(f"Failed to initialize telemetry: {e}")
        _tracer = trace.get_tracer("aevom")  # NoOp tracer
        _initialized = True

    # Metrics — separate try/except so a metrics failure doesn't kill tracing
    try:
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
            OTLPMetricExporter,
        )

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=True)
        )
        meter_provider = MeterProvider(metric_readers=[metric_reader])
        otel_metrics.set_meter_provider(meter_provider)
        _meter = otel_metrics.get_meter("aevom")
        _token_counter = _meter.create_counter(
            name="llm_agent_tokens_total",
            description="Cumulative LLM tokens consumed",
            unit="tokens",
        )
    except Exception as e:
        logger.warning(f"Metrics init failed (counter will be no-op): {e}")
        _meter = otel_metrics.get_meter("aevom")  # no-op
        _token_counter = _meter.create_counter(
            name="llm_agent_tokens_total",
            description="Cumulative LLM tokens consumed",
            unit="tokens",
        )

    return _tracer


def get_tracer() -> Tracer:
    """Get the global tracer instance, initializing if needed.

    Returns:
        The Tracer instance for creating spans.
    """
    global _tracer
    if _tracer is None:
        init_telemetry()
    return _tracer


def get_token_counter() -> Optional[otel_metrics.Counter]:
    """Get the global token counter for recording LLM token usage."""
    return _token_counter


def flush_spans(timeout_ms: int = 10000) -> bool:
    """Force-flush all pending spans to the collector.

    Must be called after ending spans and before annotating them via the
    Phoenix REST API, otherwise the spans may not exist in Phoenix yet
    (OTel's BatchSpanProcessor queues exports).

    Returns:
        True if flush succeeded, False if no provider or flush failed.
    """
    if _tracer_provider is not None and hasattr(_tracer_provider, "force_flush"):
        try:
            return _tracer_provider.force_flush(timeout_ms)
        except Exception:
            logger.exception("force_flush failed")
            return False
    return False


def set_current_chat_span(span: Optional[Span]) -> None:
    """Set the current chat request span for context propagation."""
    _current_chat_span.set(span)


def get_current_chat_span() -> Optional[Span]:
    """Get the current chat request span."""
    return _current_chat_span.get()


def set_current_iteration_span(span: Optional[Span]) -> None:
    """Set the current reasoning iteration span for context propagation."""
    _current_iteration_span.set(span)


def get_current_iteration_span() -> Optional[Span]:
    """Get the current reasoning iteration span."""
    return _current_iteration_span.get()


def set_skip_tool_spans(skip: bool) -> None:
    """When True, ToolRegistry.execute() skips OTel span creation."""
    _skip_tool_spans.set(skip)


def should_skip_tool_spans() -> bool:
    """Check whether tool span creation should be suppressed."""
    return _skip_tool_spans.get()


def set_last_tool_span_id(span_id_hex: Optional[str]) -> None:
    """Record the most recently opened tool.{name} span_id (16-char hex)."""
    _last_tool_span_id.set(span_id_hex)


def get_last_tool_span_id() -> Optional[str]:
    """Read the most recently opened tool.{name} span_id (16-char hex).

    Returns ``None`` if no tool span has been opened in this contextvars
    context, or if the producing tracer is a no-op (no valid span_id).
    """
    return _last_tool_span_id.get()


# =====================================================================
# OpenInference semantic-convention helpers
# =====================================================================
#
# Phoenix's trace UI renders LLM messages, reasoning, tool calls, and
# trace-level I/O ONLY from OpenInference attributes (openinference.span.kind
# + the flattened llm.input_messages.* / llm.output_messages.* / input.value /
# output.value schema). Aevom historically stored everything in custom
# attributes (llm.raw.*, chat.*, judge.*, tool.*) on UNKNOWN-kind spans, which
# the upgraded arize-phoenix 13.x UI does not surface in its rendered panels.
#
# These helpers are PURELY ADDITIVE: callers keep emitting all existing custom
# + profiling attributes (every field scripts/profile_run.py reads) and ALSO
# call these to make the same data render natively. Every helper is
# try/except-guarded -- telemetry must never break the training loop.
#
# Reasoning note: the canonical OpenInference spec only defines
# ``message_content.type`` values "text"/"image", so we surface reasoning via
# the reliably-rendered ``message.content`` string (reasoning concatenated with
# any final assistant text) rather than a non-standard reasoning content block.


def _safe_set(span: Optional[Span], key: str, value: Any) -> None:
    if span is None or value is None:
        return
    try:
        span.set_attribute(key, value)
    except Exception:
        pass


def set_span_kind(span: Optional[Span], kind: str) -> None:
    """Set ``openinference.span.kind`` (ALL CAPS: LLM/AGENT/TOOL/CHAIN/EVALUATOR)."""
    _safe_set(span, "openinference.span.kind", kind)


def _to_value_mime(value: Any) -> tuple[str, str]:
    if isinstance(value, str):
        return value, "text/plain"
    try:
        return json.dumps(value, default=str), "application/json"
    except Exception:
        return str(value), "text/plain"


def set_io(
    span: Optional[Span],
    input_value: Any = None,
    output_value: Any = None,
) -> None:
    """Set OpenInference ``input.value`` / ``output.value`` (+ mime) on a span.

    The trace-list Input/Output columns are read from the ROOT span's
    ``input.value`` / ``output.value``; child spans use these for their own
    detail panels.
    """
    if span is None:
        return
    if input_value is not None:
        val, mime = _to_value_mime(input_value)
        _safe_set(span, "input.value", val)
        _safe_set(span, "input.mime_type", mime)
    if output_value is not None:
        val, mime = _to_value_mime(output_value)
        _safe_set(span, "output.value", val)
        _safe_set(span, "output.mime_type", mime)


def set_input_messages(span: Optional[Span], messages: Optional[list]) -> None:
    """Flatten a chat-history list onto ``llm.input_messages.{i}.message.*``.

    Each message is a dict with ``role`` (required) and any of ``content``
    (str), ``name``, ``tool_call_id``, and ``tool_calls`` (list of
    ``{id, name, arguments}``).
    """
    if span is None or not messages:
        return
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        base = f"llm.input_messages.{i}.message"
        _safe_set(span, f"{base}.role", m.get("role") or "user")
        _safe_set(span, f"{base}.content", m.get("content"))
        _safe_set(span, f"{base}.name", m.get("name"))
        _safe_set(span, f"{base}.tool_call_id", m.get("tool_call_id"))
        for k, tc in enumerate(m.get("tool_calls") or []):
            tb = f"{base}.tool_calls.{k}.tool_call"
            _safe_set(span, f"{tb}.id", tc.get("id"))
            _safe_set(span, f"{tb}.function.name", tc.get("name", ""))
            _safe_set(span, f"{tb}.function.arguments", tc.get("arguments", ""))


def set_output_message(
    span: Optional[Span],
    *,
    role: str = "assistant",
    reasoning: Optional[str] = None,
    content: Optional[str] = None,
    tool_calls: Optional[list] = None,
) -> None:
    """Set the single assistant output message on ``llm.output_messages.0``.

    Reasoning and final content are concatenated into ``message.content`` (the
    field Phoenix renders most reliably) so per-turn chain-of-thought is always
    visible even on tool-calling turns that emit no final assistant text.
    Tool calls are flattened under ``message.tool_calls.{k}.tool_call.*``.
    """
    if span is None:
        return
    base = "llm.output_messages.0.message"
    _safe_set(span, f"{base}.role", role)
    parts = [p for p in (reasoning, content) if p]
    if parts:
        _safe_set(span, f"{base}.content", "\n\n".join(parts))
    for k, tc in enumerate(tool_calls or []):
        tb = f"{base}.tool_calls.{k}.tool_call"
        _safe_set(span, f"{tb}.id", tc.get("id"))
        _safe_set(span, f"{tb}.function.name", tc.get("name", ""))
        _safe_set(span, f"{tb}.function.arguments", tc.get("arguments", ""))


def set_llm_tools(span: Optional[Span], tool_schemas: Optional[list]) -> None:
    """Flatten advertised tool schemas onto ``llm.tools.{k}.tool.json_schema``."""
    if span is None or not tool_schemas:
        return
    for k, schema in enumerate(tool_schemas):
        try:
            js = schema if isinstance(schema, str) else json.dumps(schema, default=str)
        except Exception:
            continue
        _safe_set(span, f"llm.tools.{k}.tool.json_schema", js)


# Type variable for preserving function signatures in decorators
F = TypeVar("F", bound=Callable[..., Any])


def trace_llm_call(operation: str = "llm.call") -> Callable[[F], F]:
    """Decorator to trace LLM API calls.

    Args:
        operation: Name of the operation (e.g., "llm.complete", "llm.stream")

    Returns:
        Decorated function with tracing.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracer = get_tracer()

            # Get parent span context (iteration span if available, else chat span)
            parent = get_current_iteration_span() or get_current_chat_span()
            context = trace.set_span_in_context(parent) if parent else None

            with tracer.start_as_current_span(operation, context=context) as span:
                # Extract relevant attributes from kwargs or args
                if args and hasattr(args[0], "config"):
                    # First arg is self (LLMClient instance)
                    client = args[0]
                    span.set_attribute("llm.model", client.config.name)

                # Common kwargs
                if "temperature" in kwargs and kwargs["temperature"] is not None:
                    span.set_attribute("llm.temperature", kwargs["temperature"])
                if "max_tokens" in kwargs and kwargs["max_tokens"] is not None:
                    span.set_attribute("llm.max_tokens", kwargs["max_tokens"])
                if "tools" in kwargs and kwargs["tools"]:
                    span.set_attribute("llm.tools_count", len(kwargs["tools"]))

                try:
                    result = func(*args, **kwargs)

                    # For complete() calls, record tool calls if present
                    if hasattr(result, "tool_calls") and result.tool_calls:
                        span.set_attribute(
                            "llm.tool_calls_count", len(result.tool_calls)
                        )
                        tool_names = [
                            tc["function"]["name"] for tc in result.tool_calls
                        ]
                        span.set_attribute("llm.tool_calls", ",".join(tool_names))

                    if hasattr(result, "reasoning") and result.reasoning:
                        span.set_attribute("llm.has_reasoning", True)

                    span.set_status(Status(StatusCode.OK))
                    return result

                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper  # type: ignore

    return decorator


def trace_tool_execution(tool_name: str, arguments: dict[str, Any]) -> Span:
    """Create a span for tool execution.

    This is meant to be used as a context manager:

        with trace_tool_execution("my_tool", args) as span:
            result = await tool.execute(args)
            span.set_attribute("tool.result_type", result.content_type.value)

    Args:
        tool_name: Name of the tool being executed.
        arguments: Tool arguments (will be serialized as attributes).

    Returns:
        The span context manager.
    """
    tracer = get_tracer()

    # Get parent span context (iteration span if available)
    parent = get_current_iteration_span() or get_current_chat_span()
    context = trace.set_span_in_context(parent) if parent else None

    span = tracer.start_span(f"tool.{tool_name}", context=context)
    span.set_attribute("tool.name", tool_name)

    # OpenInference: render as a TOOL span with structured I/O in Phoenix.
    # Additive -- the tool.name / tool.arg.* attrs below are preserved.
    set_span_kind(span, "TOOL")
    set_io(span, input_value=arguments)
    _safe_set(span, "tool.parameters", _to_value_mime(arguments)[0])

    # Record the live span_id so the training resource server can recover it
    # after registry.execute() returns and target per-tool reward annotations
    # at this exact span (instead of falling back to the root chat_request).
    # No-op tracer / invalid context yields a zero span_id, which we treat
    # as "unknown" (None) to keep downstream lookups simple.
    try:
        sc = span.get_span_context()
        if sc is not None and sc.is_valid and sc.span_id != 0:
            set_last_tool_span_id(format(sc.span_id, "016x"))
        else:
            set_last_tool_span_id(None)
    except Exception:
        set_last_tool_span_id(None)

    # Add arguments as attributes (be careful with large values)
    for key, value in arguments.items():
        # Limit string values to prevent huge spans
        str_value = str(value)
        if len(str_value) > 500:
            str_value = str_value[:500] + "...[truncated]"
        span.set_attribute(f"tool.arg.{key}", str_value)

    return span


class ChatSpanContext:
    """Context manager for chat request spans.

    Creates a root span for the entire chat request and manages
    the context for child spans.

    Example:
        async with ChatSpanContext(user_message) as chat_span:
            # All operations within will be children of chat_span
            ...
    """

    def __init__(
        self,
        user_message: str,
        model_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        """Initialize chat span context.

        Args:
            user_message: The user's chat message (stored in full).
            model_name: Optional LLM model name.
            session_id: The session this turn belongs to. Recorded because it
                is the only stable key that joins a trace back to whatever
                drove it -- a benchmark task, an eval case, a support report.
                Without it the only handle is the message text, and matching
                on a prefix of that is both fragile and ambiguous the moment
                two runs send the same prompt.
        """
        self.user_message = user_message or ""  # Full message, no truncation
        self.model_name = model_name
        self.session_id = session_id
        self.span: Optional[Span] = None
        self._token = None

    def __enter__(self) -> Span:
        tracer = get_tracer()
        # An empty context makes this the trace root even when a span is
        # already active (an HTTP server span, for one), so a user turn is
        # always a trace of its own.
        self.span = tracer.start_span("chat_request", context=otel_context.Context())
        self.span.set_attribute("chat.user_message", self.user_message)
        if self.session_id:
            self.span.set_attribute("chat.session_id", self.session_id)
        if self.model_name:
            self.span.set_attribute("llm.model", self.model_name)

        # Set as current chat span for child spans
        self._token = _current_chat_span.set(self.span)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            # GeneratorExit and CancelledError are normal cleanup, not errors
            import asyncio

            is_normal_exit = (
                exc_type is None
                or exc_type is GeneratorExit
                or (isinstance(exc_val, asyncio.CancelledError) if exc_val else False)
            )
            if is_normal_exit:
                self.span.set_status(Status(StatusCode.OK))
            else:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            self.span.end()

        # Reset context
        _current_chat_span.set(None)
        return False  # Don't suppress exceptions

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


class IterationSpanContext:
    """Context manager for reasoning iteration spans.

    Creates a child span for each iteration of the tool-calling loop.

    Example:
        async with IterationSpanContext(iteration=1) as iter_span:
            response = client.complete(...)
            ...
    """

    def __init__(self, iteration: int):
        """Initialize iteration span context.

        Args:
            iteration: The iteration number (1-based).
        """
        self.iteration = iteration
        self.span: Optional[Span] = None
        self._token = None

    def __enter__(self) -> Span:
        tracer = get_tracer()

        # Parent is the chat span
        parent = get_current_chat_span()
        context = trace.set_span_in_context(parent) if parent else None

        self.span = tracer.start_span(
            f"reasoning_iteration.{self.iteration}", context=context
        )
        self.span.set_attribute("iteration.number", self.iteration)

        # Set as current iteration span for child spans (LLM calls, tools)
        self._token = _current_iteration_span.set(self.span)
        return self.span

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.span:
            # GeneratorExit and CancelledError are normal cleanup, not errors
            import asyncio

            is_normal_exit = (
                exc_type is None
                or exc_type is GeneratorExit
                or (isinstance(exc_val, asyncio.CancelledError) if exc_val else False)
            )
            if is_normal_exit:
                self.span.set_status(Status(StatusCode.OK))
            else:
                self.span.set_status(Status(StatusCode.ERROR, str(exc_val)))
                self.span.record_exception(exc_val)
            self.span.end()

        # Reset iteration context (but keep chat span)
        _current_iteration_span.set(None)
        return False

    async def __aenter__(self) -> Span:
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


def add_span_event(name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    """Add an event to the current span (iteration or chat).

    Args:
        name: Event name.
        attributes: Optional event attributes.
    """
    span = get_current_iteration_span() or get_current_chat_span()
    if span:
        span.add_event(name, attributes=attributes or {})


def get_current_span_id() -> Optional[str]:
    """Get the span_id of the current chat span as a hex string.

    This is used to pass the span_id to the frontend for feedback/annotation purposes.
    The span_id is formatted as a 16-character hex string (64-bit).

    Returns:
        The span_id as a hex string, or None if no chat span is active.
    """
    span = get_current_chat_span()
    if span is None:
        return None

    span_context = span.get_span_context()
    if span_context is None or not span_context.is_valid:
        return None

    # Format as 16-character hex string (64-bit span ID)
    return format(span_context.span_id, "016x")
