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

"""Base classes and interfaces for Aevom tools.

This module defines the abstract interface that all tools must implement,
supporting three tool types:
- LOCAL: In-process Python execution
- NVIDIA_API: External NVIDIA hosted endpoints
- CONTAINER: Sibling container services (e.g., RDKit, VMD)

It also defines the data I/O classification system for tools that store
and retrieve data artifacts via references.
"""

import logging
from abc import ABC, abstractmethod
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

from opentelemetry.trace import Status, StatusCode

from api.context.budget import CHARS_PER_TOKEN

if TYPE_CHECKING:
    from api.storage import DataStorageService

logger = logging.getLogger(__name__)


# Task-local context variables for per-request tool state.
#
# Tool instances are global singletons in the registry, shared across all
# concurrent async requests. These ContextVars replace mutable instance
# attributes to prevent race conditions where one request's await point
# allows another request to overwrite the storage/session on the same
# tool object mid-execution.
_storage_ctx: ContextVar[Optional["DataStorageService"]] = ContextVar(
    "tool_storage", default=None
)
_request_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "tool_request_id", default=None
)
_session_id_ctx: ContextVar[Optional[str]] = ContextVar(
    "tool_session_id", default=None
)
# Set by JobManager.start_job for the duration of tool.execute() when a tool
# opted into job tracking (tool.is_async == True) is running under a
# JobManager. Lets a tool report progress (e.g. an upstream-queued NIM job)
# back into its Job row without importing JobManager directly -- see
# api.jobs.manager.JobProgressHandle and internal_docs/docs/0001-architecture-boundaries.md.
_job_handle_ctx: ContextVar[Optional[Any]] = ContextVar(
    "tool_job_handle", default=None
)


# --- Tool response budget -------------------------------------------------
#
# Tool results are the only text that reaches the model without any length
# bound: the storage pattern keeps SDF/PDB payloads out of context, but a
# tool that *returns text* (python_execute printing an array, load_skill
# returning a document body) is exempt from it. A single 20-iteration turn
# can therefore exceed the context window with no graceful degradation.
#
# The budget closes that hole using the storage pattern the rest of the
# codebase already follows: overflow becomes an artifact, and the model gets
# a head/tail preview plus the reference.
DEFAULT_MAX_RESPONSE_TOKENS = 2000

# Fraction of the surviving budget given to the head of the output. Tool
# output is usually front-loaded (headers, the first rows of a table) but
# the tail often holds the summary line the model actually wants.
_HEAD_FRACTION = 0.6


def build_capped_response(
    text: str,
    max_chars: int,
    overflow_ref: Optional[str] = None,
) -> str:
    """Shrink an over-budget tool response to a head/tail preview.

    Args:
        text: The full rendered tool response.
        max_chars: Character budget for the preview content.
        overflow_ref: Reference of the stored full output, when storage was
            available. Without it the elided middle is unrecoverable, and the
            notice says so rather than pointing at a reference that does not
            exist.

    Returns:
        The preview with an inline notice explaining the elision.
    """
    head_chars = max(1, int(max_chars * _HEAD_FRACTION))
    tail_chars = max(1, max_chars - head_chars)
    elided = len(text) - head_chars - tail_chars

    if overflow_ref:
        notice = (
            f"\n\n... [{elided} characters elided. The full output is stored as "
            f"'{overflow_ref}'. Read it with python_execute: "
            f"aevom.load_text('{overflow_ref}')] ...\n\n"
        )
    else:
        notice = (
            f"\n\n... [{elided} characters elided and not recoverable "
            f"(no storage available for this call). Narrow the tool's output "
            f"and call it again if you need the middle section.] ...\n\n"
        )

    return text[:head_chars] + notice + text[-tail_chars:]


# Marks a ToolResult the tool refused on its arguments; see
# ``ToolResult.invalid_arguments``. A metadata key rather than a subclass so a
# tool signals it without changing what it returns.
INVALID_ARGUMENTS = "invalid_arguments"


class ToolType(Enum):
    """Type of tool execution environment."""
    LOCAL = "local"           # In-process Python execution
    NVIDIA_API = "nvidia_api" # NVIDIA hosted API endpoints
    CONTAINER = "container"   # Sibling container services
    MCP_SERVER = "mcp_server" # External MCP server tools


class DataIOCategory(Enum):
    """Classification of how a tool handles data storage.
    
    This determines whether a tool stores output data, retrieves input data,
    both, or neither. Tools that store/retrieve data use the DataStorageService
    to pass artifacts via references instead of raw data.
    
    Categories:
        TEXT_ONLY: Input and output are natural text, no storage needed.
        OUTPUT_STORES: Tool generates data that is stored, returns a reference.
        INPUT_RETRIEVES: Tool accepts reference parameters, retrieves data internally.
        BOTH: Tool accepts references AND stores output data.
    """
    TEXT_ONLY = "text_only"             # No storage needed (e.g., molecule_info)
    OUTPUT_STORES = "output_stores"     # Stores output, returns ref (e.g., smiles_to_sdf)
    INPUT_RETRIEVES = "input_retrieves" # Accepts refs, retrieves data (e.g., visualize_sdf)
    BOTH = "both"                       # Both input and output use storage (e.g., dock_ligand)


class ContentType(Enum):
    """Type of content returned by a tool."""
    TEXT = "text"             # Plain text response
    IMAGE_URL = "image_url"   # URL to an image
    JSON = "json"             # Structured JSON data
    ERROR = "error"           # Error message
    SDF_3D = "sdf_3d"         # 3D SDF structure for Molstar visualization
    PDB_3D = "pdb_3d"         # 3D PDB structure for Molstar visualization
    DATA_REFERENCE = "data_reference"  # Reference to stored data artifact
    STRUCTURE_QUERY = "structure_query"  # MolQL structure query for Molstar


@dataclass
class ToolResult:
    """Result from tool execution.
    
    Attributes:
        content_type: Type of content in the result.
        data: The actual result data.
        job_id: Optional job ID for async tools.
        metadata: Optional additional metadata.
        response_override: Set by the registry's response budget when the
            rendered output exceeded the tool's allowance. ``data`` is left
            untouched so callers that read it directly (visualization events,
            the training wrapper's pose-ref tracking) are unaffected; only the
            model-facing text shrinks.
        invocation_event: Set by ``ToolRegistry._execute_impl`` after this
            result is produced -- the canonical
            ``api.telemetry_contract.ToolInvocationEvent`` for this call.
            None for a ``ToolResult`` built directly by a tool (before the
            registry has seen it). See ADR 0001.
    """
    content_type: ContentType
    data: Any
    job_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    response_override: Optional[str] = None
    invocation_event: Optional[Any] = None
    
    def to_tool_response(self) -> str:
        """Convert result to string for LLM tool response.
        
        Returns:
            String representation suitable for sending back to the LLM.
        """
        if self.response_override is not None:
            return self.response_override
        if self.content_type == ContentType.TEXT:
            return str(self.data)
        elif self.content_type == ContentType.IMAGE_URL:
            # Return markdown image that LLM can include in response
            return f"Image generated successfully. URL: {self.data}"
        elif self.content_type == ContentType.JSON:
            import json
            return json.dumps(self.data)
        elif self.content_type == ContentType.ERROR:
            return f"Error: {self.data}"
        elif self.content_type == ContentType.SDF_3D:
            # SDF visualization was triggered - confirm to LLM
            label = self.metadata.get("label", "molecule")
            return f"3D structure of {label} is now displayed in the viewer."
        elif self.content_type == ContentType.PDB_3D:
            # PDB visualization was triggered - confirm to LLM
            label = self.metadata.get("label", "protein")
            return f"3D protein structure of {label} is now displayed in the viewer."
        elif self.content_type == ContentType.DATA_REFERENCE:
            # Data was stored, return reference info and guidance message
            return self.metadata.get("message", f"Data stored with reference: {self.data}")
        elif self.content_type == ContentType.STRUCTURE_QUERY:
            # Structure query was sent to frontend for execution
            return self.metadata.get("message", "Structure query executed in the 3D viewer.")
        else:
            return str(self.data)
    
    @classmethod
    def text(cls, data: str, metadata: Optional[dict] = None) -> "ToolResult":
        """Create a text result.
        
        Args:
            data: The text to return to the model.
            metadata: Structured values behind the text. Text results are
                prose to the model but the findings ledger needs the numbers,
                so tools that compute scalars should pass them here rather
                than leaving them recoverable only by re-parsing the prose.
        """
        return cls(
            content_type=ContentType.TEXT,
            data=data,
            metadata=metadata or {},
        )
    
    @classmethod
    def image_url(cls, url: str, metadata: Optional[dict] = None) -> "ToolResult":
        """Create an image URL result."""
        return cls(
            content_type=ContentType.IMAGE_URL,
            data=url,
            metadata=metadata or {},
        )
    
    @classmethod
    def json_data(cls, data: dict[str, Any]) -> "ToolResult":
        """Create a JSON result."""
        return cls(content_type=ContentType.JSON, data=data)
    
    @classmethod
    def error(cls, message: str) -> "ToolResult":
        """Create an error result."""
        return cls(content_type=ContentType.ERROR, data=message)

    @classmethod
    def invalid_arguments(cls, message: str) -> "ToolResult":
        """An error for arguments the tool refused before doing any work.

        Reads to the model exactly like any other error. The difference is who
        pays: budgeted tools are charged before they run, so that a call failing
        inside an expensive service still counts against the budget -- otherwise
        a persistently failing endpoint could be called without limit. Arguments
        the tool never accepted cost that resource nothing, and charging for
        them makes a misremembered field name as expensive as a co-folding run.
        The registry gives the call back when it sees this.

        Measured on two 100-task benchmark runs: 111 of 616 boltz2 calls were
        rejected on their arguments, each one charged against a budget of eight
        per task.
        """
        return cls(
            content_type=ContentType.ERROR,
            data=message,
            metadata={INVALID_ARGUMENTS: True},
        )
    
    @classmethod
    def sdf_3d(cls, sdf_data: str, label: str = "", metadata: Optional[dict] = None) -> "ToolResult":
        """Create an SDF 3D visualization result.
        
        Args:
            sdf_data: The SDF format string with 3D coordinates.
            label: Display label for the molecule.
            metadata: Additional metadata.
            
        Returns:
            ToolResult configured for 3D visualization.
        """
        return cls(
            content_type=ContentType.SDF_3D,
            data=sdf_data,
            metadata={"label": label, **(metadata or {})},
        )
    
    @classmethod
    def pdb_3d(cls, pdb_data: str, label: str = "", metadata: Optional[dict] = None) -> "ToolResult":
        """Create a PDB 3D visualization result.
        
        Args:
            pdb_data: The PDB format string with 3D coordinates.
            label: Display label for the protein structure.
            metadata: Additional metadata (e.g., confidence scores).
            
        Returns:
            ToolResult configured for 3D protein visualization.
        """
        return cls(
            content_type=ContentType.PDB_3D,
            data=pdb_data,
            metadata={"label": label, **(metadata or {})},
        )
    
    @classmethod
    def data_reference(
        cls,
        reference: str,
        data_type: str,
        message: str,
        metadata: Optional[dict] = None,
    ) -> "ToolResult":
        """Create a data reference result for stored artifacts.
        
        Use this when a tool stores data and returns a reference ID
        instead of the raw data. The message should guide the LLM
        on how to use the reference.
        
        Args:
            reference: The reference ID (e.g., "sdf_abc123").
            data_type: Type of data stored (e.g., "sdf", "pdb").
            message: Human-readable message explaining the stored data.
            metadata: Additional metadata about the stored artifact.
            
        Returns:
            ToolResult configured for data reference response.
            
        Example:
            return ToolResult.data_reference(
                reference="sdf_abc123",
                data_type="sdf",
                message="SDF data for ethanol stored as 'sdf_abc123'. "
                        "Pass this reference to visualize_3d to display it.",
                metadata={"smiles": "CCO"},
            )
        """
        return cls(
            content_type=ContentType.DATA_REFERENCE,
            data=reference,
            metadata={
                "data_type": data_type,
                "message": message,
                **(metadata or {}),
            },
        )
    
    @classmethod
    def structure_query(
        cls,
        query: dict,
        action: str = "highlight",
        message: str = "",
        color: str | None = None,
    ) -> "ToolResult":
        """Create a structure query result for Molstar.
        
        Args:
            query: Query specification object.
            action: Action to perform (highlight, focus, select, info).
            message: Human-readable description of the query.
            color: Optional color for highlighting.
            
        Returns:
            ToolResult configured for structure query execution.
        """
        return cls(
            content_type=ContentType.STRUCTURE_QUERY,
            data={
                "query": query,
                "action": action,
                "color": color,
            },
            metadata={"message": message},
        )


class BaseTool(ABC):
    """Abstract base class for all Aevom tools.
    
    All tools must implement this interface to be registered
    and used by the tool registry.
    
    Example:
        class WeatherTool(BaseTool):
            @property
            def name(self) -> str:
                return "get_weather"
            
            def get_schema(self) -> dict:
                return {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather for a city",
                        "parameters": {...}
                    }
                }
            
            async def execute(self, arguments: dict) -> ToolResult:
                location = arguments.get("location")
                # ... get weather ...
                return ToolResult.text(f"Temperature: {temp}°F")
    
    Data Storage Pattern:
        Tools that generate or consume large data artifacts (SDF files, PDB
        structures, etc.) should use the data storage pattern instead of
        passing raw data through the LLM:
        
        1. Override `data_io_category` to indicate storage behavior
        2. Use `self._storage` to store/retrieve data
        3. Return `ToolResult.data_reference()` for stored output
        4. Accept reference parameters and call `self._storage.retrieve()`
        
        Example:
            class SmilesToSdfTool(ContainerTool):
                @property
                def data_io_category(self) -> DataIOCategory:
                    return DataIOCategory.OUTPUT_STORES
                
                async def execute(self, arguments):
                    sdf = generate_sdf(arguments["smiles"])
                    ref = await self._storage.store("sdf", sdf.encode())
                    return ToolResult.data_reference(ref, "sdf", "SDF stored...")
    
    ContentType = ContentType  # Make ContentType accessible via ToolResult.ContentType
    """
    
    @property
    def _storage(self) -> Optional["DataStorageService"]:
        """Per-request storage service (task-local via ContextVar)."""
        return _storage_ctx.get()

    @_storage.setter
    def _storage(self, value: Optional["DataStorageService"]) -> None:
        _storage_ctx.set(value)

    @property
    def _request_id(self) -> Optional[str]:
        """Per-request ID (task-local via ContextVar)."""
        return _request_id_ctx.get()

    @_request_id.setter
    def _request_id(self, value: Optional[str]) -> None:
        _request_id_ctx.set(value)

    @property
    def _session_id(self) -> Optional[str]:
        """Per-request session ID (task-local via ContextVar)."""
        return _session_id_ctx.get()

    @_session_id.setter
    def _session_id(self, value: Optional[str]) -> None:
        _session_id_ctx.set(value)

    @property
    def _job_handle(self) -> Optional[Any]:
        """Progress handle for the Job tracking this call, if any (task-local).

        Only set when this tool opted into job tracking (``is_async`` True)
        and a ``JobManager`` is driving execution. None otherwise -- always
        check before use.
        """
        return _job_handle_ctx.get()

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of the tool (must match function name in schema)."""
        pass
    
    @property
    @abstractmethod
    def tool_type(self) -> ToolType:
        """Type of tool (local, nvidia_api, or container)."""
        pass
    
    @property
    def data_io_category(self) -> DataIOCategory:
        """Classification of how this tool handles data storage.
        
        Override to indicate that the tool stores output data,
        retrieves input data via references, or both.
        
        Returns:
            DataIOCategory.TEXT_ONLY by default (no storage needed).
        """
        return DataIOCategory.TEXT_ONLY
    
    @property
    def max_response_tokens(self) -> int:
        """Token allowance for this tool's model-facing response.
        
        Enforced by :meth:`ToolRegistry.execute`, so it applies identically
        to production and to training rollouts. Override to tighten it for
        tools whose output is unbounded by nature (arbitrary Python output,
        fetched documents).
        
        Returns:
            DEFAULT_MAX_RESPONSE_TOKENS by default.
        """
        return DEFAULT_MAX_RESPONSE_TOKENS
    
    @property
    def batch_argument(self) -> Optional[str]:
        """The array argument that lets one call cover many inputs, if any.

        Only meaningful for a tool with a call budget, where it changes the
        arithmetic completely: one ``admet_predict`` call scoring thirty
        candidates costs the same one unit as a call scoring one. Measured on
        two 100-task benchmark runs, the model averaged 4.27 molecules per call
        against a budget of fifteen calls -- it had the batching and spent the
        budget as if it did not. The ``[TOOL BUDGET]`` block says so for tools
        that name an argument here.

        None means each call covers one thing, which is the honest answer for
        ``boltz2``: it predicts affinity for one ligand per request.
        """
        return None

    @property
    def is_async(self) -> bool:
        """Whether this tool requires async job tracking.
        
        Override to return True for long-running tools that need
        job status tracking and polling.
        
        Returns:
            False by default (synchronous execution).
        """
        return False
    
    @property
    def service_url(self) -> Optional[str]:
        """Base URL for container service tools.
        
        Override for CONTAINER type tools to specify the service URL.
        
        Returns:
            None by default.
        """
        return None
    
    def set_storage(self, storage: "DataStorageService") -> None:
        """Inject storage service for the current request context.
        
        Called by the registry before tool execution. Uses ContextVar
        internally so concurrent async requests each see their own storage.
        
        Args:
            storage: DataStorageService instance for store/retrieve operations.
        """
        self._storage = storage
    
    @property
    def summary_template(self) -> str:
        """Template string for UI summary display.
        
        Override to provide a human-readable summary template that describes
        what the tool is doing. Use {arg_name} placeholders for argument
        interpolation.
        
        Example:
            @property
            def summary_template(self) -> str:
                return "Predicting structure for {sequence_ref}..."
        
        Returns:
            Template string with optional {arg_name} placeholders.
        """
        # Default fallback: use tool name
        return f"{self.name} called"
    
    def format_summary(self, arguments: dict[str, Any]) -> str:
        """Format the summary template with actual argument values.
        
        Interpolates argument values into the summary_template string.
        Unknown placeholders are left as-is to avoid errors.
        
        Args:
            arguments: Dictionary of tool arguments.
            
        Returns:
            Formatted summary string for UI display.
        """
        template = self.summary_template
        try:
            # Use format_map with a default dict to handle missing keys gracefully
            class SafeDict(dict):
                def __missing__(self, key):
                    return f"{{{key}}}"
            
            return template.format_map(SafeDict(arguments))
        except Exception:
            # Fall back to template as-is if formatting fails
            return template
    
    @abstractmethod
    def get_schema(self) -> dict:
        """Get OpenAI-compatible function schema for this tool.
        
        Returns:
            Dict with 'type' and 'function' keys matching OpenAI format.
        """
        pass
    
    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with given arguments.
        
        Args:
            arguments: Dictionary of arguments matching the schema parameters.
            
        Returns:
            ToolResult with the execution result.
        """
        pass


class ContainerTool(BaseTool):
    """Base class for tools that call sibling container services.
    
    Provides common HTTP client functionality for container-based tools.
    """
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.CONTAINER
    
    @property
    @abstractmethod
    def service_url(self) -> str:
        """Base URL of the container service (e.g., 'http://rdkit:8000')."""
        pass
    
    async def _call_service(
        self,
        endpoint: str,
        method: str = "POST",
        json_data: Optional[dict] = None,
        timeout: float = 30.0,
        retries: int = 0,
        base_url: Optional[str] = None,
        headers: Optional[dict[str, str]] = None,
        client_kwargs: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Make HTTP call to the container service.

        Supports load-balanced multi-instance services: if
        ``self.service_url`` is a comma-separated list of base URLs,
        a random one is chosen per call.  On transient failures the
        next retry picks a *different* URL from the pool.

        Emits a ``tool_service.http`` OTel span per HTTP attempt with
        ``tool.name``, ``tool.service_base_url``, ``tool.service_host``,
        ``tool.service_port``, ``tool.attempt`` attributes so we can
        attribute per-service-node load in the agent-loop profiling
        timeline. The span parents under whatever ``tool.{name}`` span
        the registry has currently set as active (so the per-attempt
        HTTP wall is a child of the whole tool call wall).

        Args:
            endpoint: API endpoint path (e.g., '/smiles-to-image').
            method: HTTP method.
            json_data: JSON body for POST requests.
            timeout: Request timeout in seconds.
            retries: Number of retry attempts on transient failures (ReadTimeout, 503).
            base_url: Use this base URL instead of ``self.service_url``.
            headers: Extra request headers, e.g. a gateway routing ``Host``.
            client_kwargs: Extra ``httpx.AsyncClient`` options (TLS material).

        Returns:
            JSON response from the service.

        Raises:
            Exception: If the service call fails after all retries.
        """
        import asyncio
        import httpx
        import os
        import random
        import time
        from contextlib import nullcontext
        from urllib.parse import urlparse

        # Best-effort tracer access -- when telemetry is not initialized
        # the spans become no-ops via the proxy tracer.
        try:
            from api.telemetry import get_tracer

            _tracer = get_tracer()
        except Exception:
            _tracer = None

        base_urls = [u.strip() for u in (base_url or self.service_url).split(",") if u.strip()]
        last_exc: Exception | None = None

        # Client-side backpressure threshold. ``timeout`` is the total budget
        # for one attempt; if a single attempt comes anywhere near that
        # budget the service is saturated (or the work is genuinely slow).
        # We warn at 50% of the budget, capped at 10s, so the threshold
        # works the same for short-timeout (rdkit, molstar) and long-timeout
        # (pyexec, vina) tools without per-tool tuning. See the v0527c
        # post-mortem in internal_docs/docs/training_scaling_curve.md for the design
        # rationale -- this loud warning is the alarm that prevents future
        # silent backpressure stalls.
        backpressure_threshold_s = float(
            os.environ.get(
                "AEVOM_TOOL_BACKPRESSURE_WARN_S",
                str(min(10.0, max(2.0, timeout * 0.5))),
            )
        )

        for attempt in range(1 + retries):
            base = random.choice(base_urls)
            url = f"{base}{endpoint}"

            parsed = urlparse(base)
            host = parsed.hostname or ""
            # Fall back to scheme-default ports if not explicit in the URL.
            if parsed.port is not None:
                port = parsed.port
            elif parsed.scheme == "https":
                port = 443
            elif parsed.scheme == "http":
                port = 80
            else:
                port = -1

            if _tracer is not None:
                span_cm = _tracer.start_as_current_span("tool_service.http")
            else:
                span_cm = nullcontext()

            with span_cm as svc_span:
                if svc_span is not None:
                    svc_span.set_attribute("tool.name", self.name)
                    svc_span.set_attribute("tool.service_base_url", base)
                    svc_span.set_attribute("tool.service_host", host)
                    svc_span.set_attribute("tool.service_port", port)
                    svc_span.set_attribute("tool.service_scheme", parsed.scheme or "")
                    svc_span.set_attribute("tool.endpoint", endpoint)
                    svc_span.set_attribute("tool.attempt", attempt + 1)
                    svc_span.set_attribute("tool.http.method", method.upper())

                t_perf_attempt_start = time.perf_counter()
                # Wire timestamp so the server can compute kernel-queue
                # wait and emit its own [AEVOM-*-BACKPRESSURE] warnings.
                # See services/pyexec/app/main.py for the consumer side.
                outbound_headers = {
                    **(headers or {}),
                    "x-aevom-client-send-ns": str(time.time_ns()),
                }

                try:
                    logger.info(
                        f"Calling container service: {method} {url} "
                        f"(attempt {attempt + 1})"
                    )
                    async with httpx.AsyncClient(timeout=timeout, **(client_kwargs or {})) as client:
                        if method.upper() == "POST":
                            response = await client.post(
                                url, json=json_data, headers=outbound_headers
                            )
                        elif method.upper() == "GET":
                            response = await client.get(
                                url, params=json_data, headers=outbound_headers
                            )
                        else:
                            raise ValueError(
                                f"Unsupported HTTP method: {method}"
                            )

                        response.raise_for_status()
                        result = response.json()
                        attempt_latency_s = time.perf_counter() - t_perf_attempt_start
                        if svc_span is not None:
                            svc_span.set_attribute(
                                "tool.http.status_code", response.status_code
                            )
                            svc_span.set_attribute(
                                "tool.http.client_latency_s",
                                round(attempt_latency_s, 4),
                            )
                            try:
                                svc_span.set_attribute(
                                    "tool.http.response_bytes",
                                    int(response.headers.get("content-length", "0")),
                                )
                            except (TypeError, ValueError):
                                pass
                            svc_span.set_status(Status(StatusCode.OK))

                        if attempt_latency_s >= backpressure_threshold_s:
                            # Loud, structured, greppable. The key metric here
                            # is "this single attempt took more than a fast
                            # path's worth of time"; the matching server-side
                            # warning (if the service is one of ours) will
                            # tell us whether the slowness was queue wait or
                            # genuine work. Don't try to throttle; scale.
                            logger.warning(
                                "[AEVOM-TOOL-BACKPRESSURE] %s -> %s%s took "
                                "%.2fs (>= warn threshold %.2fs, attempt "
                                "timeout %.1fs). attempt=%d. If this fires "
                                "repeatedly the service is saturated -- "
                                "scale workers in services.yaml or add "
                                "full-stack service nodes. Do NOT cap "
                                "concurrent requests on the client side; "
                                "backpressure must be visible.",
                                self.name,
                                base,
                                endpoint,
                                attempt_latency_s,
                                backpressure_threshold_s,
                                timeout,
                                attempt + 1,
                            )

                        logger.info(f"Container service response: {result}")
                        return result
                except (
                    httpx.ReadTimeout,
                    httpx.WriteTimeout,
                    httpx.ConnectTimeout,
                    httpx.PoolTimeout,
                    httpx.ConnectError,
                ) as e:
                    last_exc = e
                    attempt_latency_s = time.perf_counter() - t_perf_attempt_start
                    if svc_span is not None:
                        svc_span.set_attribute(
                            "tool.http.error_type", type(e).__name__
                        )
                        svc_span.set_attribute(
                            "tool.http.client_latency_s",
                            round(attempt_latency_s, 4),
                        )
                        svc_span.set_status(
                            Status(
                                StatusCode.ERROR,
                                f"{type(e).__name__} after {attempt_latency_s:.1f}s "
                                f"(timeout {timeout:.0f}s) {str(e)}".strip()[:200],
                            )
                        )
                    # Timeouts on a healthy fast-path service almost always
                    # indicate saturation. Speak loudly so this is easy to
                    # find in driver/service logs without needing Phoenix.
                    logger.warning(
                        "[AEVOM-TOOL-BACKPRESSURE] %s -> %s%s timed out "
                        "after %.2fs (%s, attempt timeout %.1fs, "
                        "attempt=%d). Likely cause: target service "
                        "saturated -- scale workers / add service nodes.",
                        self.name,
                        base,
                        endpoint,
                        attempt_latency_s,
                        type(e).__name__,
                        timeout,
                        attempt + 1,
                    )
                    if attempt < retries:
                        wait = 2 ** attempt
                        logger.warning(
                            f"Service call failed ({endpoint} -> {base}), "
                            f"retrying in {wait}s "
                            f"(attempt {attempt + 1}/{1 + retries})"
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise
                except httpx.HTTPStatusError as e:
                    attempt_latency_s = time.perf_counter() - t_perf_attempt_start
                    if svc_span is not None:
                        svc_span.set_attribute(
                            "tool.http.status_code", e.response.status_code
                        )
                        svc_span.set_attribute(
                            "tool.http.error_type", "HTTPStatusError"
                        )
                        svc_span.set_attribute(
                            "tool.http.client_latency_s",
                            round(attempt_latency_s, 4),
                        )
                        svc_span.set_status(
                            Status(StatusCode.ERROR, str(e)[:200])
                        )
                    if e.response.status_code == 503 and attempt < retries:
                        last_exc = e
                        wait = 2 ** attempt
                        logger.warning(
                            f"Service unavailable ({endpoint} -> {base}), "
                            f"retrying in {wait}s"
                        )
                        await asyncio.sleep(wait)
                        continue
                    raise
        raise last_exc  # type: ignore[misc]


class NvidiaApiError(RuntimeError):
    """A NIM call failed in a way the model should be told about verbatim.

    Carries a message written for the model, not a stack trace: on a 422 that is
    the endpoint's own validation complaint, which usually names the field to fix.
    """

    def __init__(self, message: str, *, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class NvidiaApiTool(BaseTool):
    """Base class for tools that call NVIDIA hosted API endpoints.

    Provides common authentication and HTTP client functionality
    for NVIDIA API-based tools.
    """

    # How long the NVCF gateway holds the connection open waiting for the model
    # to finish before handing back a 202 and an async job id. Five minutes
    # covers every NIM Aevom calls; the fast ones return in seconds.
    DEFAULT_POLL_SECONDS = 300

    # The client must outlive the gateway's poll window, or it aborts a request
    # the gateway is still holding open and the work is thrown away.
    _TIMEOUT_MARGIN_SECONDS = 50.0

    @property
    def tool_type(self) -> ToolType:
        return ToolType.NVIDIA_API

    @property
    @abstractmethod
    def api_endpoint(self) -> str:
        """NVIDIA API endpoint URL."""
        pass

    # Set to bind this tool instance to a self-hosted NIM rather than the
    # hosted NVIDIA endpoint: ``(url, api_key_or_None)``. Tools contributed by
    # a NIM adapter are constructed this way, since they only exist because a
    # user registered an endpoint. A session-scoped override (see
    # ``api.nim.overrides``) does the same for built-in tools, which are
    # singletons and so cannot carry per-session state on the instance.
    self_hosted: Optional[tuple[str, Optional[str]]] = None

    def _resolve_endpoint(self) -> tuple[str, Optional[str], bool]:
        """Decide where this call goes, and how it authenticates.

        Returns ``(url, api_key, is_self_hosted)``. For a self-hosted NIM the
        key may be None, which means send no ``Authorization`` header at all --
        a local NIM has no bearer token to check, and ``_get_api_key`` would
        raise before the request was ever made.

        Precedence is instance, then session, then the hosted default. An
        adapter-built tool is bound at construction and cannot be re-pointed;
        a built-in tool consults the session, so one user's self-hosted
        Boltz-2 does not redirect anyone else's.
        """
        if self.self_hosted is not None:
            url, key = self.self_hosted
            return url, key, True

        session_id = self._session_id
        if session_id:
            from api.nim.overrides import get_endpoint_override

            override = get_endpoint_override(session_id, self.name)
            if override is not None:
                logger.info(
                    "Routing %s to self-hosted NIM %s for session %s",
                    self.name, override.url, session_id,
                )
                return override.url, override.api_key, True

        return self.api_endpoint, None, False

    def _get_api_key(self) -> str:
        """Get NVIDIA API key from environment."""
        import os
        from pathlib import Path

        key_file = os.environ.get("NVIDIA_API_KEY_FILE", "/run/secrets/nvidia_api_key")
        try:
            key = Path(key_file).read_text().strip()
        except FileNotFoundError:
            raise RuntimeError(f"API key file not found: {key_file}")
        # An empty file is rejected here rather than sent as `Bearer `, because
        # health.api.nvidia.com answers a missing credential with HTTP 500 and an
        # axum "Missing request extension: Authorization<Bearer>" body, not a 401.
        # Surfaced through _describe_http_error that reads like an upstream bug,
        # which is how an unset $NVIDIA_API_KEY in CI cost a whole eval suite.
        # Same wording as api/llm/client.py so one grep finds either reader.
        if not key:
            raise RuntimeError(f"API key file is empty: {key_file}")
        return key

    @staticmethod
    def _describe_http_error(response) -> str:
        """Turn an error response into a line the model can act on.

        NIM validation errors arrive as ``{"error": "..."}``, which is specific
        enough to be worth surfacing verbatim -- "Either input_pdb or
        input_pdb_asset must be specified" tells the model exactly what to change.
        Other NVIDIA services use FastAPI's ``detail``, in either its string or
        its list-of-errors form, so all three are handled.
        """
        try:
            payload = response.json()
        except Exception:  # noqa: BLE001 - a non-JSON error body is still useful
            text = (response.text or "").strip()
            return f" Response: {text[:300]}" if text else ""

        if isinstance(payload, dict):
            # The NIM error contract wraps a stable code beside the prose:
            # {"error": {"code": "out_of_memory", "message": ..., "details": [...]}}.
            # The code is the part a caller can branch on, so it is kept in the
            # message rather than discarded in favour of the prose alone.
            error = payload.get("error")
            if isinstance(error, dict) and error.get("message"):
                code = str(error.get("code") or "").strip()
                message = str(error["message"]).strip()
                details = [str(d).strip() for d in error.get("details") or [] if d]
                rendered = f" [{code}] {message}" if code else f" {message}"
                if details:
                    rendered += " " + "; ".join(details)
                return rendered

            for key in ("error", "detail", "message"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return f" {value.strip()}"
                if isinstance(value, list):
                    # FastAPI validation shape: [{"loc": [...], "msg": "..."}].
                    parts = []
                    for item in value:
                        if not isinstance(item, dict):
                            continue
                        loc = ".".join(str(p) for p in item.get("loc", []) if p != "body")
                        msg = str(item.get("msg", "")).strip()
                        parts.append(f"{loc}: {msg}" if loc and msg else msg or loc)
                    if parts:
                        return " " + "; ".join(p for p in parts if p)
        return f" Response: {str(payload)[:300]}"

    async def _call_nvidia_api(
        self,
        json_data: dict[str, Any],
        *,
        poll_seconds: Optional[int] = None,
        endpoint: Optional[str] = None,
        service_name: Optional[str] = None,
    ) -> dict[str, Any]:
        """POST to a NIM endpoint and return the parsed JSON response.

        Args:
            json_data: Request body.
            poll_seconds: How long the gateway should wait for the result before
                falling back to an async job. Defaults to
                :attr:`DEFAULT_POLL_SECONDS`; the client timeout is derived from
                it so the two cannot drift apart.
            endpoint: Override :attr:`api_endpoint`, for services with more than
                one path.
            service_name: Name used in error messages. Defaults to the tool name.

        Raises:
            NvidiaApiError: On any non-2xx response, timeout or connection
                failure, with a message meant to be shown to the model.
        """
        import httpx

        from api.bounded_http import ResponseTooLargeError, format_limit, request_bounded

        poll = self.DEFAULT_POLL_SECONDS if poll_seconds is None else poll_seconds
        resolved_url, self_hosted_key, is_self_hosted = self._resolve_endpoint()
        url = endpoint or resolved_url
        label = service_name or self.name

        headers = {"Content-Type": "application/json"}
        if is_self_hosted:
            # No NVCF gateway in front of a self-hosted NIM: it answers the
            # POST synchronously, so NVCF-POLL-SECONDS would be a header it
            # has never heard of, and there is no NVIDIA API key to present.
            # A registered key is still sent, for a NIM behind an auth proxy.
            if self_hosted_key:
                headers["Authorization"] = f"Bearer {self_hosted_key}"
        else:
            headers["Authorization"] = f"Bearer {self._get_api_key()}"
            headers["NVCF-POLL-SECONDS"] = str(poll)

        try:
            async with httpx.AsyncClient(
                timeout=poll + self._TIMEOUT_MARGIN_SECONDS
            ) as client:
                response = await request_bounded(
                    client, "POST", url, headers=headers, json=json_data
                )
        except ResponseTooLargeError as exc:
            raise NvidiaApiError(
                f"{label} returned a response larger than Aevom's "
                f"{format_limit(exc.limit)} limit, so it was discarded. Retry with "
                "a smaller request, for example fewer samples or designs."
            ) from exc
        except httpx.TimeoutException as exc:
            raise NvidiaApiError(
                f"{label} did not respond within {poll} seconds. The service may "
                "be under load; try again, or reduce the size of the request."
            ) from exc
        except httpx.RequestError as exc:
            raise NvidiaApiError(
                f"Could not reach {label}: {exc}. The service may be unavailable."
            ) from exc

        if response.status_code == 202 and not is_self_hosted:
            # The gateway gave up waiting and queued the job. Aevom has no
            # mechanism to resume a NIM job across tool calls, so say so plainly
            # rather than returning a payload with no results in it. If a
            # JobManager is tracking this call (tool.is_async opted in), record
            # the upstream id so the queued case is observable/queryable rather
            # than only surfacing as this error message -- see ADR 0001. This
            # does not implement resumption.
            upstream_job_id = response.headers.get("NVCF-REQID")
            if upstream_job_id and self._job_handle is not None:
                try:
                    await self._job_handle.record_upstream_pending(upstream_job_id)
                except Exception:  # noqa: BLE001 - never fail the call over bookkeeping
                    logger.warning(
                        "Failed to record upstream-queued job for %s", label,
                        exc_info=True,
                    )
            raise NvidiaApiError(
                f"{label} took longer than {poll} seconds and was queued for "
                "asynchronous completion, which Aevom cannot retrieve. Retry with "
                "a smaller or simpler request.",
                status_code=202,
            )

        if response.is_redirect:
            raise NvidiaApiError(
                f"{label} redirected to "
                f"{response.headers.get('location', 'another URL')}, and Aevom "
                "does not follow redirects. The registered NIM URL needs to be "
                "the final one.",
                status_code=response.status_code,
            )

        if response.status_code >= 400:
            raise NvidiaApiError(
                f"{label} rejected the request (HTTP {response.status_code})."
                + self._describe_http_error(response),
                status_code=response.status_code,
            )

        try:
            return response.json()
        except Exception as exc:  # noqa: BLE001 - surface the body, not a parse trace
            raise NvidiaApiError(
                f"{label} returned a response that was not JSON: "
                f"{(response.text or '')[:300]}"
            ) from exc

