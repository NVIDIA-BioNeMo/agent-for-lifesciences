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

"""MCP tool adapter for heavy data types.

This module wraps MCP tools that handle heavy chem/bio data types,
transparently converting between Aevom's lightweight storage references
and the raw data that the MCP tool expects.

When a user adds an MCP tool, the adapter layer:
1. Detects heavy data types in the tool's input/output schemas
2. Rewrites the LLM-facing schema to use reference parameters
3. On execution, unpacks references to raw data before calling the MCP tool
4. On output, detects heavy data in the response and stores it
"""

import copy
import logging
from typing import TYPE_CHECKING, Any, Union

from api.storage import ArtifactNotFoundError
from api.storage.datatypes import friendly_name
from api.tools.base import BaseTool, DataIOCategory, ToolResult, ToolType

from .detector import DetectionResult, HeavyDataDetector, extract_smiles_from_content

if TYPE_CHECKING:
    from .wrapper import MCPToolWrapper

logger = logging.getLogger(__name__)


def _ref_param_name(original_name: str, data_type: str) -> str:
    """Generate the reference parameter name for an adapted input.

    If the original name already ends with the data type, just append '_ref'.
    Otherwise, replace the original name with '{data_type}_ref'.

    Examples:
        _ref_param_name("smiles", "smiles") -> "smiles_ref"
        _ref_param_name("ligand_smiles", "smiles") -> "ligand_smiles_ref"
        _ref_param_name("molecule", "smiles") -> "molecule_smiles_ref"
        _ref_param_name("input_pdb", "pdb") -> "input_pdb_ref"
    """
    lower = original_name.lower()
    # If the param name already contains or ends with the data type
    if lower.endswith(data_type):
        return f"{original_name}_ref"
    # Otherwise append type + ref
    return f"{original_name}_{data_type}_ref"


class AdaptedMCPToolWrapper(BaseTool):
    """Wraps an MCPToolWrapper to use Aevom storage references for heavy data.

    This adapter intercepts the tool's schema and execution flow:
    - Schema: Heavy-data input parameters are replaced with _ref parameters
      annotated with x-aevom-reference.
    - Execute: Reference arguments are resolved (data retrieved from storage)
      before calling the underlying MCP tool with raw data.
    - Output: If heavy data is detected in the tool response, it is stored
      and a lightweight reference is returned to the LLM instead.

    Attributes:
        inner: The original MCPToolWrapper being adapted.
        input_mappings: Mapping of original param name -> DetectionResult.
        output_mappings: List of expected heavy data types in tool output.
    """

    def __init__(
        self,
        inner: "MCPToolWrapper",
        input_mappings: list[DetectionResult],
        output_mappings: list[DetectionResult],
    ):
        """Initialize the adapted wrapper.

        Args:
            inner: The original MCPToolWrapper to wrap.
            input_mappings: Detection results for heavy input parameters.
            output_mappings: Detection results for expected heavy output.
        """
        self._inner = inner
        self._input_mappings = {m.param_name: m for m in input_mappings}
        self._output_mappings = output_mappings

        # Pre-compute the ref param name -> original param name mapping
        self._ref_to_original: dict[str, str] = {}
        for orig_name, detection in self._input_mappings.items():
            ref_name = _ref_param_name(orig_name, detection.detected_type)
            self._ref_to_original[ref_name] = orig_name

        # Detector for runtime output sniffing
        self._detector = HeavyDataDetector()

    @property
    def name(self) -> str:
        """Tool name -- delegates to the inner wrapper."""
        return self._inner.name

    @property
    def original_name(self) -> str:
        """Original MCP tool name (without prefix)."""
        return self._inner.original_name

    @property
    def tool_type(self) -> ToolType:
        """Tool type -- MCP server."""
        return ToolType.MCP_SERVER

    @property
    def data_io_category(self) -> DataIOCategory:
        """Computed from input/output mappings."""
        has_inputs = bool(self._input_mappings)
        has_outputs = bool(self._output_mappings)

        if has_inputs and has_outputs:
            return DataIOCategory.BOTH
        elif has_inputs:
            return DataIOCategory.INPUT_RETRIEVES
        elif has_outputs:
            return DataIOCategory.OUTPUT_STORES
        return DataIOCategory.TEXT_ONLY

    @property
    def server_id(self) -> str:
        """MCP server ID from inner wrapper."""
        return self._inner.server_id

    @property
    def server_name(self) -> str:
        """MCP server name from inner wrapper."""
        return self._inner.server_name

    @property
    def summary_template(self) -> str:
        """Summary template for UI display."""
        return self._inner.summary_template

    @property
    def adapted(self) -> bool:
        """Flag indicating this tool has been adapted for heavy data."""
        return True

    @property
    def adaptation_info(self) -> dict[str, Any]:
        """Structured info about the adaptations applied."""
        return {
            "input_mappings": {
                name: {
                    "detected_type": det.detected_type,
                    "confidence": det.confidence,
                    "ref_param": _ref_param_name(name, det.detected_type),
                }
                for name, det in self._input_mappings.items()
            },
            "output_mappings": [
                {
                    "detected_type": det.detected_type,
                    "confidence": det.confidence,
                }
                for det in self._output_mappings
            ],
        }

    def get_schema(self) -> dict[str, Any]:
        """Get modified OpenAI-compatible schema with reference parameters.

        Replaces detected heavy-data input parameters with _ref parameters
        annotated with x-aevom-reference. The rest of the schema is preserved.
        """
        schema = self._inner.get_schema()
        schema = copy.deepcopy(schema)

        func = schema.get("function", {})
        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = list(params.get("required", []))

        for orig_name, detection in self._input_mappings.items():
            ref_name = _ref_param_name(orig_name, detection.detected_type)
            type_desc = friendly_name(detection.detected_type)

            # Remove original parameter
            original_prop = properties.pop(orig_name, {})

            # Create reference parameter
            properties[ref_name] = {
                "type": "string",
                "description": (
                    f"Reference to stored {type_desc} "
                    f"(e.g., '{detection.detected_type}_abc123'). "
                    f"Use a stored reference from a previous tool call or "
                    f"from python_execute with aevom.store_*()."
                ),
                "x-aevom-reference": {"data_type": detection.detected_type},
            }

            # Update required list
            if orig_name in required:
                required.remove(orig_name)
                required.append(ref_name)

        params["properties"] = properties
        params["required"] = required
        func["parameters"] = params

        # Add adaptation note to description
        original_desc = func.get("description", "")
        adapted_types = [d.detected_type for d in self._input_mappings.values()]
        if self._output_mappings:
            adapted_types.extend(
                f"{d.detected_type} (output)" for d in self._output_mappings
            )
        if adapted_types:
            func["description"] = (
                f"{original_desc}\n\n"
                f"[Adapted: uses storage references for {', '.join(adapted_types)}]"
            )

        schema["function"] = func
        return schema

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the MCP tool with reference unpacking and output storage.

        1. For each adapted input parameter: retrieve raw data from storage
           using the reference, then substitute into the arguments dict.
        2. Call the underlying MCP tool with raw data arguments.
        3. If heavy output is expected or detected: store it and return a
           data reference instead of raw content.

        Args:
            arguments: Arguments from the LLM (using reference params).

        Returns:
            ToolResult -- either the raw result or a data_reference if
            heavy output was detected/stored.
        """
        # 1. Unpack reference arguments to raw data
        translated_args = dict(arguments)

        for ref_name, orig_name in self._ref_to_original.items():
            ref_value = translated_args.pop(ref_name, None)
            if ref_value is None:
                # Check if the param is required
                continue

            detection = self._input_mappings[orig_name]

            if self._storage is None:
                return ToolResult.error(
                    f"Storage service not available. Cannot retrieve "
                    f"{detection.detected_type} data for reference '{ref_value}'."
                )

            try:
                content_bytes = await self._storage.retrieve(ref_value)
                raw_data = content_bytes.decode("utf-8")
                translated_args[orig_name] = raw_data
                logger.info(
                    f"Adapted input: unpacked {ref_name}='{ref_value}' -> "
                    f"{orig_name} ({len(raw_data)} chars of {detection.detected_type})"
                )
            except ArtifactNotFoundError:
                return ToolResult.error(
                    f"Data not found for reference '{ref_value}'. "
                    f"Ensure you have a valid {detection.detected_type} reference "
                    f"from a previous tool call."
                )
            except Exception as e:
                logger.exception(f"Failed to retrieve data for {ref_value}: {e}")
                return ToolResult.error(
                    f"Failed to retrieve {detection.detected_type} data "
                    f"for reference '{ref_value}': {str(e)}"
                )

        # 2. Call the underlying MCP tool with translated arguments
        result = await self._inner.execute(translated_args)

        # 3. Handle heavy output -- store if detected
        if result.content_type.value == "error":
            return result

        return await self._maybe_store_output(result)

    async def _maybe_store_output(self, result: ToolResult) -> ToolResult:
        """Check if tool output contains heavy data and store it if so.

        Uses two strategies:
        - Pre-declared output mappings (from description analysis at add time)
        - Runtime content sniffing (regex on actual output)

        For SMILES output, individual SMILES strings are extracted and stored
        separately so each can be used independently by downstream tools.

        Args:
            result: The raw ToolResult from the MCP tool.

        Returns:
            Either the original result or a data_reference result.
        """
        content = result.to_tool_response()
        if not content or len(content) < 20:
            return result

        # Try runtime content detection
        content_detections = self._detector.detect_content(content)

        # Merge with pre-declared output mappings, preferring content detections
        output_type: str | None = None

        if content_detections:
            # Use the highest-confidence content detection
            best = max(content_detections, key=lambda d: d.confidence)
            output_type = best.detected_type
        elif self._output_mappings:
            # Fall back to description-based prediction
            output_type = self._output_mappings[0].detected_type

        if output_type is None:
            return result

        # Only store if we have storage available
        if self._storage is None:
            logger.warning(
                "Heavy output detected but storage not available -- "
                "passing raw content through"
            )
            return result

        # SMILES gets special multi-value handling
        if output_type == "smiles":
            return await self._store_smiles_output(content)

        # All other types: store as a single artifact
        return await self._store_single_output(content, output_type)

    async def _store_smiles_output(self, content: str) -> ToolResult:
        """Extract and store individual SMILES from tool output.

        Each SMILES is stored as its own artifact so downstream tools
        (e.g., generate_3d_coordinates) can reference them individually.

        Args:
            content: Raw text output containing one or more SMILES.

        Returns:
            ToolResult with reference(s) to stored SMILES data.
        """
        smiles_list = extract_smiles_from_content(content)

        if not smiles_list:
            # Detection said SMILES but extractor found nothing -- pass through
            return ToolResult.text(content)

        tool_name = self._inner.original_name
        server_name = self._inner.server_name
        references: list[str] = []

        try:
            for i, smi in enumerate(smiles_list):
                ref = await self._storage.store(
                    data_type="smiles",
                    content=smi.encode("utf-8"),
                    metadata={
                        "source": "mcp_adapter",
                        "mcp_tool": tool_name,
                        "mcp_server": server_name,
                        "index": i,
                        "batch_size": len(smiles_list),
                    },
                    description=f"SMILES {i + 1}/{len(smiles_list)} from '{tool_name}'",
                )
                references.append(ref)

            logger.info(
                f"Adapted output: stored {len(references)} SMILES individually "
                f"from MCP tool '{tool_name}' -> {references}"
            )

            # Build a summary message listing all references
            ref_lines = [f"  - '{ref}'" for ref in references]
            message = (
                f"'{tool_name}' returned {len(references)} SMILES string(s), "
                f"each stored individually:\n"
                + "\n".join(ref_lines)
                + "\n\nUse any reference with tools that accept smiles data "
                f"(e.g., generate_3d_coordinates, molecule_info)."
            )

            # Return the first reference as the primary data, with all refs in metadata
            return ToolResult.data_reference(
                reference=references[0],
                data_type="smiles",
                message=message,
                metadata={
                    "source_tool": tool_name,
                    "source_server": server_name,
                    "all_references": references,
                    "count": len(references),
                },
            )

        except Exception as e:
            logger.exception(f"Failed to store SMILES output: {e}")
            return ToolResult.text(content)

    async def _store_single_output(self, content: str, output_type: str) -> ToolResult:
        """Store tool output as a single artifact.

        Args:
            content: Raw text output.
            output_type: Aevom data type (e.g., "sdf", "pdb").

        Returns:
            ToolResult.data_reference on success, original text on failure.
        """
        tool_name = self._inner.original_name
        server_name = self._inner.server_name

        try:
            reference = await self._storage.store(
                data_type=output_type,
                content=content.encode("utf-8"),
                metadata={
                    "source": "mcp_adapter",
                    "mcp_tool": tool_name,
                    "mcp_server": server_name,
                },
                description=f"Output from MCP tool '{tool_name}'",
            )

            logger.info(
                f"Adapted output: stored {output_type} data as '{reference}' "
                f"({len(content)} chars from MCP tool '{tool_name}')"
            )

            type_desc = friendly_name(output_type)
            return ToolResult.data_reference(
                reference=reference,
                data_type=output_type,
                message=(
                    f"{type_desc.capitalize()} output from '{tool_name}' "
                    f"stored as '{reference}'. Use this reference with other tools that "
                    f"accept {output_type} data."
                ),
                metadata={
                    "source_tool": tool_name,
                    "source_server": server_name,
                },
            )

        except Exception as e:
            logger.exception(f"Failed to store heavy output: {e}")
            return ToolResult.text(content)


def adapt_mcp_tools(
    tools: list["MCPToolWrapper"],
    detector: HeavyDataDetector | None = None,
) -> list[Union["MCPToolWrapper", AdaptedMCPToolWrapper]]:
    """Post-process MCP tools to detect and adapt heavy data types.

    Examines each tool's schema for heavy data types. Tools with detected
    heavy data are wrapped in AdaptedMCPToolWrapper; others pass through
    unchanged. Both implement BaseTool, so the rest of the system is
    transparent.

    Args:
        tools: List of MCPToolWrapper instances from create_mcp_tools().
        detector: Optional custom detector. Defaults to HeavyDataDetector().

    Returns:
        List mixing MCPToolWrapper (no heavy data) and AdaptedMCPToolWrapper
        (heavy data detected). All implement BaseTool.
    """
    import asyncio

    if detector is None:
        detector = HeavyDataDetector()

    adapted_tools: list[Union["MCPToolWrapper", AdaptedMCPToolWrapper]] = []

    for tool in tools:
        schema = tool.get_schema()
        func = schema.get("function", {})
        input_schema = func.get("parameters", {})
        description = func.get("description", "")
        tool_name = tool.original_name

        # Run detection (synchronous wrapper for the async detector)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # We're already in an async context -- create a task
            import concurrent.futures

            input_detections, output_detections = _run_detection_sync(
                detector, tool_name, description, input_schema
            )
        else:
            input_detections, output_detections = asyncio.run(
                detector.analyze_tool(tool_name, description, input_schema)
            )

        if input_detections or output_detections:
            adapted = AdaptedMCPToolWrapper(
                inner=tool,
                input_mappings=input_detections,
                output_mappings=output_detections,
            )
            adapted_tools.append(adapted)
            logger.info(
                f"Adapted MCP tool '{tool_name}': "
                f"{len(input_detections)} input mapping(s), "
                f"{len(output_detections)} output mapping(s)"
            )
        else:
            adapted_tools.append(tool)
            logger.debug(f"MCP tool '{tool_name}': no heavy data detected, passing through")

    return adapted_tools


async def adapt_mcp_tools_async(
    tools: list["MCPToolWrapper"],
    detector: HeavyDataDetector | None = None,
) -> list[Union["MCPToolWrapper", AdaptedMCPToolWrapper]]:
    """Async version of adapt_mcp_tools for use in async contexts.

    Args:
        tools: List of MCPToolWrapper instances from create_mcp_tools().
        detector: Optional custom detector. Defaults to HeavyDataDetector().

    Returns:
        List mixing MCPToolWrapper and AdaptedMCPToolWrapper.
    """
    if detector is None:
        detector = HeavyDataDetector()

    adapted_tools: list[Union["MCPToolWrapper", AdaptedMCPToolWrapper]] = []

    for tool in tools:
        schema = tool.get_schema()
        func = schema.get("function", {})
        input_schema = func.get("parameters", {})
        description = func.get("description", "")
        tool_name = tool.original_name

        input_detections, output_detections = await detector.analyze_tool(
            tool_name, description, input_schema
        )

        if input_detections or output_detections:
            adapted = AdaptedMCPToolWrapper(
                inner=tool,
                input_mappings=input_detections,
                output_mappings=output_detections,
            )
            adapted_tools.append(adapted)
            logger.info(
                f"Adapted MCP tool '{tool_name}': "
                f"{len(input_detections)} input mapping(s), "
                f"{len(output_detections)} output mapping(s)"
            )
        else:
            adapted_tools.append(tool)
            logger.debug(f"MCP tool '{tool_name}': no heavy data detected, passing through")

    return adapted_tools


def _run_detection_sync(
    detector: HeavyDataDetector,
    tool_name: str,
    description: str,
    input_schema: dict[str, Any],
) -> tuple[list[DetectionResult], list[DetectionResult]]:
    """Run detection synchronously when already inside an event loop.

    Uses a new event loop in a thread to avoid nested loop issues.
    """
    import asyncio
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(
            asyncio.run,
            detector.analyze_tool(tool_name, description, input_schema),
        )
        return future.result(timeout=5.0)
