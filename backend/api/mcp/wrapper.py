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

"""MCP tool wrapper that adapts MCP tools to Aevom's BaseTool interface.

This wrapper allows MCP tools to be registered and executed through
Aevom's standard tool registry system.
"""

import logging
from typing import TYPE_CHECKING, Any

from api.tools.base import BaseTool, DataIOCategory, ToolResult, ToolType

if TYPE_CHECKING:
    from .client import MCPClient, MCPToolSchema

logger = logging.getLogger(__name__)


class MCPToolWrapper(BaseTool):
    """Wrapper that adapts an MCP tool to Aevom's BaseTool interface.
    
    This allows MCP tools discovered from remote servers to be used
    seamlessly within Aevom's tool calling infrastructure.
    
    Attributes:
        mcp_client: The MCPClient instance for executing the tool.
        schema: The MCPToolSchema describing the tool.
        server_id: Unique identifier for the MCP server.
        server_name: Human-readable name of the MCP server.
    """
    
    def __init__(
        self,
        mcp_client: "MCPClient",
        schema: "MCPToolSchema",
        server_id: str,
        server_name: str,
    ):
        """Initialize MCP tool wrapper.
        
        Args:
            mcp_client: Client for communicating with the MCP server.
            schema: Tool schema from the MCP server.
            server_id: Unique ID for the MCP server connection.
            server_name: Display name for the MCP server.
        """
        self._mcp_client = mcp_client
        self._schema = schema
        self._server_id = server_id
        self._server_name = server_name
    
    @property
    def name(self) -> str:
        """Tool name prefixed with server ID to ensure uniqueness."""
        # Prefix with server ID to avoid name collisions
        return f"mcp_{self._server_id}_{self._schema.name}"
    
    @property
    def original_name(self) -> str:
        """Original tool name from MCP server (without prefix)."""
        return self._schema.name
    
    @property
    def tool_type(self) -> ToolType:
        """MCP tools use the MCP_SERVER type."""
        return ToolType.MCP_SERVER
    
    @property
    def data_io_category(self) -> DataIOCategory:
        """MCP tools are treated as text-only by default."""
        return DataIOCategory.TEXT_ONLY
    
    @property
    def server_id(self) -> str:
        """Get the MCP server ID this tool belongs to."""
        return self._server_id
    
    @property
    def server_name(self) -> str:
        """Get the human-readable server name."""
        return self._server_name
    
    @property
    def summary_template(self) -> str:
        """Summary template for UI display."""
        return f"Calling {self._schema.name} on {self._server_name}..."
    
    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema.
        
        Returns:
            Dict matching OpenAI function calling format.
        """
        # Use prefixed name for uniqueness but keep description clear
        schema = self._schema.to_openai_schema()
        schema["function"]["name"] = self.name
        
        # Enhance description with server info
        original_desc = schema["function"].get("description", "")
        schema["function"]["description"] = (
            f"[MCP: {self._server_name}] {original_desc}"
        )
        
        return schema
    
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the MCP tool.
        
        Args:
            arguments: Arguments to pass to the tool.
            
        Returns:
            ToolResult with the execution result.
        """
        logger.info(
            f"Executing MCP tool: {self._schema.name} on server {self._server_name}"
        )
        
        try:
            # Call the tool via MCP client
            result = await self._mcp_client.call_tool(
                self._schema.name,
                arguments,
            )
            
            # Check for error response
            if result.get("isError", False):
                return ToolResult.error(
                    f"MCP tool error: {result.get('content', 'Unknown error')}"
                )
            
            # Return text result
            content = result.get("content", "")
            if not content:
                content = "Tool executed successfully (no output)"
            
            return ToolResult.text(content)
            
        except Exception as e:
            logger.exception(f"MCP tool execution failed: {e}")
            return ToolResult.error(f"MCP tool execution failed: {str(e)}")


def create_mcp_tools(
    mcp_client: "MCPClient",
    schemas: list["MCPToolSchema"],
    server_id: str,
    server_name: str,
) -> list[MCPToolWrapper]:
    """Create MCPToolWrapper instances for all tools from an MCP server.
    
    Args:
        mcp_client: Connected MCP client.
        schemas: List of tool schemas from the server.
        server_id: Unique ID for the server.
        server_name: Display name for the server.
        
    Returns:
        List of MCPToolWrapper instances.
    """
    tools = []
    for schema in schemas:
        wrapper = MCPToolWrapper(
            mcp_client=mcp_client,
            schema=schema,
            server_id=server_id,
            server_name=server_name,
        )
        tools.append(wrapper)
        logger.debug(f"Created MCP tool wrapper: {wrapper.name}")
    
    return tools
