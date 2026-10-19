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

"""MCP HTTP+SSE client for connecting to external MCP servers.

This client implements the MCP protocol over HTTP with SSE transport,
supporting tool discovery (tools/list) and tool execution (tools/call).
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from api.bounded_http import ResponseTooLargeError, format_limit, request_bounded

logger = logging.getLogger(__name__)


class MCPConnectionError(Exception):
    """Raised when connection to MCP server fails."""
    pass


class MCPAuthError(Exception):
    """Raised when MCP server authentication fails."""
    pass


class MCPToolCallError(Exception):
    """Raised when an MCP tool call fails."""
    pass


class MCPRedirectError(MCPConnectionError):
    """Raised when an MCP server answers with a redirect, which is not followed."""
    pass


class MCPResponseTooLargeError(MCPToolCallError):
    """Raised when an MCP server's response exceeds the size limit."""
    pass


@dataclass
class MCPToolSchema:
    """Schema for an MCP tool.
    
    Attributes:
        name: Unique tool name.
        description: Human-readable description.
        input_schema: JSON Schema for tool parameters.
    """
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    
    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible function schema.
        
        Returns:
            Dict matching OpenAI function calling format.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            }
        }


class MCPClient:
    """Client for MCP servers using HTTP+SSE transport.
    
    Implements the streamable-http transport protocol for MCP,
    supporting tool discovery and execution.
    
    Example:
        client = MCPClient("https://mcp.example.com", api_key="...")
        await client.connect()
        tools = await client.list_tools()
        result = await client.call_tool("search", {"query": "hello"})
    """
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """Initialize MCP client.
        
        Args:
            url: Base URL of the MCP server.
            api_key: Optional API key for authentication.
            timeout: Request timeout in seconds.
        """
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._session_id: Optional[str] = None
        self._tools: list[MCPToolSchema] = []
        self._connected = False
        self._server_info: dict[str, Any] = {}
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected to the MCP server."""
        return self._connected
    
    @property
    def tools(self) -> list[MCPToolSchema]:
        """Get list of discovered tools."""
        return self._tools
    
    @property
    def server_info(self) -> dict[str, Any]:
        """Get server information from initialization."""
        return self._server_info
    
    def _get_headers(self) -> dict[str, str]:
        """Build request headers including auth if configured."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers
    
    async def _send_jsonrpc(
        self,
        method: str,
        params: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Send a JSON-RPC request to the MCP server.
        
        Args:
            method: JSON-RPC method name.
            params: Optional parameters for the method.
            
        Returns:
            JSON-RPC result on success.
            
        Raises:
            MCPConnectionError: On connection failure.
            MCPToolCallError: On JSON-RPC error response.
        """
        request_id = str(uuid.uuid4())
        payload = {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        
        logger.debug(f"MCP request: {method} -> {self.url}")
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
                response = await request_bounded(
                    client,
                    "POST",
                    self.url,
                    headers=self._get_headers(),
                    json=payload,
                )
                
                # Check for session ID in response headers
                if "Mcp-Session-Id" in response.headers:
                    self._session_id = response.headers["Mcp-Session-Id"]
                
                if response.is_redirect:
                    raise MCPRedirectError(
                        f"MCP server at {self.url} redirected to "
                        f"{response.headers.get('location', 'another URL')}. Aevom "
                        "does not follow redirects; register the final URL instead."
                    )
                
                if response.status_code == 401:
                    raise MCPAuthError(f"Authentication failed: {response.text}")
                
                if response.status_code == 404:
                    raise MCPConnectionError(f"MCP endpoint not found at {self.url}")
                
                response.raise_for_status()
                
                # Handle SSE response (some MCP servers use SSE for all responses)
                content_type = response.headers.get("content-type", "")
                if "text/event-stream" in content_type:
                    return self._parse_sse_response(response.text, request_id)
                
                result = response.json()
                
        except ResponseTooLargeError as e:
            raise MCPResponseTooLargeError(
                f"MCP server at {self.url} returned a response larger than Aevom's "
                f"{format_limit(e.limit)} limit, so it was discarded. Retry with a "
                "request that returns less data, for example a narrower query or "
                "fewer results."
            ) from e
        except httpx.ConnectError as e:
            raise MCPConnectionError(f"Failed to connect to MCP server: {e}")
        except httpx.TimeoutException as e:
            raise MCPConnectionError(f"MCP server request timed out: {e}")
        except httpx.HTTPStatusError as e:
            raise MCPConnectionError(f"MCP server returned error: {e}")
        
        # Check for JSON-RPC error
        if "error" in result:
            error = result["error"]
            raise MCPToolCallError(
                f"MCP error {error.get('code', 'unknown')}: {error.get('message', 'Unknown error')}"
            )
        
        return result.get("result", {})
    
    def _parse_sse_response(self, sse_data: str, request_id: str) -> dict[str, Any]:
        """Parse SSE response format from MCP server.
        
        Args:
            sse_data: Raw SSE response data.
            request_id: Expected request ID for matching.
            
        Returns:
            Parsed result from the SSE event.
        """
        logger.debug(f"Parsing SSE response (expecting id={request_id}): {sse_data[:500]}")
        
        # Parse SSE events - look for data lines
        for line in sse_data.split("\n"):
            line = line.strip()
            if line.startswith("data:"):
                try:
                    data = json.loads(line[5:].strip())
                    logger.debug(f"Parsed SSE data: id={data.get('id')}, has_result={('result' in data)}")
                    
                    # Check for errors first
                    if "error" in data:
                        error = data["error"]
                        raise MCPToolCallError(
                            f"MCP error: {error.get('message', 'Unknown error')}"
                        )
                    
                    # Accept response if ID matches OR if it's a valid JSON-RPC response
                    # (some servers may not echo the exact ID format)
                    if data.get("id") == request_id or "result" in data:
                        return data.get("result", {})
                        
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse SSE line as JSON: {e}")
                    continue
        
        logger.warning(f"No valid JSON-RPC result found in SSE response")
        return {}
    
    async def connect(self) -> dict[str, Any]:
        """Connect to the MCP server and initialize session.
        
        Sends the 'initialize' request to establish the MCP session
        and discover server capabilities. Automatically tries common
        MCP endpoint paths if the initial URL fails.
        
        Returns:
            Server information dict.
            
        Raises:
            MCPConnectionError: On connection failure.
            MCPAuthError: On authentication failure.
        """
        logger.info(f"Connecting to MCP server: {self.url}")
        
        params = {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
            },
            "clientInfo": {
                "name": "Aevom",
                "version": "1.0.0",
            }
        }
        
        # Try common MCP endpoint paths
        original_url = self.url
        paths_to_try = ["", "/mcp", "/sse", "/api/mcp"]
        last_error = None
        # A redirect names the URL the user should have registered, which is
        # more useful than whatever generic failure a later path produces.
        redirect_error: Optional[MCPRedirectError] = None
        
        for path in paths_to_try:
            if path:
                self.url = original_url.rstrip("/") + path
            else:
                self.url = original_url
            
            logger.info(f"Trying MCP endpoint: {self.url}")
            
            try:
                result = await self._send_jsonrpc("initialize", params)
                
                logger.info(f"Initialize response from {self.url}: protocolVersion={result.get('protocolVersion')}, serverInfo={bool(result.get('serverInfo'))}")
                
                # Check if we got a valid MCP response
                if result.get("protocolVersion") or result.get("serverInfo"):
                    self._server_info = result
                    self._connected = True
                    
                    # Send initialized notification (don't fail if this errors)
                    try:
                        await self._send_jsonrpc("notifications/initialized")
                    except Exception as notif_err:
                        logger.debug(f"notifications/initialized returned error (this is often OK): {notif_err}")
                    
                    logger.info(
                        f"Connected to MCP server: {result.get('serverInfo', {}).get('name', 'Unknown')} "
                        f"at {self.url}"
                    )
                    
                    return result
                else:
                    # Got a response but not valid MCP - try next path
                    logger.debug(f"Invalid MCP response from {self.url}: {result}")
                    last_error = MCPConnectionError(f"Invalid MCP response from {self.url}")
                    continue
                    
            except (MCPAuthError, MCPResponseTooLargeError):
                # Auth error means we found the right endpoint, just wrong credentials.
                # An oversized answer came from a real server; other paths won't shrink it.
                raise
            except MCPRedirectError as e:
                redirect_error = redirect_error or e
                last_error = e
                logger.info(f"MCP endpoint {self.url} failed: {e}")
                continue
            except Exception as e:
                last_error = e
                logger.info(f"MCP endpoint {self.url} failed: {e}")
                continue
        
        # All paths failed
        self._connected = False
        self.url = original_url  # Restore original URL
        if redirect_error is not None:
            last_error = redirect_error
        logger.error(f"Failed to connect to MCP server at {original_url}: {last_error}")
        raise MCPConnectionError(f"Failed to connect to MCP server: {last_error}")
    
    async def list_tools(self) -> list[MCPToolSchema]:
        """Discover available tools from the MCP server.
        
        Returns:
            List of MCPToolSchema objects describing available tools.
            
        Raises:
            MCPConnectionError: If not connected.
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server. Call connect() first.")
        
        logger.info(f"Listing tools from MCP server: {self.url}")
        
        result = await self._send_jsonrpc("tools/list")
        
        tools = []
        for tool_data in result.get("tools", []):
            tool = MCPToolSchema(
                name=tool_data.get("name", ""),
                description=tool_data.get("description", ""),
                input_schema=tool_data.get("inputSchema", {}),
            )
            tools.append(tool)
            logger.debug(f"Discovered MCP tool: {tool.name}")
        
        self._tools = tools
        logger.info(f"Discovered {len(tools)} tools from MCP server")
        
        return tools
    
    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute a tool on the MCP server.
        
        Args:
            name: Name of the tool to execute.
            arguments: Arguments to pass to the tool.
            
        Returns:
            Tool execution result.
            
        Raises:
            MCPConnectionError: If not connected.
            MCPToolCallError: If tool execution fails.
        """
        if not self._connected:
            raise MCPConnectionError("Not connected to MCP server. Call connect() first.")
        
        logger.info(f"Calling MCP tool: {name}")
        
        params = {
            "name": name,
            "arguments": arguments,
        }
        
        result = await self._send_jsonrpc("tools/call", params)
        
        # MCP returns content array, extract text content
        content_parts = result.get("content", [])
        text_parts = []
        
        for part in content_parts:
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
            elif part.get("type") == "image":
                # Handle image content
                text_parts.append(f"[Image: {part.get('mimeType', 'image')}]")
            elif part.get("type") == "resource":
                # Handle embedded resource
                text_parts.append(f"[Resource: {part.get('uri', 'unknown')}]")
        
        return {
            "content": "\n".join(text_parts) if text_parts else str(result),
            "raw": result,
            "isError": result.get("isError", False),
        }
    
    async def disconnect(self) -> None:
        """Disconnect from the MCP server.
        
        Sends a clean shutdown notification if connected.
        """
        if self._connected:
            logger.info(f"Disconnecting from MCP server: {self.url}")
            self._connected = False
            self._session_id = None
            self._tools = []
