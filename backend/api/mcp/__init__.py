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

"""MCP (Model Context Protocol) client module for Aevom.

This module provides integration with external MCP servers, allowing
dynamic tool discovery and execution from remote MCP-compatible services.

Includes automatic detection and adaptation of heavy chem/bio data types
(SMILES, SDF, PDB, sequences, etc.) to use Aevom's lightweight storage
references instead of passing raw data through the LLM.
"""

from .client import MCPClient, MCPConnectionError, MCPToolSchema
from .wrapper import MCPToolWrapper
from .adapter import AdaptedMCPToolWrapper, adapt_mcp_tools, adapt_mcp_tools_async
from .detector import (
    BaseDetector,
    DetectionResult,
    HeavyDataDetector,
    HeavyTypePattern,
    RegexDetector,
    extract_smiles_from_content,
)
from .session_store import MCPSessionStore, get_mcp_session_store
from .routes import router as mcp_router

__all__ = [
    # Client
    "MCPClient",
    "MCPConnectionError",
    "MCPToolSchema",
    # Wrappers
    "MCPToolWrapper",
    "AdaptedMCPToolWrapper",
    # Detector
    "BaseDetector",
    "RegexDetector",
    "HeavyDataDetector",
    "HeavyTypePattern",
    "DetectionResult",
    "extract_smiles_from_content",
    # Adapter
    "adapt_mcp_tools",
    "adapt_mcp_tools_async",
    # Session
    "MCPSessionStore",
    "get_mcp_session_store",
    # Routes
    "mcp_router",
]
