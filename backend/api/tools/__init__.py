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

"""Tool definitions for Aevom AI agent.

This module provides:
- BaseTool: Abstract interface for all tools
- ToolRegistry: Central registry for tool discovery and execution
- DataIOCategory: Classification for tool data storage behavior
- Individual tool implementations (RDKit tools, etc.)
"""

from .base import (
    BaseTool,
    ContainerTool,
    ContentType,
    DataIOCategory,
    NvidiaApiTool,
    ToolResult,
    ToolType,
)
from .registry import ToolRegistry, get_registry
from .rdkit import RDKitMoleculeInfoTool
from .safe_notation import SafeToSmilesTool, SmilesToSafeTool

__all__ = [
    # Base classes
    "BaseTool",
    "ContainerTool",
    "NvidiaApiTool",
    "ToolType",
    "ContentType",
    "DataIOCategory",
    "ToolResult",
    # Registry
    "ToolRegistry",
    "get_registry",
    # RDKit tools
    "RDKitMoleculeInfoTool",
    # SAFE notation tools
    "SmilesToSafeTool",
    "SafeToSmilesTool",
]
