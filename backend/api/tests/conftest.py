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

"""Shared test fixtures for backend API tests.

Provides mock implementations of core services (storage, registry, LLM client)
that can be reused across test modules.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_storage():
    """Create a mock DataStorageService with common async methods stubbed."""
    storage = MagicMock()
    storage.store = AsyncMock(return_value="sdf_test1234")
    storage.retrieve = AsyncMock(return_value=b"mock data")
    storage.retrieve_with_metadata = AsyncMock(
        return_value=(b"mock data", {"session_id": "test"})
    )
    storage.delete = AsyncMock(return_value=True)
    storage.exists = AsyncMock(return_value=True)
    storage.get_data_type = MagicMock(return_value="sdf")
    storage.list_artifacts = AsyncMock(
        return_value=[
            {
                "reference": "sdf_test1234",
                "data_type": "sdf",
                "metadata": {"session_id": "sess_abc"},
            },
        ]
    )
    return storage


@pytest.fixture
def mock_registry():
    """Create a mock ToolRegistry with execute stubbed."""
    from api.tools.base import ToolResult

    registry = MagicMock()
    registry.execute = AsyncMock(
        return_value=ToolResult.text("Mock tool result")
    )
    registry.get_all_schemas = AsyncMock(return_value=[])
    registry.get_tool = MagicMock(return_value=None)
    return registry


@pytest.fixture
def mock_llm_client():
    """Create a mock LLMClient."""
    client = MagicMock()
    client.render_system_prompt = MagicMock(return_value="rendered prompt")
    client.complete = AsyncMock(return_value=MagicMock(
        content="Hello! How can I help?",
        tool_calls=None,
        reasoning=None,
    ))
    return client
