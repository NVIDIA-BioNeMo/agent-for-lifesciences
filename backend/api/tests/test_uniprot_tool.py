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

"""Tests for UniProt fetch sequence tool.

Tests cover:
1. Tool instantiation and schema validation
2. Full tool execution with mock storage
3. Error handling for invalid inputs
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from api.tools.uniprot import UniProtFetchSequenceTool


class TestUniProtToolSchema:
    """Tests for tool schema correctness."""

    def test_schema_structure(self):
        """Test that the tool schema is correctly formatted."""
        tool = UniProtFetchSequenceTool()
        schema = tool.get_schema()

        assert schema["type"] == "function"
        assert "function" in schema

        func = schema["function"]
        assert func["name"] == "fetch_protein_sequence"
        assert "description" in func
        assert "parameters" in func

        params = func["parameters"]
        assert params["type"] == "object"
        assert "gene_name" in params["properties"]
        assert "organism_id" in params["properties"]
        assert "gene_name" in params["required"]

    def test_tool_metadata(self):
        """Test tool name, type, and data IO category."""
        tool = UniProtFetchSequenceTool()
        assert tool.name == "fetch_protein_sequence"
        assert tool.data_io_category is not None


class TestUniProtToolExecution:
    """Tests for tool execution with mock storage."""

    @pytest.fixture
    def tool_with_storage(self):
        """Create a UniProt tool with mock storage."""
        tool = UniProtFetchSequenceTool()
        mock_storage = MagicMock()
        mock_storage.store = AsyncMock(return_value="sequence_test123")
        tool.set_storage(mock_storage)
        return tool, mock_storage

    @pytest.mark.asyncio
    async def test_fetch_pgk1(self, tool_with_storage):
        """Test fetching PGK1 protein sequence."""
        tool, mock_storage = tool_with_storage

        result = await tool.execute({"gene_name": "PGK1"})

        assert result.content_type.value != "error", f"Tool returned error: {result.data}"
        assert result.content_type.value == "data_reference"
        assert result.data == "sequence_test123"

        # Verify storage was called correctly
        mock_storage.store.assert_called_once()
        call_args = mock_storage.store.call_args
        assert call_args.kwargs["data_type"] == "sequence"
        assert b"M" in call_args.kwargs["content"]  # Most proteins start with M

        metadata = call_args.kwargs["metadata"]
        assert metadata["gene_name"] == "PGK1"
        assert metadata["organism_id"] == "9606"
        assert "primary_accession" in metadata
        assert "sequence_length" in metadata

    @pytest.mark.asyncio
    async def test_fetch_brca1(self, tool_with_storage):
        """Test fetching BRCA1 protein sequence."""
        tool, mock_storage = tool_with_storage
        mock_storage.store.return_value = "sequence_brca456"

        result = await tool.execute({"gene_name": "BRCA1"})

        assert result.content_type.value == "data_reference"
        metadata = mock_storage.store.call_args.kwargs["metadata"]
        assert "primary_accession" in metadata
        assert "sequence_length" in metadata


class TestUniProtToolErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_empty_gene_name(self):
        """Test error handling for empty gene name."""
        tool = UniProtFetchSequenceTool()
        result = await tool.execute({"gene_name": ""})
        assert result.content_type.value == "error"

    @pytest.mark.asyncio
    async def test_unknown_gene(self):
        """Test error handling for unknown gene."""
        tool = UniProtFetchSequenceTool()
        mock_storage = MagicMock()
        mock_storage.store = AsyncMock()
        tool.set_storage(mock_storage)

        result = await tool.execute({"gene_name": "NOTAREALGENE12345"})
        assert result.content_type.value == "error"
        assert "No protein found" in result.data

    @pytest.mark.asyncio
    async def test_missing_storage(self):
        """Test error handling when storage service is not available."""
        tool = UniProtFetchSequenceTool()
        tool._storage = None
        result = await tool.execute({"gene_name": "PGK1"})
        assert result.content_type.value == "error"
        assert "Storage service not available" in result.data
