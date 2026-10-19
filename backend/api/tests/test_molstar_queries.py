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

"""Tests for ``MolstarQueryTool`` (the ``structure_query`` tool).

Covers the new explicit ``pdb_ref`` parameter introduced by Plan A:
schema correctness and that ``execute()`` carries ``pdb_ref`` through
``result.data`` so downstream wrappers (training, production graph)
can route the query against the right structure.
"""

import pytest

from api.tools.base import ContentType
from api.tools.molstar_queries import MolstarQueryTool


@pytest.fixture
def tool() -> MolstarQueryTool:
    return MolstarQueryTool()


class TestSchema:
    def test_pdb_ref_is_required(self, tool):
        schema = tool.get_schema()
        params = schema["function"]["parameters"]
        assert "pdb_ref" in params["required"]
        assert "molql" in params["required"]
        assert "pdb_ref" in params["properties"]

    def test_pdb_ref_is_aevom_reference(self, tool):
        schema = tool.get_schema()
        prop = schema["function"]["parameters"]["properties"]["pdb_ref"]
        assert prop["x-aevom-reference"] == {"data_type": "pdb"}
        assert prop["type"] == "string"

    def test_description_mentions_pdb_ref(self, tool):
        schema = tool.get_schema()
        desc = schema["function"]["description"]
        assert "pdb_ref" in desc
        assert "vina_dock" in desc.lower() or "docked" in desc.lower()


class TestExecute:
    @pytest.mark.asyncio
    async def test_execute_carries_pdb_ref_in_result_data(self, tool):
        result = await tool.execute(
            {
                "pdb_ref": "pdb_abc12345",
                "molql": "(sel.atom.res (= atom.entity-type non-polymer))",
            }
        )
        assert result.content_type == ContentType.STRUCTURE_QUERY
        assert isinstance(result.data, dict)
        assert result.data["pdb_ref"] == "pdb_abc12345"
        assert result.data["molql"].startswith("(")
        assert result.data["action"] == "select"

    @pytest.mark.asyncio
    async def test_execute_strips_whitespace_on_pdb_ref(self, tool):
        result = await tool.execute(
            {"pdb_ref": "  pdb_abc12345  ", "molql": "(x)"}
        )
        assert result.data["pdb_ref"] == "pdb_abc12345"

    @pytest.mark.asyncio
    async def test_execute_errors_when_pdb_ref_missing(self, tool):
        result = await tool.execute({"molql": "(sel.atom.atoms)"})
        assert result.content_type == ContentType.ERROR
        assert "pdb_ref" in result.data

    @pytest.mark.asyncio
    async def test_execute_errors_when_pdb_ref_blank(self, tool):
        result = await tool.execute({"pdb_ref": "   ", "molql": "(x)"})
        assert result.content_type == ContentType.ERROR
        assert "pdb_ref" in result.data

    @pytest.mark.asyncio
    async def test_execute_errors_on_missing_molql(self, tool):
        result = await tool.execute({"pdb_ref": "pdb_abc"})
        assert result.content_type == ContentType.ERROR
        assert "MolQL" in result.data

    @pytest.mark.asyncio
    async def test_execute_errors_on_unbalanced_parens(self, tool):
        result = await tool.execute(
            {"pdb_ref": "pdb_abc", "molql": "(sel.atom.atoms"}
        )
        assert result.content_type == ContentType.ERROR
        assert "parentheses" in result.data.lower()
