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

"""Tests for LLM text-based tool call parsing and sanitization.

Verifies that extract_tool_calls_from_text() and sanitize_llm_content()
correctly handle various XML-style tool call formats from LLMs that
don't use the native function calling API.
"""

import json

import pytest

from api.llm.text_parser import extract_tool_calls_from_text, sanitize_llm_content


class TestExtractToolCallsFromText:
    """Tests for extract_tool_calls_from_text()."""

    def test_empty_content(self):
        """Test that empty/None content returns empty list."""
        assert extract_tool_calls_from_text("") == []
        assert extract_tool_calls_from_text(None) == []

    def test_no_tool_calls(self):
        """Test that regular text returns empty list."""
        assert extract_tool_calls_from_text("Hello, how can I help you?") == []

    def test_single_tool_call(self):
        """Test extraction of a single tool call."""
        content = (
            '<tool_call><function=vina_dock>'
            '<parameter=center_x>-9.02</parameter>'
            '<parameter=center_y>12.5</parameter>'
            '<parameter=center_z>0.0</parameter>'
            '</function></tool_call>'
        )
        result = extract_tool_calls_from_text(content)

        assert len(result) == 1
        call = result[0]
        assert call["type"] == "function"
        assert call["function"]["name"] == "vina_dock"

        args = json.loads(call["function"]["arguments"])
        assert args["center_x"] == -9.02
        assert args["center_y"] == 12.5
        assert args["center_z"] == 0.0

    def test_string_parameter(self):
        """Test that string parameters are preserved."""
        content = (
            '<tool_call><function=fetch_protein_sequence>'
            '<parameter=gene_name>BRCA1</parameter>'
            '</function></tool_call>'
        )
        result = extract_tool_calls_from_text(content)

        assert len(result) == 1
        args = json.loads(result[0]["function"]["arguments"])
        assert args["gene_name"] == "BRCA1"

    def test_boolean_parameters(self):
        """Test that boolean parameters are parsed correctly."""
        content = (
            '<tool_call><function=fetch_pdb>'
            '<parameter=remove_ligands>true</parameter>'
            '<parameter=keep_water>false</parameter>'
            '</function></tool_call>'
        )
        result = extract_tool_calls_from_text(content)

        args = json.loads(result[0]["function"]["arguments"])
        assert args["remove_ligands"] is True
        assert args["keep_water"] is False

    def test_integer_parameter(self):
        """Test that integer parameters are parsed correctly."""
        content = (
            '<tool_call><function=search_rcsb>'
            '<parameter=max_results>5</parameter>'
            '</function></tool_call>'
        )
        result = extract_tool_calls_from_text(content)

        args = json.loads(result[0]["function"]["arguments"])
        assert args["max_results"] == 5
        assert isinstance(args["max_results"], int)

    def test_multiple_tool_calls(self):
        """Test extraction of multiple tool calls."""
        content = (
            '<tool_call><function=fetch_protein_sequence>'
            '<parameter=gene_name>PGK1</parameter>'
            '</function></tool_call>'
            'Some text in between'
            '<tool_call><function=visualize_pdb>'
            '<parameter=pdb_ref>pdb_abc123</parameter>'
            '</function></tool_call>'
        )
        result = extract_tool_calls_from_text(content)

        assert len(result) == 2
        assert result[0]["function"]["name"] == "fetch_protein_sequence"
        assert result[1]["function"]["name"] == "visualize_pdb"

    def test_tool_call_ids_are_unique(self):
        """Test that generated tool call IDs are unique."""
        content = (
            '<tool_call><function=tool_a><parameter=x>1</parameter></function></tool_call>'
            '<tool_call><function=tool_b><parameter=y>2</parameter></function></tool_call>'
        )
        result = extract_tool_calls_from_text(content)

        ids = [call["id"] for call in result]
        assert len(set(ids)) == 2  # All unique
        assert all(id.startswith("text_call_") for id in ids)


class TestSanitizeLlmContent:
    """Tests for sanitize_llm_content()."""

    def test_empty_content(self):
        """Test that empty/None content is returned as-is."""
        assert sanitize_llm_content("") == ""
        assert sanitize_llm_content(None) is None

    def test_no_artifacts(self):
        """Test that content without tool call artifacts is unchanged."""
        content = "Here is a protein analysis result."
        assert sanitize_llm_content(content) == content

    def test_removes_tool_call_tags(self):
        """Test removal of <tool_call>...</tool_call> artifacts."""
        content = (
            "Let me dock this molecule."
            "<tool_call><function=vina_dock>"
            "<parameter=center_x>-9.0</parameter>"
            "</function></tool_call>"
        )
        result = sanitize_llm_content(content)
        assert "<tool_call>" not in result
        assert "Let me dock this molecule." in result

    def test_removes_function_tags(self):
        """Test removal of <function=name>...</function> artifacts."""
        content = (
            "Analyzing..."
            '<function=fetch_protein_sequence>'
            '<parameter=gene_name>PGK1</parameter>'
            '</function>'
        )
        result = sanitize_llm_content(content)
        assert "<function=" not in result
        assert "Analyzing..." in result

    def test_removes_incomplete_tags_at_end(self):
        """Test removal of incomplete tool call tags at end of stream."""
        content = "Starting analysis<tool_call><function=vina_dock"
        result = sanitize_llm_content(content)
        assert "<tool_call>" not in result

    def test_preserves_whitespace_when_no_changes(self):
        """Test that whitespace is preserved when no artifacts are found."""
        content = "  Hello  \n  World  "
        assert sanitize_llm_content(content) == content

    def test_strips_whitespace_after_removal(self):
        """Test that result is stripped when artifacts are removed."""
        content = "  <tool_call><function=x></function></tool_call>  Hello  "
        result = sanitize_llm_content(content)
        assert result == "Hello"
