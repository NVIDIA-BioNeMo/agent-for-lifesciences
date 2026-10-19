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

"""Selection lineage reaches the trace span.

A design run is only as explainable as its inputs. These tests pin that the
span of a tool that makes or consumes a selection carries the selection's
reference, who made it, how big it is and what it came from, so a Phoenix
query can answer "which designs came from hotspots a person picked?".
"""

import pytest

from api.tools.base import BaseTool, ToolResult, ToolType
from api.tools.registry import ToolRegistry


class RecordingSpan:
    def __init__(self):
        self.attributes = {}

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def set_status(self, *args, **kwargs):
        pass

    def record_exception(self, *args, **kwargs):
        pass


class MetadataTool(BaseTool):
    def __init__(self, metadata):
        self._metadata = metadata

    @property
    def name(self):
        return "meta_tool"

    @property
    def tool_type(self):
        return ToolType.LOCAL

    def get_schema(self):
        return {"type": "function", "function": {"name": "meta_tool", "description": "", "parameters": {}}}

    async def execute(self, arguments):
        return ToolResult.text("ok", metadata=self._metadata)


async def run(metadata):
    registry = ToolRegistry()
    registry.register(MetadataTool(metadata))
    span = RecordingSpan()
    await registry._execute_impl("meta_tool", {}, None, "s1", "r1", span)
    return span.attributes


@pytest.mark.asyncio
async def test_selection_lineage_is_recorded_on_the_span():
    attrs = await run(
        {
            "selection_ref": "selection_aaa",
            "selection_provenance": "user",
            "selection_atom_count": 31,
            "selection_residue_count": 12,
            "selection_parent": "selection_bbb",
            "design_spec_ref": "design_spec_ccc",
            "hotspot_selector_count": 12,
        }
    )
    assert attrs["tool.selection.ref"] == "selection_aaa"
    assert attrs["tool.selection.provenance"] == "user"
    assert attrs["tool.selection.atom_count"] == 31
    assert attrs["tool.selection.residue_count"] == 12
    assert attrs["tool.selection.parent"] == "selection_bbb"
    assert attrs["tool.design.spec_ref"] == "design_spec_ccc"
    assert attrs["tool.hotspot.selector_count"] == 12


@pytest.mark.asyncio
async def test_absent_keys_are_not_recorded():
    attrs = await run({"selection_ref": "selection_aaa"})
    assert attrs["tool.selection.ref"] == "selection_aaa"
    assert not any(k.startswith("tool.selection.parent") for k in attrs)
    assert "tool.design.spec_ref" not in attrs


@pytest.mark.asyncio
async def test_a_tool_without_selection_metadata_adds_nothing():
    attrs = await run({})
    assert not any(k.startswith(("tool.selection.", "tool.design.", "tool.hotspot.")) for k in attrs)
