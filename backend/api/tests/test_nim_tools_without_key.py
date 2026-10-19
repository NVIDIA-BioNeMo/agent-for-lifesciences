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

"""NIM-backed tools are withheld when only a non-NVIDIA LLM key is configured.

An Anthropic-only deployment has no credential for build.nvidia.com, so every
NIM call would fail at authentication. The tools must be absent from the
schemas instead, and named so the prompt can say why. With no key at all --
unit tests, a tool-cache replay -- nothing changes.
"""

import pytest
from api.graph import workflows
from api.llm import config as config_module
from api.llm.config import ANTHROPIC_KEY_FILE, NVIDIA_KEY_FILE, NVINF_KEY_FILE, OPENAI_KEY_FILE
from api.tools.base import BaseTool, NvidiaApiTool, ToolResult, ToolType
from api.tools.registry import ToolRegistry, nim_tools_unavailable


class StubNimTool(NvidiaApiTool):
    @property
    def name(self) -> str:
        return "stub_nim"

    @property
    def api_endpoint(self) -> str:
        return "https://example.invalid/nim"

    def get_schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": "", "parameters": {}}}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.text("done")


class StubLocalTool(BaseTool):
    @property
    def name(self) -> str:
        return "stub_local"

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    def get_schema(self) -> dict:
        return {"type": "function", "function": {"name": self.name, "description": "", "parameters": {}}}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.text("done")


@pytest.fixture
def keys(monkeypatch):
    """Key files to treat as present; every other key file is absent."""
    present: set[str] = set()
    monkeypatch.setattr(config_module, "key_file_present", lambda path: path in present)
    monkeypatch.delenv("NVIDIA_API_KEY_FILE", raising=False)
    monkeypatch.delenv("AEVOM_TOOL_CACHE_MODE", raising=False)
    return present


def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(StubNimTool())
    reg.register(StubLocalTool())
    return reg


@pytest.mark.parametrize("key_file", [ANTHROPIC_KEY_FILE, OPENAI_KEY_FILE])
def test_non_nvidia_key_only_withholds_nim_tools(keys, key_file):
    keys.add(key_file)
    reg = registry()
    assert reg.tool_names == ["stub_local"]
    assert reg.unavailable == ["stub_nim"]
    assert "stub_nim" in reg.withheld


@pytest.mark.parametrize(
    "present",
    [
        {NVIDIA_KEY_FILE},
        {NVIDIA_KEY_FILE, ANTHROPIC_KEY_FILE},
        {NVIDIA_KEY_FILE, OPENAI_KEY_FILE},
        set(),  # no key at all: unit tests and bare checkouts see every tool
    ],
)
def test_nim_tools_kept(keys, present):
    keys.update(present)
    reg = registry()
    assert reg.tool_names == ["stub_nim", "stub_local"]
    assert reg.unavailable == []


def test_nvinf_key_alone_does_not_serve_nim_tools(keys):
    keys.add(NVINF_KEY_FILE)
    assert nim_tools_unavailable()


def test_tool_cache_replay_keeps_nim_tools(keys, monkeypatch):
    keys.add(ANTHROPIC_KEY_FILE)
    monkeypatch.setenv("AEVOM_TOOL_CACHE_MODE", "replay")
    assert not nim_tools_unavailable()


def test_structure_prediction_workflow_is_dropped_with_its_tools(keys, monkeypatch):
    for module, name in (
        ("small_molecule", "build_small_molecule_graph"),
        ("structure_prediction", "build_structure_prediction_graph"),
        ("docking", "build_docking_graph"),
        ("protein_ligand", "build_protein_ligand_graph"),
    ):
        imported = __import__(f"api.graph.workflows.{module}", fromlist=[name])
        monkeypatch.setattr(imported, name, lambda *a, **k: object())

    names = {tool.name for tool in workflows.build_workflow_tools(object(), [])}
    assert "run_structure_prediction" in names

    keys.add(ANTHROPIC_KEY_FILE)
    names = {tool.name for tool in workflows.build_workflow_tools(object(), [])}
    assert "run_structure_prediction" not in names
