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

"""AEVOM_FEAT_HTTP_REQUEST_TOOL decides whether the model has ``http_request``.

Off by default. When off the tool is never registered, so it has no schema, no
LangChain binding and no execute path, and nothing in the prompt names it.
"""

import pytest

from api.secret_store.prompt import stored_keys_block
from api.tools.http_common import http_request_tool_enabled
from api.tools.registry import ToolRegistry, _register_default_tools

FLAG = "AEVOM_FEAT_HTTP_REQUEST_TOOL"


def default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    _register_default_tools(registry)
    return registry


@pytest.mark.parametrize("value", [None, "", "false", "0", "no", "TRUEISH"])
def test_anything_but_true_is_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(FLAG, raising=False)
    else:
        monkeypatch.setenv(FLAG, value)
    assert not http_request_tool_enabled()


@pytest.mark.parametrize("value", ["true", "TRUE", "1"])
def test_true_or_one_is_on(monkeypatch, value):
    monkeypatch.setenv(FLAG, value)
    assert http_request_tool_enabled()


@pytest.mark.asyncio
async def test_the_default_registry_withholds_the_tool(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    registry = default_registry()
    assert "http_request" not in registry
    assert "http_request" in registry.withheld
    names = [schema["function"]["name"] for schema in await registry.get_all_schemas()]
    assert "http_request" not in names


def test_the_flag_registers_the_tool(monkeypatch):
    monkeypatch.setenv(FLAG, "true")
    registry = default_registry()
    assert "http_request" in registry
    assert "http_request" not in registry.withheld


@pytest.mark.parametrize("enabled", [True, False])
def test_the_stored_keys_block_names_only_tools_that_exist(monkeypatch, enabled):
    monkeypatch.setenv(FLAG, "true" if enabled else "false")
    block = stored_keys_block(["NVIDIA_API_KEY"])
    assert "NVIDIA_API_KEY" in block
    assert "`secret_name`" in block
    assert ("http_request" in block) is enabled


def test_no_stored_keys_means_no_block():
    assert stored_keys_block([]) == ""
