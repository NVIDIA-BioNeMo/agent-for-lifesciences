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

"""Production-side validation tests for the structure_query graph wrapper.

The graph tool wrapper must reject ``structure_query`` calls whose
``pdb_ref`` argument does not correspond to a structure currently
loaded in the viewer. This guards against the agent's ref drifting
from what the user actually sees in Molstar.
"""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Pre-stub api.redis and api.pending_ops to avoid `api/` ending up on
# sys.path and shadowing the installed `redis` package while running
# the test under pytest. The wrapper imports these lazily inside the
# STRUCTURE_QUERY branch and never actually touches them on the
# rejection path we are exercising.
if "api.redis" not in sys.modules:
    fake_redis_mod = types.ModuleType("api.redis")
    fake_redis_mod.get_redis = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    sys.modules["api.redis"] = fake_redis_mod
if "api.pending_ops" not in sys.modules:
    fake_pending_mod = types.ModuleType("api.pending_ops")

    class _FakePending:
        def __init__(self, *_args, **_kwargs):
            pass

        async def register(self, *_args, **_kwargs):
            return None

    fake_pending_mod.RedisPendingOps = _FakePending  # type: ignore[attr-defined]
    sys.modules["api.pending_ops"] = fake_pending_mod

from api.graph.tools import (  # noqa: E402
    _graph_request_id_ctx,
    _graph_session_id_ctx,
    _graph_storage_ctx,
    _make_tool_wrapper,
    _viz_event_queue,
)
from api.tools.base import ContentType, ToolResult  # noqa: E402
from api.tools.viewer_state import StructureInfo, ViewerState  # noqa: E402


_STRUCTURE_QUERY_SCHEMA = {
    "type": "function",
    "function": {
        "name": "structure_query",
        "description": "test",
        "parameters": {
            "type": "object",
            "properties": {
                "pdb_ref": {"type": "string"},
                "molql": {"type": "string"},
            },
            "required": ["pdb_ref", "molql"],
        },
    },
}


@pytest.fixture
def viz_queue():
    return asyncio.Queue()


@pytest.fixture
def ctx_tokens(viz_queue):
    tokens = [
        _graph_storage_ctx.set(MagicMock()),
        _graph_session_id_ctx.set("sess_test"),
        _graph_request_id_ctx.set("req_test"),
        _viz_event_queue.set(viz_queue),
    ]
    yield
    for token, var in zip(reversed(tokens), [
        _viz_event_queue,
        _graph_request_id_ctx,
        _graph_session_id_ctx,
        _graph_storage_ctx,
    ]):
        var.reset(token)


def _patched_registry(monkeypatch, fake_registry):
    import api.tools.registry as registry_mod
    monkeypatch.setattr(registry_mod, "get_registry", lambda: fake_registry)


def _patched_viewer(monkeypatch, fake_viewer):
    import api.tools.viewer_state as vs_mod
    monkeypatch.setattr(
        vs_mod, "get_current_viewer_state", AsyncMock(return_value=fake_viewer)
    )


@pytest.mark.asyncio
async def test_validation_blocks_unloaded_pdb_ref(
    ctx_tokens, viz_queue, monkeypatch
):
    """Wrapper should error when pdb_ref isn't in viewer_state.structures."""
    sq_result = ToolResult(
        content_type=ContentType.STRUCTURE_QUERY,
        data={
            "pdb_ref": "pdb_unloaded",
            "molql": "(sel.atom.atoms)",
            "action": "select",
        },
        metadata={},
    )
    fake_viewer = ViewerState(
        is_empty=False,
        structures=[StructureInfo(label="loaded", pdb_ref="pdb_loaded")],
    )

    fake_registry = MagicMock()
    fake_registry.execute = AsyncMock(return_value=sq_result)
    _patched_registry(monkeypatch, fake_registry)
    _patched_viewer(monkeypatch, fake_viewer)

    wrapper = _make_tool_wrapper("structure_query", _STRUCTURE_QUERY_SCHEMA)
    out = await wrapper.coroutine(pdb_ref="pdb_unloaded", molql="(sel.atom.atoms)")

    assert "pdb_unloaded" in out
    assert "not loaded" in out.lower() or "visualize_pdb" in out
    assert viz_queue.empty(), "validation must short-circuit before queuing viz"


@pytest.mark.asyncio
async def test_validation_passes_when_pdb_ref_loaded(
    ctx_tokens, viz_queue, monkeypatch
):
    """Wrapper should queue the viz event when pdb_ref is loaded."""
    sq_result = ToolResult(
        content_type=ContentType.STRUCTURE_QUERY,
        data={"pdb_ref": "pdb_loaded", "molql": "(x)", "action": "select"},
        metadata={},
    )
    fake_viewer = ViewerState(
        is_empty=False,
        structures=[StructureInfo(label="loaded", pdb_ref="pdb_loaded")],
    )

    fake_registry = MagicMock()
    fake_registry.execute = AsyncMock(return_value=sq_result)
    _patched_registry(monkeypatch, fake_registry)
    _patched_viewer(monkeypatch, fake_viewer)

    wrapper = _make_tool_wrapper("structure_query", _STRUCTURE_QUERY_SCHEMA)
    await wrapper.coroutine(pdb_ref="pdb_loaded", molql="(x)")

    assert not viz_queue.empty(), "valid pdb_ref should queue a viz event"
    event = await viz_queue.get()
    assert event["tool_event"]["type"] == "structure_query"


@pytest.mark.asyncio
async def test_validation_skipped_when_viewer_has_no_known_refs(
    ctx_tokens, viz_queue, monkeypatch
):
    """If no structures advertise pdb_ref (e.g. legacy state), don't block."""
    sq_result = ToolResult(
        content_type=ContentType.STRUCTURE_QUERY,
        data={"pdb_ref": "pdb_anything", "molql": "(x)", "action": "select"},
        metadata={},
    )
    fake_viewer = ViewerState(
        is_empty=False,
        structures=[StructureInfo(label="legacy", pdb_ref="")],
    )

    fake_registry = MagicMock()
    fake_registry.execute = AsyncMock(return_value=sq_result)
    _patched_registry(monkeypatch, fake_registry)
    _patched_viewer(monkeypatch, fake_viewer)

    wrapper = _make_tool_wrapper("structure_query", _STRUCTURE_QUERY_SCHEMA)
    await wrapper.coroutine(pdb_ref="pdb_anything", molql="(x)")

    assert not viz_queue.empty()
