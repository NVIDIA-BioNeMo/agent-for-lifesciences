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

"""Tests for POST /api/sandbox/probe-grid.

This route exists so `aevom.probe_grid()` can hand Python code a numpy array
without the grid passing through the model's context, and without pyexec gaining
an outbound dependency on the Vina service. Three things are load-bearing and
tested here: the response carries a reference rather than the grid, the stored
bytes are the npz the array loader reads, and the new reference is added to the
calling session's allowlist -- otherwise the helper's own load_array() would be
refused by the session that just asked for the grid.

The Vina service is stubbed. The scoring function is validated in
services/vina/tests/test_probe_grid.py.
"""

import base64
import io

import numpy as np
import pytest
from fastapi import HTTPException

from api.sandbox import routes
from api.sandbox.routes import ProbeGridSandboxRequest
from api.sandbox.session import (
    ReferenceNotAllowedError,
    SessionNotFoundError,
)
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService

CENTER = [31.575, -1.590, 25.599]
SHAPE = [5, 5, 5]
ORIGIN = [30.575, -2.590, 24.599]

RECEPTOR = b"ATOM      1  N   ALA A   1      11.104   6.134  -6.504\nEND\n"


class FakeSandboxSession:
    def __init__(self, client_session_id="sess_1"):
        self.client_session_id = client_session_id


class FakeSessionManager:
    def __init__(self, error: Exception | None = None):
        self.session = FakeSandboxSession()
        self.allowed: list[str] = []
        self.error = error

    async def validate_access(self, token, reference):
        if self.error is not None:
            raise self.error
        return self.session

    async def add_allowed_ref(self, token, reference):
        self.allowed.append(reference)


def npz_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=array)
    return base64.b64encode(buffer.getvalue()).decode()


def service_response(array: np.ndarray | None = None, probes=None) -> dict:
    probes = probes if probes is not None else ["C_H", "N_D", "O_A"]
    if array is None:
        array = np.full((len(probes), *SHAPE), -0.4, dtype=np.float32)
    return {
        "success": True,
        "npz_base64": npz_base64(array),
        "probes": probes,
        "origin": ORIGIN,
        "spacing": 0.5,
        "shape": SHAPE,
        "minima": [
            {
                "probe": "O_A",
                "family": "Acceptor",
                "x": 31.6,
                "y": -1.6,
                "z": 25.6,
                "energy": -0.812,
            }
        ],
        "error": None,
    }


@pytest.fixture
def wired(monkeypatch):
    """In-memory storage, a permissive session, and a stubbed Vina service."""
    manager = FakeSessionManager()
    storage = DataStorageService(repository=InMemoryDataArtifactRepository())
    calls: list = []
    payload: dict = {"response": service_response()}

    async def fake_call(self, endpoint, method="POST", json_data=None, timeout=30.0, retries=0):
        calls.append({"endpoint": endpoint, "json": json_data, "timeout": timeout})
        outcome = payload["response"]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(routes, "get_session_manager", lambda: manager)
    monkeypatch.setattr(
        routes,
        "scoped_storage",
        lambda _session, session_id: SessionScopedStorage(storage, session_id),
    )
    monkeypatch.setattr(
        "api.tools.probe_grid.ProbeGridTool._call_service", fake_call, raising=True
    )
    return {
        "manager": manager,
        "storage": storage,
        "calls": calls,
        "payload": payload,
    }


async def receptor_ref(storage) -> str:
    return await storage.store("pdb", RECEPTOR, {"session_id": "sess_1"})


async def call(reference: str, **kwargs):
    return await routes.sandbox_probe_grid(
        ProbeGridSandboxRequest(receptor_ref=reference, center=CENTER, **kwargs),
        x_sandbox_session="tok",
        session=None,
    )


@pytest.mark.asyncio
class TestSuccessfulGrid:
    async def test_returns_a_reference_and_not_the_grid(self, wired):
        ref = await receptor_ref(wired["storage"])

        response = await call(ref)

        assert response.success
        assert response.reference.startswith("array_")

    async def test_the_stored_bytes_load_as_numpy_under_the_key_data(self, wired):
        array = np.arange(3 * 125, dtype=np.float32).reshape(3, 5, 5, 5)
        wired["payload"]["response"] = service_response(array)
        ref = await receptor_ref(wired["storage"])

        response = await call(ref)

        content = await wired["storage"].retrieve(response.reference)
        with np.load(io.BytesIO(content)) as archive:
            assert np.allclose(archive["data"], array)

    async def test_the_new_reference_becomes_readable_in_the_same_session(self, wired):
        # aevom.probe_grid() calls load_array() straight afterwards, through the
        # same session that has never seen this reference before.
        ref = await receptor_ref(wired["storage"])

        response = await call(ref)

        assert response.reference in wired["manager"].allowed

    async def test_carries_the_geometry_needed_to_index_the_array(self, wired):
        ref = await receptor_ref(wired["storage"])

        response = await call(ref)

        assert response.origin == ORIGIN
        assert response.spacing == 0.5
        assert response.shape == SHAPE
        assert response.probes == ["C_H", "N_D", "O_A"]

    async def test_returns_the_ranked_minima(self, wired):
        ref = await receptor_ref(wired["storage"])

        response = await call(ref)

        assert response.minima[0]["family"] == "Acceptor"

    async def test_the_stored_artifact_records_its_provenance(self, wired):
        ref = await receptor_ref(wired["storage"])

        response = await call(ref)

        _content, metadata = await wired["storage"].retrieve_with_metadata(
            response.reference
        )
        assert metadata["source"] == "aevom.probe_grid"
        assert metadata["receptor_ref"] == ref
        assert metadata["session_id"] == "sess_1"


@pytest.mark.asyncio
class TestServiceRequest:
    async def test_sends_the_receptor_contents_to_the_grid_endpoint(self, wired):
        ref = await receptor_ref(wired["storage"])

        await call(ref)

        sent = wired["calls"][0]
        assert sent["endpoint"] == "/probe-grid"
        assert sent["json"]["pdb_content"].startswith("ATOM")

    async def test_defaults_match_the_tool(self, wired):
        # Divergent defaults between the two front doors would make the sandbox
        # helper and the tool disagree on the same pocket.
        ref = await receptor_ref(wired["storage"])

        await call(ref)

        sent = wired["calls"][0]["json"]
        assert sent["size"] == [20.0, 20.0, 20.0]
        assert sent["spacing"] == 0.5
        assert sent["probes"] == ["C_H", "N_D", "O_A"]

    async def test_passes_the_box_through(self, wired):
        ref = await receptor_ref(wired["storage"])

        await call(ref, size=[24.0, 24.0, 24.0], spacing=0.375, max_minima=8)

        sent = wired["calls"][0]["json"]
        assert sent["size"] == [24.0, 24.0, 24.0]
        assert sent["spacing"] == 0.375
        assert sent["max_minima"] == 8

    async def test_allows_the_grid_more_time_than_a_default_service_call(self, wired):
        # A fine grid over a large box is minutes of numpy; the 30s default would
        # cut it off.
        ref = await receptor_ref(wired["storage"])

        await call(ref)

        assert wired["calls"][0]["timeout"] > 30.0


@pytest.mark.asyncio
class TestRejectedRequests:
    async def test_an_unknown_probe_is_a_400(self, wired):
        ref = await receptor_ref(wired["storage"])

        with pytest.raises(HTTPException) as exc:
            await call(ref, probes=["C_H", "Zn"])

        assert exc.value.status_code == 400
        assert "Zn" in exc.value.detail

    async def test_a_malformed_reference_is_a_400(self, wired):
        with pytest.raises(HTTPException) as exc:
            await call("not-a-reference")
        assert exc.value.status_code == 400

    async def test_a_missing_artifact_is_a_404(self, wired):
        with pytest.raises(HTTPException) as exc:
            await call("pdb_deadbeef")
        assert exc.value.status_code == 404

    async def test_an_unknown_session_is_a_401(self, wired, monkeypatch):
        manager = FakeSessionManager(error=SessionNotFoundError("no such session"))
        monkeypatch.setattr(routes, "get_session_manager", lambda: manager)
        ref = await receptor_ref(wired["storage"])

        with pytest.raises(HTTPException) as exc:
            await call(ref)

        assert exc.value.status_code == 401

    async def test_a_reference_outside_the_session_is_a_403(self, wired, monkeypatch):
        # Sandbox code must not be able to grid a receptor belonging to another
        # conversation just by guessing its reference.
        manager = FakeSessionManager(error=ReferenceNotAllowedError("not yours"))
        monkeypatch.setattr(routes, "get_session_manager", lambda: manager)
        ref = await receptor_ref(wired["storage"])

        with pytest.raises(HTTPException) as exc:
            await call(ref)

        assert exc.value.status_code == 403

    async def test_a_centre_with_two_components_is_refused_by_the_model(self):
        with pytest.raises(ValueError):
            ProbeGridSandboxRequest(receptor_ref="pdb_abc12345", center=[1.0, 2.0])

    async def test_spacing_is_bounded(self):
        # Below 0.2 A the grid is enormous for no gain; above 1 A the wells are
        # missed between points.
        with pytest.raises(ValueError):
            ProbeGridSandboxRequest(
                receptor_ref="pdb_abc12345", center=CENTER, spacing=0.05
            )


@pytest.mark.asyncio
class TestFailuresReachTheSandbox:
    """A failure returns success=False rather than raising.

    The helper turns these into an exception with the service's own message, so
    sandbox code sees why it failed. An HTTPException here would surface as a
    bare status code instead.
    """

    async def test_a_service_error_is_reported_without_storing_anything(self, wired):
        wired["payload"]["response"] = {
            "success": False,
            "error": "Receptor preparation failed.",
        }
        ref = await receptor_ref(wired["storage"])

        response = await call(ref)

        assert response.success is False
        assert "Receptor preparation failed." in response.error
        assert response.reference is None
        assert wired["manager"].allowed == []

    async def test_a_transport_failure_is_reported(self, wired):
        wired["payload"]["response"] = RuntimeError("connection refused")
        ref = await receptor_ref(wired["storage"])

        response = await call(ref)

        assert response.success is False
        assert "connection refused" in response.error

    async def test_an_unreadable_payload_is_reported(self, wired):
        wired["payload"]["response"] = {
            "success": True,
            "npz_base64": "not base64!!",
            "probes": ["C_H"],
        }
        ref = await receptor_ref(wired["storage"])

        response = await call(ref)

        assert response.success is False
        assert response.reference is None

    async def test_a_missing_payload_is_reported(self, wired):
        wired["payload"]["response"] = {"success": True, "probes": ["C_H"]}
        ref = await receptor_ref(wired["storage"])

        response = await call(ref)

        assert response.success is False
