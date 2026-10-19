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

"""Tests for `aevom.probe_grid()` in the sandbox client.

The helper's whole job is to keep a 166,000-point grid out of the model's
context: it posts a box to the backend, gets a reference back, and loads the
array through the ordinary data route. So what is pinned here is the request it
sends, the array it returns, and the errors it raises -- an exception with the
service's own text is recoverable inside sandbox code, a bare HTTP code is not.

The backend and the Vina service behind it are stubbed. The scoring function is
validated in services/vina/tests/test_probe_grid.py, and the route in
backend/api/tests/test_sandbox_probe_grid_route.py.
"""

import io
import json

import numpy as np
import pytest

from app.aevom_client import AevomClient, AevomError

CENTER = [31.575, -1.590, 25.599]
SHAPE = [5, 5, 5]
ORIGIN = [30.575, -2.590, 24.599]

MINIMUM = {
    "probe": "O_A",
    "family": "Acceptor",
    "x": 31.6,
    "y": -1.6,
    "z": 25.6,
    "energy": -0.812,
}


def npz_bytes(array) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=np.asarray(array))
    return buffer.getvalue()


class FakeBackend:
    """Stands in for POST /api/sandbox/probe-grid and the data route."""

    def __init__(self):
        self.grid = np.full((3, *SHAPE), -0.4, dtype=np.float32)
        self.requests: list[dict] = []
        self.response: dict | Exception = {}
        self.artifacts: dict[str, bytes] = {}

    def default_response(self) -> dict:
        reference = "array_probegrid1"
        self.artifacts[reference] = npz_bytes(self.grid)
        return {
            "success": True,
            "reference": reference,
            "probes": ["C_H", "N_D", "O_A"],
            "origin": ORIGIN,
            "spacing": 0.5,
            "shape": SHAPE,
            "minima": [MINIMUM],
        }


class FakeHTTPResponse:
    def __init__(self, payload: bytes):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def client(monkeypatch):
    backend = FakeBackend()
    backend.response = backend.default_response()
    # A non-empty allowlist so the client's own reference gate is live: the
    # receptor is readable, anything the session has not been told about is not.
    instance = AevomClient(
        session_token="tok",
        backend_url="http://backend:8000",
        allowed_refs={"pdb_abc12345"},
    )

    def fake_urlopen(request, timeout=None):
        # Both routes go through here, so the data route runs for real and the
        # allowlist is enforced rather than stubbed past.
        if "/api/sandbox/data/" in request.full_url:
            reference = request.full_url.rsplit("/", 1)[1]
            return FakeHTTPResponse(backend.artifacts[reference])

        backend.requests.append(
            {
                "url": request.full_url,
                "headers": dict(request.headers),
                "json": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if isinstance(backend.response, Exception):
            raise backend.response
        return FakeHTTPResponse(json.dumps(backend.response).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    instance.backend = backend
    return instance


class TestRequest:
    def test_posts_to_the_sandbox_route_with_the_session_header(self, client):
        client.probe_grid("pdb_abc12345", center=CENTER)

        sent = client.backend.requests[0]
        assert sent["url"].endswith("/api/sandbox/probe-grid")
        # urllib title-cases header names.
        assert sent["headers"]["X-sandbox-session"] == "tok"

    def test_sends_the_reference_rather_than_the_receptor(self, client):
        # The sandbox has no filesystem and the backend already holds the PDB.
        client.probe_grid("pdb_abc12345", center=CENTER)

        sent = client.backend.requests[0]["json"]
        assert sent["receptor_ref"] == "pdb_abc12345"
        assert "pdb_content" not in sent

    def test_defaults_to_a_twenty_angstrom_box(self, client):
        client.probe_grid("pdb_abc12345", center=CENTER)

        sent = client.backend.requests[0]["json"]
        assert sent["size"] == [20.0, 20.0, 20.0]
        assert sent["spacing"] == 0.5

    def test_omits_probes_so_the_backend_default_applies(self, client):
        # Restating the default here would let the two drift apart.
        client.probe_grid("pdb_abc12345", center=CENTER)
        assert "probes" not in client.backend.requests[0]["json"]

    def test_passes_a_requested_probe_list_through(self, client):
        client.probe_grid("pdb_abc12345", center=CENTER, probes=["O_A"])
        assert client.backend.requests[0]["json"]["probes"] == ["O_A"]

    def test_coerces_a_numpy_centre(self, client):
        # Pocket centres are usually computed, so they arrive as numpy scalars
        # that json cannot serialise.
        client.probe_grid("pdb_abc12345", center=np.array(CENTER))

        sent = client.backend.requests[0]["json"]
        assert sent["center"] == pytest.approx(CENTER)

    def test_a_tuple_size_is_accepted(self, client):
        client.probe_grid("pdb_abc12345", center=CENTER, size=(24, 24, 24))
        assert client.backend.requests[0]["json"]["size"] == [24.0, 24.0, 24.0]

    def test_a_two_component_centre_is_refused_before_any_request(self, client):
        with pytest.raises(AevomError, match="three components"):
            client.probe_grid("pdb_abc12345", center=[1.0, 2.0])
        assert client.backend.requests == []

    def test_allows_the_grid_more_time_than_a_default_call(self, client):
        client.probe_grid("pdb_abc12345", center=CENTER)
        assert client.backend.requests[0]["timeout"] >= 300


class TestReturnValue:
    def test_returns_the_grid_as_a_numpy_array(self, client):
        grid = client.probe_grid("pdb_abc12345", center=CENTER)

        assert isinstance(grid, np.ndarray)
        assert grid.shape == (3, 5, 5, 5)
        assert np.allclose(grid, client.backend.grid)

    def test_the_probe_axis_is_ordered_like_the_request(self, client):
        # grid[0] must be the first probe asked for, or every downstream index
        # refers to the wrong pharmacophore family.
        client.backend.grid = np.stack(
            [
                np.full(SHAPE, -0.1, dtype=np.float32),
                np.full(SHAPE, -0.2, dtype=np.float32),
                np.full(SHAPE, -0.3, dtype=np.float32),
            ]
        )
        client.backend.response = client.backend.default_response()

        grid = client.probe_grid("pdb_abc12345", center=CENTER)

        assert grid[0].max() == pytest.approx(-0.1)
        assert grid[2].max() == pytest.approx(-0.3)

    def test_details_carry_the_geometry_and_the_minima(self, client):
        out = client.probe_grid("pdb_abc12345", center=CENTER, return_details=True)

        assert out["origin"] == ORIGIN
        assert out["spacing"] == 0.5
        assert out["shape"] == SHAPE
        assert out["probes"] == ["C_H", "N_D", "O_A"]
        assert out["minima"][0]["family"] == "Acceptor"
        assert isinstance(out["grid"], np.ndarray)

    def test_details_include_the_reference_for_a_later_load(self, client):
        out = client.probe_grid("pdb_abc12345", center=CENTER, return_details=True)
        assert out["reference"].startswith("array_")

    def test_the_returned_reference_is_loadable_again(self, client):
        # The route allowlists the new reference; the client has to mirror that
        # or a second load_array() in the same run would be refused by the
        # session that just asked for the grid.
        out = client.probe_grid("pdb_abc12345", center=CENTER, return_details=True)

        assert out["reference"] in client.allowed_refs
        assert np.allclose(client.load_array(out["reference"]), client.backend.grid)

    def test_an_unrelated_reference_is_still_refused(self, client):
        # Guards the test above: it means something only because the gate is
        # live in this fixture.
        from app.aevom_client import AccessDeniedError

        client.probe_grid("pdb_abc12345", center=CENTER)
        with pytest.raises(AccessDeniedError):
            client.load_array("array_somebodyelse")


class TestErrors:
    def test_a_service_failure_carries_the_reason(self, client):
        client.backend.response = {
            "success": False,
            "error": "Receptor preparation failed.",
        }
        with pytest.raises(AevomError, match="Receptor preparation failed."):
            client.probe_grid("pdb_abc12345", center=CENTER)

    def test_a_response_without_a_reference_is_an_error(self, client):
        client.backend.response = {"success": True}
        with pytest.raises(AevomError, match="reference"):
            client.probe_grid("pdb_abc12345", center=CENTER)

    def test_an_http_error_reports_the_status_and_detail(self, client):
        import urllib.error

        client.backend.response = urllib.error.HTTPError(
            url="http://backend:8000/api/sandbox/probe-grid",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"Reference not allowed"}'),
        )
        with pytest.raises(AevomError, match="HTTP 403"):
            client.probe_grid("pdb_abc12345", center=CENTER)

    def test_a_network_error_is_reported_as_one(self, client):
        import urllib.error

        client.backend.response = urllib.error.URLError("connection refused")
        with pytest.raises(AevomError, match="Network error"):
            client.probe_grid("pdb_abc12345", center=CENTER)
