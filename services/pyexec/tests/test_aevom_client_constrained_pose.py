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

"""Tests for `aevom.constrained_pose()` in the sandbox client.

The helper exists so a model-written loop can score hundreds of candidates
without the poses reaching the transcript: it posts references and SMILES, and
gets back one row of numbers per candidate. What is pinned here is the request it
sends, the rows it returns, and the errors it raises -- an exception carrying the
service's own text is recoverable inside sandbox code, a bare HTTP code is not.

The backend and the rdkit service behind it are stubbed. The chemistry is tested
in services/rdkit/tests/test_constrained_pose.py, the route in
backend/api/tests/test_sandbox_constrained_pose_route.py.
"""

import io
import json

import pytest

from app.aevom_client import AevomClient, AevomError

FRAGMENTS = ["sdf_frag0001", "sdf_frag0002"]
POCKET = "pdb_abc12345"
POSE_SDF = "pose\n  RDKit          3D\n\nM  END\n"


def row(smiles: str, verified: bool = True, **overrides) -> dict:
    result = {
        "smiles": smiles,
        "verified": verified,
        "failed_gate": None if verified else "gate5c_pose",
        "reason": None if verified else "the bridge is too short",
        "strain": 8.2 if verified else None,
        "max_fragment_rmsd": 0.2 if verified else None,
        "min_protein_distance": 2.9 if verified else None,
        "num_clashes": 0,
        "num_linker_atoms": 6,
    }
    result.update(overrides)
    return result


class FakeBackend:
    """Stands in for POST /api/sandbox/constrained-pose and the data route."""

    def __init__(self):
        self.requests: list[dict] = []
        self.response: dict | Exception = {}
        self.artifacts: dict[str, bytes] = {}

    def default_response(self) -> dict:
        return {
            "success": True,
            "results": [row("c1ccc(-c2ccncc2)cc1"), row("CCO", verified=False)],
            "num_verified": 1,
            "pocket_atoms": 871,
            "thresholds": {"strain_max": 100.0, "rmsd_max": 1.0},
        }

    def with_pose(self, reference: str = "sdf_pose00001") -> dict:
        self.artifacts[reference] = POSE_SDF.encode()
        return {
            "success": True,
            "results": [row("c1ccc(-c2ccncc2)cc1", pose_ref=reference)],
            "num_verified": 1,
            "pocket_atoms": 871,
            "thresholds": {},
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
    # A non-empty allowlist so the client's own reference gate is live.
    instance = AevomClient(
        session_token="tok",
        backend_url="http://backend:8000",
        allowed_refs=set(FRAGMENTS) | {POCKET},
    )

    def fake_urlopen(request, timeout=None):
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
        client.constrained_pose(FRAGMENTS, ["CCO"])

        sent = client.backend.requests[0]
        assert sent["url"].endswith("/api/sandbox/constrained-pose")
        # urllib title-cases header names.
        assert sent["headers"]["X-sandbox-session"] == "tok"

    def test_sends_references_rather_than_structures(self, client):
        # The sandbox has no filesystem and the backend already holds the SDFs.
        client.constrained_pose(FRAGMENTS, ["CCO"], pocket_ref=POCKET)

        sent = client.backend.requests[0]["json"]
        assert sent["fragment_refs"] == FRAGMENTS
        assert sent["pocket_ref"] == POCKET
        assert "fragment_sdfs" not in sent

    def test_omits_the_pocket_when_there_is_none(self, client):
        client.constrained_pose(FRAGMENTS, ["CCO"])
        assert "pocket_ref" not in client.backend.requests[0]["json"]

    def test_omits_thresholds_so_the_service_default_applies(self, client):
        # Restating the calibrated defaults here would let them drift.
        client.constrained_pose(FRAGMENTS, ["CCO"])

        sent = client.backend.requests[0]["json"]
        assert "strain_max" not in sent
        assert "rmsd_max" not in sent

    def test_passes_thresholds_through_when_given(self, client):
        client.constrained_pose(FRAGMENTS, ["CCO"], strain_max=40, rmsd_max=1.5)

        sent = client.backend.requests[0]["json"]
        assert sent["strain_max"] == 40.0
        assert sent["rmsd_max"] == 1.5

    def test_does_not_store_poses_unless_asked(self, client):
        # A scan over hundreds of candidates would otherwise leave hundreds of
        # rejected poses in the artifact store.
        client.constrained_pose(FRAGMENTS, ["CCO"])
        assert client.backend.requests[0]["json"]["store_poses"] is False

    def test_a_generator_of_candidates_is_accepted(self, client):
        # Candidates usually come out of a comprehension or a filter.
        client.constrained_pose(FRAGMENTS, (s for s in ["CCO", "CCC"]))
        assert client.backend.requests[0]["json"]["smiles"] == ["CCO", "CCC"]

    def test_no_candidates_is_refused_before_any_request(self, client):
        with pytest.raises(AevomError, match="candidate"):
            client.constrained_pose(FRAGMENTS, [])
        assert client.backend.requests == []

    def test_no_fragments_is_refused_before_any_request(self, client):
        with pytest.raises(AevomError, match="fragment"):
            client.constrained_pose([], ["CCO"])
        assert client.backend.requests == []

    def test_allows_a_batch_more_time_than_a_default_call(self, client):
        client.constrained_pose(FRAGMENTS, ["CCO"])
        assert client.backend.requests[0]["timeout"] >= 300


class TestReturnValue:
    def test_returns_one_row_per_candidate(self, client):
        rows = client.constrained_pose(FRAGMENTS, ["c1ccc(-c2ccncc2)cc1", "CCO"])

        assert [r["smiles"] for r in rows] == ["c1ccc(-c2ccncc2)cc1", "CCO"]
        assert rows[0]["verified"] is True
        assert rows[0]["strain"] == 8.2

    def test_a_rejection_says_which_gate_and_why(self, client):
        rows = client.constrained_pose(FRAGMENTS, ["c1ccc(-c2ccncc2)cc1", "CCO"])

        assert rows[1]["verified"] is False
        assert rows[1]["failed_gate"] == "gate5c_pose"
        assert "too short" in rows[1]["reason"]

    def test_details_carry_the_thresholds_that_were_applied(self, client):
        out = client.constrained_pose(FRAGMENTS, ["CCO"], return_details=True)

        assert out["num_verified"] == 1
        assert out["pocket_atoms"] == 871
        assert out["thresholds"]["rmsd_max"] == 1.0

    def test_a_stored_pose_is_loadable_afterwards(self, client):
        # The route allowlists each stored pose; the client has to mirror that
        # or loading back the pose it just made would be refused.
        client.backend.response = client.backend.with_pose()

        rows = client.constrained_pose(FRAGMENTS, ["CCO"], store_poses=True)

        reference = rows[0]["pose_ref"]
        assert reference in client.allowed_refs
        assert client.load_sdf(reference) == POSE_SDF

    def test_an_unrelated_reference_is_still_refused(self, client):
        # Guards the test above: it means something only because the gate is live.
        from app.aevom_client import AccessDeniedError

        client.constrained_pose(FRAGMENTS, ["CCO"])
        with pytest.raises(AccessDeniedError):
            client.load_sdf("sdf_somebodyelse")


class TestErrors:
    def test_a_service_failure_carries_the_reason(self, client):
        client.backend.response = {
            "success": False,
            "error": "fragment 1 has no 3D coordinates",
        }
        with pytest.raises(AevomError, match="no 3D coordinates"):
            client.constrained_pose(FRAGMENTS, ["CCO"])

    def test_an_http_error_reports_the_status_and_detail(self, client):
        import urllib.error

        client.backend.response = urllib.error.HTTPError(
            url="http://backend:8000/api/sandbox/constrained-pose",
            code=403,
            msg="Forbidden",
            hdrs=None,
            fp=io.BytesIO(b'{"detail":"Reference not allowed"}'),
        )
        with pytest.raises(AevomError, match="HTTP 403"):
            client.constrained_pose(FRAGMENTS, ["CCO"])

    def test_a_network_error_is_reported_as_one(self, client):
        import urllib.error

        client.backend.response = urllib.error.URLError("connection refused")
        with pytest.raises(AevomError, match="Network error"):
            client.constrained_pose(FRAGMENTS, ["CCO"])
