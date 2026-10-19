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

"""Tests for POST /api/sandbox/constrained-pose.

This route exists so `aevom.constrained_pose()` can score hundreds of candidates
from Python without the poses or the rejected rows reaching the model's context,
and without pyexec gaining an outbound dependency on the rdkit service. What is
load-bearing here: every reference is checked against the calling session before
anything is read, pose SDFs are stripped from the response unless a reference was
asked for, and a stored pose becomes readable by the session that made it.

The rdkit service is stubbed; whether a molecule really holds its fragments is
tested in services/rdkit/tests/test_constrained_pose.py.
"""

import pytest
from fastapi import HTTPException

from api.sandbox import routes
from api.sandbox.routes import ConstrainedPoseSandboxRequest
from api.sandbox.session import (
    ReferenceNotAllowedError,
    SessionNotFoundError,
)
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService

FRAGMENT = b"""benzene
  RDKit          3D

  1  0  0  0  0  0  0  0  0  0999 V2000
    1.0000    0.0000    0.0000 C   0  0
M  END
"""
POCKET = b"ATOM      1  N   ALA A   1      11.104   6.134  -6.504\nEND\n"
POSE_SDF = "pose\n  RDKit          3D\n\nM  END\n"


class FakeSandboxSession:
    def __init__(self, client_session_id="sess_1"):
        self.client_session_id = client_session_id


class FakeSessionManager:
    def __init__(self, error: Exception | None = None):
        self.session = FakeSandboxSession()
        self.allowed: list[str] = []
        self.checked: list[str] = []
        self.error = error

    async def validate_access(self, token, reference):
        self.checked.append(reference)
        if self.error is not None:
            raise self.error
        return self.session

    async def add_allowed_ref(self, token, reference):
        self.allowed.append(reference)


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
        "pose_sdf": POSE_SDF if verified else None,
    }
    result.update(overrides)
    return result


def service_response(results=None, **overrides) -> dict:
    results = results if results is not None else [row("c1ccc(-c2ccncc2)cc1")]
    payload = {
        "success": True,
        "results": results,
        "num_verified": sum(1 for r in results if r["verified"]),
        "pocket_atoms": 871,
        "thresholds": {"strain_max": 100.0, "rmsd_max": 1.0},
        "error": None,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def wired(monkeypatch):
    """In-memory storage, a permissive session, and a stubbed rdkit service."""
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
        "api.tools.fragment_link.FragmentLinkTool._call_service", fake_call, raising=True
    )
    return {"manager": manager, "storage": storage, "calls": calls, "payload": payload}


async def stored_refs(storage) -> tuple[list[str], str]:
    fragments = [await storage.store("sdf", FRAGMENT, {"session_id": "sess_1"}) for _ in range(2)]
    pocket = await storage.store("pdb", POCKET, {"session_id": "sess_1"})
    return fragments, pocket


async def call(fragment_refs, smiles=("CCO",), **kwargs):
    return await routes.sandbox_constrained_pose(
        ConstrainedPoseSandboxRequest(
            fragment_refs=list(fragment_refs), smiles=list(smiles), **kwargs
        ),
        x_sandbox_session="tok",
        session=None,
    )


@pytest.mark.asyncio
class TestSuccessfulVerification:
    async def test_returns_the_measurements_per_candidate(self, wired):
        fragments, pocket = await stored_refs(wired["storage"])

        response = await call(fragments, pocket_ref=pocket)

        assert response.success
        assert response.num_verified == 1
        assert response.results[0]["strain"] == 8.2
        assert response.pocket_atoms == 871

    async def test_the_pose_is_dropped_unless_a_reference_is_wanted(self, wired):
        # 200 candidates times a few kB of SDF is the payload this route exists
        # to avoid; the numbers are what the scoring loop reads.
        fragments, pocket = await stored_refs(wired["storage"])

        response = await call(fragments, pocket_ref=pocket)

        assert "pose_sdf" not in response.results[0]
        assert "pose_ref" not in response.results[0]
        assert wired["manager"].allowed == []

    async def test_storing_poses_returns_references(self, wired):
        fragments, pocket = await stored_refs(wired["storage"])

        response = await call(fragments, pocket_ref=pocket, store_poses=True)

        reference = response.results[0]["pose_ref"]
        assert reference.startswith("sdf_")
        content = await wired["storage"].retrieve(reference)
        assert content.decode() == POSE_SDF

    async def test_a_stored_pose_becomes_readable_in_the_same_session(self, wired):
        # The helper loads back what it just made, through a session that has
        # never seen this reference.
        fragments, pocket = await stored_refs(wired["storage"])

        response = await call(fragments, pocket_ref=pocket, store_poses=True)

        assert response.results[0]["pose_ref"] in wired["manager"].allowed

    async def test_rejects_get_no_pose_even_when_storing(self, wired):
        fragments, pocket = await stored_refs(wired["storage"])
        wired["payload"]["response"] = service_response(
            [row("CCO"), row("CCC", verified=False)]
        )

        response = await call(fragments, pocket_ref=pocket, store_poses=True)

        assert "pose_ref" in response.results[0]
        assert "pose_ref" not in response.results[1]
        assert len(wired["manager"].allowed) == 1

    async def test_a_rejection_keeps_its_reason(self, wired):
        fragments, pocket = await stored_refs(wired["storage"])
        wired["payload"]["response"] = service_response([row("CCC", verified=False)])

        response = await call(fragments, pocket_ref=pocket)

        assert response.results[0]["failed_gate"] == "gate5c_pose"
        assert "too short" in response.results[0]["reason"]

    async def test_the_stored_pose_records_its_provenance(self, wired):
        fragments, pocket = await stored_refs(wired["storage"])

        response = await call(fragments, pocket_ref=pocket, store_poses=True)

        _content, metadata = await wired["storage"].retrieve_with_metadata(
            response.results[0]["pose_ref"]
        )
        assert metadata["source"] == "aevom.constrained_pose"
        assert metadata["fragment_refs"] == fragments
        assert metadata["session_id"] == "sess_1"


@pytest.mark.asyncio
class TestServiceRequest:
    async def test_sends_contents_not_references(self, wired):
        fragments, pocket = await stored_refs(wired["storage"])

        await call(fragments, ["CCO", "CCC"], pocket_ref=pocket)

        sent = wired["calls"][0]
        assert sent["endpoint"] == "/constrained-pose"
        assert sent["json"]["fragment_sdfs"][0].startswith("benzene")
        assert sent["json"]["pocket_pdb"].startswith("ATOM")
        assert sent["json"]["smiles"] == ["CCO", "CCC"]

    async def test_a_missing_pocket_is_sent_as_empty_not_omitted(self, wired):
        fragments, _pocket = await stored_refs(wired["storage"])

        await call(fragments)

        assert wired["calls"][0]["json"]["pocket_pdb"] == ""

    async def test_thresholds_are_only_sent_when_given(self, wired):
        fragments, _pocket = await stored_refs(wired["storage"])

        await call(fragments)
        assert "strain_max" not in wired["calls"][0]["json"]

        await call(fragments, strain_max=40.0, rmsd_max=1.5)
        assert wired["calls"][1]["json"]["strain_max"] == 40.0
        assert wired["calls"][1]["json"]["rmsd_max"] == 1.5

    async def test_poses_are_always_asked_of_the_service(self, wired):
        # Storing is the caller's choice; the route needs the pose in hand to be
        # able to honour it.
        fragments, _pocket = await stored_refs(wired["storage"])

        await call(fragments)

        assert wired["calls"][0]["json"]["include_poses"] is True

    async def test_a_batch_gets_more_time_than_a_default_service_call(self, wired):
        fragments, _pocket = await stored_refs(wired["storage"])

        await call(fragments)

        assert wired["calls"][0]["timeout"] > 30.0


@pytest.mark.asyncio
class TestRejectedRequests:
    async def test_every_reference_is_checked_against_the_session(self, wired):
        # Checking only the first fragment would let sandbox code read a second
        # one belonging to another conversation.
        fragments, pocket = await stored_refs(wired["storage"])

        await call(fragments, pocket_ref=pocket)

        assert wired["manager"].checked == fragments + [pocket]

    async def test_a_malformed_reference_is_a_400(self, wired):
        with pytest.raises(HTTPException) as exc:
            await call(["not-a-reference"])
        assert exc.value.status_code == 400

    async def test_a_missing_artifact_is_a_404(self, wired):
        with pytest.raises(HTTPException) as exc:
            await call(["sdf_deadbeef"])
        assert exc.value.status_code == 404

    async def test_an_unknown_session_is_a_401(self, wired, monkeypatch):
        manager = FakeSessionManager(error=SessionNotFoundError("no such session"))
        monkeypatch.setattr(routes, "get_session_manager", lambda: manager)
        fragments, _pocket = await stored_refs(wired["storage"])

        with pytest.raises(HTTPException) as exc:
            await call(fragments)

        assert exc.value.status_code == 401

    async def test_a_reference_outside_the_session_is_a_403(self, wired, monkeypatch):
        manager = FakeSessionManager(error=ReferenceNotAllowedError("not yours"))
        monkeypatch.setattr(routes, "get_session_manager", lambda: manager)
        fragments, _pocket = await stored_refs(wired["storage"])

        with pytest.raises(HTTPException) as exc:
            await call(fragments)

        assert exc.value.status_code == 403

    async def test_an_empty_candidate_list_is_refused_by_the_model(self):
        with pytest.raises(ValueError):
            ConstrainedPoseSandboxRequest(fragment_refs=["sdf_abc12345"], smiles=[])

    async def test_the_batch_is_bounded(self):
        with pytest.raises(ValueError):
            ConstrainedPoseSandboxRequest(
                fragment_refs=["sdf_abc12345"], smiles=["CCO"] * 500
            )


@pytest.mark.asyncio
class TestFailuresReachTheSandbox:
    """A failure returns success=False rather than raising.

    The helper turns these into an AevomError carrying the service's own
    message; an HTTPException would surface as a bare status code instead.
    """

    async def test_a_service_error_is_reported(self, wired):
        wired["payload"]["response"] = {
            "success": False,
            "error": "fragment 1 has no 3D coordinates",
        }
        fragments, _pocket = await stored_refs(wired["storage"])

        response = await call(fragments)

        assert response.success is False
        assert "no 3D coordinates" in response.error
        assert response.results is None

    async def test_a_transport_failure_is_reported(self, wired):
        wired["payload"]["response"] = RuntimeError("connection refused")
        fragments, _pocket = await stored_refs(wired["storage"])

        response = await call(fragments)

        assert response.success is False
        assert "connection refused" in response.error
