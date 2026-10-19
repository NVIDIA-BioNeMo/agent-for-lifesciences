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

"""Tests for the python_execute tool's budgets.

The service is stubbed throughout; what the sandbox does with the code is covered
in services/pyexec. What is covered here is the two budgets the tool decides:
how long the code may run, and how much of its output the model gets to see.

The timeout clamp is worth pinning because it was silently broken. The tool asked
for 180s during training runs while the pyexec service rejected anything above
60s, so those calls came back as a validation error before any code ran.
"""

import sys
import types
from unittest.mock import AsyncMock

import pytest

from api.storage.scoped import SessionScopedStorage
from api.tools.base import ContentType
from api.tools.python_exec import (
    DEFAULT_TIMEOUT_SECONDS,
    HANDWRITTEN_STRUCTURES,
    MAX_SESSION_REFS,
    MAX_TIMEOUT_SECONDS,
    TRAINING_MAX_TIMEOUT_SECONDS,
    PythonExecuteTool,
    _handwritten_structures,
    sandbox_denial_note,
)

# The ceiling enforced by services/pyexec/app/main.py, which rejects anything
# higher with a 422. Duplicated because the service is not importable here.
SERVICE_MAX_TIMEOUT_SECONDS = 180


@pytest.fixture(autouse=True)
def stub_redis(monkeypatch):
    """Stand in for api.redis, which execute() imports to find selections.

    The real module needs the redis client library, which is not installed for
    the backend test suite. Nothing here passes a selection_ref or uses the
    `selection` variable, so the connection is never touched.
    """
    module = types.ModuleType("api.redis")

    async def _get_redis():
        return None

    module.get_redis = _get_redis
    monkeypatch.setitem(sys.modules, "api.redis", module)


@pytest.fixture
def tool():
    return PythonExecuteTool()


def stub(tool, calls: list) -> None:
    """Record the request the tool would send, and report success."""

    async def _call(endpoint, method="POST", json_data=None, timeout=30.0, retries=0):
        calls.append({"body": json_data, "timeout": timeout})
        return {"success": True, "output": "ok", "error": None, "result": None}

    tool._call_service = _call


async def run(tool, calls, **arguments):
    stub(tool, calls)
    arguments.setdefault("code", "result = 1")
    return await tool.execute(arguments)


class TestTimeoutBudget:
    @pytest.mark.asyncio
    async def test_unspecified_uses_the_default(self, tool):
        calls: list = []
        await run(tool, calls)
        assert calls[0]["body"]["timeout_seconds"] == DEFAULT_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_a_caller_value_is_passed_through(self, tool):
        calls: list = []
        await run(tool, calls, timeout_seconds=45)
        assert calls[0]["body"]["timeout_seconds"] == 45

    @pytest.mark.asyncio
    async def test_above_the_ceiling_is_clamped(self, tool):
        calls: list = []
        await run(tool, calls, timeout_seconds=9999)
        assert calls[0]["body"]["timeout_seconds"] == MAX_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_below_one_is_clamped(self, tool):
        calls: list = []
        await run(tool, calls, timeout_seconds=0)
        assert calls[0]["body"]["timeout_seconds"] == 1

    @pytest.mark.asyncio
    async def test_training_runs_get_a_higher_ceiling(self, tool, monkeypatch):
        # A timeout during a rollout costs the whole episode rather than one
        # retryable turn, so training is allowed to wait longer.
        monkeypatch.setenv("AEVOM_TRAINING_RUN_ID", "run-1")
        calls: list = []
        await run(tool, calls, timeout_seconds=9999)
        assert calls[0]["body"]["timeout_seconds"] == TRAINING_MAX_TIMEOUT_SECONDS
        assert TRAINING_MAX_TIMEOUT_SECONDS > MAX_TIMEOUT_SECONDS

    def test_no_ceiling_exceeds_what_the_service_accepts(self):
        # The regression this file exists for. Either ceiling above the
        # service's own limit turns every long call into a 422.
        assert MAX_TIMEOUT_SECONDS <= SERVICE_MAX_TIMEOUT_SECONDS
        assert TRAINING_MAX_TIMEOUT_SECONDS <= SERVICE_MAX_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_the_http_timeout_outlasts_the_execution_budget(self, tool):
        # If the client gave up first, the model would see a transport failure
        # instead of the timeout message, with no partial output.
        calls: list = []
        await run(tool, calls, timeout_seconds=60)
        assert calls[0]["timeout"] > 60

    def test_the_schema_states_the_real_range(self, tool):
        described = tool.get_schema()["function"]["parameters"]["properties"]
        description = described["timeout_seconds"]["description"]
        assert str(MAX_TIMEOUT_SECONDS) in description
        assert str(DEFAULT_TIMEOUT_SECONDS) in description


class TestResponseBudget:
    def test_output_stays_tighter_than_the_default(self, tool):
        # Sandbox output is unbounded by nature, so it keeps a tighter budget
        # than tools whose output size follows from their input.
        from api.tools.base import DEFAULT_MAX_RESPONSE_TOKENS

        assert tool.max_response_tokens < DEFAULT_MAX_RESPONSE_TOKENS

    def test_there_is_room_for_a_table_of_candidates(self, tool):
        # Fifty molecules with descriptors runs to roughly 6000 characters, and
        # at the previous allowance of 800 tokens it was truncated mid-table.
        from api.tools.base import CHARS_PER_TOKEN

        assert tool.max_response_tokens * CHARS_PER_TOKEN >= 6000


class TestSchemaDocumentsTheEnvironment:
    def test_sklearn_is_advertised(self, tool):
        # The model will not reach for a library it has been told is absent.
        assert "sklearn" in tool.get_schema()["function"]["description"]

    def test_the_sa_score_helper_is_advertised(self, tool):
        # Its whole purpose is being reachable when the manual route is not.
        assert "sa_score" in tool.get_schema()["function"]["description"]

    def test_the_blocked_sklearn_submodule_is_named(self, tool):
        description = tool.get_schema()["function"]["description"]
        assert "sklearn.datasets" in description

    def test_gates_is_named_as_the_better_route_for_a_descriptor_panel(self, tool):
        # Otherwise the model reimplements the panel in Python every time.
        assert "molecule_gates" in tool.get_schema()["function"]["description"]

    def test_the_biotite_model_1_gotcha_is_documented(self, tool):
        # get_structure() returns an AtomArrayStack even for one model, so an
        # atom-length boolean mask raises an axis-0 IndexError. This was the top
        # recurring sandbox failure; the fix is to always advertise model=1.
        description = tool.get_schema()["function"]["description"]
        assert "model=1" in description
        assert "extra_fields" in description


def _scoped_storage(session_id, refs):
    """A real SessionScopedStorage over a mocked DataStorageService.

    The wrapper is real so the tool is held to the interface it gets in
    production. A hand-written fake previously accepted a session_id keyword the
    wrapper does not, which hid a TypeError that emptied every sandbox allow-list.
    """
    listed = [
        {"reference": r, "data_type": r.split("_")[0], "description": None}
        for r in refs
    ]
    base = AsyncMock()
    base.list_artifacts.side_effect = (
        lambda limit=50, session_id=None: listed[:limit]
    )
    base.exists.return_value = True  # explicit/code-detected refs are validated
    # A scope needs a session, so without one the tool has no storage at all.
    if session_id is None:
        return None
    return SessionScopedStorage(base, session_id)


class _FakeSession:
    def __init__(self, allowed_refs):
        self.id = "tok-123"
        self.allowed_refs = set(allowed_refs)


class _FakeManager:
    def __init__(self):
        self.created_with = None

    async def create_session(self, allowed_refs, ttl_minutes=None,
                             max_accesses=None, client_session_id=None):
        self.created_with = set(allowed_refs)
        return _FakeSession(allowed_refs)

    async def get_and_clear_visualization_events(self, token):
        return []


class TestSessionWideRefs:
    """execute() grants the sandbox every ref in the current session.

    The old behaviour scoped access to only the refs that appeared literally in
    the code (regex) or in data_refs, so aevom.load() raised AccessDeniedError
    for any ref the model did not spell out, and storage could not be enumerated
    from inside the sandbox at all.
    """

    def _wire(self, tool, monkeypatch, session_id, session_refs):
        calls: list = []
        stub(tool, calls)
        tool._storage = _scoped_storage(session_id, session_refs)
        mgr = _FakeManager()
        monkeypatch.setattr(
            "api.tools.python_exec.get_session_manager", lambda: mgr
        )
        return calls, mgr

    def _granted(self, calls):
        return set(calls[0]["body"]["sandbox_context"]["allowed_refs"])

    @pytest.mark.asyncio
    async def test_all_session_refs_are_granted(self, tool, monkeypatch):
        calls, mgr = self._wire(tool, monkeypatch, "sess-A",
                                ["sdf_aaaaaaaa", "pdb_bbbbbbbb"])
        await tool.execute({"code": "x = aevom.allowed_refs"})
        assert self._granted(calls) == {"sdf_aaaaaaaa", "pdb_bbbbbbbb"}
        assert mgr.created_with == {"sdf_aaaaaaaa", "pdb_bbbbbbbb"}
        tool._storage._storage.list_artifacts.assert_called_once_with(
            MAX_SESSION_REFS, session_id="sess-A"
        )

    def test_underscored_type_refs_are_detected_in_code(self, tool):
        # annotated_cif_* was invisible to the old single-segment pattern, so
        # loading an RFD4 design from the sandbox was denied.
        code = (
            'cif = aevom.load("annotated_cif_919e0b2e")\n'
            'spec = aevom.load("design_spec_0a1b2c3d")\n'
            'seq = aevom.load("sequence_59231645")'
        )
        assert tool._detect_refs_in_code(code) == {
            "annotated_cif_919e0b2e",
            "design_spec_0a1b2c3d",
            "sequence_59231645",
        }

    @pytest.mark.asyncio
    async def test_ref_not_named_in_code_is_still_granted(self, tool, monkeypatch):
        # The exact regression: a stored ref the model never typed was denied.
        calls, _ = self._wire(tool, monkeypatch, "sess-A", ["sdf_deadbeef"])
        await tool.execute({"code": "for r in aevom.allowed_refs:\n    aevom.load(r)"})
        assert "sdf_deadbeef" in self._granted(calls)

    @pytest.mark.asyncio
    async def test_explicit_and_session_refs_union(self, tool, monkeypatch):
        calls, _ = self._wire(tool, monkeypatch, "sess-A", ["sdf_aaaaaaaa"])
        await tool.execute({
            "code": "aevom.load('pdb_cccccccc')",
            "data_refs": ["smiles_dddddddd"],
        })
        assert {"sdf_aaaaaaaa", "pdb_cccccccc", "smiles_dddddddd"} <= self._granted(calls)

    @pytest.mark.asyncio
    async def test_no_storage_context_without_session(self, tool, monkeypatch):
        # No session id -> nothing to enumerate; falls back to narrow scope.
        calls, _ = self._wire(tool, monkeypatch, None, ["sdf_aaaaaaaa"])
        await tool.execute({"code": "x = aevom.allowed_refs"})
        assert "sandbox_context" not in calls[0]["body"]

    @pytest.mark.asyncio
    async def test_list_helper_returns_empty_without_session(self, tool):
        assert await tool._list_session_refs(None) == set()

    @pytest.mark.asyncio
    async def test_list_helper_swallows_listing_errors(self, tool):
        class _Boom:
            session_id = "s"

            async def list_artifacts(self, limit=50):
                raise RuntimeError("db down")

        tool._storage = _Boom()
        assert await tool._list_session_refs("s") == set()

    def test_schema_advertises_session_wide_access(self, tool):
        desc = tool.get_schema()["function"]["parameters"]["properties"]
        assert "aevom.allowed_refs" in desc["data_refs"]["description"]


class TestUnchangedBehaviour:
    @pytest.mark.asyncio
    async def test_empty_code_is_rejected(self, tool):
        result = await tool.execute({"code": "   "})
        assert result.content_type == ContentType.ERROR


# Real structures, from the reference ligands of lead-optimisation tasks.
STRUCTURES = [
    "Cc1ncc(F)cc1-c1ccc2cc(NC(=O)C3CC3)ncc2c1",
    "CC(=O)Oc1ccccc1C(=O)O",
    "Cc1cnc2[nH]cc(C(=O)c3c(F)ccc(NS(=O)(=O)c4ccccc4F)c3F)c2c1",
    "CCC[C@H](C)Oc1cc(-n2nc(CO)n(C)c2=O)c(F)cc1C(=O)Nc1c(F)cccc1Cl",
    "CN(CCC(=O)N1CCN(CCNc2c3c(nc4ccccc24)CCCC3)CC1)c1ccc(OC)cc1",
]


class TestSpottingHandwrittenStructures:
    """Molecules written out as literals, rather than produced by a program.

    This is the largest silent loss in the sandbox: over 2759 structures typed
    into it across 19 lead-optimisation tasks, 11.5% did not parse and only half
    the rest stayed inside the task's own similarity bound. The loop that builds
    them almost always ends `if mol is None: continue`, so nothing records that
    a molecule existed or why it was dropped.

    The detector is deliberately crude -- the backend image has no RDKit, so it
    recognises the shape of a structure rather than a valid one, which is right
    when the ones that matter most are the invalid ones.
    """

    def test_it_counts_structures(self):
        code = "\n".join(f'candidates.append("{s}")' for s in STRUCTURES)

        assert _handwritten_structures(code) == len(STRUCTURES)

    def test_an_unparseable_structure_still_counts(self):
        # The whole point. A splicing error looks exactly like a molecule until
        # RDKit is asked, and RDKit is not available here.
        code = 'x = "Cc1cnc2[nH]cc(C(=O)c3c(F)ccc(NHS(=O)(=O)c4ccccc4F)c3F)c2c1"'

        assert _handwritten_structures(code) == 1

    @pytest.mark.parametrize(
        "literal",
        [
            "smiles_a5230680",              # an artifact reference
            "/eval/task/reference.sdf",     # a path
            "clearance_hepatocyte_az",      # a column name
            "the 4-methoxy analogue",       # prose
            "%(asctime)s %(message)s",      # a format string
        ],
    )
    def test_it_does_not_count_things_that_are_not_molecules(self, literal):
        assert _handwritten_structures(f'x = "{literal}"') == 0

    def test_a_word_without_a_ring_or_branch_is_not_a_molecule(self):
        assert _handwritten_structures('x = "CCCCCCCCCCCCCC"') == 0


class TestSayingSoWhereItHappens:
    @pytest.mark.asyncio
    async def test_a_series_written_out_by_hand_is_named(self, tool):
        calls: list = []
        code = "\n".join(f'c.append("{s}")' for s in STRUCTURES)

        result = await run(tool, calls, code=code)

        assert f"{len(STRUCTURES)} molecules" in result.data
        # It points at the checks, and does not steer away from writing them:
        # hand-written candidates are the more productive channel by 2.7x.
        assert "mol_from_smiles" in result.data
        assert "describe_molecule" in result.data
        assert "instead" not in result.data.split("2.7x")[0].split("[")[-1]

    @pytest.mark.asyncio
    async def test_naming_a_starting_point_is_left_alone(self, tool):
        # One or two structures is how anyone begins. The note is about writing
        # out a series, not about mentioning a molecule.
        calls: list = []
        code = f'lead = "{STRUCTURES[0]}"\nprint(lead)'

        result = await run(tool, calls, code=code)

        assert "mol_from_smiles" not in result.data

    @pytest.mark.asyncio
    async def test_a_block_that_generates_its_candidates_is_left_alone(
        self, tool, monkeypatch
    ):
        # Measured from a real run: a block calling enumerate_analogs and also
        # holding 123 literals was carrying the molecules it had already tested,
        # because there is nowhere else to keep them. The note would have told
        # it to use the tool it was using.
        calls: list = []
        # Mentioning `aevom` makes execute() open a sandbox session, so this is
        # the one test in the class that needs the manager stubbed.
        monkeypatch.setattr(
            "api.tools.python_exec.get_session_manager", lambda: _FakeManager()
        )
        code = (
            "analogs = aevom.enumerate_analogs(lead, similarity_min=0.7)\n"
            + "tested = [\n"
            + ",\n".join(f'    "{s}"' for s in STRUCTURES)
            + "\n]\n"
        )

        result = await run(tool, calls, code=code)

        assert "mol_from_smiles" not in result.data

    @pytest.mark.asyncio
    async def test_the_threshold_is_where_it_says_it_is(self, tool):
        calls: list = []
        below = "\n".join(
            f'c.append("{s}")' for s in STRUCTURES[: HANDWRITTEN_STRUCTURES - 1]
        )

        result = await run(tool, calls, code=below)

        assert "mol_from_smiles" not in result.data


class _FakeOpenShell:
    def __init__(self, error=None):
        self.error = error
        self.asked_for: list = []

    async def get_endpoint(self, session_id, timeout=None):
        from api.sandbox.openshell_manager import SandboxEndpoint

        self.asked_for.append(session_id)
        if self.error:
            raise self.error
        return SandboxEndpoint(
            name="pyexec-abc",
            base_url="http://openshell-gateway:8090",
            headers={"Host": "default--pyexec-abc.openshell.localhost:8090"},
        )


class TestOpenShellRouting:
    """Under PYEXEC_MODE=openshell each call goes to the session's own sandbox."""

    def _wire(self, tool, monkeypatch, response=None, error=None):
        monkeypatch.setenv("PYEXEC_MODE", "openshell")
        calls: list = []

        async def _call(endpoint, method="POST", json_data=None, timeout=30.0, retries=0, **route):
            calls.append({"body": json_data, **route})
            return response or {"success": True, "output": "ok", "error": None, "result": None}

        tool._call_service = _call
        tool._storage = _scoped_storage("sess-A", [])
        fake = _FakeOpenShell(error)
        monkeypatch.setattr("api.tools.python_exec.get_openshell_manager", lambda: fake)
        return calls, fake

    @pytest.mark.asyncio
    async def test_the_call_goes_to_the_sessions_sandbox(self, tool, monkeypatch):
        calls, fake = self._wire(tool, monkeypatch)
        await tool.execute({"code": "result = 1"})
        assert fake.asked_for == ["sess-A"]
        assert calls[0]["base_url"] == "http://openshell-gateway:8090"
        assert calls[0]["headers"] == {"Host": "default--pyexec-abc.openshell.localhost:8090"}

    @pytest.mark.asyncio
    async def test_an_unavailable_sandbox_fails_closed(self, tool, monkeypatch):
        from api.sandbox.openshell_manager import SandboxUnavailableError

        calls, _ = self._wire(tool, monkeypatch, error=SandboxUnavailableError("all in use"))
        result = await tool.execute({"code": "result = 1"})
        assert result.content_type == ContentType.ERROR
        assert "Python sandbox unavailable: all in use" in result.data
        assert calls == []

    @pytest.mark.asyncio
    async def test_a_blocked_request_is_explained(self, tool, monkeypatch):
        response = {
            "success": False,
            "output": "",
            "error": "URLError: <urlopen error [Errno 13] Permission denied>",
            "result": None,
        }
        self._wire(tool, monkeypatch, response=response)
        result = await tool.execute({"code": "import urllib.request"})
        assert "PubChem and RCSB search" in result.data

    @pytest.mark.asyncio
    async def test_a_sandbox_timeout_names_the_exception_and_limit(self, tool, monkeypatch):
        import httpx

        self._wire(tool, monkeypatch)

        async def _timeout(*args, **kwargs):
            raise httpx.ReadTimeout("")

        tool._call_service = _timeout
        result = await tool.execute({"code": "result = 1", "timeout_seconds": 30})
        assert result.content_type == ContentType.ERROR
        assert result.data.startswith("Sandbox did not respond within 40s (ReadTimeout).")
        assert "own 30s limit" in result.data

    @pytest.mark.asyncio
    async def test_an_exception_without_a_message_is_named(self, tool, monkeypatch):
        self._wire(tool, monkeypatch)

        async def _fail(*args, **kwargs):
            raise ConnectionResetError()

        tool._call_service = _fail
        result = await tool.execute({"code": "result = 1"})
        assert result.data == "Failed to execute code: ConnectionResetError"

    @pytest.mark.asyncio
    async def test_direct_mode_is_unchanged(self, tool, monkeypatch):
        calls, fake = self._wire(tool, monkeypatch)
        monkeypatch.setenv("PYEXEC_MODE", "direct")
        await tool.execute({"code": "result = 1"})
        assert fake.asked_for == []
        assert "base_url" not in calls[0]


class TestSandboxDenialNote:
    def test_a_refused_connect_is_a_network_denial(self):
        assert "network request" in sandbox_denial_note("<urlopen error [Errno 13] Permission denied>")

    def test_a_proxy_403_is_a_network_denial(self):
        assert "network request" in sandbox_denial_note("Tunnel connection failed: 403 Forbidden")

    def test_a_blocked_write_is_a_filesystem_denial(self):
        note = sandbox_denial_note("PermissionError: [Errno 13] Permission denied: '/app/x.txt'")
        assert "filesystem write" in note
        assert "network request" not in note

    def test_ordinary_errors_get_no_note(self):
        assert sandbox_denial_note("ValueError: bad SMILES") == ""


class TestOpenShellSpanAttributes:
    @pytest.mark.asyncio
    async def test_the_tool_span_names_the_sandbox(self, tool, monkeypatch):
        recorded = {}

        class _Span:
            def set_attribute(self, key, value):
                recorded[key] = value

        monkeypatch.setattr("api.tools.python_exec.trace.get_current_span", lambda: _Span())
        TestOpenShellRouting()._wire(tool, monkeypatch)
        await tool.execute({"code": "result = 1"})
        assert recorded == {"sandbox.name": "pyexec-abc", "sandbox.wait_ms": 0.0, "sandbox.cold_start": False}
