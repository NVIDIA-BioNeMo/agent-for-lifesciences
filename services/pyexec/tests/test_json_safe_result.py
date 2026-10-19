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

"""Tests for sanitizing ``result`` before it reaches the JSON response.

A sandboxed model can assign anything to ``result`` -- a numpy scalar, an
ndarray, an RDKit ``Mol``, a set. FastAPI/Pydantic can't serialize most of
those, so ``ExecuteResponse`` crashed with an unhandled 500 *after* the code
had already run successfully -- the model saw a bare "Server error" instead
of its actual result.

Two layers are tested:

1. ``sandbox._json_safe`` / ``execute_code`` -- the root-cause fix. A
   sanitized ``result`` should always be JSON-native.
2. The FastAPI handler's defensive fallback -- even if a non-JSON-safe value
   somehow reaches ``ExecuteResponse`` construction, the endpoint returns a
   structured ``success: false`` response, not a raw 500.
"""

import pytest

pytest.importorskip("numpy", reason="numpy is provided in the sandbox namespace")

import numpy as np
from fastapi.testclient import TestClient

from app import main as pyexec_main
from app.main import app
from app.sandbox import _json_safe, execute_code


class TestJsonSafeHelper:
    def test_passes_through_json_native_scalars(self):
        assert _json_safe(None) is None
        assert _json_safe(True) is True
        assert _json_safe(5) == 5
        assert _json_safe(1.5) == 1.5
        assert _json_safe("hi") == "hi"

    def test_unwraps_numpy_scalars(self):
        for value, expected_type in (
            (np.int64(5), int),
            (np.float64(1.5), float),
            (np.bool_(True), bool),
        ):
            safe = _json_safe(value)
            assert type(safe) is expected_type
            assert safe == value

    def test_converts_ndarray_to_list(self):
        assert _json_safe(np.array([1, 2, 3])) == [1, 2, 3]
        assert all(type(v) is int for v in _json_safe(np.array([1, 2, 3])))

    def test_recurses_into_nested_containers(self):
        safe = _json_safe({"count": np.int64(2), "vals": np.array([1.0, 2.0])})
        assert safe == {"count": 2, "vals": [1.0, 2.0]}
        assert type(safe["count"]) is int

    def test_converts_set_to_list(self):
        safe = _json_safe({1, 2, 3})
        assert isinstance(safe, list)
        assert sorted(safe) == [1, 2, 3]

    def test_opaque_object_becomes_a_string_instead_of_raising(self):
        class Opaque:
            def __repr__(self):
                return "<Opaque!>"

        safe = _json_safe(Opaque())
        assert isinstance(safe, str)


class TestExecuteCodeSanitizesResult:
    def test_numpy_scalar_result_comes_back_json_native(self):
        outcome = execute_code("import numpy as np\nresult = np.int64(5)")
        assert outcome["success"] is True
        assert outcome["result"] == 5
        assert type(outcome["result"]) is int

    def test_numpy_array_result_comes_back_as_a_list(self):
        outcome = execute_code("import numpy as np\nresult = np.array([1, 2, 3])")
        assert outcome["success"] is True
        assert outcome["result"] == [1, 2, 3]

    def test_rdkit_mol_result_does_not_crash(self):
        pytest.importorskip("rdkit", reason="rdkit lives in the base image")
        outcome = execute_code(
            "from rdkit import Chem\nresult = Chem.MolFromSmiles('CCO')"
        )
        assert outcome["success"] is True
        assert isinstance(outcome["result"], str)


class TestHandlerFallbackNeverReturns500:
    def test_a_value_that_reaches_the_handler_unsanitized_still_returns_200(
        self, monkeypatch
    ):
        """Simulates layer 1 missing a case: execute_code returns a raw
        numpy scalar (as it would have before _json_safe existed). The
        handler's own try/except around ExecuteResponse(...) construction
        must degrade to a structured error, not an unhandled 500.
        """

        def _fake_execute_code(code, context=None, timeout_seconds=10, sandbox_context=None):
            return {
                "success": True,
                "output": "42\n",
                "error": None,
                "result": np.int64(42),  # deliberately unsanitized
            }

        monkeypatch.setattr(pyexec_main, "execute_code", _fake_execute_code)

        client = TestClient(app)
        r = client.post("/execute", json={"code": "result = 42"})

        assert r.status_code == 200
        body = r.json()
        assert body["success"] is False
        assert body["result"] is None
        assert "result" in body["error"].lower()
