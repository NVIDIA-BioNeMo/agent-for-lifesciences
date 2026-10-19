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

"""``PYEXEC_CONNECTION_CLOSE`` and the OpenShell relay stall.

Behind the OpenShell gateway, a connection uvicorn closes while idle is never
reported back through the relay. The gateway keeps it pooled, and a request
sent down it 5-20 s later hangs until the caller gives up. The sandbox image
sets the flag so that every response tells the gateway not to reuse the
connection. Direct mode and training leave it off.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from app import main as pyexec_main
from app.main import app


def _fake_execute_code(code, context=None, timeout_seconds=10, sandbox_context=None):
    return {"success": True, "output": "", "error": None, "result": None}


def test_responses_close_the_connection_when_enabled(monkeypatch):
    monkeypatch.setattr(pyexec_main, "CONNECTION_CLOSE", True)
    monkeypatch.setattr(pyexec_main, "execute_code", _fake_execute_code)
    client = TestClient(app)

    assert client.get("/health").headers["connection"] == "close"
    response = client.post("/execute", json={"code": "result = 1", "timeout_seconds": 5})
    assert response.status_code == 200
    assert response.headers["connection"] == "close"


def test_a_rejected_request_also_closes_the_connection(monkeypatch):
    monkeypatch.setattr(pyexec_main, "CONNECTION_CLOSE", True)
    response = TestClient(app).post("/execute", json={"timeout_seconds": 5})
    assert response.status_code == 422
    assert response.headers["connection"] == "close"


def test_connections_are_left_alone_when_disabled(monkeypatch):
    monkeypatch.setattr(pyexec_main, "CONNECTION_CLOSE", False)
    assert "connection" not in TestClient(app).get("/health").headers


def test_the_sandbox_image_enables_it():
    dockerfile = Path(__file__).resolve().parents[1] / "Dockerfile.openshell"
    assert "PYEXEC_CONNECTION_CLOSE=1" in dockerfile.read_text()
