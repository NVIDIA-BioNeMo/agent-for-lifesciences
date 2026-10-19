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

"""Tests for the execution timeout budget the /execute endpoint accepts.

The ceiling here has to be no lower than the longest budget the caller will ask
for. When it was 60s while ``python_execute`` clamped training runs to 180s,
every such call was rejected with a 422 before any code ran, surfacing to the
model as an unexplained failure rather than a timeout.
"""

import pytest
from fastapi.testclient import TestClient

from app import main as pyexec_main
from app.main import DEFAULT_TIMEOUT_SECONDS, MAX_TIMEOUT_SECONDS, app

# The clamps in backend/api/tools/python_exec.py. Duplicated rather than imported
# because the backend package is not installed in this service's image.
CALLER_INTERACTIVE_MAX = 120
CALLER_TRAINING_MAX = 180


@pytest.fixture
def client(monkeypatch):
    """A client whose requests stop at the validation boundary.

    ``execute_code`` arms a SIGALRM, which only works on the main thread, and
    TestClient runs handlers in a worker thread. Since what matters here is which
    budgets the endpoint accepts rather than what the sandbox does with them, the
    body is stubbed out -- same reasoning as test_backpressure_warn.py.
    """

    def _fake_execute_code(code, context=None, timeout_seconds=10, sandbox_context=None):
        return {"success": True, "output": "", "error": None, "result": None}

    monkeypatch.setattr(pyexec_main, "execute_code", _fake_execute_code)
    return TestClient(app)


def post(client, **body):
    body.setdefault("code", "result = 1")
    return client.post("/execute", json=body)


class TestAcceptedBudgets:
    @pytest.mark.parametrize(
        "seconds",
        [1, 10, 30, 60, CALLER_INTERACTIVE_MAX, CALLER_TRAINING_MAX],
    )
    def test_every_budget_the_caller_can_ask_for_is_accepted(self, client, seconds):
        assert post(client, timeout_seconds=seconds).status_code == 200

    def test_the_ceiling_covers_the_training_clamp(self):
        # The regression this file exists for: a caller-side clamp above the
        # service ceiling turns into a 422 rather than a longer run.
        assert MAX_TIMEOUT_SECONDS >= CALLER_TRAINING_MAX
        assert MAX_TIMEOUT_SECONDS >= CALLER_INTERACTIVE_MAX

    def test_omitting_the_budget_uses_the_default(self, client):
        assert post(client).status_code == 200
        assert DEFAULT_TIMEOUT_SECONDS <= MAX_TIMEOUT_SECONDS


class TestRejectedBudgets:
    def test_above_the_ceiling_is_rejected(self, client):
        assert post(client, timeout_seconds=MAX_TIMEOUT_SECONDS + 1).status_code == 422

    def test_zero_is_rejected(self, client):
        assert post(client, timeout_seconds=0).status_code == 422

    def test_negative_is_rejected(self, client):
        assert post(client, timeout_seconds=-5).status_code == 422


class TestReportedLimits:
    def test_info_reports_the_real_limits(self, client):
        # The /info endpoint is how the limits are discovered, so a hardcoded
        # copy drifting from the model is worse than not reporting them.
        limits = client.get("/info").json()["limits"]
        assert limits["max_timeout_seconds"] == MAX_TIMEOUT_SECONDS
        assert limits["default_timeout_seconds"] == DEFAULT_TIMEOUT_SECONDS

    def test_info_reports_scikit_learn(self, client):
        # Sandbox code may import it, so its absence should be visible here
        # rather than as an ImportError at execution time.
        assert client.get("/info").json()["libraries"]["scikit-learn"]
