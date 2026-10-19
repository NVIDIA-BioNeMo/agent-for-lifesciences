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

"""Tests for the post-2026-05-28 ``[AEVOM-PYEXEC-BACKPRESSURE]`` alarm.

Per the v0527c post-mortem, when the sandbox's offered load exceeds the
``workers: N`` pool size in ``training/configs/services.yaml``, requests
queue at the kernel/uvicorn layer and individual ``/execute`` calls can
take 60-460 s instead of 5 s. That fired a positive feedback loop with
the AsyncTrajectoryCollector and collapsed trainer step pacing.

These tests verify the loud, structured warning the server now emits
when it observes either:

1. a ``wire_wait_s`` exceeding the threshold (kernel-queue wait,
   computed from the new ``x-aevom-client-send-ns`` client-injected
   header), or
2. a total handler lifetime exceeding ``timeout_seconds + 10 s``
   (catches saturation even when the client header is missing).

The warning is intentionally *not* paired with any throttle or
semaphore: the right response to backpressure is to scale workers, not
silently mask the saturation.
"""

import logging
import time

import pytest
from fastapi.testclient import TestClient

from app import main as pyexec_main
from app.main import app


def _reset_counters() -> None:
    pyexec_main._in_flight_count = 0
    pyexec_main._in_flight_max_seen = 0
    pyexec_main._backpressure_warn_count = 0


@pytest.fixture
def stub_execute_code(monkeypatch):
    """Stub out the synchronous ``execute_code`` body.

    The real ``execute_code`` uses ``signal.SIGALRM`` for the timeout
    guard, which only works on the main thread of the main interpreter.
    Starlette's ``TestClient`` runs handlers in a worker thread, which
    means production-shaped requests crash with ``ValueError: signal
    only works in main thread`` before our handler-level
    instrumentation under test gets a chance to log/return. Bypassing
    the body here lets us exercise the FastAPI handler -- which is
    where the backpressure-warn logic lives -- without simulating the
    sandbox.
    """

    def _fake_execute_code(code, context=None, timeout_seconds=10, sandbox_context=None):
        return {
            "success": True,
            "output": "",
            "error": None,
            "result": None,
        }

    monkeypatch.setattr(pyexec_main, "execute_code", _fake_execute_code)


class TestWireWaitWarning:
    def test_wire_wait_above_threshold_logs_loud_warning(
        self, caplog, monkeypatch, stub_execute_code
    ):
        """Client header reports a 30s-old send timestamp -> wire_wait
        easily clears the 5s threshold -> we expect a loud warning."""
        _reset_counters()
        monkeypatch.setattr(pyexec_main, "BACKPRESSURE_WIRE_WAIT_WARN_S", 5.0)

        client = TestClient(app)
        ten_s_ago_ns = time.time_ns() - 30 * 1_000_000_000

        with caplog.at_level(logging.WARNING, logger="app.main"):
            r = client.post(
                "/execute",
                json={"code": "result = 1 + 1", "timeout_seconds": 5},
                headers={"x-aevom-client-send-ns": str(ten_s_ago_ns)},
            )

        assert r.status_code == 200
        assert r.json()["success"] is True
        backpressure = [
            rec.getMessage()
            for rec in caplog.records
            if "[AEVOM-PYEXEC-BACKPRESSURE]" in rec.getMessage()
        ]
        assert backpressure, (
            "expected [AEVOM-PYEXEC-BACKPRESSURE] in logs; "
            f"got {[rec.getMessage() for rec in caplog.records]}"
        )
        # Warn count tracks for /health observability.
        assert pyexec_main._backpressure_warn_count >= 1

    def test_wire_wait_below_threshold_logs_nothing(
        self, caplog, monkeypatch, stub_execute_code
    ):
        """Just-issued request (wire_wait near 0) must not emit the
        warning -- otherwise we would spam the steady-state log."""
        _reset_counters()
        monkeypatch.setattr(pyexec_main, "BACKPRESSURE_WIRE_WAIT_WARN_S", 5.0)
        # Set lifetime threshold high so the fallback branch can't fire
        # for a 1+1 program.
        monkeypatch.setattr(pyexec_main, "BACKPRESSURE_LIFETIME_WARN_S", 60.0)

        client = TestClient(app)
        with caplog.at_level(logging.WARNING, logger="app.main"):
            r = client.post(
                "/execute",
                json={"code": "result = 2 + 2", "timeout_seconds": 5},
                headers={"x-aevom-client-send-ns": str(time.time_ns())},
            )

        assert r.status_code == 200
        backpressure = [
            rec.getMessage()
            for rec in caplog.records
            if "[AEVOM-PYEXEC-BACKPRESSURE]" in rec.getMessage()
        ]
        assert not backpressure, (
            f"unexpected backpressure warning on fast path: {backpressure}"
        )
        assert pyexec_main._backpressure_warn_count == 0

    def test_missing_header_does_not_break_handler(
        self, caplog, monkeypatch, stub_execute_code
    ):
        """Legacy clients without the header still succeed; wire_wait is
        treated as 0 (no header => we can't measure queue wait)."""
        _reset_counters()
        monkeypatch.setattr(pyexec_main, "BACKPRESSURE_WIRE_WAIT_WARN_S", 5.0)
        monkeypatch.setattr(pyexec_main, "BACKPRESSURE_LIFETIME_WARN_S", 60.0)

        client = TestClient(app)
        with caplog.at_level(logging.WARNING, logger="app.main"):
            r = client.post(
                "/execute",
                json={"code": "result = 3 + 3", "timeout_seconds": 5},
            )

        assert r.status_code == 200
        assert r.json()["success"] is True


class TestHealthEndpoint:
    def test_health_reports_backpressure_counters(self):
        """``/health`` must surface the per-worker counters so post-mortem
        scripts and live monitors can read backpressure state without
        Phoenix."""
        client = TestClient(app)
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["service"] == "pyexec"
        for k in (
            "in_flight",
            "in_flight_max_seen",
            "backpressure_warn_count",
            "pid",
        ):
            assert k in body, f"/health missing {k}: {body}"
