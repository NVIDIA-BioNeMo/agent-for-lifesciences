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

"""Shared fixtures for the Vina service tests.

``gpu_worker`` tests need the real nvMolKit worker and a GPU. They skip where
the worker cannot start, unless ``AEVOM_REQUIRE_GPU_WORKER=1`` (set by
``scripts/verify-gpu-conformers.sh``), in which case an unready worker, a CPU
fallback, or a run in which no ``gpu_worker`` test passed is a failure.
"""

import os
import sys
from pathlib import Path

import pytest

TESTS = Path(__file__).resolve().parent
REQUIRE_GPU_WORKER = os.environ.get("AEVOM_REQUIRE_GPU_WORKER") == "1"

_gpu_worker_passed = 0


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu_worker: needs the real nvMolKit conformer worker and a GPU")


def pytest_runtest_logreport(report):
    global _gpu_worker_passed
    if report.when == "call" and report.passed and "gpu_worker" in report.keywords:
        _gpu_worker_passed += 1


def pytest_sessionfinish(session, exitstatus):
    if REQUIRE_GPU_WORKER and exitstatus == 0 and _gpu_worker_passed == 0:
        sys.stderr.write("\nAEVOM_REQUIRE_GPU_WORKER=1 but no gpu_worker test passed\n")
        session.exitstatus = 1


@pytest.fixture
def gpu_worker():
    """The process-wide worker client, ready, or skip (fail when required)."""
    from app import conformer_client

    conformer_client.reset_client()
    client = conformer_client.get_client()
    try:
        client.ensure_ready()
    except conformer_client.ConformerWorkerError as exc:
        conformer_client.reset_client()
        if REQUIRE_GPU_WORKER:
            pytest.fail(f"GPU conformer worker not ready: {exc}")
        pytest.skip(f"GPU conformer worker unavailable: {exc}")
    yield client
    conformer_client.reset_client()
