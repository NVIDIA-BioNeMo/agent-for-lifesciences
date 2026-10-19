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

"""Fixtures shared by every backend test suite (api/tests, api/tools/tests, api/training/tests)."""

import pytest
from opentelemetry import trace


@pytest.fixture(autouse=True, scope="session")
def _no_phoenix_export():
    """Keep tests from initialising the real Phoenix exporter.

    ``get_tracer()`` would otherwise call ``phoenix.otel.register()``, whose
    default span processor exports synchronously to localhost:4317. With no
    collector there, every span end blocks the event loop for seconds of gRPC
    retries, and the processor stays installed for the rest of the session.
    Tests that need spans install their own in-memory provider over this.
    """
    import api.telemetry as telemetry

    if not telemetry._initialized:
        telemetry._tracer = trace.NoOpTracer()
        telemetry._initialized = True
    yield
