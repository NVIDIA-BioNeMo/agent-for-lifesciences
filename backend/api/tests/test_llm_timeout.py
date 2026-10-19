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

"""Tests for the LLM client's HTTP timeout configuration.

The OpenAI SDK defaults to a 5 second connect timeout. That is shorter than
the TCP handshake to inference-api.nvidia.com takes under congestion, which
made a healthy endpoint fail every call. These tests pin an explicit, longer
connect budget so the default cannot silently come back.
"""

import httpx

from api.llm.client import (
    LLM_CONNECT_TIMEOUT_S,
    LLM_READ_TIMEOUT_S,
    LLM_TIMEOUT,
)


class TestTimeoutConstants:
    """The values themselves."""

    def test_timeout_is_an_httpx_timeout(self):
        assert isinstance(LLM_TIMEOUT, httpx.Timeout)

    def test_connect_budget_exceeds_the_sdk_default(self):
        """5s is the SDK default and is what broke; we must be well above it."""
        assert LLM_CONNECT_TIMEOUT_S > 5.0

    def test_connect_budget_covers_an_observed_slow_handshake(self):
        """A 16.5s handshake was measured on 2026-08-04; leave headroom."""
        assert LLM_CONNECT_TIMEOUT_S >= 25.0

    def test_read_budget_absorbs_queueing(self):
        """That host has queued for ~200s before emitting a first token."""
        assert LLM_READ_TIMEOUT_S >= 300.0

    def test_timeout_fields_match_the_constants(self):
        assert LLM_TIMEOUT.connect == LLM_CONNECT_TIMEOUT_S
        assert LLM_TIMEOUT.read == LLM_READ_TIMEOUT_S


class TestClientWiring:
    """The timeout must actually reach the constructed OpenAI client."""

    def test_client_uses_the_configured_timeout(self, monkeypatch, tmp_path):
        key_file = tmp_path / "key"
        key_file.write_text("test-key")

        from api.llm import client as client_module
        from api.llm.config import get_model_config

        config = get_model_config("nemotron-3-ultra")
        monkeypatch.setattr(config, "api_key_file", str(key_file))

        llm = client_module.LLMClient(model_config=config)

        assert llm._client.timeout == LLM_TIMEOUT
        assert llm._client.timeout.connect == LLM_CONNECT_TIMEOUT_S

    def test_connect_timeout_is_not_the_sdk_default(self, monkeypatch, tmp_path):
        """Regression guard for the exact failure: connect=5.0."""
        key_file = tmp_path / "key"
        key_file.write_text("test-key")

        from api.llm import client as client_module
        from api.llm.config import get_model_config

        config = get_model_config("nemotron-3-ultra")
        monkeypatch.setattr(config, "api_key_file", str(key_file))

        llm = client_module.LLMClient(model_config=config)

        assert llm._client.timeout.connect != 5.0
