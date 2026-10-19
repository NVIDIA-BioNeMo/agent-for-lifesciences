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

"""Tests for how an explicit temperature reaches the provider.

`temperature or self.config.default_temperature` treats 0.0 as absent, so every
caller asking for greedy decoding silently got the model default instead. Skill
contract extraction is the caller that cares: it asks for 0.0 to make install
reproducible, and ran at 0.7 for as long as the bug existed.
"""

import pytest

from api.llm import client as client_module
from api.llm.config import get_model_config


class _FakeMessage:
    content = '{"endpoints": []}'
    reasoning_content = None
    tool_calls = None


class _FakeChoice:
    message = _FakeMessage()
    finish_reason = "stop"


class _FakeCompletion:
    choices = [_FakeChoice()]
    usage = None


@pytest.fixture
def llm(monkeypatch, tmp_path):
    """A client whose provider call records the kwargs it was given."""
    key_file = tmp_path / "key"
    key_file.write_text("test-key")

    config = get_model_config("nemotron-3-ultra")
    monkeypatch.setattr(config, "api_key_file", str(key_file))
    instance = client_module.LLMClient(model_config=config)

    seen: dict = {}

    def _create(**kwargs):
        seen.update(kwargs)
        return _FakeCompletion()

    monkeypatch.setattr(instance._client.chat.completions, "create", _create)
    instance.seen = seen
    return instance


class TestCompleteTemperature:
    def test_zero_survives_as_zero(self, llm):
        """The regression guard: 0.0 must not fall through to the default."""
        llm.complete(messages=[{"role": "user", "content": "hi"}], temperature=0.0)
        assert llm.seen["temperature"] == 0.0

    def test_an_explicit_nonzero_value_is_used(self, llm):
        llm.complete(messages=[{"role": "user", "content": "hi"}], temperature=0.3)
        assert llm.seen["temperature"] == 0.3

    def test_none_falls_back_to_the_model_default(self, llm):
        llm.complete(messages=[{"role": "user", "content": "hi"}])
        assert llm.seen["temperature"] == llm.config.default_temperature

    def test_gemini_style_models_stay_pinned_to_their_default(self, llm, monkeypatch):
        # requires_thought_signatures locks temperature for reasoning stability,
        # and that override must still win over an explicit 0.0.
        monkeypatch.setattr(llm.config, "requires_thought_signatures", True)
        llm.complete(messages=[{"role": "user", "content": "hi"}], temperature=0.0)
        assert llm.seen["temperature"] == llm.config.default_temperature


class TestUnsupportedTemperature:
    """Models whose endpoint rejects `temperature` must be sent none at all.

    GPT-6 Astra 400s on every value but 1.0, and callers do not cooperate:
    `main.py` passes `request.temperature or 0.7` on every chat request and
    skill extraction hardcodes 0.0. Pinning is not enough -- the parameter has
    to be absent from the payload.
    """

    def test_complete_omits_temperature(self, llm, monkeypatch):
        monkeypatch.setattr(llm.config, "supports_temperature", False)
        llm.complete(messages=[{"role": "user", "content": "hi"}], temperature=0.0)
        assert "temperature" not in llm.seen

    def test_complete_omits_an_explicit_nonzero_temperature(self, llm, monkeypatch):
        monkeypatch.setattr(llm.config, "supports_temperature", False)
        llm.complete(messages=[{"role": "user", "content": "hi"}], temperature=0.7)
        assert "temperature" not in llm.seen

    def test_stream_omits_temperature(self, llm, monkeypatch):
        # The UI chat loop streams, so this is the path the 400 would surface on.
        monkeypatch.setattr(llm.config, "supports_temperature", False)
        llm.stream(messages=[{"role": "user", "content": "hi"}], temperature=0.7)
        assert "temperature" not in llm.seen

    def test_supported_models_still_send_it(self, llm):
        llm.stream(messages=[{"role": "user", "content": "hi"}], temperature=0.7)
        assert llm.seen["temperature"] == 0.7

    def test_gpt_6_astra_sends_no_temperature(self, monkeypatch, tmp_path):
        """The registry entry, not just the flag, has to produce a clean payload."""
        key_file = tmp_path / "key"
        key_file.write_text("test-key")

        config = get_model_config("gpt-6-astra")
        monkeypatch.setattr(config, "api_key_file", str(key_file))
        instance = client_module.LLMClient(model_config=config)

        seen: dict = {}

        def _create(**kwargs):
            seen.update(kwargs)
            return _FakeCompletion()

        monkeypatch.setattr(instance._client.chat.completions, "create", _create)
        instance.complete(messages=[{"role": "user", "content": "hi"}], temperature=0.0)
        assert "temperature" not in seen
        assert seen["model"] == "azure/openai/gpt-6-astra"
