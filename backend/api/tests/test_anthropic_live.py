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

"""Live smoke test against api.anthropic.com.

Opt-in: needs a populated API_KEY_ANTHROPIC at the repo root and
AEVOM_LIVE_TESTS=1, so the default suite never spends tokens or needs network.
Run with ``AEVOM_LIVE_TESTS=1 pytest backend/api/tests/test_anthropic_live.py``.
"""

import os
from pathlib import Path

import pytest
from api.llm.anthropic_models import discover
from api.llm.client import create_client
from api.llm.config import key_file_present

KEY_FILE = str(Path(__file__).resolve().parents[3] / "API_KEY_ANTHROPIC")

pytestmark = pytest.mark.skipif(
    os.getenv("AEVOM_LIVE_TESTS") != "1" or not key_file_present(KEY_FILE),
    reason="live test: set AEVOM_LIVE_TESTS=1 and populate API_KEY_ANTHROPIC",
)

TOOL = {
    "type": "function",
    "function": {
        "name": "get_formula",
        "description": "Molecular formula of a named molecule.",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
}


@pytest.fixture(scope="module")
def smallest_model():
    configs = discover(KEY_FILE)
    assert configs, "the key listed no models"
    return configs[-1]


def test_tool_loop_round_trips(smallest_model):
    client = create_client(smallest_model)
    messages = [{"role": "user", "content": "Use the tool to get the formula of caffeine."}]

    first = client.complete(messages, tools=[TOOL], max_tokens=2048)
    assert first.finish_reason == "tool_calls"
    call = first.tool_calls[0]
    assert call["function"]["name"] == "get_formula"

    messages += [
        {
            "role": "assistant",
            "content": first.content or "",
            "tool_calls": first.tool_calls,
            "provider_specific_fields": first.provider_specific_fields,
        },
        {"role": "tool", "tool_call_id": call["id"], "content": "C8H10N4O2"},
    ]
    second = client.complete(messages, tools=[TOOL], max_tokens=2048)
    assert second.finish_reason == "stop"
    assert "C8H10N4O2" in (second.content or "")
