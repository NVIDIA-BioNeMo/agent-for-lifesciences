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

"""Tests for the OpenAI Responses API client.

The agent graph speaks Chat Completions messages; the Responses API wants input
items. A tool loop breaks here when a reasoning item is not replayed verbatim
and in order (the next request 400s), when a function_call loses its call_id,
or when temperature reaches a model that refuses it. The stream tests feed
plain-dict events, the same shape the SDK's events dump to.
"""

import json

import openai
import pytest
from api.graph.llm import _retryable_code
from api.llm.client import create_client
from api.llm.config import PROVIDER_OPENAI, ModelConfig
from api.llm.openai_responses_client import (
    OUTPUT_ITEMS_KEY,
    OpenAIResponsesClient,
    OpenAIResponsesStreamingResponse,
    to_responses_input,
    to_responses_tools,
)
from langchain_core.messages import AIMessageChunk


def gpt_config(**overrides) -> ModelConfig:
    fields = dict(
        id="openai/gpt-6.1-sol",
        name="gpt-6.1-sol",
        base_url="https://api.openai.com/v1",
        api_key_file="/nonexistent/openai_api_key",
        provider=PROVIDER_OPENAI,
        supports_reasoning=True,
        retains_reasoning=True,
        supports_temperature=False,
        max_tokens=32000,
        context_window=200000,
        template_name="base",
    )
    fields.update(overrides)
    return ModelConfig(**fields)


def tool_call(call_id: str, name: str, args: dict) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_molecule",
        "description": "Look up a molecule",
        "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    },
}

REASONING = {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA-opaque", "summary": []}
LAYOUT = [
    REASONING,
    {"type": "message", "id": "msg_1"},
    {"type": "function_call", "id": "fc_1", "call_id": "call_1"},
    {"type": "function_call", "id": "fc_2", "call_id": "call_2"},
]


class TestInputConversion:
    def test_system_and_developer_messages_become_instructions(self):
        instructions, items = to_responses_input(
            [
                {"role": "system", "content": "You are Aevom."},
                {"role": "user", "content": "hi"},
                {"role": "developer", "content": "Be brief."},
            ]
        )
        assert instructions == "You are Aevom.\n\nBe brief."
        assert items == [{"role": "user", "content": "hi"}]

    def test_state_envelope_moves_to_the_end_of_the_input(self):
        envelope = "[VIEWER STATE: Empty]\n[STORAGE STATE: sdf_1 (aspirin)]\n[SESSION LOG]\nT1.0 ..."
        instructions, items = to_responses_input(
            [
                {"role": "system", "content": "You are Aevom.\n\n" + envelope},
                {"role": "user", "content": "dock it"},
                {"role": "assistant", "content": "", "tool_calls": [tool_call("call_1", "x", {})]},
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ]
        )
        assert instructions == "You are Aevom."
        assert items[-1] == {"role": "developer", "content": envelope}
        assert [i.get("type", i.get("role")) for i in items[:-1]] == ["user", "function_call", "function_call_output"]

    def test_tool_loop_without_reasoning(self):
        _, items = to_responses_input(
            [
                {"role": "user", "content": "compare caffeine and aspirin"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        tool_call("call_1", "get_molecule", {"name": "caffeine"}),
                        tool_call("call_2", "get_molecule", {"name": "aspirin"}),
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "C8H10N4O2"},
                {"role": "tool", "tool_call_id": "call_2", "content": ""},
            ]
        )
        assert [i.get("type", "message") for i in items] == [
            "message",
            "function_call",
            "function_call",
            "function_call_output",
            "function_call_output",
        ]
        assert items[1] == {
            "type": "function_call",
            "call_id": "call_1",
            "name": "get_molecule",
            "arguments": '{"name": "caffeine"}',
        }
        assert items[3] == {"type": "function_call_output", "call_id": "call_1", "output": "C8H10N4O2"}
        assert items[4]["output"] == "(no output)"

    def test_assistant_text_without_layout_precedes_calls(self):
        _, items = to_responses_input(
            [{"role": "assistant", "content": "Looking.", "tool_calls": [tool_call("call_1", "x", {})]}]
        )
        assert [i["type"] for i in items] == ["message", "function_call"]
        assert items[0]["content"] == [{"type": "output_text", "text": "Looking."}]

    def test_empty_text_is_never_sent(self):
        _, items = to_responses_input(
            [
                {"role": "user", "content": "   "},
                {"role": "user", "content": [{"type": "text", "text": ""}]},
                {"role": "assistant", "content": ""},
            ]
        )
        assert items == []

    def test_images_become_input_image(self):
        _, items = to_responses_input(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                    ],
                }
            ]
        )
        assert items[0]["content"] == [
            {"type": "input_text", "text": "what is this"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]

    def test_dict_arguments_are_serialised(self):
        _, items = to_responses_input(
            [{"role": "assistant", "tool_calls": [{"id": "c", "function": {"name": "x", "arguments": {"a": 1}}}]}]
        )
        assert items[0]["arguments"] == '{"a": 1}'


class TestReasoningReplay:
    def message(self, content="Checking both.", layout=LAYOUT):
        return {
            "role": "assistant",
            "content": content,
            "tool_calls": [tool_call("call_1", "get_molecule", {"name": "caffeine"}), tool_call("call_2", "y", {})],
            "provider_specific_fields": {OUTPUT_ITEMS_KEY: layout},
        }

    def test_layout_restores_items_in_produced_order(self):
        _, items = to_responses_input([self.message()])
        assert [i["type"] for i in items] == ["reasoning", "message", "function_call", "function_call"]
        assert items[0] == REASONING
        assert items[1]["id"] == "msg_1"
        assert items[1]["content"] == [{"type": "output_text", "text": "Checking both."}]
        assert [(i["id"], i["call_id"]) for i in items[2:]] == [("fc_1", "call_1"), ("fc_2", "call_2")]

    def test_degraded_text_comes_from_the_message_not_the_layout(self):
        _, items = to_responses_input([self.message(content="[trimmed]")])
        assert items[1]["content"][0]["text"] == "[trimmed]"

    def test_message_marker_without_text_is_dropped(self):
        _, items = to_responses_input([self.message(content="")])
        assert [i["type"] for i in items] == ["reasoning", "function_call", "function_call"]

    def test_tool_calls_missing_from_layout_are_still_sent(self):
        _, items = to_responses_input([self.message(layout=LAYOUT[:3])])
        assert [i.get("call_id") for i in items if i["type"] == "function_call"] == ["call_1", "call_2"]
        assert "id" not in items[-1]

    def test_layout_survives_langchain_chunk_merging(self):
        psf = {OUTPUT_ITEMS_KEY: LAYOUT}
        merged = AIMessageChunk(content="a") + AIMessageChunk(
            content="b", additional_kwargs={"provider_specific_fields": psf}
        )
        assert merged.additional_kwargs["provider_specific_fields"] == psf


class TestToolConversion:
    def test_schema_is_flattened(self):
        tools = to_responses_tools([SCHEMA, {"type": "function", "function": {"name": "noop"}}])
        assert tools[0] == {
            "type": "function",
            "name": "get_molecule",
            "description": "Look up a molecule",
            "parameters": SCHEMA["function"]["parameters"],
        }
        assert tools[1]["parameters"] == {"type": "object", "properties": {}}
        assert tools[1]["description"] == ""

    def test_no_tools(self):
        assert to_responses_tools(None) == []


USAGE = {
    "input_tokens": 1200,
    "output_tokens": 80,
    "input_tokens_details": {"cached_tokens": 1024},
    "output_tokens_details": {"reasoning_tokens": 40},
}


def text_stream(*texts):
    yield {"type": "response.created", "response": {"id": "resp_1"}}
    yield {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "msg_1"}}
    for text in texts:
        yield {"type": "response.output_text.delta", "output_index": 0, "delta": text}
    yield {"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "msg_1"}}
    yield {"type": "response.completed", "response": {"usage": USAGE}}


def reasoning_tool_stream():
    """reasoning, message, function_call, function_call -- the order gpt-6.1-sol produced."""
    yield {"type": "response.created", "response": {"id": "resp_1"}}
    yield {"type": "response.output_item.added", "output_index": 0, "item": {"type": "reasoning", "id": "rs_1"}}
    yield {"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 0}
    yield {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "Plan: "}
    yield {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "look up."}
    yield {"type": "response.reasoning_summary_part.added", "output_index": 0, "summary_index": 1}
    yield {"type": "response.reasoning_summary_text.delta", "output_index": 0, "delta": "Then compare."}
    yield {
        "type": "response.output_item.done",
        "output_index": 0,
        "item": {"type": "reasoning", "id": "rs_1", "encrypted_content": "gAAAA-opaque", "summary": [], "status": None},
    }
    yield {"type": "response.output_item.added", "output_index": 1, "item": {"type": "message", "id": "msg_1"}}
    yield {"type": "response.output_text.delta", "output_index": 1, "delta": "Checking both."}
    yield {"type": "response.output_item.done", "output_index": 1, "item": {"type": "message", "id": "msg_1"}}
    for index, (call_id, name) in enumerate([("call_1", "caffeine"), ("call_2", "aspirin")], start=2):
        item = {"type": "function_call", "id": f"fc_{index - 1}", "call_id": call_id, "name": "get_molecule"}
        yield {"type": "response.output_item.added", "output_index": index, "item": {**item, "arguments": ""}}
        yield {"type": "response.function_call_arguments.delta", "output_index": index, "delta": '{"name": '}
        yield {"type": "response.function_call_arguments.delta", "output_index": index, "delta": f'"{name}"}}'}
        yield {
            "type": "response.output_item.done",
            "output_index": index,
            "item": {**item, "arguments": f'{{"name": "{name}"}}'},
        }
    yield {"type": "response.completed", "response": {"usage": USAGE}}


class TestStreaming:
    def test_text_stream(self):
        response = OpenAIResponsesStreamingResponse(text_stream("Hel", "lo"))
        chunks = list(response)
        assert "".join(c.content or "" for c in chunks) == "Hello"
        assert response.finish_reason == "stop"
        assert response.get_provider_specific_fields() is None
        assert response.get_accumulated_tool_calls() == []
        assert response.usage == (1200, 80)

    def test_reasoning_and_parallel_tool_calls(self):
        response = OpenAIResponsesStreamingResponse(reasoning_tool_stream())
        chunks = list(response)

        assert "".join(c.reasoning or "" for c in chunks) == "Plan: look up.\n\nThen compare."
        assert response.content == "Checking both."
        assert response.finish_reason == "tool_calls"
        calls = response.get_accumulated_tool_calls()
        assert [c["id"] for c in calls] == ["call_1", "call_2"]
        assert json.loads(calls[1]["function"]["arguments"]) == {"name": "aspirin"}

        layout = response.get_provider_specific_fields()[OUTPUT_ITEMS_KEY]
        assert layout == LAYOUT
        assert all("index" not in entry for entry in layout)

    def test_arguments_from_deltas_when_done_item_is_missing(self):
        events = [e for e in reasoning_tool_stream() if not (
            e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
        )]
        calls = OpenAIResponsesStreamingResponse(iter(events)).drain().get_accumulated_tool_calls()
        assert json.loads(calls[0]["function"]["arguments"]) == {"name": "caffeine"}

    def test_stream_output_converts_back_to_a_valid_request(self):
        response = OpenAIResponsesStreamingResponse(reasoning_tool_stream()).drain()
        _, items = to_responses_input(
            [
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.get_accumulated_tool_calls(),
                    "provider_specific_fields": response.get_provider_specific_fields(),
                }
            ]
        )
        assert [i["type"] for i in items] == ["reasoning", "message", "function_call", "function_call"]
        assert items[0]["encrypted_content"] == "gAAAA-opaque"
        assert items[3]["arguments"] == '{"name": "aspirin"}'

    @pytest.mark.parametrize("reason, finish", [("max_output_tokens", "length"), ("content_filter", "content_filter")])
    def test_incomplete_maps_finish_reason(self, reason, finish):
        def events():
            yield {"type": "response.output_text.delta", "delta": "partial"}
            yield {"type": "response.incomplete", "response": {"incomplete_details": {"reason": reason}, "usage": {}}}

        assert OpenAIResponsesStreamingResponse(events()).drain().finish_reason == finish

    @pytest.mark.parametrize(
        "event, retry_code",
        [
            ({"type": "error", "code": "rate_limit_exceeded", "message": "slow down"}, 429),
            ({"type": "error", "code": "server_error", "message": "oops"}, 500),
            ({"type": "response.failed", "response": {"error": {"code": "server_is_overloaded"}}}, 503),
            ({"type": "error", "code": "invalid_prompt", "message": "no"}, None),
        ],
    )
    def test_in_stream_errors_are_classified_for_retry(self, event, retry_code):
        def events():
            yield {"type": "response.created"}
            yield event

        with pytest.raises(openai.APIError) as exc_info:
            list(OpenAIResponsesStreamingResponse(events()))
        assert _retryable_code(exc_info.value) == retry_code


class FakeResponses:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.events)


class FakeSDK:
    def __init__(self, events):
        self.responses = FakeResponses(events)


class TestClient:
    def test_request_shape(self):
        client = OpenAIResponsesClient(gpt_config(), sdk_client=FakeSDK([]))
        request = client.build_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}], tools=[SCHEMA]
        )
        assert "temperature" not in request
        assert request["store"] is False
        assert request["instructions"] == "sys"
        assert request["tool_choice"] == "auto"
        assert request["tools"][0]["name"] == "get_molecule"
        assert request["reasoning"] == {"summary": "auto"}
        assert request["max_output_tokens"] == 32000
        assert len(request["prompt_cache_key"]) == 32

    def test_effort_and_max_tokens_override(self):
        request = OpenAIResponsesClient(gpt_config(effort="high"), sdk_client=FakeSDK([])).build_request(
            [{"role": "user", "content": "hi"}], max_tokens=4000
        )
        assert request["reasoning"] == {"summary": "auto", "effort": "high"}
        assert request["max_output_tokens"] == 4000
        assert "tools" not in request and "instructions" not in request

    def test_cache_key_is_stable_for_the_same_prefix(self):
        client = OpenAIResponsesClient(gpt_config(), sdk_client=FakeSDK([]))
        first = client.build_request([{"role": "system", "content": "sys"}, {"role": "user", "content": "a"}], [SCHEMA])
        second = client.build_request([{"role": "system", "content": "sys"}, {"role": "user", "content": "b"}], [SCHEMA])
        other = client.build_request([{"role": "system", "content": "other"}], [SCHEMA])
        assert first["prompt_cache_key"] == second["prompt_cache_key"] != other["prompt_cache_key"]

    def test_cache_key_ignores_session_state_appended_to_the_prompt(self):
        client = OpenAIResponsesClient(gpt_config(), sdk_client=FakeSDK([]))
        prompt = "# Aevom System Prompt\n" + "x" * 20000

        def key(state):
            return client.build_request([{"role": "system", "content": prompt + state}], [SCHEMA])["prompt_cache_key"]

        assert key("[STORAGE STATE: Empty]") == key("[STORAGE STATE: sdf_1]\n[SESSION LOG]\nT1.0 ...")

    def test_complete_drains_a_stream(self):
        sdk = FakeSDK(list(reasoning_tool_stream()))
        result = OpenAIResponsesClient(gpt_config(), sdk_client=sdk).complete(
            [{"role": "user", "content": "hi"}], tools=[SCHEMA], temperature=0.3
        )
        assert sdk.responses.calls[0]["stream"] is True
        assert "temperature" not in sdk.responses.calls[0]
        assert result.content == "Checking both."
        assert result.reasoning.startswith("Plan: look up.")
        assert len(result.tool_calls) == 2
        assert result.finish_reason == "tool_calls"
        assert result.prompt_tokens == 1200 and result.completion_tokens == 80
        assert OUTPUT_ITEMS_KEY in result.provider_specific_fields

    def test_stream_returns_chunks(self):
        response = OpenAIResponsesClient(gpt_config(), sdk_client=FakeSDK(list(text_stream("ok")))).stream(
            [{"role": "user", "content": "hi"}]
        )
        assert "".join(c.content or "" for c in response) == "ok"


def test_openai_config_gets_responses_client(tmp_path):
    key = tmp_path / "key"
    key.write_text("test-key-not-real")
    assert isinstance(create_client(gpt_config(api_key_file=str(key))), OpenAIResponsesClient)
