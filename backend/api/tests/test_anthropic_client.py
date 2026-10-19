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

"""Tests for the Anthropic Messages API client.

The agent graph speaks OpenAI-format messages; Claude needs its own. The
conversions here are where a tool loop breaks: a tool_result in the wrong
place, a parallel batch split across turns, or a thinking block that does not
come back verbatim each fail the next request with a 400. The stream tests feed
plain-dict events, the same shape the SDK's events dump to.
"""

import dataclasses
import json

import anthropic
import httpx
import pytest
from api.graph.llm import _langchain_to_openai_messages, _retryable_code
from api.llm.anthropic_client import (
    CONTENT_BLOCKS_KEY,
    AnthropicLLMClient,
    AnthropicStreamingResponse,
    to_anthropic_messages,
    to_anthropic_tools,
)
from api.llm.client import LLMClient, create_client
from api.llm.config import PROVIDER_ANTHROPIC, ModelConfig, get_model_config
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage


def claude_config(**overrides) -> ModelConfig:
    fields = dict(
        id="anthropic/claude-sonnet-5-5",
        name="claude-sonnet-5-5",
        base_url="https://api.anthropic.com",
        api_key_file="/nonexistent/anthropic_api_key",
        provider=PROVIDER_ANTHROPIC,
        supports_reasoning=True,
        retains_reasoning=True,
        supports_temperature=False,
        max_tokens=32000,
        context_window=200000,
        template_name="base",
        thinking_mode="adaptive",
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


class TestMessageConversion:
    def test_system_messages_are_hoisted_and_joined(self):
        system, messages = to_anthropic_messages(
            [
                {"role": "system", "content": "You are Aevom."},
                {"role": "user", "content": "hi"},
                {"role": "system", "content": "Be brief."},
            ]
        )
        assert system == "You are Aevom.\n\nBe brief."
        assert messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]

    def test_parallel_tool_results_share_one_user_turn_before_any_text(self):
        _, messages = to_anthropic_messages(
            [
                {"role": "user", "content": "compare caffeine and aspirin"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        tool_call("toolu_1", "get_molecule", {"name": "caffeine"}),
                        tool_call("toolu_2", "get_molecule", {"name": "aspirin"}),
                    ],
                },
                {"role": "tool", "tool_call_id": "toolu_1", "content": "C8H10N4O2"},
                {"role": "tool", "tool_call_id": "toolu_2", "content": "C9H8O4"},
                {"role": "user", "content": "now summarise"},
            ]
        )
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert [b["type"] for b in messages[1]["content"]] == ["tool_use", "tool_use"]
        assert messages[1]["content"][0]["input"] == {"name": "caffeine"}
        results = messages[2]["content"]
        assert [b["type"] for b in results] == ["tool_result", "tool_result", "text"]
        assert [b.get("tool_use_id") for b in results[:2]] == ["toolu_1", "toolu_2"]

    def test_tool_error_is_flagged_and_empty_output_is_filled(self):
        _, messages = to_anthropic_messages(
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": [tool_call("toolu_1", "x", {})]},
                {"role": "tool", "tool_call_id": "toolu_1", "content": "", "is_error": True},
            ]
        )
        block = messages[-1]["content"][0]
        assert block["is_error"] is True
        assert block["content"] == "(no output)"

    def test_empty_text_is_never_sent(self):
        _, messages = to_anthropic_messages(
            [{"role": "user", "content": "   "}, {"role": "user", "content": [{"type": "text", "text": ""}]}]
        )
        assert messages == []

    def test_data_url_image_becomes_base64_block(self):
        _, messages = to_anthropic_messages(
            [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]
        )
        assert messages[0]["content"][0] == {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"},
        }

    def test_malformed_tool_arguments_do_not_raise(self):
        _, messages = to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "t", "function": {"name": "x", "arguments": "{not json"}}],
                }
            ]
        )
        assert messages[0]["content"][0]["input"] == {"raw": "{not json"}


class TestThinkingRoundTrip:
    LAYOUT = [
        {"type": "thinking", "thinking": "Need both molecules.", "signature": "sig-abc"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "text"},
        {"type": "tool_use", "id": "toolu_1"},
    ]

    def test_layout_restores_blocks_in_produced_order(self):
        _, messages = to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "Looking it up.",
                    "tool_calls": [tool_call("toolu_1", "get_molecule", {"name": "caffeine"})],
                    "provider_specific_fields": {CONTENT_BLOCKS_KEY: self.LAYOUT},
                }
            ]
        )
        blocks = messages[0]["content"]
        assert blocks[0] == {"type": "thinking", "thinking": "Need both molecules.", "signature": "sig-abc"}
        assert blocks[1] == {"type": "redacted_thinking", "data": "opaque"}
        assert blocks[2] == {"type": "text", "text": "Looking it up."}
        assert blocks[3]["type"] == "tool_use" and blocks[3]["id"] == "toolu_1"

    def test_tool_calls_missing_from_layout_are_still_sent(self):
        _, messages = to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [tool_call("toolu_1", "a", {}), tool_call("toolu_2", "b", {})],
                    "provider_specific_fields": {CONTENT_BLOCKS_KEY: self.LAYOUT},
                }
            ]
        )
        assert [b.get("id") for b in messages[0]["content"] if b["type"] == "tool_use"] == ["toolu_1", "toolu_2"]

    def test_layout_survives_langchain_chunk_merging(self):
        """Streamed chunks are summed by LangChain; the layout must come out intact."""
        psf = {CONTENT_BLOCKS_KEY: self.LAYOUT}
        merged = AIMessageChunk(content="a") + AIMessageChunk(
            content="b", additional_kwargs={"provider_specific_fields": psf}
        )
        assert merged.additional_kwargs["provider_specific_fields"] == psf


class TestToolConversion:
    def test_schema_is_translated_and_last_tool_is_cached(self):
        second = {"type": "function", "function": {"name": "noop"}}
        tools = to_anthropic_tools([SCHEMA, second])
        assert tools[0] == {
            "name": "get_molecule",
            "description": "Look up a molecule",
            "input_schema": SCHEMA["function"]["parameters"],
        }
        assert tools[1]["input_schema"] == {"type": "object", "properties": {}}
        assert tools[1]["cache_control"] == {"type": "ephemeral"}
        assert "cache_control" not in tools[0]

    def test_no_tools(self):
        assert to_anthropic_tools(None) == []


def text_stream(*texts, stop="end_turn", usage=None):
    yield {"type": "message_start", "message": {"usage": usage or {"input_tokens": 10}}}
    yield {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
    for text in texts:
        yield {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}
    yield {"type": "content_block_stop", "index": 0}
    yield {"type": "message_delta", "delta": {"stop_reason": stop}, "usage": {"output_tokens": 5}}
    yield {"type": "message_stop"}


def thinking_tool_stream():
    yield {"type": "message_start", "message": {"usage": {"input_tokens": 100}}}
    yield {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}
    yield {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Plan: "}}
    yield {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "look up."}}
    yield {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}}
    yield {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}
    yield {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Checking."}}
    for index, (call_id, name) in enumerate([("toolu_1", "caffeine"), ("toolu_2", "aspirin")], start=2):
        yield {
            "type": "content_block_start",
            "index": index,
            "content_block": {"type": "tool_use", "id": call_id, "name": "get_molecule", "input": {}},
        }
        yield {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": '{"name": '},
        }
        yield {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": f'"{name}"}}'},
        }
    yield {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 42}}
    yield {"type": "message_stop"}


class TestStreaming:
    def test_text_stream(self):
        response = AnthropicStreamingResponse(text_stream("Hel", "lo"))
        chunks = list(response)
        assert "".join(c.content or "" for c in chunks) == "Hello"
        assert response.finish_reason == "stop"
        assert response.get_provider_specific_fields() is None
        assert response.get_accumulated_tool_calls() == []

    def test_usage_counts_cached_input(self):
        usage = {"input_tokens": 10, "cache_read_input_tokens": 1000, "cache_creation_input_tokens": 50}
        response = AnthropicStreamingResponse(text_stream("x", usage=usage)).drain()
        assert response.usage == (1060, 5)

    def test_thinking_and_parallel_tool_calls(self):
        response = AnthropicStreamingResponse(thinking_tool_stream())
        chunks = list(response)

        assert "".join(c.reasoning or "" for c in chunks) == "Plan: look up."
        assert response.content == "Checking."
        assert response.finish_reason == "tool_calls"
        calls = response.get_accumulated_tool_calls()
        assert [c["id"] for c in calls] == ["toolu_1", "toolu_2"]
        assert json.loads(calls[1]["function"]["arguments"]) == {"name": "aspirin"}

        layout = response.get_provider_specific_fields()[CONTENT_BLOCKS_KEY]
        assert layout == [
            {"type": "thinking", "thinking": "Plan: look up.", "signature": "sig"},
            {"type": "text"},
            {"type": "tool_use", "id": "toolu_1"},
            {"type": "tool_use", "id": "toolu_2"},
        ]
        assert all("index" not in entry for entry in layout)

    def test_stream_output_converts_back_to_a_valid_request(self):
        response = AnthropicStreamingResponse(thinking_tool_stream()).drain()
        _, messages = to_anthropic_messages(
            [
                {
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": response.get_accumulated_tool_calls(),
                    "provider_specific_fields": response.get_provider_specific_fields(),
                }
            ]
        )
        blocks = messages[0]["content"]
        assert [b["type"] for b in blocks] == ["thinking", "text", "tool_use", "tool_use"]
        assert blocks[0]["signature"] == "sig"
        assert blocks[3]["input"] == {"name": "aspirin"}

    def test_in_stream_error_is_raised_and_retryable(self):
        def events():
            yield {"type": "message_start", "message": {"usage": {}}}
            yield {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}

        response = AnthropicStreamingResponse(events())
        with pytest.raises(anthropic.APIError) as exc_info:
            list(response)
        assert _retryable_code(exc_info.value) == 529


class FakeMessages:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return iter(self.events)


class FakeSDK:
    def __init__(self, events):
        self.messages = FakeMessages(events)


class TestClient:
    def test_request_never_sends_temperature_and_uses_auto_tool_choice(self):
        client = AnthropicLLMClient(claude_config(), sdk_client=FakeSDK([]))
        request = client.build_request(
            [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}], tools=[SCHEMA]
        )
        assert "temperature" not in request
        assert request["system"] == "sys"
        assert request["tool_choice"] == {"type": "auto"}
        assert request["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert request["max_tokens"] == 32000
        assert "output_config" not in request

    def test_budgeted_thinking_and_effort(self):
        config = claude_config(thinking_mode="enabled", thinking_budget_tokens=8192, effort="high")
        request = AnthropicLLMClient(config, sdk_client=FakeSDK([])).build_request(
            [{"role": "user", "content": "hi"}], max_tokens=4000
        )
        assert request["thinking"] == {"type": "enabled", "budget_tokens": 3999}
        assert request["output_config"] == {"effort": "high"}

    def test_thinking_budget_below_minimum_is_dropped(self):
        config = claude_config(thinking_mode="enabled")
        request = AnthropicLLMClient(config, sdk_client=FakeSDK([])).build_request(
            [{"role": "user", "content": "hi"}], max_tokens=1000
        )
        assert "thinking" not in request

    def test_no_thinking_without_support(self):
        request = AnthropicLLMClient(claude_config(thinking_mode=None), sdk_client=FakeSDK([])).build_request(
            [{"role": "user", "content": "hi"}]
        )
        assert "thinking" not in request

    def test_complete_drains_a_stream(self):
        sdk = FakeSDK(list(thinking_tool_stream()))
        result = AnthropicLLMClient(claude_config(), sdk_client=sdk).complete(
            [{"role": "user", "content": "hi"}], tools=[SCHEMA], temperature=0.3
        )
        assert sdk.messages.calls[0]["stream"] is True
        assert "temperature" not in sdk.messages.calls[0]
        assert result.content == "Checking."
        assert result.reasoning == "Plan: look up."
        assert len(result.tool_calls) == 2
        assert result.finish_reason == "tool_calls"
        assert result.prompt_tokens == 100 and result.completion_tokens == 42
        assert CONTENT_BLOCKS_KEY in result.provider_specific_fields


class TestFactory:
    def test_anthropic_config_gets_anthropic_client(self, tmp_path):
        key = tmp_path / "key"
        key.write_text("test-key-not-real")
        client = create_client(claude_config(api_key_file=str(key)))
        assert isinstance(client, AnthropicLLMClient)

    def test_static_config_gets_openai_client(self, tmp_path):
        key = tmp_path / "key"
        key.write_text("test-key-not-real")
        config = dataclasses.replace(get_model_config("nemotron-3-ultra"), api_key_file=str(key))
        assert isinstance(create_client(config), LLMClient)


def status_error(code: int, error_type: str) -> anthropic.APIStatusError:
    response = httpx.Response(code, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    return anthropic.APIStatusError("boom", response=response, body={"error": {"type": error_type}})


class TestRetryClassification:
    @pytest.mark.parametrize("code", [429, 500, 529])
    def test_transient_statuses_retry(self, code):
        assert _retryable_code(status_error(code, "overloaded_error")) == code

    def test_in_stream_error_with_200_status_is_classified_by_type(self):
        assert _retryable_code(status_error(200, "rate_limit_error")) == 429

    @pytest.mark.parametrize("code,error_type", [(400, "invalid_request_error"), (401, "authentication_error")])
    def test_client_errors_are_terminal(self, code, error_type):
        assert _retryable_code(status_error(code, error_type)) is None


class TestGraphMessageConversion:
    MESSAGES = [
        HumanMessage(content="go"),
        AIMessage(content="", tool_calls=[{"id": "toolu_1", "name": "x", "args": {}}]),
        ToolMessage(content="Error: boom", tool_call_id="toolu_1", status="error"),
    ]

    def test_failed_tool_is_flagged_for_claude(self):
        converted = _langchain_to_openai_messages(self.MESSAGES, claude_config())
        assert converted[-1]["is_error"] is True

    def test_other_providers_are_not_sent_the_flag(self):
        converted = _langchain_to_openai_messages(self.MESSAGES, get_model_config("nemotron-3-ultra"))
        assert "is_error" not in converted[-1]
