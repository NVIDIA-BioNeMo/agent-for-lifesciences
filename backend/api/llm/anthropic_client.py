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

"""LLM client for Claude models over the native Anthropic Messages API.

Presents the same surface as ``LLMClient`` -- ``config``,
``render_system_prompt``, ``complete`` and ``stream`` -- so ``ChatAevomLLM``
and the agent graph drive either one without knowing which. Callers speak
OpenAI-format messages and tool schemas; this module translates both ways.

Thinking blocks are the one thing with no OpenAI-format equivalent. They carry
an opaque ``signature`` and must go back to the API unmodified on the assistant
turn of a tool loop, so they ride along in ``provider_specific_fields`` (which
``ChatAevomLLM`` already round-trips for Gemini's thought signatures) under
``anthropic_content_blocks``.
"""

import json
import logging
from typing import Any, Iterator, Optional

import anthropic
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from api.context import check_context_budget, record_budget_on_span, record_estimate_error
from api.telemetry import get_current_chat_span, get_current_iteration_span, get_tracer

from .client import (
    LLM_CONNECT_TIMEOUT_S,
    LLM_READ_TIMEOUT_S,
    CompletionResponse,
    StreamChunk,
    _get_api_key,
    render_system_prompt,
)
from .config import ModelConfig

logger = logging.getLogger(__name__)

# Key in provider_specific_fields holding the assistant turn's block layout.
CONTENT_BLOCKS_KEY = "anthropic_content_blocks"

_THINKING_TYPES = ("thinking", "redacted_thinking")

_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "max_tokens": "length",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


# ---------------------------------------------------------------------------
# OpenAI format -> Anthropic format
# ---------------------------------------------------------------------------


def _text_of(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)


def _image_block(url: str) -> dict[str, Any]:
    if url.startswith("data:"):
        header, _, data = url.partition(",")
        media_type = header[len("data:"):].split(";", 1)[0] or "image/png"
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type, "data": data},
        }
    return {"type": "image", "source": {"type": "url", "url": url}}


def _user_blocks(content: Any) -> list[dict[str, Any]]:
    """User content as Anthropic blocks. Empty text blocks are rejected by the API."""
    if isinstance(content, list):
        blocks = []
        for part in content:
            if not isinstance(part, dict):
                text = str(part)
                if text:
                    blocks.append({"type": "text", "text": text})
            elif part.get("type") == "text" and part.get("text"):
                blocks.append({"type": "text", "text": part["text"]})
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    blocks.append(_image_block(url))
        return blocks
    text = _text_of(content)
    return [{"type": "text", "text": text}] if text.strip() else []


def _tool_use_block(tool_call: dict[str, Any]) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    arguments = function.get("arguments")
    if isinstance(arguments, dict):
        tool_input = arguments
    else:
        try:
            tool_input = json.loads(arguments) if arguments else {}
        except (json.JSONDecodeError, TypeError):
            tool_input = {"raw": arguments}
        if not isinstance(tool_input, dict):
            tool_input = {"value": tool_input}
    return {
        "type": "tool_use",
        "id": tool_call["id"],
        "name": function.get("name", ""),
        "input": tool_input,
    }


def _assistant_blocks(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild an assistant turn in the order the model produced it.

    The stored layout supplies the thinking blocks verbatim and the position of
    the text and tool_use blocks; their contents come from the message itself,
    because the context assembler may have degraded the text since.
    """
    text = _text_of(message.get("content"))
    tool_uses = [_tool_use_block(tc) for tc in message.get("tool_calls") or []]
    remaining = {block["id"]: block for block in tool_uses}

    psf = message.get("provider_specific_fields") or {}
    layout = psf.get(CONTENT_BLOCKS_KEY) if isinstance(psf, dict) else None

    blocks: list[dict[str, Any]] = []
    text_placed = False
    for entry in layout or []:
        kind = entry.get("type")
        if kind in _THINKING_TYPES:
            blocks.append({k: v for k, v in entry.items()})
        elif kind == "text" and not text_placed:
            text_placed = True
            if text.strip():
                blocks.append({"type": "text", "text": text})
        elif kind == "tool_use" and entry.get("id") in remaining:
            blocks.append(remaining.pop(entry["id"]))

    if not text_placed and text.strip():
        # Text goes after any thinking but before the tool calls.
        insert_at = next(
            (i for i, b in enumerate(blocks) if b["type"] == "tool_use"), len(blocks)
        )
        blocks.insert(insert_at, {"type": "text", "text": text})
    blocks.extend(b for b in tool_uses if b["id"] in remaining)
    return blocks


def _tool_result_block(message: dict[str, Any]) -> dict[str, Any]:
    block: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": message["tool_call_id"],
        "content": _text_of(message.get("content")) or "(no output)",
    }
    if message.get("is_error"):
        block["is_error"] = True
    return block


def to_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI-format chat messages to ``(system, messages)``.

    - System messages are hoisted and joined; Anthropic takes one system field.
    - Each ``role: tool`` message becomes a ``tool_result`` block in a user
      turn. Consecutive turns of the same role are merged, so the results of
      one parallel batch share a single user message, and they come before any
      text that follows them.
    """
    system_parts: list[str] = []
    converted: list[dict[str, Any]] = []

    def append(role: str, blocks: list[dict[str, Any]]) -> None:
        if not blocks:
            return
        if converted and converted[-1]["role"] == role:
            converted[-1]["content"].extend(blocks)
        else:
            converted.append({"role": role, "content": list(blocks)})

    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            text = _text_of(message.get("content"))
            if text.strip():
                system_parts.append(text)
        elif role == "assistant":
            append("assistant", _assistant_blocks(message))
        elif role == "tool":
            append("user", [_tool_result_block(message)])
        else:
            append("user", _user_blocks(message.get("content")))

    return "\n\n".join(system_parts), converted


def to_anthropic_tools(tools: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """OpenAI function schemas to Anthropic tool definitions.

    The last definition carries a cache breakpoint: the tool list is large and
    identical on every iteration of a turn, and it is the first thing in the
    prompt, so caching it pays off from the second call on.
    """
    converted = []
    for tool in tools or []:
        function = tool.get("function", tool)
        schema = function.get("parameters") or {"type": "object", "properties": {}}
        converted.append(
            {
                "name": function["name"],
                "description": function.get("description") or "",
                "input_schema": schema,
            }
        )
    if converted:
        converted[-1]["cache_control"] = {"type": "ephemeral"}
    return converted


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj)


class AnthropicStreamingResponse:
    """Iterate a Messages API event stream as ``StreamChunk`` objects.

    Mirrors ``StreamingResponse``: tool calls are accumulated for
    ``get_accumulated_tool_calls()`` and the span is closed when the stream
    ends or fails.
    """

    def __init__(self, events: Any, span: Any = None, estimated_prompt_tokens: int = 0):
        self._events = iter(events)
        self._span = span
        self._estimated_prompt_tokens = estimated_prompt_tokens
        self._blocks: dict[int, dict[str, Any]] = {}
        self._tool_json: dict[int, list[str]] = {}
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._finish_reason: Optional[str] = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._thinking_tokens = 0
        self._cache_read_tokens = 0
        self._cache_write_tokens = 0
        self._event_count = 0

    def __iter__(self) -> Iterator[StreamChunk]:
        return self

    def __next__(self) -> StreamChunk:
        while True:
            try:
                raw = next(self._events)
            except StopIteration:
                self._end_span()
                raise
            except Exception as e:
                self._fail_span(e)
                raise
            self._event_count += 1
            chunk = self._handle(_as_dict(raw))
            if chunk is not None:
                return chunk

    def _handle(self, event: dict[str, Any]) -> Optional[StreamChunk]:
        kind = event.get("type")

        if kind == "message_start":
            usage = (event.get("message") or {}).get("usage") or {}
            self._cache_read_tokens = usage.get("cache_read_input_tokens") or 0
            self._cache_write_tokens = usage.get("cache_creation_input_tokens") or 0
            self._input_tokens = (
                (usage.get("input_tokens") or 0) + self._cache_read_tokens + self._cache_write_tokens
            )
            return StreamChunk()

        if kind == "content_block_start":
            block = dict(event.get("content_block") or {})
            index = event.get("index", len(self._blocks))
            self._blocks[index] = block
            if block.get("type") == "tool_use":
                self._tool_json[index] = []
                block.setdefault("input", {})
            return None

        if kind == "content_block_delta":
            index = event.get("index", 0)
            delta = event.get("delta") or {}
            block = self._blocks.setdefault(index, {"type": "text", "text": ""})
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = delta.get("text", "")
                block["text"] = block.get("text", "") + text
                self._content.append(text)
                return StreamChunk(content=text) if text else None
            if delta_type == "thinking_delta":
                thinking = delta.get("thinking", "")
                block["thinking"] = block.get("thinking", "") + thinking
                self._reasoning.append(thinking)
                return StreamChunk(reasoning=thinking) if thinking else None
            if delta_type == "signature_delta":
                block["signature"] = block.get("signature", "") + delta.get("signature", "")
                return None
            if delta_type == "input_json_delta":
                self._tool_json.setdefault(index, []).append(delta.get("partial_json", ""))
                return None
            return None

        if kind == "message_delta":
            delta = event.get("delta") or {}
            usage = event.get("usage") or {}
            if usage.get("output_tokens") is not None:
                self._output_tokens = usage["output_tokens"]
            details = usage.get("output_tokens_details") or {}
            if details.get("thinking_tokens") is not None:
                self._thinking_tokens = details["thinking_tokens"]
            stop_reason = delta.get("stop_reason")
            if stop_reason:
                self._finish_reason = _FINISH_REASONS.get(stop_reason, stop_reason)
                return StreamChunk(finish_reason=self._finish_reason)
            return None

        if kind == "error":
            error = event.get("error") or {}
            raise anthropic.APIError(
                str(error.get("message") or error), request=None, body=event
            )

        # message_stop, content_block_stop, ping
        return None

    # -- accumulated results ------------------------------------------------

    def _ordered_blocks(self) -> list[dict[str, Any]]:
        return [self._blocks[i] for i in sorted(self._blocks)]

    def get_accumulated_tool_calls(self) -> list[dict[str, Any]]:
        calls = []
        for index in sorted(self._blocks):
            block = self._blocks[index]
            if block.get("type") != "tool_use":
                continue
            arguments = "".join(self._tool_json.get(index, [])) or json.dumps(block.get("input") or {})
            calls.append(
                {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {"name": block.get("name", ""), "arguments": arguments},
                }
            )
        return calls

    def get_thought_signatures(self) -> list[str]:
        return []

    def get_provider_specific_fields(self) -> Optional[dict[str, Any]]:
        """The block layout, when the turn had thinking worth sending back.

        Text and tool_use entries are position markers only; their contents
        live on the message. No entry may carry an ``index`` key: LangChain
        merges list items that share one when it concatenates chunks.
        """
        blocks = self._ordered_blocks()
        if not any(b.get("type") in _THINKING_TYPES for b in blocks):
            return None
        layout = []
        for block in blocks:
            kind = block.get("type")
            if kind == "thinking":
                layout.append(
                    {"type": "thinking", "thinking": block.get("thinking", ""), "signature": block.get("signature", "")}
                )
            elif kind == "redacted_thinking":
                layout.append({"type": "redacted_thinking", "data": block.get("data", "")})
            elif kind == "text":
                layout.append({"type": "text"})
            elif kind == "tool_use":
                layout.append({"type": "tool_use", "id": block.get("id", "")})
        return {CONTENT_BLOCKS_KEY: layout}

    @property
    def content(self) -> str:
        return "".join(self._content)

    @property
    def reasoning(self) -> str:
        return "".join(self._reasoning)

    @property
    def finish_reason(self) -> Optional[str]:
        return self._finish_reason

    @property
    def usage(self) -> tuple[int, int]:
        return self._input_tokens, self._output_tokens

    def drain(self) -> "AnthropicStreamingResponse":
        for _ in self:
            pass
        return self

    # -- telemetry ----------------------------------------------------------

    def _end_span(self) -> None:
        if not self._span:
            return
        span = self._span
        self._span = None

        span.set_attribute("llm.raw.content", self.content)
        tool_calls = self.get_accumulated_tool_calls()
        if tool_calls:
            span.set_attribute(
                "llm.raw.tool_calls",
                json.dumps([{"id": tc["id"], "function": tc["function"]} for tc in tool_calls]),
            )
            span.set_attribute("llm.tool_calls_count", len(tool_calls))
            span.set_attribute("llm.tool_calls", ",".join(tc["function"]["name"] for tc in tool_calls))
        if self._reasoning:
            span.set_attribute("llm.raw.reasoning", self.reasoning)
            span.set_attribute("llm.has_reasoning", True)
        if self._finish_reason:
            span.set_attribute("llm.finish_reason", self._finish_reason)

        span.set_attribute("llm.usage.input_tokens", self._input_tokens)
        span.set_attribute("llm.usage.output_tokens", self._output_tokens)
        span.set_attribute("llm.usage.thinking_tokens", self._thinking_tokens)
        span.set_attribute("llm.usage.cache_read_tokens", self._cache_read_tokens)
        span.set_attribute("llm.usage.cache_write_tokens", self._cache_write_tokens)
        record_estimate_error(span, self._estimated_prompt_tokens, self._input_tokens)

        from api.telemetry import get_token_accumulator, get_token_counter

        counter = get_token_counter()
        if counter and self._input_tokens:
            counter.add(self._input_tokens, {"token_type": "input"})
        if counter and self._output_tokens:
            counter.add(self._output_tokens, {"token_type": "output"})
        acc = get_token_accumulator()
        if acc:
            acc.add(self._input_tokens, self._output_tokens)

        if self._event_count == 0:
            message = "provider returned an empty stream (no events)"
            span.set_attribute("llm.error", message)
            span.set_status(Status(StatusCode.ERROR, message))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()

    def _fail_span(self, error: Exception) -> None:
        if self._span:
            self._span.set_attribute("llm.error", str(error))
            self._span.set_status(Status(StatusCode.ERROR, str(error)))
            self._span.end()
            self._span = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AnthropicLLMClient:
    """``LLMClient`` counterpart for ``provider="anthropic"`` models."""

    def __init__(self, model_config: ModelConfig, sdk_client: Any = None):
        self.config = model_config
        # The SDK's own Timeout type: it rejects an httpx object from a
        # different HTTP stack than the one it was built on.
        self._client = sdk_client or anthropic.Anthropic(
            api_key=_get_api_key(model_config.api_key_file),
            timeout=anthropic.Timeout(
                connect=LLM_CONNECT_TIMEOUT_S,
                read=LLM_READ_TIMEOUT_S,
                write=LLM_READ_TIMEOUT_S,
                pool=LLM_READ_TIMEOUT_S,
            ),
        )
        logger.info(f"AnthropicLLMClient initialized with model: {model_config.name}")

    def render_system_prompt(
        self,
        base_prompt: str,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        return render_system_prompt(self.config, base_prompt, tools)

    def build_request(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        max_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        """Keyword arguments for ``messages.create``.

        No ``temperature``: thinking requires the default, and newer models
        reject the parameter. ``tool_choice`` stays ``auto`` -- ``any`` and
        ``tool`` return a 400 on the 5.5 models.
        """
        system, converted = to_anthropic_messages(messages)
        max_out = max_tokens or self.config.max_tokens
        request: dict[str, Any] = {
            "model": self.config.name,
            "max_tokens": max_out,
            "messages": converted,
        }
        if system:
            request["system"] = system
        anthropic_tools = to_anthropic_tools(tools)
        if anthropic_tools:
            request["tools"] = anthropic_tools
            request["tool_choice"] = {"type": "auto"}

        if self.config.thinking_mode == "adaptive":
            request["thinking"] = {"type": "adaptive", "display": "summarized"}
        elif self.config.thinking_mode == "enabled":
            budget = min(self.config.thinking_budget_tokens, max_out - 1)
            if budget >= 1024:
                request["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if self.config.effort:
            request["output_config"] = {"effort": self.config.effort}
        return request

    def _open(
        self,
        span_name: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        max_tokens: Optional[int],
        parent_span: Optional[Any],
    ) -> AnthropicStreamingResponse:
        parent = parent_span or get_current_iteration_span() or get_current_chat_span()
        context = trace.set_span_in_context(parent) if parent else None
        span = get_tracer().start_span(span_name, context=context)

        request = self.build_request(messages, tools, max_tokens)
        span.set_attribute("llm.model", self.config.name)
        span.set_attribute("llm.provider", self.config.provider)
        span.set_attribute("llm.temperature_omitted", True)
        span.set_attribute("llm.max_tokens", request["max_tokens"])
        if "thinking" in request:
            span.set_attribute("llm.thinking", request["thinking"]["type"])
        if tools:
            span.set_attribute("llm.tools_count", len(tools))

        budget = check_context_budget(
            self.config, messages, tools, max_output_tokens=request["max_tokens"]
        )
        record_budget_on_span(span, budget)

        logger.info(f"Making Anthropic {span_name} request to {self.config.name}")
        try:
            events = self._client.messages.create(stream=True, **request)
        except Exception as e:
            span.set_attribute("llm.error", str(e))
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()
            raise
        return AnthropicStreamingResponse(
            events, span=span, estimated_prompt_tokens=budget.estimated_tokens
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        parent_span: Optional[Any] = None,
    ) -> AnthropicStreamingResponse:
        """Streaming request. ``temperature`` is accepted and ignored."""
        return self._open("llm.stream", messages, tools, max_tokens, parent_span)

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        parent_span: Optional[Any] = None,
    ) -> CompletionResponse:
        """Non-streaming result, gathered from a stream.

        The SDK refuses non-streaming requests it expects to run past ten
        minutes, which a large ``max_tokens`` with thinking easily is.
        """
        response = self._open("llm.complete", messages, tools, max_tokens, parent_span).drain()
        prompt_tokens, completion_tokens = response.usage
        return CompletionResponse(
            content=response.content or None,
            reasoning=response.reasoning or None,
            tool_calls=response.get_accumulated_tool_calls(),
            finish_reason=response.finish_reason,
            provider_specific_fields=response.get_provider_specific_fields(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
