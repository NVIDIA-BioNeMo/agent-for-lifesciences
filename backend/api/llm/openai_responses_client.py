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

"""LLM client for OpenAI models over the Responses API.

Presents the same surface as ``LLMClient`` -- ``config``,
``render_system_prompt``, ``complete`` and ``stream`` -- so ``ChatAevomLLM``
and the agent graph drive either one without knowing which. Chat Completions
cannot be used: it refuses function tools on the current reasoning models.

Requests are stateless (``store=False``). Reasoning items come back with
``encrypted_content`` and must be replayed, untouched and in order, on the
next request of a tool loop. They have no OpenAI chat-format equivalent, so
they ride along in ``provider_specific_fields`` (which ``ChatAevomLLM``
already round-trips) under ``openai_output_items``.
"""

import hashlib
import json
import logging
from typing import Any, Iterator, Optional

import openai
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from api.context import check_context_budget, record_budget_on_span, record_estimate_error
from api.state.format import INJECTION_SENTINEL
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

# Key in provider_specific_fields holding the assistant turn's item layout.
OUTPUT_ITEMS_KEY = "openai_output_items"

# Responses in-stream error codes, as the HTTP status the retry loop in
# api.graph.llm keys on. Anything unlisted is terminal.
_STREAM_ERROR_STATUS = {
    "rate_limit_exceeded": 429,
    "server_error": 500,
    "server_is_overloaded": 503,
    "overloaded": 503,
    "service_unavailable": 503,
}

_CACHE_KEY_PREFIX_CHARS = 8192

_INCOMPLETE_FINISH = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}


# ---------------------------------------------------------------------------
# OpenAI chat format -> Responses input items
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


def _user_item(content: Any) -> Optional[dict[str, Any]]:
    """A user message as an input item. Empty text is never sent."""
    if isinstance(content, list):
        parts = []
        for part in content:
            if not isinstance(part, dict):
                if str(part):
                    parts.append({"type": "input_text", "text": str(part)})
            elif part.get("type") == "text" and part.get("text"):
                parts.append({"type": "input_text", "text": part["text"]})
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if url:
                    parts.append({"type": "input_image", "image_url": url})
        return {"role": "user", "content": parts} if parts else None
    text = _text_of(content)
    return {"role": "user", "content": text} if text.strip() else None


def _function_call_item(tool_call: dict[str, Any], item_id: Optional[str] = None) -> dict[str, Any]:
    function = tool_call.get("function") or {}
    arguments = function.get("arguments")
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments or {})
    item = {
        "type": "function_call",
        "call_id": tool_call["id"],
        "name": function.get("name", ""),
        "arguments": arguments or "{}",
    }
    if item_id:
        item["id"] = item_id
    return item


def _message_item(text: str, item_id: Optional[str] = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }
    if item_id:
        item["id"] = item_id
    return item


def _assistant_items(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild an assistant turn in the order the model produced it.

    The stored layout supplies the reasoning items verbatim and the position
    (and item id) of the message and function_call items; their contents come
    from the message itself, because the context assembler may have degraded
    the text since.
    """
    text = _text_of(message.get("content"))
    tool_calls = {tc["id"]: tc for tc in message.get("tool_calls") or []}
    remaining = list(tool_calls)

    psf = message.get("provider_specific_fields") or {}
    layout = psf.get(OUTPUT_ITEMS_KEY) if isinstance(psf, dict) else None

    items: list[dict[str, Any]] = []
    text_placed = False
    for entry in layout or []:
        kind = entry.get("type")
        if kind == "reasoning":
            items.append(dict(entry))
        elif kind == "message" and not text_placed:
            text_placed = True
            if text.strip():
                items.append(_message_item(text, entry.get("id")))
        elif kind == "function_call" and entry.get("call_id") in remaining:
            remaining.remove(entry["call_id"])
            items.append(_function_call_item(tool_calls[entry["call_id"]], entry.get("id")))

    if not text_placed and text.strip():
        # Text goes after any reasoning but before the function calls.
        insert_at = next(
            (i for i, item in enumerate(items) if item["type"] == "function_call"), len(items)
        )
        items.insert(insert_at, _message_item(text))
    items.extend(_function_call_item(tool_calls[call_id]) for call_id in remaining)
    return items


def to_responses_input(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI chat-format messages to ``(instructions, input items)``.

    System and developer messages are hoisted into ``instructions``; each
    ``role: tool`` message becomes a ``function_call_output`` item.

    The live state envelope the graph appends to the system prompt (viewer,
    storage, session log, ...) changes on every iteration. OpenAI caches
    ``instructions`` as a whole, so leaving it there makes every call a full
    cache miss; it goes last in the input as a developer message instead,
    keeping everything before it identical from one call to the next.
    """
    instructions: list[str] = []
    envelopes: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role in ("system", "developer"):
            text = _text_of(message.get("content"))
            head, sentinel, envelope = text.partition(INJECTION_SENTINEL)
            if head.strip():
                instructions.append(head)
            if sentinel:
                envelopes.append((sentinel + envelope).strip())
        elif role == "assistant":
            items.extend(_assistant_items(message))
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message["tool_call_id"],
                    "output": _text_of(message.get("content")) or "(no output)",
                }
            )
        else:
            item = _user_item(message.get("content"))
            if item:
                items.append(item)
    if envelopes:
        items.append({"role": "developer", "content": "\n\n".join(envelopes)})
    return "\n\n".join(instructions), items


def to_responses_tools(tools: Optional[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Chat-format function schemas to flat Responses function tools."""
    converted = []
    for tool in tools or []:
        function = tool.get("function", tool)
        converted.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function.get("description") or "",
                "parameters": function.get("parameters") or {"type": "object", "properties": {}},
            }
        )
    return converted


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return dict(obj)


def _without_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _without_none(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_without_none(v) for v in value]
    return value


def _stream_error(error: dict[str, Any]) -> openai.APIError:
    code = error.get("code") or error.get("type")
    status = _STREAM_ERROR_STATUS.get(str(code))
    return openai.APIError(
        str(error.get("message") or error),
        request=None,
        body={"code": status, "type": code, "message": error.get("message")},
    )


class OpenAIResponsesStreamingResponse:
    """Iterate a Responses event stream as ``StreamChunk`` objects.

    Mirrors ``StreamingResponse``: tool calls are accumulated for
    ``get_accumulated_tool_calls()`` and the span is closed when the stream
    ends or fails.
    """

    def __init__(self, events: Any, span: Any = None, estimated_prompt_tokens: int = 0):
        self._events = iter(events)
        self._span = span
        self._estimated_prompt_tokens = estimated_prompt_tokens
        self._items: dict[int, dict[str, Any]] = {}
        self._arguments: dict[int, list[str]] = {}
        self._content: list[str] = []
        self._reasoning: list[str] = []
        self._finish_reason: Optional[str] = None
        self._input_tokens = 0
        self._output_tokens = 0
        self._reasoning_tokens = 0
        self._cached_tokens = 0
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
            try:
                chunk = self._handle(_as_dict(raw))
            except Exception as e:
                self._fail_span(e)
                raise
            if chunk is not None:
                return chunk

    def _handle(self, event: dict[str, Any]) -> Optional[StreamChunk]:
        kind = event.get("type")

        if kind == "response.created":
            return StreamChunk()

        if kind == "response.output_text.delta":
            text = event.get("delta", "")
            self._content.append(text)
            return StreamChunk(content=text) if text else None

        if kind == "response.reasoning_summary_part.added":
            if self._reasoning and not self._reasoning[-1].endswith("\n\n"):
                self._reasoning.append("\n\n")
                return StreamChunk(reasoning="\n\n")
            return None

        if kind == "response.reasoning_summary_text.delta":
            text = event.get("delta", "")
            self._reasoning.append(text)
            return StreamChunk(reasoning=text) if text else None

        if kind == "response.output_item.added":
            index = event.get("output_index", len(self._items))
            self._items[index] = dict(event.get("item") or {})
            if self._items[index].get("type") == "function_call":
                self._arguments[index] = []
            return None

        if kind == "response.function_call_arguments.delta":
            index = event.get("output_index", 0)
            self._arguments.setdefault(index, []).append(event.get("delta", ""))
            return None

        if kind == "response.output_item.done":
            index = event.get("output_index", len(self._items))
            self._items[index] = dict(event.get("item") or {})
            return None

        if kind in ("response.completed", "response.incomplete"):
            response = event.get("response") or {}
            self._record_usage(response.get("usage") or {})
            if kind == "response.incomplete":
                reason = (response.get("incomplete_details") or {}).get("reason")
                self._finish_reason = _INCOMPLETE_FINISH.get(reason, "length")
            else:
                self._finish_reason = "tool_calls" if self._function_calls() else "stop"
            return StreamChunk(finish_reason=self._finish_reason)

        if kind == "response.failed":
            raise _stream_error(((event.get("response") or {}).get("error")) or {})

        if kind == "error":
            raise _stream_error(event)

        return None

    def _record_usage(self, usage: dict[str, Any]) -> None:
        self._input_tokens = usage.get("input_tokens") or 0
        self._output_tokens = usage.get("output_tokens") or 0
        self._cached_tokens = (usage.get("input_tokens_details") or {}).get("cached_tokens") or 0
        self._reasoning_tokens = (
            (usage.get("output_tokens_details") or {}).get("reasoning_tokens") or 0
        )

    # -- accumulated results ------------------------------------------------

    def _function_calls(self) -> list[tuple[int, dict[str, Any]]]:
        return [
            (i, self._items[i])
            for i in sorted(self._items)
            if self._items[i].get("type") == "function_call"
        ]

    def get_accumulated_tool_calls(self) -> list[dict[str, Any]]:
        calls = []
        for index, item in self._function_calls():
            arguments = item.get("arguments") or "".join(self._arguments.get(index, [])) or "{}"
            calls.append(
                {
                    "id": item.get("call_id", ""),
                    "type": "function",
                    "function": {"name": item.get("name", ""), "arguments": arguments},
                }
            )
        return calls

    def get_thought_signatures(self) -> list[str]:
        return []

    def get_provider_specific_fields(self) -> Optional[dict[str, Any]]:
        """The output item layout, when the turn had reasoning to send back.

        Message and function_call entries are position markers (with their
        item ids) only; their contents live on the message. No entry may carry
        an ``index`` key: LangChain merges list items that share one when it
        concatenates chunks.
        """
        items = [self._items[i] for i in sorted(self._items)]
        if not any(item.get("type") == "reasoning" for item in items):
            return None
        layout = []
        for item in items:
            kind = item.get("type")
            if kind == "reasoning":
                entry = {
                    k: v
                    for k, v in _without_none(item).items()
                    if k in ("type", "id", "encrypted_content", "summary")
                }
                entry.setdefault("summary", [])
                layout.append(entry)
            elif kind == "message":
                layout.append({"type": "message", "id": item.get("id", "")})
            elif kind == "function_call":
                layout.append(
                    {"type": "function_call", "id": item.get("id", ""), "call_id": item.get("call_id", "")}
                )
        return {OUTPUT_ITEMS_KEY: layout}

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

    def drain(self) -> "OpenAIResponsesStreamingResponse":
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
        span.set_attribute("llm.usage.thinking_tokens", self._reasoning_tokens)
        span.set_attribute("llm.usage.cache_read_tokens", self._cached_tokens)
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


class OpenAIResponsesClient:
    """``LLMClient`` counterpart for ``provider="openai"`` models."""

    def __init__(self, model_config: ModelConfig, sdk_client: Any = None):
        self.config = model_config
        # The SDK's own Timeout type: it rejects an httpx object from a
        # different HTTP stack than the one it was built on.
        self._client = sdk_client or openai.OpenAI(
            api_key=_get_api_key(model_config.api_key_file),
            base_url=model_config.base_url,
            timeout=openai.Timeout(
                connect=LLM_CONNECT_TIMEOUT_S,
                read=LLM_READ_TIMEOUT_S,
                write=LLM_READ_TIMEOUT_S,
                pool=LLM_READ_TIMEOUT_S,
            ),
        )
        logger.info(f"OpenAIResponsesClient initialized with model: {model_config.name}")

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
        """Keyword arguments for ``responses.create``.

        No ``temperature``: the reasoning models reject it. ``store=False``
        keeps requests stateless, which is what makes reasoning items carry
        ``encrypted_content`` for replay.
        """
        instructions, items = to_responses_input(messages)
        request: dict[str, Any] = {
            "model": self.config.name,
            "input": items,
            "store": False,
            "max_output_tokens": max_tokens or self.config.max_tokens,
            "reasoning": {"summary": "auto"},
        }
        if self.config.effort:
            request["reasoning"]["effort"] = self.config.effort
        if instructions:
            request["instructions"] = instructions
        responses_tools = to_responses_tools(tools)
        if responses_tools:
            request["tools"] = responses_tools
            request["tool_choice"] = "auto"
        # Routes requests sharing the system prompt and tool list to the same
        # cache. Only the head is hashed, so a prompt that varies further down
        # (dev-mode reloads, per-mode additions) still lands on one cache.
        prefix = instructions[:_CACHE_KEY_PREFIX_CHARS] + "|" + ",".join(t["name"] for t in responses_tools)
        request["prompt_cache_key"] = hashlib.sha256(prefix.encode()).hexdigest()[:32]
        return request

    def _open(
        self,
        span_name: str,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]],
        max_tokens: Optional[int],
        parent_span: Optional[Any],
    ) -> OpenAIResponsesStreamingResponse:
        parent = parent_span or get_current_iteration_span() or get_current_chat_span()
        context = trace.set_span_in_context(parent) if parent else None
        span = get_tracer().start_span(span_name, context=context)

        request = self.build_request(messages, tools, max_tokens)
        span.set_attribute("llm.model", self.config.name)
        span.set_attribute("llm.provider", self.config.provider)
        span.set_attribute("llm.temperature_omitted", True)
        span.set_attribute("llm.max_tokens", request["max_output_tokens"])
        span.set_attribute("llm.reasoning_effort", self.config.effort or "default")
        if tools:
            span.set_attribute("llm.tools_count", len(tools))

        budget = check_context_budget(
            self.config, messages, tools, max_output_tokens=request["max_output_tokens"]
        )
        record_budget_on_span(span, budget)

        logger.info(f"Making OpenAI Responses {span_name} request to {self.config.name}")
        try:
            events = self._client.responses.create(stream=True, **request)
        except Exception as e:
            span.set_attribute("llm.error", str(e))
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()
            raise
        return OpenAIResponsesStreamingResponse(
            events, span=span, estimated_prompt_tokens=budget.estimated_tokens
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        parent_span: Optional[Any] = None,
    ) -> OpenAIResponsesStreamingResponse:
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
        """Non-streaming result, gathered from a stream."""
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
