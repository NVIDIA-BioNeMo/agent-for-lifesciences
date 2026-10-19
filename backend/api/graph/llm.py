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

"""Custom LangChain ChatModel wrapping Aevom's LLMClient.

Delegates to the existing LLMClient while conforming to LangChain's
BaseChatModel interface. This preserves NVIDIA-specific behaviour:
reasoning_content, Gemini 3 thought_signatures, model templates, and
telemetry span creation.
"""

import itertools
import json
import logging
import time
from typing import Any, Iterator, List, Optional

import anthropic
from openai import APIError, APIStatusError

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from api.llm.config import PROVIDER_ANTHROPIC

logger = logging.getLogger(__name__)

_RETRY_DELAYS = [2, 4, 8, 16, 32, 32]
_MAX_RETRY_ATTEMPTS = 7

# Codes on the HTTP response itself. 503 is here because NVIDIA answers
# "Service temporarily overloaded" that way on the non-streaming path, and
# capacity is the one thing waiting genuinely fixes. 500 and 502 are
# deliberately absent: on the response, before any of the request has been
# accepted, they are as likely to mean the request itself is unservable, and
# retrying that only multiplies the latency of a failure.
_RETRYABLE_STATUS_CODES = {429, 503, 504}

# Codes reported inside a streamed error frame rather than on the HTTP
# response. The provider already answered 200 and accepted the request, so
# anything it reports here is a serving failure and worth retrying -- which is
# why this set is wider than the one above. NVIDIA reports a full inference
# worker this way: `{"code": 500, "type": "internal_server_error",
# "message": "ResourceExhausted: Worker local total request limit reached"}`.
_RETRYABLE_STREAM_ERROR_CODES = {429, 500, 502, 503, 504}


def _unexplained_404(error: APIStatusError) -> bool:
    """Is this a 404 the provider declined to explain?

    NVIDIA's gateway answers a request it cannot place with a 404 whose body is
    empty, which says nothing is missing -- one unchanged minimal request
    repeated during an unstable window drew body-less 404s interleaved with
    200s from a model that was serving the whole time. A route or model name
    that genuinely does not exist answers `404 page not found` instead, so a
    body is what marks the failure as terminal.
    """
    return error.status_code == 404 and not getattr(error, "body", None)


# Anthropic. 529 is "overloaded", its capacity signal. An error event inside a
# stream reaches us as an APIStatusError carrying the stream's own 200, so
# those are classified by the error type in the body instead.
_ANTHROPIC_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504, 529}
_ANTHROPIC_RETRYABLE_ERROR_TYPES = {
    "overloaded_error": 529,
    "rate_limit_error": 429,
    "api_error": 500,
}


def _anthropic_retryable_code(error: anthropic.APIError) -> Optional[int]:
    status = getattr(error, "status_code", None)
    if status in _ANTHROPIC_RETRYABLE_STATUS_CODES:
        return status
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error") if isinstance(body.get("error"), dict) else body
        return _ANTHROPIC_RETRYABLE_ERROR_TYPES.get(inner.get("type"))
    return None


def _retryable_code(error: Exception) -> Optional[int]:
    """Return the status code to retry on, or None if the error is terminal."""
    if isinstance(error, anthropic.APIError):
        return _anthropic_retryable_code(error)
    if isinstance(error, APIStatusError):
        code = error.status_code
        if code in _RETRYABLE_STATUS_CODES or _unexplained_404(error):
            return code
        return None
    if isinstance(error, APIError):
        body = getattr(error, "body", None)
        code = body.get("code") if isinstance(body, dict) else None
        return code if code in _RETRYABLE_STREAM_ERROR_CODES else None
    return None


class ProviderUnavailableError(Exception):
    """The provider could not serve the request after every retry.

    Carries prose written for the user, because the streaming layer shows the
    message itself instead of an ``Error: ...`` frame: nothing about the request
    is wrong and there is nothing to act on but time.
    """

    def __init__(self, model_id: str, message: str):
        self.model_id = model_id
        super().__init__(message)


class RateLimitExhaustedError(ProviderUnavailableError):
    def __init__(self, model_id: str):
        super().__init__(
            model_id,
            "It looks like we're encountering rate limits with this model provider. "
            "Please try again a little later.",
        )


class ProviderErrorExhaustedError(ProviderUnavailableError):
    """Retries ran out on a failure that was not the provider rate limiting us."""

    def __init__(self, model_id: str):
        super().__init__(
            model_id,
            "The model provider could not serve this request after several "
            "retries. Please try again in a moment.",
        )


class EmptyStreamError(Exception):
    """A 200 whose response body ended without carrying a single event.

    The provider accepts the request, holds the connection -- 300s in every
    occurrence observed so far -- and then closes the body having sent nothing.
    No exception is raised on that path: the SDK's iterator simply stops. So
    this is raised deliberately, because the alternative is the zero-chunk
    stream reaching LangChain, which converts it into a bare
    ``ValueError: No generations found in stream.``
    """


class EmptyStreamExhaustedError(ProviderUnavailableError):
    def __init__(self, model_id: str):
        super().__init__(
            model_id,
            "The model provider accepted the request but returned an empty "
            "response, and retrying did not help. Please try again.",
        )


# The delimiter models in the content_has_think_tags family use to close
# their reasoning block. LLMClient strips it when splitting reasoning out of
# the content stream, so re-attachment has to put it back -- exactly once.
_THINK_END_TAG = "</think>"


def _attach_reasoning(entry: dict, reasoning: str, config: Any) -> None:
    """Put an assistant turn's reasoning back on the wire.

    Two shapes exist. Most reasoning models take ``reasoning_content`` as a
    sibling of ``content``. The ``content_has_think_tags`` family instead
    expects the reasoning inline, terminated by ``</think>`` -- and since
    ``LLMClient`` consumed that delimiter on the way in, this is the only
    place that can restore it.

    Args:
        entry: The assistant message being built, mutated in place.
        reasoning: Reasoning text captured from the model's own output.
        config: The ModelConfig for the call.
    """
    if getattr(config, "content_has_think_tags", False):
        content = entry.get("content") or ""
        if _THINK_END_TAG in content:
            # The delimiter survived somewhere upstream; prefixing a second
            # reasoning block would leave the model with two of them.
            return
        entry["content"] = f"{reasoning}{_THINK_END_TAG}{content}"
    else:
        entry["reasoning_content"] = reasoning


def _langchain_to_openai_messages(
    messages: List[BaseMessage],
    config: Any = None,
) -> list[dict]:
    """Convert LangChain messages to OpenAI-format dicts.

    LangGraph manages messages as BaseMessage subclasses. The existing
    LLMClient.complete() expects OpenAI-format dicts. This bridge converts
    between the two without loss.

    Args:
        messages: The LangGraph message list.
        config: ModelConfig for the call. Required to decide whether the
            model's own reasoning is sent back to it; without one, reasoning
            is dropped, which is the pre-existing behaviour.
    """
    retains_reasoning = bool(getattr(config, "retains_reasoning", False))
    is_anthropic = getattr(config, "provider", None) == PROVIDER_ANTHROPIC

    result: list[dict] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": msg.content or "",
            }
            reasoning = msg.additional_kwargs.get("reasoning")
            if retains_reasoning and reasoning:
                _attach_reasoning(entry, reasoning, config)
            if msg.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                json.dumps(tc["args"])
                                if isinstance(tc["args"], dict)
                                else tc["args"]
                            ),
                        },
                    }
                    for tc in msg.tool_calls
                ]
            # Preserve provider_specific_fields (thought signatures for Gemini 3)
            psf = msg.additional_kwargs.get("provider_specific_fields")
            if psf:
                entry["provider_specific_fields"] = psf
            result.append(entry)
        elif isinstance(msg, ToolMessage):
            tool_entry: dict[str, Any] = {
                "role": "tool",
                "content": msg.content,
                "tool_call_id": msg.tool_call_id,
                "name": msg.name or "",
            }
            # Only the Anthropic client understands this; OpenAI-compatible
            # endpoints are not sent fields they do not define.
            if is_anthropic and getattr(msg, "status", None) == "error":
                tool_entry["is_error"] = True
            result.append(tool_entry)
        else:
            result.append({"role": "user", "content": str(msg.content)})
    return result


def _openai_tools_from_langchain(
    tools: Optional[list[dict]],
) -> Optional[list[dict]]:
    """Pass through OpenAI-format tool schemas.

    LangGraph binds tools to the model and passes their schemas in the
    ``tools`` kwarg. Our tools already produce OpenAI-format schemas via
    ``ToolRegistry.get_all_schemas()``, so this is a no-op pass-through.
    """
    return tools if tools else None


class ChatAevomLLM(BaseChatModel):
    """LangChain ChatModel backed by Aevom's LLMClient.

    Usage::

        from api.llm import get_client
        from api.graph.llm import ChatAevomLLM

        llm = ChatAevomLLM(client=get_client())
        # Bind tools (optional)
        llm_with_tools = llm.bind_tools(tool_schemas)
    """

    client: Any  # LLMClient instance — use Any to avoid Pydantic validation issues
    model_config_obj: Any = None  # Optional ModelConfig override

    class Config:
        arbitrary_types_allowed = True

    def _with_rate_limit_retry(self, call_fn):
        """Wrap an LLM call with exponential back-off for retryable failures."""
        from api.graph.tools import push_retry_status

        model_id = self.client.config.id
        last_error = None

        for attempt in range(_MAX_RETRY_ATTEMPTS):
            try:
                result = call_fn()
                if attempt > 0:
                    push_retry_status(None)
                return result
            except (APIError, anthropic.APIError, EmptyStreamError) as e:
                if isinstance(e, EmptyStreamError):
                    reason = "empty response"
                    busy_message = f"{model_id} returned an empty response, retrying..."
                    retry_message = f"Retrying {model_id}"
                else:
                    code = _retryable_code(e)
                    if code is None:
                        raise
                    reason = str(code)
                    busy_message = f"{code} {model_id} is a little busy..."
                    retry_message = f"{code} Retrying {model_id}"

                last_error = e
                logger.warning(
                    f"Retryable {reason} for {model_id} (attempt {attempt + 1}/{_MAX_RETRY_ATTEMPTS})"
                )
                # Every attempt reports. The back-off now runs to ~94s, and a
                # client told only about attempts 1 and 3 spent most of that
                # watching a stream with nothing on it.
                push_retry_status({
                    "message": busy_message if attempt == 0 else retry_message,
                    "attempt": attempt + 1,
                })

                if attempt < _MAX_RETRY_ATTEMPTS - 1:
                    time.sleep(_RETRY_DELAYS[attempt])

        push_retry_status(None)
        if isinstance(last_error, EmptyStreamError):
            raise EmptyStreamExhaustedError(model_id) from last_error
        if _retryable_code(last_error) == 429:
            raise RateLimitExhaustedError(model_id) from last_error
        raise ProviderErrorExhaustedError(model_id) from last_error

    @property
    def _llm_type(self) -> str:
        return "aevom-llm"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.client.config.name}

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation delegating to LLMClient.complete()."""
        openai_msgs = _langchain_to_openai_messages(messages, self.client.config)
        tools = _openai_tools_from_langchain(kwargs.pop("tools", None))
        temperature = kwargs.pop("temperature", None)
        max_tokens = kwargs.pop("max_tokens", None)

        response = self._with_rate_limit_retry(
            lambda: self.client.complete(
                messages=openai_msgs,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )

        return self._completion_to_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Streaming generation delegating to LLMClient.stream()."""
        openai_msgs = _langchain_to_openai_messages(messages, self.client.config)
        tools = _openai_tools_from_langchain(kwargs.pop("tools", None))
        temperature = kwargs.pop("temperature", None)
        max_tokens = kwargs.pop("max_tokens", None)

        # Priming the stream -- pulling its first chunk -- has to happen inside
        # the retry wrapper. The provider can accept the request with a 200 and
        # then make the very first thing in the body an error frame, so opening
        # the stream is no evidence it will produce anything. Retrying is only
        # safe this early, before any chunk has been yielded downstream: there
        # is no partial reasoning or content for a second attempt to duplicate.
        def _open_primed_stream():
            resp = self.client.stream(
                messages=openai_msgs,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            first = next(resp, None)
            if first is None:
                # The body ended without a single event, so nothing was buffered
                # either -- there is no partial output a retry could duplicate.
                raise EmptyStreamError(
                    f"{self.client.config.id} returned a response body with no events"
                )
            return resp, first

        streaming_resp, first_chunk = self._with_rate_limit_retry(_open_primed_stream)

        chunks = itertools.chain([first_chunk], streaming_resp)

        for chunk in chunks:
            lc_chunk = AIMessageChunk(content=chunk.content or "")

            if chunk.reasoning:
                lc_chunk.additional_kwargs["reasoning"] = chunk.reasoning

            if chunk.thought_signatures:
                lc_chunk.additional_kwargs.setdefault(
                    "provider_specific_fields", {}
                )["thought_signatures"] = chunk.thought_signatures

            yield ChatGenerationChunk(message=lc_chunk)

        # After stream ends, attach accumulated tool calls if any
        accumulated = streaming_resp.get_accumulated_tool_calls()
        if accumulated:
            tool_calls_lc = []
            for tc in accumulated:
                args_str = tc["function"]["arguments"]
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": args_str}
                tool_calls_lc.append({
                    "name": tc["function"]["name"],
                    "args": args,
                    "id": tc["id"],
                })

            final_chunk = AIMessageChunk(
                content="",
                tool_calls=tool_calls_lc,
            )
            psf = streaming_resp.get_provider_specific_fields()
            if psf:
                final_chunk.additional_kwargs["provider_specific_fields"] = psf

            yield ChatGenerationChunk(message=final_chunk)

    # ------------------------------------------------------------------
    # Tool binding — converts LangChain tool objects to OpenAI schemas
    # ------------------------------------------------------------------

    def bind_tools(
        self,
        tools: list[Any],
        **kwargs: Any,
    ) -> Any:
        """Bind tools so they're sent with every LLM call.

        Accepts either:
        - Raw OpenAI-format schema dicts (from ToolRegistry.get_all_schemas())
        - LangChain BaseTool / StructuredTool instances

        Uses LangChain's ``Runnable.bind()`` to attach schemas as kwargs
        that are forwarded to ``_generate()`` / ``_stream()`` on every call.
        """
        schemas: list[dict] = []
        for t in tools:
            if isinstance(t, dict):
                schemas.append(t)
            elif hasattr(t, "tool_call_schema"):
                schema = {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.args_schema.model_json_schema() if t.args_schema else {},
                    },
                }
                schemas.append(schema)
            else:
                raise ValueError(f"Unsupported tool type: {type(t)}")

        return self.bind(tools=schemas, **kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _completion_to_chat_result(response) -> ChatResult:
        """Convert LLMClient CompletionResponse to LangChain ChatResult."""
        additional_kwargs: dict[str, Any] = {}

        if response.reasoning:
            additional_kwargs["reasoning"] = response.reasoning

        if response.provider_specific_fields:
            additional_kwargs["provider_specific_fields"] = (
                response.provider_specific_fields
            )

        tool_calls_lc = []
        if response.tool_calls:
            for tc in response.tool_calls:
                args_str = tc["function"]["arguments"]
                try:
                    args = json.loads(args_str)
                except (json.JSONDecodeError, TypeError):
                    args = {"raw": args_str}
                tc_entry: dict[str, Any] = {
                    "name": tc["function"]["name"],
                    "args": args,
                    "id": tc["id"],
                }
                tool_calls_lc.append(tc_entry)

                # Preserve per-tool thought_signature (Gemini 3)
                tc_psf = tc.get("provider_specific_fields")
                if tc_psf:
                    tc_entry["provider_specific_fields"] = tc_psf

        ai_message = AIMessage(
            content=response.content or "",
            tool_calls=tool_calls_lc,
            additional_kwargs=additional_kwargs,
        )

        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])
