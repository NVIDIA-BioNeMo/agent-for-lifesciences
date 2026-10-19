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

"""LLM client wrapper using OpenAI SDK.

Provides a unified interface for LLM calls with support for
streaming, tool calling, reasoning content, and template-based prompts.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

import httpx
from openai import OpenAI
from opentelemetry.trace import Status, StatusCode

from api.context import (
    check_context_budget,
    record_budget_on_span,
    record_estimate_error,
)
from api.telemetry import get_tracer, get_current_iteration_span, get_current_chat_span
from .config import PROVIDER_ANTHROPIC, PROVIDER_OPENAI, ModelConfig, get_active_model
from .template_loader import get_template_loader

logger = logging.getLogger(__name__)

# The SDK default is connect=5s, which is shorter than the TCP handshake to
# inference-api.nvidia.com actually takes when that path is congested -- 16.5s
# measured on 2026-08-04. Every attempt then failed at 5s and the call aborted
# after the retries, so a healthy endpoint looked like a dead one. Read stays
# long because reasoning models on that host queue for minutes before emitting
# a first token.
LLM_CONNECT_TIMEOUT_S = float(os.getenv("AEVOM_LLM_CONNECT_TIMEOUT", "30"))
LLM_READ_TIMEOUT_S = float(os.getenv("AEVOM_LLM_READ_TIMEOUT", "600"))
LLM_TIMEOUT = httpx.Timeout(
    connect=LLM_CONNECT_TIMEOUT_S,
    read=LLM_READ_TIMEOUT_S,
    write=LLM_READ_TIMEOUT_S,
    pool=LLM_READ_TIMEOUT_S,
)


@dataclass
class StreamChunk:
    """A chunk from the streaming response.
    
    Attributes:
        content: Regular response content.
        reasoning: Reasoning/thinking content (if model supports it).
        tool_calls: Tool call deltas (partial tool call data).
        finish_reason: Reason for finishing (if this is the final chunk).
        thought_signatures: Gemini 3 thought signatures (may arrive in final chunk).
    """
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: Optional[list[dict[str, Any]]] = None
    finish_reason: Optional[str] = None
    thought_signatures: Optional[list[str]] = None


@dataclass
class CompletionResponse:
    """Response from a non-streaming completion.
    
    Attributes:
        content: The response content.
        reasoning: Reasoning content (if available).
        tool_calls: List of tool calls requested by the model.
        finish_reason: Reason for finishing.
        provider_specific_fields: Provider-specific fields including thought_signatures for Gemini 3.
    """
    content: Optional[str] = None
    reasoning: Optional[str] = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    finish_reason: Optional[str] = None
    provider_specific_fields: Optional[dict[str, Any]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _get_api_key(key_file: str) -> str:
    """Read API key from mounted secret file.

    Args:
        key_file: Path to the API key file.

    Returns:
        API key string.

    Raises:
        RuntimeError: If the key file is missing, is a directory, or is empty.
            A directory at this path almost always means a Docker bind mount
            silently created an empty stub directory because the host file was
            missing during ``docker compose up``. Preflight
            (``scripts/check-prerequisites.sh``) and the
            ``create_host_path: false`` bind option in the compose files are
            the primary guards; this branch is a clearer last-ditch error.
    """
    path = Path(key_file)
    try:
        if path.is_dir():
            raise RuntimeError(
                f"API key path is a directory, not a file: {key_file}. "
                "This usually means the host key file was missing when the "
                "container started and Docker created an empty directory in "
                "its place. Remove the stub directory on the host (e.g. "
                "`rm -rf API_KEY_NVINF`), recreate it as a file with your "
                "API key, and restart the stack. Run `make check` to verify."
            )
        key = path.read_text().strip()
    except FileNotFoundError:
        raise RuntimeError(f"API key file not found: {key_file}")
    if not key:
        raise RuntimeError(f"API key file is empty: {key_file}")
    return key


def render_system_prompt(
    config: ModelConfig,
    base_prompt: str,
    tools: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Render ``base_prompt`` through the model's Jinja2 template.

    Shared by every client so a model renders the same prompt whichever
    provider serves it.
    """
    template_loader = get_template_loader()
    template_name = config.template_name

    # Check if template exists, fall back to base if not
    if not template_loader.has_template(template_name):
        logger.warning(
            f"Template '{template_name}' not found for model {config.id}, "
            f"using base template"
        )
        template_name = "base"

    context = {
        "system_prompt": base_prompt,
        "supports_reasoning": config.supports_reasoning,
        "model_name": config.name,
        "model_id": config.id,
        "tools": tools or [],
    }

    try:
        rendered = template_loader.render(template_name, context)
        logger.debug(f"Rendered system prompt using template '{template_name}'")
        return rendered
    except Exception as e:
        logger.error(f"Failed to render template '{template_name}': {e}")
        # Fall back to base prompt on error
        return base_prompt


class LLMClient:
    """Client for LLM interactions using OpenAI SDK.
    
    Wraps the OpenAI SDK with configuration-driven model selection
    and support for reasoning content.
    
    Example:
        client = LLMClient()
        
        # Non-streaming completion
        response = client.complete(messages, tools)
        
        # Streaming
        for chunk in client.stream(messages):
            if chunk.reasoning:
                print(f"Thinking: {chunk.reasoning}")
            if chunk.content:
                print(chunk.content, end="")
    """
    
    def __init__(self, model_config: Optional[ModelConfig] = None):
        """Initialize the LLM client.
        
        Args:
            model_config: Model configuration. If None, uses active model from env.
        """
        self.config = model_config or get_active_model()
        self._client = OpenAI(
            base_url=self.config.base_url,
            api_key=_get_api_key(self.config.api_key_file),
            timeout=LLM_TIMEOUT,
        )
        self._template_loader = get_template_loader()
        logger.info(f"LLMClient initialized with model: {self.config.name}")
    
    def render_system_prompt(
        self,
        base_prompt: str,
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> str:
        """Render the system prompt using the model's template.
        
        Uses the model's configured template to format the system prompt
        with model-specific instructions for reasoning and tool usage.
        
        Args:
            base_prompt: The base system prompt content (e.g., from prompts.md).
            tools: Optional list of tool schemas to include in the prompt.
            
        Returns:
            Rendered system prompt string.
        """
        return render_system_prompt(self.config, base_prompt, tools)
    
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        parent_span: Optional[Any] = None,
    ) -> CompletionResponse:
        """Make a non-streaming completion request.
        
        Used primarily for tool call detection where we need
        the complete response before processing.
        
        Args:
            messages: List of chat messages.
            tools: Optional list of tool schemas.
            temperature: Override default temperature (ignored for Gemini 3, and
                for models whose endpoint rejects the parameter entirely).
            max_tokens: Override default max tokens.
            parent_span: Optional explicit parent span (for thread pool contexts
                where ContextVars don't propagate).
            
        Returns:
            CompletionResponse with content and tool calls.
        """
        from opentelemetry import trace
        
        tracer = get_tracer()
        # Use explicit parent_span if provided (for thread pool contexts),
        # otherwise try to get from ContextVars
        parent = parent_span or get_current_iteration_span() or get_current_chat_span()
        context = trace.set_span_in_context(parent) if parent else None
        
        with tracer.start_as_current_span("llm.complete", context=context) as span:
            # For Gemini 3 (requires_thought_signatures), lock temperature at config value
            # Temperature must be 1.0 for reasoning engine stability
            if self.config.requires_thought_signatures:
                effective_temperature = self.config.default_temperature
            else:
                # `is not None` rather than truthiness: temperature=0.0 is a
                # deliberate request for greedy decoding, not an absent value.
                effective_temperature = (
                    temperature
                    if temperature is not None
                    else self.config.default_temperature
                )
            
            # Set span attributes
            span.set_attribute("llm.model", self.config.name)
            if self.config.supports_temperature:
                span.set_attribute("llm.temperature", effective_temperature)
            else:
                span.set_attribute("llm.temperature_omitted", True)
            span.set_attribute("llm.max_tokens", max_tokens or self.config.max_tokens)
            if tools:
                span.set_attribute("llm.tools_count", len(tools))
            
            kwargs: dict[str, Any] = {
                "model": self.config.name,
                "messages": messages,
                "max_tokens": max_tokens or self.config.max_tokens,
                "stream": False,
            }
            
            # Endpoints that reject the parameter outright get no temperature at
            # all -- sending one 400s the request regardless of its value.
            if self.config.supports_temperature:
                kwargs["temperature"] = effective_temperature
            
            # Add presence_penalty if configured
            if self.config.presence_penalty is not None:
                kwargs["presence_penalty"] = self.config.presence_penalty
            
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            
            extra_body = self.config.get_extra_body()
            if extra_body:
                kwargs["extra_body"] = extra_body
            
            budget = check_context_budget(
                self.config,
                messages,
                tools,
                max_output_tokens=kwargs["max_tokens"],
            )
            record_budget_on_span(span, budget)
            
            try:
                logger.info(f"Making completion request to {self.config.name}")
                response = self._client.chat.completions.create(**kwargs)

                # --- Token usage ---
                usage = getattr(response, "usage", None)
                prompt_tok = usage.prompt_tokens if usage else 0
                completion_tok = usage.completion_tokens if usage else 0

                span.set_attribute("llm.usage.input_tokens", prompt_tok)
                span.set_attribute("llm.usage.output_tokens", completion_tok)
                record_estimate_error(span, budget.estimated_tokens, prompt_tok)

                from api.telemetry import get_token_counter, get_token_accumulator

                counter = get_token_counter()
                if counter and prompt_tok:
                    counter.add(prompt_tok, {"token_type": "input"})
                if counter and completion_tok:
                    counter.add(completion_tok, {"token_type": "output"})

                acc = get_token_accumulator()
                if acc:
                    acc.add(prompt_tok, completion_tok)

                message = response.choices[0].message
                
                # Extract tool calls WITH thought_signature preservation for Gemini 3
                tool_calls = []
                if message.tool_calls:
                    for tc in message.tool_calls:
                        tool_call_dict: dict[str, Any] = {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        # Preserve per-tool-call thought_signature (Gemini 3)
                        # The signature is an opaque blob - never truncate or modify
                        tc_psf = getattr(tc, "provider_specific_fields", None)
                        if tc_psf:
                            # Convert to dict if needed (may be Pydantic model)
                            if hasattr(tc_psf, "model_dump"):
                                tc_psf = tc_psf.model_dump()
                            elif hasattr(tc_psf, "__dict__"):
                                tc_psf = dict(tc_psf)
                            if tc_psf.get("thought_signature"):
                                tool_call_dict["provider_specific_fields"] = {
                                    "thought_signature": tc_psf["thought_signature"]
                                }
                        tool_calls.append(tool_call_dict)
                
                # Capture message-level provider_specific_fields (Gemini 3 thought_signatures)
                provider_specific_fields = None
                msg_psf = getattr(message, "provider_specific_fields", None)
                if msg_psf:
                    # Convert to dict if needed
                    if hasattr(msg_psf, "model_dump"):
                        provider_specific_fields = msg_psf.model_dump()
                    elif hasattr(msg_psf, "__dict__"):
                        provider_specific_fields = dict(msg_psf)
                    else:
                        provider_specific_fields = msg_psf
                
                # Store RAW data for training (no truncation, no processing)
                span.set_attribute("llm.raw.content", message.content or "")
                if message.tool_calls:
                    raw_tool_calls = json.dumps([{
                        "id": tc.id,
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments  # Keep as raw JSON string
                        }
                    } for tc in message.tool_calls])
                    span.set_attribute("llm.raw.tool_calls", raw_tool_calls)
                
                # Record reasoning content (raw only - no duplicates)
                reasoning = getattr(message, "reasoning_content", None)
                
                # Fallback: extract clear-text reasoning from content for models
                # that embed it before a </think> delimiter (Nemotron Super)
                if not reasoning and self.config.content_has_think_tags and message.content:
                    think_parts = message.content.split("</think>", 1)
                    if len(think_parts) == 2:
                        raw_thinking = think_parts[0]
                        if raw_thinking.startswith("<think>"):
                            raw_thinking = raw_thinking[len("<think>"):]
                        reasoning = raw_thinking.strip()
                        message.content = think_parts[1].strip() or None
                
                if reasoning:
                    span.set_attribute("llm.raw.reasoning", reasoning)
                    span.set_attribute("llm.has_reasoning", True)
                
                # Record thought signatures in span for debugging
                if provider_specific_fields and provider_specific_fields.get("thought_signatures"):
                    span.set_attribute("llm.has_thought_signatures", True)
                    span.set_attribute(
                        "llm.thought_signatures_count", 
                        len(provider_specific_fields["thought_signatures"])
                    )
                
                # Record tool calls in span (backward compat - processed format)
                if tool_calls:
                    span.set_attribute("llm.tool_calls_count", len(tool_calls))
                    tool_names = [tc["function"]["name"] for tc in tool_calls]
                    span.set_attribute("llm.tool_calls", ",".join(tool_names))
                
                span.set_status(Status(StatusCode.OK))
                
                return CompletionResponse(
                    content=message.content,
                    reasoning=reasoning,
                    tool_calls=tool_calls,
                    finish_reason=response.choices[0].finish_reason,
                    provider_specific_fields=provider_specific_fields,
                    prompt_tokens=prompt_tok,
                    completion_tokens=completion_tok,
                )
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                # Log full request payload for 400 errors to help debug missing signatures
                if hasattr(e, 'status_code') and e.status_code == 400:
                    logger.error(
                        f"400 error from {self.config.name}. "
                        f"This may indicate missing thought signatures. "
                        f"Message count: {len(messages)}"
                    )
                raise
    
    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        parent_span: Optional[Any] = None,
    ) -> "StreamingResponse":
        """Make a streaming completion request.
        
        Args:
            messages: List of chat messages.
            tools: Optional list of tool schemas.
            temperature: Override default temperature (ignored for Gemini 3, and
                for models whose endpoint rejects the parameter entirely).
            max_tokens: Override default max tokens.
            parent_span: Optional explicit parent span (for thread pool contexts
                where ContextVars don't propagate).
            
        Returns:
            StreamingResponse iterator yielding StreamChunks.
        """
        from opentelemetry import trace
        
        tracer = get_tracer()
        # Use explicit parent_span if provided (for thread pool contexts),
        # otherwise try to get from ContextVars
        parent = parent_span or get_current_iteration_span() or get_current_chat_span()
        context = trace.set_span_in_context(parent) if parent else None
        
        # For Gemini 3 (requires_thought_signatures), lock temperature at config value
        if self.config.requires_thought_signatures:
            effective_temperature = self.config.default_temperature
        else:
            # See `complete()`: 0.0 must survive as 0.0.
            effective_temperature = (
                temperature
                if temperature is not None
                else self.config.default_temperature
            )
        
        # Start a span for the streaming call
        span = tracer.start_span("llm.stream", context=context)
        span.set_attribute("llm.model", self.config.name)
        if self.config.supports_temperature:
            span.set_attribute("llm.temperature", effective_temperature)
        else:
            span.set_attribute("llm.temperature_omitted", True)
        span.set_attribute("llm.max_tokens", max_tokens or self.config.max_tokens)
        if tools:
            span.set_attribute("llm.tools_count", len(tools))
        
        kwargs: dict[str, Any] = {
            "model": self.config.name,
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
        }
        
        # See `complete()`: omitted entirely where the endpoint rejects it.
        if self.config.supports_temperature:
            kwargs["temperature"] = effective_temperature
        
        # Add presence_penalty if configured
        if self.config.presence_penalty is not None:
            kwargs["presence_penalty"] = self.config.presence_penalty
        
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        
        extra_body = self.config.get_extra_body()
        if extra_body:
            kwargs["extra_body"] = extra_body
        
        kwargs["stream_options"] = {"include_usage": True}

        budget = check_context_budget(
            self.config,
            messages,
            tools,
            max_output_tokens=kwargs["max_tokens"],
        )
        record_budget_on_span(span, budget)

        logger.info(f"Making streaming request to {self.config.name}")

        def _create() -> Any:
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception as e:
                if "stream_options" not in str(e):
                    raise
                logger.info("Provider rejected stream_options, retrying without it")
                del kwargs["stream_options"]
                return self._client.chat.completions.create(**kwargs)

        try:
            stream = _create()
        except Exception as e:
            # This span is only handed to StreamingResponse below, so until then
            # closing it is this function's job -- and an unended span is never
            # exported, which is how the provider's body-less 404s produced a
            # failed `chat_request` with no `llm.stream` child to explain it.
            span.set_attribute("llm.error", str(e))
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.end()
            raise
        
        # Pass span and config to StreamingResponse for thought signature handling
        return StreamingResponse(
            stream, 
            self.config.supports_reasoning, 
            span=span,
            requires_thought_signatures=self.config.requires_thought_signatures,
            content_has_think_tags=self.config.content_has_think_tags,
            estimated_prompt_tokens=budget.estimated_tokens,
        )


class StreamingResponse:
    """Iterator for streaming responses.
    
    Handles parsing of streaming chunks including reasoning content
    and Gemini 3 thought signatures.
    
    CRITICAL: For Gemini 3, thought signatures may arrive in the final chunk
    even if no text is present. The parser MUST continue until finish_reason
    is received, regardless of whether text is still appearing.
    """
    
    # The </think> tag that models use to end the reasoning block
    _THINK_END_TAG = "</think>"
    
    def __init__(
        self, 
        stream, 
        supports_reasoning: bool, 
        span=None,
        requires_thought_signatures: bool = False,
        content_has_think_tags: bool = False,
        estimated_prompt_tokens: int = 0,
    ):
        """Initialize streaming response.
        
        Args:
            stream: The OpenAI streaming response.
            supports_reasoning: Whether to look for reasoning_content.
            span: Optional telemetry span to end when streaming completes.
            requires_thought_signatures: Whether to capture Gemini 3 thought signatures.
            content_has_think_tags: Whether reasoning is embedded in content before </think>.
            estimated_prompt_tokens: What the context budget predicted for this
                prompt. Compared against the provider's own count once the
                usage chunk arrives, which is what calibrates the estimator.
        """
        self._stream = stream
        self._supports_reasoning = supports_reasoning
        self._requires_thought_signatures = requires_thought_signatures
        self._content_has_think_tags = content_has_think_tags
        self._tool_call_buffer: dict[int, dict[str, Any]] = {}
        self._span = span
        self._has_reasoning = False
        self._reasoning_content: list[str] = []
        self._content_chunks: list[str] = []
        # Gemini 3 thought signature accumulation
        self._thought_signatures: list[str] = []
        self._per_tool_signatures: dict[int, str] = {}
        # Token usage (populated from final streaming chunk)
        self._usage_prompt_tokens = 0
        self._usage_completion_tokens = 0
        self._estimated_prompt_tokens = estimated_prompt_tokens
        # Think-tag streaming state: reasoning arrives before </think> in content
        self._think_in_reasoning: bool = content_has_think_tags
        self._think_tag_buffer: str = ""
        self._think_skipped_open_tag: bool = False
        # Counts events actually delivered by the provider. Zero means the body
        # ended empty, which is a serving failure rather than an empty answer.
        self._chunk_count = 0
    
    def __iter__(self):
        """Iterate over stream chunks."""
        return self
    
    def __next__(self) -> StreamChunk:
        """Get next chunk from stream.
        
        CRITICAL: This continues until finish_reason is received.
        For Gemini 3, thought signatures may be in the final chunk
        even when no text is present.
        """
        try:
            chunk = next(self._stream)
        except StopIteration:
            # End span when stream completes
            self._end_span()
            raise
        except Exception as e:
            self._fail_span(e)
            raise
        
        self._chunk_count += 1
        
        # Check for usage on EVERY chunk — with stream_options.include_usage,
        # the final chunk carries usage but has choices=[] (no content).
        usage = getattr(chunk, "usage", None)
        if usage:
            self._usage_prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            self._usage_completion_tokens = getattr(usage, "completion_tokens", 0) or 0

        if not chunk.choices:
            return StreamChunk()
        
        delta = chunk.choices[0].delta
        finish_reason = chunk.choices[0].finish_reason
        
        # Extract reasoning content if available
        reasoning = None
        if self._supports_reasoning:
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                self._has_reasoning = True
                self._reasoning_content.append(reasoning)
        
        # Capture thought_signatures from delta (Gemini 3)
        # CRITICAL: Signatures may arrive in final chunk - must not terminate early
        thought_signatures = None
        if self._requires_thought_signatures:
            delta_psf = getattr(delta, "provider_specific_fields", None)
            if delta_psf:
                # Convert to dict if needed
                if hasattr(delta_psf, "model_dump"):
                    delta_psf = delta_psf.model_dump()
                elif hasattr(delta_psf, "__dict__"):
                    delta_psf = dict(delta_psf)
                
                sigs = delta_psf.get("thought_signatures")
                if sigs:
                    thought_signatures = sigs
                    # Accumulate - treat as immutable opaque blobs
                    self._thought_signatures.extend(sigs)
        
        # Handle tool calls - they come in pieces across multiple chunks
        tool_calls = None
        if delta.tool_calls:
            tool_calls = []
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in self._tool_call_buffer:
                    self._tool_call_buffer[idx] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""}
                    }
                
                if tc_delta.id:
                    self._tool_call_buffer[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        self._tool_call_buffer[idx]["function"]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        self._tool_call_buffer[idx]["function"]["arguments"] += tc_delta.function.arguments
                
                # Capture per-tool-call thought_signature (Gemini 3)
                if self._requires_thought_signatures:
                    tc_psf = getattr(tc_delta, "provider_specific_fields", None)
                    if tc_psf:
                        if hasattr(tc_psf, "model_dump"):
                            tc_psf = tc_psf.model_dump()
                        elif hasattr(tc_psf, "__dict__"):
                            tc_psf = dict(tc_psf)
                        sig = tc_psf.get("thought_signature")
                        if sig:
                            # Store per-tool signature - immutable, opaque blob
                            self._per_tool_signatures[idx] = sig
                
                tool_calls.append(self._tool_call_buffer[idx])
        
        # --- Think-tag parsing: split reasoning from content for models that
        #     embed reasoning before </think> in the content stream. ---
        content_out = delta.content
        if self._content_has_think_tags and delta.content and not reasoning:
            content_out, reasoning = self._parse_think_content(delta.content)
        
        # Accumulate content for telemetry (raw, pre-parse)
        if delta.content:
            self._content_chunks.append(delta.content)
        
        # Don't end the span on finish_reason — the usage-only chunk
        # (choices=[], usage={...}) arrives AFTER this one. _end_span()
        # is called from the StopIteration handler once usage is captured.
        
        return StreamChunk(
            content=content_out,
            reasoning=reasoning,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            thought_signatures=thought_signatures,
        )
    
    def _parse_think_content(self, text: str) -> tuple[Optional[str], Optional[str]]:
        """Parse a content chunk, splitting reasoning from content at </think>.
        
        Handles cross-chunk boundaries where </think> may be split across
        two streaming chunks (e.g., "</thi" + "nk>...").
        
        Also strips any leading <think> tag from the first reasoning chunk.
        
        Returns:
            (content_out, reasoning_out) — at most one is non-None per call.
        """
        tag = self._THINK_END_TAG
        
        if not self._think_in_reasoning:
            # Already past the think region, pass through as content
            return text, None
        
        # Strip optional opening <think> tag from first content chunk
        if not self._think_skipped_open_tag:
            self._think_skipped_open_tag = True
            if text.startswith("<think>"):
                text = text[len("<think>"):]
                if not text:
                    return None, None
        
        # Prepend any buffered partial-tag text from a previous chunk
        if self._think_tag_buffer:
            text = self._think_tag_buffer + text
            self._think_tag_buffer = ""
        
        # Check for complete </think> tag in the combined text
        idx = text.find(tag)
        if idx != -1:
            # Found the end tag — split
            self._think_in_reasoning = False
            before = text[:idx]
            after = text[idx + len(tag):]
            
            reasoning_out = before if before else None
            if reasoning_out:
                self._has_reasoning = True
                self._reasoning_content.append(reasoning_out)
            
            # Strip leading whitespace/newline from content after tag
            content_out = after.lstrip("\n") if after else None
            return content_out, reasoning_out
        
        # Check if the text ends with a partial match for "</think>"
        # e.g., text ending in "<", "</", "</t", "</th", etc.
        # Buffer these trailing chars to combine with the next chunk.
        for k in range(1, len(tag)):
            if text.endswith(tag[:k]):
                safe = text[:-k]
                self._think_tag_buffer = text[-k:]
                reasoning_out = safe if safe else None
                if reasoning_out:
                    self._has_reasoning = True
                    self._reasoning_content.append(reasoning_out)
                return None, reasoning_out
        
        # Fully inside reasoning, no partial tag match at the end
        self._has_reasoning = True
        self._reasoning_content.append(text)
        return None, text
    
    def _end_span(self) -> None:
        """End the telemetry span for this streaming response."""
        if self._span:
            # Store RAW accumulated data for training (no truncation)
            full_content = "".join(self._content_chunks) if self._content_chunks else ""
            self._span.set_attribute("llm.raw.content", full_content)
            
            if self._tool_call_buffer:
                raw_tool_calls = json.dumps([{
                    "id": tc["id"],
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"]
                    }
                } for tc in self._tool_call_buffer.values()])
                self._span.set_attribute("llm.raw.tool_calls", raw_tool_calls)
                self._span.set_attribute("llm.tool_calls_count", len(self._tool_call_buffer))
            
            if self._has_reasoning:
                full_reasoning = "".join(self._reasoning_content)
                self._span.set_attribute("llm.raw.reasoning", full_reasoning)
                self._span.set_attribute("llm.has_reasoning", True)
            
            # Record thought signatures in telemetry
            if self._thought_signatures:
                self._span.set_attribute("llm.has_thought_signatures", True)
                self._span.set_attribute("llm.thought_signatures_count", len(self._thought_signatures))
            
            # Token usage
            self._span.set_attribute("llm.usage.input_tokens", self._usage_prompt_tokens)
            self._span.set_attribute("llm.usage.output_tokens", self._usage_completion_tokens)
            record_estimate_error(
                self._span, self._estimated_prompt_tokens, self._usage_prompt_tokens
            )

            from api.telemetry import get_token_counter, get_token_accumulator

            counter = get_token_counter()
            if counter and self._usage_prompt_tokens:
                counter.add(self._usage_prompt_tokens, {"token_type": "input"})
            if counter and self._usage_completion_tokens:
                counter.add(self._usage_completion_tokens, {"token_type": "output"})

            acc = get_token_accumulator()
            if acc:
                acc.add(self._usage_prompt_tokens, self._usage_completion_tokens)

            if self._chunk_count == 0:
                # Reached through StopIteration like any completed stream, so
                # this used to close OK -- which is how three ~300s zero-token
                # stalls stayed invisible to `phoenix_query.py errors`.
                message = "provider returned an empty stream (no events)"
                self._span.set_attribute("llm.error", message)
                self._span.set_status(Status(StatusCode.ERROR, message))
            else:
                self._span.set_status(Status(StatusCode.OK))
            self._span.end()
            self._span = None
    
    def _fail_span(self, error: Exception) -> None:
        """Close the span for a stream that died before completing.

        Only StopIteration reached `_end_span`, so a provider error abandoned
        the span unfinished and the failure never reached Phoenix at all -- it
        was visible only in the container log.
        """
        if self._span:
            self._span.set_attribute("llm.error", str(error))
            self._span.set_status(Status(StatusCode.ERROR, str(error)))
            self._span.end()
            self._span = None
    
    def get_accumulated_tool_calls(self) -> list[dict[str, Any]]:
        """Get all accumulated tool calls WITH thought signatures attached.
        
        For Gemini 3, each tool call may have a per-call thought_signature
        that must be preserved in the conversation history.
        
        Returns:
            List of complete tool call dictionaries with provider_specific_fields.
        """
        result = []
        for idx, tc in sorted(self._tool_call_buffer.items()):
            tc_copy = dict(tc)  # Shallow copy
            tc_copy["function"] = dict(tc["function"])  # Copy nested dict
            
            # Attach per-tool thought_signature if present
            if idx in self._per_tool_signatures:
                tc_copy["provider_specific_fields"] = {
                    "thought_signature": self._per_tool_signatures[idx]
                }
            result.append(tc_copy)
        return result
    
    def get_thought_signatures(self) -> list[str]:
        """Get all accumulated message-level thought signatures.
        
        Returns:
            List of thought signature strings (opaque blobs).
        """
        return list(self._thought_signatures)
    
    def get_provider_specific_fields(self) -> Optional[dict[str, Any]]:
        """Get provider_specific_fields for the complete message.
        
        Returns:
            Dict with thought_signatures if present, None otherwise.
        """
        if self._thought_signatures:
            return {"thought_signatures": list(self._thought_signatures)}
        return None


def create_client(model_config: Optional[ModelConfig] = None) -> Any:
    """Build the client that speaks ``model_config``'s provider.

    Returns an ``LLMClient``, an ``AnthropicLLMClient`` or an
    ``OpenAIResponsesClient``; all expose ``config``, ``render_system_prompt``,
    ``complete`` and ``stream``.
    """
    config = model_config or get_active_model()
    if config.provider == PROVIDER_ANTHROPIC:
        from .anthropic_client import AnthropicLLMClient

        return AnthropicLLMClient(config)
    if config.provider == PROVIDER_OPENAI:
        from .openai_responses_client import OpenAIResponsesClient

        return OpenAIResponsesClient(config)
    return LLMClient(model_config=config)


# Singleton client instance
_client: Optional[Any] = None


def get_client() -> Any:
    """Get or create the singleton client for the active model.
    
    Returns:
        The global client instance (see ``create_client``).
    """
    global _client
    if _client is None:
        _client = create_client()
    return _client


def reset_client() -> None:
    """Reset the singleton client (useful for testing or model switching)."""
    global _client
    _client = None

