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

"""Rebuild LangChain messages from the durable conversation record.

The boundary between the record and the model's context. The record stores
what happened; this turns a slice of it back into messages a provider will
accept.

"Will accept" is the hard part. A ``ToolMessage`` whose announcing tool call
is missing, or an assistant tool call whose result is missing, is a provider
400 -- and a 400 that surfaces as a generic bad-request is considerably harder
to diagnose than the missing history it replaced. So tool calls and their
results are only ever restored as a complete unit.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)


def _build_assistant(record: dict[str, Any]) -> AIMessage:
    """Rebuild an assistant message, chain of thought included.

    Reasoning goes back where the model produced it, on
    ``additional_kwargs``, because that is where ``retains_reasoning`` looks
    for it when re-emitting it to the provider.
    """
    additional_kwargs: dict[str, Any] = {}
    if record.get("reasoning"):
        additional_kwargs["reasoning"] = record["reasoning"]
    if record.get("extra"):
        additional_kwargs.update(record["extra"])

    return AIMessage(
        content=record.get("content") or "",
        tool_calls=record.get("tool_calls") or [],
        additional_kwargs=additional_kwargs,
    )


def rehydrate_messages(transcript: list[dict[str, Any]]) -> list[BaseMessage]:
    """Turn recorded messages into LangChain messages the provider will accept.

    Args:
        transcript: Rows from ``ConversationStore.get_transcript``, in order.

    Returns:
        Messages with every tool call paired to its results. Any pair that did
        not survive intact -- because the request was interrupted between the
        call and the result, most often -- is dropped whole rather than
        half-restored.
    """
    messages: list[BaseMessage] = []
    index = 0

    while index < len(transcript):
        record = transcript[index]
        role = record.get("role")
        index += 1

        if role == "system":
            messages.append(SystemMessage(content=record.get("content") or ""))
            continue
        if role == "user":
            messages.append(HumanMessage(content=record.get("content") or ""))
            continue
        if role == "tool":
            # Only reachable when a result outlived its call, which the block
            # handling below would otherwise have consumed.
            logger.debug("Dropping orphaned tool result from rehydrated history")
            continue
        if role != "assistant":
            continue

        assistant = _build_assistant(record)
        if not assistant.tool_calls:
            messages.append(assistant)
            continue

        results: list[ToolMessage] = []
        while index < len(transcript) and transcript[index].get("role") == "tool":
            result = transcript[index]
            index += 1
            results.append(
                ToolMessage(
                    content=result.get("content") or "",
                    tool_call_id=result.get("tool_call_id") or "",
                    name=result.get("name") or "",
                )
            )

        announced = {call.get("id") for call in assistant.tool_calls}
        answered = {result.tool_call_id for result in results}
        if announced - answered:
            logger.info(
                f"Dropping an incomplete tool-call block from rehydrated history: "
                f"{len(announced - answered)} of {len(announced)} calls unanswered"
            )
            continue

        messages.append(assistant)
        messages.extend(results)

    return messages
