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

"""Text-based tool call parsing and sanitization.

Some LLM models (especially smaller ones) output tool calls as XML-style
text instead of using the native function calling API. This module provides
utilities to detect, extract, and sanitize such text-based tool calls.

Example formats handled:
    <tool_call><function=vina_dock><parameter=center_x>-9.02</parameter></function></tool_call>

Usage:
    from api.llm.text_parser import extract_tool_calls_from_text, sanitize_llm_content

    # Extract structured tool calls from raw LLM text
    tool_calls = extract_tool_calls_from_text(llm_content)

    # Remove tool call artifacts from content before displaying
    clean_content = sanitize_llm_content(llm_content)
"""

import json
import logging
import re
import uuid

logger = logging.getLogger(__name__)


# Regex patterns to detect and strip textual tool call artifacts
# Some models output tool calls as text instead of using the proper API
TOOL_CALL_TEXT_PATTERNS = [
    # <tool_call> ... </tool_call> format
    re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL),
    # <function=name> ... </function> format
    re.compile(r"<function=[^>]*>.*?</function>", re.DOTALL),
    # Incomplete variants at end of stream
    re.compile(r"<tool_call>.*$", re.DOTALL),
    re.compile(r"<function=[^>]*>.*$", re.DOTALL),
]

# Pattern to extract tool calls from XML-style text format
# Example: <tool_call><function=vina_dock><parameter=center_x>-9.02</parameter>...</function></tool_call>
TOOL_CALL_EXTRACT_PATTERN = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>", re.DOTALL
)
PARAMETER_EXTRACT_PATTERN = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>", re.DOTALL
)


def extract_tool_calls_from_text(content: str) -> list[dict]:
    """Extract tool calls from XML-style text format.

    Some models (especially smaller ones) output tool calls as text instead of
    using the native function calling API. This function parses those text-based
    tool calls and returns them in the standard format.

    Args:
        content: Raw content from the LLM that may contain XML tool calls.

    Returns:
        List of tool call dictionaries in the standard format, or empty list if none found.
    """
    if not content:
        return []

    tool_calls = []

    for match in TOOL_CALL_EXTRACT_PATTERN.finditer(content):
        function_name = match.group(1)
        params_text = match.group(2)

        # Extract parameters
        arguments = {}
        for param_match in PARAMETER_EXTRACT_PATTERN.finditer(params_text):
            param_name = param_match.group(1)
            param_value = param_match.group(2).strip()

            # Try to parse as JSON value (number, boolean, etc.)
            try:
                # Try as number first
                if "." in param_value:
                    arguments[param_name] = float(param_value)
                else:
                    arguments[param_name] = int(param_value)
            except ValueError:
                # Keep as string
                if param_value.lower() == "true":
                    arguments[param_name] = True
                elif param_value.lower() == "false":
                    arguments[param_name] = False
                else:
                    arguments[param_name] = param_value

        # Generate a unique ID for the tool call
        tool_call_id = f"text_call_{uuid.uuid4().hex[:8]}"

        tool_calls.append(
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": json.dumps(arguments),
                },
            }
        )

        logger.info(
            f"Extracted text-based tool call: {function_name} with args {arguments}"
        )

    return tool_calls


def sanitize_llm_content(content: str) -> str:
    """Remove textual tool call artifacts from LLM content.

    Some models (especially smaller ones) output tool calls in a textual format
    like <tool_call><function=name>...</function></tool_call> instead of using
    the proper function calling API. This function strips such artifacts.

    Args:
        content: Raw content from the LLM.

    Returns:
        Sanitized content with tool call text removed.
    """
    if not content:
        return content

    result = content
    was_modified = False
    for pattern in TOOL_CALL_TEXT_PATTERNS:
        new_result = pattern.sub("", result)
        if new_result != result:
            was_modified = True
            result = new_result

    # Only strip whitespace if we actually removed tool call artifacts
    # Don't strip normal content chunks as it breaks markdown paragraph structure
    if was_modified:
        result = result.strip()

    return result
