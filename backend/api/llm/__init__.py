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

"""LLM client module for Aevom.

Provides configuration-driven LLM integration with OpenAI SDK.
"""

from .config import ModelConfig, get_model_config, get_active_model
from .client import LLMClient, StreamChunk, create_client, get_client
from .template_loader import (
    TemplateLoader,
    get_template_loader,
    render_template,
)
from .text_parser import (
    extract_tool_calls_from_text,
    sanitize_llm_content,
)

__all__ = [
    "ModelConfig",
    "get_model_config",
    "get_active_model",
    "LLMClient",
    "StreamChunk",
    "create_client",
    "get_client",
    "TemplateLoader",
    "get_template_loader",
    "render_template",
    "extract_tool_calls_from_text",
    "sanitize_llm_content",
]



