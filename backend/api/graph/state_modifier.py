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

"""Prompt state formatters for the LangGraph agent.

Formatting is delegated to :mod:`api.state.format` so production and training
share identical state envelopes; this module re-exports it for the graph and
its subgraphs.

Assembling the message list around that envelope now belongs to
:class:`api.context.ContextAssembler`, which does the same job under a token
budget.
"""

import logging

from api.state.format import (
    format_selections,
    format_storage_state,
    format_viewer_state,
    inject_state_into_prompt,
)

logger = logging.getLogger(__name__)


__all__ = [
    "format_selections",
    "format_storage_state",
    "format_viewer_state",
    "inject_state_into_prompt",
]
