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

"""Context assembly and budgeting.

The conversation record and the model's context are different things: the
record is everything that happened, the context is a view of it rendered to
fit a token budget. This package owns the second half of that -- measuring
what a prompt costs and, in later stages, deciding what to leave out.

``budget`` measures what a prompt costs. ``assembler`` decides what to leave
out when that cost exceeds the model's window, degrading older content through
four levels rather than dropping it. ``rehydrate`` turns stored records back
into messages a provider will accept.
"""

from .assembler import (
    AssemblyReport,
    ContextAssembler,
    build_context_assembler,
    record_assembly_on_span,
    split_into_turns,
)
from .budget import (
    CONTEXT_SAFETY_MARGIN_TOKENS,
    BudgetReport,
    check_context_budget,
    estimate_messages_tokens,
    estimate_tokens,
    record_budget_on_span,
    record_estimate_error,
)
from .rehydrate import rehydrate_messages
from .responses import DegradeReport, degrade_responses_input

__all__ = [
    "CONTEXT_SAFETY_MARGIN_TOKENS",
    "AssemblyReport",
    "BudgetReport",
    "ContextAssembler",
    "DegradeReport",
    "build_context_assembler",
    "degrade_responses_input",
    "record_assembly_on_span",
    "split_into_turns",
    "check_context_budget",
    "estimate_messages_tokens",
    "estimate_tokens",
    "record_budget_on_span",
    "record_estimate_error",
    "rehydrate_messages",
]
