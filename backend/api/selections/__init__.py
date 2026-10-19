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

"""Selection registry for tracking available selections across conversation turns.

This module provides a Redis-backed registry for tracking selections created by
structure_query so they can be accessed by subsequent tools like python_execute.

All public functions are async and require a session_id parameter.
"""

from .registry import (
    PROVENANCE_AGENT,
    PROVENANCE_USER,
    SelectionInfo,
    register_selection,
    get_available_selections,
    get_latest_selection,
    get_role_layers,
    active_role_layers,
    get_latest_nonempty_selection,
    get_selection_by_query_id,
    clear_selections,
    get_selection_state_condensed,
)
from .residues import (
    ResidueRef,
    format_residue_listing,
    residues_from_atoms,
    residues_from_selection,
)
from .format import (
    format_selection_success,
    format_selection_empty,
    format_selection_empty_with_hints,
    format_selection_timeout,
    format_selection_no_data,
    format_selection_query_error,
    is_molql_error_summary,
    molql_error_summary,
    MOLQL_ERROR_PREFIX,
)

__all__ = [
    "get_role_layers",
    "active_role_layers",
    "PROVENANCE_AGENT",
    "PROVENANCE_USER",
    "ResidueRef",
    "format_residue_listing",
    "residues_from_atoms",
    "residues_from_selection",
    "SelectionInfo",
    "register_selection",
    "get_available_selections",
    "get_latest_selection",
    "get_latest_nonempty_selection",
    "get_selection_by_query_id",
    "clear_selections",
    "get_selection_state_condensed",
    "format_selection_success",
    "format_selection_empty",
    "format_selection_empty_with_hints",
    "format_selection_timeout",
    "format_selection_no_data",
    "format_selection_query_error",
    "is_molql_error_summary",
    "molql_error_summary",
    "MOLQL_ERROR_PREFIX",
]
