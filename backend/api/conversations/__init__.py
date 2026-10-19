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

"""The durable conversation record.

The backend, not the browser, owns what happened. This package holds the
read/write path for that record: transcripts (conversations, turns, messages)
and the findings ledger extracted from tool executions.

Context assembly, which renders a *view* of this record under a token budget,
lives in :mod:`api.context`. Keeping them apart is the whole point -- the
record can be as long as the session needs while the view stays bounded.
"""

from .candidates import (
    canonical,
    capture_from_result,
    get_candidates_source,
    merge_in_memory,
    rank,
    reset_candidates_source,
    set_candidates_source,
)
from .findings import (
    SCALAR_KEYS,
    extract_finding,
    format_session_log,
    get_findings_source,
    reset_findings_source,
    search_findings_in_memory,
    set_findings_source,
)
from .recorder import (
    TurnRecorder,
    get_recorder,
    reset_recorder,
    set_recorder,
)
from .store import ConversationStore

__all__ = [
    "SCALAR_KEYS",
    "ConversationStore",
    "TurnRecorder",
    "canonical",
    "capture_from_result",
    "extract_finding",
    "format_session_log",
    "get_candidates_source",
    "get_findings_source",
    "get_recorder",
    "merge_in_memory",
    "rank",
    "reset_candidates_source",
    "reset_findings_source",
    "reset_recorder",
    "search_findings_in_memory",
    "set_candidates_source",
    "set_findings_source",
    "set_recorder",
]
