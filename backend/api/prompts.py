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

"""System prompts for Aevom AI scientist agent.

This module loads the system prompt from prompts.md in the project root.
This ensures a single source of truth that is both human-readable (markdown)
and usable by the Python backend.

In development mode (AEVOM_DEV_MODE=true), the prompt is reloaded from disk
on each request if the file has changed, enabling rapid prompt iteration.
"""

import os
from pathlib import Path

# Container path (volume-mounted from project root)
_prompts_file = Path("/prompts.md")
_cached_prompt: str | None = None
_cached_mtime: float = 0


def get_system_prompt() -> str:
    """Get system prompt, reloading from disk in dev mode if file changed."""
    global _cached_prompt, _cached_mtime

    dev_mode = os.getenv("AEVOM_DEV_MODE", "").lower() == "true"

    if dev_mode:
        # Check if file was modified
        current_mtime = _prompts_file.stat().st_mtime
        if current_mtime != _cached_mtime:
            _cached_prompt = _prompts_file.read_text()
            _cached_mtime = current_mtime
    elif _cached_prompt is None:
        # Production: load once
        _cached_prompt = _prompts_file.read_text()

    return _cached_prompt


# Backwards compatibility (production behavior - loaded at import time)
system_prompt = _prompts_file.read_text()
