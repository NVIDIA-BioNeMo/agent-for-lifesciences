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

"""The system-prompt block that names a session's stored API keys."""

from api.tools.http_common import http_request_tool_enabled


def stored_keys_block(names: list[str]) -> str:
    """Name the stored keys (never their values) so the first call can use one.

    Without the names the model only learns auth is needed after an upstream
    401. Returns an empty string when nothing is stored.
    """
    if not names:
        return ""
    tool = (
        "call http_request or the skill's tool"
        if http_request_tool_enabled()
        else "call the skill's tool"
    )
    return (
        f"[STORED API KEYS: {len(names)}]\n"
        f"Available key names: {', '.join(names)}.\n"
        f"When an API a skill uses needs a key, {tool} with "
        "`secret_name` set to one of these EXACT names on the first "
        "attempt (do not put the key or a placeholder in headers)."
    )
