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

"""Session-scoped API-key secrets for the Aevom agent.

Users add API keys (e.g. for NVIDIA-hosted NIMs) that skills need. Secrets are
stored per browser session in Redis (``secret:{session_id}``) and referenced by
name. Skill endpoint tools (and the built-in ``http_request`` tool, when
``AEVOM_FEAT_HTTP_REQUEST_TOOL`` enables it) resolve a secret by name
server-side and inject it into the outgoing request; the raw value never
reaches the model or the pyexec sandbox.
"""

from .routes import router as secrets_router
from .session_store import (
    DEFAULT_SESSION_TTL,
    SecretMeta,
    SecretSessionStore,
    get_secret_session_store,
)

__all__ = [
    "DEFAULT_SESSION_TTL",
    "SecretMeta",
    "SecretSessionStore",
    "get_secret_session_store",
    "secrets_router",
]
