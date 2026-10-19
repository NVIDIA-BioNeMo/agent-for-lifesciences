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

"""Self-hosted NVIDIA NIM support.

A user can point Aevom at a NIM they run themselves. Two things come of that:
the NIM may contribute tools Aevom did not have (RFD4-Proteina contributes
binder design), and it may stand in for a hosted NVIDIA endpoint a built-in
tool already uses (a self-hosted Boltz-2 serves the ``boltz2`` tool).

Which of those applies is decided by an adapter keyed on the model id the NIM
reports, since a NIM announces a model and not a capability. See
``api.nim.adapters``.
"""

from .adapters import NIMAdapter, NIMEndpoint, all_adapters, get_adapter
from .client import NIMAuthError, NIMClient, NIMConnectionError, NIMInfo
from .overrides import EndpointOverride, get_endpoint_override, overridden_tools
from .routes import router as nim_router
from .session_store import NIMServerConfig, NIMSessionStore, get_nim_session_store

__all__ = [
    # Client
    "NIMClient",
    "NIMInfo",
    "NIMConnectionError",
    "NIMAuthError",
    # Adapters
    "NIMAdapter",
    "NIMEndpoint",
    "get_adapter",
    "all_adapters",
    # Endpoint overrides
    "EndpointOverride",
    "get_endpoint_override",
    "overridden_tools",
    # Session
    "NIMServerConfig",
    "NIMSessionStore",
    "get_nim_session_store",
    # Routes
    "nim_router",
]
