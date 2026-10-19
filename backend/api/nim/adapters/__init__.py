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

"""Registry mapping a NIM's model id to what Aevom can do with it.

Adding support for another NIM means adding a module here and listing it in
``_ADAPTERS``. Nothing else in the NIM package knows which models exist.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import NIMAdapter, NIMEndpoint
from .rfd4_proteina import ADAPTER as RFD4_PROTEINA_ADAPTER
from .boltz2 import ADAPTER as BOLTZ2_ADAPTER

logger = logging.getLogger(__name__)

# Every model Aevom recognises behind a self-hosted NIM.
_ADAPTERS: tuple[NIMAdapter, ...] = (
    RFD4_PROTEINA_ADAPTER,
    BOLTZ2_ADAPTER,
)


def get_adapter(model_id: str) -> Optional[NIMAdapter]:
    """The adapter claiming this model id, or None if no one does.

    None is a supported outcome, not a failure: an unrecognised NIM still
    registers, and is shown as reachable but without tools.
    """
    for adapter in _ADAPTERS:
        if adapter.matches(model_id):
            return adapter
    logger.info("No NIM adapter claims model id %r", model_id)
    return None


def all_adapters() -> tuple[NIMAdapter, ...]:
    """Every registered adapter, for introspection and tests."""
    return _ADAPTERS


__all__ = ["NIMAdapter", "NIMEndpoint", "get_adapter", "all_adapters"]
