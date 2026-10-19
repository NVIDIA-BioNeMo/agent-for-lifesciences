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

"""Minting a session: the body of ``POST /api/session/create``."""

from __future__ import annotations

import uuid

from fastapi import Request, Response

from .ownership import SessionOwnership, owner_ttl_seconds
from .principal import mint_owner_token, owner_token_from, set_owner_cookie


async def open_owned_session(
    request: Request, response: Response, ownership: SessionOwnership
) -> dict[str, str]:
    """Create a session owned by the caller and return its id and owner token.

    A caller that already holds an owner token (cookie or Bearer) keeps it, so
    one browser's sessions share an owner; otherwise a token is minted. The
    token is set as an HttpOnly cookie for the browser and returned for
    scripts, which send it back as ``Authorization: Bearer``.
    """
    token = owner_token_from(request) or mint_owner_token()
    session_id = str(uuid.uuid4())
    await ownership.bind(session_id, token)
    set_owner_cookie(response, token, owner_ttl_seconds())
    return {"session_id": session_id, "owner_token": token}
