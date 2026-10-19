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

"""Who is calling: the owner token a request carries.

``POST /api/session/create`` mints a random owner token and binds each session
it creates to it. Browsers carry the token in an HttpOnly cookie; scripts and
eval harnesses send it as ``Authorization: Bearer``. A ``session_id`` names the
resource being acted on; the token proves the caller may act on it.
"""

from __future__ import annotations

import hashlib
import os
import re
import secrets
from typing import Optional

from fastapi import Request, Response

OWNER_COOKIE = "aevom_owner"

# token_urlsafe(32) yields 43 characters. Anything shorter was not minted here,
# and a guessable value a client chose for itself must not become an identity.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{43,128}$")


def mint_owner_token() -> str:
    """A new, unguessable owner token."""
    return secrets.token_urlsafe(32)


def is_well_formed(token: Optional[str]) -> bool:
    return bool(token) and bool(_TOKEN_RE.match(token))


def hash_token(token: str) -> str:
    """What is stored server-side, so a Redis dump does not yield usable tokens."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def owner_token_from(request: Request) -> Optional[str]:
    """The owner token on a request: a Bearer header first, then the cookie.

    Returns None when neither is present or well formed.
    """
    authorization = request.headers.get("authorization", "")
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() == "bearer":
        token = credentials.strip()
        return token if is_well_formed(token) else None
    token = request.cookies.get(OWNER_COOKIE)
    return token if is_well_formed(token) else None


def cookie_secure() -> bool:
    """Whether the owner cookie is marked Secure (read per-call for tests).

    On by default. Local stacks served over plain HTTP turn it off, or the
    browser would never send the cookie back.
    """
    return os.environ.get("AEVOM_COOKIE_SECURE", "true").lower() in ("true", "1")


def set_owner_cookie(response: Response, token: str, max_age: int) -> None:
    response.set_cookie(
        OWNER_COOKIE,
        token,
        max_age=max_age,
        httponly=True,
        secure=cookie_secure(),
        samesite="lax",
        path="/",
    )
