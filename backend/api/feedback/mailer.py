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

"""Compose and relay a user feedback email.

Split from the route handler so subject construction -- the part with a real
correctness constraint -- is testable without a relay or an event loop.
"""

from __future__ import annotations

import logging
from email.message import EmailMessage

import aiosmtplib

from .config import SEND_TIMEOUT_SECONDS, SUBJECT_PREVIEW_CHARS, SmtpConfig

logger = logging.getLogger(__name__)

SUBJECT_PREFIX = "[aevom Feedback]: "


def build_subject(message: str) -> str:
    """``"[aevom Feedback]: "`` plus the first 20 characters of the message.

    Whitespace is collapsed before slicing. That is not cosmetic: a bare CRLF in
    a header value is an email header injection, so feedback beginning with
    ``"x\\r\\nBcc: someone@elsewhere"`` must not be able to reach the header
    unaltered. Collapsing also means a message that opens with a newline yields
    a subject preview of real words rather than of the indentation.
    """
    flattened = " ".join(message.split())
    return f"{SUBJECT_PREFIX}{flattened[:SUBJECT_PREVIEW_CHARS]}"


def build_message(message: str, config: SmtpConfig) -> EmailMessage:
    """Build the outgoing email. The body is the feedback text, verbatim.

    The From address has to be one the relay will accept from the authenticating
    account, which is rarely the address feedback should land at -- so it comes
    from ``FEEDBACK_FROM_ADDRESS`` and the To from ``FEEDBACK_EMAIL_ADDRESS``.
    """
    email = EmailMessage()
    email["From"] = config.sender
    email["To"] = config.recipient
    email["Subject"] = build_subject(message)
    email.set_content(message)
    return email


async def send_feedback(message: str, config: SmtpConfig) -> None:
    """Relay one feedback message. Raises on any SMTP or network failure."""
    await aiosmtplib.send(
        build_message(message, config),
        hostname=config.host,
        port=config.port,
        start_tls=config.start_tls,
        username=config.username,
        password=config.password,
        timeout=SEND_TIMEOUT_SECONDS,
    )
    logger.info(
        "Relayed %d chars of user feedback from %s to %s via %s:%d",
        len(message),
        config.sender,
        config.recipient,
        config.host,
        config.port,
    )
