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

"""Configuration for the user feedback email relay.

The single read site for every feedback env var, so ``api.main`` (which reports
the feature flag through ``/api/info``) and the route handler cannot disagree
about whether the feature is on.

``FEEDBACK_ENABLED`` is read at import time, matching
``STORAGE_ACCESS_API_ENABLED`` in ``api.main``. The SMTP settings are read per
request instead: they are deployment wiring rather than a shape the app is built
around, and reading them late means a relay can be pointed somewhere else
without a code change to how it is cached.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

FEEDBACK_ENABLED = os.getenv("AEVOM_FEAT_FEEDBACK_EMAIL", "false").lower() in (
    "true",
    "1",
)

MIN_MESSAGE_CHARS = 30
MAX_MESSAGE_CHARS = 5000
SUBJECT_PREVIEW_CHARS = 20

DEFAULT_SMTP_PORT = 587
SEND_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class SmtpConfig:
    """Everything needed to hand one message to a relay.

    ``sender`` and ``recipient`` are separate because authenticated relays
    require the From address to belong to the authenticating account, while the
    address feedback should *arrive* at is usually a distribution list that
    nobody authenticates as.
    """

    sender: str
    recipient: str
    host: str
    port: int
    start_tls: bool
    username: str | None
    password: str | None


def _password() -> str | None:
    """Read the relay password from a mounted file, or fall back to env.

    The file form matches how ``NVIDIA_API_KEY_FILE`` is handled in
    ``api.llm.client``, and is what a k8s Secret volume gives you. The plain env
    var is the form an ``optional: true`` ``secretKeyRef`` produces, and is what
    Compose users will reach for.
    """
    path = os.environ.get("SMTP_PASSWORD_FILE", "").strip()
    if path:
        try:
            return Path(path).read_text().strip() or None
        except OSError as exc:
            # Not fatal: many relays accept unauthenticated mail from inside the
            # cluster. Authentication will fail loudly at send time if it was
            # actually required, and that error is the useful one.
            logger.warning("Could not read SMTP_PASSWORD_FILE %s: %s", path, exc)
            return None
    return os.environ.get("SMTP_PASSWORD", "").strip() or None


def load_smtp_config() -> SmtpConfig | None:
    """Build the relay config from the environment.

    Returns ``None`` when the two vars with no sensible default --
    ``FEEDBACK_EMAIL_ADDRESS`` and ``SMTP_HOST`` -- are unset, which is how a
    deployment with the flag flipped on but no relay configured turns into a
    clear 503 rather than a connection error to ``localhost:587``.

    ``FEEDBACK_FROM_ADDRESS`` is optional and falls back to the recipient. A
    relay that accepts unauthenticated mail does not care who the sender claims
    to be, so making it required would be ceremony for those deployments.
    """
    recipient = os.environ.get("FEEDBACK_EMAIL_ADDRESS", "").strip()
    host = os.environ.get("SMTP_HOST", "").strip()
    if not recipient or not host:
        return None

    sender = os.environ.get("FEEDBACK_FROM_ADDRESS", "").strip() or recipient

    raw_port = os.environ.get("SMTP_PORT", "").strip()
    try:
        port = int(raw_port) if raw_port else DEFAULT_SMTP_PORT
    except ValueError:
        logger.warning("Ignoring non-numeric SMTP_PORT %r", raw_port)
        port = DEFAULT_SMTP_PORT

    return SmtpConfig(
        sender=sender,
        recipient=recipient,
        host=host,
        port=port,
        start_tls=os.environ.get("SMTP_STARTTLS", "true").lower() in ("true", "1"),
        username=os.environ.get("SMTP_USERNAME", "").strip() or None,
        password=_password(),
    )
