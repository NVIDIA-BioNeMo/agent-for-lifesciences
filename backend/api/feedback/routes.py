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

"""API route for submitting free-text user feedback by email.

Mounted at ``/api/feedback/email``, deliberately distinct from ``main.py``'s
``POST /api/feedback``, which does something unrelated: it writes thumbs-up/down
Phoenix span annotations against a trace.
"""

from __future__ import annotations

import logging

import aiosmtplib
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .config import (
    FEEDBACK_ENABLED,
    MAX_MESSAGE_CHARS,
    MIN_MESSAGE_CHARS,
    load_smtp_config,
)
from .mailer import send_feedback

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feedback", tags=["feedback"])


class FeedbackEmailRequest(BaseModel):
    """Free-text feedback from the UI's feedback modal."""

    message: str


@router.post("/email")
async def submit_feedback_email(request: FeedbackEmailRequest):
    """Relay one free-text feedback message to the configured inbox."""
    if not FEEDBACK_ENABLED:
        raise HTTPException(status_code=404, detail="Feedback is disabled")

    # The modal enforces the same bounds, but a browser is not a validator.
    message = request.message.strip()
    if len(message) < MIN_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Feedback must be at least {MIN_MESSAGE_CHARS} characters",
        )
    if len(message) > MAX_MESSAGE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Feedback must be at most {MAX_MESSAGE_CHARS} characters",
        )

    config = load_smtp_config()
    if config is None:
        logger.error(
            "Feedback is enabled but FEEDBACK_EMAIL_ADDRESS or SMTP_HOST is unset"
        )
        raise HTTPException(
            status_code=503, detail="Feedback email is not configured"
        )

    try:
        await send_feedback(message, config)
    except (aiosmtplib.SMTPException, OSError, TimeoutError) as exc:
        # The relay's own error text can name internal hosts and credentials, so
        # it goes to the log rather than to the browser.
        logger.exception("Failed to relay feedback email: %s", exc)
        raise HTTPException(
            status_code=502, detail="Could not send feedback email"
        )

    return {"status": "sent"}
