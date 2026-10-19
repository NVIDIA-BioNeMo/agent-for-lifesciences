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

"""Tests for the free-text feedback email relay.

Handlers are exercised directly rather than through a TestClient, because
``api.main`` cannot be imported under pytest -- ``api/redis.py`` shadows the
``redis`` package once pytest puts ``backend/api`` on ``sys.path``. Routing
itself is FastAPI's problem; what matters here is the flag gate, the length
bounds, and that nothing a user types can escape into an email header.

Skips wholesale if ``aiosmtplib`` is missing, which means the devcontainer image
predates it being added to ``backend/requirements.txt``. Rebuild the container.
"""

import pytest

pytest.importorskip("aiosmtplib")

import aiosmtplib  # noqa: E402
from fastapi import HTTPException  # noqa: E402

from api.feedback import mailer, routes  # noqa: E402
from api.feedback.config import (  # noqa: E402
    MAX_MESSAGE_CHARS,
    MIN_MESSAGE_CHARS,
    load_smtp_config,
)

SMTP_ENV = {
    # The destination is a distribution list; the sender is the service account
    # the relay authenticates, which is a different address.
    "FEEDBACK_EMAIL_ADDRESS": "aevom-feedback-dl@example.com",
    "FEEDBACK_FROM_ADDRESS": "svc-aevom@example.com",
    "SMTP_HOST": "smtp.example.com",
}


@pytest.fixture
def enabled(monkeypatch):
    """Flag on, relay configured, and every send captured instead of sent."""
    monkeypatch.setattr(routes, "FEEDBACK_ENABLED", True)
    for key, value in SMTP_ENV.items():
        monkeypatch.setenv(key, value)
    for key in ("SMTP_PORT", "SMTP_STARTTLS", "SMTP_USERNAME", "SMTP_PASSWORD",
                "SMTP_PASSWORD_FILE"):
        monkeypatch.delenv(key, raising=False)

    sent = []

    async def fake_send(message, **kwargs):
        sent.append((message, kwargs))

    monkeypatch.setattr(mailer.aiosmtplib, "send", fake_send)
    return sent


async def submit(message: str):
    return await routes.submit_feedback_email(
        routes.FeedbackEmailRequest(message=message)
    )


class TestSubjectLine:
    def test_prefix_and_first_twenty_characters(self):
        subject = mailer.build_subject("Docking results look wrong for 1ABC")
        assert subject == "[aevom Feedback]: Docking results look"
        assert len(subject.removeprefix("[aevom Feedback]: ")) == 20

    def test_short_message_is_not_padded(self):
        assert mailer.build_subject("a" * 5) == "[aevom Feedback]: aaaaa"

    def test_crlf_cannot_reach_the_header(self):
        """A header value containing CRLF is an email header injection.

        The injected text may still appear as characters -- it is a subject
        preview, after all -- but with no line break it cannot start a header.
        """
        subject = mailer.build_subject("hi\r\nBcc: attacker@evil.test\r\n")
        assert "\r" not in subject and "\n" not in subject
        assert subject.splitlines() == [subject]

    def test_leading_whitespace_does_not_eat_the_preview(self):
        subject = mailer.build_subject("\n\n   Vina scores are inverted")
        assert subject == "[aevom Feedback]: Vina scores are inve"

    def test_injection_survives_the_full_build(self):
        """The composed message must not gain headers the user asked for."""
        config = mailer.SmtpConfig(
            sender="from@example.com",
            recipient="to@example.com",
            host="smtp.example.com",
            port=587,
            start_tls=True,
            username=None,
            password=None,
        )
        email = mailer.build_message("x\r\nBcc: attacker@evil.test", config)
        assert email["Bcc"] is None
        assert email["To"] == "to@example.com"


class TestLengthBounds:
    @pytest.mark.asyncio
    async def test_one_short_of_the_minimum_is_rejected(self, enabled):
        with pytest.raises(HTTPException) as exc:
            await submit("a" * (MIN_MESSAGE_CHARS - 1))
        assert exc.value.status_code == 400
        assert str(MIN_MESSAGE_CHARS) in exc.value.detail
        assert not enabled

    @pytest.mark.asyncio
    async def test_exactly_the_minimum_is_accepted(self, enabled):
        assert await submit("a" * MIN_MESSAGE_CHARS) == {"status": "sent"}
        assert len(enabled) == 1

    @pytest.mark.asyncio
    async def test_exactly_the_maximum_is_accepted(self, enabled):
        assert await submit("a" * MAX_MESSAGE_CHARS) == {"status": "sent"}
        assert len(enabled) == 1

    @pytest.mark.asyncio
    async def test_one_past_the_maximum_is_rejected(self, enabled):
        with pytest.raises(HTTPException) as exc:
            await submit("a" * (MAX_MESSAGE_CHARS + 1))
        assert exc.value.status_code == 400
        assert str(MAX_MESSAGE_CHARS) in exc.value.detail
        assert not enabled

    @pytest.mark.asyncio
    async def test_padding_does_not_buy_length(self, enabled):
        """Whitespace is stripped before measuring, so spaces are not content."""
        with pytest.raises(HTTPException) as exc:
            await submit("short" + " " * 100)
        assert exc.value.status_code == 400
        assert not enabled


class TestFlagGate:
    @pytest.mark.asyncio
    async def test_disabled_looks_like_it_does_not_exist(self, monkeypatch):
        monkeypatch.setattr(routes, "FEEDBACK_ENABLED", False)
        with pytest.raises(HTTPException) as exc:
            await submit("a" * MIN_MESSAGE_CHARS)
        assert exc.value.status_code == 404

    @pytest.mark.asyncio
    async def test_the_gate_precedes_validation(self, monkeypatch):
        """A disabled feature must not leak its rules through error messages."""
        monkeypatch.setattr(routes, "FEEDBACK_ENABLED", False)
        with pytest.raises(HTTPException) as exc:
            await submit("too short")
        assert exc.value.status_code == 404


class TestConfiguration:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("missing", ["FEEDBACK_EMAIL_ADDRESS", "SMTP_HOST"])
    async def test_unconfigured_relay_is_unavailable_not_broken(
        self, enabled, monkeypatch, missing
    ):
        monkeypatch.delenv(missing)
        with pytest.raises(HTTPException) as exc:
            await submit("a" * MIN_MESSAGE_CHARS)
        assert exc.value.status_code == 503
        assert not enabled

    def test_defaults_when_only_the_required_vars_are_set(self, enabled):
        config = load_smtp_config()
        assert config is not None
        assert config.port == 587
        assert config.start_tls is True
        assert config.username is None
        assert config.password is None

    def test_sender_falls_back_to_the_recipient(self, enabled, monkeypatch):
        """Unauthenticated relays do not care who the sender claims to be."""
        monkeypatch.delenv("FEEDBACK_FROM_ADDRESS")
        config = load_smtp_config()
        assert config.sender == SMTP_ENV["FEEDBACK_EMAIL_ADDRESS"]
        assert config.recipient == SMTP_ENV["FEEDBACK_EMAIL_ADDRESS"]

    def test_an_empty_sender_is_treated_as_unset(self, enabled, monkeypatch):
        """A blank ConfigMap value must behave like an absent one."""
        monkeypatch.setenv("FEEDBACK_FROM_ADDRESS", "   ")
        assert load_smtp_config().sender == SMTP_ENV["FEEDBACK_EMAIL_ADDRESS"]

    def test_a_sender_alone_does_not_make_the_relay_usable(
        self, enabled, monkeypatch
    ):
        """FEEDBACK_FROM_ADDRESS is not a substitute for a destination."""
        monkeypatch.delenv("FEEDBACK_EMAIL_ADDRESS")
        assert load_smtp_config() is None

    def test_non_numeric_port_falls_back_rather_than_raising(
        self, enabled, monkeypatch
    ):
        monkeypatch.setenv("SMTP_PORT", "not-a-port")
        assert load_smtp_config().port == 587

    def test_password_is_read_from_the_mounted_file(
        self, enabled, monkeypatch, tmp_path
    ):
        secret = tmp_path / "smtp_password"
        secret.write_text("  relay-secret\n")
        monkeypatch.setenv("SMTP_PASSWORD_FILE", str(secret))
        assert load_smtp_config().password == "relay-secret"

    def test_missing_password_file_does_not_break_the_config(
        self, enabled, monkeypatch, tmp_path
    ):
        """Many relays accept unauthenticated mail; let the send fail instead."""
        monkeypatch.setenv("SMTP_PASSWORD_FILE", str(tmp_path / "absent"))
        assert load_smtp_config().password is None


class TestRelay:
    @pytest.mark.asyncio
    async def test_the_body_is_the_message_verbatim(self, enabled):
        message = "Line one.\n\nLine two has  double  spaces and a tab\there."
        await submit(message + "x" * MIN_MESSAGE_CHARS)

        email, _ = enabled[0]
        assert email.get_content().rstrip("\n") == message + "x" * MIN_MESSAGE_CHARS

    @pytest.mark.asyncio
    async def test_the_sender_is_the_service_account_not_the_destination(
        self, enabled
    ):
        """An authenticated relay rejects a From it did not authenticate."""
        await submit("a" * MIN_MESSAGE_CHARS)
        email, kwargs = enabled[0]
        assert email["From"] == SMTP_ENV["FEEDBACK_FROM_ADDRESS"]
        assert email["To"] == SMTP_ENV["FEEDBACK_EMAIL_ADDRESS"]
        assert email["From"] != email["To"]
        assert email["Subject"].startswith("[aevom Feedback]: ")
        assert kwargs["hostname"] == SMTP_ENV["SMTP_HOST"]
        assert kwargs["port"] == 587

    @pytest.mark.asyncio
    async def test_the_destination_list_is_the_only_recipient(self, enabled):
        """No Cc or Bcc, so the sender mailbox does not accumulate copies."""
        await submit("a" * MIN_MESSAGE_CHARS)
        email, _ = enabled[0]
        assert email["Cc"] is None
        assert email["Bcc"] is None
        assert email.get_all("To") == [SMTP_ENV["FEEDBACK_EMAIL_ADDRESS"]]

    @pytest.mark.asyncio
    async def test_send_is_bounded_by_a_timeout(self, enabled):
        """An unreachable relay must not hold the request open indefinitely."""
        await submit("a" * MIN_MESSAGE_CHARS)
        _, kwargs = enabled[0]
        assert kwargs["timeout"] == pytest.approx(15.0)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "error",
        [
            aiosmtplib.SMTPException("relay refused"),
            ConnectionRefusedError("nothing listening"),
            TimeoutError("relay never answered"),
        ],
    )
    async def test_relay_failure_becomes_a_bad_gateway(
        self, enabled, monkeypatch, error
    ):
        async def boom(message, **kwargs):
            raise error

        monkeypatch.setattr(mailer.aiosmtplib, "send", boom)
        with pytest.raises(HTTPException) as exc:
            await submit("a" * MIN_MESSAGE_CHARS)
        assert exc.value.status_code == 502

    @pytest.mark.asyncio
    async def test_relay_error_text_is_not_returned_to_the_caller(
        self, enabled, monkeypatch
    ):
        """It can name internal hosts and credentials; it belongs in the log."""

        async def boom(message, **kwargs):
            raise aiosmtplib.SMTPException(
                "535 auth failed for user svc-aevom on relay.internal"
            )

        monkeypatch.setattr(mailer.aiosmtplib, "send", boom)
        with pytest.raises(HTTPException) as exc:
            await submit("a" * MIN_MESSAGE_CHARS)
        assert "relay.internal" not in exc.value.detail
        assert "svc-aevom" not in exc.value.detail
