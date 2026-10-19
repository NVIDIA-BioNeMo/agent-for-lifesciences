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

"""Length-prefixed JSON frames over pipes, with deadlines that bound partial I/O.

Shared by the Vina service (parent) and the GPU conformer worker. Each side
loads this file by path, so it must stay standard-library only and must not
import anything from the Vina application.

A frame is a 4-byte big-endian payload length followed by that many bytes of
UTF-8 JSON encoding one object. File descriptors passed here must be in
non-blocking mode: every read and write is driven by ``poll`` against an
absolute ``time.monotonic()`` deadline, so a peer that sends half a frame or
stops draining its pipe cannot block the caller past that deadline.
"""

from __future__ import annotations

import json
import os
import select
import struct
import time
from typing import Any, Optional

PROTOCOL_VERSION = 1

_HEADER = struct.Struct(">I")


class FrameError(Exception):
    """The byte stream does not hold a valid frame."""


class FrameTimeout(FrameError):
    """The deadline passed before the frame was fully read or written."""


class FrameEOF(FrameError):
    """The peer closed its end in the middle of a frame, or before a write."""


class FrameTooLarge(FrameError):
    """A frame exceeds the configured size limit."""


def deadline_after(seconds: Optional[float]) -> Optional[float]:
    """Absolute monotonic deadline *seconds* from now, or None for no deadline."""
    return None if seconds is None else time.monotonic() + seconds


def _wait(fd: int, events: int, deadline: Optional[float], what: str) -> None:
    if deadline is None:
        timeout_ms = None
    else:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise FrameTimeout(f"timed out {what}")
        timeout_ms = max(1, int(remaining * 1000))
    poller = select.poll()
    poller.register(fd, events | select.POLLERR | select.POLLHUP)
    if not poller.poll(timeout_ms):
        raise FrameTimeout(f"timed out {what}")


def encode_frame(obj: dict[str, Any], max_bytes: int) -> bytes:
    try:
        payload = json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise FrameError(f"cannot encode frame: {exc}") from exc
    if len(payload) > max_bytes:
        raise FrameTooLarge(f"frame of {len(payload)} bytes exceeds limit of {max_bytes}")
    return _HEADER.pack(len(payload)) + payload


def write_frame(fd: int, obj: dict[str, Any], *, max_bytes: int, deadline: Optional[float]) -> None:
    """Write one frame, finishing before *deadline* or raising FrameTimeout."""
    view = memoryview(encode_frame(obj, max_bytes))
    while view:
        try:
            written = os.write(fd, view)
        except BlockingIOError:
            written = 0
        except BrokenPipeError as exc:
            raise FrameEOF("peer closed the pipe during write") from exc
        view = view[written:]
        if view:
            _wait(fd, select.POLLOUT, deadline, "writing frame")


def _read_exact(fd: int, n: int, deadline: Optional[float], allow_clean_eof: bool) -> Optional[bytes]:
    chunks: list[bytes] = []
    received = 0
    while received < n:
        try:
            chunk = os.read(fd, n - received)
        except BlockingIOError:
            _wait(fd, select.POLLIN, deadline, "reading frame")
            continue
        if not chunk:
            if allow_clean_eof and received == 0:
                return None
            raise FrameEOF(f"peer closed the pipe after {received} of {n} bytes")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def _reject_constant(name: str) -> float:
    raise FrameError(f"non-finite number {name} in frame")


def read_frame(
    fd: int,
    *,
    max_bytes: int,
    deadline: Optional[float],
    frame_timeout: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Read one frame.

    *deadline* bounds the whole read, including the wait for its first byte.
    *frame_timeout*, if given, additionally bounds the time from the first
    byte to the last, so a reader that waits indefinitely for the next request
    still cannot be stalled by a frame that never finishes.

    Returns None on a clean EOF before any byte of a new frame.
    """
    first = _read_exact(fd, 1, deadline, allow_clean_eof=True)
    if first is None:
        return None
    if frame_timeout is not None:
        frame_deadline = time.monotonic() + frame_timeout
        deadline = frame_deadline if deadline is None else min(deadline, frame_deadline)

    rest = _read_exact(fd, _HEADER.size - 1, deadline, allow_clean_eof=False)
    (length,) = _HEADER.unpack(first + rest)
    if length > max_bytes:
        raise FrameTooLarge(f"peer announced a {length}-byte frame; limit is {max_bytes}")

    payload = _read_exact(fd, length, deadline, allow_clean_eof=False)
    try:
        obj = json.loads(payload.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrameError(f"frame is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise FrameError(f"frame holds {type(obj).__name__}, expected an object")
    return obj
