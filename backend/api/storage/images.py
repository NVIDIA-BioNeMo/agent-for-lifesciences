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

"""URL-segment handling for stored molecule depictions.

Depictions are `png` data artifacts served by ``GET /api/images/{reference}``.
That endpoint takes a bare reference and checks the caller owns the artifact's
session. The reference is also restricted to `png` here before it reaches
storage, so the endpoint cannot become a generic artifact reader.

The check lives here rather than inline in ``api.main`` so it is unit-testable:
``api.main`` cannot be imported under pytest, because ``api/redis.py`` shadows
the ``redis`` package once pytest puts ``backend/api`` on ``sys.path``.
"""

from __future__ import annotations

from typing import Optional

from .datatypes import DataType

_PREFIX = f"{DataType.PNG.value}_"
_EXTENSION = ".png"


def normalize_png_reference(segment: str) -> Optional[str]:
    """Resolve a URL path segment to a `png` artifact reference.

    Args:
        segment: The raw ``{reference}`` path segment. A trailing ``.png`` is
            accepted because model-authored markdown tends to add the extension
            (``![mol](/api/images/png_a1b2c3d4.png)``).

    Returns:
        The artifact reference, or None when the segment does not name a `png`
        artifact -- in which case the caller must 404 rather than look it up.
    """
    if segment.endswith(_EXTENSION):
        segment = segment[: -len(_EXTENSION)]

    if not segment.startswith(_PREFIX) or len(segment) == len(_PREFIX):
        return None

    return segment
