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

"""Tests for how GET /api/images/{reference} resolves a depiction.

Depictions used to be PNG files on a ReadWriteMany volume shared between the
rdkit and api pods. They are now `png` data artifacts, which is what makes them
readable from any api replica -- rdkit sits behind an HPA, so a file written by
one rdkit pod was never guaranteed to be visible to the pod that served it.

The endpoint takes a bare reference, because model-authored markdown embeds
``/api/images/png_<hex>``. It looks up the artifact's owning session and requires
the caller's owner token to own it (see test_session_ownership.py). It also
refuses any non-`png` reference, so it cannot become a generic artifact reader.

These exercise ``normalize_png_reference`` and the storage round-trip rather than
the route function, because ``api.main`` cannot be imported under pytest --
``api/redis.py`` shadows the ``redis`` package once pytest puts ``backend/api``
on ``sys.path``.
"""

import pytest

from api.storage.datatypes import DataType, content_type_for
from api.storage.images import normalize_png_reference
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.service import ArtifactNotFoundError, DataStorageService

# A one-pixel PNG. Real bytes rather than a b"..." placeholder so the round-trip
# is checked against something a browser would actually render.
PNG_BYTES = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c63000100000500010d0a2db40000000049454e44ae"
    "426082"
)


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


class TestNormalizePngReference:
    def test_a_png_reference_passes_through(self):
        assert normalize_png_reference("png_a1b2c3d4") == "png_a1b2c3d4"

    def test_a_trailing_extension_is_stripped(self):
        """Model-authored markdown tends to write /api/images/png_a1b2c3d4.png."""
        assert normalize_png_reference("png_a1b2c3d4.png") == "png_a1b2c3d4"

    @pytest.mark.parametrize(
        "segment",
        [
            "sdf_a1b2c3d4",  # a real artifact, but not an image
            "pdb_a1b2c3d4",
            "no-underscore-at-all",
            "../../etc/passwd",
            "png_",  # prefix with no id
            "png",
            "",
        ],
    )
    def test_anything_else_is_rejected(self, segment):
        assert normalize_png_reference(segment) is None


@pytest.mark.asyncio
class TestStoredDepictions:
    async def test_bytes_round_trip_unchanged(self, storage):
        reference = await storage.store("png", PNG_BYTES)

        assert normalize_png_reference(reference) == reference
        assert await storage.retrieve(reference) == PNG_BYTES

    async def test_the_datatype_member_yields_the_same_reference_as_its_value(
        self, storage
    ):
        """`store(DataType.PNG, ...)` must not produce `DataType.PNG_<hex>`.

        Every caller today passes a bare string, so nothing would have caught
        this. It matters here because a member is the obvious thing for a future
        `smiles_to_image` tool to reach for, and an f-string formats a str enum
        member by its repr from Python 3.11 on -- the reference would store fine
        and then 404 forever, since the endpoint matches on the `png_` prefix.
        """
        reference = await storage.store(DataType.PNG, PNG_BYTES)

        assert reference.startswith("png_")
        assert normalize_png_reference(reference) == reference
        assert await storage.retrieve(reference) == PNG_BYTES

    async def test_the_served_media_type_is_png(self):
        assert content_type_for("png") == "image/png"

    async def test_an_unknown_reference_raises_not_found(self, storage):
        """What the endpoint turns into a 404."""
        with pytest.raises(ArtifactNotFoundError):
            await storage.retrieve("png_deadbeef")
