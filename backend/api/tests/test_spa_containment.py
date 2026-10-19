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

"""The SPA catch-all serves files from the build directory and nowhere else.

It used to join the request path onto the static root unchecked, so ``..``
segments, an absolute path (``//etc/passwd`` -- pathlib drops the root when
joined with an absolute path) or a symlink could read any file the process
could. Requests are sent with raw targets so the client cannot normalise the
traversal away before it reaches the route.
"""

import httpx
import pytest
from fastapi import FastAPI

from api.spa import mount_spa

INDEX = b"<html>index</html>"
SECRET = b"outside the build"


@pytest.fixture
def layout(tmp_path):
    static = tmp_path / "static"
    (static / "assets").mkdir(parents=True)
    (static / "index.html").write_bytes(INDEX)
    (static / "favicon.ico").write_bytes(b"icon")
    (static / "assets" / "app.js").write_bytes(b"js")
    (tmp_path / "secret.txt").write_bytes(SECRET)
    (static / "escape").symlink_to(tmp_path / "secret.txt")
    return tmp_path


@pytest.fixture
def client(layout):
    app = FastAPI()

    @app.get("/api/real")
    async def real():
        return {"ok": True}

    mount_spa(app, layout / "static")
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def raw_get(client: httpx.AsyncClient, target: str) -> httpx.Response:
    request = client.build_request("GET", "http://test/")
    request.url = request.url.copy_with(raw_path=target.encode())
    return await client.send(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        "/../secret.txt",
        "/%2e%2e/secret.txt",
        "/assets/../../secret.txt",
        "/%2e%2e%2fsecret.txt",
        "//etc/passwd",
        "/escape",
    ],
)
async def test_paths_outside_the_build_get_the_index(client, layout, target):
    response = await raw_get(client, target)
    assert SECRET not in response.content
    assert b"root:" not in response.content
    assert response.content == INDEX


@pytest.mark.asyncio
async def test_a_file_in_the_build_is_served(client):
    response = await raw_get(client, "/favicon.ico")
    assert response.status_code == 200
    assert response.content == b"icon"


@pytest.mark.asyncio
async def test_assets_are_served(client):
    response = await raw_get(client, "/assets/app.js")
    assert response.status_code == 200
    assert response.content == b"js"


@pytest.mark.asyncio
async def test_a_client_side_route_gets_the_index(client):
    response = await raw_get(client, "/some/client/route")
    assert response.status_code == 200
    assert response.content == INDEX


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["/api", "/api/unknown", "/api/real/extra"])
async def test_an_unclaimed_api_path_is_a_404(client, target):
    response = await raw_get(client, target)
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_api_routes_registered_first_still_win(client):
    response = await raw_get(client, "/api/real")
    assert response.status_code == 200
    assert response.json() == {"ok": True}
