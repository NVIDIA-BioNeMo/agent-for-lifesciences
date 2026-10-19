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

"""Skill install from a GitHub URL cannot steer its authenticated requests.

The owner, repo and branch parsed from the URL are spliced into api.github.com
URLs sent with GITHUB_TOKEN. Anything outside GitHub's own character set is
refused before a request is made.
"""

import pytest

from api.skills import routes
from api.skills.parser import SkillParseError


class _Response:
    def __init__(self, payload=None, content=b""):
        self._payload = payload or {}
        self.status_code = 200
        self.content = content

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class RecordingClient:
    urls: list[str] = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url, params=None):
        RecordingClient.urls.append(url)
        if "/git/trees/" in url:
            return _Response({"tree": [{"type": "blob", "path": "skill/SKILL.md"}]})
        if url.startswith("https://api.github.com/repos/"):
            return _Response({"default_branch": "main"})
        return _Response(content=b"---\nname: s\n---\n")


@pytest.fixture
def client(monkeypatch):
    RecordingClient.urls = []
    monkeypatch.setattr(routes.httpx, "AsyncClient", RecordingClient)
    return RecordingClient


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/../../user",
        "https://github.com/owner/..",
        "https://github.com/owner/repo/tree/..",
        "https://github.com/owner/repo/tree/main?x=1",
        "https://github.com/owner/repo%2F..",
        "https://github.com/own?er/repo",
        "https://github.com/owner/repo#frag",
        "https://github.com/owner/repo/tree/ma%2Fin",
        "https://gitlab.com/owner/repo",
    ],
)
async def test_unsafe_urls_are_refused_before_any_request(client, url):
    with pytest.raises(SkillParseError):
        await routes._fetch_github_skill(url)
    assert client.urls == []


@pytest.mark.asyncio
async def test_a_tree_url_requests_only_the_expected_paths(client):
    files = await routes._fetch_github_skill(
        "https://github.com/NVIDIA/skills-repo.v2/tree/release-1.0/skill"
    )
    assert [path for path, _ in files] == ["skill/SKILL.md"]
    assert client.urls == [
        "https://api.github.com/repos/NVIDIA/skills-repo.v2/git/trees/release-1.0",
        "https://raw.githubusercontent.com/NVIDIA/skills-repo.v2/release-1.0/skill/SKILL.md",
    ]


@pytest.mark.asyncio
async def test_a_repo_url_looks_up_the_default_branch(client):
    await routes._fetch_github_skill("https://github.com/owner/repo")
    assert client.urls[0] == "https://api.github.com/repos/owner/repo"
    assert client.urls[1] == "https://api.github.com/repos/owner/repo/git/trees/main"
