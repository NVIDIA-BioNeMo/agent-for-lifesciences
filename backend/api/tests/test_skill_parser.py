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

"""Tests for skill bundle parsing and validation."""

import pytest

from api.skills.parser import (
    MAX_FILE_BYTES,
    ParsedResource,
    SkillParseError,
    parse_skill_bundle,
    parse_skill_markdown,
    split_frontmatter,
)

VALID_SKILL_MD = b"""---
name: Test Skill
description: A test skill.
allowed_tools: [alpha, beta]
---
# Body

Do the thing.
"""


class TestSplitFrontmatter:
    def test_parses_meta_and_body(self):
        meta, body = split_frontmatter(VALID_SKILL_MD.decode())
        assert meta["name"] == "Test Skill"
        assert meta["description"] == "A test skill."
        assert body.startswith("# Body")

    def test_missing_frontmatter_raises(self):
        with pytest.raises(SkillParseError):
            split_frontmatter("# No frontmatter here")


class TestParseSkillMarkdown:
    def test_requires_name(self):
        text = "---\ndescription: x\n---\nbody"
        with pytest.raises(SkillParseError):
            parse_skill_markdown(text)

    def test_requires_description(self):
        text = "---\nname: x\n---\nbody"
        with pytest.raises(SkillParseError):
            parse_skill_markdown(text)

    def test_allowed_tools_list(self):
        parsed = parse_skill_markdown(VALID_SKILL_MD.decode())
        assert parsed.allowed_tools == ["alpha", "beta"]

    def test_allowed_tools_comma_string(self):
        text = "---\nname: x\ndescription: y\nallowed_tools: a, b\n---\nbody"
        parsed = parse_skill_markdown(text)
        assert parsed.allowed_tools == ["a", "b"]

    def test_allowed_tools_absent(self):
        text = "---\nname: x\ndescription: y\n---\nbody"
        parsed = parse_skill_markdown(text)
        assert parsed.allowed_tools == []

    def test_carries_resources(self):
        resources = [ParsedResource(path="reference.md", content=b"ref")]
        parsed = parse_skill_markdown(VALID_SKILL_MD.decode(), resources)
        assert len(parsed.resources) == 1
        assert parsed.resources[0].path == "reference.md"


class TestParseSkillBundle:
    def test_basic_bundle(self):
        parsed = parse_skill_bundle(
            [
                ("myskill/SKILL.md", VALID_SKILL_MD),
                ("myskill/reference.md", b"reference body"),
            ]
        )
        assert parsed.name == "Test Skill"
        assert [r.path for r in parsed.resources] == ["reference.md"]

    def test_nested_root_detection(self):
        parsed = parse_skill_bundle(
            [
                ("top/skill-dir/SKILL.md", VALID_SKILL_MD),
                ("top/skill-dir/docs/extra.md", b"extra"),
            ]
        )
        assert [r.path for r in parsed.resources] == ["docs/extra.md"]

    def test_skips_disallowed_extensions(self):
        parsed = parse_skill_bundle(
            [
                ("s/SKILL.md", VALID_SKILL_MD),
                ("s/reference.md", b"ref"),
                ("s/script.py", b"print('x')"),
                ("s/logo.png", b"\x89PNG"),
            ]
        )
        assert [r.path for r in parsed.resources] == ["reference.md"]

    def test_missing_skill_md_raises(self):
        with pytest.raises(SkillParseError):
            parse_skill_bundle([("s/reference.md", b"ref")])

    def test_path_traversal_rejected(self):
        with pytest.raises(SkillParseError):
            parse_skill_bundle(
                [("SKILL.md", VALID_SKILL_MD), ("../evil.md", b"x")]
            )

    def test_absolute_path_rejected(self):
        with pytest.raises(SkillParseError):
            parse_skill_bundle(
                [("SKILL.md", VALID_SKILL_MD), ("/etc/passwd", b"x")]
            )

    def test_oversized_resource_rejected(self):
        big = b"x" * (MAX_FILE_BYTES + 1)
        with pytest.raises(SkillParseError):
            parse_skill_bundle(
                [("s/SKILL.md", VALID_SKILL_MD), ("s/big.txt", big)]
            )

    def test_shallowest_skill_md_chosen(self):
        # Two SKILL.md; the shallowest defines the root.
        parsed = parse_skill_bundle(
            [
                ("SKILL.md", VALID_SKILL_MD),
                ("nested/SKILL.md", VALID_SKILL_MD),
                ("reference.md", b"ref"),
            ]
        )
        # root is "" -> both nested/SKILL.md and reference.md are resources
        paths = sorted(r.path for r in parsed.resources)
        assert "reference.md" in paths
