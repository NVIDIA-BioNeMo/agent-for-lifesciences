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

"""Tests for the curated builtin skill registry loading from disk."""

from api.skills.binding import bound_skill_ids
from api.skills.registry import SkillRegistry, get_skill_registry


class TestBuiltinSkillLoading:
    def test_loads_curated_example_skill(self):
        registry = SkillRegistry()
        registry.load_builtin_skills()  # default backend/skills dir

        example = registry.get("example_skill")
        assert example is not None
        assert example.name
        assert example.description
        assert example.source == "builtin"
        # The curated example ships a reference.md for Level 3 disclosure.
        assert "reference.md" in example.resources
        assert "report" in example.resources["reference.md"].lower()

    def test_loads_the_fragment_assembly_skill_with_its_references(self):
        registry = SkillRegistry()
        registry.load_builtin_skills()

        skill = registry.get("fragment-assembly")
        assert skill is not None
        assert set(skill.resources) == {
            "references/linker-library.md",
            "references/geometry-cookbook.md",
            "references/worked-example.md",
        }

    def test_fragment_assembly_names_only_tools_that_exist(self):
        # A skill naming a tool that is not registered tells the model to call
        # something it cannot call.
        from api.tools.registry import get_registry

        registry = SkillRegistry()
        registry.load_builtin_skills()
        skill = registry.get("fragment-assembly")

        assert set(skill.allowed_tools) <= set(get_registry().tool_names)

    def test_load_is_idempotent(self):
        registry = SkillRegistry()
        registry.load_builtin_skills()
        count = len(registry)
        registry.load_builtin_skills()
        assert len(registry) == count

    def test_missing_directory_is_safe(self, tmp_path):
        # base_dir names the first-party skills directory only. Its absence must
        # not take down the bound NIM skills, which load from the submodule.
        registry = SkillRegistry()
        registry.load_builtin_skills(tmp_path / "does_not_exist")
        assert registry.get("example_skill") is None
        assert set(bound_skill_ids()).issubset({s.id for s in registry.list()})

    def test_singleton(self):
        assert get_skill_registry() is get_skill_registry()
