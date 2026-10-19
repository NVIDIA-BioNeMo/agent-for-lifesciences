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

"""Tests for skills bound to builtin tools instead of synthesized ones.

The NIM skills are an inversion of the usual arrangement: the prose comes from a
pinned upstream submodule and must stay byte-identical, while the tool is
hand-written in Aevom. That creates three failure modes worth guarding, and each
one is silent:

- The submodule pin moves and a skill Aevom has no tool for joins the catalog,
  telling the model to hand-build HTTP requests to an endpoint nobody vetted.
- A binding is mistaken for a contract, so a duplicate synthesized tool appears
  and the user is asked to authorize an outbound host that is already covered.
- The upstream prose -- which is written for an agent with a shell, ``requests``
  and a filesystem -- reaches the model without the correction saying that none
  of those exist here.
"""

import pytest

from api.skills.annotate import catalog_tool_hint, resource_banner, skill_body_banner
from api.skills.binding import (
    LIBRARY_SKILLS,
    NIM_SKILL_BINDINGS,
    BoundTool,
    SkillBinding,
    announcement_from_binding,
    bound_skill_ids,
    get_binding,
    library_skill_ids,
    library_skills_root,
    nim_skills_root,
)
from api.skills.endpoint_tool import synthesize_skill_tools
from api.skills.registry import SkillRegistry
from api.skills.session_store import LoadedSkill, SkillMeta
from api.skills.tools import _format_skill_body
from api.tools.registry import ToolRegistry, _register_default_tools

pytestmark = pytest.mark.skipif(
    nim_skills_root() is None,
    reason="bionemo-agent-toolkit submodule not checked out (run 'make submodules')",
)

# Parametrized from the allowlist rather than restated, so a new binding is
# covered by the verbatim-prose and no-synthesis checks the moment it is added.
BOUND_SKILL_IDS = bound_skill_ids()
LIBRARY_SKILL_IDS = library_skill_ids()


@pytest.fixture(scope="module")
def registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.load_builtin_skills()
    return registry


class TestAllowlist:
    def test_only_the_bound_nim_skills_are_registered(self, registry):
        # The submodule carries roughly ten NIM skill directories. Loading it by
        # directory scan would register all of them, including `meta-skills`,
        # which is not a NIM, and skills for endpoints Aevom cannot call.
        loaded = {skill.id for skill in registry.list()}
        on_disk = {
            path.name
            for path in nim_skills_root().iterdir()
            if path.is_dir() and (path / "SKILL.md").is_file()
        }
        assert set(bound_skill_ids()) <= loaded
        assert (on_disk - set(bound_skill_ids())) & loaded == set()

    def test_the_catalog_is_exactly_the_expected_skills(self, registry):
        assert {skill.id for skill in registry.list()} == {
            "example_skill",
            "fragment-assembly",
            "structure-selection-analysis",
            "rfdiffusion-nim",
            "proteinmpnn-nim",
            "genmol-nim",
            "boltz2-nim",
            "msa-search-nim",
            *LIBRARY_SKILL_IDS,
        }

    def test_every_binding_names_a_registered_builtin_tool(self):
        # A binding that names a tool nobody registered would announce a name the
        # model cannot call.
        tool_registry = ToolRegistry()
        _register_default_tools(tool_registry)
        for binding in NIM_SKILL_BINDINGS:
            for name in binding.tool_names:
                assert name in tool_registry, (
                    f"{binding.skill_id} names unregistered tool {name}"
                )

    def test_every_binding_resolves_to_a_skill_on_disk(self):
        for binding in NIM_SKILL_BINDINGS:
            skill_md = nim_skills_root() / binding.skill_id / "SKILL.md"
            assert skill_md.is_file(), f"{binding.skill_id} has no SKILL.md"


class TestProseIsUnmodified:
    """Ingest must store upstream prose verbatim; corrections are read-time only."""

    @pytest.mark.parametrize("skill_id", BOUND_SKILL_IDS)
    def test_body_matches_the_submodule_file(self, registry, skill_id):
        source = (nim_skills_root() / skill_id / "SKILL.md").read_text(encoding="utf-8")
        # The parser strips frontmatter and surrounding whitespace; everything
        # after that must be the upstream bytes.
        _, _, after_frontmatter = source.partition("---\n")
        expected = after_frontmatter.split("\n---\n", 1)[1].strip()
        assert registry.get(skill_id).body == expected

    @pytest.mark.parametrize("skill_id", BOUND_SKILL_IDS)
    def test_references_match_the_submodule_files(self, registry, skill_id):
        skill = registry.get(skill_id)
        assert skill.resources, "expected the bundled references to be loaded"
        for path, content in skill.resources.items():
            on_disk = (nim_skills_root() / skill_id / path).read_text(encoding="utf-8")
            assert content == on_disk

    def test_the_eval_harness_is_not_offered_as_a_resource(self, registry):
        # proteinmpnn-nim ships evals/harbor/.../instruction.md, which is test
        # infrastructure for the toolkit, not reference material for the model.
        paths = registry.get("proteinmpnn-nim").resources
        assert all(path.startswith("references/") for path in paths)
        assert not any("eval" in path for path in paths)

    def test_upstream_agent_tools_do_not_leak_into_allowed_tools(self, registry):
        # Upstream declares `allowed-tools: Bash, Read, Write, AskUserQuestion`.
        # None of those exist in Aevom, and load_skill renders allowed_tools to
        # the model verbatim, so they must not be picked up.
        for skill_id in bound_skill_ids():
            assert registry.get(skill_id).allowed_tools == []


class TestNoToolSynthesis:
    @pytest.mark.parametrize("skill_id", BOUND_SKILL_IDS)
    def test_a_bound_skill_synthesizes_nothing(self, registry, skill_id):
        skill = registry.get(skill_id)
        # No contract means no duplicate HTTP tool, and no consent gate opened
        # for a host the builtin tool already reaches on its own vetted path.
        assert skill.contract is None
        assert synthesize_skill_tools(skill.id, skill.name, skill.contract) == []

    def test_a_bound_skill_still_announces_its_tools(self, registry):
        skill = registry.get("rfdiffusion-nim")
        assert skill.announcement is not None
        assert skill.announcement.short_names == ("rfdiffusion_design_backbone",)


class TestDisclosureLevels:
    """The announcement has to reach all three levels the model can read."""

    def test_level_1_catalog_names_the_builtin_tool(self, registry):
        skill = registry.get("proteinmpnn-nim")
        assert (
            catalog_tool_hint(skill.announcement)
            == " [has tools: proteinmpnn_design_sequence]"
        )

    def test_level_2_banner_leads_the_body(self, registry):
        skill = registry.get("rfdiffusion-nim")
        loaded = LoadedSkill(
            id=skill.id,
            name=skill.name,
            body=skill.body,
            allowed_tools=skill.allowed_tools,
            resource_paths=list(skill.resources),
            source=skill.source,
            binding=skill.binding,
        )
        body = _format_skill_body(loaded)
        assert body.startswith("[READ FIRST")
        assert "rfdiffusion_design_backbone" in body
        # The prose is still there in full, below the correction.
        assert body.index("READ FIRST") < body.index("# RFDiffusion NIM")

    def test_level_2_banner_supersedes_the_local_docker_prose(self, registry):
        # Aevom has no shell, so a model that follows the upstream "Local Docker"
        # section will try to `docker run` and stall the turn.
        banner = skill_body_banner(registry.get("rfdiffusion-nim").announcement)
        assert "docker run" in banner
        assert "Local Docker" in banner
        assert "requests" in banner

    def test_level_3_resource_banner_names_the_builtin_tool(self, registry):
        banner = resource_banner(registry.get("msa-search-nim").announcement)
        assert "`msa_search`" in banner
        assert "Reminder" in banner

    def test_de_novo_guidance_survives_into_the_banner(self, registry):
        # The endpoint rejects a de novo request with no input structure, and the
        # tool covers that. If the banner does not say so, the model will hunt
        # for a structure it does not need.
        banner = skill_body_banner(registry.get("rfdiffusion-nim").announcement)
        assert "Omit `pdb_ref` for de novo" in banner

    def test_the_broken_upstream_score_pairing_is_corrected(self, registry):
        # Upstream's snippet filters headers containing "native" or "wt", but the
        # service labels the input row `>input`, so its filter keeps that row and
        # misaligns every score.
        banner = skill_body_banner(registry.get("proteinmpnn-nim").announcement)
        assert "`>input`" in banner
        assert "misaligns" in banner

    def test_the_safe_not_smiles_correction_survives_into_the_banner(self, registry):
        # The upstream prose calls the request field `smiles`. A model that takes
        # the field name at face value hands GenMol a plain SMILES, which is
        # accepted and quietly generates worse molecules.
        banner = skill_body_banner(registry.get("genmol-nim").announcement)
        assert "must carry SAFE" in banner
        assert "smiles_to_safe" in banner
        # The `.smi` file the prose writes has nowhere to go in Aevom.
        assert "Save And Report Output" in banner

    def test_unavailable_endpoints_are_named(self, registry):
        # msa_search wraps the monomer endpoint only. Without this the model will
        # try to use it for a complex.
        banner = skill_body_banner(registry.get("msa-search-nim").announcement)
        assert "Not available as a tool in Aevom" in banner
        assert "Paired MSA" in banner


class TestBindingLookup:
    def test_get_binding_returns_the_table_entry(self):
        assert get_binding("rfdiffusion-nim").tool_names == (
            "rfdiffusion_design_backbone",
        )

    def test_unknown_skill_has_no_binding(self):
        assert get_binding("not-a-skill") is None

    def test_a_binding_with_no_tools_announces_nothing(self):
        assert announcement_from_binding(SkillBinding("x", tools=())) is None
        assert announcement_from_binding(None) is None

    def test_a_session_skill_cannot_claim_a_nim_binding_by_name(self):
        # Bindings are keyed by skill id, so a user installing a skill called
        # "rfdiffusion-nim" would otherwise inherit its builtin tools.
        impostor = SkillMeta(
            id="rfdiffusion-nim",
            name="Impostor",
            description="d",
            body_ref="ref_1",
        )
        assert impostor.binding is None
        assert impostor.announcement is None


class TestAnnouncementRendering:
    def test_takes_and_returns_are_both_rendered(self):
        announcement = announcement_from_binding(
            SkillBinding(
                "demo",
                tools=(
                    BoundTool(
                        name="demo_tool",
                        summary="Do the thing.",
                        takes="`a_ref`.",
                        returns="a `pdb` reference.",
                    ),
                ),
            )
        )
        body = "\n".join(announcement.body_lines)
        assert "`demo_tool` — Do the thing." in body
        assert "Takes: `a_ref`." in body
        assert "Returns: a `pdb` reference." in body

    def test_a_tool_without_takes_or_returns_renders_one_line(self):
        announcement = announcement_from_binding(
            SkillBinding("demo", tools=(BoundTool(name="t", summary="s."),))
        )
        assert list(announcement.body_lines) == ["* `t` — s."]

    def test_a_reference_only_binding_forbids_pasting_raw_data(self):
        announcement = announcement_from_binding(
            SkillBinding("demo", tools=(BoundTool(name="t", summary="s."),))
        )
        assert "never paste raw" in announcement.trailer

    def test_a_literal_taking_binding_does_not_forbid_pasting_smiles(self):
        # `molecule_gates` takes SMILES literals. Telling the model never to
        # paste a SMILES would make the tool it is being pointed at uncallable.
        announcement = announcement_from_binding(
            SkillBinding(
                "demo",
                tools=(BoundTool(name="t", summary="s."),),
                reference_only=False,
            )
        )
        assert "never paste raw" not in announcement.trailer
        assert "storage reference or a literal value" in announcement.trailer


class TestLibrarySkills:
    """Prose for the packages installed in the python_execute sandbox.

    Loaded through the same root-plus-allowlist path as the NIM skills so that
    swapping in an upstream submodule later is a one-function change. These
    tests pin the parts of that arrangement a refactor could silently break.
    """

    @pytest.mark.parametrize("skill_id", LIBRARY_SKILL_IDS)
    def test_every_library_skill_is_registered(self, registry, skill_id):
        assert registry.get(skill_id) is not None

    @pytest.mark.parametrize("skill_id", LIBRARY_SKILL_IDS)
    def test_body_matches_the_file_on_disk(self, registry, skill_id):
        source = (library_skills_root() / skill_id / "SKILL.md").read_text(
            encoding="utf-8"
        )
        _, _, after_frontmatter = source.partition("---\n")
        expected = after_frontmatter.split("\n---\n", 1)[1].strip()
        assert registry.get(skill_id).body == expected

    @pytest.mark.parametrize("skill_id", LIBRARY_SKILL_IDS)
    def test_a_library_skill_synthesizes_nothing(self, registry, skill_id):
        # No contract means no synthesized HTTP tool and no consent gate for a
        # host nobody vetted. These skills describe local packages.
        skill = registry.get(skill_id)
        assert skill.contract is None
        assert synthesize_skill_tools(skill.id, skill.name, skill.contract) == []

    def test_every_library_binding_names_a_registered_builtin_tool(self):
        tool_registry = ToolRegistry()
        _register_default_tools(tool_registry)
        for entry in LIBRARY_SKILLS:
            if entry.binding is None:
                continue
            for name in entry.binding.tool_names:
                assert name in tool_registry, (
                    f"{entry.skill_id} names unregistered tool {name}"
                )

    def test_an_unbound_library_skill_announces_nothing(self, registry):
        # biotite and openbabel are pure sandbox prose. Announcing a tool would
        # send the model after a name that does not resolve.
        for skill_id in ("openbabel", "biotite", "biopython"):
            assert registry.get(skill_id).announcement is None

    def test_plip_routes_to_the_tool_and_supersedes_the_import(self, registry):
        # PLIP is deliberately absent from the sandbox: its API is file-based
        # and its protonation step races on a fixed path. A model that follows
        # upstream prose will write `import plip` and get an ImportError.
        banner = skill_body_banner(registry.get("plip").announcement)
        assert "plip_interactions" in banner
        assert "import plip" in banner
        assert "PDBComplex" in banner

    def test_rdkit_points_at_molecule_gates_before_hand_rolling(self, registry):
        banner = skill_body_banner(registry.get("rdkit").announcement)
        assert "molecule_gates" in banner
        assert "smiles_to_safe" in banner
        # The tool takes SMILES literals, so the reference-only prohibition
        # must not fire here.
        assert "never paste raw" not in banner

    def test_the_banner_names_python_execute_not_http_request(self, registry):
        # The default lede is written for NIM skills, where the temptation is a
        # hand-built HTTP call. For a library the temptation is hand-written
        # sandbox code, and naming the wrong one makes the correction land
        # nowhere.
        banner = skill_body_banner(registry.get("rdkit").announcement)
        assert "python_execute" in banner
        assert "http_request" not in banner

    @pytest.mark.parametrize("skill_id", LIBRARY_SKILL_IDS)
    def test_each_skill_isolates_its_aevom_corrections(self, skill_id):
        # The convention that makes a later submodule swap cheap: everything
        # Aevom-specific lives under one heading, ready to migrate into the
        # binding's notes when upstream prose replaces the body.
        source = (library_skills_root() / skill_id / "SKILL.md").read_text(
            encoding="utf-8"
        )
        assert "## Running this in Aevom" in source

    @pytest.mark.parametrize("skill_id", LIBRARY_SKILL_IDS)
    def test_no_upstream_agent_tools_leak_into_allowed_tools(self, registry, skill_id):
        assert registry.get(skill_id).allowed_tools == []
