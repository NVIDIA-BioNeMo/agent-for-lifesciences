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

"""Tests for announcing a skill's synthesized tools inside its own prose.

A SKILL.md tells the model to build an HTTP request by hand and read raw fields
out of the response. Once a typed tool exists, those instructions are actively
wrong, and a model follows the first applicable instruction it reads. These tests
check that the correction reaches all three disclosure levels, and that skills
without a usable contract pay nothing.

The banner functions take a ``ToolAnnouncement`` rather than a contract, because
a skill's tools can also be builtins named by a binding; see
``test_skill_binding.py`` for that path.
"""

import pytest

from api.skills.annotate import catalog_tool_hint, resource_banner, skill_body_banner
from api.skills.binding import announcement_from_contract
from api.skills.contract import SkillContract, parse_contract
from api.skills.format import format_skill_catalog
from api.skills.registry import SkillDefinition, get_skill_registry
from api.skills.session_store import (
    SkillMeta,
    SkillSessionStore,
    get_skill_session_store,
)
from api.skills.tools import LoadSkillResourceTool, LoadSkillTool, _format_skill_body
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService

ENDPOINT_DATA = {
    "tool_suffix": "predict",
    "summary": "Predict a protein-ligand complex.",
    "url": "https://api.test/predict",
    "inputs": [
        {
            "name": "sequence",
            "data_type": "sequence",
            "json_path": "$.polymers[0].sequence",
        },
        {"name": "ligand", "data_type": "smiles", "json_path": "$.ligands[0].smiles"},
    ],
    "scalar_params": [{"name": "recycling_steps", "type": "integer"}],
    "outputs": [
        {
            "name": "structure",
            "json_path": "$.structures[0].structure",
            "data_type": "mmcif",
        }
    ],
}

SKILL_ID = "skill_ab12cd34"


@pytest.fixture
def confirmed_contract() -> SkillContract:
    return parse_contract([ENDPOINT_DATA], source="llm").model_copy(
        update={"confirmed": True}
    )


@pytest.fixture
def unconfirmed_contract() -> SkillContract:
    return parse_contract([ENDPOINT_DATA], source="llm")


def announce(contract) -> object:
    """The announcement a contract produces, as the banner functions receive it."""
    return announcement_from_contract(SKILL_ID, contract)


class TestCatalogHint:
    def test_names_the_endpoints(self, confirmed_contract):
        assert catalog_tool_hint(announce(confirmed_contract)) == " [has tools: predict]"

    def test_unconfirmed_contract_is_silent(self, unconfirmed_contract):
        # Nothing was synthesized, so advertising a tool would send the model
        # after a name that does not resolve.
        assert catalog_tool_hint(announce(unconfirmed_contract)) == ""

    def test_absent_contract_is_silent(self):
        assert catalog_tool_hint(announce(None)) == ""
        assert catalog_tool_hint(announce(SkillContract())) == ""
        assert catalog_tool_hint(None) == ""


class TestBodyBanner:
    def test_leads_the_body_so_it_is_read_first(self, confirmed_contract):
        banner = skill_body_banner(announce(confirmed_contract))
        assert banner.startswith("[READ FIRST")

    def test_names_the_full_tool_name(self, confirmed_contract):
        banner = skill_body_banner(announce(confirmed_contract))
        assert f"`{SKILL_ID}_predict`" in banner

    def test_lists_the_ref_parameters_the_tool_takes(self, confirmed_contract):
        banner = skill_body_banner(announce(confirmed_contract))
        assert "`sequence_ref`" in banner
        assert "`ligand_smiles_ref`" in banner
        assert "`recycling_steps`" in banner

    def test_maps_the_response_field_the_prose_names_onto_a_reference(
        self, confirmed_contract
    ):
        # The prose below the banner says to read structures[0].structure. Say
        # what comes back instead, or the model will go looking for it.
        banner = skill_body_banner(announce(confirmed_contract))
        assert "structures[0].structure" in banner
        assert "reference" in banner

    def test_mentions_the_array_salvage_path(self, confirmed_contract):
        banner = skill_body_banner(announce(confirmed_contract))
        assert "python_execute" in banner
        assert "aevom.load" in banner

    def test_supersedes_the_raw_http_instructions(self, confirmed_contract, monkeypatch):
        monkeypatch.setenv("AEVOM_FEAT_HTTP_REQUEST_TOOL", "true")
        banner = skill_body_banner(announce(confirmed_contract))
        assert "instead of building the HTTP request yourself" in banner
        assert "http_request" in banner

    def test_names_no_http_request_tool_when_it_is_disabled(
        self, confirmed_contract, monkeypatch
    ):
        monkeypatch.delenv("AEVOM_FEAT_HTTP_REQUEST_TOOL", raising=False)
        banner = skill_body_banner(announce(confirmed_contract))
        assert "instead of building the HTTP request yourself:" in banner
        assert "http_request" not in banner

    def test_empty_without_an_actionable_contract(self, unconfirmed_contract):
        assert skill_body_banner(announce(unconfirmed_contract)) == ""
        assert skill_body_banner(announce(None)) == ""


class TestResourceBanner:
    def test_restates_the_tool_name(self, confirmed_contract):
        banner = resource_banner(announce(confirmed_contract))
        assert f"`{SKILL_ID}_predict`" in banner

    def test_empty_without_an_actionable_contract(self, unconfirmed_contract):
        assert resource_banner(announce(unconfirmed_contract)) == ""
        assert resource_banner(announce(None)) == ""


class TestCatalogRendering:
    def _meta(self, contract):
        return SkillMeta(
            id=SKILL_ID,
            name="Boltz-2",
            description="Predict structures.",
            body_ref="pdb_x",
            contract=contract.to_dict() if contract else None,
        )

    def test_marks_skills_that_have_tools(self, confirmed_contract):
        catalog = format_skill_catalog([self._meta(confirmed_contract)])
        assert "[has tools: predict]" in catalog
        assert "name the tools to use" in catalog

    def test_stays_compact_for_toolless_skills(self, unconfirmed_contract):
        catalog = format_skill_catalog([self._meta(unconfirmed_contract)])
        assert "has tools" not in catalog
        assert "name the tools to use" not in catalog

    @pytest.mark.parametrize("enabled", [True, False])
    def test_names_http_request_only_when_it_is_enabled(
        self, confirmed_contract, monkeypatch, enabled
    ):
        monkeypatch.setenv("AEVOM_FEAT_HTTP_REQUEST_TOOL", "true" if enabled else "false")
        catalog = format_skill_catalog([self._meta(confirmed_contract)])
        assert ("http_request" in catalog) is enabled
        assert "take and return data references" in catalog


# --------------------------------------------------------------------------
# Level 2 and 3 through the actual disclosure tools
# --------------------------------------------------------------------------

class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value


@pytest.fixture
def fake_redis(monkeypatch):
    redis = FakeRedis()

    async def _get_redis(self):
        return redis

    monkeypatch.setattr(SkillSessionStore, "_get_redis", _get_redis)
    return redis


@pytest.fixture(autouse=True)
def clear_registry():
    get_skill_registry()._skills.clear()
    yield
    get_skill_registry()._skills.clear()


@pytest.fixture
def storage():
    base = DataStorageService(repository=InMemoryDataArtifactRepository())
    return SessionScopedStorage(base, session_id="sess_1")


@pytest.mark.asyncio
class TestDisclosureTools:
    async def test_body_banner_precedes_the_prose(
        self, fake_redis, storage, confirmed_contract
    ):
        get_skill_registry()._skills["boltz2"] = SkillDefinition(
            id="boltz2",
            name="Boltz-2",
            description="d",
            body="POST the sequence to /predict and read structures[0].structure.",
            contract=confirmed_contract,
        )
        skill = await get_skill_session_store().get_skill("sess_1", storage, "boltz2")
        body = _format_skill_body(skill)
        assert body.index("READ FIRST") < body.index("POST the sequence")

    async def test_load_skill_includes_the_banner(
        self, fake_redis, storage, confirmed_contract
    ):
        get_skill_registry()._skills["boltz2"] = SkillDefinition(
            id="boltz2",
            name="Boltz-2",
            description="d",
            body="Build a POST request to /predict yourself.",
            contract=confirmed_contract,
        )
        tool = LoadSkillTool()
        tool._session_id = "sess_1"
        tool._storage = storage

        result = await tool.execute({"skill_id": "boltz2"})
        assert "READ FIRST" in result.data
        assert "boltz2_predict" in result.data

    async def test_load_skill_resource_includes_the_reminder(
        self, fake_redis, storage, confirmed_contract
    ):
        get_skill_registry()._skills["boltz2"] = SkillDefinition(
            id="boltz2",
            name="Boltz-2",
            description="d",
            body="body",
            resources={"reference.md": "Set the Authorization header yourself."},
            contract=confirmed_contract,
        )
        tool = LoadSkillResourceTool()
        tool._session_id = "sess_1"
        tool._storage = storage

        result = await tool.execute({"skill_id": "boltz2", "path": "reference.md"})
        assert "Reminder" in result.data
        assert "boltz2_predict" in result.data
        # The original resource text is still there; the banner only prefixes it.
        assert "Set the Authorization header yourself." in result.data

    async def test_toolless_skill_body_is_unchanged(self, fake_redis, storage):
        get_skill_registry()._skills["notes"] = SkillDefinition(
            id="notes",
            name="Notes",
            description="d",
            body="Just instructions.",
        )
        tool = LoadSkillTool()
        tool._session_id = "sess_1"
        tool._storage = storage

        result = await tool.execute({"skill_id": "notes"})
        assert "READ FIRST" not in result.data
        assert result.data.startswith("Just instructions.")
