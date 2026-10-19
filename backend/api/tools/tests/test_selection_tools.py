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

"""The agent's selection tools: select_residues, combine_selections, show_selection.

They share one contract with a selection the user picks by hand -- same
artifact, same registry -- and differ only in provenance. These tests run them
against a stored structure with no viewer and no Redis server.
"""

import json

import pytest

from api.selections.registry import get_available_selections
from api.storage import ArtifactNotFoundError
from api.tools import selection_tools
from api.tools.base import ContentType
from api.tools.selection_tools import (
    CombineSelectionsTool,
    SelectResiduesTool,
    ShowSelectionTool,
    show_event,
)

from api.tests.test_atom_selection import CIF as WHOLE_RESIDUE_CIF
from api.tests.test_selection_ops import CIF
from api.tests.test_user_selections import FakeRedis


class Storage:
    """Stores and retrieves, like the real service does."""

    def __init__(self):
        self.items = {"mmcif_t": CIF.encode(), "mmcif_whole": WHOLE_RESIDUE_CIF.encode()}
        self.n = 0
        self.committed = 0

    async def store(self, data_type, content, metadata=None, description=None):
        self.n += 1
        ref = f"{data_type}_{self.n:03d}"
        self.items[ref] = content
        self.items[ref + ":meta"] = metadata
        return ref

    async def retrieve(self, reference):
        if reference not in self.items:
            raise ArtifactNotFoundError(reference)
        return self.items[reference]

    async def commit(self):
        self.committed += 1


@pytest.fixture
def redis(monkeypatch):
    fake = FakeRedis()
    monkeypatch.setattr(selection_tools, "_get_redis", lambda: fake)
    return fake


@pytest.fixture
def storage():
    return Storage()


def make(tool_cls, storage):
    tool = tool_cls()
    tool.set_storage(storage)
    tool._session_id = "s1"
    return tool


def stored_selection(storage, ref):
    return json.loads(storage.items[ref])


def text_of(result):
    assert result.content_type == ContentType.TEXT, result.data
    return result.data


@pytest.mark.asyncio
class TestSelectResidues:
    async def test_creates_an_agent_selection_by_author_number(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56, "115"], "name": "epitope"}
        )
        ref = result.metadata["selection_ref"]
        data = stored_selection(storage, ref)
        assert data["provenance"] == "agent"
        assert data["scope"] == "residue"
        assert data["name"] == "epitope"
        assert {a["residue"]["authSeqNumber"] for a in data["atoms"]} == {56, 115}
        # Label ids ride along, so the design tool and the viewer can both use it.
        assert {a["residue"]["seqNumber"] for a in data["atoms"]} == {40, 99}
        assert storage.committed == 1
        # What the trace span will record about this selection.
        assert result.metadata["selection_provenance"] == "agent"
        assert result.metadata["selection_residue_count"] == 2
        assert "selection_parent" not in result.metadata

    async def test_is_listed_for_the_next_turn(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56], "name": "epitope"}
        )
        (info,) = await get_available_selections(redis, "s1")
        assert info.reference == result.metadata["selection_ref"]
        assert info.provenance == "agent"
        assert info.name == "epitope"

    async def test_reply_names_the_residues_and_reference(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": ["115-117"]}
        )
        text = text_of(result)
        assert result.metadata["selection_ref"] in text
        assert "3 residues" in text
        assert "M115" in text

    async def test_shows_in_the_viewer_with_label_ids(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56]}
        )
        (event,) = result.metadata["visualization_events"]
        assert event["type"] == "show_selection"
        # This fixture's Tyr holds 3 of its 12 atoms, so it is shown atom by atom.
        assert event["residues"] == []
        assert event["atoms"] == [
            {"chain": "A", "seq": 40, "name": n} for n in ("N", "CA", "CG")
        ]
        assert event["pdb_ref"] == "mmcif_t"
        assert event["focus"] is True

    async def test_can_decline_to_show(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56], "show": False}
        )
        assert "visualization_events" not in result.metadata

    async def test_missing_residues_are_reported_not_hidden(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56, 9999]}
        )
        assert "Not found in chain A: 9999" in text_of(result)

    async def test_nothing_found_stores_nothing(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [9999]}
        )
        assert result.content_type == ContentType.ERROR
        assert "Nothing was selected" in result.data
        assert await get_available_selections(redis, "s1") == []

    async def test_label_numbering_can_be_asked_for(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [40], "numbering": "label"}
        )
        data = stored_selection(storage, result.metadata["selection_ref"])
        assert {a["residue"]["authSeqNumber"] for a in data["atoms"]} == {56}

    @pytest.mark.parametrize(
        "args",
        [
            {"chain": "A", "residues": [1]},
            {"structure_ref": "mmcif_t", "residues": [1]},
            {"structure_ref": "mmcif_t", "chain": "A"},
        ],
    )
    async def test_missing_arguments_are_rejected_as_invalid(self, storage, redis, args):
        result = await make(SelectResiduesTool, storage).execute(args)
        assert result.content_type == ContentType.ERROR
        assert result.metadata.get("invalid_arguments")

    async def test_an_expired_structure_says_so(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_gone", "chain": "A", "residues": [1]}
        )
        assert "may have expired" in result.data

    async def test_works_without_redis(self, storage, monkeypatch):
        monkeypatch.setattr(selection_tools, "_get_redis", lambda: None)
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56]}
        )
        assert result.metadata["selection_ref"] in storage.items

    async def test_a_selection_is_not_a_structure(self, storage, redis):
        first = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56]}
        )
        result = await make(SelectResiduesTool, storage).execute(
            {
                "structure_ref": first.metadata["selection_ref"],
                "chain": "A",
                "residues": [56],
            }
        )
        assert result.content_type == ContentType.ERROR
        assert "Could not read any atoms" in result.data


@pytest.mark.asyncio
@pytest.mark.asyncio
class TestAtomLevelTools:
    async def test_per_residue_atoms_are_stored_as_an_atom_selection(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": ["56:CA,CG", 115]}
        )
        data = stored_selection(storage, result.metadata["selection_ref"])
        assert data["scope"] == "atom"
        by_residue = {}
        for a in data["atoms"]:
            by_residue.setdefault(a["residue"]["authSeqNumber"], []).append(a["name"])
        assert by_residue == {56: ["CA", "CG"], 115: ["CA"]}
        # The reply and the next turn's prompt both name the chosen atoms.
        assert "Y56[CA,CG]" in result.data

    async def test_the_viewer_gets_whole_residues_as_residues_and_chosen_atoms_as_atoms(
        self, storage, redis
    ):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_whole", "chain": "A", "residues": [56, "115:SD,CE", 116]}
        )
        (event,) = result.metadata["visualization_events"]
        assert event["residues"] == [{"chain": "A", "seq": 40}, {"chain": "A", "seq": 100}]
        assert event["atoms"] == [
            {"chain": "A", "seq": 99, "name": "SD"},
            {"chain": "A", "seq": 99, "name": "CE"},
        ]

    async def test_whole_residues_carry_no_atom_list(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_whole", "chain": "A", "residues": [56]}
        )
        (event,) = result.metadata["visualization_events"]
        assert "atoms" not in event

    async def test_whole_residues_stay_residue_scoped(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56]}
        )
        assert stored_selection(storage, result.metadata["selection_ref"])["scope"] == "residue"

    async def test_the_keyword_form_is_accepted(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56], "atoms": "backbone"}
        )
        data = stored_selection(storage, result.metadata["selection_ref"])
        assert {a["name"] for a in data["atoms"]} <= {"N", "CA", "C", "O"}
        assert data["scope"] == "atom"

    async def test_an_atom_the_residue_lacks_is_named_in_the_reply(self, storage, redis):
        result = await make(SelectResiduesTool, storage).execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": ["56:CA,NZ"]}
        )
        assert "56:NZ" in result.data

    async def test_subtracting_an_atom_leaves_the_rest_of_its_residue(self, storage, redis):
        select = make(SelectResiduesTool, storage)
        both = await select.execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56], "atoms": ["CA", "CG", "N"]}
        )
        one = await select.execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": ["56:CA"]}
        )
        result = await make(CombineSelectionsTool, storage).execute(
            {
                "operation": "subtract",
                "selection_ref": both.metadata["selection_ref"],
                "other_selection_ref": one.metadata["selection_ref"],
            }
        )
        data = stored_selection(storage, result.metadata["selection_ref"])
        assert sorted(a["name"] for a in data["atoms"]) == ["CG", "N"]
        assert data["scope"] == "atom"

    async def test_two_whole_residue_selections_still_combine_by_residue(self, storage, redis):
        select = make(SelectResiduesTool, storage)
        a = await select.execute({"structure_ref": "mmcif_t", "chain": "A", "residues": [56, 115]})
        b = await select.execute({"structure_ref": "mmcif_t", "chain": "A", "residues": [115]})
        result = await make(CombineSelectionsTool, storage).execute(
            {"operation": "subtract", "selection_ref": a.metadata["selection_ref"],
             "other_selection_ref": b.metadata["selection_ref"]}
        )
        data = stored_selection(storage, result.metadata["selection_ref"])
        assert {x["residue"]["authSeqNumber"] for x in data["atoms"]} == {56}
        assert data["scope"] == "residue"


@pytest.mark.asyncio
class TestCombineSelections:
    async def pair(self, storage, redis):
        select = make(SelectResiduesTool, storage)
        a = await select.execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56, 115], "show": False}
        )
        b = await select.execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [115, 117], "show": False}
        )
        return a.metadata["selection_ref"], b.metadata["selection_ref"]

    @pytest.mark.parametrize(
        "operation, expected",
        [("union", {56, 115, 117}), ("intersect", {115}), ("subtract", {56})],
    )
    async def test_set_operations(self, storage, redis, operation, expected):
        a, b = await self.pair(storage, redis)
        result = await make(CombineSelectionsTool, storage).execute(
            {"operation": operation, "selection_ref": a, "other_selection_ref": b}
        )
        data = stored_selection(storage, result.metadata["selection_ref"])
        assert {x["residue"]["authSeqNumber"] for x in data["atoms"]} == expected
        # Lineage for the trace span: the result descends from the first input.
        assert result.metadata["selection_parent"] == a

    async def test_records_its_parent(self, storage, redis):
        a, b = await self.pair(storage, redis)
        result = await make(CombineSelectionsTool, storage).execute(
            {"operation": "union", "selection_ref": a, "other_selection_ref": b}
        )
        assert stored_selection(storage, result.metadata["selection_ref"])["parent"] == a

    async def test_combines_a_users_selection_with_the_agents(self, storage, redis):
        a, _ = await self.pair(storage, redis)
        user_atoms = [x for x in stored_selection(storage, a)["atoms"] if x["residue"]["authSeqNumber"] == 56]
        user = await storage.store(
            "selection",
            json.dumps({"atoms": user_atoms, "provenance": "user"}).encode(),
        )
        _, b = await self.pair(storage, redis)
        result = await make(CombineSelectionsTool, storage).execute(
            {"operation": "union", "selection_ref": user, "other_selection_ref": b}
        )
        data = stored_selection(storage, result.metadata["selection_ref"])
        assert {x["residue"]["authSeqNumber"] for x in data["atoms"]} == {56, 115, 117}

    async def test_expand_needs_a_structure_and_radius(self, storage, redis):
        a, _ = await self.pair(storage, redis)
        result = await make(CombineSelectionsTool, storage).execute(
            {"operation": "expand", "selection_ref": a}
        )
        assert result.metadata.get("invalid_arguments")

    async def test_expand_grows_by_whole_residues(self, storage, redis):
        select = make(SelectResiduesTool, storage)
        a = await select.execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56], "show": False}
        )
        result = await make(CombineSelectionsTool, storage).execute(
            {
                "operation": "expand",
                "selection_ref": a.metadata["selection_ref"],
                "structure_ref": "mmcif_t",
                "radius": 5,
            }
        )
        data = stored_selection(storage, result.metadata["selection_ref"])
        assert {x["residue"]["authSeqNumber"] for x in data["atoms"]} == {56, 5}

    async def test_an_empty_result_is_an_error_not_an_empty_selection(self, storage, redis):
        a, _ = await self.pair(storage, redis)
        result = await make(CombineSelectionsTool, storage).execute(
            {"operation": "subtract", "selection_ref": a, "other_selection_ref": a}
        )
        assert result.content_type == ContentType.ERROR
        assert "empty" in result.data

    async def test_a_second_operand_is_required_for_set_operations(self, storage, redis):
        a, _ = await self.pair(storage, redis)
        result = await make(CombineSelectionsTool, storage).execute(
            {"operation": "union", "selection_ref": a}
        )
        assert result.metadata.get("invalid_arguments")

    async def test_unknown_operation_lists_the_choices(self, storage, redis):
        a, _ = await self.pair(storage, redis)
        result = await make(CombineSelectionsTool, storage).execute(
            {"operation": "xor", "selection_ref": a}
        )
        assert "union" in result.data


@pytest.mark.asyncio
class TestShowSelection:
    async def test_shows_a_user_selection(self, storage, redis):
        select = make(SelectResiduesTool, storage)
        made = await select.execute(
            {"structure_ref": "mmcif_t", "chain": "A", "residues": [56], "show": False}
        )
        result = await make(ShowSelectionTool, storage).execute(
            {"selection_ref": made.metadata["selection_ref"], "colour": "#ff8800", "focus": False}
        )
        (event,) = result.metadata["visualization_events"]
        assert {a["seq"] for a in event["atoms"]} == {40}
        assert event["colour"] == "#ff8800"
        assert event["focus"] is False

    async def test_a_bad_colour_is_rejected(self, storage, redis):
        result = await make(ShowSelectionTool, storage).execute(
            {"selection_ref": "selection_001", "colour": "orange"}
        )
        assert result.metadata.get("invalid_arguments")

    async def test_not_a_selection(self, storage, redis):
        result = await make(ShowSelectionTool, storage).execute({"selection_ref": "mmcif_t"})
        assert "not a selection" in result.data


class TestShowEvent:
    def test_residues_are_unique_and_in_label_numbering(self):
        atoms = [
            {"residue": {"chainId": "A", "seqNumber": 40}},
            {"residue": {"chainId": "A", "seqNumber": 40}},
            {"residue": {"chainId": "B", "seqNumber": 1}},
        ]
        event = show_event("selection_1", atoms, name=None)
        assert event["residues"] == [{"chain": "A", "seq": 40}, {"chain": "B", "seq": 1}]

    def test_ligand_atoms_without_a_sequence_number_cannot_be_addressed(self):
        atoms = [{"residue": {"chainId": "C", "seqNumber": None}}]
        assert show_event("selection_1", atoms, name=None)["residues"] == []

    def test_a_huge_selection_is_truncated_and_says_so(self):
        atoms = [{"residue": {"chainId": "A", "seqNumber": i}} for i in range(1, 2501)]
        event = show_event("selection_1", atoms, name=None)
        assert len(event["residues"]) == 2000
        assert event["truncated"] == 2500
