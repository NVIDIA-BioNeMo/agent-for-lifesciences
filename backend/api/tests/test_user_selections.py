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

"""A selection made by hand is the same artifact an agent query makes.

Covers the residue view both producers share, the registry fields that tell
them apart, the eviction rule that protects the user's work, how the prompt
block reads them, and the POST endpoint's handling of a pick that has no tool
call behind it.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.selections import (
    PROVENANCE_AGENT,
    PROVENANCE_USER,
    ResidueRef,
    format_residue_listing,
    register_selection,
    residues_from_atoms,
)
from api.selections.registry import (
    MAX_SELECTIONS,
    SelectionInfo,
    get_available_selections,
)
from api.selections.models import SelectionData
from api.selections.service import store_selection_data
from api.state.format import format_selections


def atom(name, comp, label_seq, auth_seq=None, chain="A", auth_chain=None, ins="", entity="polymer"):
    """An atom as the viewer posts it; author fields only when given."""
    residue = {
        "name": comp,
        "seqNumber": label_seq,
        "chainId": chain,
        "insCode": ins,
        "entityType": entity,
    }
    if auth_seq is not None:
        residue["authSeqNumber"] = auth_seq
        residue["authChainId"] = auth_chain or chain
    return {"name": name, "element": name[0], "id": 1, "coords": [0, 0, 0], "residue": residue}


def whole(comp, label_seq, auth_seq=None, **kwargs):
    """Every heavy atom of a standard residue, so it reads as a whole residue."""
    from api.selections.residues import STANDARD_HEAVY_ATOMS

    names = ["N", "CA", "C", "O"] + [f"X{i}" for i in range(STANDARD_HEAVY_ATOMS[comp] - 4)]
    return [atom(n, comp, label_seq, auth_seq, **kwargs) for n in names]


class FakeRedis:
    """Just enough of redis.asyncio for the registry: get, set, delete."""

    def __init__(self):
        self.values: dict[str, str] = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def delete(self, key):
        self.values.pop(key, None)


class TestResidueView:
    def test_atoms_collapse_to_residues_in_viewer_order(self):
        residues = residues_from_atoms(
            [*whole("TYR", 40, 56), *whole("MET", 99, 115)]
        )
        assert [r.token for r in residues] == ["Y56", "M115"]

    def test_both_numbering_systems_are_kept(self):
        (residue,) = residues_from_atoms([atom("CA", "TYR", 40, 56)])
        assert (residue.label_seq, residue.auth_seq) == (40, 56)
        assert residue.auth_known

    def test_a_legacy_payload_is_marked_as_not_independently_known(self):
        (residue,) = residues_from_atoms([atom("CA", "TYR", 40)])
        assert not residue.auth_known
        assert residue.auth_seq == 40

    def test_a_ligand_has_no_label_sequence_number(self):
        (residue,) = residues_from_atoms([atom("C1", "HEM", 0, 601, entity="non-polymer")])
        assert residue.label_seq is None
        assert not residue.is_polymer

    def test_insertion_codes_keep_residues_apart(self):
        residues = residues_from_atoms(
            [atom("CA", "SER", 10, 52), atom("CA", "THR", 11, 52, ins="A")]
        )
        assert len(residues) == 2

    def test_a_residue_missing_its_number_is_skipped_not_fatal(self):
        bad = atom("CA", "GLY", 1)
        bad["residue"]["seqNumber"] = None
        assert residues_from_atoms([bad]) == []

    def test_round_trips_through_a_dict(self):
        (residue,) = residues_from_atoms([atom("CA", "TYR", 40, 56)])
        assert ResidueRef.from_dict(residue.to_dict()) == residue


class TestListing:
    def test_author_numbering_and_one_letter_codes(self):
        listing = format_residue_listing(
            residues_from_atoms([*whole("TYR", 40, 56), *whole("MET", 99, 115)])
        )
        assert listing == "A: Y56 M115"

    def test_a_run_is_one_token(self):
        atoms = [a for i in range(10, 20) for a in whole("GLY", i, i + 16)]
        listing = format_residue_listing(residues_from_atoms(atoms))
        assert listing == "A: G26-G35 (10 res)"

    def test_grouped_by_chain(self):
        listing = format_residue_listing(
            residues_from_atoms(
                [*whole("TYR", 1, 5, chain="A"), *whole("LYS", 1, 7, chain="B")]
            )
        )
        assert listing == "A: Y5 | B: K7"

    def test_the_rest_is_counted_not_named(self):
        atoms = [atom("CA", "ALA", i, i * 3) for i in range(1, 31)]
        listing = format_residue_listing(residues_from_atoms(atoms), max_listed=5)
        assert listing.endswith("(+25 more residues)")
        assert listing.count("A") < 30

    def test_unusual_residue_is_named_in_full(self):
        listing = format_residue_listing(
            residues_from_atoms([atom("FE", "HEM", 0, 601, entity="non-polymer")])
        )
        assert listing == "A: HEM601"

    def test_nothing_selected_lists_nothing(self):
        assert format_residue_listing([]) == ""


class TestRegistry:
    @pytest.mark.asyncio
    async def test_user_fields_survive_the_round_trip(self):
        redis = FakeRedis()
        await register_selection(
            redis, "s1", "selection_u", molql="", action="select", atom_count=3,
            residue_count=2, summary="x", provenance=PROVENANCE_USER,
            name="epitope", parent="selection_p", scope="residue",
            residue_listing="A: Y56 M115",
        )
        (info,) = await get_available_selections(redis, "s1")
        assert (info.provenance, info.name, info.parent, info.scope) == (
            "user", "epitope", "selection_p", "residue",
        )
        assert info.residue_listing == "A: Y56 M115"
        assert info.query_id is None

    def test_an_old_registry_entry_reads_as_an_agent_selection(self):
        info = SelectionInfo.from_dict(
            {"reference": "selection_old", "molql": "(sel.atom.all)", "atom_count": 5}
        )
        assert info.provenance == PROVENANCE_AGENT
        assert info.name is None and info.scope == "atom"

    @pytest.mark.asyncio
    async def test_agent_probes_are_evicted_before_a_user_pick(self):
        redis = FakeRedis()
        await register_selection(
            redis, "s1", "selection_mine", "", "select", 3, 1, "", provenance=PROVENANCE_USER
        )
        for i in range(MAX_SELECTIONS + 4):
            await register_selection(redis, "s1", f"selection_probe{i}", "q", "select", 1, 1, "")

        refs = [s.reference for s in await get_available_selections(redis, "s1")]
        assert len(refs) == MAX_SELECTIONS
        assert "selection_mine" in refs
        assert "selection_probe0" not in refs
        assert f"selection_probe{MAX_SELECTIONS + 3}" in refs

    @pytest.mark.asyncio
    async def test_when_everything_is_a_user_pick_the_oldest_goes(self):
        redis = FakeRedis()
        for i in range(MAX_SELECTIONS + 1):
            await register_selection(
                redis, "s1", f"selection_u{i}", "", "select", 1, 1, "", provenance=PROVENANCE_USER
            )
        refs = [s.reference for s in await get_available_selections(redis, "s1")]
        assert len(refs) == MAX_SELECTIONS
        assert "selection_u0" not in refs

    @pytest.mark.asyncio
    async def test_the_newest_selection_is_never_the_one_dropped(self):
        # A waiting tool call looks it up by query_id; dropping it would hang
        # the turn on a selection that was just made.
        redis = FakeRedis()
        for i in range(MAX_SELECTIONS):
            await register_selection(
                redis, "s1", f"selection_u{i}", "", "select", 1, 1, "", provenance=PROVENANCE_USER
            )
        await register_selection(redis, "s1", "selection_new", "q", "select", 1, 1, "", query_id="q9")
        refs = [s.reference for s in await get_available_selections(redis, "s1")]
        assert refs[0] == "selection_new"


class TestPromptBlock:
    def test_a_user_pick_says_so_and_what_it_holds(self):
        out = format_selections(
            [
                {
                    "reference": "selection_u1",
                    "atom_count": 12,
                    "residue_count": 3,
                    "provenance": "user",
                    "name": "epitope",
                    "residue_listing": "A: Y56 M115 Y123",
                }
            ]
        )
        assert "epitope = selection_u1" in out
        assert "picked by the user" in out
        assert "A: Y56 M115 Y123" in out

    def test_an_agent_probe_keeps_the_existing_wording(self):
        out = format_selections(
            [
                {
                    "reference": "selection_a",
                    "atom_count": 10,
                    "residue_count": 2,
                    "residue_listing": "A: Y56 M115",
                }
            ]
        )
        assert out == "[SELECTION STATE: selection_a (10 atoms, 2 residues)]"

    def test_user_picks_are_not_pushed_out_by_newer_probes(self):
        probes = [
            {"reference": f"selection_p{i}", "atom_count": 1, "residue_count": 1}
            for i in range(8)
        ]
        mine = {
            "reference": "selection_mine",
            "atom_count": 1,
            "residue_count": 1,
            "provenance": "user",
        }
        out = format_selections(probes + [mine], max_selections=5)
        assert "selection_mine" in out
        assert out.index("selection_mine") < out.index("selection_p0")


class TestStoreSelectionData:
    """The body of ``POST /api/selection``, with storage and redis faked."""

    @pytest.fixture
    def parts(self):
        storage = MagicMock()
        storage.store = AsyncMock(return_value="selection_abc12345")
        ack = AsyncMock(return_value=True)
        commit = AsyncMock()
        return storage, ack, commit

    def _payload(self, **overrides):
        data = {
            "atoms": [
                atom("CA", "TYR", 40, 56),
                atom("CG", "TYR", 40, 56),
                atom("CA", "MET", 99, 115),
            ],
            "atom_count": 999,  # deliberately wrong: the server must not trust it
            "residue_count": 999,
            "provenance": "user",
            "scope": "residue",
            "name": "epitope",
        }
        data.update(overrides)
        return SelectionData(**data)

    async def _store(self, parts, redis=None, **overrides):
        storage, ack, commit = parts
        return await store_selection_data(
            self._payload(**overrides),
            storage=storage,
            redis_client=redis or FakeRedis(),
            session_id="s1",
            acknowledge=ack,
            commit=commit,
        )

    @pytest.mark.asyncio
    async def test_a_user_pick_needs_no_query_id_and_acknowledges_nothing(self, parts):
        result = await self._store(parts)
        parts[1].assert_not_awaited()
        assert result["reference"] == "selection_abc12345"
        assert result["provenance"] == "user"

    @pytest.mark.asyncio
    async def test_counts_are_derived_from_the_atoms_for_a_user_pick(self, parts):
        result = await self._store(parts)
        assert (result["atom_count"], result["residue_count"]) == (3, 2)
        assert result["residue_listing"] == "A: Y56[CA,CG] M115[CA]"

    @pytest.mark.asyncio
    async def test_registered_for_the_prompt_with_its_provenance(self, parts):
        redis = FakeRedis()
        await self._store(parts, redis=redis)
        (info,) = await get_available_selections(redis, "s1")
        assert info.provenance == "user" and info.name == "epitope"
        assert info.residue_listing == "A: Y56[CA,CG] M115[CA]"

    @pytest.mark.asyncio
    async def test_the_artifact_keeps_both_numbering_systems(self, parts):
        await self._store(parts)
        stored = json.loads(parts[0].store.await_args.kwargs["content"])
        residue = stored["atoms"][0]["residue"]
        assert (residue["seqNumber"], residue["authSeqNumber"]) == (40, 56)
        assert stored["provenance"] == "user"

    @pytest.mark.asyncio
    async def test_the_artifact_says_who_made_it(self, parts):
        await self._store(parts)
        description = parts[0].store.await_args.kwargs["description"]
        assert description.startswith("User selection 'epitope'")
        assert "Y56" in description

    @pytest.mark.asyncio
    async def test_the_row_is_committed_before_it_is_registered(self, parts):
        redis = FakeRedis()
        order = []
        parts[2].side_effect = lambda: order.append("commit")
        original_set = redis.set

        async def recording_set(*args, **kwargs):
            order.append("register")
            await original_set(*args, **kwargs)

        redis.set = recording_set
        await self._store(parts, redis=redis)
        assert order == ["commit", "register"]

    @pytest.mark.asyncio
    async def test_an_agent_query_still_acknowledges_its_pending_op(self, parts):
        await self._store(
            parts, provenance="agent", query_id="q1", molql="(sel.atom.all)", scope="atom"
        )
        parts[1].assert_awaited_once_with("q1")

    @pytest.mark.asyncio
    async def test_an_agent_selection_keeps_the_clients_counts(self, parts):
        # The viewer caps the atoms it sends; its counts are the truthful ones.
        result = await self._store(parts, provenance="agent", query_id="q1", scope="atom")
        assert result["atom_count"] == 999

    def test_a_payload_from_before_either_producer_still_validates(self):
        data = SelectionData(
            query_id="q", molql="m", action="select", atoms=[], atom_count=0,
            residue_count=0, summary="s",
        )
        assert data.provenance == "agent" and data.scope == "atom"
