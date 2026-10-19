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

"""``aevom.load_selection()``'s residue helpers and ``aevom.store_selection()``.

Residues carry two numbering systems that differ for many structures (5O45's
author numbers run 16 ahead of its label numbers). The helpers must hand back
both, and say so when a stored selection predates author ids, because an
analysis that quietly treats label numbers as author numbers picks the wrong
residue.
"""

import json

import pytest

from app.aevom_client import AevomClient, AevomError, Selection


def atom(name, comp, label_seq, auth_seq=None, chain="A", auth_chain=None, ins=""):
    residue = {
        "name": comp,
        "seqNumber": label_seq,
        "chainId": chain,
        "insCode": ins,
        "entityType": "polymer",
    }
    if auth_seq is not None:
        residue["authSeqNumber"] = auth_seq
        residue["authChainId"] = auth_chain or chain
    return {"name": name, "element": name[0], "id": 1, "coords": [0, 0, 0], "residue": residue}


class TestResidues:
    def test_each_residue_once_in_order(self):
        sel = Selection(
            atoms=[
                atom("CA", "TYR", 40, 56),
                atom("CG", "TYR", 40, 56),
                atom("CA", "MET", 99, 115),
            ]
        )
        assert [(r["name"], r["number"]) for r in sel.residues()] == [("TYR", 56), ("MET", 115)]

    def test_both_numbering_systems_are_returned(self):
        (residue,) = Selection(atoms=[atom("CA", "TYR", 40, 56)]).residues()
        assert (residue["number"], residue["label_seq"]) == (56, 40)
        assert residue["author_known"] is True

    def test_chains_differ_between_systems(self):
        (residue,) = Selection(atoms=[atom("CA", "GLY", 1, 5, chain="B", auth_chain="C")]).residues()
        assert (residue["chain"], residue["label_chain"]) == ("C", "B")

    def test_an_old_selection_is_marked_not_independently_known(self):
        (residue,) = Selection(atoms=[atom("CA", "TYR", 40)]).residues()
        assert residue["number"] == 40
        assert residue["author_known"] is False

    def test_insertion_codes_are_kept(self):
        residues = Selection(
            atoms=[atom("CA", "SER", 10, 52), atom("CA", "THR", 11, 52, ins="A")]
        ).residues()
        assert [r["ins_code"] for r in residues] == ["", "A"]

    def test_residue_numbers_can_be_limited_to_a_chain(self):
        sel = Selection(
            atoms=[atom("CA", "TYR", 40, 56), atom("CA", "GLY", 1, 5, chain="B", auth_chain="C")]
        )
        assert sel.residue_numbers("C") == [5]
        assert sorted(sel.residue_numbers()) == [5, 56]

    def test_it_is_still_a_dict(self):
        sel = Selection(atoms=[], atom_count=0, provenance="user")
        assert sel["provenance"] == "user"
        assert json.loads(json.dumps(sel))["atom_count"] == 0

    def test_an_empty_selection_has_no_residues(self):
        assert Selection().residues() == []


class FakeHTTPResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def client(monkeypatch):
    instance = AevomClient(
        session_token="tok",
        backend_url="http://backend:8000",
        allowed_refs={"mmcif_target01", "selection_known01"},
    )
    instance.sent = []
    instance.reply = {"success": True, "reference": "selection_new00001"}

    def fake_urlopen(request, timeout=None):
        if "/api/sandbox/data/" in request.full_url:
            stored = Selection(atoms=[atom("CA", "TYR", 40, 56)], provenance="user", name="mine")
            return FakeHTTPResponse(json.dumps(stored).encode())
        instance.sent.append((request.full_url, json.loads(request.data.decode())))
        if isinstance(instance.reply, Exception):
            raise instance.reply
        return FakeHTTPResponse(json.dumps(instance.reply).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return instance


class TestAtoms:
    def test_a_residue_lists_its_atoms_once_in_order(self):
        (residue,) = Selection(
            atoms=[
                atom("OH", "TYR", 40, 56),
                atom("CZ", "TYR", 40, 56),
                atom("OH", "TYR", 40, 56),  # an alternate location repeats the name
            ]
        ).residues()
        assert residue["atoms"] == ["OH", "CZ"]
        assert residue["partial"] is True

    def test_every_heavy_atom_is_a_whole_residue(self):
        names = ["N", "CA", "C", "O"]
        (residue,) = Selection(atoms=[atom(n, "GLY", 1, 5) for n in names]).residues()
        assert residue["partial"] is False

    def test_a_non_standard_residue_is_never_partial(self):
        (residue,) = Selection(atoms=[atom("FE", "HEM", 0, 601)]).residues()
        assert residue["partial"] is False

    def test_the_table_matches_the_backends(self):
        from api.selections.residues import STANDARD_HEAVY_ATOMS
        from app.aevom_client import _STANDARD_HEAVY_ATOMS

        assert _STANDARD_HEAVY_ATOMS == STANDARD_HEAVY_ATOMS


class TestLoadSelection:
    def test_returns_a_selection_with_helpers(self, client):
        sel = client.load_selection("selection_known01")
        assert isinstance(sel, Selection)
        assert sel["provenance"] == "user"
        assert sel.residue_numbers() == [56]


class TestStoreSelection:
    def test_posts_residues_and_returns_the_reference(self, client):
        reference = client.store_selection(
            "mmcif_target01", [56, "100-120"], chain="A", name="epitope", atoms=["CA"]
        )
        assert reference == "selection_new00001"
        (url, body), = client.sent
        assert url.endswith("/api/sandbox/selection")
        assert body == {
            "structure_ref": "mmcif_target01",
            "chain": "A",
            "residues": [56, "100-120"],
            "numbering": "author",
            "name": "epitope",
            "atoms": ["CA"],
        }

    def test_per_residue_atoms_and_keywords_pass_through(self, client):
        client.store_selection(
            "mmcif_target01", [56, "115:SD", "123:CZ,OH"], chain="A", atoms="backbone"
        )
        (_, body), = client.sent
        assert body["residues"] == [56, "115:SD", "123:CZ,OH"]
        assert body["atoms"] == "backbone"

    def test_the_new_reference_becomes_readable(self, client):
        reference = client.store_selection("mmcif_target01", [56], chain="A")
        assert reference in client.allowed_refs

    def test_a_refusal_carries_the_servers_reason(self, client):
        client.reply = {"success": False, "error": "None of those residues exist in chain A: 9999."}
        with pytest.raises(AevomError, match="9999"):
            client.store_selection("mmcif_target01", [9999], chain="A")

    def test_a_missing_reference_is_an_error(self, client):
        client.reply = {"success": True}
        with pytest.raises(AevomError, match="reference"):
            client.store_selection("mmcif_target01", [56], chain="A")
