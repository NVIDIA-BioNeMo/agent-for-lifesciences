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

"""Tests for the browser stand-in.

The bridge is the one part of the harness that must behave like a component it
does not share code with (``frontend/src/App.tsx``), so the assertions here are
about matching that component's observable behaviour: what it POSTs, when it
declines to POST, and what text it puts in front of the model.
"""

from __future__ import annotations

import pytest

from aevom_eval.driver import molstar_bridge as bridge_module
from aevom_eval.driver.molstar_bridge import MAX_ATOMS, MolstarBridge
from aevom_eval.driver.http import HttpError


def _atom(atom_id: int, *, resno: int = 20, name: str = "CA") -> dict:
    return {
        "name": name,
        "element": name[0],
        "id": atom_id,
        "coords": [1.0, 2.0, 3.0],
        "residue": {
            "name": "GLY",
            "seqNumber": resno,
            "chainId": "A",
            "insCode": "",
            "entityType": "polymer",
        },
    }


@pytest.fixture
def posts(monkeypatch):
    """Capture every POST the bridge makes, and script molstar's replies."""
    calls: list[tuple[str, dict]] = []
    replies: dict[str, object] = {}

    def fake_post_json(url, payload, timeout=None, token=None):
        calls.append((url, payload))
        for fragment, reply in replies.items():
            if fragment in url:
                if isinstance(reply, Exception):
                    raise reply
                return reply
        return {}

    monkeypatch.setattr(bridge_module, "post_json", fake_post_json)
    return calls, replies


@pytest.fixture
def bridge():
    return MolstarBridge("http://api:8000", "http://molstar-headless:8020")


def _selection_posts(calls):
    return [(url, payload) for url, payload in calls if "/api/selection" in url]


def _query_event(molql="(sel.atom.all)", **overrides):
    event = {
        "type": "structure_query",
        "query_id": "q-1",
        "query": {"molql": molql, "action": "select"},
    }
    event.update(overrides)
    return event


def test_visualize_pdb_loads_the_viewer(bridge, posts):
    calls, _ = posts
    bridge.note_tool_event(
        {"type": "visualize_pdb", "pdb": "ATOM ...", "label": "1CRN"}, "s1"
    )

    assert bridge.loaded_pdb == "ATOM ..."
    assert bridge.loaded_label == "1CRN"
    assert calls == [], "loading a structure posts nothing"


def test_query_executes_molql_and_posts_the_selection(bridge, posts):
    calls, replies = posts
    replies["/query"] = {
        "atoms": [_atom(1), _atom(2)],
        "atom_count": 2,
        "residue_count": 1,
        "summary": "Selected 2 atoms in 1 residues",
    }
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "PDBDATA"}, "s1")
    bridge.note_tool_event(_query_event("(sel.atom.res)"), "s1")

    query_url, query_payload = calls[0]
    assert query_url.endswith("/query")
    assert query_payload == {"pdb_data": "PDBDATA", "molql": "(sel.atom.res)"}

    (selection_url, selection), = _selection_posts(calls)
    assert selection_url == "http://api:8000/api/selection?session_id=s1"
    assert selection["query_id"] == "q-1"
    assert selection["molql"] == "(sel.atom.res)"
    assert selection["action"] == "select"
    assert selection["atom_count"] == 2
    assert selection["residue_count"] == 1
    assert selection["summary"] == "Selected 2 atoms in 1 residues"
    assert selection["atoms"][0] == _atom(1)


def test_mmcif_reaches_the_query_unchanged(bridge, posts):
    """boltz2 output must be queryable without a new HTTP contract.

    The bridge stays format-agnostic: it forwards whatever text the event carried
    and molstar-headless detects mmCIF from the `data_` block header. Teaching
    the bridge about formats instead would put the detection in two places.
    """
    calls, replies = posts
    replies["/query"] = {"atoms": [_atom(1)], "atom_count": 1, "residue_count": 1}
    mmcif = "data_boltz2\n#\n_atom_site.group_PDB\nATOM\n"

    bridge.note_tool_event(
        {"type": "visualize_pdb", "pdb": mmcif, "format": "mmcif"}, "s1"
    )
    bridge.note_tool_event(_query_event("(sel.atom.all)"), "s1")

    _, query_payload = calls[0]
    assert query_payload == {"pdb_data": mmcif, "molql": "(sel.atom.all)"}


def test_event_session_id_wins_over_the_harness_session(bridge, posts):
    """App.tsx prefers event.session_id; the graph's id is the authoritative one."""
    calls, replies = posts
    replies["/query"] = {"atoms": [], "atom_count": 0, "residue_count": 0}
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "harness-session")
    bridge.note_tool_event(_query_event(session_id="graph-session"), "harness-session")

    (url, _), = _selection_posts(calls)
    assert "session_id=graph-session" in url


def test_query_without_a_loaded_structure_posts_an_empty_selection(bridge, posts):
    """An empty viewer selects nothing; it must still unblock the graph."""
    calls, _ = posts
    bridge.note_tool_event(_query_event(), "s1")

    assert not any(url.endswith("/query") for url, _ in calls), (
        "must not query molstar without a structure"
    )
    (_, selection), = _selection_posts(calls)
    assert selection["atoms"] == []
    assert selection["atom_count"] == 0
    assert selection["summary"] == "Selected 0 atoms"


def test_query_without_molql_posts_nothing(bridge, posts):
    """Mirrors App.tsx's `event.query?.molql` guard.

    Posting anyway would be kinder to the run but would hide a backend defect
    that production users would experience as a stall.
    """
    calls, _ = posts
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "s1")
    bridge.note_tool_event(
        {"type": "structure_query", "query_id": "q-1", "query": {"action": "select"}},
        "s1",
    )

    assert _selection_posts(calls) == []
    assert bridge.events[-1]["error"] == "no_molql"


def test_unparseable_molql_still_posts_so_the_graph_unblocks(bridge, posts):
    calls, replies = posts
    replies["/query"] = HttpError(422, "http://molstar-headless:8020/query", "bad expr")
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "s1")
    bridge.note_tool_event(_query_event("(nonsense"), "s1")

    (_, selection), = _selection_posts(calls)
    assert selection["atom_count"] == 0
    # The model reads this string, so it uses the prefix App.tsx posts and
    # carries the engine's message: without it "0 atoms" is indistinguishable
    # from a legitimately empty selection.
    assert selection["summary"] == "MolQL error: bad expr"
    assert "bad expr" in bridge.events[-1]["error"]


def test_engine_message_is_extracted_from_the_json_error_body(bridge, posts):
    calls, replies = posts
    body = (
        '{"error": "MolQL query failed: expected EOF at line 1 column 1, got '
        "'(sel.atom.re...'\", \"atoms\": [], \"atom_count\": 0}"
    )
    replies["/query"] = HttpError(422, "http://molstar-headless:8020/query", body)
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "s1")
    bridge.note_tool_event(_query_event("(sel.atom.res (in atom.resno [1, 2]))"), "s1")

    (_, selection), = _selection_posts(calls)
    assert selection["atom_count"] == 0
    assert selection["summary"] == (
        "MolQL error: expected EOF at line 1 column 1, got '(sel.atom.re...'"
    )


def test_molstar_being_unreachable_does_not_abort_the_turn(bridge, posts):
    calls, replies = posts
    replies["/query"] = ConnectionError("connection refused")
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "s1")
    bridge.note_tool_event(_query_event(), "s1")

    assert len(_selection_posts(calls)) == 1


def test_atoms_are_capped_but_the_count_is_truthful(bridge, posts):
    """Matches MAX_ATOMS in App.tsx: cap the payload, report the real total."""
    calls, replies = posts
    total = MAX_ATOMS + 25
    replies["/query"] = {
        "atoms": [_atom(i) for i in range(total)],
        "atom_count": total,
        "residue_count": 300,
        "summary": f"Selected {total} atoms in 300 residues",
    }
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "s1")
    bridge.note_tool_event(_query_event(), "s1")

    (_, selection), = _selection_posts(calls)
    assert len(selection["atoms"]) == MAX_ATOMS
    assert selection["atom_count"] == total


def test_a_failed_selection_post_is_recorded_not_raised(bridge, posts):
    """The graph will time out on its own; raising would lose the trajectory."""
    calls, replies = posts
    replies["/query"] = {"atoms": [_atom(1)], "atom_count": 1, "residue_count": 1}
    replies["/api/selection"] = HttpError(500, "http://api:8000/api/selection", "boom")

    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "s1")
    bridge.note_tool_event(_query_event(), "s1")

    assert any(e["kind"] == "selection_post_failed" for e in bridge.events)


def test_events_record_enough_to_tell_no_query_from_empty_query(bridge, posts):
    """The verifier needs this distinction to score the structure_query task."""
    _, replies = posts
    replies["/query"] = {
        "atoms": [],
        "atom_count": 0,
        "residue_count": 0,
        "summary": "No atoms matched the query",
    }
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "s1")
    bridge.note_tool_event(_query_event("(sel.atom.res (= atom.resno 999))"), "s1")

    queries = [e for e in bridge.events if e["kind"] == "structure_query"]
    assert len(queries) == 1
    assert queries[0]["atom_count"] == 0
    assert queries[0]["error"] is None, "an empty match is not an error"


def test_atom_payload_is_projected_onto_the_backend_model(bridge, posts):
    """Extra or missing fields must not turn into a 422 that looks like a stall."""
    calls, replies = posts
    replies["/query"] = {
        "atoms": [{"name": "CA", "element": "C", "id": 7, "unexpected": True}],
        "atom_count": 1,
        "residue_count": 1,
    }
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "s1")
    bridge.note_tool_event(_query_event(), "s1")

    (_, selection), = _selection_posts(calls)
    atom = selection["atoms"][0]
    assert set(atom) == {"name", "element", "id", "coords", "residue"}
    assert atom["coords"] == [0.0, 0.0, 0.0]
    assert atom["residue"]["insCode"] == ""


def test_author_numbering_is_forwarded_beside_the_label_pair(bridge, posts):
    """The backend resolves hotspots from author ids; dropping them here would
    make every eval selection fall back to the ambiguous legacy reading."""
    calls, replies = posts
    atom = _atom(7, resno=40)
    atom["residue"]["authSeqNumber"] = 56
    atom["residue"]["authChainId"] = "A"
    replies["/query"] = {"atoms": [atom], "atom_count": 1, "residue_count": 1}
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "s1")
    bridge.note_tool_event(_query_event(), "s1")

    (_, selection), = _selection_posts(calls)
    residue = selection["atoms"][0]["residue"]
    assert residue["seqNumber"] == 40
    assert residue["authSeqNumber"] == 56
    assert residue["authChainId"] == "A"


def test_author_numbering_is_not_invented_when_the_viewer_sent_none(bridge, posts):
    calls, replies = posts
    replies["/query"] = {"atoms": [_atom(7)], "atom_count": 1, "residue_count": 1}
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "P"}, "s1")
    bridge.note_tool_event(_query_event(), "s1")

    (_, selection), = _selection_posts(calls)
    residue = selection["atoms"][0]["residue"]
    assert "authSeqNumber" not in residue
    assert "authChainId" not in residue


def test_show_selection_records_residues_and_chosen_atoms(bridge, posts):
    calls, _ = posts
    bridge.note_tool_event(
        {
            "type": "show_selection",
            "selection_ref": "selection_abc",
            "residues": [{"chain": "A", "seq": 40}],
            "atoms": [{"chain": "A", "seq": 99, "name": "SD"}, {"chain": "A", "seq": 99, "name": "CE"}],
        },
        "s1",
    )

    assert calls == [], "showing something posts nothing"
    (shown,) = [e for e in bridge.events if e["kind"] == "viewer_show"]
    assert (shown["residue_count"], shown["atom_count"]) == (1, 2)


def test_the_selection_post_carries_the_owner_token(monkeypatch):
    # The backend only accepts a selection from the session's owner, which
    # the browser proves with its cookie and the harness with this token.
    tokens: dict[str, object] = {}

    def fake_post_json(url, payload, timeout=None, token=None):
        tokens[url] = token
        return {"atoms": [], "atom_count": 0, "residue_count": 0, "summary": ""}

    monkeypatch.setattr(bridge_module, "post_json", fake_post_json)
    bridge = MolstarBridge("http://api:8000", "http://molstar-headless:8020", "tok")
    bridge.note_tool_event({"type": "visualize_pdb", "pdb": "PDBDATA"}, "s1")
    bridge.note_tool_event(_query_event(), "s1")

    assert tokens["http://api:8000/api/selection?session_id=s1"] == "tok"
    assert tokens["http://molstar-headless:8020/query"] is None
