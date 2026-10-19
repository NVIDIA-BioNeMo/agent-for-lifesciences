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

"""create_design_spec, validate_design_spec, and design_protein_binder's spec path."""

import json

import pytest

from api.selections.design_spec import parse_design_spec
from api.tools.base import ContentType
from api.tools.design_spec_tools import CreateDesignSpecTool, ValidateDesignSpecTool
from api.tools.rfd4_proteina import DesignMotifScaffoldTool, DesignProteinBinderTool
from api.tools.rfdiffusion import RFDiffusionTool
from api.tools.tests.test_rfd4_proteina import (
    DESIGN_RESPONSE,
    EPITOPE_SELECTION,
    FakeStorage,
    TARGET_CIF,
    error_text,
)


class Store(FakeStorage):
    """FakeStorage that can read back what it stored."""

    async def retrieve(self, reference):
        for item in self.stored:
            if item["reference"] == reference:
                return item["content"]
        return await super().retrieve(reference)

    async def commit(self):
        pass


def make(cls, **kwargs):
    storage = Store(
        {
            "mmcif_target": TARGET_CIF.encode(),
            "mmcif_other": TARGET_CIF.encode(),
            "selection_epitope": json.dumps(EPITOPE_SELECTION).encode(),
            "selection_other": json.dumps(EPITOPE_SELECTION).encode(),
            "pdb_target": b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n",
            "selection_garbage": b"not json",
        }
    )
    tool = cls(**kwargs)
    tool._storage = storage
    return tool, storage


def text(result):
    assert result.content_type == ContentType.TEXT, result.data
    return result.data


@pytest.fixture
def binder():
    return make(
        DesignProteinBinderTool,
        endpoint_url="http://localhost:8000/biology/ipd/rfd4-proteina/generative",
        api_key=None,
        metric_tags=["design"],
        server_name="Local RFD4",
    )


@pytest.mark.asyncio
class TestCreate:
    async def test_stores_a_spec_that_reads_back(self):
        tool, storage = make(CreateDesignSpecTool)
        result = await tool.execute(
            {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_epitope", "name": "v1"}
        )
        reference = result.metadata["design_spec_ref"]
        assert reference.startswith("design_spec_")
        spec = parse_design_spec(storage.stored[0]["content"])
        assert (spec.target_ref, spec.name) == ("mmcif_target", "v1")
        assert spec.layer("hotspot").selection_ref == "selection_epitope"
        assert storage.stored[0]["metadata"]["layer_count"] == 1

    async def test_an_edit_is_a_new_spec_that_names_its_parent(self):
        tool, storage = make(CreateDesignSpecTool)
        first = await tool.execute(
            {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_epitope"}
        )
        base = first.metadata["design_spec_ref"]
        second = await tool.execute(
            {"based_on_spec_ref": base, "hotspot_selection_ref": "selection_other"}
        )
        spec = parse_design_spec(storage.stored[1]["content"])
        assert spec.parent == base
        assert spec.target_ref == "mmcif_target"
        assert spec.layer("hotspot").selection_ref == "selection_other"
        # The original is untouched.
        assert parse_design_spec(storage.stored[0]["content"]).layer("hotspot").selection_ref == "selection_epitope"
        assert second.metadata["design_spec_ref"] != base

    async def test_a_spec_is_for_one_target(self):
        tool, _ = make(CreateDesignSpecTool)
        first = await tool.execute(
            {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_epitope"}
        )
        result = await tool.execute(
            {
                "based_on_spec_ref": first.metadata["design_spec_ref"],
                "target_ref": "mmcif_other",
                "hotspot_selection_ref": "selection_other",
            }
        )
        assert result.metadata.get("invalid_arguments")

    async def test_a_missing_artifact_fails_now_not_at_design_time(self):
        tool, storage = make(CreateDesignSpecTool)
        result = await tool.execute(
            {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_gone"}
        )
        assert "may have expired" in error_text(result)
        assert storage.stored == []

    async def test_a_non_selection_is_refused(self):
        tool, _ = make(CreateDesignSpecTool)
        result = await tool.execute(
            {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_garbage"}
        )
        assert "not a selection" in error_text(result)

    async def test_the_selection_is_required(self):
        tool, _ = make(CreateDesignSpecTool)
        result = await tool.execute({"target_ref": "mmcif_target"})
        assert result.metadata.get("invalid_arguments")


@pytest.mark.asyncio
class TestValidate:
    async def test_lists_what_rfd4_will_be_told(self):
        tool, _ = make(ValidateDesignSpecTool)
        reply = text(
            await tool.execute(
                {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_epitope"}
            )
        )
        assert "sel('A/*/40/[CG,OH]')" in reply
        assert "label numbering" in reply

    async def test_reports_the_numbering_translation_the_design_tool_would_make(self):
        tool, _ = make(ValidateDesignSpecTool)
        reply = text(
            await tool.execute(
                {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_epitope"}
            )
        )
        # EPITOPE_SELECTION is in the viewer's author numbering (56); RFD4 is
        # told 40, and the reply says the translation happened.
        assert "Residue A56 in the viewer is A40" in reply

    async def test_works_from_a_saved_spec(self):
        creator, storage = make(CreateDesignSpecTool)
        made = await creator.execute(
            {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_epitope"}
        )
        validator, _ = make(ValidateDesignSpecTool)
        validator._storage = storage
        reply = text(await validator.execute({"design_spec_ref": made.metadata["design_spec_ref"]}))
        assert "sel('A/*/40" in reply

    async def test_pdb_is_refused_as_the_design_tool_would(self):
        tool, _ = make(ValidateDesignSpecTool)
        result = await tool.execute(
            {"target_ref": "pdb_target", "hotspot_selection_ref": "selection_epitope"}
        )
        assert "mmCIF" in error_text(result)

    async def test_needs_a_spec_or_both_references(self):
        tool, _ = make(ValidateDesignSpecTool)
        result = await tool.execute({"target_ref": "mmcif_target"})
        assert result.metadata.get("invalid_arguments")

    async def test_a_selection_on_nothing_in_the_target_is_an_error(self):
        tool, storage = make(ValidateDesignSpecTool)
        storage.artifacts["selection_elsewhere"] = json.dumps(
            {
                "atoms": [
                    {
                        "name": "CA",
                        "residue": {
                            "name": "ALA",
                            "chainId": "Z",
                            "seqNumber": 5,
                            "authChainId": "Z",
                            "authSeqNumber": 5,
                        },
                    }
                ]
            }
        ).encode()
        result = await tool.execute(
            {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_elsewhere"}
        )
        assert "No usable hotspot" in error_text(result)


@pytest.mark.asyncio
class TestDesignWithSpec:
    @pytest.fixture
    def sent(self, monkeypatch):
        captured = {}

        async def fake_call(self, json_data, **kwargs):
            captured.update(json_data)
            return DESIGN_RESPONSE

        monkeypatch.setattr(DesignProteinBinderTool, "_call_nvidia_api", fake_call)
        return captured

    async def spec_for(self, storage, target="mmcif_target", selection="selection_epitope"):
        creator = CreateDesignSpecTool()
        creator._storage = storage
        made = await creator.execute({"target_ref": target, "hotspot_selection_ref": selection})
        return made.metadata["design_spec_ref"]

    async def test_a_spec_alone_is_enough(self, binder, sent):
        tool, storage = binder
        ref = await self.spec_for(storage)
        await tool.execute({"design_spec_ref": ref})
        assert sent["input"]["spec"]["conditions"]["C_HOT"]["true"][0]["select"] == (
            "sel('A/*/40/[CG,OH]')"
        )

    async def test_a_spec_gives_the_same_request_as_the_two_references(self, binder, sent):
        tool, storage = binder
        ref = await self.spec_for(storage)
        await tool.execute({"design_spec_ref": ref, "seed": 1})
        via_spec = json.loads(json.dumps(sent))
        sent.clear()
        await tool.execute(
            {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_epitope", "seed": 1}
        )
        assert sent == via_spec

    async def test_the_design_remembers_which_spec_made_it(self, binder, sent):
        tool, storage = binder
        ref = await self.spec_for(storage)
        await tool.execute({"design_spec_ref": ref})
        design = next(i for i in storage.stored if i["data_type"] == "annotated_cif")
        assert design["metadata"]["design_spec_ref"] == ref

    async def test_the_result_carries_lineage_for_the_trace_span(self, binder, sent):
        tool, storage = binder
        ref = await self.spec_for(storage)
        result = await tool.execute({"design_spec_ref": ref})
        meta = result.metadata
        assert meta["design_spec_ref"] == ref
        assert meta["selection_ref"]
        assert meta["selection_provenance"] in {"user", "agent", "unknown"}
        assert meta["hotspot_selector_count"] >= 1

    async def test_a_conflicting_target_is_refused(self, binder, sent):
        tool, storage = binder
        ref = await self.spec_for(storage)
        result = await tool.execute({"design_spec_ref": ref, "target_ref": "mmcif_other"})
        assert "is not the spec's target" in error_text(result)
        assert sent == {}

    async def test_a_conflicting_hotspot_is_refused(self, binder, sent):
        tool, storage = binder
        ref = await self.spec_for(storage)
        result = await tool.execute(
            {"design_spec_ref": ref, "hotspot_selection_ref": "selection_other"}
        )
        assert "is not the spec's hotspot" in error_text(result)

    async def test_matching_references_alongside_a_spec_are_fine(self, binder, sent):
        tool, storage = binder
        ref = await self.spec_for(storage)
        await tool.execute(
            {
                "design_spec_ref": ref,
                "target_ref": "mmcif_target",
                "hotspot_selection_ref": "selection_epitope",
            }
        )
        assert sent

    async def test_a_missing_spec_says_so(self, binder, sent):
        tool, _ = binder
        result = await tool.execute({"design_spec_ref": "design_spec_gone"})
        assert "design spec" in error_text(result)

    async def test_the_missing_hotspot_message_points_at_the_user_selection(self, binder):
        tool, _ = binder
        result = await tool.execute({"target_ref": "mmcif_target"})
        assert "SELECTION STATE" in error_text(result)

    async def test_warnings_reach_the_reply(self, binder, sent):
        tool, storage = binder
        # The two-residue test target is almost entirely one chain: the whole
        # of it selected is not an epitope.
        whole = {
            "atoms": [
                {
                    "name": "CA",
                    "residue": {
                        "name": "X",
                        "chainId": "A",
                        "seqNumber": seq,
                        "authChainId": "A",
                        "authSeqNumber": auth,
                    },
                }
                for seq, auth in ((1, 17), (40, 56), (99, 115), (115, 131))
            ]
        }
        storage.artifacts["selection_whole"] = json.dumps(whole).encode()
        result = await tool.execute(
            {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_whole"}
        )
        assert "most of the chain" in result.to_tool_response()


# --- Roles ------------------------------------------------------------------


def residues_selection(*residues):
    """A selection of whole residues given as ``(name, author_seq)`` pairs."""
    from api.selections.residues import STANDARD_HEAVY_ATOMS

    atoms = []
    for comp, author_seq in residues:
        names = ["N", "CA", "C", "O"] + [f"X{i}" for i in range(STANDARD_HEAVY_ATOMS[comp] - 4)]
        atoms.extend(
            {
                "name": name,
                "element": name[0],
                "id": 1,
                "coords": [0.0, 0.0, 0.0],
                "residue": {
                    "name": comp,
                    "seqNumber": author_seq - 16,
                    "chainId": "A",
                    "insCode": "",
                    "entityType": "polymer",
                    "authSeqNumber": author_seq,
                    "authChainId": "A",
                },
            }
            for name in names
        )
    return {"atoms": atoms}


KEEP_SELECTION = residues_selection(("TYR", 56), ("MET", 115))


def make_with_roles(cls, **kwargs):
    tool, storage = make(cls, **kwargs)
    storage.artifacts["selection_keep"] = json.dumps(KEEP_SELECTION).encode()
    return tool, storage


class FakeRoleRedis:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def set(self, key, value, ex=None):
        self.values[key] = value

    async def delete(self, key):
        self.values.pop(key, None)


@pytest.mark.asyncio
class TestRolesInSpecs:
    async def test_layers_are_recorded_with_their_roles(self):
        tool, storage = make_with_roles(CreateDesignSpecTool)
        result = await tool.execute(
            {
                "target_ref": "mmcif_target",
                "layers": [
                    {"condition": "hotspot", "selection_ref": "selection_epitope"},
                    {"condition": "keep", "selection_ref": "selection_keep"},
                ],
            }
        )
        spec = parse_design_spec(storage.stored[0]["content"])
        assert [l.condition for l in spec.layers] == ["hotspot", "keep"]
        assert "keep selection_keep" in text(result)

    async def test_role_works_as_an_alias_for_condition(self):
        # The call the agent made in a live run, which was refused before.
        tool, storage = make_with_roles(CreateDesignSpecTool)
        result = await tool.execute(
            {
                "target_ref": "mmcif_target",
                "layers": [
                    {"role": "hotspot", "selection_ref": "selection_epitope"},
                    {"role": "keep", "selection_ref": "selection_keep"},
                ],
            }
        )
        spec = parse_design_spec(storage.stored[0]["content"])
        assert [l.condition for l in spec.layers] == ["hotspot", "keep"]
        assert "keep selection_keep" in text(result)

    def test_the_schema_offers_both_names(self):
        from api.tools.design_spec_tools import _LAYER_SCHEMA

        assert _LAYER_SCHEMA["properties"]["role"]["enum"] == _LAYER_SCHEMA["properties"]["condition"]["enum"]
        assert "condition" not in _LAYER_SCHEMA["required"]

    async def test_an_unknown_role_is_refused_with_the_list(self):
        tool, _ = make_with_roles(CreateDesignSpecTool)
        result = await tool.execute(
            {
                "target_ref": "mmcif_target",
                "layers": [{"condition": "vibes", "selection_ref": "selection_keep"}],
            }
        )
        assert "hotspot" in error_text(result)

    async def test_nothing_to_record_is_an_argument_error(self):
        tool, _ = make_with_roles(CreateDesignSpecTool)
        result = await tool.execute({"target_ref": "mmcif_target"})
        assert result.metadata.get("invalid_arguments")

    async def test_my_roles_are_the_users_latest_selection_for_each_role(self, monkeypatch):
        from api.selections.registry import register_selection

        redis = FakeRoleRedis()
        for ref, role in [("selection_epitope", "hotspot"), ("selection_keep", "keep")]:
            await register_selection(
                redis, "s1", ref, "", "select", 5, 2, "", provenance="user", role=role
            )
        monkeypatch.setattr("api.tools.design_spec_tools._get_redis", lambda: redis)
        tool, storage = make_with_roles(CreateDesignSpecTool)
        tool._session_id = "s1"
        await tool.execute({"target_ref": "mmcif_target", "use_my_roles": True})
        spec = parse_design_spec(storage.stored[0]["content"])
        assert {l.condition: l.selection_ref for l in spec.layers} == {
            "hotspot": "selection_epitope",
            "keep": "selection_keep",
        }

    async def test_an_explicit_hotspot_beats_the_users(self, monkeypatch):
        from api.selections.registry import register_selection

        redis = FakeRoleRedis()
        await register_selection(
            redis, "s1", "selection_keep", "", "select", 5, 2, "", provenance="user", role="hotspot"
        )
        monkeypatch.setattr("api.tools.design_spec_tools._get_redis", lambda: redis)
        tool, storage = make_with_roles(CreateDesignSpecTool)
        tool._session_id = "s1"
        await tool.execute(
            {
                "target_ref": "mmcif_target",
                "use_my_roles": True,
                "hotspot_selection_ref": "selection_epitope",
            }
        )
        spec = parse_design_spec(storage.stored[0]["content"])
        assert spec.layer("hotspot").selection_ref == "selection_epitope"

    async def test_validate_shows_the_contig_and_conditions_the_roles_make(self):
        tool, _ = make_with_roles(ValidateDesignSpecTool)
        reply = text(
            await tool.execute(
                {
                    "target_ref": "mmcif_target",
                    "layers": [
                        {"condition": "hotspot", "selection_ref": "selection_epitope"},
                        {"condition": "keep", "selection_ref": "selection_keep"},
                    ],
                }
            )
        )
        assert "Contig: 60-120, /0, A40-99" in reply
        assert "C_SEQ true" in reply

    async def test_validate_says_a_hotspot_outside_keep_cannot_run(self):
        tool, storage = make_with_roles(ValidateDesignSpecTool)
        storage.artifacts["selection_one"] = json.dumps(residues_selection(("TYR", 56))).encode()
        result = await tool.execute(
            {
                "target_ref": "mmcif_target",
                "layers": [
                    {"condition": "hotspot", "selection_ref": "selection_epitope"},
                    {"condition": "keep", "selection_ref": "selection_one"},
                ],
            }
        )
        assert "outside Keep" in error_text(result)

    async def test_validate_can_target_rfdiffusion_and_names_what_it_drops(self):
        tool, storage = make_with_roles(ValidateDesignSpecTool)
        result = await tool.execute(
            {
                "target_ref": "mmcif_target",
                "engine": "rfdiffusion1",
                "layers": [
                    {"condition": "hotspot", "selection_ref": "selection_epitope"},
                    {"condition": "keep", "selection_ref": "selection_keep"},
                    {"condition": "float", "selection_ref": "selection_keep"},
                ],
            }
        )
        reply = text(result)
        assert "RFdiffusion" in reply and "Float was dropped" in reply
        assert "hotspot_res" in reply

    async def test_the_design_tool_runs_a_spec_with_kept_ranges(self, binder, monkeypatch):
        captured = {}

        async def fake_call(self, json_data, **kwargs):
            captured.update(json_data)
            return DESIGN_RESPONSE

        monkeypatch.setattr(DesignProteinBinderTool, "_call_nvidia_api", fake_call)
        tool, storage = binder
        storage.artifacts["selection_keep"] = json.dumps(KEEP_SELECTION).encode()
        creator = CreateDesignSpecTool()
        creator._storage = storage
        made = await creator.execute(
            {
                "target_ref": "mmcif_target",
                "layers": [
                    {"condition": "hotspot", "selection_ref": "selection_epitope"},
                    {"condition": "keep", "selection_ref": "selection_keep"},
                ],
            }
        )
        await tool.execute({"design_spec_ref": made.metadata["design_spec_ref"]})
        body = captured["input"]["spec"]
        assert body["contig"] == "60-120, /0, A40-99"
        assert body["conditions"]["C_SEQ"]["true"] == [{"select": "sel('A/*/[40-99]')"}]

    async def test_a_spec_that_cannot_run_is_not_sent(self, binder, monkeypatch):
        called = []

        async def fake_call(self, json_data, **kwargs):
            called.append(1)
            return DESIGN_RESPONSE

        monkeypatch.setattr(DesignProteinBinderTool, "_call_nvidia_api", fake_call)
        tool, storage = binder
        storage.artifacts["selection_one"] = json.dumps(residues_selection(("TYR", 56))).encode()
        creator = CreateDesignSpecTool()
        creator._storage = storage
        made = await creator.execute(
            {
                "target_ref": "mmcif_target",
                "layers": [
                    {"condition": "hotspot", "selection_ref": "selection_epitope"},
                    {"condition": "keep", "selection_ref": "selection_one"},
                ],
            }
        )
        result = await tool.execute({"design_spec_ref": made.metadata["design_spec_ref"]})
        assert "outside Keep" in error_text(result)
        assert called == []


@pytest.fixture
def motif():
    return make(
        DesignMotifScaffoldTool,
        endpoint_url="http://localhost:8000/biology/ipd/rfd4-proteina/generative",
        api_key=None,
        server_name="Local RFD4",
    )


async def _spec(storage, layers, **extra):
    creator = CreateDesignSpecTool()
    creator._storage = storage
    storage.artifacts["selection_keep"] = json.dumps(KEEP_SELECTION).encode()
    made = await creator.execute({"target_ref": "mmcif_target", "layers": layers, **extra})
    return made.metadata["design_spec_ref"]


@pytest.mark.asyncio
class TestMotifScaffoldTool:
    async def test_it_sends_the_motif_scaffolding_task_with_the_kept_residues(
        self, motif, monkeypatch
    ):
        captured = {}

        async def fake_call(self, json_data, **kwargs):
            captured.update(json_data)
            return DESIGN_RESPONSE

        monkeypatch.setattr(DesignMotifScaffoldTool, "_call_nvidia_api", fake_call)
        tool, storage = motif
        ref = await _spec(
            storage,
            [
                {"condition": "keep", "selection_ref": "selection_keep"},
                {"condition": "fix_sequence", "selection_ref": "selection_keep"},
            ],
        )
        result = await tool.execute({"design_spec_ref": ref, "num_designs": 2})

        assert captured["task"] == "motif_scaffolding"
        spec = captured["input"]["spec"]
        assert "A40" in spec["contig"] and "A99" in spec["contig"]
        assert "/0" not in spec["contig"], "a motif is one protein, not a binder and a target"
        assert "C_HOT" not in spec["conditions"]
        assert captured["sampling"]["num_replicates"] == 2
        assert "metrics" not in captured, "binder metric tags do not apply to a scaffold"
        assert result.content_type != ContentType.TEXT or "error" not in str(result.data).lower()

    async def test_the_layout_on_keep_sets_the_designed_lengths(self, motif, monkeypatch):
        captured = {}

        async def fake_call(self, json_data, **kwargs):
            captured.update(json_data)
            return DESIGN_RESPONSE

        monkeypatch.setattr(DesignMotifScaffoldTool, "_call_nvidia_api", fake_call)
        tool, storage = motif
        ref = await _spec(
            storage,
            [
                {
                    "condition": "keep",
                    "selection_ref": "selection_keep",
                    "value": {"n_term": [11, 13], "linkers": [[21, 23]], "c_term": [31, 33]},
                }
            ],
        )
        await tool.execute({"design_spec_ref": ref})
        contig = captured["input"]["spec"]["contig"]
        assert "11-13" in contig and "21-23" in contig and "31-33" in contig

    async def test_each_scaffold_is_stored_and_summarised(self, motif, monkeypatch):
        async def fake_call(self, json_data, **kwargs):
            return DESIGN_RESPONSE

        monkeypatch.setattr(DesignMotifScaffoldTool, "_call_nvidia_api", fake_call)
        tool, storage = motif
        ref = await _spec(storage, [{"condition": "keep", "selection_ref": "selection_keep"}])
        result = await tool.execute({"design_spec_ref": ref, "description": "carbonic scaffold"})
        kinds = [item["data_type"] for item in storage.stored if item["reference"] != ref]
        assert kinds.count("annotated_cif") == 2 and kinds.count("sequence") == 2
        body = result.to_tool_response()
        assert "Scaffolded the motif into 2 design(s)" in body
        assert "renumbered" in body
        assert result.metadata["design_spec_ref"] == ref

    async def test_a_spec_without_keep_is_refused_before_any_call(self, motif, monkeypatch):
        called = []

        async def fake_call(self, json_data, **kwargs):
            called.append(1)
            return DESIGN_RESPONSE

        monkeypatch.setattr(DesignMotifScaffoldTool, "_call_nvidia_api", fake_call)
        tool, storage = motif
        ref = await _spec(
            storage, [{"condition": "hotspot", "selection_ref": "selection_epitope"}]
        )
        assert "no keep layer" in error_text(await tool.execute({"design_spec_ref": ref}))
        assert called == []

    async def test_a_spec_is_required(self, motif):
        tool, _ = motif
        assert "design_spec_ref is required" in error_text(await tool.execute({}))

    async def test_a_spec_for_the_other_engine_is_refused(self, motif):
        tool, storage = motif
        ref = await _spec(
            storage,
            [{"condition": "keep", "selection_ref": "selection_keep"}],
            engine="rfdiffusion1",
        )
        assert "rfdiffusion1" in error_text(await tool.execute({"design_spec_ref": ref}))

    async def test_a_spec_that_cannot_run_says_why(self, motif):
        tool, storage = motif
        storage.artifacts["selection_one"] = json.dumps(residues_selection(("TYR", 56))).encode()
        ref = await _spec(
            storage,
            [
                {"condition": "keep", "selection_ref": "selection_one"},
                {"condition": "fix_position", "selection_ref": "selection_keep"},
            ],
        )
        assert "outside Keep" in error_text(await tool.execute({"design_spec_ref": ref}))

    async def test_the_schema_requires_the_spec(self, motif):
        tool, _ = motif
        fn = tool.get_schema()["function"]
        assert fn["name"] == "design_motif_scaffold"
        assert fn["parameters"]["required"] == ["design_spec_ref"]


PDB_TARGET = (
    "".join(
        f"ATOM  {i:5d}  CA  {comp} A{seq:4d}    {0.0:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C\n"
        for i, (comp, seq) in enumerate([("TYR", 56), ("MET", 99), ("MET", 115)], 1)
    )
    + "END\n"
)


@pytest.fixture
def rfdiffusion(monkeypatch):
    tool, storage = make(RFDiffusionTool)
    storage.artifacts["pdb_spec_target"] = PDB_TARGET.encode()
    sent = {}

    async def fake_call(self, json_data, **kwargs):
        sent.update(json_data)
        return {"output_pdb": PDB_TARGET}

    monkeypatch.setattr(RFDiffusionTool, "_call_nvidia_api", fake_call)
    return tool, storage, sent


async def _rfdiffusion_spec(storage, layers, target="pdb_spec_target", **extra):
    creator = CreateDesignSpecTool()
    creator._storage = storage
    storage.artifacts["selection_keep"] = json.dumps(KEEP_SELECTION).encode()
    made = await creator.execute(
        {"target_ref": target, "engine": "rfdiffusion1", "layers": layers, **extra}
    )
    return made.metadata["design_spec_ref"]


@pytest.mark.asyncio
class TestRFdiffusionFromSpec:
    async def test_a_spec_is_compiled_to_contigs_and_hotspots(self, rfdiffusion):
        tool, storage, sent = rfdiffusion
        ref = await _rfdiffusion_spec(
            storage,
            [
                {"condition": "hotspot", "selection_ref": "selection_epitope"},
                {"condition": "keep", "selection_ref": "selection_keep"},
            ],
        )
        result = await tool.execute({"design_spec_ref": ref, "binder_length_min": 70, "binder_length_max": 90})
        assert result.content_type == ContentType.DATA_REFERENCE, result.data
        assert sent["input_pdb"] == PDB_TARGET, "the spec's target is what is sent"
        assert "70-90" in sent["contigs"] and "A56" in sent["contigs"]
        assert sent["hotspot_res"] == ["A56", "A99"] or "A56" in sent["hotspot_res"]
        assert result.metadata["design_spec_ref"] == ref

    async def test_roles_it_cannot_take_are_reported_not_hidden(self, rfdiffusion):
        tool, storage, sent = rfdiffusion
        ref = await _rfdiffusion_spec(
            storage,
            [
                {"condition": "hotspot", "selection_ref": "selection_epitope"},
                {"condition": "keep", "selection_ref": "selection_keep"},
                {"condition": "float", "selection_ref": "selection_keep"},
            ],
        )
        result = await tool.execute({"design_spec_ref": ref})
        assert "Float was dropped" in result.to_tool_response()

    async def test_fix_sequence_is_reported_as_positions_to_check(self, rfdiffusion):
        tool, storage, _ = rfdiffusion
        ref = await _rfdiffusion_spec(
            storage,
            [
                {"condition": "hotspot", "selection_ref": "selection_epitope"},
                {"condition": "keep", "selection_ref": "selection_keep"},
                {"condition": "fix_sequence", "selection_ref": "selection_keep"},
            ],
        )
        result = await tool.execute({"design_spec_ref": ref})
        assert "A:56" in result.to_tool_response()
        assert "no fixed-positions option" in result.to_tool_response()

    async def test_a_spec_cannot_be_mixed_with_contigs(self, rfdiffusion):
        tool, storage, sent = rfdiffusion
        result = await tool.execute({"design_spec_ref": "design_spec_x", "contigs": "50-60"})
        assert "alone" in error_text(result)
        assert sent == {}

    async def test_an_mmcif_target_is_refused_with_the_way_out(self, rfdiffusion):
        tool, storage, sent = rfdiffusion
        ref = await _rfdiffusion_spec(
            storage,
            [
                {"condition": "hotspot", "selection_ref": "selection_epitope"},
                {"condition": "keep", "selection_ref": "selection_keep"},
            ],
            target="mmcif_target",
        )
        assert "reads PDB" in error_text(await tool.execute({"design_spec_ref": ref}))
        assert sent == {}

    async def test_an_rfd4_spec_is_pointed_at_the_rfd4_tool(self, rfdiffusion):
        tool, storage, sent = rfdiffusion
        creator = CreateDesignSpecTool()
        creator._storage = storage
        made = await creator.execute(
            {
                "target_ref": "mmcif_target",
                "engine": "rfd4",
                "layers": [{"condition": "hotspot", "selection_ref": "selection_epitope"}],
            }
        )
        result = await tool.execute({"design_spec_ref": made.metadata["design_spec_ref"]})
        assert "design_protein_binder" in error_text(result)
        assert sent == {}

    async def test_the_plain_contigs_call_is_unchanged(self, rfdiffusion):
        tool, _, sent = rfdiffusion
        result = await tool.execute({"contigs": "80-120"})
        assert result.content_type == ContentType.DATA_REFERENCE
        assert sent["contigs"] == "80-120" and "hotspot_res" not in sent
