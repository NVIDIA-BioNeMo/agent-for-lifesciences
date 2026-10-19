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

"""Binder design: selection in, condition spec out, designs stored.

The request shape asserted here is not invented. It was POSTed to a live
RFD4-Proteina NIM, which accepted it through schema validation, contig
parsing, selection resolution and condition application -- the local GPU is
too old to run the model, so the request died in CUDA, which is precisely the
signal that everything above CUDA was right.

The response fixture mirrors the NIM's documented ``DesignResponse``.
"""

import json

import pytest
import pytest_asyncio

from api.storage.mmcif_numbering import build_numbering_map
from api.tools.base import ContentType
from api.tools.rfd4_proteina import (
    DEFAULT_BINDER_LENGTH_MAX,
    DEFAULT_BINDER_LENGTH_MIN,
    DesignProteinBinderTool,
    advice_for_error,
    build_condition_spec,
    build_contig,
    build_hotspot_spec,
    read_design_chains,
)

# Author numbering runs 16 ahead of label numbering, as in 5O45.
TARGET_CIF = """\
data_TARGET
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
ATOM   1 N CA . ASN A 1 1   ? 1.000 1.000 1.000 17  ASN A CA
ATOM   2 N CA . TYR A 1 40  ? 2.000 2.000 1.000 56  TYR A CA
ATOM   3 C CG . TYR A 1 40  ? 2.000 2.000 2.000 56  TYR A CG
ATOM   4 O OH . TYR A 1 40  ? 2.000 2.000 3.000 56  TYR A OH
ATOM   5 N CA . MET A 1 99  ? 3.000 3.000 1.000 115 MET A CA
ATOM   6 S SD . MET A 1 99  ? 3.000 3.000 2.000 115 MET A SD
ATOM   7 N CA . ASN A 1 115 ? 4.000 4.000 1.000 131 ASN A CA
HETATM 8 O O  . HOH B 2 .   ? 9.000 9.000 9.000 301 HOH A O
#
"""


def atom(name, comp, seq, chain="A"):
    return {
        "name": name,
        "element": name[0],
        "id": 1,
        "coords": [0.0, 0.0, 0.0],
        "residue": {
            "name": comp,
            "seqNumber": seq,
            "chainId": chain,
            "insCode": "",
            "entityType": "polymer",
        },
    }


EPITOPE_SELECTION = {
    "query_id": "q1",
    "molql": "(sel.atom.res ...)",
    "atoms": [
        atom("CG", "TYR", 56),
        atom("OH", "TYR", 56),
        atom("SD", "MET", 115),
    ],
    "atom_count": 3,
    "residue_count": 2,
}

_DESIGN_COLUMNS = (
    "group_PDB",
    "type_symbol",
    "label_atom_id",
    "label_alt_id",
    "label_comp_id",
    "label_asym_id",
    "label_entity_id",
    "label_seq_id",
    "Cartn_x",
    "Cartn_y",
    "Cartn_z",
    "mask_hotspot_1_atom",
)


def design_cif(name, binder_residues, target_residues):
    """A design as the NIM returns one: binder on a fresh chain, target kept.

    The contig preserves the target's chain id (A) and gives the run with no
    preserved segment a fresh one (B), so the two are told apart by chain.
    """
    header = [f"data_{name}", "#", "loop_"]
    header += [f"_atom_site.{column}" for column in _DESIGN_COLUMNS]
    rows = [
        f"ATOM N CA . {comp} {chain} 1 {seq} 1.0 1.0 1.0 False"
        for chain, count, cycle in (
            ("B", binder_residues, BINDER_RESIDUES_CYCLE),
            ("A", target_residues, TARGET_RESIDUES_CYCLE),
        )
        for seq, comp in zip(
            range(1, count + 1),
            (cycle[i % len(cycle)] for i in range(count)),
        )
    ]
    return "\n".join(header + rows + ["#", ""])


# The two chains use disjoint residues so a sequence can be told to be the
# binder's or the target's by reading it.
BINDER_RESIDUES_CYCLE = ("ALA", "LEU", "SER")
TARGET_RESIDUES_CYCLE = ("TYR", "MET")


def expected_chain_sequence(count, cycle):
    """The one-letter sequence design_cif writes for a chain of this size."""
    letters = {"ALA": "A", "LEU": "L", "SER": "S", "TYR": "Y", "MET": "M"}
    return "".join(letters[cycle[i % len(cycle)]] for i in range(count))


def sequence_of(length):
    """The NIM's string for a design: binder first, then the target.

    Binder first even though the binder sits on chain B and the target on A:
    the string follows the contig, not the chain letters.
    """
    binder = length - TARGET_RESIDUES
    return expected_chain_sequence(
        binder, BINDER_RESIDUES_CYCLE
    ) + expected_chain_sequence(TARGET_RESIDUES, TARGET_RESIDUES_CYCLE)


# One design, shaped like the NIM's DesignResponse. Each design is the binder
# plus the whole target, which is what makes the complex longer than the
# requested binder length.
BINDER_0_BINDER_RESIDUES = 12
BINDER_1_BINDER_RESIDUES = 30
TARGET_RESIDUES = 20
DESIGN_RESPONSE = {
    "request_id": "req-1",
    "task": "unconditional",
    "model": {"name": "BakerLab/rfd4-proteina"},
    "runtime": {"backend": "trt"},
    "schedule": {"num_steps": 20},
    "designs": [
        {
            "design_id": "binder_0",
            "replicate_index": 0,
            "origin": "sample",
            "sequence": sequence_of(BINDER_0_BINDER_RESIDUES + TARGET_RESIDUES),
            "structure": {
                "format": "mmcif",
                "filename": "binder_0.cif",
                "content": design_cif(
                    "BINDER0", BINDER_0_BINDER_RESIDUES, TARGET_RESIDUES
                ),
            },
            "metrics": {"hotspot_obedience.hotspot_obedience": 0.67},
            "diagnostics": {},
        },
        {
            "design_id": "binder_1",
            "replicate_index": 1,
            "origin": "sample",
            "sequence": sequence_of(BINDER_1_BINDER_RESIDUES + TARGET_RESIDUES),
            "structure": {
                "format": "mmcif",
                "filename": "binder_1.cif",
                "content": design_cif(
                    "BINDER1", BINDER_1_BINDER_RESIDUES, TARGET_RESIDUES
                ),
            },
            "metrics": {"hotspot_obedience.hotspot_obedience": 1.0},
            "diagnostics": {},
        },
    ],
    "aggregated_metrics": [
        {"metric": "unconditional.pass", "reduction": "mean", "value": 0.5, "n": 2}
    ],
    "warnings": ["one design was dropped by a filter"],
    "duration_ms": 1234.0,
}


class FakeStorage:
    """Records what a tool stored, and hands back references for it."""

    def __init__(self, artifacts=None):
        self.artifacts = dict(artifacts or {})
        self.stored = []
        self._next = 0

    async def retrieve(self, reference):
        from api.storage import ArtifactNotFoundError

        if reference not in self.artifacts:
            raise ArtifactNotFoundError(reference)
        return self.artifacts[reference]

    async def store(self, data_type, content, metadata=None, description=None):
        self._next += 1
        reference = f"{data_type}_{self._next:03d}"
        self.stored.append(
            {
                "reference": reference,
                "data_type": data_type,
                "content": content,
                "metadata": metadata or {},
                "description": description,
            }
        )
        return reference


def error_text(result) -> str:
    """Assert a result is an error and hand back its message."""
    assert result.content_type == ContentType.ERROR, result.data
    return str(result.data)


def reply_text(result) -> str:
    """The text the model actually receives for a stored-data result."""
    return result.to_tool_response()


@pytest.fixture
def numbering():
    return build_numbering_map(TARGET_CIF)


@pytest.fixture
def tool(monkeypatch):
    """A tool bound to a fake endpoint, with storage attached."""
    instance = DesignProteinBinderTool(
        endpoint_url="http://localhost:8000/biology/ipd/rfd4-proteina/generative",
        api_key=None,
        metric_tags=["design", "hotspot_conditioning"],
        server_name="Local RFD4",
    )
    instance._storage = FakeStorage(
        {
            "mmcif_target": TARGET_CIF.encode(),
            "selection_epitope": json.dumps(EPITOPE_SELECTION).encode(),
            "pdb_target": b"ATOM      1  CA  ALA A   1       0.000   0.000   0.000\n",
            "selection_garbage": b"not json",
        }
    )
    return instance


class TestHotspotSelectors:
    def test_hotspots_are_emitted_in_label_numbering(self, numbering):
        spec = build_hotspot_spec(EPITOPE_SELECTION, numbering)
        assert spec.selectors == [
            "sel('A/*/40/[CG,OH]')",
            "sel('A/*/99/SD')",
        ]

    def test_translation_is_reported_to_the_user(self, numbering):
        # A silent renumbering is indistinguishable from a bug, so the tool
        # has to say it happened.
        spec = build_hotspot_spec(EPITOPE_SELECTION, numbering)
        assert any("A56" in note and "A40" in note for note in spec.notes)

    def test_non_polymer_atoms_are_dropped_with_a_reason(self, numbering):
        selection = {"atoms": [atom("O", "HOH", 301)]}
        spec = build_hotspot_spec(selection, numbering)
        assert spec.selectors == []
        assert any("not" in note for note in spec.notes)

    def test_residue_absent_from_the_target_is_dropped(self, numbering):
        # The common user error: a selection made on a different structure.
        selection = {"atoms": [atom("CA", "GLY", 9999)]}
        spec = build_hotspot_spec(selection, numbering)
        assert spec.selectors == []
        assert any("9999" in note for note in spec.notes)

    def test_whole_residue_selection_omits_the_atom_list(self, numbering):
        selection = {
            "atoms": [atom(f"C{i}", "TYR", 56) for i in range(12)]
        }
        spec = build_hotspot_spec(selection, numbering)
        assert spec.selectors == ["sel('A/*/40')"]

    def test_empty_selection_is_reported_not_crashed(self, numbering):
        spec = build_hotspot_spec({"atoms": []}, numbering)
        assert spec.selectors == []
        assert spec.notes


def viewer_atom(name, comp, auth_seq, label_seq, auth_chain="A", label_chain="A"):
    """An atom as the viewer emits it: label ids in seqNumber/chainId, and the
    author ids beside them in authSeqNumber/authChainId."""
    return {
        "name": name,
        "element": name[0],
        "id": 1,
        "coords": [0.0, 0.0, 0.0],
        "residue": {
            "name": comp,
            "seqNumber": label_seq,
            "chainId": label_chain,
            "authSeqNumber": auth_seq,
            "authChainId": auth_chain,
            "insCode": "",
            "entityType": "polymer",
        },
    }


class TestViewerNumbering:
    """The viewer reports label numbering in seqNumber and author numbering
    beside it. Before the two were separated, ``build_hotspot_spec`` read the
    label number as an author one: on 5O45 (author = label + 16) a pick of
    author 56 arrived as 40, was looked up as *author* 40, and silently
    conditioned label 24 -- a different residue -- with no error anywhere."""

    def test_pick_of_author_56_conditions_label_40(self, numbering):
        selection = {"atoms": [viewer_atom("CG", "TYR", auth_seq=56, label_seq=40)]}
        spec = build_hotspot_spec(selection, numbering)
        assert spec.selectors == ["sel('A/*/40/CG')"]

    def test_legacy_payload_without_author_fields_is_read_as_author(self, numbering):
        selection = {"atoms": [atom("CG", "TYR", 56)]}
        spec = build_hotspot_spec(selection, numbering)
        assert spec.selectors == ["sel('A/*/40/CG')"]

    def test_author_fields_win_over_the_label_fields(self, numbering):
        # Same residue, but seqNumber carries an unrelated label number (a PDB
        # copy of the structure, where label ids are synthesised). The author
        # number is stable across formats, so it decides, and the user is told
        # the two copies disagree.
        selection = {"atoms": [viewer_atom("CG", "TYR", auth_seq=56, label_seq=56)]}
        spec = build_hotspot_spec(selection, numbering)
        assert spec.selectors == ["sel('A/*/40/CG')"]
        assert any("different copy" in note for note in spec.notes)

    def test_matching_label_numbers_raise_no_disagreement_note(self, numbering):
        selection = {"atoms": [viewer_atom("CG", "TYR", auth_seq=56, label_seq=40)]}
        spec = build_hotspot_spec(selection, numbering)
        assert not any("different copy" in note for note in spec.notes)


class TestContig:
    def test_binder_first_then_target_span_in_label_numbering(self, numbering):
        assert build_contig(numbering, ["A"], 50, 120) == "50-120, /0, A1-115"

    def test_fixed_length_binder_is_a_single_number(self, numbering):
        assert build_contig(numbering, ["A"], 80, 80) == "80, /0, A1-115"


class TestConditionSpec:
    @pytest.fixture
    def request_body(self, numbering):
        spec = build_hotspot_spec(EPITOPE_SELECTION, numbering)
        return build_condition_spec(
            example_id="pdl1_binder",
            contig=build_contig(numbering, spec.chains, 50, 120),
            target_chains=spec.chains,
            hotspot_selectors=spec.selectors,
            target_cif=TARGET_CIF,
            num_designs=2,
            num_steps=20,
            metric_tags=["design", "hotspot_conditioning"],
        )

    def test_binder_design_is_the_unconditional_task(self, request_body):
        # Counter-intuitive but correct: the target is held by conditions,
        # not by the task.
        assert request_body["task"] == "unconditional"

    def test_target_backbone_is_held_but_side_chains_are_free(self, request_body):
        selector = request_body["input"]["spec"]["conditions"]["C_CRD"]["true"][0]
        assert "is_protein_backbone" in selector["select"]

    def test_target_sequence_is_held(self, request_body):
        # Without C_SEQ the model redesigns the thing we are binding to.
        assert request_body["input"]["spec"]["conditions"]["C_SEQ"]["true"] == [
            {"select": "sel('A')"}
        ]

    def test_hotspots_become_one_entry_each(self, request_body):
        hotspots = request_body["input"]["spec"]["conditions"]["C_HOT"]["true"]
        assert [entry["select"] for entry in hotspots] == [
            "sel('A/*/40/[CG,OH]')",
            "sel('A/*/99/SD')",
        ]

    def test_structure_travels_as_an_inline_file_the_spec_names(self, request_body):
        files = request_body["input"]["files"]
        assert files[0]["filename"] == request_body["input"]["spec"]["input"]
        assert files[0]["content"] == TARGET_CIF

    def test_sampling_carries_the_design_count(self, request_body):
        assert request_body["sampling"]["num_replicates"] == 2

    def test_metric_tags_are_passed_through(self, request_body):
        assert request_body["metrics"]["tags"] == ["design", "hotspot_conditioning"]

    def test_no_metrics_block_when_nothing_is_available(self, numbering):
        # A deployment reporting no available tags must get no tags, since an
        # unavailable tag is a request error rather than a skipped metric.
        body = build_condition_spec(
            example_id="x",
            contig="50, /0, A1-115",
            target_chains=["A"],
            hotspot_selectors=["sel('A/*/40')"],
            target_cif=TARGET_CIF,
            num_designs=1,
            metric_tags=[],
        )
        assert "metrics" not in body


@pytest.mark.asyncio
class TestExecute:
    async def test_rejects_pdb_input(self, tool):
        result = await tool.execute(
            {"target_ref": "pdb_target", "hotspot_selection_ref": "selection_epitope"}
        )
        assert "mmCIF" in error_text(result)

    async def test_requires_a_selection(self, tool):
        result = await tool.execute({"target_ref": "mmcif_target"})
        assert "structure_query" in error_text(result)

    async def test_rejects_a_non_selection_artifact(self, tool):
        result = await tool.execute(
            {"target_ref": "mmcif_target", "hotspot_selection_ref": "selection_garbage"}
        )
        assert error_text(result)

    async def test_rejects_an_inverted_length_range(self, tool):
        result = await tool.execute(
            {
                "target_ref": "mmcif_target",
                "hotspot_selection_ref": "selection_epitope",
                "binder_length_min": 200,
                "binder_length_max": 50,
            }
        )
        assert "cannot exceed" in error_text(result)

    async def test_rejects_too_few_diffusion_steps(self, tool):
        # The checkpoint refuses flat sampling below 20 steps. Catching it
        # here turns a round trip and a 422 into an immediate answer.
        result = await tool.execute(
            {
                "target_ref": "mmcif_target",
                "hotspot_selection_ref": "selection_epitope",
                "num_steps": 1,
            }
        )
        assert "20" in error_text(result)

    async def test_sends_the_spec_the_live_nim_accepted(self, tool, monkeypatch):
        sent = {}

        async def fake_call(self, json_data, **kwargs):
            sent.update(json_data)
            return DESIGN_RESPONSE

        monkeypatch.setattr(DesignProteinBinderTool, "_call_nvidia_api", fake_call)

        await tool.execute(
            {
                "target_ref": "mmcif_target",
                "hotspot_selection_ref": "selection_epitope",
                "binder_length_min": 50,
                "binder_length_max": 120,
            }
        )

        assert sent["task"] == "unconditional"
        assert sent["input"]["spec"]["contig"] == "50-120, /0, A1-115"
        assert sent["input"]["spec"]["conditions"]["C_HOT"]["true"][0]["select"] == (
            "sel('A/*/40/[CG,OH]')"
        )


@pytest_asyncio.fixture
async def result(tool, monkeypatch):
    """One completed run against a stubbed NIM, with no range requested."""

    async def fake_call(self, json_data, **kwargs):
        return DESIGN_RESPONSE

    monkeypatch.setattr(DesignProteinBinderTool, "_call_nvidia_api", fake_call)
    return await tool.execute(
        {
            "target_ref": "mmcif_target",
            "hotspot_selection_ref": "selection_epitope",
            "description": "PD-L1 binder",
        }
    )


@pytest.mark.asyncio
class TestResponseHandling:
    async def test_each_design_is_stored_as_an_annotated_cif(self, tool, result):
        kinds = [item["data_type"] for item in tool._storage.stored]
        assert kinds.count("annotated_cif") == 2

    async def test_each_design_sequence_is_stored(self, tool, result):
        kinds = [item["data_type"] for item in tool._storage.stored]
        assert kinds.count("sequence") == 2

    async def test_metrics_are_stored_as_json(self, tool, result):
        metrics = [i for i in tool._storage.stored if i["data_type"] == "json"]
        assert len(metrics) == 1
        payload = json.loads(metrics[0]["content"])
        assert payload["aggregated_metrics"][0]["metric"] == "unconditional.pass"

    async def test_designs_are_ranked_by_hotspot_obedience(self, tool, result):
        # binder_1 scores 1.0 against binder_0's 0.67, so it leads.
        first = tool._storage.stored[0]
        assert first["metadata"]["design_id"] == "binder_1"

    async def test_the_reply_names_the_best_design_first(self, result):
        body = reply_text(result)
        assert body.index("binder_1") < body.index("binder_0")

    async def test_the_reply_carries_no_structure_text(self, result):
        # Returning mmCIF to the model would spend the context window on
        # coordinates it cannot use.
        assert "data_BINDER0" not in reply_text(result)

    async def test_in_silico_success_rate_is_surfaced(self, result):
        assert "unconditional.pass" in reply_text(result)

    async def test_nim_warnings_reach_the_user(self, result):
        assert "dropped by a filter" in reply_text(result)

    async def test_numbering_translation_is_recorded_on_the_artifact(self, tool, result):
        design = next(
            i for i in tool._storage.stored if i["data_type"] == "annotated_cif"
        )
        assert design["metadata"]["numbering_translated"] is True

    async def test_empty_design_list_is_an_error_not_a_success(self, tool, monkeypatch):
        async def fake_call(self, json_data, **kwargs):
            return {**DESIGN_RESPONSE, "designs": [], "warnings": ["nothing survived"]}

        monkeypatch.setattr(DesignProteinBinderTool, "_call_nvidia_api", fake_call)
        result = await tool.execute(
            {
                "target_ref": "mmcif_target",
                "hotspot_selection_ref": "selection_epitope",
            }
        )
        assert "nothing survived" in error_text(result)


class TestReportedLengths:
    """What the model is told about how long a design is.

    Upstream's rule is that the requested range bounds the designed segment
    alone: "the length counts only what gets designed". The target rides along
    in the output, so the complex is longer, and reporting one number for both
    had the model telling users their binder was twice the size they asked for.
    """

    def test_binder_and_target_are_counted_separately(self):
        chains = read_design_chains(design_cif("D", 96, 100), target_chains=["A"])
        assert (chains.binder_length, chains.target_length) == (96, 100)

    def test_the_binder_is_the_chain_the_target_is_not(self):
        # Binder written first, target kept as A, so the binder is chain B.
        # Reading the first chain written as "chain A" is the mistake.
        chains = read_design_chains(design_cif("D", 12, 20), target_chains=["A"])
        assert list(chains.binder) == ["B"]
        assert list(chains.target) == ["A"]

    def test_each_chain_reads_back_as_its_own_sequence(self):
        chains = read_design_chains(design_cif("D", 12, 20), target_chains=["A"])
        assert chains.binder_sequence == expected_chain_sequence(
            12, BINDER_RESIDUES_CYCLE
        )
        assert chains.target["A"] == expected_chain_sequence(
            20, TARGET_RESIDUES_CYCLE
        )

    def test_the_description_names_which_chain_is_which(self):
        chains = read_design_chains(design_cif("D", 32, 100), target_chains=["A"])
        assert chains.describe() == (
            "binder on chain B (32 residues), target on chain A (100 residues)"
        )

    def test_an_unrecognised_residue_is_x_not_dropped(self):
        cif = design_cif("D", 3, 2).replace("ALA", "UNK", 1)
        assert read_design_chains(cif, ["A"]).binder_sequence.startswith("X")

    def test_a_file_without_an_atom_site_loop_has_no_chains(self):
        chains = read_design_chains("data_EMPTY\n#\n", target_chains=["A"])
        assert (chains.binder_length, chains.target_length) == (0, 0)

    @pytest.mark.asyncio
    async def test_the_stored_sequence_is_the_binder_alone(self, tool, result):
        # The NIM's string is binder + target. A tool handed this reference to
        # fold or score must get the binder, not a 50-residue chimera.
        sequences = [i for i in tool._storage.stored if i["data_type"] == "sequence"]
        best = next(s for s in sequences if s["metadata"]["design_id"] == "binder_1")
        assert best["content"].decode() == expected_chain_sequence(
            BINDER_1_BINDER_RESIDUES, BINDER_RESIDUES_CYCLE
        )
        assert best["metadata"]["scope"] == "binder"
        assert best["metadata"]["chains"] == ["B"]

    @pytest.mark.asyncio
    async def test_the_sequence_reference_says_it_excludes_the_target(
        self, tool, result
    ):
        sequences = [i for i in tool._storage.stored if i["data_type"] == "sequence"]
        assert "target excluded" in sequences[0]["description"]

    @pytest.mark.asyncio
    async def test_an_unreadable_structure_labels_the_sequence_as_the_complex(
        self, tool, monkeypatch
    ):
        # Without a readable structure the chain boundary is unknown, so the
        # NIM's string is kept but must not pass as the binder.
        broken = {
            **DESIGN_RESPONSE,
            "designs": [
                {
                    **DESIGN_RESPONSE["designs"][0],
                    "structure": {
                        "format": "mmcif",
                        "filename": "x.cif",
                        "content": "data_BROKEN\n#\n",
                    },
                }
            ],
        }

        async def fake_call(self, json_data, **kwargs):
            return broken

        monkeypatch.setattr(DesignProteinBinderTool, "_call_nvidia_api", fake_call)
        await tool.execute(
            {
                "target_ref": "mmcif_target",
                "hotspot_selection_ref": "selection_epitope",
            }
        )

        sequence = next(
            i for i in tool._storage.stored if i["data_type"] == "sequence"
        )
        assert sequence["metadata"]["scope"] == "complex"
        assert "chain boundary unknown" in sequence["description"]

    @pytest.mark.asyncio
    async def test_the_reply_says_the_binder_is_not_chain_a(self, result):
        body = reply_text(result)
        assert "The binder is NOT chain A" in body
        assert "binder on chain B" in body
        assert "target on chain A" in body

    @pytest.mark.asyncio
    async def test_the_reply_gives_the_binder_length_not_the_complex(self, result):
        body = reply_text(result)
        assert f"{BINDER_1_BINDER_RESIDUES}-residue binder" in body
        assert (
            f"{BINDER_1_BINDER_RESIDUES + TARGET_RESIDUES} residues with the target"
            in body
        )

    @pytest.mark.asyncio
    async def test_the_binder_length_is_recorded_on_the_artifact(self, tool, result):
        design = next(
            i for i in tool._storage.stored if i["data_type"] == "annotated_cif"
        )
        assert design["metadata"]["binder_length"] == BINDER_1_BINDER_RESIDUES
        assert design["metadata"]["target_length"] == TARGET_RESIDUES

    @pytest.mark.asyncio
    async def test_a_defaulted_range_says_so(self, result):
        # The range reaching the NIM is the only record of what was asked for:
        # _validate_lengths cannot tell an omitted value from an explicit one.
        assert (
            f"{DEFAULT_BINDER_LENGTH_MIN}-{DEFAULT_BINDER_LENGTH_MAX} residues "
            "(tool default; none was given)"
        ) in reply_text(result)

    @pytest.mark.asyncio
    async def test_a_requested_range_is_echoed_back(self, tool, monkeypatch):
        async def fake_call(self, json_data, **kwargs):
            return DESIGN_RESPONSE

        monkeypatch.setattr(DesignProteinBinderTool, "_call_nvidia_api", fake_call)
        result = await tool.execute(
            {
                "target_ref": "mmcif_target",
                "hotspot_selection_ref": "selection_epitope",
                "binder_length_min": 80,
                "binder_length_max": 100,
            }
        )

        body = reply_text(result)
        assert "Binder length requested: 80-100 residues" in body
        assert "tool default" not in body


class TestErrorAdvice:
    def test_out_of_memory_suggests_shrinking_the_job(self):
        advice = advice_for_error("Local RFD4 rejected the request. [out_of_memory] no")
        assert "fewer designs" in advice

    def test_cuda_failure_points_at_the_gpu(self):
        # The exact failure on this host: an old card the model cannot use.
        advice = advice_for_error("[internal_error] AcceleratorError: CUDA error")
        assert "A100" in advice or "too old" in advice

    def test_unknown_code_is_passed_through_unchanged(self):
        assert advice_for_error("something else") == "something else"
