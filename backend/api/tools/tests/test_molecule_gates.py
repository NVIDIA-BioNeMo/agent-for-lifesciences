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

"""Tests for the molecule_gates tool.

The service is stubbed throughout: RDKit is not installed in the devcontainer,
and the descriptor arithmetic is the rdkit suite's business anyway. What matters
here is that a failing molecule comes back with the violated bound named rather
than a bare verdict, that overridden thresholds actually reach the service, and
that a similarity gate cannot be requested in a way that silently does not run.
"""

import csv
import io

import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.base import ContentType
from api.tools.molecule_gates import (
    MAX_MOLECULES,
    MAX_RENDERED_ROWS,
    MoleculeGatesTool,
)

DEFAULT_THRESHOLDS = {
    "mw_max": 600.0,
    "logp_min": -1.0,
    "logp_max": 5.0,
    "tpsa_max": 140.0,
    "hbd_max": 5,
    "hba_max": 10,
    "rotatable_bonds_max": 10,
    "formal_charge_min": -2,
    "formal_charge_max": 2,
    "sa_score_max": 4.5,
    "pains_max": 0,
}


def molecule(
    smiles: str,
    passed: bool = True,
    failures: list[str] | None = None,
    properties: dict | None = None,
    similarity: dict | None = None,
    alerts: dict | None = None,
    gates: list[dict] | None = None,
) -> dict:
    """Build one result entry as the service would return it."""
    return {
        "smiles": smiles,
        "canonical_smiles": smiles,
        "properties": {
            "mw": 320.4,
            "logp": 3.1,
            "tpsa": 62.0,
            "hbd": 1,
            "hba": 4,
            "rotatable_bonds": 5,
            "formal_charge": 0,
            "sa_score": 2.4,
            "qed": 0.81,
            "num_rings": 3,
            "num_aromatic_rings": 2,
            "heavy_atoms": 23,
            "molecular_formula": "C18H20N2O3",
            **(properties or {}),
        },
        "alerts": {"pains": [], "brenk": [], "nih": [], **(alerts or {})},
        "similarity": {"murcko_scaffold": "c1ccccc1", **(similarity or {})},
        "gates": gates
        if gates is not None
        else [
            {
                "name": "mw",
                "label": "molecular weight",
                "value": 320.4,
                "bound": "<= 600",
                "passed": True,
                "detail": "",
            }
        ],
        "passed": passed,
        "num_failed": len(failures or []),
        "failures": failures or [],
    }


def response(
    results: list[dict],
    invalid: list[str] | None = None,
    unknown: list[str] | None = None,
    thresholds: dict | None = None,
    reference_smiles: str = "",
    reference_scaffold: str = "",
) -> dict:
    """Build a successful service response."""
    return {
        "success": True,
        "results": results,
        "invalid_smiles": invalid or [],
        "thresholds": thresholds if thresholds is not None else dict(DEFAULT_THRESHOLDS),
        "unknown_thresholds": unknown or [],
        "reference_smiles": reference_smiles,
        "reference_scaffold": reference_scaffold,
        "error": None,
    }


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.fixture
def tool(storage):
    instance = MoleculeGatesTool()
    instance._storage = storage
    return instance


def stub(tool, payload, calls: list | None = None):
    """Replace the tool's service call with a canned response."""

    async def _call(endpoint, method="POST", json_data=None, timeout=30.0, retries=0):
        if calls is not None:
            calls.append(json_data)
        if isinstance(payload, Exception):
            raise payload
        return payload

    tool._call_service = _call


class TestInputHandling:
    @pytest.mark.asyncio
    async def test_requires_some_molecules(self, tool):
        result = await tool.execute({})
        assert result.content_type == ContentType.ERROR
        assert "smiles_refs" in result.data

    @pytest.mark.asyncio
    async def test_scores_inline_smiles(self, tool):
        calls: list = []
        stub(tool, response([molecule("CCO")]), calls)

        result = await tool.execute({"smiles": ["CCO"]})

        assert result.content_type == ContentType.TEXT
        assert calls[0]["smiles"] == ["CCO"]

    @pytest.mark.asyncio
    async def test_accepts_a_bare_string_where_a_list_is_declared(self, tool):
        calls: list = []
        stub(tool, response([molecule("CCO")]), calls)

        await tool.execute({"smiles": "CCO"})

        assert calls[0]["smiles"] == ["CCO"]

    @pytest.mark.asyncio
    async def test_reads_a_multi_line_reference_as_many_molecules(
        self, tool, storage
    ):
        ref = await storage.store("smiles", b"CCO\nCCC\n\nc1ccccc1\n")
        calls: list = []
        stub(
            tool,
            response(
                [molecule("CCO"), molecule("CCC"), molecule("c1ccccc1")]
            ),
            calls,
        )

        result = await tool.execute({"smiles_refs": [ref]})

        assert calls[0]["smiles"] == ["CCO", "CCC", "c1ccccc1"]
        assert result.metadata["smiles_origin"]["CCO"] == ref

    @pytest.mark.asyncio
    async def test_deduplicates_across_both_input_forms(self, tool, storage):
        ref = await storage.store("smiles", b"CCO\nCCC\n")
        calls: list = []
        stub(tool, response([molecule("CCO"), molecule("CCC")]), calls)

        await tool.execute({"smiles": ["CCO"], "smiles_refs": [ref]})

        assert calls[0]["smiles"] == ["CCO", "CCC"]

    @pytest.mark.asyncio
    async def test_rejects_an_unknown_reference(self, tool):
        stub(tool, response([molecule("CCO")]))

        result = await tool.execute({"smiles_refs": ["smiles_deadbeef"]})

        assert result.content_type == ContentType.ERROR
        assert "smiles_deadbeef" in result.data

    @pytest.mark.asyncio
    async def test_refuses_an_oversized_batch_before_calling_the_service(self, tool):
        calls: list = []
        stub(tool, response([molecule("CCO")]), calls)

        result = await tool.execute(
            {"smiles": [f"C{'C' * index}O" for index in range(MAX_MOLECULES + 5)]}
        )

        assert result.content_type == ContentType.ERROR
        assert str(MAX_MOLECULES) in result.data
        assert calls == []

    @pytest.mark.asyncio
    async def test_refuses_references_without_storage(self, tool):
        tool._storage = None
        result = await tool.execute({"smiles_refs": ["smiles_abc12345"]})
        assert result.content_type == ContentType.ERROR
        assert "inline" in result.data


class TestThresholds:
    @pytest.mark.asyncio
    async def test_passes_overrides_through(self, tool):
        calls: list = []
        stub(tool, response([molecule("CCO")]), calls)

        await tool.execute(
            {"smiles": ["CCO"], "thresholds": {"mw_max": 500, "logp_max": 3}}
        )

        assert calls[0]["thresholds"] == {"mw_max": 500, "logp_max": 3}

    @pytest.mark.asyncio
    async def test_omits_thresholds_when_not_given(self, tool):
        calls: list = []
        stub(tool, response([molecule("CCO")]), calls)

        await tool.execute({"smiles": ["CCO"]})

        assert "thresholds" not in calls[0]

    @pytest.mark.asyncio
    async def test_rejects_a_non_object_threshold_argument(self, tool):
        calls: list = []
        stub(tool, response([molecule("CCO")]), calls)

        result = await tool.execute({"smiles": ["CCO"], "thresholds": "mw_max=500"})

        assert result.content_type == ContentType.ERROR
        assert "mw_max" in result.data
        assert calls == []

    @pytest.mark.asyncio
    async def test_reports_the_thresholds_that_were_applied(self, tool):
        # Without this the model cannot tell whether its override landed.
        stub(tool, response([molecule("CCO")]))

        result = await tool.execute({"smiles": ["CCO"]})

        assert "Thresholds applied:" in result.data
        assert "mw_max=600.0" in result.data

    @pytest.mark.asyncio
    async def test_reports_unrecognized_threshold_names(self, tool):
        stub(tool, response([molecule("CCO")], unknown=["mw_maximum"]))

        result = await tool.execute({"smiles": ["CCO"]})

        assert "mw_maximum" in result.data
        assert "Recognized" in result.data


class TestSingleMolecule:
    @pytest.mark.asyncio
    async def test_states_the_verdict_first(self, tool):
        stub(tool, response([molecule("CCO")]))

        result = await tool.execute({"smiles": ["CCO"]})

        assert result.data.splitlines()[0].startswith("CCO PASSES")

    @pytest.mark.asyncio
    async def test_names_every_violated_bound(self, tool):
        stub(
            tool,
            response(
                [
                    molecule(
                        "CCO",
                        passed=False,
                        failures=[
                            "molecular weight 641.2 violates <= 600",
                            "logP 6.1 violates -1 to 5",
                        ],
                        properties={"mw": 641.2, "logp": 6.1},
                    )
                ]
            ),
        )

        result = await tool.execute({"smiles": ["CCO"]})

        assert "FAILS" in result.data
        assert "641.2 violates <= 600" in result.data
        assert "logP 6.1 violates -1 to 5" in result.data

    @pytest.mark.asyncio
    async def test_lists_each_gate_with_its_limit(self, tool):
        stub(
            tool,
            response(
                [
                    molecule(
                        "CCO",
                        gates=[
                            {
                                "name": "sa_score",
                                "label": "synthetic accessibility",
                                "value": 5.2,
                                "bound": "<= 4.5",
                                "passed": False,
                                "detail": "",
                            }
                        ],
                        passed=False,
                        failures=["synthetic accessibility 5.2 violates <= 4.5"],
                    )
                ]
            ),
        )

        result = await tool.execute({"smiles": ["CCO"]})

        assert "[FAIL] synthetic accessibility: 5.2 (limit <= 4.5)" in result.data

    @pytest.mark.asyncio
    async def test_names_the_structural_alerts_it_tripped(self, tool):
        stub(
            tool,
            response(
                [
                    molecule(
                        "CCO",
                        alerts={"pains": ["quinone_A(370)"], "brenk": ["nitro group"]},
                        passed=False,
                        failures=["PAINS alerts 1 violates <= 0"],
                    )
                ]
            ),
        )

        result = await tool.execute({"smiles": ["CCO"]})

        assert "quinone_A(370)" in result.data
        assert "nitro group" in result.data

    @pytest.mark.asyncio
    async def test_reports_the_murcko_scaffold(self, tool):
        stub(
            tool,
            response([molecule("CCO", similarity={"murcko_scaffold": "c1ccncc1"})]),
        )

        result = await tool.execute({"smiles": ["CCO"]})

        assert "Murcko scaffold: c1ccncc1" in result.data


class TestBatch:
    @pytest.mark.asyncio
    async def test_leads_with_the_pass_count(self, tool):
        stub(
            tool,
            response(
                [
                    molecule("CCO"),
                    molecule("CCC", passed=False, failures=["logP 6.1 violates -1 to 5"]),
                ]
            ),
        )

        result = await tool.execute({"smiles": ["CCO", "CCC"]})

        assert result.data.startswith("1 of 2 molecules pass every gate.")

    @pytest.mark.asyncio
    async def test_separates_passing_from_failing_with_reasons(self, tool):
        stub(
            tool,
            response(
                [
                    molecule("CCO"),
                    molecule(
                        "CCCCCCCC",
                        passed=False,
                        failures=["logP 6.1 violates -1 to 5"],
                    ),
                ]
            ),
        )

        result = await tool.execute({"smiles": ["CCO", "CCCCCCCC"]})

        assert "PASSING:" in result.data
        assert "FAILING:" in result.data
        assert "CCCCCCCC -- logP 6.1 violates -1 to 5" in result.data

    @pytest.mark.asyncio
    async def test_renders_a_comparable_table(self, tool):
        stub(tool, response([molecule("CCO"), molecule("CCC")]))

        result = await tool.execute({"smiles": ["CCO", "CCC"]})

        assert "molecule" in result.data
        for header in ("MW", "logP", "TPSA", "HBD", "HBA", "RB", "SA", "QED"):
            assert header in result.data

    @pytest.mark.asyncio
    async def test_omits_similarity_columns_without_a_reference(self, tool):
        stub(tool, response([molecule("CCO"), molecule("CCC")]))

        result = await tool.execute({"smiles": ["CCO", "CCC"]})

        assert "scafTan" not in result.data

    @pytest.mark.asyncio
    async def test_shows_similarity_columns_with_a_reference(self, tool):
        stub(
            tool,
            response(
                [
                    molecule(
                        "CCO",
                        similarity={
                            "tanimoto": 0.42,
                            "scaffold_tanimoto": 0.31,
                            "scaffold_mcs_ratio": 0.55,
                        },
                    )
                ]
                * 2,
                reference_smiles="c1ccccc1O",
                reference_scaffold="c1ccccc1",
            ),
        )

        result = await tool.execute(
            {"smiles": ["CCO", "CCC"], "reference_smiles": "c1ccccc1O"}
        )

        assert "scafTan" in result.data
        assert "Reference: c1ccccc1O" in result.data
        assert "Reference scaffold: c1ccccc1" in result.data

    @pytest.mark.asyncio
    async def test_caps_the_rendered_rows_and_says_so(self, tool):
        many = [molecule(f"C{'C' * index}O") for index in range(MAX_RENDERED_ROWS + 5)]
        stub(tool, response(many))

        result = await tool.execute({"smiles": [entry["smiles"] for entry in many]})

        assert f"first {MAX_RENDERED_ROWS} of {len(many)} rows" in result.data
        assert "aevom.load(" in result.data


class TestStoredResults:
    @pytest.mark.asyncio
    async def test_stores_every_row_as_csv(self, tool, storage):
        stub(
            tool,
            response(
                [
                    molecule("CCO"),
                    molecule(
                        "CCC",
                        passed=False,
                        failures=["logP 6.1 violates -1 to 5"],
                        alerts={"brenk": ["nitro group"]},
                    ),
                ]
            ),
        )

        result = await tool.execute({"smiles": ["CCO", "CCC"]})

        ref = result.metadata["csv_ref"]
        rows = list(csv.DictReader(io.StringIO((await storage.retrieve(ref)).decode())))
        assert [row["smiles"] for row in rows] == ["CCO", "CCC"]
        assert rows[0]["passed"] == "True"
        assert rows[1]["failures"] == "logP 6.1 violates -1 to 5"
        assert rows[1]["brenk"] == "1"

    @pytest.mark.asyncio
    async def test_renders_without_storage(self, tool):
        # A missing store should cost the CSV, not the answer.
        tool._storage = None
        stub(tool, response([molecule("CCO")]))

        result = await tool.execute({"smiles": ["CCO"]})

        assert result.content_type == ContentType.TEXT
        assert "csv_ref" not in result.metadata

    @pytest.mark.asyncio
    async def test_metadata_carries_the_per_gate_verdict(self, tool):
        # The candidate ledger reads this instead of re-parsing the table.
        stub(
            tool,
            response(
                [
                    molecule("CCO"),
                    molecule("CCC", passed=False, failures=["logP 6.1 violates -1 to 5"]),
                ]
            ),
        )

        result = await tool.execute({"smiles": ["CCO", "CCC"]})

        assert result.metadata["num_passed"] == 1
        assert result.metadata["passing_smiles"] == ["CCO"]
        assert result.metadata["verdicts"]["CCC"]["failures"] == [
            "logP 6.1 violates -1 to 5"
        ]
        assert result.metadata["thresholds"]["mw_max"] == 600.0


class TestFailures:
    @pytest.mark.asyncio
    async def test_surfaces_a_service_error(self, tool):
        stub(tool, {"success": False, "error": "similarity_min requires reference_smiles"})

        result = await tool.execute({"smiles": ["CCO"]})

        assert result.content_type == ContentType.ERROR
        assert "requires reference_smiles" in result.data

    @pytest.mark.asyncio
    async def test_surfaces_a_transport_failure(self, tool):
        stub(tool, RuntimeError("connection refused"))

        result = await tool.execute({"smiles": ["CCO"]})

        assert result.content_type == ContentType.ERROR
        assert "connection refused" in result.data

    @pytest.mark.asyncio
    async def test_names_the_unparseable_inputs(self, tool):
        stub(tool, response([molecule("CCO")], invalid=["not-a-molecule"]))

        result = await tool.execute({"smiles": ["CCO", "not-a-molecule"]})

        assert "not-a-molecule" in result.data
        assert result.metadata["invalid_smiles"] == ["not-a-molecule"]

    @pytest.mark.asyncio
    async def test_errors_when_nothing_could_be_scored(self, tool):
        stub(tool, response([], invalid=["nope"]))

        result = await tool.execute({"smiles": ["nope"]})

        assert result.content_type == ContentType.ERROR
        assert "nope" in result.data


class TestSchema:
    def test_documents_the_defaults_and_the_override_names(self, tool):
        description = tool.get_schema()["function"]["description"]
        assert "MW <= 600" in description
        assert "SA score <= 4.5" in description
        assert "similarity_min" in description

    def test_declares_the_reference_annotation_on_smiles_refs(self, tool):
        params = tool.get_schema()["function"]["parameters"]["properties"]
        assert params["smiles_refs"]["x-aevom-reference"] == {"data_type": "smiles"}


class TestTheReferenceCanBeAHandle:
    """Re-typing the parent is the one input here that can be wrong silently.

    `reference_smiles` was free text, so the agent transcribed the reference
    ligand from an SDF by hand. The transcription parses -- this is not the
    unparseable-SMILES problem -- it is simply a different molecule: a
    regio-isomer, an extra methyl, a lost carbonyl. Every similarity number in
    the call is then measured against a molecule that is not the parent, and the
    run reports a passing Tanimoto that the scorer will not reproduce. Measured
    in 5 of 120 benchmark runs, 3 of which submitted an answer that fails the
    similarity gate against the real reference.
    """

    @pytest.mark.asyncio
    async def test_a_stored_reference_is_used(self, tool, storage):
        calls: list = []
        stub(tool, response([molecule("c1ccccc1C")]), calls)
        ref = await storage.store(data_type="smiles", content=b"c1ccccc1O")

        await tool.execute({"smiles": ["c1ccccc1C"], "reference_ref": ref})

        assert calls[-1]["reference_smiles"] == "c1ccccc1O"

    @pytest.mark.asyncio
    async def test_the_stored_one_wins_and_the_disagreement_is_reported(
        self, tool, storage
    ):
        calls: list = []
        stub(tool, response([molecule("c1ccccc1C")]), calls)
        ref = await storage.store(data_type="smiles", content=b"c1ccccc1O")

        result = await tool.execute({
            "smiles": ["c1ccccc1C"],
            "reference_ref": ref,
            "reference_smiles": "c1ccccc1N",      # a different molecule
        })

        assert calls[-1]["reference_smiles"] == "c1ccccc1O"
        assert "not the same molecule" in result.data
        assert "c1ccccc1N" in result.data

    @pytest.mark.asyncio
    async def test_agreement_is_not_worth_a_paragraph(self, tool, storage):
        stub(tool, response([molecule("c1ccccc1C")]))
        ref = await storage.store(data_type="smiles", content=b"c1ccccc1O")

        result = await tool.execute({
            "smiles": ["c1ccccc1C"],
            "reference_ref": ref,
            "reference_smiles": "c1ccccc1O",
        })

        assert "not the same molecule" not in result.data

    @pytest.mark.asyncio
    async def test_a_missing_reference_says_so(self, tool):
        stub(tool, response([molecule("c1ccccc1C")]))

        result = await tool.execute(
            {"smiles": ["c1ccccc1C"], "reference_ref": "smiles_deadbeef"}
        )

        assert result.content_type == ContentType.ERROR
        assert "smiles_deadbeef" in result.data

    def test_the_schema_offers_the_handle(self):
        properties = MoleculeGatesTool().get_schema()["function"]["parameters"][
            "properties"
        ]

        assert "reference_ref" in properties
        assert properties["reference_ref"]["x-aevom-reference"]["data_type"] == [
            "smiles",
            "sdf",
        ]
