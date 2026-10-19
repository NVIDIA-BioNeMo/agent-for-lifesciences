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

"""Tests for the admet_predict tool.

The service is stubbed throughout -- loading the real ensemble takes tens of
seconds and a GPU-sized image, and none of what is interesting here lives in the
model. What is interesting is the presentation: a table the model can compare
analogues in, a CSV it can rank in python_execute, and enough labelling that a
number cannot be read in the wrong direction. Plus the two ways a batch call goes
subtly wrong -- an unparseable SMILES silently dropped, and duplicates padding
the table.
"""

import csv
import io

import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.admet import MAX_MOLECULES, MAX_RENDERED_ROWS, AdmetPredictTool
from api.tools.base import ContentType

PROPERTY_INFO = [
    {
        "name": "hERG",
        "known": True,
        "label": "hERG blocking",
        "abbrev": "hERG",
        "category": "Toxicity",
        "task_type": "classification",
        "units": "",
        "direction": "lower",
        "note": "Cardiac liability.",
    },
    {
        "name": "BBB_Martins",
        "known": True,
        "label": "Blood-brain barrier penetration",
        "abbrev": "BBB",
        "category": "Distribution",
        "task_type": "classification",
        "units": "",
        "direction": "context",
        "note": "Required for a CNS target.",
    },
    {
        "name": "PPBR_AZ",
        "known": True,
        "label": "Plasma protein binding rate",
        "abbrev": "PPBR",
        "category": "Distribution",
        "task_type": "regression",
        "units": "%",
        "direction": "context",
        "note": "",
    },
]


def response(
    smiles: list[str],
    invalid: list[str] | None = None,
    unknown: list[str] | None = None,
    percentiles: dict[str, float] | None = None,
) -> dict:
    """Build a successful service response for the given molecules."""
    return {
        "success": True,
        "results": [
            {
                "smiles": entry,
                "properties": {
                    "hERG": 0.1 + index / 100,
                    "BBB_Martins": 0.5,
                    "PPBR_AZ": 90.0 + index,
                },
                "percentiles": percentiles or {},
            }
            for index, entry in enumerate(smiles)
        ],
        "invalid_smiles": invalid or [],
        "property_info": PROPERTY_INFO,
        "unknown_properties": unknown or [],
        "error": None,
    }


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.fixture
def tool(storage):
    instance = AdmetPredictTool()
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
        stub(tool, response(["CCO"]), calls)

        result = await tool.execute({"smiles": ["CCO"]})

        assert result.content_type == ContentType.TEXT
        assert calls[0]["smiles"] == ["CCO"]

    @pytest.mark.asyncio
    async def test_accepts_a_bare_string_where_a_list_is_declared(self, tool):
        # Models pass a scalar for an array field often enough that rejecting it
        # would cost a turn to no purpose.
        calls: list = []
        stub(tool, response(["CCO"]), calls)

        await tool.execute({"smiles": "CCO"})

        assert calls[0]["smiles"] == ["CCO"]

    @pytest.mark.asyncio
    async def test_reads_a_stored_reference(self, tool, storage):
        ref = await storage.store("smiles", b"CCO\n")
        calls: list = []
        stub(tool, response(["CCO"]), calls)

        result = await tool.execute({"smiles_refs": [ref]})

        assert calls[0]["smiles"] == ["CCO"]
        assert result.metadata["smiles_origin"] == {"CCO": ref}

    @pytest.mark.asyncio
    async def test_reads_a_multi_line_reference_as_many_molecules(
        self, tool, storage
    ):
        ref = await storage.store("smiles", b"CCO\nCCC\n\nc1ccccc1\n")
        calls: list = []
        stub(tool, response(["CCO", "CCC", "c1ccccc1"]), calls)

        await tool.execute({"smiles_refs": [ref]})

        assert calls[0]["smiles"] == ["CCO", "CCC", "c1ccccc1"]

    @pytest.mark.asyncio
    async def test_tolerates_the_smi_name_column(self, tool, storage):
        ref = await storage.store("smiles", b"CCO ethanol\nCCC\tpropane\n")
        calls: list = []
        stub(tool, response(["CCO", "CCC"]), calls)

        await tool.execute({"smiles_refs": [ref]})

        assert calls[0]["smiles"] == ["CCO", "CCC"]

    @pytest.mark.asyncio
    async def test_deduplicates_across_both_input_forms(self, tool, storage):
        ref = await storage.store("smiles", b"CCO\nCCC\n")
        calls: list = []
        stub(tool, response(["CCO", "CCC"]), calls)

        await tool.execute({"smiles": ["CCO"], "smiles_refs": [ref]})

        assert calls[0]["smiles"] == ["CCO", "CCC"]

    @pytest.mark.asyncio
    async def test_rejects_an_unknown_reference(self, tool):
        stub(tool, response(["CCO"]))

        result = await tool.execute({"smiles_refs": ["smiles_deadbeef"]})

        assert result.content_type == ContentType.ERROR
        assert "smiles_deadbeef" in result.data

    @pytest.mark.asyncio
    async def test_rejects_a_malformed_reference(self, tool):
        stub(tool, response(["CCO"]))

        result = await tool.execute({"smiles_refs": ["not-a-reference"]})

        assert result.content_type == ContentType.ERROR

    @pytest.mark.asyncio
    async def test_refuses_an_oversized_batch_before_calling_the_service(self, tool):
        calls: list = []
        stub(tool, response(["CCO"]), calls)

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

    @pytest.mark.asyncio
    async def test_passes_requested_properties_through(self, tool):
        calls: list = []
        stub(tool, response(["CCO"]), calls)

        await tool.execute({"smiles": ["CCO"], "properties": ["herg", "bbb"]})

        assert calls[0]["properties"] == ["herg", "bbb"]

    @pytest.mark.asyncio
    async def test_omits_properties_when_not_requested(self, tool):
        calls: list = []
        stub(tool, response(["CCO"]), calls)

        await tool.execute({"smiles": ["CCO"]})

        assert "properties" not in calls[0]


class TestSingleMoleculeRendering:
    @pytest.mark.asyncio
    async def test_labels_each_property_and_its_direction(self, tool):
        stub(tool, response(["CCO"]))

        result = await tool.execute({"smiles": ["CCO"]})

        assert "hERG blocking" in result.data
        assert "lower is better" in result.data
        # A context-dependent property must not be given a direction.
        bbb_line = next(
            line for line in result.data.splitlines() if "Blood-brain" in line
        )
        assert "better" not in bbb_line

    @pytest.mark.asyncio
    async def test_shows_units_for_a_regression_property(self, tool):
        stub(tool, response(["CCO"]))
        result = await tool.execute({"smiles": ["CCO"]})
        assert "%" in result.data

    @pytest.mark.asyncio
    async def test_reports_drugbank_percentiles_when_present(self, tool):
        stub(tool, response(["CCO"], percentiles={"hERG": 42.0}))
        result = await tool.execute({"smiles": ["CCO"]})
        assert "percentile" in result.data


class TestBatchRendering:
    @pytest.mark.asyncio
    async def test_renders_a_table_with_abbreviated_headers(self, tool):
        stub(tool, response(["CCO", "CCC", "c1ccccc1"]))

        result = await tool.execute({"smiles": ["CCO", "CCC", "c1ccccc1"]})

        assert "molecule" in result.data
        assert "hERG" in result.data
        assert "PPBR" in result.data
        assert "c1ccccc1" in result.data

    @pytest.mark.asyncio
    async def test_a_table_carries_a_legend_expanding_every_header(self, tool):
        stub(tool, response(["CCO", "CCC"]))

        result = await tool.execute({"smiles": ["CCO", "CCC"]})

        assert "Legend:" in result.data
        assert "Plasma protein binding rate" in result.data
        assert "depends on the programme" in result.data

    @pytest.mark.asyncio
    async def test_truncates_the_rendered_table_but_not_the_csv(self, tool, storage):
        smiles = [f"C{'C' * index}O" for index in range(MAX_RENDERED_ROWS + 8)]
        stub(tool, response(smiles))

        result = await tool.execute({"smiles": smiles})

        assert str(len(smiles)) in result.data
        assert "python_execute" in result.data
        stored = (await storage.retrieve(result.metadata["csv_ref"])).decode()
        assert len(stored.strip().splitlines()) == len(smiles) + 1

    @pytest.mark.asyncio
    async def test_long_smiles_are_truncated_in_the_table_only(self, tool, storage):
        long_smiles = "C" * 120
        stub(tool, response([long_smiles, "CCO"]))

        result = await tool.execute({"smiles": [long_smiles, "CCO"]})

        assert long_smiles not in result.data
        stored = (await storage.retrieve(result.metadata["csv_ref"])).decode()
        assert long_smiles in stored


class TestResultsCsv:
    @pytest.mark.asyncio
    async def test_stores_one_row_per_molecule_with_a_header(self, tool, storage):
        stub(tool, response(["CCO", "CCC"]))

        result = await tool.execute({"smiles": ["CCO", "CCC"]})

        ref = result.metadata["csv_ref"]
        assert ref.startswith("csv_")
        rows = list(csv.reader(io.StringIO((await storage.retrieve(ref)).decode())))
        assert rows[0] == ["smiles", "hERG", "BBB_Martins", "PPBR_AZ"]
        assert [row[0] for row in rows[1:]] == ["CCO", "CCC"]

    @pytest.mark.asyncio
    async def test_the_reference_is_offered_to_the_model(self, tool):
        stub(tool, response(["CCO", "CCC"]))
        result = await tool.execute({"smiles": ["CCO", "CCC"]})
        assert result.metadata["csv_ref"] in result.data

    @pytest.mark.asyncio
    async def test_predictions_still_render_without_storage(self, tool):
        # Storage is unavailable in some training paths; losing the CSV must not
        # lose the answer.
        tool._storage = None
        stub(tool, response(["CCO", "CCC"]))

        result = await tool.execute({"smiles": ["CCO", "CCC"]})

        assert result.content_type == ContentType.TEXT
        assert "hERG" in result.data
        assert "csv_ref" not in result.metadata


class TestPartialAndFailedResults:
    @pytest.mark.asyncio
    async def test_names_the_smiles_the_service_could_not_parse(self, tool):
        stub(tool, response(["CCO"], invalid=["not-a-molecule"]))

        result = await tool.execute({"smiles": ["CCO", "not-a-molecule"]})

        assert "not-a-molecule" in result.data
        assert result.metadata["invalid_smiles"] == ["not-a-molecule"]

    @pytest.mark.asyncio
    async def test_summarizes_a_long_list_of_invalid_smiles(self, tool):
        invalid = [f"junk{index}" for index in range(9)]
        stub(tool, response(["CCO"], invalid=invalid))

        result = await tool.execute({"smiles": ["CCO", *invalid]})

        assert "9 SMILES could not be parsed" in result.data
        assert "and 4 more" in result.data

    @pytest.mark.asyncio
    async def test_reports_ignored_property_names(self, tool):
        stub(tool, response(["CCO"], unknown=["frobnicity"]))

        result = await tool.execute(
            {"smiles": ["CCO"], "properties": ["herg", "frobnicity"]}
        )

        assert "frobnicity" in result.data

    @pytest.mark.asyncio
    async def test_surfaces_a_service_level_failure(self, tool):
        stub(
            tool,
            {
                "success": False,
                "error": "ADMET-AI model weights were not found",
                "unknown_properties": [],
            },
        )

        result = await tool.execute({"smiles": ["CCO"]})

        assert result.content_type == ContentType.ERROR
        assert "weights were not found" in result.data

    @pytest.mark.asyncio
    async def test_a_failure_lists_the_names_it_did_not_recognize(self, tool):
        stub(
            tool,
            {
                "success": False,
                "error": "None of the requested properties are known",
                "unknown_properties": ["frobnicity"],
            },
        )

        result = await tool.execute(
            {"smiles": ["CCO"], "properties": ["frobnicity"]}
        )

        assert "frobnicity" in result.data

    @pytest.mark.asyncio
    async def test_an_unreachable_service_says_it_may_be_warming_up(self, tool):
        stub(tool, RuntimeError("connection refused"))

        result = await tool.execute({"smiles": ["CCO"]})

        assert result.content_type == ContentType.ERROR
        assert "warming up" in result.data

    @pytest.mark.asyncio
    async def test_an_empty_result_set_is_an_error_not_an_empty_table(self, tool):
        stub(
            tool,
            {
                "success": True,
                "results": [],
                "invalid_smiles": ["junk"],
                "property_info": PROPERTY_INFO,
                "unknown_properties": [],
            },
        )

        result = await tool.execute({"smiles": ["junk"]})

        assert result.content_type == ContentType.ERROR
        assert "junk" in result.data


class TestMetadata:
    @pytest.mark.asyncio
    async def test_carries_the_numbers_for_the_findings_ledger(self, tool):
        stub(tool, response(["CCO", "CCC"]))

        result = await tool.execute({"smiles": ["CCO", "CCC"]})

        assert result.metadata["num_molecules"] == 2
        assert result.metadata["properties"] == [
            "hERG",
            "BBB_Martins",
            "PPBR_AZ",
        ]
        assert result.metadata["predictions"]["CCO"]["hERG"] == pytest.approx(0.1)


class TestSchema:
    def test_declares_no_required_arguments(self):
        # Either input form is sufficient, so requiring one would reject the
        # other; the check is in execute() instead.
        schema = AdmetPredictTool().get_schema()["function"]
        assert schema["parameters"].get("required", []) == []

    def test_marks_the_reference_argument_for_the_reference_resolver(self):
        schema = AdmetPredictTool().get_schema()["function"]
        refs = schema["parameters"]["properties"]["smiles_refs"]
        assert refs["x-aevom-reference"] == {"data_type": "smiles"}

    def test_warns_that_predictions_are_not_measurements(self):
        description = AdmetPredictTool().get_schema()["function"]["description"]
        assert "not measurements" in description
