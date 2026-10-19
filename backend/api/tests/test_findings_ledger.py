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

"""Tests for finding extraction and session-log rendering.

The ledger's value rests on one claim: a number a tool measured is recorded
once and re-rendered verbatim forever. These tests hold that claim, mostly by
checking that scalars survive extraction and that the rendered line stays
short enough to be worth keeping in every prompt.
"""

import pytest

from api.conversations.findings import (
    build_summary,
    extract_finding,
    format_session_log,
)
from api.tools.base import ContentType, ToolResult


class TestReferenceExtraction:
    """Refs are the part that must never be mangled."""

    def test_input_refs_come_from_arguments(self):
        result = ToolResult.text("done")
        finding = extract_finding(
            "vina_dock",
            {"receptor_pdb_ref": "pdb_a1b2c3d4", "ligand_sdf_ref": "sdf_11223344"},
            result,
        )
        assert finding["input_refs"] == ["pdb_a1b2c3d4", "sdf_11223344"]

    def test_output_ref_comes_from_a_data_reference_result(self):
        result = ToolResult.data_reference("sdf_deadbeef", "sdf", "stored")
        finding = extract_finding("download_pubchem_sdf", {"cid": 2244}, result)
        assert finding["output_refs"] == ["sdf_deadbeef"]

    def test_a_ref_is_not_both_input_and_output(self):
        result = ToolResult.data_reference("sdf_deadbeef", "sdf", "stored")
        finding = extract_finding("noop", {"ref": "sdf_deadbeef"}, result)
        assert finding["input_refs"] == ["sdf_deadbeef"]
        assert finding["output_refs"] == []

    def test_refs_announced_in_tool_text_are_captured(self):
        """python_execute names the artifacts it created only in its output."""
        result = ToolResult.text(
            "Output:\nStored caffeine SMILES as: smiles_2904b978\n"
            "Stored ibuprofen SMILES as: smiles_72b2e1cc\n"
        )
        finding = extract_finding("python_execute", {"code": "..."}, result)
        assert finding["output_refs"] == ["smiles_2904b978", "smiles_72b2e1cc"]

    def test_refs_named_in_a_failure_are_not_claimed_as_outputs(self):
        result = ToolResult.error("Reference sdf_a1b2c3d4 not found in this session")
        finding = extract_finding("visualize_3d", {}, result)
        assert finding["output_refs"] == []

    def test_multi_segment_type_refs_survive_intact(self):
        """skill_file_x recorded as file_x would not resolve."""
        result = ToolResult.text("ok")
        finding = extract_finding("load_skill", {"ref": "skill_file_a1b2c3d4"}, result)
        assert finding["input_refs"] == ["skill_file_a1b2c3d4"]

    def test_hex_that_is_also_alphabetic_still_parses(self):
        result = ToolResult.text("ok")
        finding = extract_finding("load_skill", {"ref": "skill_file_abcdefab"}, result)
        assert finding["input_refs"] == ["skill_file_abcdefab"]


class TestScalarExtraction:
    """The numbers the whole design exists to preserve."""

    def test_docking_affinity_is_captured(self):
        result = ToolResult.data_reference(
            "sdf_pose1234",
            "sdf",
            "Docked.",
            metadata={"affinity_kcal_mol": -9.5},
        )
        finding = extract_finding("vina_dock", {"ligand_sdf_ref": "sdf_a1b2c3d4"}, result)
        assert finding["scalars"] == {"affinity_kcal_mol": -9.5}
        assert "-9.50 kcal/mol" in finding["summary"]

    def test_fold_confidence_metrics_are_captured(self):
        result = ToolResult.data_reference(
            "pdb_fold1234",
            "pdb",
            "Folded.",
            metadata={"plddt_score": 87.3, "iptm_score": 0.812, "ptm_score": 0.774},
        )
        finding = extract_finding("openfold3", {"sequence_ref": "sequence_aabbccdd"}, result)
        assert finding["scalars"] == {"plddt": 87.3, "iptm": 0.812, "ptm": 0.774}

    def test_alternate_key_for_the_same_quantity_normalizes(self):
        """OpenFold3 reports complex_plddt_score for multimers."""
        result = ToolResult.data_reference(
            "pdb_fold1234", "pdb", "Folded.", metadata={"complex_plddt_score": 91.0}
        )
        finding = extract_finding("openfold3", {}, result)
        assert finding["scalars"]["plddt"] == 91.0

    def test_molecular_properties_are_captured(self):
        result = ToolResult.text(
            "Molecule info",
            metadata={"molecular_weight": 412.5, "logp": 4.8, "tpsa": 62.3},
        )
        finding = extract_finding("molecule_info", {"smiles_ref": "smiles_a1b2c3d4"}, result)
        assert finding["scalars"] == {
            "molecular_weight": 412.5,
            "logp": 4.8,
            "tpsa": 62.3,
        }

    def test_non_numeric_metadata_is_ignored_rather_than_crashing(self):
        result = ToolResult.text("ok", metadata={"affinity_kcal_mol": "not a number"})
        finding = extract_finding("vina_dock", {}, result)
        assert finding["scalars"] == {}


class TestSummaryRendering:
    def test_units_are_rendered_so_scalars_are_not_confusable(self):
        line = build_summary(
            "vina_dock",
            {"ligand_sdf_ref": "sdf_a1b2c3d4"},
            {"affinity_kcal_mol": -9.5},
            ["sdf_pose1234"],
            success=True,
        )
        assert line == (
            "vina_dock(ligand_sdf_ref=sdf_a1b2c3d4) -> -9.50 kcal/mol [sdf_pose1234]"
        )

    def test_bulky_arguments_are_dropped_from_the_line(self):
        """A block of Python in the log would drown the facts it surfaces."""
        line = build_summary(
            "python_execute",
            {"code": "x = 1\n" * 100, "purpose": "count"},
            {},
            [],
            success=True,
        )
        assert "x = 1" not in line
        assert "purpose=count" in line

    def test_failures_are_recorded_not_silently_dropped(self):
        result = ToolResult.error("Receptor has no binding site")
        finding = extract_finding("vina_dock", {"receptor_pdb_ref": "pdb_a1b2c3d4"}, result)
        assert finding["success"] is False
        assert "failed" in finding["summary"]
        assert "binding site" in finding["summary"]

    def test_a_tool_without_scalars_falls_back_to_its_own_text(self):
        result = ToolResult.text("Found 3 structures: 1ABC, 2DEF, 3GHI")
        finding = extract_finding("search_rcsb", {"return_type": "entry"}, result)
        assert "1ABC" in finding["summary"]

    def test_an_argument_too_large_to_render_still_leaves_a_row(self):
        """search_rcsb takes a nested query object, which exceeds the arg limit."""
        query = {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_entity_source_organism.rcsb_gene_name.value",
                "operator": "exact_match",
                "value": "KRAS",
            },
        }
        result = ToolResult.text("Searched for 'KRAS'. total_count: 3 — showing 3.")
        finding = extract_finding("search_rcsb", {"query": query}, result)
        assert finding["success"] is True
        # The query is dropped from the rendered line, so the tool's own text is
        # the only record of what was searched for.
        assert "KRAS" in finding["summary"]

    def test_summary_stays_short_enough_to_keep_forever(self):
        result = ToolResult.text("word " * 500)
        finding = extract_finding("some_tool", {}, result)
        assert len(finding["summary"]) < 200


class TestSessionLog:
    @pytest.fixture
    def findings(self):
        return [
            {
                "turn_seq": 1,
                "iteration": 2,
                "summary": "vina_dock(lig=sdf_a1b2c3d4) -> -9.50 kcal/mol [sdf_pose1234]",
            },
            {
                "turn_seq": 1,
                "iteration": 4,
                "summary": "vina_dock(lig=sdf_d4d4d4d4) -> -6.10 kcal/mol [sdf_pose5678]",
            },
            {
                "turn_seq": 2,
                "iteration": 1,
                "summary": "molecule_info(sdf_d4d4d4d4) -> MW 412.5, LogP 4.80",
            },
        ]

    def test_renders_turn_and_iteration_coordinates(self, findings):
        out = format_session_log(findings)
        assert out.startswith("[SESSION LOG]")
        assert "T1.2 vina_dock" in out
        assert "T2.1 molecule_info" in out

    def test_measurements_appear_verbatim(self, findings):
        """No model call, so no opportunity to paraphrase a number."""
        out = format_session_log(findings)
        assert "-9.50 kcal/mol" in out
        assert "-6.10 kcal/mol" in out

    def test_empty_ledger_renders_nothing(self):
        assert format_session_log([]) == ""

    def test_truncation_says_what_it_hid(self, findings):
        out = format_session_log(findings, max_entries=2)
        assert "most recent 2 of 3" in out
        assert "T1.2" not in out
        assert "T2.1" in out

    def test_rendering_is_deterministic(self, findings):
        assert format_session_log(findings) == format_session_log(findings)
