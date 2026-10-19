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

"""Tests for how a workflow subgraph's result crosses back to the parent.

The child's trajectory never reaches the parent -- that firewall is why
subgraphs are the best context mechanism in the system. The cost is that
anything the child measured has to cross the boundary deliberately. These
tests hold the numbers and the artifact references to that boundary.
"""

from api.graph.workflows import _render_subgraph_result


class TestRendering:
    def test_the_childs_own_answer_leads(self):
        out = _render_subgraph_result({"summary": "Docked and visualised."})
        assert out.startswith("Docked and visualised.")

    def test_artifacts_are_named_so_the_parent_can_use_them(self):
        """The parent never saw the child's tool output, so refs must be listed."""
        out = _render_subgraph_result({
            "summary": "Done.",
            "artifacts_created": ["pdb_c3c3c3c3", "sdf_a1b2c3d4"],
        })
        assert "Artifacts created: pdb_c3c3c3c3, sdf_a1b2c3d4" in out

    def test_measurements_cross_as_the_tools_reported_them(self):
        out = _render_subgraph_result({
            "summary": "The ligand binds strongly.",
            "findings": [
                {"summary": "vina_dock(lig=sdf_a1b2c3d4) -> -9.50 kcal/mol"},
                {"summary": "openfold3(seq=sequence_11223344) -> pLDDT 87.3"},
            ],
        })
        assert "-9.50 kcal/mol" in out
        assert "pLDDT 87.3" in out

    def test_an_empty_workflow_still_says_something(self):
        out = _render_subgraph_result({})
        assert "no output" in out

    def test_nothing_measured_adds_no_empty_section(self):
        out = _render_subgraph_result({"summary": "Done.", "findings": []})
        assert out == "Done."

    def test_findings_without_summaries_add_no_empty_section(self):
        out = _render_subgraph_result({"summary": "Done.", "findings": [{}]})
        assert out == "Done."
