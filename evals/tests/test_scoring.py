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

"""Tests for the eval scoring rules.

The rules decide whether a run passes, so they are the part of the harness most
worth pinning down. Deliberately Harbor-free: these run in the plain
devcontainer, with no stack and no Harbor install.
"""

from __future__ import annotations

import pytest

from aevom_eval.scoring import (
    ANSWER,
    ARTIFACTS,
    JOB_AND_POLICY,
    NO_ERRORS,
    OVERALL,
    TOOL_SELECTION,
    score_trajectory,
)


def trajectory(**overrides) -> dict:
    base = {
        "session_id": "s1",
        "final_content": "The molecular weight of caffeine is 194.19 g/mol.",
        "tool_calls": [
            {"name": "search_pubchem", "success": True},
            {"name": "get_pubchem_properties", "success": True},
        ],
        "artifacts": [{"data_type": "sdf", "reference": "sdf_1"}],
        "bridge_events": [],
        "errors": [],
        "hit_iteration_limit": False,
    }
    base.update(overrides)
    return base


def names(score) -> dict[str, bool]:
    return {check.name: check.passed for check in score.checks}


class TestToolSelection:
    def test_required_tool_present(self):
        score = score_trajectory(
            trajectory(), {"required_tools": ["search_pubchem"]}
        )
        assert score.rewards[TOOL_SELECTION] == 1.0

    def test_required_tool_absent(self):
        score = score_trajectory(trajectory(), {"required_tools": ["dock_ligand"]})
        assert score.rewards[TOOL_SELECTION] == 0.0
        assert "was never called" in score.failures[0].detail

    def test_partial_credit_across_required_tools(self):
        score = score_trajectory(
            trajectory(), {"required_tools": ["search_pubchem", "dock_ligand"]}
        )
        assert score.rewards[TOOL_SELECTION] == 0.5

    def test_forbidden_tool(self):
        score = score_trajectory(
            trajectory(), {"forbidden_tools": ["get_pubchem_properties"]}
        )
        assert score.rewards[TOOL_SELECTION] == 0.0

    def test_sequence_allows_interleaving(self):
        """Other calls between the required ones are fine; order is what matters."""
        traj = trajectory(
            tool_calls=[
                {"name": "search_pubchem", "success": True},
                {"name": "python_execute", "success": True},
                {"name": "get_pubchem_properties", "success": True},
            ]
        )
        score = score_trajectory(
            traj,
            {"required_tool_sequence": ["search_pubchem", "get_pubchem_properties"]},
        )
        assert score.rewards[TOOL_SELECTION] == 1.0

    def test_sequence_rejects_wrong_order(self):
        traj = trajectory(
            tool_calls=[
                {"name": "get_pubchem_properties", "success": True},
                {"name": "search_pubchem", "success": True},
            ]
        )
        score = score_trajectory(
            traj,
            {"required_tool_sequence": ["search_pubchem", "get_pubchem_properties"]},
        )
        assert score.rewards[TOOL_SELECTION] == 0.0


class TestToolArguments:
    """The design tools take their whole specification through their arguments.

    ``rfdiffusion_design_backbone`` selects de novo, motif scaffolding and binder
    design through one ``contigs`` string, so "was it called" cannot distinguish
    a scaffolded motif from a design that ignored the input structure.
    """

    def design(self, **args):
        return trajectory(
            tool_calls=[{"name": "rfdiffusion_design_backbone", "success": True, "args": args}]
        )

    def test_present(self):
        score = score_trajectory(
            self.design(contigs="A25-35/0 50-80", hotspot_res=["A54"]),
            {"required_tool_arguments": [
                {"tool": "rfdiffusion_design_backbone", "arg": "hotspot_res", "present": True}
            ]},
        )
        assert score.rewards[TOOL_SELECTION] == 1.0

    def test_empty_list_counts_as_absent(self):
        score = score_trajectory(
            self.design(contigs="100", hotspot_res=[]),
            {"required_tool_arguments": [
                {"tool": "rfdiffusion_design_backbone", "arg": "hotspot_res", "present": True}
            ]},
        )
        assert score.rewards[TOOL_SELECTION] == 0.0

    def test_absence_is_assertable(self):
        """De novo design means omitting pdb_ref, not passing a placeholder."""
        score = score_trajectory(
            self.design(contigs="100"),
            {"required_tool_arguments": [
                {"tool": "rfdiffusion_design_backbone", "arg": "pdb_ref", "present": False}
            ]},
        )
        assert score.rewards[TOOL_SELECTION] == 1.0

        score = score_trajectory(
            self.design(contigs="100", pdb_ref="pdb_1"),
            {"required_tool_arguments": [
                {"tool": "rfdiffusion_design_backbone", "arg": "pdb_ref", "present": False}
            ]},
        )
        assert score.rewards[TOOL_SELECTION] == 0.0

    @pytest.mark.parametrize(
        "contigs,expected",
        [
            # A chain-qualified span then a chain break: the input structure is
            # being built on.
            ("A25-109/0 100", True),
            ("B1-85/0 60-100", True),
            # A bare length is a de novo design; the input structure was ignored.
            ("100", False),
            ("80-120", False),
        ],
    )
    def test_matches_regex(self, contigs, expected):
        score = score_trajectory(
            self.design(contigs=contigs),
            {"required_tool_arguments": [
                {
                    "tool": "rfdiffusion_design_backbone",
                    "arg": "contigs",
                    "matches": r"[A-Za-z]\d+-\d+/0",
                }
            ]},
        )
        assert (score.rewards[TOOL_SELECTION] == 1.0) is expected

    def test_min_items(self):
        expect = {"required_tool_arguments": [
            {"tool": "rfdiffusion_design_backbone", "arg": "hotspot_res", "min_items": 2}
        ]}
        assert score_trajectory(
            self.design(hotspot_res=["A54", "A58", "A61"]), expect
        ).rewards[TOOL_SELECTION] == 1.0
        assert score_trajectory(
            self.design(hotspot_res=["A54"]), expect
        ).rewards[TOOL_SELECTION] == 0.0

    def test_predicates_are_combined(self):
        expect = {"required_tool_arguments": [
            {
                "tool": "proteinmpnn_design_sequence",
                "arg": "chains",
                "present": True,
                "min_items": 1,
            }
        ]}
        traj = trajectory(
            tool_calls=[{
                "name": "proteinmpnn_design_sequence",
                "success": True,
                "args": {"chains": ["A"]},
            }]
        )
        assert score_trajectory(traj, expect).rewards[TOOL_SELECTION] == 1.0

    def test_any_call_satisfies(self):
        """Getting it wrong and then correcting is recovery, priced under no_errors."""
        traj = trajectory(
            tool_calls=[
                {"name": "rfdiffusion_design_backbone", "success": False, "args": {"contigs": "100"}},
                {"name": "rfdiffusion_design_backbone", "success": True, "args": {"contigs": "A1-85/0 100"}},
            ]
        )
        score = score_trajectory(
            traj,
            {"required_tool_arguments": [
                {"tool": "rfdiffusion_design_backbone", "arg": "contigs", "matches": r"/0"}
            ]},
        )
        assert score.rewards[TOOL_SELECTION] == 1.0

    def test_uncalled_tool_fails_with_a_useful_detail(self):
        score = score_trajectory(
            trajectory(),
            {"required_tool_arguments": [
                {"tool": "msa_search", "arg": "sequence_ref", "present": True}
            ]},
        )
        assert score.rewards[TOOL_SELECTION] == 0.0
        assert "never called" in score.failures[0].detail

    def test_failure_detail_reports_what_was_seen(self):
        score = score_trajectory(
            self.design(contigs="80-120"),
            {"required_tool_arguments": [
                {"tool": "rfdiffusion_design_backbone", "arg": "contigs", "matches": r"/0"}
            ]},
        )
        assert "80-120" in score.failures[0].detail

    def test_missing_args_dict_does_not_crash(self):
        """A tool_call_end with no matching start records no args at all."""
        traj = trajectory(
            tool_calls=[{"name": "rfdiffusion_design_backbone", "success": True, "args": None}]
        )
        score = score_trajectory(
            traj,
            {"required_tool_arguments": [
                {"tool": "rfdiffusion_design_backbone", "arg": "contigs", "present": True}
            ]},
        )
        assert score.rewards[TOOL_SELECTION] == 0.0


class TestAnswer:
    def test_contains_is_case_insensitive(self):
        score = score_trajectory(trajectory(), {"answer_contains": ["CAFFEINE"]})
        assert score.rewards[ANSWER] == 1.0

    def test_excludes(self):
        score = score_trajectory(trajectory(), {"answer_excludes": ["aspirin"]})
        assert score.rewards[ANSWER] == 1.0

    def test_contains_any(self):
        score = score_trajectory(
            trajectory(), {"answer_contains_any": [["theine", "caffeine"]]}
        )
        assert score.rewards[ANSWER] == 1.0

    @pytest.mark.parametrize(
        "answer,wanted,tolerance,expected",
        [
            ("MW is 194.19 g/mol", 194.19, 0.0, True),
            # A rounded restatement should still pass.
            ("MW is 194.2 g/mol", 194.19, 0.05, True),
            # A number that merely contains the digits must not.
            ("id 1194.19", 194.19, 0.05, False),
            ("affinity -7.3 kcal/mol", -7.5, 0.5, True),
            ("affinity -9.1 kcal/mol", -7.5, 0.5, False),
        ],
    )
    def test_numeric_tolerance(self, answer, wanted, tolerance, expected):
        """Numbers are compared numerically; substring matching gets this wrong."""
        score = score_trajectory(
            trajectory(final_content=answer),
            {"answer_numbers": [{"value": wanted, "tolerance": tolerance}]},
        )
        assert (score.rewards[ANSWER] == 1.0) is expected

    def test_bare_number_expectation(self):
        score = score_trajectory(
            trajectory(final_content="exactly 42 items"), {"answer_numbers": [42]}
        )
        assert score.rewards[ANSWER] == 1.0

    def test_min_chars_catches_an_empty_answer(self):
        score = score_trajectory(
            trajectory(final_content=""), {"answer_min_chars": 50}
        )
        assert score.rewards[ANSWER] == 0.0


class TestStructureQuery:
    def test_success_requires_atoms_not_just_a_call(self):
        """The pre-bridge failure mode was a call that returned nothing."""
        traj = trajectory(
            bridge_events=[
                {"kind": "structure_query", "molql": "(x)", "atom_count": 0, "error": None}
            ]
        )
        score = score_trajectory(traj, {"structure_query_must_succeed": True})
        assert score.rewards[ANSWER] == 0.0
        assert "none selected any atoms" in score.failures[0].detail

    def test_never_issued_is_reported_distinctly(self):
        score = score_trajectory(trajectory(), {"structure_query_must_succeed": True})
        assert "never issued" in score.failures[0].detail

    def test_a_successful_query_passes_and_reports_the_selection(self):
        traj = trajectory(
            bridge_events=[
                {
                    "kind": "structure_query",
                    "molql": "(sel.atom.all)",
                    "atom_count": 50,
                    "residue_count": 7,
                    "error": None,
                }
            ]
        )
        score = score_trajectory(traj, {"structure_query_must_succeed": True})
        assert score.rewards[ANSWER] == 1.0
        assert "50 atoms in 7 residue(s)" in names_detail(score, "structure_query_succeeded")

    def test_min_selection_residues_uses_the_largest_selection(self):
        traj = trajectory(
            bridge_events=[
                {"kind": "structure_query", "atom_count": 4, "residue_count": 1},
                {"kind": "structure_query", "atom_count": 50, "residue_count": 7},
            ]
        )
        score = score_trajectory(traj, {"min_selection_residues": 7})
        assert score.rewards[ANSWER] == 1.0

    def test_failed_queries_are_excluded_from_the_largest_selection(self):
        traj = trajectory(
            bridge_events=[
                {
                    "kind": "structure_query",
                    "atom_count": 99,
                    "residue_count": 99,
                    "error": "parse failure",
                }
            ]
        )
        score = score_trajectory(traj, {"min_selection_residues": 7})
        assert score.rewards[ANSWER] == 0.0


class TestArtifacts:
    def test_required_type(self):
        score = score_trajectory(trajectory(), {"required_artifact_types": ["sdf"]})
        assert score.rewards[ARTIFACTS] == 1.0

    def test_missing_type_names_what_was_there(self):
        score = score_trajectory(trajectory(), {"required_artifact_types": ["pdb"]})
        assert score.rewards[ARTIFACTS] == 0.0
        assert "'sdf'" in score.failures[0].detail

    def test_min_artifacts(self):
        score = score_trajectory(trajectory(artifacts=[]), {"min_artifacts": 1})
        assert score.rewards[ARTIFACTS] == 0.0


class TestErrors:
    def test_failed_tool_calls_within_budget(self):
        traj = trajectory(
            tool_calls=[
                {"name": "download_pubchem_sdf", "success": False},
                {"name": "search_pubchem", "success": True},
            ]
        )
        score = score_trajectory(traj, {"max_failed_tool_calls": 1})
        assert score.rewards[NO_ERRORS] == 1.0

    def test_failed_tool_calls_over_budget(self):
        traj = trajectory(
            tool_calls=[
                {"name": "a", "success": False},
                {"name": "b", "success": False},
            ]
        )
        score = score_trajectory(traj, {"max_failed_tool_calls": 1})
        assert score.rewards[NO_ERRORS] == 0.0

    def test_iteration_limit(self):
        score = score_trajectory(
            trajectory(hit_iteration_limit=True), {"allow_iteration_limit": False}
        )
        assert score.rewards[NO_ERRORS] == 0.0

    def test_driver_errors(self):
        score = score_trajectory(
            trajectory(errors=["bridge error: boom"]), {"max_driver_errors": 0}
        )
        assert score.rewards[NO_ERRORS] == 0.0


class TestAggregation:
    def test_undeclared_dimensions_are_absent(self):
        """A task silent on artifacts must not be credited for them."""
        score = score_trajectory(trajectory(), {"answer_contains": ["caffeine"]})
        assert set(score.rewards) == {ANSWER, OVERALL}

    def test_overall_is_the_mean_of_declared_dimensions(self):
        score = score_trajectory(
            trajectory(),
            {
                "answer_contains": ["caffeine"],  # passes -> 1.0
                "required_artifact_types": ["pdb"],  # fails -> 0.0
            },
        )
        assert score.rewards[ANSWER] == 1.0
        assert score.rewards[ARTIFACTS] == 0.0
        assert score.rewards[OVERALL] == 0.5

    def test_a_missing_trajectory_scores_zero_and_says_why(self):
        score = score_trajectory(None, {"answer_contains": ["caffeine"]})
        assert score.rewards[OVERALL] == 0.0
        assert score.rewards[ANSWER] == 0.0
        assert "failed to drive Aevom" in score.failures[0].detail


class TestJobAndPolicy:
    """The dimension shipped before the fields it reads were on the wire.

    That made it not merely inert but harmful: it entered the mean and could
    only ever contribute 0, so declaring it deflated the run's reward. These
    pin the three-way outcome that fixes it.
    """

    def _docked(self, **call_overrides) -> dict:
        call = {"name": "vina_dock", "success": True}
        call.update(call_overrides)
        return trajectory(tool_calls=[call])

    def test_matching_status_passes(self):
        score = score_trajectory(
            self._docked(job_status="completed"),
            {"required_job_status": [{"tool": "vina_dock", "status": "completed"}]},
        )
        assert score.rewards[JOB_AND_POLICY] == 1.0

    def test_wrong_status_fails(self):
        score = score_trajectory(
            self._docked(job_status="failed"),
            {"required_job_status": [{"tool": "vina_dock", "status": "completed"}]},
        )
        assert score.rewards[JOB_AND_POLICY] == 0.0

    def test_a_tool_that_never_ran_is_a_real_failure(self):
        # Not a measurement gap: the agent simply didn't do the work.
        score = score_trajectory(
            trajectory(),
            {"required_job_status": [{"tool": "vina_dock", "status": "completed"}]},
        )
        assert score.rewards[JOB_AND_POLICY] == 0.0

    def test_an_unreported_field_is_not_scored_at_all(self):
        # The tool ran, but the backend never sent job_status. Grading that as
        # a failure would score the harness rather than the agent.
        score = score_trajectory(
            self._docked(),
            {"required_job_status": [{"tool": "vina_dock", "status": "completed"}]},
        )
        assert JOB_AND_POLICY not in score.rewards

    def test_an_unreported_field_does_not_deflate_the_overall_reward(self):
        """The regression this whole guard exists for."""
        score = score_trajectory(
            self._docked(),
            {
                "answer_contains": ["caffeine"],  # passes -> 1.0
                "required_job_status": [{"tool": "vina_dock", "status": "completed"}],
            },
        )
        assert score.rewards[ANSWER] == 1.0
        assert score.rewards[OVERALL] == 1.0

    def test_policy_decision_is_scored_the_same_way(self):
        score = score_trajectory(
            self._docked(policy_decision="allow"),
            {"policy_decision_expected": [{"tool": "vina_dock", "decision": "allow"}]},
        )
        assert score.rewards[JOB_AND_POLICY] == 1.0


def names_detail(score, name: str) -> str:
    return next(check.detail for check in score.checks if check.name == name)
