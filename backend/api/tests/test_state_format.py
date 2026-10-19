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

"""Tests for ``api.state.format`` (shared formatters for state injection)."""

from api.state.format import (
    MAX_STORAGE_ARTIFACTS_IN_PROMPT,
    format_active_structure,
    format_nim_notices,
    format_selections,
    format_storage_state,
    format_viewer_state,
    inject_state_into_prompt,
)


class TestFormatViewerState:
    def test_empty(self):
        assert (
            format_viewer_state({"is_empty": True, "structures": []})
            == "[VIEWER STATE: Empty - no structures loaded]"
        )

    def test_includes_pdb_ref(self):
        out = format_viewer_state(
            {
                "structures": [
                    {
                        "label": "MyPDB",
                        "structure_type": "complex",
                        "pdb_ref": "pdb_xyz",
                        "chain_ids": ["A"],
                        "residue_count": 100,
                        "atom_count": 800,
                        "has_ligand": True,
                    }
                ]
            }
        )
        assert "ref=pdb_xyz" in out
        assert "MyPDB" in out
        assert "ligand" in out


class TestFormatStorageState:
    def test_empty(self):
        assert (
            format_storage_state({"artifacts": []})
            == "[STORAGE STATE: Empty - no artifacts stored]"
        )

    def test_truncates_to_cap(self):
        artifacts = [
            {"reference": f"pdb_{i:04d}", "data_type": "pdb", "description": "x"}
            for i in range(MAX_STORAGE_ARTIFACTS_IN_PROMPT + 5)
        ]
        out = format_storage_state({"artifacts": artifacts})
        assert "and 5 more" in out

    def test_truncates_long_description(self):
        artifacts = [
            {"reference": "pdb_a", "data_type": "pdb", "description": "x" * 200}
        ]
        out = format_storage_state({"artifacts": artifacts})
        assert "..." in out


class TestFormatActiveStructure:
    def test_none(self):
        assert "None" in format_active_structure(None)

    def test_with_label_from_storage(self):
        out = format_active_structure(
            "pdb_winner",
            {
                "artifacts": [
                    {
                        "reference": "pdb_winner",
                        "description": "winner-ligand docked complex",
                    }
                ]
            },
        )
        assert "pdb_winner" in out
        assert "winner-ligand" in out

    def test_without_label(self):
        out = format_active_structure("pdb_xyz", {"artifacts": []})
        assert "pdb_xyz" in out


class TestFormatSelections:
    def test_empty(self):
        assert "No selections" in format_selections([])

    def test_listed(self):
        out = format_selections(
            [{"reference": "selection_a", "atom_count": 10, "residue_count": 2}]
        )
        assert "selection_a" in out
        assert "10 atoms" in out


def _sel(ref, role=None, value=None, atoms=5, user=True, listing=""):
    return {
        "reference": ref,
        "atom_count": atoms,
        "residue_count": 2 if atoms else 0,
        "provenance": "user" if user else "agent",
        "role": role,
        "role_value": value,
        "residue_listing": listing,
    }


class TestFormatSelectionRoles:
    """The model must see the user's roles, and only the ones in force."""

    def test_role_is_named_and_listing_shown(self):
        out = format_selections([_sel("selection_h", "hotspot", listing="A: Y56 M115")])
        assert "design role: hotspot" in out
        assert "A: Y56 M115" in out

    def test_parameter_is_part_of_the_role(self):
        out = format_selections([_sel("selection_d", "hbond", "DONOR")])
        assert "design role: hbond=DONOR" in out

    def test_a_role_is_pointed_at_use_my_roles(self):
        out = format_selections([_sel("selection_h", "hotspot")])
        assert "use_my_roles=true" in out
        assert "use_my_roles" not in format_selections([_sel("selection_a", user=False)])

    def test_the_newest_selection_for_a_role_wins(self):
        # Newest first, as the prompt receives them.
        out = format_selections(
            [_sel("selection_new", "hotspot"), _sel("selection_old", "hotspot")]
        )
        assert "selection_new" in out and "selection_old" not in out

    def test_a_removed_role_disappears_with_everything_older(self):
        out = format_selections(
            [
                _sel("selection_gone", "hotspot", atoms=0),
                _sel("selection_old", "hotspot"),
                _sel("selection_probe", user=False),
            ]
        )
        assert "selection_gone" not in out and "selection_old" not in out
        assert "selection_probe" in out
        assert "use_my_roles" not in out

    def test_parameters_are_separate_roles(self):
        out = format_selections(
            [_sel("selection_d", "hbond", "DONOR"), _sel("selection_a", "hbond", "ACCEPTOR")]
        )
        assert "selection_d" in out and "selection_a" in out

    def test_only_removed_roles_leaves_nothing(self):
        assert "No selections" in format_selections([_sel("selection_gone", "keep", atoms=0)])

    def test_roles_are_not_pushed_out_by_probes(self):
        probes = [_sel(f"selection_p{i}", user=False) for i in range(8)]
        out = format_selections(probes + [_sel("selection_role", "keep")])
        assert "selection_role" in out

    def test_roles_do_not_use_up_the_window(self):
        roles = [_sel(f"selection_r{i}", role, None) for i, role in enumerate(["hotspot", "keep", "float"])]
        probes = [_sel(f"selection_p{i}", user=False) for i in range(5)]
        out = format_selections(roles + probes)
        assert all(f"selection_p{i}" in out for i in range(5))
        assert "more" not in out


class TestFormatNimNotices:
    def test_none_means_no_block(self):
        assert format_nim_notices([]) == ""

    def test_names_the_nim_and_what_confirming_would_add(self):
        out = format_nim_notices([{"name": "RFD4", "tools": ["design_protein_binder"]}])
        assert "RFD4 NIM registered but not confirmed (would add design_protein_binder)" in out
        assert "ask the user to confirm" in out
        assert out.startswith("[NIM STATE") and out.endswith("]")

    def test_several_nims_share_one_line(self):
        out = format_nim_notices([{"name": "A", "tools": []}, {"name": "B", "tools": ["t"]}])
        assert "\n" not in out
        assert "A NIM registered but not confirmed;" in out
        assert "B NIM registered but not confirmed (would add t)" in out

    def test_a_nim_with_no_listed_tools_still_reads_cleanly(self):
        assert "would add" not in format_nim_notices([{"name": "A"}])

    def test_the_envelope_carries_it_only_when_there_is_one(self):
        kwargs = {"viewer_state": {"is_empty": True}, "storage_state": {"is_empty": True}, "selections": []}
        quiet = inject_state_into_prompt("BASE", **kwargs)
        loud = inject_state_into_prompt(
            "BASE", **kwargs, nim_notices=[{"name": "RFD4", "tools": ["design_protein_binder"]}]
        )
        assert "[NIM STATE" not in quiet
        assert "[NIM STATE: RFD4 NIM registered but not confirmed" in loud


class TestInjectStateIntoPrompt:
    def test_full_envelope_includes_active_structure(self):
        out = inject_state_into_prompt(
            "BASE",
            viewer_state={"is_empty": True},
            storage_state={
                "is_empty": False,
                "artifacts": [
                    {"reference": "pdb_a", "data_type": "pdb", "description": "apo"}
                ],
            },
            selections=[],
            active_pdb_ref="pdb_a",
        )
        assert out.startswith("BASE")
        assert "[VIEWER STATE" in out
        assert "[STORAGE STATE" in out
        assert "[ACTIVE STRUCTURE" in out
        assert "[SELECTION STATE" in out
        assert "pdb_a" in out

    def test_no_active_structure_when_none_passed(self):
        out = inject_state_into_prompt(
            "BASE",
            viewer_state={"is_empty": True},
            storage_state={"is_empty": True},
            selections=[],
            active_pdb_ref=None,
        )
        assert "[ACTIVE STRUCTURE" not in out
