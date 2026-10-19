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

"""Tests for exit-vector geometry and linker sizing.

Two committed PDBBind cases carry the load, both built by
``tests/pdbbind_cases.py`` exactly the way SMDD-Bench builds a Fragment Assembly
task -- BRICS-decompose a crystal ligand, keep two fragments of at least six heavy
atoms at least 6 A apart, hide the rest:

- **1uwh**: sorafenib in BRAF. Chlorophenyl and pyridine, anchors 10.0 A apart,
  eight linker atoms hidden between them. The long-linker regime, where a
  saturated chain cannot pay the rotatable-bond gate.
- **4o76**: a pyrimidine kinase inhibitor. Pyrimidine and N-methylpiperazine,
  anchors 6.6 A apart, five linker atoms. The short end of the same problem.

Because the answer is known, the assertions are about recovering it: the true
attachment pair should rank near the top, and the true linker length should fall
inside the recommended window. The wider PDBBind sweep behind those two numbers
lives in ``test_pdbbind_validation.py``.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

from app import fragment_link as fl
from app.main import app
from tests.pdbbind_cases import fixture_ids, load_fixture

client = TestClient(app)

BENZENE_SDF = """benzene
     RDKit          3D

  6  6  0  0  0  0  0  0  0  0999 V2000
    1.4000    0.0000    0.0000 C   0  0
    0.7000    1.2124    0.0000 C   0  0
   -0.7000    1.2124    0.0000 C   0  0
   -1.4000    0.0000    0.0000 C   0  0
   -0.7000   -1.2124    0.0000 C   0  0
    0.7000   -1.2124    0.0000 C   0  0
  1  2  2  0
  2  3  1  0
  3  4  2  0
  4  5  1  0
  5  6  2  0
  6  1  1  0
M  END
"""


# A slab along +x, which only the C0 vector points at. Two things make it a wall
# rather than an obstacle: it is wide enough that a 45 degree cone cannot bend
# past its edge, and it is close enough that not even one atom fits in front of
# it -- 2.6 A from the anchor, against the 2.4 A clearance a probe point needs.
WALL = np.array(
    [[4.0, y, z] for y in np.arange(-9.0, 9.5, 1.5) for z in np.arange(-9.0, 9.5, 1.5)]
)


def shifted_benzene(offset: tuple[float, float, float]) -> str:
    """A second benzene, translated, for building a synthetic two-fragment case."""
    mol = Chem.MolFromMolBlock(BENZENE_SDF)
    conformer = mol.GetConformer()
    for index in range(mol.GetNumAtoms()):
        position = conformer.GetAtomPosition(index)
        conformer.SetAtomPosition(
            index, (position.x + offset[0], position.y + offset[1], position.z + offset[2])
        )
    return Chem.MolToMolBlock(mol)


class TestFragmentLoading:
    def test_loads_a_posed_fragment(self):
        mol, error = fl.load_fragment(BENZENE_SDF)
        assert error is None
        assert mol.GetNumHeavyAtoms() == 6
        assert mol.GetConformer().Is3D()

    def test_empty_input_is_reported(self):
        mol, error = fl.load_fragment("")
        assert mol is None
        assert "empty" in error

    def test_unparseable_input_is_reported(self):
        mol, error = fl.load_fragment("not an sdf at all")
        assert mol is None
        assert "parse" in error

    def test_two_dimensional_coordinates_are_rejected(self):
        flat = Chem.MolToMolBlock(Chem.MolFromSmiles("c1ccccc1"))
        mol, error = fl.load_fragment(flat)
        assert mol is None
        assert "2D" in error


class TestPocketParsing:
    def test_reads_heavy_atoms_and_skips_hydrogens_and_water(self):
        pdb = (
            "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00  0.00           C\n"
            "ATOM      3  H   ALA A   1      11.500  10.000  10.000  1.00  0.00           H\n"
            "HETATM    4  O   HOH A   2      20.000  20.000  20.000  1.00  0.00           O\n"
        )
        coords = fl.parse_pocket_coords(pdb)
        assert coords.shape == (2, 3)
        assert np.allclose(coords[0], [10.0, 10.0, 10.0])

    def test_no_pocket_is_an_empty_array_not_an_error(self):
        assert fl.parse_pocket_coords("").shape == (0, 3)


class TestSites:
    def test_benzene_has_six_symmetric_sites(self):
        mol, _ = fl.load_fragment(BENZENE_SDF)
        sites = fl.enumerate_sites(mol, 0, np.zeros((0, 3)))
        assert len(sites) == 6
        # Every exit vector points radially out of the ring.
        for site in sites:
            assert site.outwardness > 0.99
            assert abs(np.linalg.norm(site.direction) - 1.0) < 1e-6

    def test_atoms_without_a_hydrogen_are_not_sites(self):
        # Naphthalene's two fusion carbons carry no hydrogen.
        mol = Chem.AddHs(Chem.MolFromSmiles("c1ccc2ccccc2c1"))
        from rdkit.Chem import AllChem

        AllChem.EmbedMolecule(mol, randomSeed=3)
        sites = fl.enumerate_sites(mol, 0, np.zeros((0, 3)))
        assert len(sites) == 8

    def test_a_wall_of_protein_closes_the_vectors_facing_it(self):
        mol, _ = fl.load_fragment(BENZENE_SDF)
        sites = {site.atom_index: site for site in fl.enumerate_sites(mol, 0, WALL)}
        assert sites[0].free_depth < 1.5
        assert sites[0].blocked
        assert sites[3].free_depth == fl.MAX_DEPTH
        assert not sites[3].blocked

    def test_a_patch_too_small_to_grow_around_is_not_a_wall(self):
        """The probe is a cone, so what obstructs it has to obstruct the cone.

        A nine-atom patch four angstrom out does not: a substituent is free to
        bend past its edge, and reporting that vector as blocked is the error
        that made real pockets read as walls.
        """
        mol, _ = fl.load_fragment(BENZENE_SDF)
        patch = np.array([[5.4, y, z] for y in (-2, 0, 2) for z in (-2, 0, 2)])
        sites = {site.atom_index: site for site in fl.enumerate_sites(mol, 0, patch)}
        assert sites[0].free_depth > fl.BOND_LENGTH
        assert not sites[0].blocked

    def test_targets_reorder_the_sites(self):
        mol, _ = fl.load_fragment(BENZENE_SDF)
        target = [[8.0, 0.0, 0.0]]
        sites = fl.enumerate_sites(
            mol, 0, np.zeros((0, 3)), np.array(target, dtype=float)
        )
        assert sites[0].atom_index == 0
        assert sites[0].target_alignment > 0.99
        assert sites[0].target_distance == pytest.approx(6.6, abs=0.1)

    def test_the_deepest_vector_does_not_win_by_pointing_out_of_the_protein(self):
        """Depth is necessary and not sufficient, so burial has to break the tie.

        The wall in front of C0 is a slab with a gap in it, so C0 has real room
        and real protein around it. C3 points away from the slab entirely: it has
        unlimited depth and nothing to bind, and ranking it first is how growth
        gets aimed into solvent.
        """
        mol, _ = fl.load_fragment(BENZENE_SDF)
        # A cup around C0: the slab, plus flanking walls, all open along +x.
        cup = np.vstack(
            [
                [[x, y, 6.0] for x in np.arange(2.0, 12.5, 1.5) for y in np.arange(-6.0, 6.5, 1.5)],
                [[x, y, -6.0] for x in np.arange(2.0, 12.5, 1.5) for y in np.arange(-6.0, 6.5, 1.5)],
                [[x, 6.0, z] for x in np.arange(2.0, 12.5, 1.5) for z in np.arange(-6.0, 6.5, 1.5)],
                [[x, -6.0, z] for x in np.arange(2.0, 12.5, 1.5) for z in np.arange(-6.0, 6.5, 1.5)],
            ]
        )
        sites = {site.atom_index: site for site in fl.enumerate_sites(mol, 0, cup)}
        assert sites[0].free_depth == fl.MAX_DEPTH
        assert sites[3].free_depth == fl.MAX_DEPTH
        assert sites[0].burial > sites[3].burial
        assert sites[0].score > sites[3].score

    def test_burial_is_absent_rather_than_zero_without_a_pocket(self):
        mol, _ = fl.load_fragment(BENZENE_SDF)
        sites = fl.enumerate_sites(mol, 0, np.zeros((0, 3)))
        assert all(site.burial is None for site in sites)
        # And a site with nothing measured still scores on its geometry alone.
        assert sites[0].score > 0.0

    def test_a_blocked_vector_cannot_outrank_an_open_one_by_aiming(self):
        mol, _ = fl.load_fragment(BENZENE_SDF)
        # The target sits behind the wall, so C0 points straight at it and has
        # nowhere to grow. It must not come first.
        sites = fl.enumerate_sites(mol, 0, WALL, np.array([[9.0, 0.0, 0.0]]))
        assert sites[0].atom_index != 0


class TestLinkerSizing:
    @pytest.mark.parametrize(
        "distance,expected_min,expected_max",
        [(2.5, 1, 4), (6.0, 3, 7), (8.0, 5, 9), (10.0, 6, 11)],
    )
    def test_window_matches_the_documented_formula(self, distance, expected_min, expected_max):
        assert fl.linker_atom_range(distance) == (expected_min, expected_max)

    def test_window_never_goes_below_one_atom(self):
        lower, upper = fl.linker_atom_range(0.5)
        assert lower == 1
        assert upper >= lower

    def test_window_agrees_with_the_measured_library(self):
        from app import linker_library as lib

        for distance in (4.0, 6.0, 8.0, 10.0):
            lower, upper = fl.linker_atom_range(distance)
            for geometry in lib.matching_linkers(distance, one_per_family=False):
                if geometry.path_atoms == 0:
                    continue
                assert lower <= geometry.path_atoms <= upper, (
                    f"{geometry.name} spans {distance} A with "
                    f"{geometry.path_atoms} path atoms, outside [{lower}, {upper}]"
                )

    def test_mask_window_admits_the_ring_bearing_bridges(self):
        from app import linker_library as lib

        for distance in (6.0, 8.0, 10.0):
            path_window = fl.linker_atom_range(distance)
            mask_min, mask_max = fl.mask_atom_range(*path_window)
            assert mask_max > path_window[1]
            for geometry in lib.matching_linkers(distance, one_per_family=False):
                if geometry.num_linker_atoms == 0:
                    continue
                assert mask_min <= geometry.num_linker_atoms <= mask_max, (
                    f"{geometry.name} needs {geometry.num_linker_atoms} atoms, "
                    f"outside the mask window [{mask_min}, {mask_max}] for {distance} A"
                )

    def test_chain_cost_is_one_bond_more_than_the_atom_count(self):
        assert fl.saturated_chain_rb_cost(6) == 7


class TestPairGeometry:
    def test_facing_vectors_score_higher_than_averted_ones(self):
        sdf_a = BENZENE_SDF
        sdf_b = shifted_benzene((9.0, 0.0, 0.0))
        report, error = fl.analyze([sdf_a, sdf_b])
        assert error is None
        best = report["pairs"][0]
        assert best["exit_vector_angle"] > 150
        assert best["alignment_a"] > 0.9
        assert best["alignment_b"] > 0.9
        assert best["anchor_distance"] == pytest.approx(6.2, abs=0.3)

    def test_separation_is_reported(self):
        report, _ = fl.analyze([BENZENE_SDF, shifted_benzene((9.0, 0.0, 0.0))])
        separation = report["fragment_separation"][0]
        assert separation["closest_heavy_distance"] == pytest.approx(6.2, abs=0.3)
        assert separation["centroid_distance"] == pytest.approx(9.0, abs=0.1)

    def test_tube_clearance_sees_an_obstruction_between_the_anchors(self):
        blocker = np.array([[4.5, 0.0, 0.0]])
        pdb = "".join(
            f"ATOM  {index:5d}  C   ALA A   1    {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
            "  1.00  0.00           C\n"
            for index, x in enumerate(blocker[:, 0], start=1)
        )
        report, _ = fl.analyze([BENZENE_SDF, shifted_benzene((9.0, 0.0, 0.0))], pocket_pdb=pdb)
        best = report["pairs"][0]
        assert best["tube_clearance"] is not None
        assert best["tube_clearance"] < 1.0

    def test_rotatable_bond_headroom_charges_for_the_two_new_bonds(self):
        report, _ = fl.analyze([BENZENE_SDF, shifted_benzene((9.0, 0.0, 0.0))])
        assert report["rotatable_bonds_used"] == 0
        assert report["rb_headroom"] == fl.DEFAULT_RB_MAX - fl.NEW_BOND_RB_COST

    def test_a_long_span_on_a_tight_budget_demands_a_rigid_linker(self):
        report, _ = fl.analyze(
            [BENZENE_SDF, shifted_benzene((14.0, 0.0, 0.0))], rb_max=4
        )
        best = report["pairs"][0]
        assert best["requires_rigid_linker"]
        assert any("rigid" in warning for warning in best["warnings"])


class TestTemplates:
    def test_marker_form_carries_one_attachment_point_per_fragment(self):
        mol_a, _ = fl.load_fragment(BENZENE_SDF)
        mol_b, _ = fl.load_fragment(shifted_benzene((9.0, 0.0, 0.0)))
        template, error = fl.marker_template(mol_a, 0, mol_b, 3)
        assert error is None
        assert template.count("*") == 2
        assert len(template.split(".")) == 2

    def test_safe_form_leaves_the_mask_between_the_fragments(self):
        mol_a, _ = fl.load_fragment(BENZENE_SDF)
        mol_b, _ = fl.load_fragment(shifted_benzene((9.0, 0.0, 0.0)))
        template, error = fl.safe_template(mol_a, 0, 3, 7, mol_b, 3)
        assert error is None
        tokens = template.split(".")
        assert len(tokens) == 3
        assert tokens[1] == "[*{3-7}]"
        # The two bonds the mask will make are the ring-closure digits its
        # neighbours are left holding.
        digits = [character for token in tokens for character in token if character.isdigit()]
        assert digits, "no ring-closure bookkeeping survived the encoding"

    def test_safe_form_for_a_single_fragment_is_a_growth_template(self):
        mol, _ = fl.load_fragment(BENZENE_SDF)
        template, error = fl.safe_template(mol, 0, 2, 5)
        assert error is None
        assert template.endswith(".[*{2-5}]")

    def test_placeholder_avoids_an_element_the_fragment_already_has(self):
        from rdkit.Chem import AllChem

        mol = Chem.AddHs(Chem.MolFromSmiles("[SiH3]c1ccccc1"))
        AllChem.EmbedMolecule(mol, randomSeed=5)
        # Atom 3 is a ring CH; atom 0 is the silicon the placeholder must avoid.
        template, error = fl.safe_template(mol, 3, 2, 4)
        assert error is None, error
        assert template.endswith(".[*{2-4}]")
        assert "Ge" not in template, "the placeholder itself must not survive"

    def test_templates_appear_on_every_reported_pair(self):
        report, _ = fl.analyze([BENZENE_SDF, shifted_benzene((9.0, 0.0, 0.0))])
        for pair in report["pairs"]:
            assert pair["templates"]["safe"], pair["templates"]["safe_error"]
            assert pair["templates"]["marker"], pair["templates"]["marker_error"]
            assert pair["templates"]["mask_min"] >= 1


class TestGrowth:
    def test_one_fragment_gives_growth_vectors_not_pairs(self):
        report, error = fl.analyze([BENZENE_SDF])
        assert error is None
        assert report["pairs"] == []
        assert report["growth"]["vectors"]
        assert report["fragment_separation"] is None

    def test_growth_vectors_are_aimed_at_supplied_hotspots(self):
        report, _ = fl.analyze([BENZENE_SDF], target_points=[[8.0, 0.0, 0.0]])
        assert report["growth"]["vectors"][0]["atom_index"] == 0


class TestAssembledCandidates:
    def test_candidates_are_only_built_when_asked(self):
        without, _ = fl.analyze([BENZENE_SDF, shifted_benzene((9.0, 0.0, 0.0))])
        assert without["candidates"] == []
        with_candidates, _ = fl.analyze(
            [BENZENE_SDF, shifted_benzene((9.0, 0.0, 0.0))], assemble_candidates=True
        )
        assert with_candidates["candidates"]

    def test_every_candidate_is_a_real_molecule_containing_both_fragments(self):
        report, _ = fl.analyze(
            [BENZENE_SDF, shifted_benzene((9.0, 0.0, 0.0))], assemble_candidates=True
        )
        benzene = Chem.MolFromSmiles("c1ccccc1")
        for candidate in report["candidates"]:
            mol = Chem.MolFromSmiles(candidate["smiles"])
            assert mol is not None, candidate["smiles"]
            assert len(mol.GetSubstructMatches(benzene)) >= 2


class TestErrors:
    def test_no_fragments(self):
        report, error = fl.analyze([])
        assert report is None
        assert "no fragments" in error

    def test_bad_fragment_names_its_index(self):
        report, error = fl.analyze([BENZENE_SDF, "garbage"])
        assert report is None
        assert error.startswith("fragment 1")

    def test_malformed_target_points(self):
        report, error = fl.analyze([BENZENE_SDF], target_points=[[1.0, 2.0]])
        assert report is None
        assert "target_points" in error

    def test_missing_pocket_is_a_warning_not_an_error(self):
        report, error = fl.analyze([BENZENE_SDF, shifted_benzene((9.0, 0.0, 0.0))])
        assert error is None
        assert any("No pocket" in warning for warning in report["warnings"])


class TestCommittedCases:
    """The known-answer cases: does the geometry recover what was hidden?"""

    @pytest.mark.parametrize("pdb_id", fixture_ids())
    def test_true_linker_length_is_inside_the_recommended_window(self, pdb_id):
        """Criterion V1, per case. The window predicts *path* atoms, so that is
        what it is checked against; the hidden piece can carry more than it spans.
        """
        case = load_fixture(pdb_id)
        lower, upper = fl.linker_atom_range(case.anchor_distance)
        assert lower <= case.true_path_atoms <= upper
        mask_low, mask_high = fl.mask_atom_range(lower, upper)
        assert mask_low <= case.true_linker_atoms <= mask_high

    def test_the_true_attachment_pair_usually_ranks_in_the_top_five(self):
        """Criterion V2, over the committed cases.

        Aggregate rather than per-case on purpose. Ranking exit vectors is a
        prediction, not a computation -- it is right about four cases in five over
        PDBBind -- so asserting it per fixture would only record which fixtures
        happened to be easy. ``test_pdbbind_validation.py`` measures the rate that
        the 80% target is actually about.
        """
        found = 0
        for pdb_id in fixture_ids():
            case = load_fixture(pdb_id)
            report, error = fl.analyze(
                case.fragment_sdfs, pocket_pdb=case.pocket_pdb, max_pairs=5
            )
            assert error is None, error
            ranked = [(pair["atom_a"], pair["atom_b"]) for pair in report["pairs"]]
            found += case.true_anchors in ranked
        assert found >= len(fixture_ids()) - 1

    def test_sorafenib_span_is_expensive_for_a_chain_and_cheap_for_a_rod(self):
        case = load_fixture("1uwh")
        report, _ = fl.analyze(case.fragment_sdfs, pocket_pdb=case.pocket_pdb)
        assert report["pocket_atoms"] > 500
        best = report["pairs"][0]
        assert best["anchor_distance"] > 8.0
        # Sorafenib's own fragments are two rigid rings, so the chain that spans
        # 10 A does fit under RB <= 10 -- at the cost of most of the budget.
        assert best["saturated_chain_rb_cost"] >= 6
        assert not best["requires_rigid_linker"]
        offered = {entry["name"] for entry in best["linkers"]}
        assert offered, "no library entry offered for a 10 A span"
        # Every bridge the library offers for this span is cheaper than the chain,
        # which is the whole reason to look one up.
        cheapest = min(entry["rb_cost"] for entry in best["linkers"])
        assert cheapest < best["saturated_chain_rb_cost"]
        from app import linker_library as lib

        assert all(lib.LINKERS_BY_NAME[name].rigid for name in offered), offered

    def test_a_flexible_fragment_pair_is_told_to_use_a_rigid_bridge(self):
        # Same span, but the fragments have already spent the budget: the two new
        # bonds plus a chain long enough to reach can no longer fit.
        case = load_fixture("1uwh")
        report, _ = fl.analyze(case.fragment_sdfs, pocket_pdb=case.pocket_pdb, rb_max=6)
        best = report["pairs"][0]
        assert best["requires_rigid_linker"]
        assert any("rigid" in warning for warning in best["warnings"])

    def test_committed_cases_carry_a_pocket_and_two_fragments(self):
        assert fixture_ids(), "no committed fixtures found"
        for pdb_id in fixture_ids():
            case = load_fixture(pdb_id)
            assert len(case.fragment_sdfs) == 2
            assert len(fl.parse_pocket_coords(case.pocket_pdb)) > 100
            assert case.separation >= 6.0


class TestEndpoint:
    def test_link_geometry_endpoint_returns_pairs(self):
        response = client.post(
            "/link-geometry",
            json={
                "fragment_sdfs": [BENZENE_SDF, shifted_benzene((9.0, 0.0, 0.0))],
                "assemble_candidates": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["pairs"]
        assert body["candidates"]
        assert body["rb_max"] == fl.DEFAULT_RB_MAX

    def test_link_geometry_endpoint_reports_a_bad_fragment(self):
        response = client.post("/link-geometry", json={"fragment_sdfs": ["nope"]})
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert "fragment 0" in body["error"]

    def test_link_geometry_endpoint_rejects_an_empty_fragment_list(self):
        response = client.post("/link-geometry", json={"fragment_sdfs": []})
        assert response.status_code == 422

    def test_linker_library_endpoint_lists_measured_entries(self):
        response = client.get("/linker-library")
        assert response.status_code == 200
        body = response.json()
        assert body["unmeasurable"] == []
        names = {entry["name"] for entry in body["linkers"]}
        assert "phenyl_1_4" in names
        entry = next(item for item in body["linkers"] if item["name"] == "phenyl_1_4")
        assert entry["span_min"] == pytest.approx(5.79, abs=0.15)
        assert entry["rigid"] is True
        assert entry["note"]
