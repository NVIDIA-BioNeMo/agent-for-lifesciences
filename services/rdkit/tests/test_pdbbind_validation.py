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

"""Does any of this work on real structures?

The unit suites check that each piece does what it says. This one checks that what
it says is worth saying, against crystal structures with known answers: PDBBind
ligands cut into fragment-linker-fragment, the fragments handed over with their
poses, the linker hidden. The criteria come from the plan.

===  ==================  ===================================================
V1   linker sizing       the true path length is inside the recommended range
V2   exit vectors        the true attachment pair is in the top 5
V3   specificity         wrong variants are rejected while the truth is kept
V4   pose recovery       the constrained pose lands near the crystal pose
V6   end to end          the whole loop emits a candidate that clears gates
===  ==================  ===================================================

V5 is Vina's local scoring mode and lives in ``services/vina/tests/test_score_mode.py``,
because it needs the docking binary rather than the PDBBind mount.

The whole module skips without the PDBBind mount, which is how CI runs it. It is
slow -- a few seconds per case, and each case is verified four times -- so the
sample size is an environment variable::

    PDBBIND_CASES=60 pytest tests/test_pdbbind_validation.py -v -s

Failures here are findings, not flakes: they mean the geometry model or the
verifier has drifted away from what real ligands do, and the numbers printed
alongside say in which direction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdMolAlign

from app import constrained_pose as cp
from app import fragment_link as fl
from app import gates
from app import linker_library as lib
from tests.pdbbind_cases import DEFAULT_ROOT, LinkCase, iter_cases

PDBBIND_ROOT = os.environ.get("PDBBIND_ROOT", DEFAULT_ROOT)
CASE_LIMIT = int(os.environ.get("PDBBIND_CASES", "20"))

pytestmark = pytest.mark.skipif(
    not os.path.isdir(os.path.join(PDBBIND_ROOT, "P-L")),
    reason=f"no PDBBind mount at {PDBBIND_ROOT}",
)

# Variants are built by bridging the true anchors with a saturated chain of the
# wrong length, and by bridging the right length from the wrong atom. Lengths are
# in path atoms, against the true path length.
TOO_SHORT = -3
TOO_LONG = 4


@dataclass
class Measured:
    """One case, verified as itself and as three things it is not."""

    case: LinkCase
    sizing_range: tuple[int, int]
    pair_rank: Optional[int]
    true: cp.PoseResult
    recovery: Optional[float]
    variants: dict[str, cp.PoseResult]


def _rank_of_true_pair(case: LinkCase) -> Optional[int]:
    report, error = fl.analyze(case.fragment_sdfs, pocket_pdb=case.pocket_pdb, max_pairs=5)
    if error or not report:
        return None
    for position, pair in enumerate(report["pairs"]):
        if (pair["atom_a"], pair["atom_b"]) == case.true_anchors:
            return position + 1
    return None


def _recovery(case: LinkCase, result: cp.PoseResult) -> Optional[float]:
    """Heavy-atom RMSD of the constrained pose against the crystal pose.

    Symmetry-aware, because a phenyl flipped 180 degrees is the same pose and
    ``GetBestRMS`` would otherwise charge 2 A for it.
    """
    if not result.pose_sdf:
        return None
    crystal = Chem.RemoveHs(Chem.MolFromMolBlock(case.ligand_sdf))
    posed = Chem.RemoveHs(Chem.MolFromMolBlock(result.pose_sdf))
    if crystal is None or posed is None:
        return None
    try:
        return float(rdMolAlign.CalcRMS(posed, crystal))
    except (ValueError, RuntimeError):
        return None


def _variants(case: LinkCase, fragments: list[Chem.Mol], protein: np.ndarray) -> dict[str, str]:
    """Three molecules that pass gates 1-4 and should fail on geometry."""
    anchor_a, anchor_b = case.true_anchors
    path = case.true_path_atoms
    built: dict[str, str] = {}

    for label, atoms in (
        ("too_short", path + TOO_SHORT),
        ("too_long", path + TOO_LONG),
    ):
        if atoms < 1:
            continue
        smiles, _ = lib.assemble(
            fragments[0], anchor_a, fragments[1], anchor_b, "[*:1]" + "C" * atoms + "[*:2]"
        )
        if smiles:
            built[label] = smiles

    # Right length, wrong place: the same chain hung off a different site on the
    # first fragment, as far from the true anchor as the fragment allows.
    positions = fragments[0].GetConformer().GetPositions()
    sites = fl.enumerate_sites(fragments[0], 0, protein)
    elsewhere = sorted(
        (site.atom_index for site in sites if site.atom_index != anchor_a),
        key=lambda index: -float(np.linalg.norm(positions[index] - positions[anchor_a])),
    )
    if elsewhere and path >= 1:
        smiles, _ = lib.assemble(
            fragments[0], elsewhere[0], fragments[1], anchor_b, "[*:1]" + "C" * path + "[*:2]"
        )
        if smiles:
            built["wrong_vector"] = smiles
    return built


@pytest.fixture(scope="module")
def measured() -> list[Measured]:
    rows: list[Measured] = []
    for case in iter_cases(PDBBIND_ROOT, limit=CASE_LIMIT):
        fragments = [Chem.RemoveHs(fl.load_fragment(sdf)[0]) for sdf in case.fragment_sdfs]
        protein = fl.parse_pocket_coords(case.pocket_pdb)

        # Thresholds off for the measurement itself: every test below applies its
        # own, and a result that was gated away carries no numbers to compare.
        true = cp.verify_one(
            case.ligand_smiles, fragments, protein, strain_max=None, rmsd_max=99.0
        )
        rows.append(
            Measured(
                case=case,
                sizing_range=fl.linker_atom_range(case.anchor_distance),
                pair_rank=_rank_of_true_pair(case),
                true=true,
                recovery=_recovery(case, true),
                variants={
                    label: cp.verify_one(
                        smiles, fragments, protein, strain_max=None, rmsd_max=99.0
                    )
                    for label, smiles in _variants(case, fragments, protein).items()
                },
            )
        )
    if not rows:
        pytest.skip(f"no usable cases under {PDBBIND_ROOT}")
    return rows


def _accepts(result: cp.PoseResult) -> bool:
    """The verdict the shipped thresholds would return for this result."""
    if result.failed_gate and result.failed_gate != "gate5c_pose":
        return False
    if result.max_fragment_rmsd is None or result.max_fragment_rmsd > cp.DEFAULT_RMSD_MAX:
        return False
    if result.strain is not None and result.strain > cp.DEFAULT_STRAIN_MAX:
        return False
    return (
        result.min_protein_distance is None
        or result.min_protein_distance >= cp.POSE_CLASH_DISTANCE
    )


class TestV1Sizing:
    def test_the_true_linker_length_is_inside_the_recommended_range(self, measured):
        inside = [
            row for row in measured
            if row.sizing_range[0] <= row.case.true_path_atoms <= row.sizing_range[1]
        ]
        rate = len(inside) / len(measured)
        misses = [
            f"{row.case.pdb_id} d={row.case.anchor_distance:.1f} "
            f"path={row.case.true_path_atoms} range={list(row.sizing_range)}"
            for row in measured
            if row not in inside
        ]
        print(f"\nV1 sizing: {len(inside)}/{len(measured)} = {rate:.0%}")
        for miss in misses:
            print(f"  missed: {miss}")
        assert rate >= 0.90

    def test_the_mask_window_admits_the_whole_hidden_piece(self, measured):
        """The SAFE mask is what GenMol is actually given, so it has to fit."""
        inside = []
        for row in measured:
            low, high = fl.mask_atom_range(*row.sizing_range)
            if low <= row.case.true_linker_atoms <= high:
                inside.append(row)
        print(f"\nmask window: {len(inside)}/{len(measured)} = {len(inside) / len(measured):.0%}")
        assert len(inside) / len(measured) >= 0.90


class TestV2ExitVectors:
    def test_the_true_attachment_pair_ranks_in_the_top_five(self, measured):
        found = [row for row in measured if row.pair_rank is not None]
        rate = len(found) / len(measured)
        ranks = [row.pair_rank for row in found]
        print(
            f"\nV2 exit vectors: {len(found)}/{len(measured)} = {rate:.0%} in the top 5"
            f" (rank 1 for {sum(1 for r in ranks if r == 1)})"
        )
        for row in measured:
            if row.pair_rank is None:
                print(f"  missed: {row.case.pdb_id} true pair {row.case.true_anchors}")
        assert rate >= 0.80


class TestV3Specificity:
    def test_wrong_variants_are_rejected(self, measured):
        by_label: dict[str, list[bool]] = {}
        for row in measured:
            for label, result in row.variants.items():
                by_label.setdefault(label, []).append(not _accepts(result))

        print("\nV3 specificity")
        for label, verdicts in sorted(by_label.items()):
            print(
                f"  {label:12s} rejected {sum(verdicts)}/{len(verdicts)} "
                f"= {sum(verdicts) / len(verdicts):.0%}"
            )
        every = [verdict for verdicts in by_label.values() for verdict in verdicts]
        print(f"  {'overall':12s} rejected {sum(every)}/{len(every)} = {sum(every) / len(every):.0%}")
        assert sum(every) / len(every) >= 0.80

    def test_the_true_molecule_is_kept(self, measured):
        kept = [row for row in measured if _accepts(row.true)]
        rate = len(kept) / len(measured)
        print(f"\nV3 sensitivity: kept {len(kept)}/{len(measured)} = {rate:.0%}")
        for row in measured:
            if row not in kept:
                print(
                    f"  dropped: {row.case.pdb_id} rmsd={row.true.max_fragment_rmsd} "
                    f"strain={row.true.strain} clash={row.true.min_protein_distance}"
                )
        assert rate >= 0.70

    def test_the_separation_is_reported_not_just_the_rate(self, measured):
        """The plan asks for the distributions, so print them."""
        pools = {"true": [row.true for row in measured]}
        for row in measured:
            for label, result in row.variants.items():
                pools.setdefault(label, []).append(result)

        print("\nsignal distributions (median [p10, p90])")
        for label, results in pools.items():
            for name, values in (
                ("fit rmsd", [r.max_fragment_rmsd for r in results if r.max_fragment_rmsd is not None]),
                ("strain", [r.strain for r in results if r.strain is not None]),
                ("protein", [r.min_protein_distance for r in results if r.min_protein_distance is not None]),
            ):
                if values:
                    print(
                        f"  {label:12s} {name:9s} {np.median(values):7.2f} "
                        f"[{np.percentile(values, 10):.2f}, {np.percentile(values, 90):.2f}]"
                    )
        assert pools["true"]


@dataclass
class Solved:
    """What the whole loop produced for one case, without any generative model."""

    case: LinkCase
    num_candidates: int
    survivors: list[tuple[str, cp.PoseResult]]


def _solve(case: LinkCase) -> Solved:
    """Run what ``fragment_link`` runs: geometry, library, verification, Table 6."""
    report, error = fl.analyze(
        case.fragment_sdfs,
        pocket_pdb=case.pocket_pdb,
        max_pairs=3,
        assemble_candidates=True,
        max_candidates_per_pair=8,
    )
    if error or not report:
        return Solved(case=case, num_candidates=0, survivors=[])

    smiles = [candidate["smiles"] for candidate in report.get("candidates") or []]
    if not smiles:
        return Solved(case=case, num_candidates=0, survivors=[])

    results, error = cp.verify(smiles, case.fragment_sdfs, pocket_pdb=case.pocket_pdb)
    if error or not results:
        return Solved(case=case, num_candidates=len(smiles), survivors=[])

    verified = [result for result in results if result.verified]
    if not verified:
        return Solved(case=case, num_candidates=len(smiles), survivors=[])

    report, error = gates.evaluate([result.smiles for result in verified])
    if error or report is None:
        return Solved(case=case, num_candidates=len(smiles), survivors=[])

    passed = {
        row["smiles"]
        for row in report.results
        if all(gate["passed"] for gate in row["gates"])
    }
    return Solved(
        case=case,
        num_candidates=len(smiles),
        survivors=[
            (result.smiles, result) for result in verified if result.smiles in passed
        ],
    )


@pytest.fixture(scope="module")
def solved() -> list[Solved]:
    rows = [_solve(case) for case in iter_cases(PDBBIND_ROOT, limit=CASE_LIMIT)]
    if not rows:
        pytest.skip(f"no usable cases under {PDBBIND_ROOT}")
    return rows


class TestV6EndToEnd:
    """The whole loop, library only -- no GenMol, no network, no GPU.

    This is the floor of the tool's usefulness rather than its ceiling: a case
    the library cannot solve may still be solved by a generated linker. What the
    floor being non-trivial buys is a tool that answers something on its own,
    within milliseconds, before any model or oracle is involved.
    """

    def test_the_loop_emits_a_candidate_that_clears_every_gate(self, solved):
        won = [row for row in solved if row.survivors]
        rate = len(won) / len(solved)
        print(
            f"\nV6 end to end: {len(won)}/{len(solved)} = {rate:.0%} solved from the "
            f"library alone"
        )
        for row in solved:
            if not row.survivors:
                print(
                    f"  unsolved: {row.case.pdb_id} d={row.case.anchor_distance:.1f} "
                    f"from {row.num_candidates} candidate(s)"
                )
        assert rate >= 0.30

    def test_a_survivor_holds_its_poses_cheaply(self, solved):
        """A survivor with 300 kcal/mol of strain is a pass on paper only."""
        strains = [
            result.strain
            for row in solved
            for _smiles, result in row.survivors
            if result.strain is not None
        ]
        if not strains:
            pytest.skip("no survivors to measure")
        print(
            f"\nV6 survivor strain: median {np.median(strains):.1f} "
            f"[{np.percentile(strains, 10):.1f}, {np.percentile(strains, 90):.1f}] "
            f"kcal/mol over {len(strains)} survivor(s)"
        )
        assert np.median(strains) < cp.DEFAULT_STRAIN_MAX


class TestV4PoseRecovery:
    def test_the_constrained_pose_lands_on_the_crystal_pose(self, measured):
        trusted = [
            row for row in measured
            if row.recovery is not None
            and (row.true.strain is None or row.true.strain <= cp.DEFAULT_STRAIN_MAX)
        ]
        recoveries = [row.recovery for row in trusted]
        under = [value for value in recoveries if value < 1.0]
        print(
            f"\nV4 pose recovery: n={len(recoveries)} median={np.median(recoveries):.2f} "
            f"under 1.0 A {len(under)}/{len(recoveries)} = {len(under) / len(recoveries):.0%} "
            f"max={max(recoveries):.2f}"
        )
        assert np.median(recoveries) < 1.0
        assert len(under) / len(recoveries) >= 0.70
