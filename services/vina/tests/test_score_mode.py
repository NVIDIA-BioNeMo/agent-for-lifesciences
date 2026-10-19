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

"""Tests for pose-preserving scoring: ``score_mode`` of search, local or score.

The point of these modes is that the caller already has a pose worth keeping --
a constrained embedding, a crystal ligand, a scaffold hop off a known binder --
and a global search would throw it away. So the tests that matter are about what
happens to the pose, not about the number that comes back.

Criterion V5 from the plan is :class:`TestLocalMinimisation`: from the 3ert
crystal pose, local minimisation moves the ligand less than 1 A and lands within
about 1 kcal/mol of what a search finds. It runs only where the ``vina`` binary
exists, in the manner of the probe-grid golden test.

Run with: pytest services/vina/tests/test_score_mode.py
"""

import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.docking import parse_scored_pose, run_docking
from app.main import app
from app.preparation import prepare_ligand, prepare_receptor

DATA = Path(__file__).parent / "data"
HAS_VINA = shutil.which("vina") is not None

client = TestClient(app)


def coordinates(pdbqt: str) -> np.ndarray:
    return np.array(
        [
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            for line in pdbqt.splitlines()
            if line.startswith(("ATOM", "HETATM"))
        ]
    )


def rmsd(first: np.ndarray, second: np.ndarray) -> float:
    """Straight per-atom RMSD. No superposition -- movement is the measurement."""
    assert first.shape == second.shape
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


@pytest.fixture(scope="module")
def crystal_case():
    """The 3ert pocket and its tamoxifen ligand, prepared, plus the box centre."""
    receptor, _atoms, error = prepare_receptor(
        (DATA / "pocket.pdb").read_text(), remove_water=True, remove_heteroatoms=True
    )
    assert error is None, error
    ligand, _n, _rot, error = prepare_ligand(sdf_content=(DATA / "ligand.sdf").read_text())
    assert error is None, error
    center = tuple(float(value) for value in (DATA / "pocket_center.txt").read_text().split())
    return receptor, ligand, center


LOCAL_ONLY_STDOUT = """
Computing Vina grid ... done.
Performing local search ... done.
Estimated Free Energy of Binding   : -9.371 (kcal/mol) [=(1)+(2)+(3)-(4)]
(1) Final Intermolecular Energy    : -14.027 (kcal/mol)
"""

MINIMAL_POSE = """REMARK SMILES c1ccccc1
ROOT
ATOM      1  C   UNL     1       1.000   2.000   3.000  1.00  0.00     0.000 C
ENDROOT
TORSDOF 0
"""


class TestParsing:
    """``--local_only`` and ``--score_only`` do not write REMARK VINA RESULT."""

    def test_the_energy_is_read_from_stdout(self):
        poses = parse_scored_pose(LOCAL_ONLY_STDOUT, MINIMAL_POSE)
        assert len(poses) == 1
        assert poses[0].affinity == pytest.approx(-9.371)
        assert poses[0].rank == 1

    def test_the_pose_is_carried_through_without_the_remarks(self):
        poses = parse_scored_pose(LOCAL_ONLY_STDOUT, MINIMAL_POSE)
        assert "REMARK" not in poses[0].pdbqt
        assert "ATOM      1  C" in poses[0].pdbqt
        assert coordinates(poses[0].pdbqt).tolist() == [[1.0, 2.0, 3.0]]

    def test_no_energy_line_means_no_pose(self):
        assert parse_scored_pose("Computing Vina grid ... done.", MINIMAL_POSE) == []

    def test_no_atoms_means_no_pose(self):
        assert parse_scored_pose(LOCAL_ONLY_STDOUT, "REMARK nothing here") == []


class TestValidation:
    def test_an_unknown_mode_is_refused_before_anything_runs(self):
        success, poses, affinity, elapsed, gpu, error = run_docking(
            receptor_pdbqt="ATOM", ligand_pdbqt="ATOM",
            center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0),
            score_mode="minimise",
        )
        assert not success
        assert "minimise" in error
        assert elapsed == 0

    def test_the_schema_rejects_an_unknown_mode(self):
        response = client.post(
            "/dock",
            json={
                "receptor_pdbqt": "ATOM",
                "ligand_pdbqt": "ATOM",
                "center_x": 0.0,
                "center_y": 0.0,
                "center_z": 0.0,
                "score_mode": "minimise",
            },
        )
        assert response.status_code == 422

    def test_search_is_still_the_default(self):
        from app.models import DockAutoRequest, DockRequest

        assert DockRequest(
            receptor_pdbqt="x", ligand_pdbqt="y", center_x=0, center_y=0, center_z=0
        ).score_mode == "search"
        assert DockAutoRequest(
            receptor_pdb="x", ligand_sdf="y", center_x=0, center_y=0, center_z=0
        ).score_mode == "search"


@pytest.mark.skipif(not HAS_VINA, reason="vina binary not installed")
class TestLocalMinimisation:
    """Criterion V5, against the 3ert crystal pose."""

    @pytest.fixture(scope="class")
    def results(self, crystal_case):
        receptor, ligand, center = crystal_case
        box = (20.0, 20.0, 20.0)
        outcome = {}
        for mode in ("score", "local", "search"):
            success, poses, affinity, _ms, _gpu, error = run_docking(
                receptor_pdbqt=receptor,
                ligand_pdbqt=ligand,
                center=center,
                size=box,
                exhaustiveness=8,
                score_mode=mode,
            )
            assert success, f"{mode}: {error}"
            outcome[mode] = (poses, affinity)
        return ligand, outcome

    def test_score_mode_returns_the_pose_it_was_given(self, results):
        ligand, outcome = results
        poses, affinity = outcome["score"]
        assert len(poses) == 1
        assert rmsd(coordinates(poses[0].pdbqt), coordinates(ligand)) == 0.0
        assert affinity < 0

    def test_local_minimisation_keeps_the_pose(self, results):
        ligand, outcome = results
        poses, _affinity = outcome["local"]
        assert len(poses) == 1
        moved = rmsd(coordinates(poses[0].pdbqt), coordinates(ligand))
        assert moved < 1.0, f"local minimisation moved the ligand {moved:.2f} A"

    def test_local_minimisation_improves_on_the_raw_score(self, results):
        """Minimising can only help, or the minimiser is not one."""
        _ligand, outcome = results
        assert outcome["local"][1] <= outcome["score"][1] + 0.01

    def test_local_affinity_is_close_to_what_a_search_finds(self, results):
        """The crystal pose is near the global optimum, so the two should agree."""
        _ligand, outcome = results
        assert outcome["local"][1] == pytest.approx(outcome["search"][1], abs=1.5)

    def test_a_search_is_free_to_move_the_ligand_and_does(self, results):
        ligand, outcome = results
        poses, _affinity = outcome["search"]
        assert len(poses) > 1
        worst = max(rmsd(coordinates(p.pdbqt), coordinates(ligand)) for p in poses)
        assert worst > 1.0


@pytest.mark.skipif(not HAS_VINA, reason="vina binary not installed")
class TestEndpoints:
    def test_dock_reports_the_mode_it_ran(self, crystal_case):
        receptor, ligand, center = crystal_case
        response = client.post(
            "/dock",
            json={
                "receptor_pdbqt": receptor,
                "ligand_pdbqt": ligand,
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "score_mode": "local",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True, body["error"]
        assert body["docking_mode"] == "local"
        assert body["gpu_accelerated"] is False
        assert len(body["poses"]) == 1

    def test_dock_auto_keeps_the_supplied_conformer(self, crystal_case):
        """A ligand over the flexibility threshold would normally be re-embedded.

        The receptor goes in pre-prepared because ``/dock-auto`` mis-detects this
        fixture's PDB as PDBQT already -- it carries formal charges in the element
        column -- which fails identically in every score mode and is not what this
        test is about.
        """
        receptor, _ligand, center = crystal_case
        response = client.post(
            "/dock-auto",
            json={
                "receptor_pdb": receptor,
                "ligand_sdf": (DATA / "ligand.sdf").read_text(),
                "center_x": center[0],
                "center_y": center[1],
                "center_z": center[2],
                "score_mode": "local",
                "rot_bond_threshold": 0,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True, body["error"]
        assert body["docking_mode"] == "local"
        assert body["conformers_generated"] == 0

        crystal = coordinates(
            prepare_ligand(sdf_content=(DATA / "ligand.sdf").read_text())[0]
        )
        assert rmsd(coordinates(body["poses"][0]["pdbqt"]), crystal) < 1.0


@pytest.mark.skipif(not HAS_VINA, reason="vina binary not installed")
class TestNoConformerWorker:
    """Only a GPU ensemble search generates conformers; nothing else reaches the worker."""

    @pytest.fixture
    def gpu_pipeline(self, monkeypatch):
        """The pipeline with GPU routing on and every conformer route booby-trapped."""
        from app import conformer_client, pipeline

        def forbidden(*_args, **_kwargs):
            raise AssertionError("conformer generation / worker reached")

        monkeypatch.setattr(pipeline, "VINA_GPU_ENABLED", True)
        monkeypatch.setattr(pipeline, "generate_conformers_with_report", forbidden)
        monkeypatch.setattr(conformer_client, "get_client", forbidden)
        return pipeline

    @pytest.mark.parametrize("mode", ["score", "local"])
    def test_pose_preserving_modes(self, gpu_pipeline, crystal_case, mode):
        receptor, _ligand, center = crystal_case
        result = gpu_pipeline.run_auto_dock(
            receptor_pdb=receptor,
            ligand_sdf=(DATA / "ligand.sdf").read_text(),
            center=center,
            size=(20.0, 20.0, 20.0),
            exhaustiveness=1,
            rot_bond_threshold=0,
            score_mode=mode,
        )
        assert result["success"] is True, result["error"]
        assert result["docking_mode"] == mode
        assert result["conformers_generated"] == 0

    def test_flexible_search(self, crystal_case, monkeypatch):
        from app import conformer_client, pipeline

        def forbidden(*_args, **_kwargs):
            raise AssertionError("conformer generation / worker reached")

        monkeypatch.setattr(pipeline, "generate_conformers_with_report", forbidden)
        monkeypatch.setattr(conformer_client, "get_client", forbidden)
        receptor, _ligand, center = crystal_case
        result = pipeline.run_auto_dock(
            receptor_pdb=receptor,
            ligand_sdf=(DATA / "ligand.sdf").read_text(),
            center=center,
            size=(20.0, 20.0, 20.0),
            exhaustiveness=1,
            num_modes=1,
            rot_bond_threshold=100,
        )
        assert result["success"] is True, result["error"]
        assert result["docking_mode"] == "flexible"
        assert result["conformers_generated"] == 0
