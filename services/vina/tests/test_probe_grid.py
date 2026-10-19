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

"""Tests for probe interaction-energy grids.

Six checks, in the order they were used to validate the implementation:

- V1 the vectorised grid reproduces a scalar transcription of the published
  scoring function,
- V2 it reproduces ``vina --write_maps`` point for point (the golden test),
- V3 the repulsive wall is where the protein is,
- V4 hydrogen-bond geometry falls out of the energy rather than being imposed,
- V5 the grid prefers real ligand positions over distance-matched decoys,
- V6 cropping the receptor to the box changes nothing.

V2 is the one that catches real bugs: it caught sulfur being left out of the
heteroatom test in hydrophobic typing, which had quietly turned every cysteine
and methionine carbon into a hydrophobe. It runs only where the ``vina`` binary
exists, which in practice means the vina test image.

The remaining tests pin the residue-template typing, which deliberately
*disagrees* with Vina's own perception -- that is the point of it -- and so
cannot be checked against the maps.

Run with: pytest services/vina/tests/test_probe_grid.py
"""

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.preparation import prepare_receptor
from app.probe_grid import (
    CUTOFF,
    PROBES,
    RESIDUE_TEMPLATES,
    W_GAUSS1,
    W_GAUSS2,
    W_HBOND,
    W_HYDROPHOBIC,
    W_REPULSION,
    Receptor,
    build_receptor,
    compute_grids,
    crop_to_box,
    find_minima,
    grid_axes,
    run_probe_grid,
)

DATA = Path(__file__).parent / "data"
HAS_VINA = shutil.which("vina") is not None

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def pocket_pdb() -> str:
    return (DATA / "pocket.pdb").read_text()


@pytest.fixture(scope="module")
def center() -> tuple[float, float, float]:
    values = (DATA / "pocket_center.txt").read_text().split()
    return (float(values[0]), float(values[1]), float(values[2]))


@pytest.fixture(scope="module")
def receptor_pdbqt(pocket_pdb) -> str:
    pdbqt, _atoms, error = prepare_receptor(pdb_content=pocket_pdb)
    assert error is None, error
    return pdbqt


@pytest.fixture(scope="module")
def perceived(receptor_pdbqt) -> Receptor:
    """Typed the way Vina types a prepared receptor."""
    return build_receptor(receptor_pdbqt, use_templates=False)


@pytest.fixture(scope="module")
def templated(receptor_pdbqt) -> Receptor:
    """Typed from residue chemistry, which is the production path."""
    return build_receptor(receptor_pdbqt, use_templates=True)


@pytest.fixture(scope="module")
def ligand_atoms() -> list[tuple[str, np.ndarray]]:
    """Heavy atoms of the co-crystal ligand as (element, xyz)."""
    lines = (DATA / "ligand.sdf").read_text().splitlines()
    n_atoms = int(lines[3][0:3])
    out = []
    for line in lines[4 : 4 + n_atoms]:
        element = line[31:34].strip()
        if element == "H":
            continue
        xyz = np.array([float(line[0:10]), float(line[10:20]), float(line[20:30])])
        out.append((element, xyz))
    return out


def atoms_named(rec: Receptor, residue: str, atom: str) -> list[int]:
    """Indices of every ``atom`` in every ``residue`` of the receptor."""
    out = []
    for i, label in enumerate(rec.labels):
        _chain, res, name = label.split(":")
        if name == atom and res.rstrip("0123456789-") == residue:
            out.append(i)
    return out


# =============================================================================
# Residue template typing
# =============================================================================


class TestResidueTemplates:
    """The typing that the map comparison cannot check, because it differs.

    OpenBabel perceives hydrogen-bond roles from geometry with no knowledge of
    residue chemistry. Donor/acceptor typing is the entire channel where this
    grid outperforms a geometric buriedness score, so an error here silently
    discards its value while still producing plausible output.
    """

    def test_all_twenty_standard_residues_are_covered(self):
        assert len(RESIDUE_TEMPLATES) == 20

    def test_most_of_a_protein_is_typed_from_templates(self, templated):
        fraction = sum(1 for s in templated.source if s == "template") / len(templated)
        assert fraction > 0.9

    def test_proline_nitrogen_neither_donates_nor_accepts(self, templated):
        # It has no hydrogen, so it cannot donate, and its lone pair is
        # delocalised into the amide, so it does not accept. Perception sees a
        # donor wherever cropping the pocket left a proline at a chain break,
        # because protonating that free end produces a secondary ammonium that
        # does not exist in the intact protein.
        indices = atoms_named(templated, "PRO", "N")
        assert indices, "fixture has no proline"
        for i in indices:
            assert not templated.donor[i]
            assert not templated.acceptor[i]

    def test_guanidinium_nitrogens_lose_their_perceived_acceptor_role(
        self, templated, perceived
    ):
        # Arginine is protonated at pH 7.4 and its nitrogen lone pairs are
        # delocalised across the guanidinium, so NH1 and NH2 donate only.
        # Perception marks them acceptors as well, which would put a spurious
        # acceptor-probe well against the most common cationic residue in a
        # pocket.
        indices = atoms_named(templated, "ARG", "NH1") + atoms_named(
            templated, "ARG", "NH2"
        )
        assert indices, "fixture has no arginine"
        for i in indices:
            assert perceived.acceptor[i], "perception no longer needs correcting"
            assert templated.donor[i] and not templated.acceptor[i]

    def test_the_template_actually_corrects_perception(self, templated, perceived):
        # Guards against the templates silently agreeing with perception, which
        # would make this whole path dead weight.
        changed = np.flatnonzero(
            (templated.donor != perceived.donor)
            | (templated.acceptor != perceived.acceptor)
        )
        assert len(changed) > 0
        # Every correction removes a hydrogen-bond role rather than adding one.
        # Perception errs by seeing roles that residue chemistry rules out, and
        # a template that invented a donor or acceptor would be the more
        # dangerous failure: it plants a well where nothing can bind.
        for i in changed:
            assert not (templated.donor[i] and not perceived.donor[i])
            assert not (templated.acceptor[i] and not perceived.acceptor[i])

    def test_backbone_amide_donates_and_does_not_accept(self, templated):
        for i in atoms_named(templated, "LEU", "N"):
            assert templated.donor[i]
            assert not templated.acceptor[i]

    def test_backbone_carbonyl_accepts_and_does_not_donate(self, templated):
        indices = atoms_named(templated, "LEU", "O")
        assert indices
        for i in indices:
            assert templated.acceptor[i]
            assert not templated.donor[i]

    def test_hydroxyls_both_donate_and_accept(self, templated):
        for residue, atom in (("SER", "OG"), ("THR", "OG1"), ("TYR", "OH")):
            indices = atoms_named(templated, residue, atom)
            assert indices, f"fixture has no {residue}"
            for i in indices:
                assert templated.donor[i], f"{residue} {atom}"
                assert templated.acceptor[i], f"{residue} {atom}"

    def test_carboxylates_accept_only(self, templated):
        # Deprotonated at pH 7.4.
        for residue, atom in (("ASP", "OD1"), ("ASP", "OD2"), ("GLU", "OE1")):
            for i in atoms_named(templated, residue, atom):
                assert templated.acceptor[i]
                assert not templated.donor[i]

    def test_charged_amines_donate_only(self, templated):
        for residue, atom in (("LYS", "NZ"), ("ARG", "NH1"), ("TRP", "NE1")):
            indices = atoms_named(templated, residue, atom)
            assert indices, f"fixture has no {residue}"
            for i in indices:
                assert templated.donor[i]
                assert not templated.acceptor[i]

    def test_carbon_bonded_to_sulfur_is_not_hydrophobic(self, templated):
        # Vina's heteroatom is anything that is not carbon or hydrogen, so
        # sulfur counts. Methionine's CG and CE flank SD; CB does not.
        for atom in ("CG", "CE"):
            indices = atoms_named(templated, "MET", atom)
            assert indices, "fixture has no methionine"
            for i in indices:
                assert not templated.hydrophobic[i], f"MET {atom}"
        for i in atoms_named(templated, "MET", "CB"):
            assert templated.hydrophobic[i]

    def test_pure_carbon_side_chains_are_hydrophobic(self, templated):
        for residue, atom in (("LEU", "CD1"), ("VAL", "CG1"), ("PHE", "CZ")):
            indices = atoms_named(templated, residue, atom)
            assert indices, f"fixture has no {residue}"
            for i in indices:
                assert templated.hydrophobic[i], f"{residue} {atom}"

    def test_carbon_bonded_to_nitrogen_or_oxygen_is_polar(self, templated):
        for residue, atom in (("LEU", "CA"), ("LEU", "C"), ("ARG", "CZ")):
            for i in atoms_named(templated, residue, atom):
                assert not templated.hydrophobic[i], f"{residue} {atom}"

    def test_histidine_nitrogens_defer_to_the_prepared_structure(self, templated):
        # Which imidazole nitrogen carries the hydrogen is a tautomer and pH
        # question, not a covalent-structure one, so the template must not
        # pretend to know it.
        indices = atoms_named(templated, "HIS", "ND1") + atoms_named(
            templated, "HIS", "NE2"
        )
        assert indices, "fixture has no histidine"
        for i in indices:
            assert templated.source[i] == "pdbqt"

    def test_the_two_methods_agree_on_hydrophobicity(self, templated, perceived):
        # Connectivity from templates and bond detection by distance are
        # independent, so agreement is a real cross-check on both.
        assert np.array_equal(templated.hydrophobic, perceived.hydrophobic)


# =============================================================================
# Grid geometry
# =============================================================================


class TestGridGeometry:
    def test_the_centre_lands_on_a_grid_point(self, center):
        origin, shape = grid_axes(center, (20.0, 20.0, 20.0), 0.5)
        middle = origin + np.array(shape) // 2 * 0.5
        assert np.allclose(middle, center)

    def test_point_count_is_odd_like_autodock(self, center):
        _origin, shape = grid_axes(center, (20.0, 20.0, 20.0), 0.5)
        assert shape == (41, 41, 41)
        assert all(n % 2 == 1 for n in shape)

    def test_spacing_changes_the_count_not_the_extent(self, center):
        coarse_origin, coarse_shape = grid_axes(center, (20.0,) * 3, 0.5)
        fine_origin, fine_shape = grid_axes(center, (20.0,) * 3, 0.25)
        assert np.allclose(coarse_origin, fine_origin)
        assert fine_shape == (81, 81, 81)
        extent_coarse = (coarse_shape[0] - 1) * 0.5
        extent_fine = (fine_shape[0] - 1) * 0.25
        assert extent_coarse == pytest.approx(extent_fine)


class TestCropSufficiency:
    """V6: an atom out of range of every grid point contributes exactly zero."""

    def test_cropping_removes_atoms(self, templated, center):
        origin, shape = grid_axes(center, (10.0, 10.0, 10.0), 0.5)
        cropped = crop_to_box(templated, origin, shape, 0.5)
        assert len(cropped) < len(templated)

    def test_cropping_does_not_change_a_single_energy(self, templated, center):
        origin, shape = grid_axes(center, (10.0, 10.0, 10.0), 0.5)
        cropped = crop_to_box(templated, origin, shape, 0.5)
        probes = ("C_H", "O_A")

        full = compute_grids(templated, origin, shape, 0.5, probes)
        part = compute_grids(cropped, origin, shape, 0.5, probes)
        # Bit-for-bit would be too strong -- the summation order differs -- but
        # the discarded contributions are identically zero, so the residual is
        # floating-point noise and nothing else.
        assert np.max(np.abs(full - part)) < 1e-12


# =============================================================================
# Energy
# =============================================================================


def scalar_energy(
    point: np.ndarray, rec: Receptor, probe_name: str
) -> float:
    """A direct transcription of the published scoring function.

    Deliberately naive -- a Python loop over atoms with no vectorisation, no
    caching and no cutoff arithmetic -- so that agreement with the production
    grid means the optimisations preserved the formula.
    """
    probe = PROBES[probe_name]
    total = 0.0
    for i in range(len(rec.coords)):
        r = math.dist(point, rec.coords[i])
        if r >= CUTOFF:
            continue
        d = r - (probe.radius + rec.radii[i])
        total += W_GAUSS1 * math.exp(-((d / 0.5) ** 2))
        total += W_GAUSS2 * math.exp(-(((d - 3.0) / 2.0) ** 2))
        if d < 0:
            total += W_REPULSION * d * d
        if probe.hydrophobic and rec.hydrophobic[i]:
            total += W_HYDROPHOBIC * min(max(1.5 - d, 0.0), 1.0)
        if (probe.donor and rec.acceptor[i]) or (probe.acceptor and rec.donor[i]):
            total += W_HBOND * min(max(-d / 0.7, 0.0), 1.0)
    return total


class TestAgainstScalarReference:
    """V1: the vectorised grid is the published formula and nothing else."""

    @pytest.mark.parametrize("probe", ["C_H", "N_D", "O_A"])
    def test_every_point_of_a_small_grid_matches(self, templated, center, probe):
        origin, shape = grid_axes(center, (3.0, 3.0, 3.0), 1.0)
        cropped = crop_to_box(templated, origin, shape, 1.0)
        grid = compute_grids(cropped, origin, shape, 1.0, (probe,))[0]

        worst = 0.0
        for index in np.ndindex(shape):
            point = origin + np.array(index) * 1.0
            worst = max(worst, abs(grid[index] - scalar_energy(point, cropped, probe)))
        assert worst < 1e-9


class TestPhysicalBehaviour:
    """V3 and V4: the grid behaves like the interaction it claims to model."""

    @pytest.fixture(scope="class")
    def grids(self, templated, center):
        origin, shape = grid_axes(center, (16.0, 16.0, 16.0), 0.5)
        cropped = crop_to_box(templated, origin, shape, 0.5)
        probes = ("C_H", "N_D", "O_A")
        return (
            compute_grids(cropped, origin, shape, 0.5, probes),
            probes,
            origin,
            shape,
            cropped,
        )

    def test_the_protein_interior_is_strongly_repulsive(self, grids):
        # A point on a receptor atom is a hard clash: the repulsion term is
        # quadratic in the overlap and dominates everything else.
        _values, _probes, origin, shape, rec = grids
        buried = rec.coords[
            np.argmin(np.linalg.norm(rec.coords - (origin + np.array(shape) / 2 * 0.5), axis=1))
        ]
        assert scalar_energy(buried, rec, "C_H") > 5.0

    def test_the_open_pocket_is_attractive(self, grids):
        values, _probes, _origin, _shape, _rec = grids
        assert values.min() < -0.5

    def test_hydrogen_bond_distance_emerges_from_the_energy(self, grids):
        # Nothing in the code says 2.8 A. The acceptor probe's well sits where
        # the h-bond ramp and the repulsion balance, and that has to land on the
        # experimental heavy-atom donor-acceptor distance or the term is wrong.
        values, probes, origin, _shape, rec = grids
        acceptor_grid = values[probes.index("O_A")]
        minima = find_minima(
            acceptor_grid, "O_A", origin, 0.5, max_minima=12, min_separation=2.0
        )
        assert minima

        donors = rec.coords[rec.donor]
        assert len(donors)
        distances = []
        for m in minima:
            point = np.array([m.x, m.y, m.z])
            distances.append(float(np.min(np.linalg.norm(donors - point, axis=1))))

        median = float(np.median(distances))
        assert 2.6 < median < 3.5, f"median donor distance {median:.2f} A"


class TestMinima:
    def test_minima_are_separated(self, templated, center):
        origin, shape = grid_axes(center, (16.0,) * 3, 0.5)
        cropped = crop_to_box(templated, origin, shape, 0.5)
        grid = compute_grids(cropped, origin, shape, 0.5, ("O_A",))[0]
        minima = find_minima(grid, "O_A", origin, 0.5, max_minima=8, min_separation=3.0)

        points = [np.array([m.x, m.y, m.z]) for m in minima]
        for i, a in enumerate(points):
            for b in points[i + 1 :]:
                assert np.linalg.norm(a - b) >= 3.0

    def test_minima_are_ranked_and_negative(self, templated, center):
        origin, shape = grid_axes(center, (16.0,) * 3, 0.5)
        cropped = crop_to_box(templated, origin, shape, 0.5)
        grid = compute_grids(cropped, origin, shape, 0.5, ("C_H",))[0]
        minima = find_minima(grid, "C_H", origin, 0.5, max_minima=5)

        energies = [m.energy for m in minima]
        assert energies == sorted(energies)
        assert all(e < 0 for e in energies)

    def test_a_minimum_indexes_back_to_its_own_grid_value(self, templated, center):
        origin, shape = grid_axes(center, (16.0,) * 3, 0.5)
        cropped = crop_to_box(templated, origin, shape, 0.5)
        grid = compute_grids(cropped, origin, shape, 0.5, ("O_A",))[0]
        for m in find_minima(grid, "O_A", origin, 0.5, max_minima=5):
            index = np.round((np.array([m.x, m.y, m.z]) - origin) / 0.5).astype(int)
            assert grid[tuple(index)] == pytest.approx(m.energy)

    def test_probe_families_name_the_ligand_atom_not_the_receptor(self):
        # An acceptor probe finds its wells beside receptor donors, so its
        # minima mark where a ligand ACCEPTOR belongs. Getting this backwards
        # would invert every pharmacophore hypothesis built on the output.
        assert PROBES["O_A"].family == "Acceptor"
        assert PROBES["N_D"].family == "Donor"
        assert PROBES["C_H"].family == "Hydrophobe"


class TestLigandRecovery:
    """V5: the grid prefers where ligand atoms really are."""

    def test_hydrophobic_wells_sit_on_the_ligand_carbons(
        self, templated, center, ligand_atoms
    ):
        # Decoys are matched on distance to the nearest receptor atom, so a
        # score that only measured burial would come out at chance. Anything
        # above that is the chemistry.
        spacing = 0.5
        origin, shape = grid_axes(center, (20.0,) * 3, spacing)
        cropped = crop_to_box(templated, origin, shape, spacing)
        grid = compute_grids(cropped, origin, shape, spacing, ("C_H",))[0]

        def burial(points: np.ndarray) -> np.ndarray:
            deltas = points[:, None, :] - cropped.coords[None, :, :]
            return np.min(np.linalg.norm(deltas, axis=2), axis=1)

        def energy_at(points: np.ndarray) -> np.ndarray:
            index = np.round((points - origin) / spacing).astype(int)
            inside = np.all((index >= 0) & (index < np.array(shape)), axis=1)
            index = index[inside]
            return grid[index[:, 0], index[:, 1], index[:, 2]], inside

        carbons = np.array([xyz for element, xyz in ligand_atoms if element == "C"])
        assert len(carbons) >= 10
        true_energy, kept = energy_at(carbons)
        true_burial = burial(carbons)[kept]

        # Candidate decoys: every grid point, from which we draw a burial-matched
        # partner for each true position.
        rng = np.random.default_rng(0)
        grid_points = (
            origin
            + spacing
            * np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
        ).reshape(-1, 3)
        sample = grid_points[rng.choice(len(grid_points), 4000, replace=False)]
        sample_burial = burial(sample)
        sample_energy = grid[
            tuple(np.round((sample - origin) / spacing).astype(int).T)
        ]

        decoys = []
        for b in true_burial:
            close = np.flatnonzero(np.abs(sample_burial - b) < 0.35)
            if len(close):
                decoys.append(sample_energy[rng.choice(close)])
        assert len(decoys) >= 10

        wins = sum(
            1 for t in true_energy for d in decoys if t < d
        ) / (len(true_energy) * len(decoys))
        assert wins > 0.65, f"burial-matched AUC {wins:.3f}"


# =============================================================================
# V2: the golden regression against vina --write_maps
# =============================================================================


PROBE_LIGAND_TYPES = ["C", "A", "OA", "NA", "N"]


def write_probe_ligand(path: Path, center) -> None:
    """A dummy rigid ligand whose only job is to name the maps Vina writes.

    Vina emits one affinity map per atom type present in the ligand, so asking
    for maps means handing it a ligand that carries the types you want. The
    geometry is irrelevant to the result: a map depends on the probe type, not
    on where the dummy sat.
    """
    lines = ["ROOT"]
    # Strung along x and centred on the box: Vina refuses to score a ligand
    # whose atoms fall outside it.
    first = -1.6 * (len(PROBE_LIGAND_TYPES) - 1) / 2
    for i, ad_type in enumerate(PROBE_LIGAND_TYPES):
        x, y, z = center[0] + first + i * 1.6, center[1], center[2]
        lines.append(
            f"ATOM  {i + 1:>5} {ad_type[:2].ljust(3):<4} LIG A   1    "
            f"{x:>8.3f}{y:>8.3f}{z:>8.3f}  1.00  0.00     0.000 {ad_type:<2}"
        )
    lines += ["ENDROOT", "TORSDOF 0"]
    path.write_text("\n".join(lines) + "\n")


def read_autodock_map(path: Path):
    """Parse an AutoDock ``.map`` into (values, spacing, nelements, center)."""
    spacing = nelements = map_center = None
    values = []
    for line in path.read_text().splitlines():
        if line.startswith("SPACING"):
            spacing = float(line.split()[1])
        elif line.startswith("NELEMENTS"):
            nelements = [int(v) for v in line.split()[1:4]]
        elif line.startswith("CENTER"):
            map_center = [float(v) for v in line.split()[1:4]]
        elif line.startswith(("GRID_", "MACROMOLECULE")):
            continue
        elif line.strip():
            values.append(float(line))
    return np.array(values), spacing, nelements, map_center


@pytest.mark.skipif(not HAS_VINA, reason="vina binary not installed")
class TestAgainstVinaMaps:
    """The golden test: our energies against Vina's own, point for point.

    Typing is forced to the PDBQT path so this compares the *energy* and not the
    typing choice -- the residue templates deliberately differ from Vina, and
    ``TestResidueTemplates`` covers them instead.

    Assertions are restricted to the attractive region. Inside the clash core
    Vina's lookup-table interpolation dominates and the two disagree by up to
    0.6, which says nothing about this implementation: an analytic evaluation is
    the more accurate of the two there, and no minimum ever lands in a clash.
    """

    @pytest.fixture(scope="class")
    def maps(self, receptor_pdbqt, center):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            receptor = work / "receptor.pdbqt"
            receptor.write_text(receptor_pdbqt)
            ligand = work / "probe.pdbqt"
            write_probe_ligand(ligand, center)

            result = subprocess.run(
                [
                    "vina",
                    "--receptor", str(receptor),
                    "--ligand", str(ligand),
                    "--center_x", str(center[0]),
                    "--center_y", str(center[1]),
                    "--center_z", str(center[2]),
                    "--size_x", "12", "--size_y", "12", "--size_z", "12",
                    "--spacing", "0.5",
                    "--force_even_voxels",
                    "--write_maps", str(work / "ref"),
                    "--score_only",
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            assert result.returncode == 0, result.stderr or result.stdout

            written = sorted(work.glob("ref.*.map"))
            assert written, f"vina wrote no maps: {result.stdout}"
            # Read eagerly: the directory goes away when the fixture returns.
            yield {p.name.split(".")[1]: read_autodock_map(p) for p in written}

    def test_vina_wrote_maps_we_know_how_to_reproduce(self, maps):
        assert set(maps) & set(PROBES), f"no comparable maps in {sorted(maps)}"

    def test_energies_match_in_the_attractive_region(self, maps, perceived):
        worst_mean = worst_max = 0.0
        report = []

        for probe_name, (reference, spacing, nelements, map_center) in maps.items():
            if probe_name not in PROBES:
                continue
            size = tuple(n * spacing for n in nelements)
            origin, shape = grid_axes(map_center, size, spacing)
            cropped = crop_to_box(perceived, origin, shape, spacing)
            grid = compute_grids(cropped, origin, shape, spacing, (probe_name,))[0]

            # AutoDock maps run x fastest and z slowest.
            ours = grid.transpose(2, 1, 0).ravel()
            assert ours.shape == reference.shape, (probe_name, ours.shape, reference.shape)

            attractive = reference < 0
            assert attractive.sum() > 100, f"{probe_name} has no attractive region"
            deviation = np.abs(ours[attractive] - reference[attractive])
            worst_mean = max(worst_mean, float(deviation.mean()))
            worst_max = max(worst_max, float(deviation.max()))
            report.append(
                f"{probe_name}: n={attractive.sum()} "
                f"mean={deviation.mean():.2e} max={deviation.max():.2e}"
            )

        assert report, "no probes compared"
        # Measured at 4e-4 mean and 9e-3 max across five probe types; the bounds
        # leave room for a Vina point release without hiding a real regression.
        assert worst_mean < 1e-3, "; ".join(report)
        assert worst_max < 0.02, "; ".join(report)

    def test_the_shape_of_the_well_is_reproduced(self, maps, perceived):
        # A near-perfect correlation is a stronger claim than a small mean
        # difference: it says the wells are in the same places, not merely that
        # the numbers are all small.
        for probe_name, (reference, spacing, nelements, map_center) in maps.items():
            if probe_name not in PROBES:
                continue
            size = tuple(n * spacing for n in nelements)
            origin, shape = grid_axes(map_center, size, spacing)
            cropped = crop_to_box(perceived, origin, shape, spacing)
            grid = compute_grids(cropped, origin, shape, spacing, (probe_name,))[0]
            ours = grid.transpose(2, 1, 0).ravel()

            attractive = reference < 0
            correlation = np.corrcoef(ours[attractive], reference[attractive])[0, 1]
            assert correlation > 0.9999, f"{probe_name}: r={correlation:.6f}"


# =============================================================================
# Entry point and failure modes
# =============================================================================


class TestRunProbeGrid:
    def test_the_default_probes_are_the_three_pharmacophore_families(
        self, receptor_pdbqt, center
    ):
        result = run_probe_grid(receptor_pdbqt, center, (10.0, 10.0, 10.0))
        assert result.probes == ("C_H", "N_D", "O_A")
        assert result.grids.shape == (3, 21, 21, 21)
        assert {m.family for m in result.minima} <= {
            "Hydrophobe",
            "Donor",
            "Acceptor",
        }

    def test_minima_are_pooled_and_ranked_across_probes(self, receptor_pdbqt, center):
        result = run_probe_grid(
            receptor_pdbqt, center, (10.0,) * 3, max_minima=3
        )
        energies = [m.energy for m in result.minima]
        assert energies == sorted(energies)

    def test_extra_probes_share_the_distance_calculation(self, receptor_pdbqt, center):
        # Not a timing assertion -- just that asking for more probes returns
        # more grids of the same geometry, which is what makes it cheap.
        result = run_probe_grid(
            receptor_pdbqt, center, (8.0,) * 3, probe_names=tuple(PROBES)
        )
        assert result.grids.shape[0] == len(PROBES)

    def test_an_unknown_probe_is_rejected_by_name(self, receptor_pdbqt, center):
        with pytest.raises(ValueError, match="Unknown probe"):
            run_probe_grid(receptor_pdbqt, center, (8.0,) * 3, probe_names=("Xx",))

    def test_a_box_off_the_protein_is_rejected(self, receptor_pdbqt):
        with pytest.raises(ValueError, match="No receptor atoms"):
            run_probe_grid(receptor_pdbqt, (900.0, 900.0, 900.0), (8.0,) * 3)

    def test_an_empty_receptor_is_rejected(self, center):
        with pytest.raises(ValueError, match="No typed heavy atoms"):
            run_probe_grid("REMARK nothing here\n", center, (8.0,) * 3)
