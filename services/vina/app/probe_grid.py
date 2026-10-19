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

"""Probe interaction-energy grids, implementing AutoDock Vina's scoring function.

The question this answers is "where in this pocket would a hydrogen-bond donor,
an acceptor, or a hydrophobic group most want to sit", using only the receptor.
It is the physics half of pharmacophore-site prediction: a ligand atom of a given
type placed at a grid point, scored against every receptor atom in range.

Why Vina's functional form rather than a plain Lennard-Jones: Vina 1.2.5 can
write its own affinity maps, so this is checkable against a reference
implementation numerically rather than merely "correlated with". Matching Vina
also keeps the grid consistent with the docking tool this service already ships.
See ``tests/test_probe_grid.py`` for that comparison.

Vina's inter-molecular score, over heavy atoms only, in terms of the surface
distance ``d = r - R_i - R_j`` (Trott & Olson 2010):

    gauss1      exp(-(d/0.5)^2)                              w = -0.035579
    gauss2      exp(-((d-3)/2)^2)                            w = -0.005156
    repulsion   d^2 if d < 0 else 0                          w = +0.840245
    hydrophobic 1 for d<0.5, 0 for d>1.5, linear between     w = -0.035069
                (only between two hydrophobic atoms)
    hbond       1 for d<-0.7, 0 for d>0, linear between      w = -0.587439
                (only between a donor and an acceptor)

with an 8 A cutoff on r.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

W_GAUSS1 = -0.035579
W_GAUSS2 = -0.005156
W_REPULSION = 0.840245
W_HYDROPHOBIC = -0.035069
W_HBOND = -0.587439
CUTOFF = 8.0

# Longest bond the PDBQT fallback needs to detect is C-S at 1.81 A; the shortest
# 1,3-contact it must not mistake for a bond starts near 2.2 A.
BOND_CUTOFF = 2.0

# Vina X-S radii, by element.
XS_RADIUS = {
    "C": 1.9, "N": 1.8, "O": 1.7, "S": 2.0, "P": 2.1,
    "F": 1.5, "Cl": 1.8, "Br": 2.0, "I": 2.2,
    "Fe": 1.2, "Zn": 1.2, "Mg": 1.2, "Mn": 1.2, "Ca": 1.2, "Cu": 1.2, "Na": 1.2,
    "K": 1.2, "Hg": 1.2, "Ni": 1.2, "Co": 1.2, "Cd": 1.2,
}
DEFAULT_RADIUS = 1.9


@dataclass(frozen=True)
class Probe:
    """A ligand atom type placed at each grid point.

    Named for the XS type Vina uses in its map filenames, so a grid can be
    compared against ``vina --write_maps`` output without a translation step.

    ``family`` is the pharmacophore family of the *ligand* atom, which is the
    inverse of what it pairs with: an acceptor probe finds its wells next to
    receptor donors, and a minimum there is where a ligand acceptor belongs.
    """

    radius: float
    hydrophobic: bool
    donor: bool
    acceptor: bool
    family: str


PROBES: dict[str, Probe] = {
    "C_H": Probe(1.9, True, False, False, "Hydrophobe"),
    "C_P": Probe(1.9, False, False, False, "Steric"),
    "N_P": Probe(1.8, False, False, False, "Steric"),
    "N_A": Probe(1.8, False, False, True, "Acceptor"),
    "N_D": Probe(1.8, False, True, False, "Donor"),
    "O_A": Probe(1.7, False, False, True, "Acceptor"),
    "O_DA": Probe(1.7, False, True, True, "Donor/Acceptor"),
}

DEFAULT_PROBES = ("C_H", "N_D", "O_A")

# AutoDock PDBQT type -> (element, is_acceptor). HD is a polar hydrogen: it does
# not interact in the XS model, it only marks its heavy neighbour as a donor.
AD_TYPE = {
    "C": ("C", False), "A": ("C", False),
    "N": ("N", False), "NA": ("N", True), "NS": ("N", True),
    "OA": ("O", True), "OS": ("O", True), "O": ("O", True),
    "SA": ("S", True), "S": ("S", False),
    "P": ("P", False), "F": ("F", False), "Cl": ("Cl", False), "CL": ("Cl", False),
    "Br": ("Br", False), "BR": ("Br", False), "I": ("I", False),
    "Fe": ("Fe", False), "FE": ("Fe", False), "Zn": ("Zn", False),
    "ZN": ("Zn", False), "Mg": ("Mg", False), "MG": ("Mg", False),
    "Mn": ("Mn", False), "MN": ("Mn", False), "Ca": ("Ca", False),
    "CA": ("Ca", False), "Cu": ("Cu", False), "CU": ("Cu", False),
}


# =============================================================================
# Residue templates
# =============================================================================
#
# Why not just read the PDBQT types, which is what Vina itself does: because
# perception from geometry gets protein chemistry wrong in ways that matter
# here. OpenBabel types a proline backbone nitrogen as a hydrogen-bond acceptor.
# It has no hydrogen (so it cannot donate) and its lone pair is delocalised into
# the amide (so it does not accept), and marking it an acceptor puts a spurious
# donor-probe well against every proline in the pocket. Donor/acceptor typing is
# the whole channel where this grid beats a geometric buriedness score, so an
# error there discards the value while still producing plausible-looking output.
#
# Only side-chain bonds are listed. Inter-residue bonds are omitted deliberately:
# hydrophobicity is "carbon with no heteroatom neighbour", and no carbon's only
# heteroatom neighbour is across a peptide or disulfide bond -- CA always has its
# own backbone N, and CB of cysteine always has its own SG.

_BACKBONE_BONDS = (("N", "CA"), ("CA", "C"), ("C", "O"), ("C", "OXT"))

# Backbone amide N donates (it carries an H) and does not accept. Backbone
# carbonyl O and the C-terminal OXT accept and do not donate. Proline overrides
# the nitrogen below.
_BACKBONE_DONORS = frozenset({"N"})
_BACKBONE_ACCEPTORS = frozenset({"O", "OXT"})


@dataclass(frozen=True)
class ResidueTemplate:
    """Heavy-atom connectivity and hydrogen-bond roles for one residue type."""

    bonds: tuple[tuple[str, str], ...]
    donors: frozenset[str] = frozenset()
    acceptors: frozenset[str] = frozenset()
    # Atoms whose role depends on protonation state rather than on covalent
    # structure, so a template cannot know it. Typed from the PDBQT instead,
    # which reflects the pH the receptor was prepared at.
    state_dependent: frozenset[str] = frozenset()
    # Overrides the backbone defaults, for proline's tertiary amide nitrogen.
    not_donors: frozenset[str] = frozenset()


def _template(
    side_chain: tuple[tuple[str, str], ...],
    donors: tuple[str, ...] = (),
    acceptors: tuple[str, ...] = (),
    state_dependent: tuple[str, ...] = (),
    not_donors: tuple[str, ...] = (),
) -> ResidueTemplate:
    return ResidueTemplate(
        bonds=_BACKBONE_BONDS + side_chain,
        donors=frozenset(donors) | (_BACKBONE_DONORS - frozenset(not_donors)),
        acceptors=frozenset(acceptors) | _BACKBONE_ACCEPTORS,
        state_dependent=frozenset(state_dependent),
        not_donors=frozenset(not_donors),
    )


RESIDUE_TEMPLATES: dict[str, ResidueTemplate] = {
    "GLY": _template(()),
    "ALA": _template((("CA", "CB"),)),
    "VAL": _template(
        (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2")),
    ),
    "LEU": _template(
        (("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2")),
    ),
    "ILE": _template(
        (("CA", "CB"), ("CB", "CG1"), ("CB", "CG2"), ("CG1", "CD1")),
    ),
    # The nitrogen closes the pyrrolidine ring, so it carries no hydrogen: not a
    # donor, and not an acceptor either.
    "PRO": _template(
        (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "N")),
        not_donors=("N",),
    ),
    "PHE": _template(
        (
            ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
            ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
        ),
    ),
    "TYR": _template(
        (
            ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
            ("CD1", "CE1"), ("CD2", "CE2"), ("CE1", "CZ"), ("CE2", "CZ"),
            ("CZ", "OH"),
        ),
        donors=("OH",),
        acceptors=("OH",),
    ),
    "TRP": _template(
        (
            ("CA", "CB"), ("CB", "CG"), ("CG", "CD1"), ("CG", "CD2"),
            ("CD1", "NE1"), ("NE1", "CE2"), ("CD2", "CE2"), ("CD2", "CE3"),
            ("CE2", "CZ2"), ("CE3", "CZ3"), ("CZ2", "CH2"), ("CZ3", "CH2"),
        ),
        donors=("NE1",),
    ),
    "SER": _template(
        (("CA", "CB"), ("CB", "OG")),
        donors=("OG",),
        acceptors=("OG",),
    ),
    "THR": _template(
        (("CA", "CB"), ("CB", "OG1"), ("CB", "CG2")),
        donors=("OG1",),
        acceptors=("OG1",),
    ),
    # Sulfur is XS type S_P in Vina: it never takes part in the hydrogen-bond
    # term, whichever way OpenBabel types it.
    "CYS": _template((("CA", "CB"), ("CB", "SG"))),
    "MET": _template((("CA", "CB"), ("CB", "CG"), ("CG", "SD"), ("SD", "CE"))),
    "ASN": _template(
        (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "ND2")),
        donors=("ND2",),
        acceptors=("OD1",),
    ),
    "GLN": _template(
        (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "NE2")),
        donors=("NE2",),
        acceptors=("OE1",),
    ),
    # Deprotonated at pH 7.4: both carboxylate oxygens accept, neither donates.
    "ASP": _template(
        (("CA", "CB"), ("CB", "CG"), ("CG", "OD1"), ("CG", "OD2")),
        acceptors=("OD1", "OD2"),
    ),
    "GLU": _template(
        (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "OE1"), ("CD", "OE2")),
        acceptors=("OE1", "OE2"),
    ),
    "LYS": _template(
        (("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "CE"), ("CE", "NZ")),
        donors=("NZ",),
    ),
    "ARG": _template(
        (
            ("CA", "CB"), ("CB", "CG"), ("CG", "CD"), ("CD", "NE"),
            ("NE", "CZ"), ("CZ", "NH1"), ("CZ", "NH2"),
        ),
        donors=("NE", "NH1", "NH2"),
    ),
    # Which imidazole nitrogen carries the hydrogen is a tautomer and pH
    # question, not a covalent-structure one, so both are deferred to the PDBQT.
    "HIS": _template(
        (
            ("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("CG", "CD2"),
            ("ND1", "CE1"), ("CE1", "NE2"), ("CD2", "NE2"),
        ),
        state_dependent=("ND1", "NE2"),
    ),
}

# Common alternative residue names that are chemically the standard residue.
RESIDUE_ALIASES = {
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
    "HSP": "HIS", "CYX": "CYS", "CYM": "CYS", "ASH": "ASP", "GLH": "GLU",
    "LYN": "LYS", "ARN": "ARG", "TYM": "TYR", "MSE": "MET",
}


# =============================================================================
# Receptor typing
# =============================================================================


@dataclass
class RawAtom:
    """One heavy atom as it appears in the prepared PDBQT."""

    name: str
    res_name: str
    chain: str
    res_seq: str
    element: str
    ad_acceptor: bool
    xyz: tuple[float, float, float]

    @property
    def residue_key(self) -> tuple[str, str, str]:
        return (self.chain, self.res_seq, self.res_name)

    @property
    def label(self) -> str:
        return f"{self.chain or '_'}:{self.res_name}{self.res_seq}:{self.name}"


@dataclass
class Receptor:
    """Receptor heavy atoms typed for the XS scoring function."""

    coords: np.ndarray
    radii: np.ndarray
    donor: np.ndarray
    acceptor: np.ndarray
    hydrophobic: np.ndarray
    labels: list[str]
    # "template" or "pdbqt", per atom. Reported so a caller can see how much of
    # the receptor fell back to geometric perception.
    source: list[str]

    def __len__(self) -> int:
        return len(self.coords)


def parse_pdbqt_atoms(text: str) -> tuple[list[RawAtom], np.ndarray]:
    """Split a PDBQT into heavy atoms and polar-hydrogen positions.

    Polar hydrogens (AutoDock type ``HD``) do not interact in the XS model; they
    are kept only because they mark their heavy neighbour as a donor.
    """
    heavy: list[RawAtom] = []
    polar_h: list[list[float]] = []

    for line in text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except (ValueError, IndexError):
            continue
        ad = line[77:79].strip()
        if ad in ("HD", "H"):
            if ad == "HD":
                polar_h.append(list(xyz))
            continue
        if ad not in AD_TYPE:
            continue
        element, ad_acceptor = AD_TYPE[ad]
        heavy.append(
            RawAtom(
                name=line[12:16].strip(),
                res_name=line[17:20].strip().upper(),
                chain=line[21:22].strip(),
                res_seq=line[22:27].strip(),
                element=element,
                ad_acceptor=ad_acceptor,
                xyz=xyz,
            )
        )

    return heavy, np.array(polar_h) if polar_h else np.zeros((0, 3))


def _perceived_typing(
    atoms: list[RawAtom], polar_h: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donor, acceptor and hydrophobic flags perceived from the PDBQT.

    This is what Vina itself does with a prepared receptor, so it is both the
    fallback for anything without a residue template and the typing to use when
    comparing against ``vina --write_maps``.
    """
    coords = np.array([a.xyz for a in atoms]) if atoms else np.zeros((0, 3))
    n = len(atoms)
    acceptor = np.array([a.ad_acceptor for a in atoms], dtype=bool)

    # Donor: an N or O carrying a polar hydrogen.
    donor = np.zeros(n, dtype=bool)
    if len(polar_h):
        for i, atom in enumerate(atoms):
            if atom.element in ("N", "O"):
                if np.min(np.linalg.norm(polar_h - coords[i], axis=1)) < 1.3:
                    donor[i] = True

    # Hydrophobic: a carbon bonded to no heteroatom, plus the halogens. Vina's
    # heteroatom is anything that is not carbon or hydrogen, so sulfur counts --
    # missing that is a real bug the map comparison caught, because it turned
    # every cysteine and methionine carbon into a hydrophobe.
    hydrophobic = np.zeros(n, dtype=bool)
    is_hetero = np.array([a.element != "C" for a in atoms], dtype=bool)
    hetero_pos = coords[is_hetero] if is_hetero.any() else np.zeros((0, 3))
    for i, atom in enumerate(atoms):
        if atom.element in ("F", "Cl", "Br", "I"):
            hydrophobic[i] = True
        elif atom.element == "C":
            if not len(hetero_pos):
                hydrophobic[i] = True
            else:
                d = np.linalg.norm(hetero_pos - coords[i], axis=1)
                hydrophobic[i] = bool(np.min(d) > BOND_CUTOFF)

    return donor, acceptor, hydrophobic


def build_receptor(pdbqt_text: str, use_templates: bool = True) -> Receptor:
    """Type a prepared receptor for the XS scoring function.

    With ``use_templates`` (the default), standard amino acids are typed from
    their known chemistry and everything else -- cofactors, metals, modified
    residues, ligands -- falls back to perception from the PDBQT. Set it False
    to reproduce Vina's own typing exactly, which is what the map comparison in
    the tests needs in order to isolate the energy calculation from the typing
    choice.
    """
    atoms, polar_h = parse_pdbqt_atoms(pdbqt_text)
    if not atoms:
        raise ValueError("No typed heavy atoms found in the prepared receptor")

    coords = np.array([a.xyz for a in atoms])
    donor, acceptor, hydrophobic = _perceived_typing(atoms, polar_h)
    source = ["pdbqt"] * len(atoms)

    if use_templates:
        # Which atoms of each residue are actually present, so a disordered or
        # truncated side chain is typed on what the structure contains.
        present: dict[tuple[str, str, str], set[str]] = {}
        for atom in atoms:
            present.setdefault(atom.residue_key, set()).add(atom.name)

        for i, atom in enumerate(atoms):
            res_name = RESIDUE_ALIASES.get(atom.res_name, atom.res_name)
            template = RESIDUE_TEMPLATES.get(res_name)
            if template is None or atom.name in template.state_dependent:
                continue
            adjacency = TEMPLATE_ADJACENCY[res_name].get(atom.name)
            if adjacency is None:
                continue

            donor[i] = atom.name in template.donors
            acceptor[i] = atom.name in template.acceptors
            if atom.element == "C":
                bonded = adjacency & present[atom.residue_key]
                hydrophobic[i] = all(_element_of(nb) == "C" for nb in bonded)
            else:
                hydrophobic[i] = False
            source[i] = "template"

    return Receptor(
        coords=coords,
        radii=np.array([XS_RADIUS.get(a.element, DEFAULT_RADIUS) for a in atoms]),
        donor=donor,
        acceptor=acceptor,
        hydrophobic=hydrophobic,
        labels=[a.label for a in atoms],
        source=source,
    )


def _build_adjacency() -> dict[str, dict[str, frozenset[str]]]:
    """Atom name -> bonded atom names, per residue type. Built once at import."""
    index: dict[str, dict[str, frozenset[str]]] = {}
    for res_name, template in RESIDUE_TEMPLATES.items():
        neighbours: dict[str, set[str]] = {}
        for a, b in template.bonds:
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)
        index[res_name] = {k: frozenset(v) for k, v in neighbours.items()}
    return index


TEMPLATE_ADJACENCY = _build_adjacency()


def _element_of(atom_name: str) -> str:
    """Element of a standard amino-acid atom, from its PDB name.

    Safe for the 20 standard residues only, where every heavy atom is C, N, O or
    S and the name begins with the element symbol. Anything else is typed from
    the PDBQT and never reaches this function.
    """
    return atom_name[0] if atom_name else "C"


# =============================================================================
# Grid geometry
# =============================================================================


def grid_axes(
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    spacing: float,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Origin and shape of a box grid, using AutoDock's conventions.

    AutoDock counts an even number of intervals per axis and therefore an odd
    number of points, with the requested centre landing exactly on a grid point.
    Reproducing that is what lets the tests compare point-for-point against
    ``vina --write_maps`` without interpolating.
    """
    origin = np.zeros(3)
    shape = []
    for axis in range(3):
        intervals = max(2, 2 * int(round(size[axis] / (2.0 * spacing))))
        half = intervals / 2.0 * spacing
        origin[axis] = center[axis] - half
        shape.append(intervals + 1)
    return origin, (shape[0], shape[1], shape[2])


def crop_to_box(
    rec: Receptor,
    origin: np.ndarray,
    shape: tuple[int, int, int],
    spacing: float,
) -> Receptor:
    """Drop receptor atoms that cannot reach any grid point.

    Exact rather than approximate: an atom further than the 8 A cutoff from the
    box on any axis contributes zero to every point in it. Validation on a 16 A
    crop of a full receptor found the energies at the ligand atoms unchanged to
    the last bit.
    """
    upper = origin + (np.array(shape) - 1) * spacing
    inside = np.all(
        (rec.coords >= origin - CUTOFF) & (rec.coords <= upper + CUTOFF), axis=1
    )
    if inside.all():
        return rec
    keep = np.flatnonzero(inside)
    return Receptor(
        coords=rec.coords[keep],
        radii=rec.radii[keep],
        donor=rec.donor[keep],
        acceptor=rec.acceptor[keep],
        hydrophobic=rec.hydrophobic[keep],
        labels=[rec.labels[i] for i in keep],
        source=[rec.source[i] for i in keep],
    )


# =============================================================================
# Energy
# =============================================================================


def compute_grids(
    rec: Receptor,
    origin: np.ndarray,
    shape: tuple[int, int, int],
    spacing: float,
    probe_names: tuple[str, ...],
) -> np.ndarray:
    """Vina inter-molecular energy at every grid point, for each probe.

    Returns an array of shape ``(len(probe_names), nx, ny, nz)``.

    The loop runs over receptor atoms rather than grid points. On a regular
    lattice the set of points within the cutoff of an atom is a contiguous
    sub-box, found by arithmetic on the indices, which makes the neighbour
    search exact and free -- no tree to build, no ragged index lists, and no
    risk of dropping a coincident point the way a sparse distance matrix does.
    Each atom's distances are computed once and shared across every probe, and
    the steric part is shared across probes of equal radius, which is why asking
    for three probes costs barely more than asking for one.
    """
    probes = [PROBES[name] for name in probe_names]
    grids = np.zeros((len(probes), *shape))
    cutoff_sq = CUTOFF * CUTOFF

    # Probes of equal radius see the same surface distance, so gauss1, gauss2
    # and repulsion are computed once per distinct radius.
    radius_groups: dict[float, list[int]] = {}
    for index, probe in enumerate(probes):
        radius_groups.setdefault(probe.radius, []).append(index)

    for atom in range(len(rec)):
        centre = rec.coords[atom]
        lo = np.maximum(
            np.ceil((centre - CUTOFF - origin) / spacing).astype(int), 0
        )
        hi = np.minimum(
            np.floor((centre + CUTOFF - origin) / spacing).astype(int),
            np.array(shape) - 1,
        )
        if np.any(lo > hi):
            continue

        sub_shape = tuple(hi - lo + 1)
        offsets = [
            origin[k] + spacing * np.arange(lo[k], hi[k] + 1) - centre[k]
            for k in range(3)
        ]
        r_sq = (
            offsets[0][:, None, None] ** 2
            + offsets[1][None, :, None] ** 2
            + offsets[2][None, None, :] ** 2
        )
        within = r_sq < cutoff_sq
        if not within.any():
            continue
        r = np.sqrt(r_sq[within])

        atom_radius = rec.radii[atom]
        block = (
            slice(lo[0], hi[0] + 1),
            slice(lo[1], hi[1] + 1),
            slice(lo[2], hi[2] + 1),
        )

        for radius, indices in radius_groups.items():
            d = r - (radius + atom_radius)
            steric = (
                W_GAUSS1 * np.exp(-((d / 0.5) ** 2))
                + W_GAUSS2 * np.exp(-(((d - 3.0) / 2.0) ** 2))
                + W_REPULSION * np.where(d < 0, d * d, 0.0)
            )
            for index in indices:
                probe = probes[index]
                energy = steric
                if probe.hydrophobic and rec.hydrophobic[atom]:
                    energy = energy + W_HYDROPHOBIC * np.clip(1.5 - d, 0.0, 1.0)
                # A donor probe pairs with receptor acceptors, and vice versa.
                pairs = (probe.donor and rec.acceptor[atom]) or (
                    probe.acceptor and rec.donor[atom]
                )
                if pairs:
                    energy = energy + W_HBOND * np.clip(-d / 0.7, 0.0, 1.0)

                scattered = np.zeros(sub_shape)
                scattered[within] = energy
                grids[index][block] += scattered

    return grids


# =============================================================================
# Minima
# =============================================================================


@dataclass(frozen=True)
class Minimum:
    """One ranked interaction hotspot."""

    probe: str
    family: str
    x: float
    y: float
    z: float
    energy: float

    def as_dict(self) -> dict:
        return {
            "probe": self.probe,
            "family": self.family,
            "x": round(float(self.x), 3),
            "y": round(float(self.y), 3),
            "z": round(float(self.z), 3),
            "energy": round(float(self.energy), 4),
        }


def find_minima(
    grid: np.ndarray,
    probe_name: str,
    origin: np.ndarray,
    spacing: float,
    max_minima: int = 5,
    min_separation: float = 2.0,
) -> list[Minimum]:
    """Best-scoring, well-separated points of one probe grid.

    Greedy non-maximum suppression rather than a plain argsort: the lowest
    hundred points of a grid at 0.5 A spacing are all the same well, so without
    a separation constraint the ranked list describes one site several times.
    """
    flat = grid.ravel()
    negative = np.flatnonzero(flat < 0.0)
    if not len(negative):
        return []

    # Only the deepest candidates can survive suppression, and sorting the whole
    # grid to find them is wasted work on a 166k-point box.
    budget = min(len(negative), max(2000, max_minima * 200))
    deepest = negative[np.argpartition(flat[negative], budget - 1)[:budget]]
    order = deepest[np.argsort(flat[deepest])]

    probe = PROBES[probe_name]
    accepted: list[Minimum] = []
    accepted_xyz: list[np.ndarray] = []
    for index in order:
        point = origin + spacing * np.array(np.unravel_index(index, grid.shape))
        if any(
            np.linalg.norm(point - other) < min_separation for other in accepted_xyz
        ):
            continue
        accepted.append(
            Minimum(
                probe=probe_name,
                family=probe.family,
                x=point[0],
                y=point[1],
                z=point[2],
                energy=float(flat[index]),
            )
        )
        accepted_xyz.append(point)
        if len(accepted) >= max_minima:
            break

    return accepted


# =============================================================================
# Entry point
# =============================================================================


@dataclass
class ProbeGridResult:
    """Everything the endpoint needs to answer with."""

    grids: np.ndarray
    probes: tuple[str, ...]
    origin: np.ndarray
    spacing: float
    shape: tuple[int, int, int]
    minima: list[Minimum]
    num_receptor_atoms: int
    num_template_typed: int


def run_probe_grid(
    pdbqt_text: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    spacing: float = 0.5,
    probe_names: tuple[str, ...] = DEFAULT_PROBES,
    use_templates: bool = True,
    max_minima: int = 5,
    min_separation: float = 2.0,
) -> ProbeGridResult:
    """Compute probe grids for a prepared receptor and rank their minima."""
    unknown = [name for name in probe_names if name not in PROBES]
    if unknown:
        raise ValueError(
            f"Unknown probe type(s): {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(PROBES))}"
        )

    receptor = build_receptor(pdbqt_text, use_templates=use_templates)
    origin, shape = grid_axes(center, size, spacing)
    cropped = crop_to_box(receptor, origin, shape, spacing)
    if not len(cropped):
        raise ValueError(
            "No receptor atoms within reach of the requested box. Check that "
            "the centre is on the receptor."
        )

    logger.info(
        "Probe grid: %d points x %d probes against %d atoms "
        "(%d cropped away, %d typed from templates)",
        int(np.prod(shape)),
        len(probe_names),
        len(cropped),
        len(receptor) - len(cropped),
        sum(1 for s in cropped.source if s == "template"),
    )

    grids = compute_grids(cropped, origin, shape, spacing, probe_names)

    minima: list[Minimum] = []
    for index, name in enumerate(probe_names):
        minima.extend(
            find_minima(
                grids[index],
                name,
                origin,
                spacing,
                max_minima=max_minima,
                min_separation=min_separation,
            )
        )
    minima.sort(key=lambda m: m.energy)

    return ProbeGridResult(
        grids=grids,
        probes=tuple(probe_names),
        origin=origin,
        spacing=spacing,
        shape=shape,
        minima=minima,
        num_receptor_atoms=len(cropped),
        num_template_typed=sum(1 for s in cropped.source if s == "template"),
    )
