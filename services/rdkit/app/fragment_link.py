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

"""Exit-vector geometry: where a fragment can be substituted, and what fits there.

SMDD-Bench hands a Fragment Assembly task one or two fragments **already posed**
in the pocket and valence-complete -- no dummy atoms, no marked attachment
points. Appendix H.5's ``fragment_0.sdf`` parses to plain ``c1ccccc1``. So before
anything can be linked, something has to answer a question no existing Aevom tool
asks: *which* hydrogen do you replace, and can a bridge of plausible length reach
from there to its partner without going through the protein?

That is what this module computes, per fragment and per candidate pair:

- **Substitutable sites.** Every heavy atom carrying a hydrogen, with the
  outward unit vector along which a substituent would grow. Taken from an
  explicit hydrogen's position when the input has one and from the
  heavy-neighbour centroid otherwise, so a fragment written either way behaves
  the same.
- **Free depth.** How far that vector travels before a protein heavy atom is
  within clashing distance. A site pointing into the protein wall has a free
  depth near zero and is not a growth vector however exposed it looks in 2D.
- **Pair descriptors.** Anchor separation, the angle between the two exit
  vectors, how squarely each one points at its partner, and the clearance of the
  straight tube between the anchors. DeLinker showed that exit-vector distance
  and angle are what carry the signal for long linkers
  (`Imrie 2020 <https://pmc.ncbi.nlm.nih.gov/articles/PMC7189367/>`_), which is
  the only regime SMDD's two-fragment tasks contain -- pairs must sit at least
  6 A apart.
- **Linker size and rotatable-bond budget.** The atom-count window that can
  span the gap, and what a saturated chain of that length would cost against
  ``RB <= 10``. When the chain cannot be afforded, that is not a footnote: it is
  the finding, and it means a rigid aryl, alkyne or fused bridge is the only
  answer that passes.
- **Ready-to-use templates.** GenMol linker-design strings in both accepted
  encodings, plus assembled candidate SMILES from the measured
  :mod:`linker_library`, which is the path that needs no model at all.

Nothing here is a verdict on a molecule. Deciding whether a candidate actually
holds both fragments in place is :mod:`constrained_pose`'s job.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import safe as sf
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from . import linker_library as lib

logger = logging.getLogger(__name__)

# A probe point this close to a protein heavy atom counts as obstructed. 2.4 A
# is the same floor the tube clearance already normalises against, and it is
# below what real substituents do: measured against SMDD-Bench witness ligands,
# grown atoms pack as close as 2.85 A to the protein with a 3.65 A median. The
# 3.0 A this used to be sat above that, so a probe walking into an occupiable
# pocket reported a wall -- on smdd_005_P00519_0 it called the one vector the
# witness grows from blocked, and the 21 atoms it grows there fit.
PROBE_CLEARANCE = 2.4

# Depth probe: start one bond out from the anchor, step finely enough that a
# 2.4 A sphere cannot be stepped over, and stop past the longest reach a
# substituent has been observed to take (11.3 A on the task above).
BOND_LENGTH = 1.5
DEPTH_STEP = 0.25
MAX_DEPTH = 12.0

# The depth probe is a cone of rays, not one ray. A substituent bends, so a
# single ray along the exit vector reports the first wall its bond angle happens
# to point at rather than the room actually available: on the task above the
# axial ray reads 3.25 A where the witness reaches 11.32, and a 45 degree cone
# reads 11.50. The half-angle is the whole calibration -- at 60 degrees the cone
# reaches around walls and reports 14 A for the two vectors that genuinely are
# blocked, losing the distinction it exists to make.
CONE_HALF_ANGLE = 45.0
CONE_RINGS = (22.5, 45.0)
CONE_SPOKES = 8

# Burial: protein heavy atoms within BURIAL_RADIUS of a point BURIAL_PROBE along
# the exit vector. Depth alone cannot tell a deep pocket from open water -- a
# vector pointing out of the protein never hits anything and so reports maximum
# depth, which is how a solvent-facing methyl came to carry the largest growth
# mask on offer. On the task above this reads 22 for that methyl against 64 to
# 88 for the five vectors that point into the pocket, so BURIAL_FULL is set
# where a vector is credited as enclosed.
BURIAL_PROBE = 4.0
BURIAL_RADIUS = 8.0
BURIAL_FULL = 60.0

# Interior of the anchor-to-anchor segment sampled for tube clearance. The ends
# are skipped because the anchors are fragment atoms already in van der Waals
# contact with the protein -- including them would report every buried pair as
# blocked.
TUBE_MARGIN = 1.5
TUBE_STEP = 0.25

# SMDD-Bench Table 6. Kept here rather than imported from gates.py because this
# module needs only this one bound, and the two new bonds a linker adds are
# charged against it before any linker exists.
DEFAULT_RB_MAX = 10
NEW_BOND_RB_COST = 2

# Atoms a ring-bearing bridge carries per atom it puts on the anchor-to-anchor
# path, measured across the linker library. See :func:`mask_atom_range`.
RING_ATOM_FACTOR = 1.7

# Elements used as the placeholder in a SAFE template, in preference order. The
# placeholder occupies the position the generation mask will take; it has to be
# an element neither fragment contains so the cut can be aimed at it.
PLACEHOLDER_ELEMENTS = ("Si", "Ge", "Se", "Te")

# Depth at which a growth vector is considered fully open, for scoring only.
DEPTH_SCALE = 9.0

# Pair ranking weights. Their only job is ordering candidate pairs so the true
# attachment pair of a known complex lands in the top few; no downstream decision
# reads the score itself. Calibrated on 46 PDBBind complexes built the way
# SMDD-Bench builds its tasks (see tests/test_fragment_link.py and the plan's V2
# criterion): the four-term score below puts the true pair in the top 5 in 93% of
# them and the top 10 in 98%, against 87% top-5 for facing-and-clearance alone.
W_FACING = 0.35
W_PROXIMITY = 0.35
W_CLEARANCE = 0.15
W_DEPTH = 0.15

# Anchor separation beyond the fragments' closest approach at which the proximity
# term is fully discounted. A bridge that spans much more than the narrowest gap
# spends atoms and molecular weight to reach around the fragments rather than
# between them.
PROXIMITY_SCALE = 6.0


# =============================================================================
# Structure input
# =============================================================================


def parse_pocket_coords(pdb_text: str) -> np.ndarray:
    """Heavy-atom coordinates from a PDB, as an (N, 3) array.

    A hand-rolled column parse rather than ``Chem.MolFromPDBBlock`` because the
    only thing wanted is coordinates, and RDKit's PDB reader returns None for
    the truncated pockets and non-standard residues that turn up in practice --
    turning "the clash check has no protein" into "the request failed".
    Hydrogens and waters are dropped: contacts are judged heavy-atom to
    heavy-atom, and a crystallographic water is not a wall.
    """
    coords: list[tuple[float, float, float]] = []
    for line in pdb_text.splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        if line[17:20].strip().upper() in ("HOH", "WAT", "DOD"):
            continue
        element = line[76:78].strip() or line[12:16].strip()[:1]
        if element.upper() in ("H", "D"):
            continue
        try:
            coords.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
        except (ValueError, IndexError):
            continue
    return np.array(coords, dtype=float) if coords else np.zeros((0, 3))


def load_fragment(sdf_text: str) -> tuple[Optional[Chem.Mol], Optional[str]]:
    """Parse one fragment SDF, keeping its hydrogens and its pose.

    ``removeHs=False`` because an explicit hydrogen is the most direct statement
    of where a substituent would go, and this module uses it when it is there.
    """
    if not sdf_text or not sdf_text.strip():
        return None, "fragment SDF is empty"

    mol = Chem.MolFromMolBlock(sdf_text, removeHs=False, sanitize=True)
    if mol is None:
        # Sanitisation is where fragments of a decomposed ligand tend to fail
        # (an aromatic ring cut open, a nitrogen left over-valent). Parsing
        # without it and then reporting what sanitisation objected to is far
        # more useful than "could not parse".
        raw = Chem.MolFromMolBlock(sdf_text, removeHs=False, sanitize=False)
        if raw is None:
            return None, "could not parse fragment SDF"
        try:
            Chem.SanitizeMol(raw)
        except Exception as exc:  # noqa: BLE001
            return None, f"fragment failed RDKit sanitization: {exc}"
        mol = raw

    if mol.GetNumConformers() == 0:
        return None, "fragment SDF carries no coordinates"
    if not mol.GetConformer().Is3D():
        return None, "fragment SDF coordinates are 2D; a pose is required"
    return mol, None


# =============================================================================
# Substitutable sites
# =============================================================================


@dataclass(frozen=True)
class Site:
    """One substitutable position on a fragment, with its outward geometry."""

    site_id: str
    fragment: int
    atom_index: int
    element: str
    aromatic: bool
    in_ring: bool
    num_hydrogens: int
    hydrogen_index: Optional[int]
    anchor: np.ndarray
    direction: np.ndarray
    tip: np.ndarray
    free_depth: Optional[float]
    nearest_protein: Optional[float]
    outwardness: float
    burial: Optional[float] = None
    target_alignment: Optional[float] = None
    target_distance: Optional[float] = None

    @property
    def score(self) -> float:
        """Site quality in [0, 1]: room in front of it, aimed where it should be.

        Free depth multiplies rather than adds. A vector with nothing but protein
        in front of it is not a usable attachment point at any orientation, and an
        additive score would still rank it above a well-placed one if it happened
        to point at a target -- which is exactly the mistake that puts a linker
        through the pocket wall. STRIFE's premise is the same one: elaboration has
        to aim where there is room.

        Burial multiplies for the same reason in the other direction. Room is
        necessary and not sufficient: the emptiest vector on a buried fragment is
        usually the one pointing out of the protein, where a substituent is free
        to grow and gains nothing by it.

        ``target_alignment`` is measured against the caller's hotspots on a
        one-fragment task and against the partner fragment on a linking task, so
        the same expression serves both.
        """
        depth = 1.0 if self.free_depth is None else min(self.free_depth / DEPTH_SCALE, 1.0)
        enclosure = 1.0 if self.burial is None else self.burial
        if self.target_alignment is None:
            return round(depth * enclosure * (0.7 + 0.3 * max(self.outwardness, 0.0)), 4)
        return round(depth * enclosure * (0.5 + 0.5 * max(self.target_alignment, 0.0)), 4)

    @property
    def blocked(self) -> bool:
        """True when the protein sits within one bond length of the exit vector."""
        return self.free_depth is not None and self.free_depth < BOND_LENGTH

    def as_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "fragment": self.fragment,
            "atom_index": self.atom_index,
            "element": self.element,
            "aromatic": self.aromatic,
            "in_ring": self.in_ring,
            "num_hydrogens": self.num_hydrogens,
            "blocked": self.blocked,
            "anchor": [round(float(v), 3) for v in self.anchor],
            "direction": [round(float(v), 3) for v in self.direction],
            "free_depth": None if self.free_depth is None else round(self.free_depth, 2),
            "nearest_protein": (
                None if self.nearest_protein is None else round(self.nearest_protein, 2)
            ),
            "outwardness": round(self.outwardness, 3),
            "burial": None if self.burial is None else round(self.burial, 3),
            "target_alignment": (
                None if self.target_alignment is None else round(self.target_alignment, 3)
            ),
            "target_distance": (
                None if self.target_distance is None else round(self.target_distance, 2)
            ),
            "score": self.score,
        }


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])


def cone_directions(direction: np.ndarray) -> np.ndarray:
    """The axis plus rings of spokes tilted off it, within the cone half-angle.

    Deterministic rather than sampled: the same fragment has to yield the same
    depths on every call, and a fixed lattice of 17 rays resolves a cone this
    wide as well as a random draw of any size worth paying for.
    """
    axis = _unit(direction)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(axis @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    side = _unit(np.cross(axis, reference))
    up = np.cross(axis, side)

    directions = [axis]
    for tilt in CONE_RINGS:
        polar = math.radians(tilt)
        for spoke in range(CONE_SPOKES):
            azimuth = 2.0 * math.pi * spoke / CONE_SPOKES
            directions.append(
                _unit(
                    math.cos(polar) * axis
                    + math.sin(polar) * (math.cos(azimuth) * side + math.sin(azimuth) * up)
                )
            )
    return np.array(directions)


def march(
    anchor: np.ndarray,
    direction: np.ndarray,
    protein: np.ndarray,
    max_depth: float = MAX_DEPTH,
) -> float:
    """How far a single ray from ``anchor`` runs before the protein obstructs it.

    Marched rather than solved analytically because the answer wanted is "how
    far until the first obstruction", and marching gives that for an arbitrary
    set of obstacles in three lines. The march starts one bond length out: the
    anchor's own protein contacts are not obstructions to its substituent.
    """
    distance = BOND_LENGTH
    reached = 0.0
    while distance <= max_depth:
        point = anchor + direction * distance
        if float(np.min(np.linalg.norm(protein - point, axis=1))) < PROBE_CLEARANCE:
            break
        reached = distance
        distance += DEPTH_STEP
    return reached


def free_depth(
    anchor: np.ndarray,
    direction: np.ndarray,
    protein: np.ndarray,
    max_depth: float = MAX_DEPTH,
) -> float:
    """How far from the anchor a substituent can reach around ``direction``.

    The deepest ray in the cone, not the axial one. A substituent is free to bend
    away from the exit vector, so what limits it is the best path within the cone
    rather than the one direction the bond happens to point.
    """
    if len(protein) == 0:
        return max_depth
    return max(
        march(anchor, ray, protein, max_depth) for ray in cone_directions(direction)
    )


def burial(anchor: np.ndarray, direction: np.ndarray, protein: np.ndarray) -> Optional[float]:
    """How enclosed the space in front of the vector is, in [0, 1].

    Protein heavy atoms near a point out along the vector, which separates a
    pocket from the outside of the protein. Depth cannot: a vector pointing into
    solvent runs to the cap unobstructed and scores as the most open one on the
    fragment, when it is the one place a substituent buys no contacts.
    """
    if len(protein) == 0:
        return None
    probe = anchor + _unit(direction) * BURIAL_PROBE
    neighbours = int(np.sum(np.linalg.norm(protein - probe, axis=1) < BURIAL_RADIUS))
    return min(neighbours / BURIAL_FULL, 1.0)


def enumerate_sites(
    mol: Chem.Mol,
    fragment_index: int,
    protein: np.ndarray,
    targets: Optional[np.ndarray] = None,
) -> list[Site]:
    """Every substitutable heavy atom of a fragment, with its exit geometry.

    One site per heavy atom, not per hydrogen: a CH2 has two hydrogens pointing
    different ways, and the one worth reporting is whichever has more room, since
    which of a pair a substituent takes is a consequence of the substituent, not
    a separate design choice.
    """
    positions = mol.GetConformer().GetPositions()
    heavy_indices = [
        atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1
    ]
    centroid = positions[heavy_indices].mean(axis=0) if heavy_indices else positions.mean(axis=0)

    sites: list[Site] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        num_hydrogens = atom.GetTotalNumHs(includeNeighbors=True)
        if num_hydrogens < 1:
            continue

        anchor = positions[atom.GetIdx()]
        explicit_hs = [
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        ]
        heavy_neighbors = [
            neighbor.GetIdx()
            for neighbor in atom.GetNeighbors()
            if neighbor.GetAtomicNum() > 1
        ]

        if explicit_hs:
            candidates = [(_unit(positions[h] - anchor), h) for h in explicit_hs]
        elif heavy_neighbors:
            # Away from the mean of the heavy neighbours: along the bond for a
            # terminal atom, out of the ring plane's centre for an aromatic CH,
            # along the missing tetrahedral vertex for an sp3 CH.
            neighbor_centroid = positions[heavy_neighbors].mean(axis=0)
            candidates = [(_unit(anchor - neighbor_centroid), None)]
        else:
            continue

        best: Optional[Site] = None
        for direction, hydrogen_index in candidates:
            depth = free_depth(anchor, direction, protein) if len(protein) else None
            nearest = (
                float(np.min(np.linalg.norm(protein - anchor, axis=1)))
                if len(protein)
                else None
            )
            alignment, target_distance = _target_geometry(anchor, direction, targets)
            site = Site(
                site_id=f"F{fragment_index}:{atom.GetSymbol()}{atom.GetIdx()}",
                fragment=fragment_index,
                atom_index=atom.GetIdx(),
                element=atom.GetSymbol(),
                aromatic=atom.GetIsAromatic(),
                in_ring=atom.IsInRing(),
                num_hydrogens=num_hydrogens,
                hydrogen_index=hydrogen_index,
                anchor=anchor,
                direction=direction,
                tip=anchor + direction * BOND_LENGTH,
                free_depth=depth,
                burial=burial(anchor, direction, protein) if len(protein) else None,
                nearest_protein=nearest,
                outwardness=float(np.dot(direction, _unit(anchor - centroid))),
                target_alignment=alignment,
                target_distance=target_distance,
            )
            if best is None or site.score > best.score:
                best = site
        if best is not None:
            sites.append(best)

    sites.sort(key=lambda site: -site.score)
    return sites


def _target_geometry(
    anchor: np.ndarray, direction: np.ndarray, targets: Optional[np.ndarray]
) -> tuple[Optional[float], Optional[float]]:
    """Best alignment with a supplied hotspot, and how far away that hotspot is.

    "Best" is by alignment rather than by distance: a hotspot 3 A behind the
    fragment is not reachable from this site, however close it is.
    """
    if targets is None or len(targets) == 0:
        return None, None
    offsets = targets - anchor
    distances = np.linalg.norm(offsets, axis=1)
    usable = distances > 1e-6
    if not np.any(usable):
        return None, None
    alignments = (offsets[usable] / distances[usable, None]) @ direction
    best = int(np.argmax(alignments))
    return float(alignments[best]), float(distances[usable][best])


# =============================================================================
# Pair descriptors and linker sizing
# =============================================================================


def linker_atom_range(anchor_distance: float) -> tuple[int, int]:
    """Heavy-atom count window for a bridge spanning ``anchor_distance``.

    A bridge of ``n`` atoms makes ``n + 1`` bonds, each contributing between
    about 1.1 A (a zig-zag chain's projection onto the axis) and 1.5 A (a fully
    extended chain or a rigid rod) of span, plus a one-atom cushion on each side.

    The cushions are not decoration. Measured on 46 PDBBind complexes decomposed
    the way SMDD-Bench builds its tasks, real crystal linkers span 0.77 to 1.26 A
    of anchor separation per bond, median 1.06 -- more coiled than the taut
    estimate. The window this returns corresponds to roughly 0.8-1.35 A per bond,
    which is why every one of those 46 true linker lengths falls inside it.
    """
    lower = math.ceil(anchor_distance / 1.5) - 1
    upper = math.floor(anchor_distance / 1.1) + 2
    lower = max(1, lower)
    return lower, max(lower, upper)


def mask_atom_range(path_min: int, path_max: int) -> tuple[int, int]:
    """Turn a path-length window into GenMol mask bounds.

    ``[*{m-n}]`` bounds the *size of the generated fragment*, not the number of
    atoms it puts between the anchors, and the two differ whenever the bridge
    contains a ring. Every ring-bearing entry in :mod:`linker_library` carries
    between 1.5 and 1.7 times as many atoms as it places on the path -- a
    para-phenylene is six atoms for a four-atom path, a triazole five for three.
    A mask ceiling taken straight from the path window therefore forbids exactly
    the rigid bridges that a long span needs.
    """
    return path_min, math.ceil(RING_ATOM_FACTOR * path_max)


def saturated_chain_rb_cost(num_atoms: int) -> int:
    """Rotatable bonds a saturated chain of ``num_atoms`` would spend.

    ``n + 1`` bonds, all rotatable. This is the number that makes the long-linker
    case interesting: at ``n = 6`` a chain has already spent 7 of the 10 bonds
    Table 6 allows for the entire molecule.
    """
    return num_atoms + 1


@dataclass
class Pair:
    """Two sites on different fragments, and what it would take to bridge them."""

    site_a: Site
    site_b: Site
    anchor_distance: float
    tip_distance: float
    exit_vector_angle: float
    alignment_a: float
    alignment_b: float
    tube_clearance: Optional[float]
    linker_min: int
    linker_max: int
    rb_headroom: int
    separation: Optional[float] = None
    templates: dict = field(default_factory=dict)
    linkers: list = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        """Pair quality in [0, 1], from four terms that each fail differently.

        - **facing**: do the two exit vectors point at each other. This is
          DeLinker's finding about long linkers, reduced to one number.
        - **proximity**: does this pair bridge the narrowest gap between the
          fragments, or reach around them. Empirically the strongest single term.
        - **clearance**: is the straight tube between the anchors free of protein.
        - **depth**: is there room in front of both exit vectors at all.
        """
        facing = ((self.alignment_a + self.alignment_b) / 2.0 + 1.0) / 2.0
        if self.tube_clearance is None:
            clearance = 1.0
        else:
            clearance = min(max((self.tube_clearance - 2.4) / 1.6, 0.0), 1.0)
        depths = [
            1.0 if site.free_depth is None else min(site.free_depth / DEPTH_SCALE, 1.0)
            for site in (self.site_a, self.site_b)
        ]
        if self.separation is None:
            proximity = 1.0
        else:
            excess = self.anchor_distance - self.separation
            proximity = min(max(1.0 - excess / PROXIMITY_SCALE, 0.0), 1.0)
        return round(
            W_FACING * facing
            + W_PROXIMITY * proximity
            + W_CLEARANCE * clearance
            + W_DEPTH * min(depths),
            4,
        )

    @property
    def pair_id(self) -> str:
        return f"{self.site_a.site_id}|{self.site_b.site_id}"

    @property
    def requires_rigid_linker(self) -> bool:
        """True when no saturated chain long enough fits the rotatable-bond gate."""
        return saturated_chain_rb_cost(self.linker_min) > self.rb_headroom

    def as_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "site_a": self.site_a.site_id,
            "site_b": self.site_b.site_id,
            "fragment_a": self.site_a.fragment,
            "fragment_b": self.site_b.fragment,
            "atom_a": self.site_a.atom_index,
            "atom_b": self.site_b.atom_index,
            "anchor_distance": round(self.anchor_distance, 2),
            "tip_distance": round(self.tip_distance, 2),
            "exit_vector_angle": round(self.exit_vector_angle, 1),
            "alignment_a": round(self.alignment_a, 3),
            "alignment_b": round(self.alignment_b, 3),
            "tube_clearance": (
                None if self.tube_clearance is None else round(self.tube_clearance, 2)
            ),
            "linker_atoms_min": self.linker_min,
            "linker_atoms_max": self.linker_max,
            "rb_headroom": self.rb_headroom,
            "saturated_chain_rb_cost": saturated_chain_rb_cost(self.linker_min),
            "requires_rigid_linker": self.requires_rigid_linker,
            "score": self.score,
            "templates": self.templates,
            "linkers": [entry.as_dict() for entry in self.linkers],
            "warnings": self.warnings,
        }


def tube_clearance(
    anchor_a: np.ndarray, anchor_b: np.ndarray, protein: np.ndarray
) -> Optional[float]:
    """Closest protein approach along the straight line between two anchors.

    A straight tube is the optimistic case, so a clear one does not prove a
    linker fits, but a blocked one says the bridge has to arc -- which is a real
    constraint on which library entries can work and worth knowing before
    embedding anything.
    """
    if len(protein) == 0:
        return None
    span = float(np.linalg.norm(anchor_b - anchor_a))
    if span <= 2 * TUBE_MARGIN:
        return None
    direction = (anchor_b - anchor_a) / span
    offsets = np.arange(TUBE_MARGIN, span - TUBE_MARGIN, TUBE_STEP)
    if len(offsets) == 0:
        return None
    points = anchor_a + direction * offsets[:, None]
    return float(np.min(np.linalg.norm(protein[None, :, :] - points[:, None, :], axis=2)))


def pair_sites(
    site_a: Site,
    site_b: Site,
    protein: np.ndarray,
    rb_headroom: int,
    separation: Optional[float] = None,
) -> Pair:
    """Descriptors for bridging one pair of sites.

    ``separation`` is the closest heavy-atom approach between the two fragments,
    which the proximity term of :attr:`Pair.score` needs to tell a bridge across
    the narrowest gap from one reaching around the outside.
    """
    anchor_distance = float(np.linalg.norm(site_b.anchor - site_a.anchor))
    tip_distance = float(np.linalg.norm(site_b.tip - site_a.tip))
    axis = _unit(site_b.anchor - site_a.anchor)
    angle = math.degrees(
        math.acos(float(np.clip(np.dot(site_a.direction, site_b.direction), -1.0, 1.0)))
    )
    linker_min, linker_max = linker_atom_range(anchor_distance)

    return Pair(
        site_a=site_a,
        site_b=site_b,
        anchor_distance=anchor_distance,
        tip_distance=tip_distance,
        exit_vector_angle=angle,
        alignment_a=float(np.dot(site_a.direction, axis)),
        alignment_b=float(np.dot(site_b.direction, -axis)),
        tube_clearance=tube_clearance(site_a.anchor, site_b.anchor, protein),
        linker_min=linker_min,
        linker_max=linker_max,
        rb_headroom=rb_headroom,
        separation=separation,
    )


# =============================================================================
# GenMol templates
# =============================================================================


def _placeholder_element(mols: list[Chem.Mol]) -> Optional[str]:
    """An element none of the fragments contains, to stand in for the mask."""
    present = {
        atom.GetSymbol() for mol in mols for atom in mol.GetAtoms()
    }
    for element in PLACEHOLDER_ELEMENTS:
        if element not in present:
            return element
    return None


def marker_template(
    mol_a: Chem.Mol,
    site_a: int,
    mol_b: Optional[Chem.Mol] = None,
    site_b: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    """GenMol v2's dummy-marker form: ``CC(*)c1ccccc1.*c1ccc(F)cc1``.

    Each fragment carries one bare attachment point and the model decides what
    joins them. No mask bounds, so no way to tell it how long the bridge should
    be -- which is why the SAFE form is emitted alongside this one.
    """
    marked_a, error = lib.mark_attachment(mol_a, site_a, lib.ATTACH_A)
    if error:
        return None, f"fragment A: {error}"
    pieces = [_bare_attachment(marked_a)]

    if mol_b is not None and site_b is not None:
        marked_b, error = lib.mark_attachment(mol_b, site_b, lib.ATTACH_A)
        if error:
            return None, f"fragment B: {error}"
        pieces.append(_bare_attachment(marked_b))

    return ".".join(pieces), None


def _bare_attachment(mol: Chem.Mol) -> str:
    """SMILES with unnumbered ``*`` attachment points, which is what v2 expects."""
    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(editable.GetMol())


def safe_template(
    mol_a: Chem.Mol,
    site_a: int,
    linker_min: int,
    linker_max: int,
    mol_b: Optional[Chem.Mol] = None,
    site_b: Optional[int] = None,
) -> tuple[Optional[str], Optional[str]]:
    """GenMol's documented linker form: fragments plus an interior ``[*{m-n}]``.

    NVIDIA's own example is
    ``c14ncnc2[nH]ccc12.C136CN5C1.[*{5-15}].S5(=O)(=O)CC.C6C#N``: the mask is a
    fragment in its own right, and the bonds it will make are the ring-closure
    digits left dangling in its neighbours. Getting those digits right by string
    surgery is not feasible, so this builds a real molecule with a placeholder
    atom where the linker goes, lets ``safe-mol`` do the numbering, and swaps the
    placeholder's fragment for the mask.

    The mask is placed *between* the fragment tokens, matching the documented
    example. Fragment order in a SAFE string carries no chemistry -- the bonds
    are in the digits -- but the model was trained on strings that read this way.
    """
    fragments = [mol_a] if mol_b is None else [mol_a, mol_b]
    element = _placeholder_element(fragments)
    if element is None:
        return None, "fragments already contain every usable placeholder element"

    placeholder = (
        f"[*:{lib.ATTACH_A}][{element}]"
        if mol_b is None
        else f"[*:{lib.ATTACH_A}][{element}][*:{lib.ATTACH_B}]"
    )
    composite, error = lib.assemble(mol_a, site_a, mol_b, site_b, placeholder)
    if error or composite is None:
        return None, f"could not build the placeholder molecule: {error}"

    try:
        encoded = sf.encode(composite, slicer=[f"[{element}]~[!{element}]"])
    except Exception as exc:  # noqa: BLE001 - fall back to the marker form
        return None, f"SAFE encoding of the placeholder molecule failed: {exc}"
    if not encoded:
        return None, "SAFE encoding of the placeholder molecule returned nothing"

    tokens = encoded.split(".")
    placeholder_tokens = [index for index, token in enumerate(tokens) if element in token]
    if len(placeholder_tokens) != 1:
        return None, (
            f"expected exactly one {element} fragment in the SAFE encoding, "
            f"found {len(placeholder_tokens)}"
        )

    mask = f"[*{{{linker_min}-{linker_max}}}]"
    remaining = [token for index, token in enumerate(tokens) if index != placeholder_tokens[0]]
    if len(remaining) == 1:
        return f"{remaining[0]}.{mask}", None
    middle = len(remaining) // 2
    return ".".join(remaining[:middle] + [mask] + remaining[middle:]), None


# =============================================================================
# Entry point
# =============================================================================


def analyze(
    fragment_sdfs: list[str],
    pocket_pdb: str = "",
    target_points: Optional[list[list[float]]] = None,
    max_sites_per_fragment: int = 12,
    max_pairs: int = 10,
    rb_max: int = DEFAULT_RB_MAX,
    assemble_candidates: bool = False,
    max_candidates_per_pair: int = 6,
    rigid_linkers_only: bool = False,
) -> tuple[Optional[dict], Optional[str]]:
    """Analyse fragment poses for linking or growth.

    Args:
        fragment_sdfs: One SDF block per fragment, each carrying its pose.
        pocket_pdb: Receptor PDB. Without it, free depth and tube clearance are
            reported as null and the geometry is topological only.
        target_points: Hotspot coordinates, typically ``probe_grid`` minima, used
            to aim growth vectors on a one-fragment task.
        max_sites_per_fragment: Sites reported per fragment, best first.
        max_pairs: Candidate pairs reported, best first.
        rb_max: Rotatable-bond ceiling the whole molecule must respect.
        assemble_candidates: Also return complete candidate SMILES built from the
            measured linker library.
        max_candidates_per_pair: Library entries assembled per pair.
        rigid_linkers_only: Restrict library matches to the rigid entries.

    Returns:
        Tuple of ``(report, error_message)``.
    """
    if not fragment_sdfs:
        return None, "no fragments supplied"

    mols: list[Chem.Mol] = []
    for index, sdf_text in enumerate(fragment_sdfs):
        mol, error = load_fragment(sdf_text)
        if error or mol is None:
            return None, f"fragment {index}: {error}"
        mols.append(mol)

    protein = parse_pocket_coords(pocket_pdb) if pocket_pdb else np.zeros((0, 3))
    targets = (
        np.array(target_points, dtype=float)
        if target_points
        else None
    )
    if targets is not None and (targets.ndim != 2 or targets.shape[1] != 3):
        return None, "target_points must be a list of [x, y, z] coordinates"

    warnings: list[str] = []
    if len(protein) == 0:
        warnings.append(
            "No pocket supplied, so free depth and tube clearance are unknown and "
            "site ranking reflects fragment geometry only."
        )
    if len(mols) > 2:
        warnings.append(
            f"{len(mols)} fragments given. Pairs are reported for every combination, "
            "but a molecule joining three fragments needs a branch point no pair "
            "descriptor describes."
        )

    fragment_reports = []
    all_sites: list[Site] = []
    total_rb = 0
    for index, mol in enumerate(mols):
        sites = enumerate_sites(mol, index, protein, _targets_for(index, mols, targets))
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(Chem.RemoveHs(mol))
        total_rb += rotatable
        heavy = mol.GetNumHeavyAtoms()
        fragment_reports.append(
            {
                "index": index,
                "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)),
                "num_heavy_atoms": heavy,
                "rotatable_bonds": rotatable,
                "molecular_weight": round(
                    rdMolDescriptors.CalcExactMolWt(Chem.RemoveHs(mol)), 2
                ),
                "num_sites": len(sites),
                "sites": [site.as_dict() for site in sites[:max_sites_per_fragment]],
            }
        )
        all_sites.extend(sites[:max_sites_per_fragment])
        if not sites:
            warnings.append(
                f"Fragment {index} has no substitutable hydrogen, so it cannot be "
                "extended without removing one of its heavy atoms."
            )

    rb_headroom = rb_max - total_rb - NEW_BOND_RB_COST
    if rb_headroom < 0:
        warnings.append(
            f"The fragments already spend {total_rb} rotatable bonds; with the two "
            f"bonds a linker adds there is no room under RB <= {rb_max}. Any bridge "
            "has to be rigid, and even then the gate may be unreachable."
        )

    report: dict = {
        "fragments": fragment_reports,
        "rotatable_bonds_used": total_rb,
        "rb_headroom": rb_headroom,
        "rb_max": rb_max,
        "pocket_atoms": int(len(protein)),
        "pairs": [],
        "candidates": [],
        "warnings": warnings,
    }

    if len(mols) == 1:
        report["growth"] = _growth_report(
            mols[0], all_sites, max_sites_per_fragment, rb_headroom
        )
        report["fragment_separation"] = None
        return report, None

    separations = _fragment_separations(mols)
    report["fragment_separation"] = separations
    closest = {
        (row["fragment_a"], row["fragment_b"]): row["closest_heavy_distance"]
        for row in separations
    }

    pairs: list[Pair] = []
    for site_a in all_sites:
        for site_b in all_sites:
            if site_a.fragment >= site_b.fragment:
                continue
            pairs.append(
                pair_sites(
                    site_a,
                    site_b,
                    protein,
                    rb_headroom,
                    separation=closest.get((site_a.fragment, site_b.fragment)),
                )
            )
    pairs.sort(key=lambda pair: -pair.score)
    pairs = pairs[:max_pairs]

    candidates: list[dict] = []
    for pair in pairs:
        mol_a, mol_b = mols[pair.site_a.fragment], mols[pair.site_b.fragment]
        mask_min, mask_max = mask_atom_range(pair.linker_min, pair.linker_max)
        safe_string, safe_error = safe_template(
            mol_a,
            pair.site_a.atom_index,
            mask_min,
            mask_max,
            mol_b,
            pair.site_b.atom_index,
        )
        marker, marker_error = marker_template(
            mol_a, pair.site_a.atom_index, mol_b, pair.site_b.atom_index
        )
        pair.templates = {
            "safe": safe_string,
            "safe_error": safe_error,
            "marker": marker,
            "marker_error": marker_error,
            "mask_min": mask_min,
            "mask_max": mask_max,
        }
        pair.linkers = lib.matching_linkers(
            pair.anchor_distance,
            rb_headroom=max(rb_headroom, 0),
            rigid_only=rigid_linkers_only,
            max_results=max_candidates_per_pair,
        )
        if not pair.linkers:
            pair.warnings.append(
                f"No library entry spans {pair.anchor_distance:.1f} A within a "
                f"rotatable-bond headroom of {rb_headroom}; this pair needs a "
                "generated linker or a different exit vector."
            )
        if pair.requires_rigid_linker:
            pair.warnings.append(
                f"A saturated chain long enough for {pair.anchor_distance:.1f} A "
                f"costs {saturated_chain_rb_cost(pair.linker_min)} rotatable bonds "
                f"against a headroom of {rb_headroom}. Use a rigid bridge: aryl, "
                "heteroaryl, alkyne, amide or a fused ring."
            )

        if assemble_candidates:
            for entry in pair.linkers:
                smiles, error = lib.assemble(
                    mol_a,
                    pair.site_a.atom_index,
                    mol_b,
                    pair.site_b.atom_index,
                    lib.LINKERS_BY_NAME[entry.name].smiles,
                )
                if error or smiles is None:
                    logger.debug(f"Assembly failed for {entry.name}: {error}")
                    continue
                candidates.append(
                    {
                        "smiles": smiles,
                        "pair_id": pair.pair_id,
                        "linker": entry.name,
                        "linker_atoms": entry.num_linker_atoms,
                        "linker_rb_cost": entry.rb_cost,
                        "anchor_distance": round(pair.anchor_distance, 2),
                        "atom_a": pair.site_a.atom_index,
                        "atom_b": pair.site_b.atom_index,
                    }
                )

    report["pairs"] = [pair.as_dict() for pair in pairs]
    report["candidates"] = candidates
    return report, None


def _targets_for(
    index: int, mols: list[Chem.Mol], explicit: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """What this fragment's exit vectors should be aiming at.

    Explicit targets win: a caller passing ``probe_grid`` minima wants vectors
    ranked by those. Otherwise, on a linking task, the thing a fragment's exit
    vector should point at is its partner, so the partner's centroid stands in.
    That is what makes the top-ranked sites the plausible attachment points
    rather than merely the most solvent-exposed ones.
    """
    if explicit is not None and len(explicit):
        return explicit
    if len(mols) < 2:
        return None
    centroids = [
        mol.GetConformer().GetPositions()[
            [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        ].mean(axis=0)
        for position, mol in enumerate(mols)
        if position != index
    ]
    return np.array(centroids)


def _fragment_separations(mols: list[Chem.Mol]) -> list[dict]:
    """Closest heavy-atom approach between each pair of fragments.

    SMDD-Bench builds two-fragment tasks only from fragments at least 6 A apart,
    so this number tells the caller which regime it is in before it looks at a
    single exit vector.
    """
    separations = []
    for i in range(len(mols)):
        for j in range(i + 1, len(mols)):
            coords_i = mols[i].GetConformer().GetPositions()[
                [a.GetIdx() for a in mols[i].GetAtoms() if a.GetAtomicNum() > 1]
            ]
            coords_j = mols[j].GetConformer().GetPositions()[
                [a.GetIdx() for a in mols[j].GetAtoms() if a.GetAtomicNum() > 1]
            ]
            distances = np.linalg.norm(coords_i[:, None, :] - coords_j[None, :, :], axis=2)
            separations.append(
                {
                    "fragment_a": i,
                    "fragment_b": j,
                    "closest_heavy_distance": round(float(np.min(distances)), 2),
                    "centroid_distance": round(
                        float(np.linalg.norm(coords_i.mean(axis=0) - coords_j.mean(axis=0))), 2
                    ),
                }
            )
    return separations


def _growth_report(
    mol: Chem.Mol, sites: list[Site], max_sites: int, rb_headroom: int
) -> dict:
    """Growth templates for a one-fragment task.

    A single fragment is not a linking problem, it is an elaboration problem:
    pick the vector with room in front of it, aim it at a hotspot, and extend.
    The mask bounds come from the free depth rather than from a partner's
    position, since there is no partner to reach.
    """
    entries = []
    for site in sites[:max_sites]:
        depth = site.free_depth if site.free_depth is not None else MAX_DEPTH
        path_min, path_max = linker_atom_range(max(depth, 1.5))
        mask_min, mask_max = mask_atom_range(path_min, path_max)
        safe_string, safe_error = safe_template(mol, site.atom_index, mask_min, mask_max)
        marker, marker_error = marker_template(mol, site.atom_index)
        entries.append(
            {
                "site_id": site.site_id,
                "atom_index": site.atom_index,
                "free_depth": None if site.free_depth is None else round(site.free_depth, 2),
                "burial": None if site.burial is None else round(site.burial, 3),
                "blocked": site.blocked,
                "score": site.score,
                "linker_atoms_min": path_min,
                "linker_atoms_max": path_max,
                "mask_min": mask_min,
                "mask_max": mask_max,
                "templates": {
                    "safe": safe_string,
                    "safe_error": safe_error,
                    "marker": marker,
                    "marker_error": marker_error,
                },
            }
        )
    return {"rb_headroom": rb_headroom, "vectors": entries}
