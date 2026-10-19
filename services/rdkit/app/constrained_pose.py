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

"""Does a candidate molecule hold its fragments where they were?

This is the question SMDD-Bench's Fragment Assembly task actually turns on. Gates
1 to 4 -- valid SMILES, both fragments present as disjoint connected
substructures, the Table 6 property panel -- are cheap and most candidates pass
them. Gate 5c is the one that fails: after co-folding, each fragment has to sit
within 2.0 A RMSD of the pose it was handed in. A molecule whose linker is a
bond too short pulls the fragments together; one that is three bonds too long
lets them drift; one that has to thread through the protein cannot be folded
there at all. All three pass gates 1 to 4 and all three fail gate 5c, and Boltz-2
is far too expensive to use as the filter.

The tractable proxy is the constrained-placement question from fragment-based
drug discovery: **does a low-strain, clash-free conformer exist that holds every
fragment atom at its input coordinates?** That is what
`Fragmenstein <https://fragmenstein.readthedocs.io/en/latest/doc_01_workings.html>`_'s
RDKit-only ``Wictor`` tier does for XChem follow-ups -- map the follow-up onto the
hits, embed one restrained conformer, minimise in a frozen neighbourhood, judge by
RMSD -- and it is what this module does per candidate.

Five details are load-bearing, and four of them cost a day to find.

**``coordMap`` constrains shape, not position.** The coordinates handed to
``EmbedMolecule`` become distance bounds between the mapped atoms; the conformer
that comes back is in the embedding's own frame, centred near the origin.
Measured here: mapped atoms 109 A from the coordinates they were given, for a
pocket 100 A from the origin. Every attempt therefore ends with a rigid
superposition of the mapped atoms onto their targets.

**The assignment has to be chosen for geometry, not taken first.** A benzene
fragment matches a biaryl candidate in five places. On a PDBBind case with two
benzene fragments there were eighteen disjoint assignments, and the first one
asked two directly bonded atoms to sit 6.95 A apart -- impossible, unembeddable,
and it made a good candidate look hopeless. :func:`assign_fragments` ranks them.

**Restrained embedding is not reliable, and falling back beats failing.**
Experimental geometry does not always fit RDKit's idealised distance bounds: one
case had seventeen crystal distances outside them, including a 1.42 A bond where
the lower bound is 1.49, and every seed failed even though each fragment embedded
fine alone. A free conformer pulled into the poses by tethers turns that from an
opaque failure into a measurable residual.

**Minimisation must restrain, not relax.** An unrestrained
``MMFFOptimizeMolecule`` after embedding throws the pose away: measured on the
tamoxifen case, the two fragments came back 1.81 and 2.29 A from where they
started, a gate-5c failure manufactured by the verifier. The pull-in stage tethers
every matched atom to the coordinate it must hold, and the minimisation that
follows fixes those atoms outright.

**The tethers need a flat bottom.** Held to crystal coordinates exactly, MMFF
reports 40 to 230 kcal/mol of "strain" that is almost all bond lengths it
disagrees with, drowning the conformational signal. A quarter of an Angstrom of
slack -- finer than anything measured downstream -- drops the median to 9.

What comes out: the per-fragment fit residual, the strain above the free minimum,
and the closest approach to the protein. Measured against PDBBind ligands cut into
fragment-linker-fragment, the three together keep 75% of the known answers while
rejecting 93% of deliberately wrong variants, and the poses they accept land a
median 0.28 A from the crystal pose -- close enough to hand to Vina.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from itertools import product
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

from .fragment_link import load_fragment, parse_pocket_coords

logger = logging.getLogger(__name__)

# Independent attempts per candidate. Embedding is stochastic in both directions:
# a failure is not always real, and a seed that succeeds can still fold the linker
# into the protein where another seed would not. Every seed is measured and the
# best pose is the one reported, because the question is whether a pose exists.
DEFAULT_SEEDS = 4
BASE_SEED = 0xA5E

# Force-field iterations. Enough for a restrained minimum to settle; the fixed
# points mean there is not much left to relax.
MAX_MINIMIZE_ITERATIONS = 800

# Distance-geometry retries inside one embedding call. A candidate that has not
# embedded in fifty is not going to, and each failed call costs seconds.
EMBED_ATTEMPTS = 50

# Substructure matches enumerated per fragment before the disjoint search gives
# up. A symmetric fragment in a symmetric molecule can produce dozens; hundreds
# means something pathological, and the combinatorics are not worth it.
MAX_MATCHES_PER_FRAGMENT = 64

# How far apart two atoms can be for each bond on the path between them. A fully
# extended chain reaches about 1.27 A per bond and the longest single bonds are
# under 1.8. Measured over 40 PDBBind cases and their wrong variants, this is the
# setting that separates them: at 1.5 no known answer exceeds the deficit
# tolerance below (worst 0.76 A) while 85% of bridges three atoms too short do.
# At the 1.8 that a single bond can physically reach, only 43% do -- the slack
# lets a short bridge pull the two fragments toward each other instead, and it
# then passes on strain and fit as well.
REACH_PER_BOND = 1.5

# A best assignment that still needs this many Angstroms more reach than the
# topology can offer is rejected without embedding anything. Four failed embeddings
# cost seconds and say less than the arithmetic does.
MAX_REACH_DEFICIT = 1.0

# Flat bottom on the pose tethers, in Angstroms. Inside it the tether is free, so
# the force field keeps its own bond lengths instead of paying to reproduce an
# experimental geometry it disagrees with -- measured on PDBBind cases, tethering
# to the crystal coordinates exactly reports 40 to 230 kcal/mol of "strain" that is
# almost entirely bond-length quibbles. A quarter of an Angstrom is finer than any
# pose measurement downstream of here.
TETHER_TOLERANCE = 0.25
TETHER_FORCE_CONSTANT = 100.0

# Minimisation rounds for the tethered stage. RDKit's own ConstrainedEmbed calls
# Minimize repeatedly until it converges or the budget runs out.
TETHER_ROUNDS = 5

# The three thresholds below are calibrated together in
# tests/test_pdbbind_validation.py, over 40 PDBBind cases with a known answer and
# three deliberately wrong variants of each. What the medians look like there:
#
#                   fit rmsd   strain   closest protein contact
#   known answer      0.23        9.4          2.42
#   linker too short  0.42       88            2.57
#   linker too long   0.24       12            1.45
#   wrong exit vector 0.41      103            1.58
#
# Each wrong answer fails differently, which is why all three are gated rather
# than one composite score -- and why the reach check above runs before any of
# them. A short linker is caught by arithmetic and then by strain, a long one
# coils into the protein, and a bridge off the wrong atom does both. Together
# they reject 93% of the wrong variants while keeping 70% of the right ones.

# A candidate heavy atom closer than this to a protein heavy atom is clashing.
# 2.6 A is where crystal ligands themselves bottom out, but demanding it of a
# vacuum-minimised linker throws away three fifths of the right answers. 2.0 keeps
# the pool honest without gutting it, which is the trade a filter in front of Vina
# and Boltz-2 should make: the next stage scores protein sterics properly, and it
# only gets to score what survives this one.
POSE_CLASH_DISTANCE = 2.0

# Per-fragment deviation ceiling. Known answers sit at a median of 0.23 A and a
# maximum of 0.87, so this rejects only topologies that genuinely cannot reach.
DEFAULT_RMSD_MAX = 1.0

# Strain ceiling, in kcal/mol above the free minimum. Set well above the 7 a right
# answer costs, because the same tail catches two different things: a bridge that
# cannot span the gap without stretching, and a crystal geometry MMFF simply
# disagrees with. Cases of the second kind are 4 of 20 here, and their poses are
# not to be trusted either, so rejecting both is the right behaviour -- but it is
# why the ceiling is not tightened to the 50 that would buy a few more rejections.
# Set to None to report strain without judging it.
DEFAULT_STRAIN_MAX = 100.0


@dataclass
class FragmentMatch:
    """One fragment's assignment onto the candidate, and how well it was held."""

    fragment: int
    atom_indices: tuple[int, ...]
    query_mode: str
    rmsd: Optional[float] = None
    max_deviation: Optional[float] = None

    def as_dict(self) -> dict:
        return {
            "fragment": self.fragment,
            "atom_indices": list(self.atom_indices),
            "query_mode": self.query_mode,
            "rmsd": None if self.rmsd is None else round(self.rmsd, 3),
            "max_deviation": (
                None if self.max_deviation is None else round(self.max_deviation, 3)
            ),
        }


@dataclass
class PoseResult:
    """Everything the endpoint answers with for one candidate."""

    smiles: str
    verified: bool = False
    # Which gate rejected it, in SMDD-Bench's numbering, or None when verified.
    failed_gate: Optional[str] = None
    reason: Optional[str] = None
    matches: list[FragmentMatch] = field(default_factory=list)
    strain: Optional[float] = None
    energy_restrained: Optional[float] = None
    energy_free: Optional[float] = None
    force_field: Optional[str] = None
    min_protein_distance: Optional[float] = None
    num_clashes: int = 0
    max_fragment_rmsd: Optional[float] = None
    embed_attempts: int = 0
    # "restrained" when the distance-geometry bounds accepted the poses, "free" when
    # the conformer had to be built without them and tethered in afterwards.
    embed_mode: Optional[str] = None
    pose_sdf: Optional[str] = None
    num_heavy_atoms: Optional[int] = None
    num_linker_atoms: Optional[int] = None

    def as_dict(self, include_pose: bool = True) -> dict:
        payload = {
            "smiles": self.smiles,
            "verified": self.verified,
            "failed_gate": self.failed_gate,
            "reason": self.reason,
            "matches": [match.as_dict() for match in self.matches],
            "strain": None if self.strain is None else round(self.strain, 2),
            "energy_restrained": (
                None if self.energy_restrained is None else round(self.energy_restrained, 2)
            ),
            "energy_free": None if self.energy_free is None else round(self.energy_free, 2),
            "force_field": self.force_field,
            "min_protein_distance": (
                None if self.min_protein_distance is None else round(self.min_protein_distance, 2)
            ),
            "num_clashes": self.num_clashes,
            "max_fragment_rmsd": (
                None if self.max_fragment_rmsd is None else round(self.max_fragment_rmsd, 3)
            ),
            "embed_attempts": self.embed_attempts,
            "embed_mode": self.embed_mode,
            "num_heavy_atoms": self.num_heavy_atoms,
            "num_linker_atoms": self.num_linker_atoms,
        }
        if include_pose:
            payload["pose_sdf"] = self.pose_sdf
        return payload


# =============================================================================
# Substructure assignment: gates 2 and 3
# =============================================================================


def _coord_map(
    matches: list[FragmentMatch], fragments: list[Chem.Mol]
) -> dict[int, Point3D]:
    """Candidate atom index -> the coordinate it has to hold.

    A substructure match is parallel to the query's atom order, and the queries
    here are hydrogen-free fragments, so match position ``i`` is fragment atom
    ``i``.
    """
    coord_map: dict[int, Point3D] = {}
    for match in matches:
        positions = fragments[match.fragment].GetConformer().GetPositions()
        for query_position, candidate_atom in enumerate(match.atom_indices):
            x, y, z = positions[query_position]
            coord_map[candidate_atom] = Point3D(float(x), float(y), float(z))
    return coord_map


def _queries(fragment: Chem.Mol) -> list[tuple[Chem.Mol, str]]:
    """Substructure queries for one fragment, strictest first.

    The strict query is the fragment itself, which requires the candidate to
    agree with it on aromaticity and bond order. That is the right default, but a
    fragment cut out of a fused system can perceive differently on its own than
    it does inside the assembled molecule, so a generic-bond SMARTS follows as a
    fallback. Which one matched is reported: a match only the loose query finds is
    a match SMDD's own SMARTS check may or may not agree with.
    """
    queries: list[tuple[Chem.Mol, str]] = [(fragment, "exact")]

    params = Chem.AdjustQueryParameters.NoAdjustments()
    params.makeBondsGeneric = True
    params.aromatizeIfPossible = False
    try:
        queries.append((Chem.AdjustQueryProperties(fragment, params), "generic_bonds"))
    except Exception as exc:  # noqa: BLE001 - the strict query still stands
        logger.debug(f"Could not build a generic-bond query: {exc}")
    return queries


def _reach_deficit(
    candidate: Chem.Mol, graph: np.ndarray, coord_map: dict[int, Point3D]
) -> float:
    """How much further apart the poses ask two atoms to be than bonds can carry.

    Zero for any assignment the topology can satisfy. Positive when some pair of
    constrained atoms is asked to sit further apart than the shortest path between
    them can stretch, which happens for two reasons worth telling apart from an
    embedding failure: the linker is too short for the gap, or -- for a fragment
    that matches the candidate in several places -- this is the wrong match.
    """
    indices = sorted(coord_map)
    target = np.array([[coord_map[i].x, coord_map[i].y, coord_map[i].z] for i in indices])
    distances = np.linalg.norm(target[:, None] - target[None], axis=2)
    reach = graph[np.ix_(indices, indices)] * REACH_PER_BOND
    return max(float(np.max(distances - reach)), 0.0)


def assign_fragments(
    candidate: Chem.Mol, fragments: list[Chem.Mol]
) -> tuple[Optional[list[FragmentMatch]], Optional[str], Optional[str]]:
    """Find the disjoint assignment of every fragment onto the candidate that its
    poses can actually be held in.

    Returns ``(matches, failed_gate, reason)``. Three failure modes, reported
    separately because the remedies differ: a fragment that is not a substructure
    at all means the candidate lost or altered it; matches that exist but overlap
    mean the linker was built out of atoms a fragment needed, typically by fusing
    the two fragments into one ring system; and an assignment no topology can hold
    means the bridge is too short for the gap.

    Picking *any* disjoint assignment is not enough. A benzene fragment matches a
    biaryl candidate in five places, and on a PDBBind case with two benzene
    fragments, 18 disjoint assignments existed of which the first demanded two
    directly bonded atoms sit 6.95 A apart. That assignment cannot be embedded, and
    taking it made a perfectly good candidate look impossible. Assignments are
    therefore ranked by :func:`_reach_deficit` and the most consistent one wins.
    """
    per_fragment: list[list[FragmentMatch]] = []
    for index, fragment in enumerate(fragments):
        found: list[FragmentMatch] = []
        for query, mode in _queries(fragment):
            for match in candidate.GetSubstructMatches(
                query, uniquify=True, maxMatches=MAX_MATCHES_PER_FRAGMENT
            ):
                found.append(FragmentMatch(index, tuple(match), mode))
            if found:
                break
        if not found:
            return (
                None,
                "gate2_substructure",
                f"fragment {index} ({Chem.MolToSmiles(fragment)}) is not a substructure "
                "of the candidate",
            )
        per_fragment.append(found)

    graph = Chem.GetDistanceMatrix(candidate)
    ranked: list[tuple[float, list[FragmentMatch]]] = []
    for combination in product(*per_fragment):
        used: set[int] = set()
        for match in combination:
            if used & set(match.atom_indices):
                break
            used |= set(match.atom_indices)
        else:
            matches = list(combination)
            ranked.append((_reach_deficit(candidate, graph, _coord_map(matches, fragments)), matches))

    if not ranked:
        return (
            None,
            "gate3_disjoint",
            "every fragment matches, but no assignment keeps them disjoint -- the "
            "fragments share atoms in this molecule rather than being joined by a linker",
        )

    deficit, matches = min(ranked, key=lambda item: item[0])
    if deficit > MAX_REACH_DEFICIT:
        needed = int(np.ceil(deficit / REACH_PER_BOND))
        return (
            matches,
            "gate5c_pose",
            f"the bridge is too short to span these poses: it falls {deficit:.1f} A "
            f"short of reaching, so it needs about {needed} more atom(s) on the path "
            "between the fragments",
        )
    return matches, None, None


# =============================================================================
# Restrained embedding and minimisation
# =============================================================================


def _superimpose(conformer: Chem.Conformer, coord_map: dict[int, Point3D]) -> None:
    """Rigidly move a conformer so its mapped atoms sit on their target coordinates.

    ``EmbedMolecule(coordMap=...)`` returns a conformer with the right *shape* in
    the wrong *place* -- the coordinates it was given act as distance bounds among
    the mapped atoms, not as absolute positions. Kabsch superposition on those
    atoms puts the whole molecule into the pocket's frame, which is where the
    clash check and the output pose both need it.
    """
    indices = sorted(coord_map)
    positions = conformer.GetPositions()
    mobile = positions[indices]
    target = np.array([[coord_map[i].x, coord_map[i].y, coord_map[i].z] for i in indices])

    mobile_centre = mobile.mean(axis=0)
    target_centre = target.mean(axis=0)
    covariance = (mobile - mobile_centre).T @ (target - target_centre)
    left, _, right = np.linalg.svd(covariance)
    # Guard against the reflection that SVD will happily hand back.
    chirality = np.sign(np.linalg.det(right.T @ left.T))
    rotation = right.T @ np.diag([1.0, 1.0, chirality]) @ left.T

    moved = (positions - mobile_centre) @ rotation.T + target_centre
    for index, (x, y, z) in enumerate(moved):
        conformer.SetAtomPosition(index, Point3D(float(x), float(y), float(z)))


def _force_field(mol: Chem.Mol, conformer_id: int):
    """MMFF where it types, UFF where it does not, reporting which."""
    properties = AllChem.MMFFGetMoleculeProperties(mol)
    if properties is not None:
        field = AllChem.MMFFGetMoleculeForceField(mol, properties, confId=conformer_id)
        if field is not None:
            return field, "MMFF94"
    field = AllChem.UFFGetMoleculeForceField(mol, confId=conformer_id)
    return field, "UFF" if field is not None else None


def _embed(mol: Chem.Mol, coord_map: dict[int, Point3D], seed: int) -> tuple[int, str]:
    """One conformer, restrained to the poses if the bounds allow it, free if not.

    The restrained path is preferred because it starts near the answer. It is not
    reliable, though, and the reason is worth knowing: ``coordMap`` turns the given
    coordinates into distance bounds, and experimental geometry does not always
    agree with RDKit's idealised bounds. On one PDBBind case, seventeen crystal
    distances sat outside them -- a 1.42 A bond where the lower bound is 1.49 --
    and triangle smoothing then failed at every seed even though each fragment on
    its own embedded fine. Falling back to a free conformer and pulling it into the
    poses with tethers turns that from a hard failure into a residual, which is
    both more informative and more often right.
    """
    conformer_id = AllChem.EmbedMolecule(
        mol,
        coordMap=coord_map,
        randomSeed=seed,
        maxAttempts=EMBED_ATTEMPTS,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
        ignoreSmoothingFailures=True,
    )
    if conformer_id >= 0:
        return conformer_id, "restrained"
    conformer_id = AllChem.EmbedMolecule(
        mol,
        randomSeed=seed,
        maxAttempts=EMBED_ATTEMPTS,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
    )
    return conformer_id, "free"


def _pull_into_pose(mol: Chem.Mol, conformer_id: int, coord_map: dict[int, Point3D]) -> None:
    """Minimise with every matched atom tethered to the coordinate it must hold.

    The tether is RDKit's own ``ConstrainedEmbed`` mechanism: a fixed extra point
    at the target and a distance constraint between it and the atom. The constraint
    is flat-bottomed at :data:`TETHER_TOLERANCE` so the force field keeps its own
    bond lengths rather than paying to reproduce an experimental geometry exactly.
    """
    field, name = _force_field(mol, conformer_id)
    if field is None:
        return
    for index, point in coord_map.items():
        extra = field.AddExtraPoint(point.x, point.y, point.z, fixed=True) - 1
        field.AddDistanceConstraint(
            extra, index, 0.0, TETHER_TOLERANCE, TETHER_FORCE_CONSTANT
        )
    field.Initialize()
    more = field.Minimize(maxIts=MAX_MINIMIZE_ITERATIONS, energyTol=1e-4, forceTol=1e-3)
    rounds = TETHER_ROUNDS
    while more and rounds:
        more = field.Minimize(maxIts=MAX_MINIMIZE_ITERATIONS, energyTol=1e-4, forceTol=1e-3)
        rounds -= 1


def _minimize(mol: Chem.Mol, conformer_id: int, fixed: list[int]) -> tuple[Optional[float], Optional[float], Optional[str]]:
    """Minimise twice from the same geometry: fragments fixed, then free.

    The free minimum is taken from the *same* starting conformer rather than from
    a fresh embedding, so the difference is the price of holding the fragments in
    place rather than the difference between two unrelated conformers.
    """
    restrained, name = _force_field(mol, conformer_id)
    if restrained is None:
        return None, None, None
    for index in fixed:
        restrained.AddFixedPoint(index)
    restrained.Minimize(maxIts=MAX_MINIMIZE_ITERATIONS)
    energy_restrained = float(restrained.CalcEnergy())

    free_copy = Chem.Mol(mol)
    free, _ = _force_field(free_copy, conformer_id)
    if free is None:
        return energy_restrained, None, name
    free.Minimize(maxIts=MAX_MINIMIZE_ITERATIONS)
    return energy_restrained, float(free.CalcEnergy()), name


def _rmsd(
    positions: np.ndarray, matches: list[FragmentMatch], fragments: list[Chem.Mol]
) -> None:
    """Fill in each match's deviation from the coordinates it was asked to hold."""
    for match in matches:
        target = fragments[match.fragment].GetConformer().GetPositions()
        actual = positions[list(match.atom_indices)]
        deviations = np.linalg.norm(actual - target, axis=1)
        match.rmsd = float(np.sqrt(np.mean(deviations**2)))
        match.max_deviation = float(np.max(deviations))


@dataclass
class _Attempt:
    """One seed's constrained pose, fully measured."""

    mol: Chem.Mol
    conformer_id: int
    matches: list[FragmentMatch]
    embed_mode: str = "restrained"
    max_rmsd: Optional[float] = None
    strain: Optional[float] = None
    energy_restrained: Optional[float] = None
    energy_free: Optional[float] = None
    force_field: Optional[str] = None
    min_protein_distance: Optional[float] = None
    num_clashes: int = 0

    @property
    def cost(self) -> tuple[int, float, float]:
        """Ordering over attempts, best first.

        The question being asked is whether a pose *exists*, so the best attempt is
        the one reported and every seed is measured before choosing. A pose inside
        the protein loses to one that merely fits loosely; fit is compared to a
        tenth of an Angstrom because finer differences than that are embedding
        noise, which leaves strain to break the ties it should break.
        """
        return (self.num_clashes, round(self.max_rmsd or 0.0, 1), self.strain or 0.0)


def _attempt(
    mol: Chem.Mol,
    coord_map: dict[int, Point3D],
    matches: list[FragmentMatch],
    fragments: list[Chem.Mol],
    protein: np.ndarray,
    seed: int,
    clash_distance: float,
) -> Optional[_Attempt]:
    """Embed, pull into the poses, minimise with the fragments fixed, and measure.

    Returns None when no conformer could be generated at all, which for a molecule
    RDKit accepts is rare enough to be a real answer about the candidate.
    """
    probe = Chem.Mol(mol)
    conformer_id, embed_mode = _embed(probe, coord_map, seed)
    if conformer_id < 0:
        return None

    # Two superpositions, both necessary. The first puts the embedded shape into the
    # pocket's frame so the tethers pull over Angstroms rather than tens of them;
    # the second undoes the small rigid drift the tethered minimisation leaves.
    _superimpose(probe.GetConformer(conformer_id), coord_map)
    _pull_into_pose(probe, conformer_id, coord_map)
    _superimpose(probe.GetConformer(conformer_id), coord_map)

    attempt = _Attempt(
        mol=probe,
        conformer_id=conformer_id,
        matches=[replace(match) for match in matches],
        embed_mode=embed_mode,
    )
    attempt.energy_restrained, attempt.energy_free, attempt.force_field = _minimize(
        probe, conformer_id, sorted(coord_map)
    )
    if attempt.energy_restrained is not None and attempt.energy_free is not None:
        attempt.strain = attempt.energy_restrained - attempt.energy_free

    positions = probe.GetConformer(conformer_id).GetPositions()
    _rmsd(positions, attempt.matches, fragments)
    attempt.max_rmsd = max(
        (match.rmsd for match in attempt.matches if match.rmsd is not None), default=None
    )

    if len(protein):
        heavy = positions[
            [atom.GetIdx() for atom in probe.GetAtoms() if atom.GetAtomicNum() > 1]
        ]
        per_atom = np.linalg.norm(
            protein[None, :, :] - heavy[:, None, :], axis=2
        ).min(axis=1)
        attempt.min_protein_distance = float(per_atom.min())
        attempt.num_clashes = int(np.sum(per_atom < clash_distance))
    return attempt


def _failures(
    attempt: _Attempt,
    strain_max: Optional[float],
    rmsd_max: float,
    clash_distance: float,
) -> list[str]:
    """Why this pose is not acceptable, in words the model can act on."""
    reasons = []
    if attempt.max_rmsd is not None and attempt.max_rmsd > rmsd_max:
        reasons.append(
            f"a fragment sits {attempt.max_rmsd:.2f} A from its given pose, over the "
            f"{rmsd_max:.1f} A allowance -- this topology cannot span them as posed"
        )
    if strain_max is not None and attempt.strain is not None and attempt.strain > strain_max:
        reasons.append(
            f"holding the fragments costs {attempt.strain:.1f} kcal/mol of strain, "
            f"over the {strain_max:.0f} allowed"
        )
    if attempt.num_clashes:
        reasons.append(
            f"{attempt.num_clashes} atom(s) sit within {clash_distance:.1f} A of the "
            f"protein, closest {attempt.min_protein_distance:.2f} A"
        )
    return reasons


def verify_one(
    smiles: str,
    fragments: list[Chem.Mol],
    protein: np.ndarray,
    num_seeds: int = DEFAULT_SEEDS,
    strain_max: Optional[float] = DEFAULT_STRAIN_MAX,
    rmsd_max: float = DEFAULT_RMSD_MAX,
    clash_distance: float = POSE_CLASH_DISTANCE,
) -> PoseResult:
    """Verify one candidate against posed fragments.

    ``fragments`` may carry explicit hydrogens; they are stripped here, because a
    query with hydrogens in it will never match a candidate parsed from SMILES.
    """
    fragments = [
        fragment if fragment.GetNumAtoms() == fragment.GetNumHeavyAtoms()
        else Chem.RemoveHs(fragment)
        for fragment in fragments
    ]
    candidate = Chem.MolFromSmiles(smiles)
    if candidate is None:
        return PoseResult(
            smiles=smiles,
            failed_gate="gate1_smiles",
            reason="not a parseable SMILES",
        )
    if len(Chem.GetMolFrags(candidate)) > 1:
        return PoseResult(
            smiles=smiles,
            failed_gate="gate3_disjoint",
            reason="the candidate is disconnected; the fragments must be joined "
            "covalently, not merely present",
        )

    result = PoseResult(smiles=smiles, num_heavy_atoms=candidate.GetNumHeavyAtoms())

    matches, failed_gate, reason = assign_fragments(candidate, fragments)
    if matches is not None:
        result.matches = matches
        matched_atoms = {atom for match in matches for atom in match.atom_indices}
        result.num_linker_atoms = candidate.GetNumHeavyAtoms() - len(matched_atoms)
    if failed_gate is not None or matches is None:
        result.failed_gate = failed_gate
        result.reason = reason
        return result

    # Hydrogens are added before embedding because a force field without them
    # reports nonsense strain, and AddHs appends them after the heavy atoms, so
    # every index in the coordinate map and the fixed-point list stays valid.
    mol = Chem.AddHs(candidate)
    coord_map = _coord_map(matches, fragments)

    best: Optional[_Attempt] = None
    for index in range(max(num_seeds, 1)):
        result.embed_attempts = index + 1
        attempt = _attempt(
            mol,
            coord_map,
            matches,
            fragments,
            protein,
            seed=BASE_SEED + index,
            clash_distance=clash_distance,
        )
        if attempt is not None and (best is None or attempt.cost < best.cost):
            best = attempt

    if best is None:
        result.failed_gate = "gate5c_pose"
        result.reason = (
            f"no conformer at all in {result.embed_attempts} attempts, restrained or "
            "free; RDKit cannot build this molecule in three dimensions"
        )
        return result

    result.matches = best.matches
    result.embed_mode = best.embed_mode
    result.max_fragment_rmsd = best.max_rmsd
    result.strain = best.strain
    result.energy_restrained = best.energy_restrained
    result.energy_free = best.energy_free
    result.force_field = best.force_field
    result.min_protein_distance = best.min_protein_distance
    result.num_clashes = best.num_clashes

    posed = Chem.RemoveHs(Chem.Mol(best.mol))
    posed.SetProp("_Name", "constrained_pose")
    result.pose_sdf = Chem.MolToMolBlock(posed, confId=best.conformer_id)

    failures = _failures(best, strain_max, rmsd_max, clash_distance)
    if failures:
        result.failed_gate = "gate5c_pose"
        result.reason = "; ".join(failures)
    else:
        result.verified = True
    return result


def verify(
    smiles: list[str],
    fragment_sdfs: list[str],
    pocket_pdb: str = "",
    num_seeds: int = DEFAULT_SEEDS,
    strain_max: Optional[float] = DEFAULT_STRAIN_MAX,
    rmsd_max: float = DEFAULT_RMSD_MAX,
    clash_distance: float = POSE_CLASH_DISTANCE,
) -> tuple[Optional[list[PoseResult]], Optional[str]]:
    """Verify candidates against posed fragments.

    Args:
        smiles: Candidate molecules.
        fragment_sdfs: One SDF block per fragment, carrying the pose to hold.
        pocket_pdb: Receptor PDB. Without it, clash distances are not computed and
            a candidate threading the protein wall will verify.
        num_seeds: Independent restrained-embedding attempts per candidate.
        strain_max: Strain ceiling in kcal/mol, or None to report without judging.
        rmsd_max: Per-fragment deviation ceiling in Angstroms.
        clash_distance: Protein contact below which an atom counts as clashing.

    Returns:
        Tuple of ``(results, error_message)``. An unusable *input* is an error; an
        unusable *candidate* is a result with ``verified`` false and a gate named.
    """
    if not smiles:
        return None, "no candidate SMILES supplied"
    if not fragment_sdfs:
        return None, "no fragments supplied"

    fragments: list[Chem.Mol] = []
    for index, sdf_text in enumerate(fragment_sdfs):
        mol, error = load_fragment(sdf_text)
        if error or mol is None:
            return None, f"fragment {index}: {error}"
        # Queries and coordinate maps both run over heavy atoms, and a fragment
        # written with explicit hydrogens would otherwise never match a candidate
        # parsed from SMILES.
        fragments.append(Chem.RemoveHs(mol))

    protein = parse_pocket_coords(pocket_pdb) if pocket_pdb else np.zeros((0, 3))

    return [
        verify_one(
            candidate,
            fragments,
            protein,
            num_seeds=num_seeds,
            strain_max=strain_max,
            rmsd_max=rmsd_max,
            clash_distance=clash_distance,
        )
        for candidate in smiles
    ], None
