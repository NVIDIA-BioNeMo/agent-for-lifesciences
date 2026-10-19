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

"""A curated linker library, with spans and rotatable-bond costs measured here.

Fragment linking has two ways to propose a bridge. A generative model (GenMol,
DiffLinker) invents one; a library looks one up by geometry. DeLinker's own
evaluation calls the lookup a "database baseline" and beats it by 200% in the
long-linker regime, which is the regime SMDD-Bench's Fragment Assembly tasks
live in -- so the library is not the ambition here. It is the floor: it needs no
network, no GPU and no API key, it answers in milliseconds, and every candidate
it emits is a real molecule made of motifs a medicinal chemist would recognise.

Two numbers decide whether an entry can bridge a given pair, and both are
**measured rather than asserted**:

- ``span_min``/``span_max``: the anchor-to-anchor distance the linker can hold,
  taken from ETKDG conformers of the linker capped with a benzene ring at each
  end, measured between the two ring carbons that carry it. Anchor-to-anchor is
  exactly what :mod:`fragment_link` computes from the input poses, so the
  comparison is apples to apples.
- ``rb_cost``: rotatable bonds contributed, from ``CalcNumRotatableBonds`` on
  that same capped molecule. Phenyl caps rather than methyl caps because RDKit's
  definition excludes bonds to terminal atoms; a methyl-capped chain would
  under-report its own attachment bonds by two, which is precisely the cost that
  matters against SMDD-Bench's ``RB <= 10``.

Measurement is cached per process. It costs a few hundred embeddings once.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors

logger = logging.getLogger(__name__)

# Conformers per entry for the span measurement. A rigid aryl bridge is settled
# by a handful; a butyl chain needs enough to find both its extended and its
# coiled extremes, and the reported range is only as honest as that sampling.
SPAN_CONFORMERS = 24
RIGID_SPAN_CONFORMERS = 4
SPAN_SEED = 0xF00D

# Both ends of a library linker are dummy atoms carrying these map numbers, which
# is what makes Chem.molzip able to join fragment - linker - fragment in one call.
ATTACH_A = 1
ATTACH_B = 2


@dataclass(frozen=True)
class Linker:
    """One curated bridge, with its geometry measured on first use."""

    name: str
    smiles: str
    family: str
    rigid: bool
    note: str

    @property
    def num_linker_atoms(self) -> int:
        """Heavy atoms the linker itself contributes, dummies excluded."""
        mol = Chem.MolFromSmiles(self.smiles)
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)

    @property
    def is_direct_bond(self) -> bool:
        """True for the zero-atom entry, which joins the fragments to each other.

        It needs its own path everywhere: ``molzip`` refuses a linker whose two
        ends are the same bond, so a direct join is done by labelling both
        fragments with the same attachment number instead.
        """
        return self.num_linker_atoms == 0


# Ordered roughly by span. `family` groups entries so a caller can ask for one
# representative of each shape rather than five variations on an amide, and
# `rigid` marks the entries whose span survives contact with a real force field
# -- the ones worth reaching for when the rotatable-bond budget is tight.
LINKER_LIBRARY: tuple[Linker, ...] = (
    Linker(
        name="direct_bond",
        smiles="[*:1][*:2]",
        family="bond",
        rigid=True,
        note="Fragments bonded directly; only reachable for anchors under ~2 A apart",
    ),
    Linker(
        name="methylene",
        smiles="[*:1]C[*:2]",
        family="chain",
        rigid=False,
        note="Shortest flexible bridge",
    ),
    Linker(
        name="ether",
        smiles="[*:1]O[*:2]",
        family="chain",
        rigid=False,
        note="One-atom bridge with an acceptor; shortens the span versus CH2",
    ),
    Linker(
        name="ethylene",
        smiles="[*:1]CC[*:2]",
        family="chain",
        rigid=False,
        note="Two-carbon chain",
    ),
    Linker(
        name="alkyne",
        smiles="[*:1]C#C[*:2]",
        family="rigid_rod",
        rigid=True,
        note="Linear and rigid; spends no rotatable bonds on the span itself",
    ),
    Linker(
        name="amide",
        smiles="[*:1]C(=O)N[*:2]",
        family="amide",
        rigid=True,
        note="Planar, directional, donor plus acceptor; the workhorse bridge",
    ),
    Linker(
        name="reverse_amide",
        smiles="[*:1]NC(=O)[*:2]",
        family="amide",
        rigid=True,
        note="Amide with the carbonyl pointing the other way",
    ),
    Linker(
        name="urea",
        smiles="[*:1]NC(=O)N[*:2]",
        family="amide",
        rigid=True,
        note="Two donors and an acceptor, planar",
    ),
    Linker(
        name="ester",
        smiles="[*:1]C(=O)O[*:2]",
        family="amide",
        rigid=True,
        note="Planar but metabolically labile",
    ),
    Linker(
        name="sulfonamide",
        smiles="[*:1]S(=O)(=O)N[*:2]",
        family="sulfonamide",
        rigid=False,
        note="Tetrahedral sulfur, strongly polar, adds TPSA",
    ),
    Linker(
        name="propylene",
        smiles="[*:1]CCC[*:2]",
        family="chain",
        rigid=False,
        note="Three-carbon chain",
    ),
    Linker(
        name="butylene",
        smiles="[*:1]CCCC[*:2]",
        family="chain",
        rigid=False,
        note="Four-carbon chain; four rotatable bonds is most of a tight budget",
    ),
    Linker(
        name="peg2",
        smiles="[*:1]OCCO[*:2]",
        family="chain",
        rigid=False,
        note="Flexible and solubilising; expensive in rotatable bonds",
    ),
    Linker(
        name="triazole",
        smiles="[*:1]c1cn([*:2])nn1",
        family="heteroaryl",
        rigid=True,
        note="Click chemistry product; rigid, planar, three acceptors",
    ),
    Linker(
        name="oxadiazole",
        smiles="[*:1]c1nnc([*:2])o1",
        family="heteroaryl",
        rigid=True,
        note="Rigid amide bioisostere",
    ),
    Linker(
        name="phenyl_1_3",
        smiles="[*:1]c1cccc([*:2])c1",
        family="aryl",
        rigid=True,
        note="Meta-phenylene; bends the span by about 60 degrees",
    ),
    Linker(
        name="phenyl_1_4",
        smiles="[*:1]c1ccc([*:2])cc1",
        family="aryl",
        rigid=True,
        note="Para-phenylene; the canonical rigid span",
    ),
    Linker(
        name="pyridine_2_5",
        smiles="[*:1]c1ccc([*:2])nc1",
        family="heteroaryl",
        rigid=True,
        note="Para-phenylene span with an acceptor nitrogen",
    ),
    Linker(
        name="pyrimidine_2_5",
        smiles="[*:1]c1cnc([*:2])nc1",
        family="heteroaryl",
        rigid=True,
        note="Two acceptors, rigid; lowers logP versus phenyl",
    ),
    Linker(
        name="piperazine_1_4",
        smiles="[*:1]N1CCN([*:2])CC1",
        family="saturated_ring",
        rigid=True,
        note="Chair-locked span with a basic nitrogen; common solubilising bridge",
    ),
    Linker(
        name="piperidine_1_4",
        smiles="[*:1]N1CCC([*:2])CC1",
        family="saturated_ring",
        rigid=True,
        note="One basic nitrogen, sp3 exit at the 4-position",
    ),
    Linker(
        name="cyclohexane_1_4",
        smiles="[*:1]C1CCC([*:2])CC1",
        family="saturated_ring",
        rigid=True,
        note="Aliphatic phenyl bioisostere, trans-diequatorial span",
    ),
    Linker(
        name="bicyclopentane_1_3",
        smiles="[*:1]C12CC([*:2])(C1)C2",
        family="saturated_ring",
        rigid=True,
        note="Bicyclo[1.1.1]pentane; para-phenyl span with no aromatic ring",
    ),
    Linker(
        name="amide_phenyl",
        smiles="[*:1]C(=O)Nc1ccc([*:2])cc1",
        family="composite",
        rigid=True,
        note="Anilide; long, rigid, directional",
    ),
    Linker(
        name="phenyl_amide",
        smiles="[*:1]c1ccc(C(=O)N[*:2])cc1",
        family="composite",
        rigid=True,
        note="Anilide with the ring on the other side",
    ),
    Linker(
        name="phenyl_ether",
        smiles="[*:1]c1ccc(O[*:2])cc1",
        family="composite",
        rigid=False,
        note="Aryl ether; one rotatable bond buys a long span",
    ),
    Linker(
        name="alkynyl_phenyl",
        smiles="[*:1]C#Cc1ccc([*:2])cc1",
        family="rigid_rod",
        rigid=True,
        note="Rigid rod near 8 A; for spans no chain can reach inside the RB budget",
    ),
    Linker(
        name="phenyl_methylene",
        smiles="[*:1]c1ccc(C[*:2])cc1",
        family="composite",
        rigid=False,
        note="Benzyl bridge; one flexible joint at the far end",
    ),
    Linker(
        name="phenyl_piperazine",
        smiles="[*:1]c1ccc(N2CCN([*:2])CC2)cc1",
        family="composite",
        rigid=True,
        note="Aryl-piperazine; long, soluble, only one rotatable joint",
    ),
    Linker(
        name="alkynyl_bicyclopentane",
        smiles="[*:1]C#CC12CC([*:2])(C1)C2",
        family="rigid_rod",
        rigid=True,
        note="Alkyne plus a bicyclopentane; rigid, aliphatic, near 7.5 A",
    ),
    Linker(
        name="phenyl_bicyclopentane",
        smiles="[*:1]c1ccc(C23CC([*:2])(C2)C3)cc1",
        family="rigid_rod",
        rigid=True,
        note="Fills the 9-10 A band that no chain can reach affordably",
    ),
    Linker(
        name="biphenyl",
        smiles="[*:1]c1ccc(-c2ccc([*:2])cc2)cc1",
        family="rigid_rod",
        rigid=True,
        note="Rigid rod near 10 A; heavy, and pushes MW and logP",
    ),
    Linker(
        name="bis_alkynyl_phenyl",
        smiles="[*:1]C#Cc1ccc(C#C[*:2])cc1",
        family="rigid_rod",
        rigid=True,
        note="Longest rigid entry, near 11 A, at zero rotatable-bond cost",
    ),
    Linker(
        name="phenyl_amide_phenyl",
        smiles="[*:1]c1ccc(C(=O)Nc2ccc([*:2])cc2)cc1",
        family="composite",
        rigid=True,
        note="Twelve-atom directional span; check molecular weight before using",
    ),
)

LINKERS_BY_NAME: dict[str, Linker] = {entry.name: entry for entry in LINKER_LIBRARY}


@dataclass(frozen=True)
class LinkerGeometry:
    """Measured properties of one library entry."""

    name: str
    span_min: float
    span_max: float
    rb_cost: int
    num_linker_atoms: int
    # Atoms on the shortest path between the two anchors. Lower than
    # num_linker_atoms for anything containing a ring: a para-phenylene carries
    # six atoms but puts four of them on the path, which is why an atom-count
    # window derived from a distance has to be scaled before it becomes a
    # generation mask bound.
    path_atoms: int
    num_conformers: int

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "span_min": round(self.span_min, 2),
            "span_max": round(self.span_max, 2),
            "rb_cost": self.rb_cost,
            "num_linker_atoms": self.num_linker_atoms,
            "path_atoms": self.path_atoms,
        }


_geometry_cache: dict[str, LinkerGeometry] = {}


def _phenyl_capped(linker_smiles: str) -> tuple[Optional[Chem.Mol], tuple[int, int]]:
    """Join benzene rings to both ends of a linker, returning the anchor indices.

    The anchors are the ring carbons that carry the linker: measuring between
    them gives the anchor-to-anchor distance that :mod:`fragment_link` derives
    from the input fragment poses.
    """
    linker = Chem.MolFromSmiles(linker_smiles)
    if linker is None:
        return None, (-1, -1)

    # A zero-atom linker means the caps bond to each other, so both carry the
    # same attachment number and the linker itself drops out of the join.
    direct = sum(1 for atom in linker.GetAtoms() if atom.GetAtomicNum() > 1) == 0
    cap_a = Chem.MolFromSmiles(f"c1ccccc1[*:{ATTACH_A}]")
    cap_b = Chem.MolFromSmiles(f"c1ccccc1[*:{ATTACH_A if direct else ATTACH_B}]")
    if cap_a is None or cap_b is None:
        return None, (-1, -1)

    # Marking the cap attachment carbons before the zip is the only reliable way
    # to find them afterwards: molzip renumbers.
    for cap, tag in ((cap_a, "capA"), (cap_b, "capB")):
        for atom in cap.GetAtoms():
            if atom.GetAtomicNum() == 0:
                neighbor = atom.GetNeighbors()[0]
                neighbor.SetProp("_link_cap", tag)

    combined = Chem.CombineMols(cap_a, cap_b) if direct else Chem.CombineMols(
        Chem.CombineMols(cap_a, linker), cap_b
    )
    try:
        joined = Chem.molzip(combined)
        Chem.SanitizeMol(joined)
    except Exception as exc:  # noqa: BLE001 - a bad library entry must not 500
        logger.warning(f"Could not cap linker {linker_smiles}: {exc}")
        return None, (-1, -1)

    anchors = {}
    for atom in joined.GetAtoms():
        if atom.HasProp("_link_cap"):
            anchors[atom.GetProp("_link_cap")] = atom.GetIdx()
    if "capA" not in anchors or "capB" not in anchors:
        return None, (-1, -1)

    return joined, (anchors["capA"], anchors["capB"])


def measure(entry: Linker) -> Optional[LinkerGeometry]:
    """Measure one entry's span range and rotatable-bond cost, with caching."""
    if entry.name in _geometry_cache:
        return _geometry_cache[entry.name]

    capped, (anchor_a, anchor_b) = _phenyl_capped(entry.smiles)
    if capped is None:
        return None

    rb_cost = rdMolDescriptors.CalcNumRotatableBonds(capped)

    mol = Chem.AddHs(capped)
    params = AllChem.ETKDGv3()
    params.randomSeed = SPAN_SEED
    params.pruneRmsThresh = 0.5
    num_conformers = RIGID_SPAN_CONFORMERS if entry.rigid else SPAN_CONFORMERS
    conformer_ids = AllChem.EmbedMultipleConfs(mol, numConfs=num_conformers, params=params)
    if not conformer_ids:
        logger.warning(f"Could not embed capped linker {entry.name}")
        return None

    # Minimising matters for the flexible entries: raw ETKDG output includes
    # strained extremes that would widen the reported span past what the linker
    # can actually hold.
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)

    spans = []
    for conformer_id in conformer_ids:
        positions = mol.GetConformer(conformer_id).GetPositions()
        spans.append(float(np.linalg.norm(positions[anchor_a] - positions[anchor_b])))

    geometry = LinkerGeometry(
        name=entry.name,
        span_min=min(spans),
        span_max=max(spans),
        rb_cost=rb_cost,
        num_linker_atoms=entry.num_linker_atoms,
        path_atoms=max(len(Chem.GetShortestPath(capped, anchor_a, anchor_b)) - 2, 0),
        num_conformers=len(spans),
    )
    _geometry_cache[entry.name] = geometry
    return geometry


def measure_all() -> dict[str, LinkerGeometry]:
    """Measure every entry, skipping any that cannot be built or embedded."""
    return {
        entry.name: geometry
        for entry in LINKER_LIBRARY
        if (geometry := measure(entry)) is not None
    }


def matching_linkers(
    anchor_distance: float,
    rb_headroom: Optional[int] = None,
    tolerance: float = 0.75,
    rigid_only: bool = False,
    max_results: int = 8,
    one_per_family: bool = True,
) -> list[LinkerGeometry]:
    """Library entries that can bridge ``anchor_distance``, cheapest bridge first.

    ``tolerance`` widens the measured span window on both sides: the anchor
    distance comes from a pose the linker has to accommodate rather than dictate,
    and the constrained-embedding check downstream is the real arbiter of whether
    a near miss works. Ranking is by rotatable-bond cost and then by size, since
    the rotatable-bond gate is the one that fails candidates here and every heavy
    atom spends molecular-weight budget.
    """
    matches = []
    for entry in LINKER_LIBRARY:
        if rigid_only and not entry.rigid:
            continue
        geometry = measure(entry)
        if geometry is None:
            continue
        if rb_headroom is not None and geometry.rb_cost > rb_headroom:
            continue
        if geometry.span_min - tolerance <= anchor_distance <= geometry.span_max + tolerance:
            matches.append((entry, geometry))

    matches.sort(key=lambda pair: (pair[1].rb_cost, pair[1].num_linker_atoms, pair[0].name))

    selected: list[LinkerGeometry] = []
    seen_families: set[str] = set()
    for entry, geometry in matches:
        if one_per_family and entry.family in seen_families:
            continue
        seen_families.add(entry.family)
        selected.append(geometry)
        if len(selected) >= max_results:
            break
    return selected


def assemble(
    mol_a: Chem.Mol,
    site_a: int,
    mol_b: Optional[Chem.Mol],
    site_b: Optional[int],
    linker_smiles: str,
) -> tuple[Optional[str], Optional[str]]:
    """Bond one or two fragments through a linker and return the product SMILES.

    Attachment is by :func:`rdkit.Chem.molzip` over dummy atoms carrying map
    numbers, which is the only join that keeps aromaticity and stereochemistry
    intact without hand-editing bonds. Fragment coordinates are dropped: the
    product is a topology, and giving it a pose is
    :mod:`constrained_pose`'s job.

    Args:
        mol_a: First fragment.
        site_a: Index in ``mol_a`` of the atom that loses a hydrogen.
        mol_b: Second fragment, or None for a one-fragment growth.
        site_b: Index in ``mol_b`` of the atom that loses a hydrogen.
        linker_smiles: Library entry, with ``[*:1]`` and (for two fragments)
            ``[*:2]`` marking its ends.

    Returns:
        Tuple of ``(smiles, error_message)``.
    """
    linker = Chem.MolFromSmiles(linker_smiles)
    if linker is None:
        return None, f"unparseable linker SMILES: {linker_smiles}"
    direct = sum(1 for atom in linker.GetAtoms() if atom.GetAtomicNum() > 1) == 0

    marked_a, error = mark_attachment(mol_a, site_a, ATTACH_A)
    if error:
        return None, f"fragment A: {error}"

    pieces = [marked_a] if direct else [marked_a, linker]

    if mol_b is not None and site_b is not None:
        marked_b, error = mark_attachment(
            mol_b, site_b, ATTACH_A if direct else ATTACH_B
        )
        if error:
            return None, f"fragment B: {error}"
        pieces.append(marked_b)
    elif direct:
        return None, "a direct bond needs two fragments"

    combined = pieces[0]
    for piece in pieces[1:]:
        combined = Chem.CombineMols(combined, piece)

    try:
        joined = Chem.molzip(combined)
        Chem.SanitizeMol(joined)
    except Exception as exc:  # noqa: BLE001 - report, do not raise, per candidate
        return None, f"could not join fragments through {linker_smiles}: {exc}"

    # An unconsumed dummy means the linker had an end no fragment claimed, which
    # would leave a reactive attachment point in a molecule we are about to score.
    if any(atom.GetAtomicNum() == 0 for atom in joined.GetAtoms()):
        return None, f"unpaired attachment point remains after joining {linker_smiles}"

    joined = Chem.RemoveHs(joined)
    return Chem.MolToSmiles(joined), None


def mark_attachment(
    mol: Chem.Mol, site: int, map_number: int
) -> tuple[Optional[Chem.Mol], Optional[str]]:
    """Replace one hydrogen on ``site`` with a mapped dummy atom.

    Works from a hydrogen when the input carries explicit ones and from the
    implicit count otherwise, so a fragment written either way behaves the same.
    The returned molecule has no coordinates and no explicit hydrogens.
    """
    if site < 0 or site >= mol.GetNumAtoms():
        return None, f"atom index {site} is out of range for a {mol.GetNumAtoms()}-atom fragment"

    atom = mol.GetAtomWithIdx(site)
    if atom.GetAtomicNum() <= 1:
        return None, f"atom {site} is a hydrogen, not a substitutable heavy atom"

    editable = Chem.RWMol(mol)
    hydrogens = [
        neighbor.GetIdx()
        for neighbor in editable.GetAtomWithIdx(site).GetNeighbors()
        if neighbor.GetAtomicNum() == 1
    ]

    if hydrogens:
        # Turning an existing hydrogen into the dummy keeps the valence exactly
        # as it was, which is why this path is preferred when one is available.
        dummy = editable.GetAtomWithIdx(hydrogens[0])
        dummy.SetAtomicNum(0)
        dummy.SetAtomMapNum(map_number)
        dummy.SetNoImplicit(True)
    else:
        if atom.GetTotalNumHs(includeNeighbors=True) < 1:
            return None, f"atom {site} has no hydrogen to substitute"
        new_atom = Chem.Atom(0)
        new_atom.SetAtomMapNum(map_number)
        new_index = editable.AddAtom(new_atom)
        editable.AddBond(site, new_index, Chem.BondType.SINGLE)
        target = editable.GetAtomWithIdx(site)
        if target.GetNumExplicitHs() > 0:
            target.SetNumExplicitHs(target.GetNumExplicitHs() - 1)

    result = editable.GetMol()
    try:
        Chem.SanitizeMol(result)
        result = Chem.RemoveHs(result)
        Chem.SanitizeMol(result)
    except Exception as exc:  # noqa: BLE001
        return None, f"substituting atom {site} left an invalid molecule: {exc}"

    result.RemoveAllConformers()
    return result, None
