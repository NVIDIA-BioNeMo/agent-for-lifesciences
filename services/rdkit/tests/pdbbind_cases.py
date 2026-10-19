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

"""Build Fragment Assembly cases from PDBBind, the way SMDD-Bench builds its own.

SMDD-Bench's Fragment Assembly tasks come from co-folded witness complexes:
BRICS-decompose the witness ligand, keep fragments of at least six heavy atoms,
hide everything else, and hand the agent the fragment poses plus the pocket. Every
such task therefore has a known answer -- the ligand it was cut from -- which is
what makes it possible to check exit-vector ranking, linker sizing and pose
recovery against ground truth rather than against plausibility.

PDBBind supplies the same construction from crystal structures. This module turns
one PDBBind entry directory into a :class:`LinkCase`, and is used two ways:

- ``tests/test_pdbbind_validation.py`` walks the whole set when the mount is
  present, and reports the V1-V6 criteria.
- ``python -m tests.pdbbind_cases <entry_dir> ...`` writes a case out as fixture
  files under ``tests/data/``, which is how the committed cases were made and how
  a new one is added.

**Cut exactly two bonds.** The construction that matters is the one that leaves
three pieces -- fragment, linker, fragment -- because then the hidden piece *is* a
linker. Splitting at every BRICS bond and keeping the two widest-separated pieces
looks like the same thing and is not: measured over 500 PDBBind entries it hides a
median 60% of the ligand, so the "answer" is a 25-atom substructure that no
verifier can be expected to reproduce, and a deliberately-wrong variant built by
bridging the two fragments with a chain comes out *smaller and floppier* than the
truth. Every specificity number measured that way is backwards. Cutting two bonds
instead hides a median 29% of the ligand as a 5-10 atom linker, and 15% of PDBBind
entries yield such a case.

Selection bounds, and why each one is not arbitrary:

``MIN_HEAVY_ATOMS``/``MIN_SEPARATION``
    SMDD's stated construction: fragments of at least six heavy atoms whose
    closest heavy atoms are at least 6 A apart.
``MAX_SEPARATION``/``MAX_TRUE_LINKER_ATOMS``/``MAX_LIGAND_MW``
    Implied by the rest of the benchmark rather than stated. A task is only
    solvable if the assembled molecule can pass Table 6, so a 25 A gap bridged by
    a coiled 29-atom peptide chain is not in the distribution -- and PDBBind is
    full of peptides and lipids that produce exactly that.
``MIN_COVERAGE``
    A backstop on the two-cut rule for the rare ligand whose linker is a tenth of
    it: below this the case is measuring generation, not linking.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Descriptors

RDLogger.DisableLog("rdApp.*")

MIN_HEAVY_ATOMS = 6
MIN_SEPARATION = 6.0
MAX_SEPARATION = 12.0
MAX_TRUE_LINKER_ATOMS = 10
MAX_LIGAND_MW = 600.0
MIN_COVERAGE = 0.55

DEFAULT_ROOT = "/pdbbind"


@dataclass
class LinkCase:
    """One linking task with a known answer."""

    pdb_id: str
    ligand_smiles: str
    ligand_sdf: str
    pocket_pdb: str
    fragment_sdfs: list[str]
    fragment_smiles: list[str]
    # Index inside each fragment SDF of the atom that carried the true linker.
    true_anchors: tuple[int, int]
    # Heavy atoms in the hidden piece, all of them. This is what a generated
    # molecule has to supply, and what a SAFE mask window has to admit.
    true_linker_atoms: int
    # Heavy atoms on the shortest bond path from one anchor to the other,
    # exclusive. Always at most the above, and less whenever the linker carries a
    # ring or a branch: a piperazine has six atoms but a path of four. This is the
    # count a span-to-atoms estimate is predicting, so it is the one V1 checks.
    true_path_atoms: int
    separation: float
    anchor_distance: float
    # Share of the ligand's heavy atoms the two fragments account for. What is left
    # is the hidden linker, so this says how much of the answer was given away.
    coverage: float = 1.0
    rejections: list[str] = field(default_factory=list)

    def metadata(self) -> dict:
        """Everything but the structures, for committing next to the fixtures."""
        return {
            "pdb_id": self.pdb_id,
            "ligand_smiles": self.ligand_smiles,
            "fragment_smiles": self.fragment_smiles,
            "true_anchors": list(self.true_anchors),
            "true_linker_atoms": self.true_linker_atoms,
            "true_path_atoms": self.true_path_atoms,
            "separation": round(self.separation, 2),
            "anchor_distance": round(self.anchor_distance, 2),
            "coverage": round(self.coverage, 3),
        }


def _splits(ligand: Chem.Mol) -> list[dict]:
    """Every fragment-linker-fragment split of a ligand at two BRICS bonds.

    A pair of BRICS bonds only gives a linking case when it leaves exactly three
    pieces and the middle one touches both cuts -- that is what makes it a linker
    rather than a branch. Pairs that cut a ring open, or that cut two branches off
    the same core, are skipped by those two conditions.
    """
    bonds = [pair for pair, _ in BRICS.FindBRICSBonds(ligand)]
    if len(bonds) < 2:
        return []
    positions = ligand.GetConformer().GetPositions()
    heavy_total = ligand.GetNumHeavyAtoms()

    found = []
    for first in range(len(bonds)):
        for second in range(first + 1, len(bonds)):
            cuts = [bonds[first], bonds[second]]
            bond_ids = [ligand.GetBondBetweenAtoms(a, b).GetIdx() for a, b in cuts]
            broken = Chem.FragmentOnBonds(ligand, bond_ids, addDummies=False)
            groups = Chem.GetMolFrags(broken)
            if len(groups) != 3:
                continue

            cut_atoms = {atom for cut in cuts for atom in cut}
            middles = [
                index for index, group in enumerate(groups) if len(cut_atoms & set(group)) == 2
            ]
            if len(middles) != 1:
                continue
            middle = middles[0]
            left, right = (index for index in range(3) if index != middle)

            heavy = [
                [atom for atom in group if ligand.GetAtomWithIdx(atom).GetAtomicNum() > 1]
                for group in groups
            ]
            if min(len(heavy[left]), len(heavy[right])) < MIN_HEAVY_ATOMS:
                continue
            if not 1 <= len(heavy[middle]) <= MAX_TRUE_LINKER_ATOMS:
                continue

            separation = float(
                np.min(
                    np.linalg.norm(
                        positions[heavy[left]][:, None] - positions[heavy[right]][None], axis=2
                    )
                )
            )
            if not MIN_SEPARATION <= separation <= MAX_SEPARATION:
                continue

            coverage = (len(heavy[left]) + len(heavy[right])) / heavy_total
            if coverage < MIN_COVERAGE:
                continue

            # The anchors are the atoms the linker was bonded to, read straight off
            # the cut bonds rather than inferred.
            anchors = {}
            for side in (left, right):
                for cut in cuts:
                    for atom in cut:
                        if atom in groups[side]:
                            anchors[side] = atom
            if len(anchors) != 2:
                continue

            path = Chem.GetDistanceMatrix(ligand)[anchors[left]][anchors[right]]
            found.append(
                {
                    "groups": groups,
                    "sides": (left, right),
                    "middle": middle,
                    "anchors": (anchors[left], anchors[right]),
                    "linker_atoms": len(heavy[middle]),
                    "path_atoms": int(path) - 1,
                    "separation": separation,
                    "coverage": coverage,
                    "bond_ids": bond_ids,
                }
            )
    return found


def build_case(entry_dir: str) -> tuple[Optional[LinkCase], Optional[str]]:
    """Decompose one PDBBind entry into a two-fragment linking case.

    Returns ``(None, reason)`` when the entry does not yield one, with the reason
    naming which bound rejected it so a survey can report the funnel. When a ligand
    admits several splits the widest is taken, because the long-linker regime is
    where SMDD's two-fragment instances sit and where a linker library stops being
    enough.
    """
    pdb_id = os.path.basename(entry_dir.rstrip("/"))
    ligand_path = os.path.join(entry_dir, f"{pdb_id}_ligand.sdf")
    pocket_path = os.path.join(entry_dir, f"{pdb_id}_pocket.pdb")
    if not os.path.exists(ligand_path) or not os.path.exists(pocket_path):
        return None, "missing ligand or pocket file"

    ligand = Chem.MolFromMolFile(ligand_path, removeHs=False)
    if ligand is None:
        return None, "unparseable ligand"
    if Descriptors.MolWt(Chem.RemoveHs(ligand)) > MAX_LIGAND_MW:
        return None, "ligand too heavy"

    splits = _splits(ligand)
    if not splits:
        return None, "no fragment-linker-fragment split within the bounds"
    split = max(splits, key=lambda item: (item["separation"], item["coverage"]))

    # addDummies=False keeps the pieces valence-complete after sanitisation, which
    # is how SMDD's fragments arrive: plain molecules with no attachment marks.
    broken = Chem.FragmentOnBonds(ligand, split["bond_ids"], addDummies=False)
    pieces = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=False)
    fragments = []
    for side in split["sides"]:
        piece = pieces[side]
        try:
            Chem.SanitizeMol(piece)
        except Exception as exc:  # noqa: BLE001 - a fragment RDKit rejects is not usable
            return None, f"fragment does not sanitise: {exc}"
        fragments.append(piece)

    positions = ligand.GetConformer().GetPositions()
    anchor_left, anchor_right = split["anchors"]
    local = [
        {atom: index for index, atom in enumerate(split["groups"][side])}
        for side in split["sides"]
    ]

    return (
        LinkCase(
            pdb_id=pdb_id,
            ligand_smiles=Chem.MolToSmiles(Chem.RemoveHs(ligand)),
            ligand_sdf=Chem.MolToMolBlock(ligand),
            pocket_pdb=open(pocket_path).read(),
            fragment_sdfs=[Chem.MolToMolBlock(fragment) for fragment in fragments],
            fragment_smiles=[
                Chem.MolToSmiles(Chem.RemoveHs(fragment)) for fragment in fragments
            ],
            true_anchors=(local[0][anchor_left], local[1][anchor_right]),
            true_linker_atoms=split["linker_atoms"],
            true_path_atoms=split["path_atoms"],
            separation=split["separation"],
            anchor_distance=float(
                np.linalg.norm(positions[anchor_left] - positions[anchor_right])
            ),
            coverage=split["coverage"],
        ),
        None,
    )


def iter_cases(
    root: str = DEFAULT_ROOT, limit: Optional[int] = None, seed: int = 7
) -> Iterator[LinkCase]:
    """Yield cases from a PDBBind tree, sampled deterministically.

    Sampling rather than walking in order because PDBBind's directories are
    grouped by deposition year, so the first N entries are not a sample of it.
    """
    import random

    entries = sorted(glob.glob(os.path.join(root, "P-L", "*", "*", "")))
    random.Random(seed).shuffle(entries)
    yielded = 0
    for entry in entries:
        case, _ = build_case(entry)
        if case is None:
            continue
        yield case
        yielded += 1
        if limit is not None and yielded >= limit:
            return


DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def write_fixture(case: LinkCase, data_dir: str = DATA_DIR) -> list[str]:
    """Write one case out as committed fixture files, returning their paths."""
    os.makedirs(data_dir, exist_ok=True)
    written = []
    files = {
        f"{case.pdb_id}_pocket.pdb": case.pocket_pdb,
        f"{case.pdb_id}_ligand.sdf": case.ligand_sdf,
        f"{case.pdb_id}_fragment_0.sdf": case.fragment_sdfs[0],
        f"{case.pdb_id}_fragment_1.sdf": case.fragment_sdfs[1],
        f"{case.pdb_id}_case.json": json.dumps(case.metadata(), indent=2) + "\n",
    }
    for name, content in files.items():
        path = os.path.join(data_dir, name)
        with open(path, "w") as handle:
            handle.write(content)
        written.append(path)
    return written


def load_fixture(pdb_id: str, data_dir: str = DATA_DIR) -> LinkCase:
    """Rebuild a case from its committed fixture files."""
    metadata = json.load(open(os.path.join(data_dir, f"{pdb_id}_case.json")))
    return LinkCase(
        pdb_id=pdb_id,
        ligand_smiles=metadata["ligand_smiles"],
        ligand_sdf=open(os.path.join(data_dir, f"{pdb_id}_ligand.sdf")).read(),
        pocket_pdb=open(os.path.join(data_dir, f"{pdb_id}_pocket.pdb")).read(),
        fragment_sdfs=[
            open(os.path.join(data_dir, f"{pdb_id}_fragment_{index}.sdf")).read()
            for index in (0, 1)
        ],
        fragment_smiles=metadata["fragment_smiles"],
        true_anchors=tuple(metadata["true_anchors"]),
        true_linker_atoms=metadata["true_linker_atoms"],
        true_path_atoms=metadata["true_path_atoms"],
        separation=metadata["separation"],
        anchor_distance=metadata["anchor_distance"],
        coverage=metadata.get("coverage", 1.0),
    )


def fixture_ids(data_dir: str = DATA_DIR) -> list[str]:
    """PDB ids of every committed fixture case."""
    return sorted(
        os.path.basename(path)[: -len("_case.json")]
        for path in glob.glob(os.path.join(data_dir, "*_case.json"))
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python -m tests.pdbbind_cases <pdbbind_entry_dir> ...")
        raise SystemExit(2)
    for argument in sys.argv[1:]:
        case, reason = build_case(argument)
        if case is None:
            print(f"{argument}: rejected ({reason})")
            continue
        paths = write_fixture(case)
        print(f"{case.pdb_id}: {case.fragment_smiles} d={case.anchor_distance:.2f} "
              f"n_true={case.true_linker_atoms} path={case.true_path_atoms} "
              f"coverage={case.coverage:.0%} -> {len(paths)} files")
