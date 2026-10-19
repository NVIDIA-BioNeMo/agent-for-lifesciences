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

"""Aevom client for sandboxed Python execution.

This module provides the `aevom` client that is pre-loaded into the sandbox
execution environment, allowing Python code to load stored data artifacts
(PDB files, SDF files, SMILES, selections, etc.) by reference.

================================================================================
OVERVIEW
================================================================================

The aevom client is the bridge between your Python code and Aevom's data store.
It allows you to:

1. LOAD data by reference (e.g., aevom.load("pdb_abc123"))
2. STORE new data and get back a reference (e.g., aevom.store_smiles("CCO"))
3. VISUALIZE molecules in the MolStar 3D viewer
4. CONVERT between formats using RDKit helpers

================================================================================
REFERENCE SYSTEM
================================================================================

Aevom uses a reference-based data storage pattern. Instead of passing large
molecular data directly, you work with short reference IDs like:

    - "pdb_abc123"     -> PDB protein structure
    - "sdf_xyz789"     -> SDF 3D molecule
    - "smiles_def456"  -> SMILES string
    - "sequence_ghi"   -> Protein sequence
    - "selection_jkl"  -> Atom selection from viewer
    - "array_mno345"   -> Numeric array (PAE / pLDDT matrix, distances, ...)

This keeps data out of the LLM context and enables efficient tool chaining.

Numeric outputs that tools cannot summarize -- confidence matrices, per-residue
scores -- are stored as "array_*" references specifically so you can analyze them
here with numpy instead of the model trying to read them as text.

================================================================================
QUICK REFERENCE - COMMONLY USED METHODS
================================================================================

Loading Data:
    aevom.load(ref)              -> Auto-detect type and return data
    aevom.load_pdb(ref)          -> Load PDB as string
    aevom.load_sdf(ref)          -> Load SDF as string
    aevom.load_smiles(ref)       -> Load SMILES as string
    aevom.load_coords(ref)       -> Load coordinates as numpy array
    aevom.load_array(ref)        -> Load numeric array as numpy array
    aevom.load_rdkit_mol(ref)    -> Load SDF as RDKit Mol object

Storing Data:
    aevom.store_smiles(smiles)   -> Store SMILES, returns reference
    aevom.store_sdf(sdf)         -> Store SDF, returns reference
    aevom.store_pdb(pdb)         -> Store PDB, returns reference
    aevom.store_array(arr)       -> Store numeric array, returns reference

Visualization:
    aevom.visualize(mol_or_sdf)  -> Display in 3D viewer
    aevom.visualize_pdb(pdb)     -> Display protein in 3D viewer

RDKit Helpers:
    aevom.mol_from_smiles(smi)   -> Create RDKit Mol from SMILES
    aevom.mol_from_sdf(sdf)      -> Create RDKit Mol from SDF
    aevom.embed_mol_3d(mol)      -> Add 3D coordinates to mol
    aevom.canonical_smiles(mol)  -> Get canonical SMILES from mol
    aevom.sa_score(mol_or_smi)   -> Synthetic accessibility, 1 (easy) to 10

================================================================================
EXAMPLE WORKFLOW
================================================================================

    # 1. Store a SMILES from PubChem lookup
    import pubchempy as pcp
    compound = pcp.get_compounds("aspirin", "name")[0]
    smiles_ref = aevom.store_smiles(compound.canonical_smiles,
                                     description="Aspirin from PubChem")
    print(f"Stored as: {smiles_ref}")

    # 2. The LLM can now call generate_3d_coordinates(smiles_ref=smiles_ref)
    #    which returns an sdf_ref

    # 3. Load and analyze the 3D structure
    mol = aevom.load_rdkit_mol("sdf_xyz789")
    from rdkit.Chem import Descriptors
    print(f"Molecular weight: {Descriptors.MolWt(mol)}")

    # 4. Visualize in MolStar
    aevom.visualize(mol, label="Aspirin")

================================================================================
"""

import base64
import json
import re
import os
import sys
import urllib.request
import urllib.error
from typing import Any, Optional, Union

_sascorer: Any = None
_sascorer_loaded = False


# Hydrogen counts written as if they were atoms. SMILES infers hydrogen, so `NH`
# is nitrogen followed by hydrogen-the-element and `CH3` is carbon followed by
# three hydrogens -- neither of which is what anyone means. Measured across 2759
# molecules an agent wrote by hand into this sandbox, this was the single largest
# cause of an unparseable string: 137 of 318 failures.
# Hydrogen is not in the organic subset: outside square brackets there is no
# valid SMILES in which a bare H appears, so finding one is proof rather than
# suspicion. Reporting the atom in front of it turns the message from a rule
# into the reader's own mistake -- "NH is not an atom" rather than "H is not
# allowed here".
_ELEMENTS_BEFORE_H = ("Br", "Cl", "N", "O", "S", "P", "C", "F", "I", "B")

# Outside square brackets SMILES admits only the organic subset -- B C N O P S
# F Cl Br I and their aromatic lower-case forms -- so the letters that can
# legally appear there are a short, closed list. Inside brackets any element
# symbol goes, which is why this is only ever applied outside them.
_ORGANIC_SUBSET_CHARS = set("BCNOPSFIlrbcnops")
_STRUCTURAL_CHARS = set("0123456789()=#-+\\/@.%*$:~[]")

# Group formulas, which read correctly to a chemist and are not SMILES.
_GROUP_FORMULAS = {
    "CF3": "C(F)(F)F -- and FC(F)(F)c1... when attached, not C(F)(F)Fc1...",
    "OMe": "OC",
    "OEt": "OCC",
    "NMe2": "N(C)C",
    "Me": "C",
    "Et": "CC",
    "Ph": "c1ccccc1",
    "Ac": "C(C)=O",
    "CCl3": "C(Cl)(Cl)Cl",
    "NO2": "[N+](=O)[O-]",
    "SO2": "S(=O)=O",
    "SO3H": "S(=O)(=O)O",
    "COOH": "C(=O)O",
    "CONH2": "C(N)=O",
    "OCF3": "OC(F)(F)F",
}


def _bare_hydrogen(smiles: str) -> Optional[tuple[str, str]]:
    """The first H outside square brackets, with whatever element precedes it."""
    depth = 0
    for index, character in enumerate(smiles):
        if character == "[":
            depth += 1
        elif character == "]":
            depth = max(0, depth - 1)
        elif character == "H" and depth == 0:
            preceding = ""
            for element in _ELEMENTS_BEFORE_H:
                if smiles[:index].endswith(element):
                    preceding = element
                    break
            trailing = smiles[index + 1: index + 2]
            token = preceding + "H" + (trailing if trailing.isdigit() else "")
            return token, preceding
    return None


def _unbalanced_ring_digit(smiles: str) -> Optional[str]:
    """The first ring-closure digit opened an odd number of times, if any."""
    counts: dict[str, int] = {}
    index = 0
    while index < len(smiles):
        character = smiles[index]
        if character == "[":
            closing = smiles.find("]", index)
            index = len(smiles) if closing < 0 else closing + 1
            continue
        if character == "%" and smiles[index + 1: index + 3].isdigit():
            counts[smiles[index: index + 3]] = counts.get(smiles[index: index + 3], 0) + 1
            index += 3
            continue
        if character.isdigit():
            counts[character] = counts.get(character, 0) + 1
        index += 1
    for digit, count in counts.items():
        if count % 2:
            return digit
    return None


# Groups that do not survive contact with water, and so are not substituents any
# lead-optimisation campaign can propose. This list exists because the enumerator
# below makes them by accident: a one-cut swap is blind to what it is attaching
# to, so cutting the O-CH3 bond of an anisole and re-attaching "F" from the
# default substituent list yields Ar-O-F, an aryl hypofluorite. Measured over
# four ordinary motifs -- anisole, thioanisole, phenol, a cyclopropylamide --
# 31 of 269 enumerated analogues (12%) carried one of these, and PAINS, Brenk
# and NIH alert catalogues pass every one.
#
# Deliberately short. Hydrazides, hydroxylamines and cyclic endoperoxides are
# all real drug chemistry (isoniazid, artemisinin), so they are not here; the
# cost of refusing a molecule someone wanted is higher than the cost of letting
# an unusual one through. What is here cannot be isolated at all.
UNSTABLE_GROUPS = (
    ("hypohalite (O-halogen)", "[OX2]-[F,Cl,Br,I]"),
    ("sulfenyl halide (S-halogen)", "[SX2]-[F,Cl,Br,I]"),
    ("N-halamine (N-halogen)", "[NX3]-[F,Cl,Br,I]"),
    ("acyl halide", "[CX3](=O)-[F,Cl,Br,I]"),
    ("acyclic peroxide", "[OX2R0]-[OX2R0]"),
    ("anhydride", "[CX3](=O)-[OX2]-[CX3]=O"),
)

_UNSTABLE_PATTERNS: list = []


def _unstable_patterns() -> list:
    """Compile the alert SMARTS once, on first use."""
    global _UNSTABLE_PATTERNS
    if not _UNSTABLE_PATTERNS:
        from rdkit import Chem

        _UNSTABLE_PATTERNS = [
            (name, Chem.MolFromSmarts(smarts)) for name, smarts in UNSTABLE_GROUPS
        ]
    return _UNSTABLE_PATTERNS


def unstable_group(mol_or_smiles: Any) -> Optional[str]:
    """Name the non-isolable group in a molecule, or None if there is none.

    Args:
        mol_or_smiles: An RDKit Mol or a SMILES string.

    Returns:
        The name of the first group from :data:`UNSTABLE_GROUPS` found, or None.

    Example:
        aevom.unstable_group("FOc1ccccc1")   # 'hypohalite (O-halogen)'
        aevom.unstable_group("COc1ccccc1")   # None
    """
    from rdkit import Chem

    mol = (
        Chem.MolFromSmiles(mol_or_smiles)
        if isinstance(mol_or_smiles, str)
        else mol_or_smiles
    )
    if mol is None:
        return None
    for name, pattern in _unstable_patterns():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return name
    return None


# What a molecule is made of, named. The order matters: the first pattern that
# matches an atom set wins, so the specific forms come before the general ones
# (a sulfonamide before an amine, a carboxylic acid before a hydroxyl).
FUNCTIONAL_GROUPS = (
    # Groups that cannot be isolated, first, because naming them is the point.
    ("aryl hypohalite (O-halogen)", "[OX2]-[F,Cl,Br,I]"),
    ("sulfenyl halide (S-halogen)", "[SX2]-[F,Cl,Br,I]"),
    ("N-halamine (N-halogen)", "[NX3]-[F,Cl,Br,I]"),
    ("acyl halide", "[CX3](=O)-[F,Cl,Br,I]"),
    ("acyclic peroxide", "[OX2R0]-[OX2R0]"),
    ("anhydride", "[CX3](=O)-[OX2]-[CX3]=O"),
    ("gem-halohydrin", "[OX2H]-[CX4]([F,Cl,Br,I])"),
    ("alpha-halo amine", "[NX3;!$(N[CX3]=O)]-[CX4]-[F,Cl,Br,I]"),
    ("hemiaminal", "[NX3]-[CX4;H1,H2]-[OX2H]"),
    ("hemiacetal", "[OX2]-[CX4;H1]-[OX2H]"),
    # Ordinary medicinal chemistry.
    ("sulfonamide", "[SX4](=O)(=O)[NX3]"),
    ("sulfone", "[SX4](=O)(=O)[#6]"),
    ("sulfoxide", "[SX3](=O)[#6]"),
    ("carboxylic acid", "[CX3](=O)[OX2H1]"),
    ("ester", "[CX3](=O)[OX2H0][#6]"),
    ("carbamate", "[NX3][CX3](=O)[OX2H0]"),
    ("urea", "[NX3][CX3](=O)[NX3]"),
    ("amide", "[NX3][CX3](=O)[#6]"),
    ("aldehyde", "[CX3H1](=O)[#6]"),
    ("ketone", "[#6][CX3](=O)[#6]"),
    ("nitrile", "[NX1]#[CX2]"),
    ("nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]"),
    ("guanidine", "[NX3][CX3](=[NX2])[NX3]"),
    ("amidine", "[NX3][CX3]=[NX2]"),
    ("aniline", "[NX3;H2,H1,H0][c]"),
    ("phenol", "[OX2H][c]"),
    ("alcohol", "[OX2H][CX4]"),
    ("aryl ether", "[OX2]([#6])[c]"),
    ("ether", "[OX2]([CX4])[CX4]"),
    ("thioether", "[SX2]([#6])[#6]"),
    ("thiol", "[SX2H]"),
    ("tertiary amine", "[NX3;H0;!$(N[CX3]=[O,N,S]);!$(N[a])]"),
    ("secondary amine", "[NX3;H1;!$(N[CX3]=[O,N,S]);!$(N[a])]"),
    ("primary amine", "[NX3;H2;!$(N[CX3]=[O,N,S]);!$(N[a])]"),
    # Named, not refused: these are isolable, and some are drugs. But the model
    # interpolates CH3 -> CH2F -> CHF2 -> CF3 as if all four were equally inert,
    # and the monofluorides at a benzylic or alpha-heteroatom position are
    # alkylating agents rather than metabolic blocks. 151 such structures were
    # typed by hand across 72 of 120 benchmark runs, and prompt guidance did not
    # change the rate.
    ("benzylic halide", "[c][CX4;H1,H2][F,Cl,Br,I]"),
    ("alpha-halo ether", "[OX2][CX4;H1,H2][F,Cl,Br,I]"),
    ("trifluoromethyl", "[CX4](F)(F)F"),
    ("aryl fluoride", "[c][F]"),
    ("aryl chloride", "[c][Cl]"),
    ("aryl bromide", "[c][Br]"),
    ("alkyl halide", "[CX4][F,Cl,Br,I]"),
    ("alkyne", "[CX2]#[CX2]"),
    ("alkene", "[CX3]=[CX3]"),
)

_GROUP_PATTERNS: list = []


def _group_patterns() -> list:
    global _GROUP_PATTERNS
    if not _GROUP_PATTERNS:
        from rdkit import Chem

        _GROUP_PATTERNS = [
            (name, Chem.MolFromSmarts(smarts)) for name, smarts in FUNCTIONAL_GROUPS
        ]
    return _GROUP_PATTERNS


def describe_molecule(mol_or_smiles: Any) -> dict:
    """Say what a molecule is actually made of, so it need not be read by eye.

    A SMILES string is not legible. ``OC(F)(F)c1ccccc1`` and
    ``C(F)(F)Oc1ccccc1`` differ by which atom carries the ring, and in traces of
    real runs an agent wrote the first while describing the second -- shipping a
    gem-fluorohydrin it believed was a difluoromethyl ether. Both parse, both
    clear every drug-likeness gate, and nothing downstream disagrees. This
    returns the graph's own account of itself: formula, rings, and the groups
    present by name, with the non-isolable ones named first.

    Args:
        mol_or_smiles: An RDKit Mol or a SMILES string.

    Returns:
        ``{"formula", "heavy_atoms", "rings", "groups", "unstable"}`` where
        ``rings`` describes each ring system and ``groups`` is a list of names.

    Raises:
        AevomError: If the input is not a valid molecule.

    Example:
        d = aevom.describe_molecule("OC(F)(F)c1ccccc1")
        print(d["groups"])     # ['gem-halohydrin', 'alcohol', ...]
        print(d["unstable"])   # None here -- it is odd, not impossible
    """
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise AevomError(
                f"Invalid SMILES: {mol_or_smiles!r}\n{diagnose_smiles(mol_or_smiles)}"
            )
    else:
        mol = mol_or_smiles
    if mol is None:
        raise AevomError("Cannot describe None")

    rings = []
    ring_info = mol.GetRingInfo()
    for atoms in ring_info.AtomRings():
        aromatic = all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in atoms)
        hetero = sorted(
            {
                mol.GetAtomWithIdx(i).GetSymbol()
                for i in atoms
                if mol.GetAtomWithIdx(i).GetSymbol() != "C"
            }
        )
        rings.append(
            {
                "size": len(atoms),
                "aromatic": aromatic,
                "heteroatoms": hetero,
                "atoms": list(atoms),
            }
        )
    rings.sort(key=lambda r: (-r["size"], r["aromatic"]))

    groups: list[str] = []
    for name, pattern in _group_patterns():
        if pattern is not None and mol.HasSubstructMatch(pattern):
            groups.append(name)

    return {
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "rings": rings,
        "groups": groups,
        "unstable": unstable_group(mol),
    }


def diagnose_smiles(smiles: str) -> str:
    """Say what is wrong with a SMILES string, in the terms that made it wrong.

    RDKit answers "is this a molecule" with ``None``, which is the right answer
    to that question and no help with the next one. Every message here names a
    mistake that was actually made: these five classes account for all 318
    unparseable structures in a 2759-molecule sample of agent-written SMILES.

    Args:
        smiles: The string that failed to parse.

    Returns:
        One sentence naming the defect, and where possible what to write
        instead. Falls back to a general description when nothing matches.
    """
    try:
        from rdkit import Chem
    except ImportError:  # pragma: no cover - RDKit is always present in pyexec
        return "RDKit is not available, so the string could not be diagnosed."

    raw = Chem.MolFromSmiles(smiles, sanitize=False)
    if raw is not None:
        # The syntax is fine; chemistry failed. RDKit will say which atoms.
        for problem in Chem.DetectChemistryProblems(raw):
            kind = problem.GetType()
            if kind in ("KekulizeException", "AtomKekulizeException"):
                return (
                    f"{problem.Message()} A lowercase ring has to be a real "
                    "aromatic system: check that the ring closed with the atom "
                    "count you intended, and that any ring nitrogen bearing a "
                    "hydrogen is written [nH]. Editing a substituent on a ring "
                    "often drops or adds a ring atom by accident."
                )
            if kind == "AtomValenceException":
                return (
                    f"{problem.Message()} An atom has more bonds than it can "
                    "take, which usually means a group was written in the "
                    "direction it is spoken rather than the direction it bonds: "
                    "a trifluoromethyl on a ring is FC(F)(F)c1..., not "
                    "C(F)(F)Fc1..., which asks the carbon for a fifth bond."
                )
            return problem.Message()
        return "The string parsed but could not be sanitized."

    if re.search(r"\[#\d|\[[A-Za-z]+[XDHRvh]\d", smiles):
        return (
            "This is a SMARTS query pattern, not a SMILES string. SMARTS "
            "describes a set of molecules and is read by Chem.MolFromSmarts; "
            "a SMILES names one molecule."
        )

    if "..." in smiles:
        return "The string contains '...', so a placeholder was left in it."

    if len(smiles) > 25 and set(smiles) <= set("ACDEFGHIKLMNPQRSTVWY"):
        return (
            "This looks like a protein sequence in one-letter code, not a "
            "SMILES string."
        )

    bare = _bare_hydrogen(smiles)
    if bare:
        token, element = bare
        if element:
            return (
                f"'{token}' is not an atom. SMILES infers hydrogens, so the "
                f"element is written {element} on its own; spelling the "
                f"hydrogen out reads as {element} bonded to a separate hydrogen "
                f"atom, which leaves no bond for whatever comes next. Write "
                f"{element}, or [{token}] if the count really is being fixed."
            )
        return (
            "A bare H is not an atom in SMILES -- hydrogen is inferred, and is "
            "written [H] on the rare occasion it has to be explicit."
        )

    for formula, correct in _GROUP_FORMULAS.items():
        if formula in smiles:
            return (
                f"'{formula}' is a group formula, not SMILES. Write {correct}."
            )

    digit = _unbalanced_ring_digit(smiles)
    if digit:
        return (
            f"Ring-closure digit {digit} appears an odd number of times, so a "
            "ring was opened and never closed. Re-using a digit that is still "
            "open, or deleting one half of a pair while editing, both do this."
        )

    # Characters that appear in no SMILES at all. Reaching this point means the
    # string was never SMILES to begin with -- a chemical name, or a formula
    # with placeholder atoms in it -- and saying which character gave it away is
    # more use than saying the whole string is unparseable.
    outside, depth = [], 0
    for character in smiles:
        if character == "[":
            depth += 1
        elif character == "]":
            depth = max(0, depth - 1)
        elif depth == 0:
            outside.append(character)
    stray = sorted(
        {c for c in outside
         if c not in _ORGANIC_SUBSET_CHARS and c not in _STRUCTURAL_CHARS}
    )
    if stray:
        shown = ", ".join(repr(c) for c in stray[:4])
        return (
            f"{shown} cannot appear in SMILES. A name like "
            "'methylcarboxamide' has to be written out as structure, and a "
            "placeholder atom is [*], not X or R."
        )

    if smiles.count("(") != smiles.count(")"):
        return "Unbalanced parentheses."
    if smiles.count("[") != smiles.count("]"):
        return "Unbalanced square brackets."

    return "Not parseable as SMILES."


def _load_sascorer() -> Any:
    """Import RDKit's contributed synthetic accessibility scorer, or None.

    Kept at module scope so the fingerprint database it unpickles on first use
    is paid for once per worker rather than once per call.
    """
    global _sascorer, _sascorer_loaded
    if _sascorer_loaded:
        return _sascorer

    _sascorer_loaded = True
    try:
        from rdkit.Chem import RDConfig

        contrib = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if contrib not in sys.path:
            sys.path.append(contrib)
        import sascorer  # type: ignore[import-not-found]

        _sascorer = sascorer
    except Exception:
        _sascorer = None
    return _sascorer


class AevomError(Exception):
    """Base exception for Aevom client errors."""

    pass


class ReferenceNotFoundError(AevomError):
    """Raised when a reference cannot be found in the data store."""

    pass


class AccessDeniedError(AevomError):
    """Raised when access to a reference is denied."""

    pass


class VisualizationError(AevomError):
    """Raised when visualization fails."""

    pass


class StorageError(AevomError):
    """Raised when storing data fails."""

    pass


# Heavy atoms in each standard residue. A residue with fewer selected atoms is
# partial. Mirrors api.selections.residues.STANDARD_HEAVY_ATOMS; the sandbox
# cannot import the backend, and a test keeps the two in step.
_STANDARD_HEAVY_ATOMS = {
    "ALA": 5, "ARG": 11, "ASN": 8, "ASP": 8, "CYS": 6,
    "GLN": 9, "GLU": 9, "GLY": 4, "HIS": 10, "ILE": 8,
    "LEU": 8, "LYS": 9, "MET": 8, "PHE": 11, "PRO": 7,
    "SER": 6, "THR": 7, "TRP": 14, "TYR": 12, "VAL": 7,
}  # fmt: skip


class Selection(dict):
    """A stored selection: a plain dict with residue-level helpers.

    It *is* the dict ``load_selection`` has always returned (``atoms``,
    ``atom_count``, ``summary`` ...), so existing code is unaffected. The extra
    keys a selection now carries -- ``provenance`` (``"user"`` for residues
    picked by hand, ``"agent"`` otherwise), ``name``, ``parent`` -- are ordinary
    dict entries when present.
    """

    def residues(self) -> list[dict]:
        """The selection's residues, once each, in the order they appear.

        Each residue is a dict with both numbering systems, because they differ
        for many structures:

        - ``number`` / ``chain``: author numbering -- what the viewer and the
          PDB entry show, and what a person means by "residue 56".
        - ``label_seq`` / ``label_chain``: label numbering -- what design tools
          such as RFD4 call ``res_id``.
        - ``name`` (three-letter code), ``ins_code``, ``entity_type``.
        - ``atoms``: names of this residue's atoms that the selection holds.
        - ``partial``: True when only some of a standard residue's atoms are
          selected (a user right-clicked individual atoms, or
          ``select_residues`` was given ``"56:OH"``). A hotspot built from a
          partial residue conditions on those atoms only.

        A selection posted before author ids were recorded has only label ids;
        ``number`` then repeats ``label_seq`` and ``author_known`` is False.
        """
        seen: dict[tuple, dict] = {}
        for atom in self.get("atoms", []):
            residue = atom.get("residue") or {}
            label_chain = residue.get("chainId")
            label_seq = residue.get("seqNumber")
            if label_chain is None:
                continue
            author_known = "authSeqNumber" in residue and "authChainId" in residue
            chain = residue.get("authChainId", label_chain)
            number = residue.get("authSeqNumber", label_seq)
            ins_code = residue.get("insCode", "") or ""
            key = (chain, number, ins_code, residue.get("name"))
            atom_name = atom.get("name")
            if key in seen:
                if atom_name and atom_name not in seen[key]["atoms"]:
                    seen[key]["atoms"].append(atom_name)
                continue
            seen[key] = {
                "chain": chain,
                "number": number,
                "ins_code": ins_code,
                "label_chain": label_chain,
                "label_seq": label_seq,
                "name": residue.get("name", ""),
                "entity_type": residue.get("entityType", ""),
                "author_known": author_known,
                "atoms": [atom_name] if atom_name else [],
            }
        residues = list(seen.values())
        for r in residues:
            expected = _STANDARD_HEAVY_ATOMS.get(str(r["name"]).upper())
            r["partial"] = bool(r["atoms"]) and expected is not None and len(r["atoms"]) < expected
        return residues

    def residue_numbers(self, chain: Optional[str] = None) -> list[int]:
        """Author residue numbers, optionally for one chain (author chain id)."""
        return [
            r["number"]
            for r in self.residues()
            if r["number"] is not None and (chain is None or r["chain"] == chain)
        ]


class AevomClient:
    """Client for loading and storing molecular data in Aevom's sandbox environment.

    This client is pre-loaded as `aevom` in the sandbox globals, allowing
    Python code to interact with Aevom's data storage system using references.

    The client provides methods for:

    1. LOADING DATA - Retrieve stored molecular data by reference
       - load()           : Auto-detect type and load any reference
       - load_pdb()       : Load PDB protein structure as string
       - load_sdf()       : Load SDF molecule as string
       - load_smiles()    : Load SMILES string
       - load_sequence()  : Load protein sequence as dict
       - load_selection() : Load a selection (query, user pick, or agent);
                            .residues() gives both numbering systems
       - load_coords()    : Load coordinates as numpy array (N, 3)
       - load_array()     : Load a numeric array artifact as numpy array
       - load_rdkit_mol() : Load SDF as RDKit Mol object
       - load_pdb_structure() : Load PDB with full atom info as dict
       - load_sdf_structure() : Load SDF with atoms and bonds as dict

    2. STORING DATA - Store new data and receive a reference back
       - store()          : Generic store any data type
       - store_smiles()   : Store SMILES string -> "smiles_abc123"
       - store_sdf()      : Store SDF data -> "sdf_xyz789"
       - store_pdb()      : Store PDB data -> "pdb_def456"
       - store_array()    : Store numeric array -> "array_ghi012"

    3. VISUALIZATION - Display molecules in MolStar 3D viewer
       - visualize()      : Display RDKit mol or SDF string
       - visualize_pdb()  : Display PDB protein structure

    4. RDKIT HELPERS - Convert between molecular formats
       - mol_from_smiles()   : SMILES -> RDKit Mol (2D)
       - mol_from_sdf()      : SDF -> RDKit Mol
       - mol_to_sdf()        : RDKit Mol -> SDF string
       - embed_mol_3d()      : Add 3D coordinates to mol
       - canonical_smiles()  : RDKit Mol -> canonical SMILES

    5. DATABASE LOOKUPS
       - lookup_ccd_code()   : Find PDB CCD code for a molecule name

    Examples:
        # Load a PDB structure
        pdb_data = aevom.load("pdb_abc123")

        # Store a SMILES and get a reference
        ref = aevom.store_smiles("CCO", description="Ethanol")
        print(ref)  # "smiles_abc123"

        # Load coordinates as numpy array for calculations
        coords = aevom.load_coords("pdb_abc123")
        center = coords.mean(axis=0)

        # Load SDF as RDKit molecule for cheminformatics
        mol = aevom.load_rdkit_mol("sdf_xyz789")
        from rdkit.Chem import Descriptors
        print(f"MW: {Descriptors.MolWt(mol)}")

        # Visualize a molecule
        aevom.visualize(mol, label="My Molecule")

    Note:
        This client is automatically available as `aevom` in the sandbox.
        You do not need to import or instantiate it.
    """

    def __init__(
        self,
        session_token: str,
        backend_url: str,
        allowed_refs: Optional[set[str]] = None,
    ):
        """Initialize the Aevom client.

        Args:
            session_token: Session token for backend authentication.
            backend_url: Base URL of the backend API (e.g., "http://backend:8000").
            allowed_refs: Optional set of allowed reference IDs.
        """
        self._session_token = session_token
        self._backend_url = backend_url.rstrip("/")
        self._allowed_refs = allowed_refs or set()
        self._cache: dict[str, bytes] = {}

    def _get_data_type(self, reference: str) -> str:
        """Extract data type from reference ID prefix."""
        # Mirrors the backend's DataType values that contain an underscore;
        # the sandbox cannot import the backend's datatypes module.
        for type_name in ("annotated_cif", "design_spec", "skill_file"):
            if reference.startswith(f"{type_name}_") and len(reference) > len(type_name) + 1:
                return type_name
        parts = reference.split("_", 1)
        if len(parts) != 2:
            raise AevomError(f"Invalid reference format: {reference}")
        return parts[0]

    def _fetch_data(self, reference: str) -> bytes:
        """Fetch data from the backend API.

        Args:
            reference: Reference ID to fetch.

        Returns:
            Raw bytes of the stored data.

        Raises:
            ReferenceNotFoundError: If reference doesn't exist.
            AccessDeniedError: If access to reference is denied.
            AevomError: For other errors.
        """
        # Check cache first
        if reference in self._cache:
            return self._cache[reference]

        # Check if reference is allowed (if restriction is in place)
        if self._allowed_refs and reference not in self._allowed_refs:
            raise AccessDeniedError(
                f"Reference '{reference}' is not accessible in this session. "
                f"Allowed references: {self._allowed_refs}"
            )

        # Build request
        url = f"{self._backend_url}/api/sandbox/data/{reference}"
        headers = {
            "X-Sandbox-Session": self._session_token,
        }

        req = urllib.request.Request(url, headers=headers)

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read()
                # Cache the result
                self._cache[reference] = data
                return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise ReferenceNotFoundError(
                    f"Reference '{reference}' not found in data store"
                )
            elif e.code == 401:
                raise AccessDeniedError(
                    f"Session expired or invalid. Cannot access '{reference}'"
                )
            elif e.code == 403:
                raise AccessDeniedError(f"Access denied for reference '{reference}'")
            elif e.code == 429:
                raise AevomError("Rate limit exceeded. Too many data access requests.")
            else:
                raise AevomError(f"Failed to load '{reference}': HTTP {e.code}")
        except urllib.error.URLError as e:
            raise AevomError(f"Network error loading '{reference}': {e}")

    def load(self, reference: str) -> Union[str, dict, bytes]:
        """Load data from the store by reference (auto-detects type).

        This is the universal loading method that automatically parses data
        based on the reference type prefix. Use this when you want automatic
        type handling, or use the specific load_* methods for type safety.

        Type Detection:
            The reference prefix determines how data is parsed:
            - "pdb_*", "sdf_*", "mol_*", "smiles_*", "safe_*", "csv_*",
              "sequence_*" -> Returns string
            - "selection_*", "json_*" -> Returns dict (parsed JSON)
            - Other types -> Returns raw bytes

        Args:
            reference: Reference ID with type prefix (e.g., "pdb_abc123",
                       "smiles_xyz789", "selection_def456").

        Returns:
            - String for molecular formats (PDB, SDF, SMILES) and sequences
            - Dict for structured data (selections, JSON)
            - Bytes for unknown types

        Raises:
            ReferenceNotFoundError: If reference doesn't exist
            AccessDeniedError: If reference is not accessible
            AevomError: For other errors

        Example:
            # Load different types - method auto-detects format
            pdb_data = aevom.load("pdb_abc123")      # Returns string
            smiles = aevom.load("smiles_xyz789")     # Returns string
            selection = aevom.load("selection_def") # Returns dict

            # For type safety, prefer specific methods:
            pdb_data = aevom.load_pdb("pdb_abc123")
            smiles = aevom.load_smiles("smiles_xyz789")
        """
        data = self._fetch_data(reference)
        data_type = self._get_data_type(reference)

        # Text-based molecular formats
        if data_type in (
            "pdb", "sdf", "mol", "mol2", "mmcif", "cif", "annotated_cif", "smiles", "safe",
            "sequence", "dna", "rna", "msa", "fasta", "text", "csv",
        ):
            return data.decode("utf-8")

        # JSON-based formats
        if data_type in ("selection", "json"):
            return json.loads(data.decode("utf-8"))

        # Numeric arrays (e.g. PAE / pLDDT matrices from a skill endpoint)
        if data_type == "array":
            return self._decode_array(data)

        # Default to raw bytes
        return data

    def load_array(self, reference: str) -> Any:
        """Load a numeric array artifact as a numpy array.

        ``array`` references hold auxiliary numeric outputs that are too large
        and too uninteresting to put in the model's context but are exactly what
        you want in Python: PAE matrices, per-residue pLDDT, contact maps,
        distance matrices.

        Args:
            reference: An ``array_*`` reference, e.g. from a skill endpoint tool.

        Returns:
            numpy.ndarray with the original shape and dtype.

        Raises:
            AevomError: If the artifact is not a readable array.

        Example:
            pae = aevom.load_array("array_abc123")
            print(pae.shape)              # (256, 256)
            print(pae.mean(), pae.max())

            # Interface confidence between two chains
            print(pae[:120, 120:].mean())
        """
        return self._decode_array(self._fetch_data(reference))

    def _decode_array(self, data: bytes) -> Any:
        """Decode a stored ``.npz`` payload into a numpy array."""
        import io

        try:
            import numpy as np
        except ImportError:
            raise AevomError("numpy is not available in this environment")

        try:
            with np.load(io.BytesIO(data), allow_pickle=False) as archive:
                keys = list(archive.keys())
                if not keys:
                    raise AevomError("Array artifact is empty")
                # Single-array artifacts use the key "data"; fall back to the
                # first key for archives written elsewhere.
                return archive["data"] if "data" in archive else archive[keys[0]]
        except AevomError:
            raise
        except Exception as e:
            raise AevomError(f"Failed to decode array artifact: {e}")

    def load_table(self, reference: str) -> list[dict[str, Any]]:
        """Load a stored CSV as a list of row dicts, numbers already parsed.

        The sandbox has neither ``csv`` nor ``pandas``, so a caller told to
        "rank or filter the rest in python_execute" was left splitting on commas
        by hand -- which fails on the first quoted SMILES containing one, and
        silently leaves a carriage return welded to the last field of every row
        when the file was written with CRLF. Measured across 120 benchmark runs,
        64 of them tried to ``import pandas`` or ``import csv`` and were refused;
        20 went on to raise a KeyError or TypeError on a column.

        Values that look numeric are returned as floats and everything else as
        a string, so a threshold comparison works without a cast.

        Args:
            reference: A stored ``csv`` reference.

        Returns:
            One dict per row, keyed by the file's own header names.

        Raises:
            AevomError: If the reference does not hold parseable CSV.

        Example:
            rows = aevom.load_table("csv_a1b2c3d4")
            print(rows[0].keys())                     # the column names
            keep = [r for r in rows if r["CYP3A4_Veith"] < 0.5]
        """
        import csv as _csv
        import io as _io

        raw = self.load(reference)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if not isinstance(raw, str):
            raise AevomError(
                f"{reference} holds {type(raw).__name__}, not text; "
                "load_table expects a stored CSV."
            )

        try:
            rows = list(_csv.DictReader(_io.StringIO(raw)))
        except Exception as exc:
            raise AevomError(f"Could not parse {reference} as CSV: {exc}")
        if not rows:
            return []

        parsed: list[dict[str, Any]] = []
        for row in rows:
            typed: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                text = (value or "").strip()
                try:
                    typed[key.strip()] = float(text)
                except (TypeError, ValueError):
                    typed[key.strip()] = text
            parsed.append(typed)
        return parsed

    def load_text(self, reference: str) -> str:
        """Load a plain-text artifact as a string.

        ``text_*`` references hold tool output that exceeded its context
        budget. The model receives a head/tail preview plus this reference;
        the full output is here, and is usually best handled by searching it
        rather than printing it back out.

        Args:
            reference: A ``text_*`` reference from a capped tool response.

        Returns:
            The full stored text.

        Example:
            full = aevom.load_text("text_abc123")
            hits = [ln for ln in full.splitlines() if "affinity" in ln]
            print(hits[:10])
        """
        return self._fetch_data(reference).decode("utf-8")

    def load_pdb(self, reference: str) -> str:
        """Load a PDB structure as string.

        Args:
            reference: PDB reference ID.

        Returns:
            PDB format string.
        """
        data = self._fetch_data(reference)
        return data.decode("utf-8")

    def load_sdf(self, reference: str) -> str:
        """Load an SDF molecule as string.

        SDF (Structure Data File) format contains 3D atomic coordinates
        and bond information for a molecule. This is the format used
        for 3D visualization in MolStar.

        Args:
            reference: SDF reference ID (e.g., "sdf_abc123").
                       Typically obtained from generate_3d_coordinates tool.

        Returns:
            SDF format string containing the molecule's 3D structure.

        Example:
            sdf_content = aevom.load_sdf("sdf_abc123")
            # Parse with RDKit
            mol = aevom.mol_from_sdf(sdf_content)

        See Also:
            load_rdkit_mol() - Load directly as RDKit Mol object
            load_sdf_structure() - Load with parsed atom/bond info
        """
        data = self._fetch_data(reference)
        return data.decode("utf-8")

    def load_smiles(self, reference: str) -> str:
        """Load a SMILES string from the data store.

        SMILES (Simplified Molecular Input Line Entry System) is a compact
        string notation for representing molecular structures. SMILES stored
        via store_smiles() can be retrieved with this method.

        Args:
            reference: SMILES reference ID (e.g., "smiles_abc123").
                       Obtained from store_smiles() or python_execute code.

        Returns:
            SMILES string (e.g., "CCO" for ethanol, "c1ccccc1" for benzene).

        Example:
            # Retrieve a stored SMILES
            smiles = aevom.load_smiles("smiles_abc123")
            print(f"SMILES: {smiles}")  # e.g., "CC(=O)Oc1ccccc1C(=O)O"

            # Convert to RDKit mol
            mol = aevom.mol_from_smiles(smiles)

        See Also:
            store_smiles() - Store a SMILES string
            mol_from_smiles() - Create RDKit Mol from SMILES
        """
        data = self._fetch_data(reference)
        return data.decode("utf-8")

    def load_safe(self, reference: str) -> str:
        """Load a SAFE fragment notation string from the data store.

        SAFE (Sequential Attachment-based Fragment Embedding) writes a molecule
        as a dot-separated sequence of fragments. It is a valid SMILES, but the
        fragment ordering is what generative models such as GenMol and SAFE-GPT
        are trained on.

        Args:
            reference: SAFE reference ID (e.g., "safe_abc123").

        Returns:
            SAFE string (e.g., "c1ccccc1.CC(=O)O" or a masked scaffold like
            "C1CC(=O)NC1.[*{10-15}]").

        Example:
            safe_str = aevom.load_safe("safe_abc123")
            print(f"{len(safe_str.split('.'))} fragments")

            # Back to a molecule
            smiles = aevom.smiles_from_safe(safe_str)

        See Also:
            store_safe() - Store a SAFE string
            safe_from_smiles() - Encode a SMILES as SAFE
            smiles_from_safe() - Decode a SAFE string back to SMILES
        """
        data = self._fetch_data(reference)
        return data.decode("utf-8")

    def load_sequence(self, reference: str) -> dict:
        """Load a protein sequence as dict.

        Args:
            reference: Sequence reference ID.

        Returns:
            Dict with at least 'sequence' and 'length'.
        """
        text = self._fetch_data(reference).decode("utf-8").strip()
        # Sequence tools store the bare residue string; only a JSON object
        # payload carries extra metadata.
        if text.startswith("{"):
            return json.loads(text)
        return {"sequence": text, "length": len(text)}

    def load_selection(self, reference: str) -> Selection:
        """Load a selection: one made by a structure query, by the user picking
        residues in the viewer, or by ``select_residues``.

        Args:
            reference: Selection reference ID.

        Returns:
            A dict with 'atoms', 'atom_count', 'residue_count', 'provenance'
            ("user" or "agent"), and when set 'name' and 'parent'. Call
            ``.residues()`` for the residues with both numbering systems, or
            ``.residue_numbers(chain)`` for author residue numbers.
        """
        data = self._fetch_data(reference)
        return Selection(json.loads(data.decode("utf-8")))

    def store_selection(
        self,
        structure_ref: str,
        residues: Any,
        chain: str,
        name: Optional[str] = None,
        numbering: str = "author",
        atoms: Optional[Any] = None,
        parent: Optional[str] = None,
    ) -> str:
        """Make a selection from residue numbers and return its reference.

        The result is the same kind of reference a user's pick or
        ``select_residues`` produces: it is listed in the selection state, can be
        shown with ``show_selection``, and can be used as a design hotspot.

        Args:
            structure_ref: A ``pdb_*`` or ``mmcif_*`` reference.
            residues: Residue numbers and ranges, e.g. ``[56, "100-120", "52A"]``.
                Follow a residue with ``:`` and atom names to take only those
                atoms of it: ``"56:OH"``, ``"123:CZ,OH"``, ``"100-110:sidechain"``.
            chain: Chain id, in the same numbering system as ``numbering``.
            name: Short name shown with the selection, e.g. ``"epitope"``.
            numbering: ``"author"`` (the default; what the viewer shows) or
                ``"label"``.
            atoms: Keep only these atoms of every residue listed: names such as
                ``["CA"]``, or ``"backbone"`` / ``"sidechain"``. A residue
                written ``"56:OH"`` uses its own atoms instead.
            parent: Reference of a selection this one derives from.

        Returns:
            The ``selection_*`` reference.

        Raises:
            AevomError: If the residues are not found or the request is refused.
                The message says which residues are missing.

        Example:
            hot = [r["number"] for r in aevom.load_selection(ref).residues()
                   if r["name"] in ("TYR", "TRP")]
            new_ref = aevom.store_selection(pdb_ref, hot, chain="A", name="aromatics")
        """
        url = f"{self._backend_url}/api/sandbox/selection"
        headers = {
            "X-Sandbox-Session": self._session_token,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "structure_ref": structure_ref,
            "chain": chain,
            "residues": list(residues),
            "numbering": numbering,
        }
        if name:
            payload["name"] = name
        if atoms:
            payload["atoms"] = atoms if isinstance(atoms, str) else list(atoms)
        if parent:
            payload["parent"] = parent

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                pass
            raise AevomError(f"store_selection failed: HTTP {e.code}. {detail}")
        except urllib.error.URLError as e:
            raise AevomError(f"Network error storing selection: {e}")

        if not result.get("success"):
            raise AevomError(f"store_selection failed: {result.get('error')}")
        reference = result.get("reference")
        if not reference:
            raise AevomError("store_selection did not return a reference")
        self._allowed_refs.add(reference)
        return reference

    def load_coords(self, reference: str) -> Any:
        """Load coordinates from a structure as numpy array.

        Extracts atomic coordinates from PDB/SDF/selection data and
        returns them as a numpy array of shape (N, 3).

        Args:
            reference: Reference to a PDB, SDF, or selection.

        Returns:
            numpy.ndarray of shape (N, 3) with XYZ coordinates.

        Example:
            coords = aevom.load_coords("pdb_abc123")
            center = coords.mean(axis=0)
        """
        import numpy as np

        data_type = self._get_data_type(reference)

        if data_type == "selection":
            # Selection data has atoms with coords directly
            selection = self.load_selection(reference)
            atoms = selection.get("atoms", [])
            if not atoms:
                return np.array([]).reshape(0, 3)
            return np.array([a["coords"] for a in atoms])

        elif data_type == "pdb":
            # Parse PDB format
            pdb_text = self.load_pdb(reference)
            return self._parse_pdb_coords(pdb_text)

        elif data_type in ("sdf", "mol"):
            # Parse SDF/MOL format
            sdf_text = self.load_sdf(reference)
            return self._parse_sdf_coords(sdf_text)

        else:
            raise AevomError(
                f"Cannot extract coordinates from data type '{data_type}'. "
                f"Supported types: selection, pdb, sdf, mol"
            )

    def _parse_pdb_coords(self, pdb_text: str) -> Any:
        """Parse coordinates from PDB format text.

        Args:
            pdb_text: PDB format string.

        Returns:
            numpy array of shape (N, 3).
        """
        import numpy as np

        coords = []
        for line in pdb_text.split("\n"):
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue

        if not coords:
            return np.array([]).reshape(0, 3)
        return np.array(coords)

    def _parse_sdf_coords(self, sdf_text: str) -> Any:
        """Parse coordinates from SDF/MOL format text.

        Args:
            sdf_text: SDF format string.

        Returns:
            numpy array of shape (N, 3).
        """
        import numpy as np

        coords = []
        lines = sdf_text.split("\n")

        # Find the counts line (line 4 in V2000 format)
        if len(lines) < 4:
            return np.array([]).reshape(0, 3)

        try:
            counts_line = lines[3]
            num_atoms = int(counts_line[:3])

            # Parse atom block (starts at line 5)
            for i in range(4, 4 + num_atoms):
                if i >= len(lines):
                    break
                parts = lines[i].split()
                if len(parts) >= 3:
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    coords.append([x, y, z])
        except (ValueError, IndexError):
            pass

        if not coords:
            return np.array([]).reshape(0, 3)
        return np.array(coords)

    def load_rdkit_mol(self, reference: str) -> Any:
        """Load an SDF as an RDKit Mol object.

        Args:
            reference: SDF reference ID.

        Returns:
            rdkit.Chem.Mol object.

        Raises:
            AevomError: If RDKit is not available or parsing fails.

        Example:
            mol = aevom.load_rdkit_mol("sdf_xyz789")
            mw = Descriptors.MolWt(mol)
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise AevomError("RDKit is not available in this environment")

        sdf_text = self.load_sdf(reference)

        mol = Chem.MolFromMolBlock(sdf_text)
        if mol is None:
            raise AevomError(
                f"Failed to parse SDF from '{reference}' as RDKit molecule"
            )

        return mol

    def load_pdb_structure(self, reference: str) -> dict:
        """Load a PDB file and parse it into a structured dict.

        Returns a dict with atom information preserving all PDB columns
        (atom names, residue names, chain IDs, residue numbers, etc.).

        Args:
            reference: PDB reference ID.

        Returns:
            Dict with 'atoms' list, each containing full PDB atom info.

        Example:
            structure = aevom.load_pdb_structure("pdb_abc123")
            for atom in structure['atoms']:
                print(f"{atom['name']} in {atom['residue_name']}{atom['residue_num']}")
        """
        pdb_text = self.load_pdb(reference)

        atoms = []
        for line in pdb_text.split("\n"):
            if line.startswith(("ATOM", "HETATM")):
                try:
                    atom = {
                        "record_type": line[0:6].strip(),
                        "atom_num": int(line[6:11]),
                        "name": line[12:16].strip(),
                        "alt_loc": line[16:17].strip(),
                        "residue_name": line[17:20].strip(),
                        "chain_id": line[21:22].strip(),
                        "residue_num": int(line[22:26]),
                        "insertion_code": line[26:27].strip(),
                        "x": float(line[30:38]),
                        "y": float(line[38:46]),
                        "z": float(line[46:54]),
                        "coords": [
                            float(line[30:38]),
                            float(line[38:46]),
                            float(line[46:54]),
                        ],
                        "occupancy": float(line[54:60]) if line[54:60].strip() else 1.0,
                        "temp_factor": float(line[60:66])
                        if line[60:66].strip()
                        else 0.0,
                        "element": line[76:78].strip() if len(line) > 76 else "",
                        "charge": line[78:80].strip() if len(line) > 78 else "",
                    }
                    atoms.append(atom)
                except (ValueError, IndexError):
                    continue

        return {
            "atoms": atoms,
            "atom_count": len(atoms),
            "chains": list(set(a["chain_id"] for a in atoms if a["chain_id"])),
            "residue_count": len(
                set(
                    (a["chain_id"], a["residue_num"], a["insertion_code"])
                    for a in atoms
                )
            ),
        }

    def load_sdf_structure(self, reference: str) -> dict:
        """Load an SDF file and parse it into a structured dict.

        Returns a dict with atom and bond information preserving
        all SDF columns.

        Args:
            reference: SDF reference ID.

        Returns:
            Dict with 'atoms' and 'bonds' lists containing full SDF info.
        """
        sdf_text = self.load_sdf(reference)
        lines = sdf_text.split("\n")

        atoms = []
        bonds = []

        if len(lines) < 4:
            return {"atoms": atoms, "bonds": bonds, "atom_count": 0, "bond_count": 0}

        try:
            # Parse counts line
            counts_line = lines[3]
            num_atoms = int(counts_line[:3])
            num_bonds = int(counts_line[3:6])

            # Parse atom block
            for i in range(4, 4 + num_atoms):
                if i >= len(lines):
                    break
                line = lines[i]
                parts = line.split()
                if len(parts) >= 4:
                    atom = {
                        "x": float(parts[0]),
                        "y": float(parts[1]),
                        "z": float(parts[2]),
                        "coords": [float(parts[0]), float(parts[1]), float(parts[2])],
                        "symbol": parts[3],
                        "mass_diff": int(parts[4]) if len(parts) > 4 else 0,
                        "charge": int(parts[5]) if len(parts) > 5 else 0,
                        "index": len(atoms),
                    }
                    atoms.append(atom)

            # Parse bond block
            bond_start = 4 + num_atoms
            for i in range(bond_start, bond_start + num_bonds):
                if i >= len(lines):
                    break
                line = lines[i]
                parts = line.split()
                if len(parts) >= 3:
                    bond = {
                        "atom1": int(parts[0]) - 1,  # 0-indexed
                        "atom2": int(parts[1]) - 1,
                        "order": int(parts[2]),
                        "stereo": int(parts[3]) if len(parts) > 3 else 0,
                    }
                    bonds.append(bond)
        except (ValueError, IndexError):
            pass

        return {
            "atoms": atoms,
            "bonds": bonds,
            "atom_count": len(atoms),
            "bond_count": len(bonds),
        }

    def clear_cache(self) -> None:
        """Clear the local data cache."""
        self._cache.clear()

    @property
    def allowed_refs(self) -> set[str]:
        """Get the set of allowed references for this session."""
        return self._allowed_refs.copy()

    # =========================================================================
    # Store Methods - Store data back to the backend
    # =========================================================================

    def store(
        self,
        data_type: str,
        content: Union[str, bytes],
        metadata: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Store data in Aevom's backend data store (generic method).

        This is the low-level storage method. For convenience, use the
        type-specific methods instead:
        - store_smiles() for SMILES strings
        - store_sdf() for SDF molecular data
        - store_pdb() for PDB protein structures

        The stored data receives a reference ID that can be used to:
        1. Retrieve the data later with load() or load_*() methods
        2. Pass to LLM tools that accept reference parameters
        3. Share data between Python code and tool calls

        Args:
            data_type: Type prefix for the reference ID. Common types:
                       - "smiles" -> Creates "smiles_abc123" reference
                       - "sdf" -> Creates "sdf_abc123" reference
                       - "pdb" -> Creates "pdb_abc123" reference
                       - "json" -> Creates "json_abc123" reference
            content: String content (encoded as UTF-8) or raw bytes, which are
                     sent base64-encoded. Use bytes for binary types like
                     "array".
            metadata: Optional dict with additional info stored alongside
                      the content. Retrieved with retrieve_with_metadata().
            description: Human-readable description shown in storage listings.

        Returns:
            Reference ID (e.g., "smiles_abc12345", "sdf_xyz789") that uniquely
            identifies the stored data.

        Raises:
            StorageError: If storage fails (network error, invalid data, etc.)

        Example:
            # Store custom JSON data
            import json
            data = {"analysis": "complete", "results": [1, 2, 3]}
            ref = aevom.store(
                "json",
                json.dumps(data),
                description="Analysis results"
            )
            print(f"Stored as: {ref}")  # "json_abc12345"

        See Also:
            store_smiles() - Store SMILES strings
            store_sdf() - Store SDF molecular data
            store_pdb() - Store PDB structures
        """
        url = f"{self._backend_url}/api/sandbox/store"
        headers = {
            "X-Sandbox-Session": self._session_token,
            "Content-Type": "application/json",
        }

        payload: dict[str, Any] = {
            "data_type": data_type,
            "metadata": metadata or {},
        }
        if isinstance(content, (bytes, bytearray)):
            payload["content_base64"] = base64.b64encode(bytes(content)).decode("ascii")
        else:
            payload["content"] = content
        if description:
            payload["description"] = description

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                reference = result.get("reference")
                if not reference:
                    raise StorageError("Backend did not return a reference")

                # Add to allowed refs for this session
                self._allowed_refs.add(reference)

                return reference
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise StorageError(f"Failed to store data: HTTP {e.code}. {error_body}")
        except urllib.error.URLError as e:
            raise StorageError(f"Network error storing data: {e}")

    def store_sdf(
        self,
        sdf_content: str,
        metadata: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Store SDF (3D molecular structure) data in the data store.

        SDF (Structure Data File) format contains 3D atomic coordinates and
        bond connectivity for a molecule. Use this to store:
        - Molecules with 3D coordinates from RDKit embedding
        - Modified molecular structures
        - Results from conformer generation

        Args:
            sdf_content: SDF/MOL format string. Must contain valid V2000 or
                         V3000 format data with atom coordinates.
            metadata: Optional dict with additional info. Useful keys:
                      - smiles: Original SMILES string
                      - name: Common name of the molecule
                      - source: Where the structure came from
            description: Human-readable description for tracking.

        Returns:
            Reference ID (e.g., "sdf_abc12345") that can be passed to:
            - visualize_3d(sdf_ref=...) to display in viewer
            - vina_dock(ligand_sdf_ref=...) for docking
            - load_sdf() or load_rdkit_mol() to retrieve

        Example:
            # Create 3D structure with RDKit and store
            mol = aevom.mol_from_smiles("CCO")
            aevom.embed_mol_3d(mol)
            sdf_content = aevom.mol_to_sdf(mol)
            ref = aevom.store_sdf(
                sdf_content,
                metadata={"smiles": "CCO", "name": "ethanol"},
                description="Ethanol 3D structure"
            )
            print(f"Stored as: {ref}")

        See Also:
            load_sdf() - Retrieve stored SDF
            load_rdkit_mol() - Load SDF as RDKit Mol
            visualize() - Display in 3D viewer
        """
        return self.store("sdf", sdf_content, metadata, description)

    def store_pdb(
        self,
        pdb_content: str,
        metadata: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Store PDB structure data in the data store.

        PDB (Protein Data Bank) format contains atomic coordinates for
        protein structures. Use this to store predicted structures,
        modified proteins, or analysis results.

        Args:
            pdb_content: PDB format string containing ATOM/HETATM records.
            metadata: Optional dict with additional info. Useful keys:
                      - protein_id: UniProt ID (e.g., "P12345")
                      - gene_name: Gene name (e.g., "PGK1")
                      - source: Where the structure came from
            description: Human-readable description for tracking.

        Returns:
            Reference ID (e.g., "pdb_abc12345") that can be passed to
            other tools like visualize_pdb or used with load_pdb().

        Example:
            # Store a modified structure
            ref = aevom.store_pdb(
                pdb_content,
                metadata={"protein_id": "P12345", "modification": "ligand_bound"},
                description="PGK1 with docked aspirin"
            )
            print(f"Stored as: {ref}")  # "pdb_abc12345"

        See Also:
            load_pdb() - Retrieve stored PDB
            visualize_pdb() - Display PDB in viewer
        """
        return self.store("pdb", pdb_content, metadata, description)

    def store_smiles(
        self,
        smiles: str,
        metadata: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Store a SMILES string in the data store.

        SMILES (Simplified Molecular Input Line Entry System) is a compact
        string notation for representing molecular structures. Storing SMILES
        creates a reference that can be passed to tools like generate_3d_coordinates.

        This is the PRIMARY WAY to get molecules into the Aevom pipeline from
        Python code. Common sources of SMILES include:
        - PubChem lookups (pubchempy)
        - RDKit molecule manipulation
        - Manual specification of novel compounds

        Args:
            smiles: SMILES string representing the molecule.
                    Examples:
                    - "CCO" (ethanol)
                    - "c1ccccc1" (benzene)
                    - "CC(=O)Oc1ccccc1C(=O)O" (aspirin)
                    - "Cn1cnc2c1c(=O)n(c(=O)n2C)C" (caffeine)
            metadata: Optional dict with additional info. Useful keys:
                      - name: Common name of the molecule
                      - source: Where the SMILES came from (e.g., "PubChem")
                      - cid: PubChem CID if from PubChem
            description: Human-readable description for tracking.

        Returns:
            Reference ID (e.g., "smiles_abc12345") that can be passed to:
            - generate_3d_coordinates(smiles_ref=...) to create 3D structure
            - load_smiles() to retrieve the SMILES string

        Example:
            # From PubChem lookup
            import pubchempy as pcp
            compound = pcp.get_compounds("ibuprofen", "name")[0]
            ref = aevom.store_smiles(
                compound.canonical_smiles,
                metadata={"name": "ibuprofen", "cid": compound.cid},
                description="Ibuprofen from PubChem"
            )
            print(f"Stored as: {ref}")  # "smiles_abc12345"

            # From RDKit manipulation
            mol = aevom.mol_from_smiles("c1ccccc1")
            mol = Chem.AddHs(mol)
            new_smiles = aevom.canonical_smiles(mol)
            ref = aevom.store_smiles(new_smiles, description="Modified benzene")

            # The LLM can now call:
            # generate_3d_coordinates(smiles_ref=ref) -> returns sdf_ref

        See Also:
            load_smiles() - Retrieve stored SMILES
            mol_from_smiles() - Create RDKit Mol from SMILES
        """
        if not smiles or not smiles.strip():
            raise StorageError("SMILES string cannot be empty")
        return self.store("smiles", smiles.strip(), metadata, description)

    def store_safe(
        self,
        safe_string: str,
        metadata: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Store a SAFE fragment notation string in the data store.

        SAFE (Sequential Attachment-based Fragment Embedding) writes a molecule
        as a dot-separated sequence of fragments. Store it as SAFE rather than
        as SMILES when the consumer is a generative model: GenMol and SAFE-GPT
        both require the fragment ordering and both name their input field
        "smiles", so a plain SMILES is accepted and quietly produces worse
        output. Keeping the type distinct is what stops that mix-up.

        Args:
            safe_string: SAFE string. May include GenMol generation masks, e.g.
                    - "[*{20-30}]" (de novo)
                    - "C1CC(=O)NC1.[*{10-15}]" (scaffold decoration)
                    - "[*{5-10}].c1ccccc1.[*{5-10}]" (motif extension)
            metadata: Optional dict with additional info. Useful keys:
                      - name: Common name of the molecule
                      - slicer: Fragmentation algorithm used ("brics", ...)
                      - smiles: The SMILES this was encoded from
            description: Human-readable description for tracking.

        Returns:
            Reference ID (e.g., "safe_abc12345") that can be passed to
            safe_to_smiles, to a GenMol skill or MCP tool, or to load_safe().

        Example:
            safe_str = aevom.safe_from_smiles("CC(Cc1ccc(cc1)C(C(=O)O)C)C")
            ref = aevom.store_safe(
                safe_str,
                metadata={"name": "ibuprofen", "slicer": "brics"},
                description="Ibuprofen in SAFE notation",
            )

            # Masked scaffold, ready for GenMol
            scaffold_ref = aevom.store_safe(
                aevom.scaffold_to_safe("C1CC(=O)NC1", 10, 15),
                description="Lactam scaffold with 10-15 token mask",
            )

        See Also:
            load_safe() - Retrieve stored SAFE
            safe_from_smiles() - Encode a SMILES as SAFE
            scaffold_to_safe() - Encode and append a generation mask
        """
        if not safe_string or not safe_string.strip():
            raise StorageError("SAFE string cannot be empty")
        return self.store("safe", safe_string.strip(), metadata, description)

    def store_array(
        self,
        array: Any,
        metadata: Optional[dict[str, Any]] = None,
        description: Optional[str] = None,
    ) -> str:
        """Store a numeric array (or anything castable to one) as an artifact.

        Use this for derived numeric results -- a filtered PAE matrix, computed
        distances, per-residue scores -- so they can be handed to another tool or
        reloaded in a later ``python_execute`` call without going through the
        model's context.

        Args:
            array: A numpy array, or a nested numeric list/tuple.
            metadata: Optional dict stored alongside the array.
            description: Human-readable description for tracking.

        Returns:
            Reference ID (e.g. "array_abc12345").

        Raises:
            StorageError: If the value is not a numeric array or storage fails.

        Example:
            pae = aevom.load_array("array_abc123")
            interface = pae[:120, 120:]
            ref = aevom.store_array(interface, description="Chain A/B interface PAE")
        """
        import io

        try:
            import numpy as np
        except ImportError:
            raise StorageError("numpy is not available in this environment")

        try:
            arr = np.asarray(array)
        except (TypeError, ValueError) as e:
            raise StorageError(f"Value cannot be converted to an array: {e}")

        if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
            raise StorageError(
                f"store_array requires numeric data, got dtype '{arr.dtype}'. "
                "Use aevom.store('json', json.dumps(value)) for non-numeric data."
            )

        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=arr)

        meta = dict(metadata or {})
        meta.setdefault("shape", list(arr.shape))
        meta.setdefault("dtype", str(arr.dtype))

        return self.store("array", buffer.getvalue(), meta, description)

    # =========================================================================
    # Specialized Computation - Work delegated to Aevom's services
    # =========================================================================

    def probe_grid(
        self,
        receptor_ref: str,
        center: Any,
        size: Any = (20.0, 20.0, 20.0),
        spacing: float = 0.5,
        probes: Optional[list] = None,
        max_minima: int = 5,
        return_details: bool = False,
    ) -> Any:
        """Map where each pharmacophore type would want to bind in a pocket.

        Places a probe atom at every point of a box around ``center`` and scores
        it against the receptor with AutoDock Vina's interaction function. A
        minimum in the acceptor-probe grid is a position where a *ligand*
        acceptor would be well satisfied, because a receptor donor points at it.

        Nothing is docked and no ligand is needed: the result depends only on
        the receptor, which is what makes it usable before a molecule exists.

        The grid is computed by Aevom's Vina service and stored as an artifact,
        so the array comes back over the ordinary data route and never passes
        through the model's context.

        Args:
            receptor_ref: A ``pdb_*`` reference for the receptor.
            center: Pocket centre ``[x, y, z]`` in angstrom.
            size: Box dimensions ``[x, y, z]`` in angstrom. Default 20 each way.
            spacing: Grid spacing in angstrom. Default 0.5.
            probes: Probe types, default ``["C_H", "N_D", "O_A"]`` -- hydrophobe,
                donor and acceptor. Also available: ``C_P``, ``N_P``, ``N_A``,
                ``O_DA``.
            max_minima: Ranked minima to return per probe.
            return_details: Return a dict with the grid plus its geometry and
                minima, instead of the bare array.

        Returns:
            A numpy array of shape ``(n_probes, nx, ny, nz)``, ordered to match
            ``probes``. With ``return_details=True``, a dict with keys
            ``grid``, ``probes``, ``origin``, ``spacing``, ``shape``,
            ``minima`` and ``reference``.

            Grid point ``(i, j, k)`` is at ``origin + spacing * (i, j, k)``.

        Raises:
            AevomError: If the receptor cannot be read or the service fails.

        Example:
            grid = aevom.probe_grid("pdb_abc12345", center=[12.4, -3.1, 25.6])
            donor = grid[1]                       # the N_D probe
            print(donor.min())                    # best donor site energy

            out = aevom.probe_grid(ref, center, return_details=True)
            for m in out["minima"][:3]:
                print(m["family"], m["x"], m["y"], m["z"], m["energy"])
        """
        center = [float(v) for v in center]
        size = [float(v) for v in size]
        if len(center) != 3 or len(size) != 3:
            raise AevomError("center and size must each have three components")

        url = f"{self._backend_url}/api/sandbox/probe-grid"
        headers = {
            "X-Sandbox-Session": self._session_token,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "receptor_ref": receptor_ref,
            "center": center,
            "size": size,
            "spacing": float(spacing),
            "max_minima": int(max_minima),
        }
        if probes:
            payload["probes"] = list(probes)

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            # Generous: a fine grid over a large box is minutes of numpy in the
            # service, and there is no partial result to fall back to.
            with urllib.request.urlopen(request, timeout=300) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                pass
            raise AevomError(f"probe_grid failed: HTTP {e.code}. {detail}")
        except urllib.error.URLError as e:
            raise AevomError(f"Network error computing probe grid: {e}")

        if not result.get("success"):
            raise AevomError(f"probe_grid failed: {result.get('error')}")

        reference = result.get("reference")
        if not reference:
            raise AevomError("probe_grid did not return a grid reference")
        self._allowed_refs.add(reference)

        grid = self.load_array(reference)
        if not return_details:
            return grid
        return {
            "grid": grid,
            "probes": result.get("probes"),
            "origin": result.get("origin"),
            "spacing": result.get("spacing"),
            "shape": result.get("shape"),
            "minima": result.get("minima"),
            "reference": reference,
        }

    def constrained_pose(
        self,
        fragment_refs: Any,
        smiles: Any,
        pocket_ref: Optional[str] = None,
        num_seeds: int = 4,
        strain_max: Optional[float] = None,
        rmsd_max: Optional[float] = None,
        store_poses: bool = False,
        return_details: bool = False,
    ) -> Any:
        """Ask whether each candidate can hold the fragments where they were posed.

        Fragment Assembly's decisive gate is geometric: the answer has to contain
        both fragments *and* be a molecule that keeps them roughly where the
        witness complex put them. This is that question, answered offline. Each
        candidate is built with its fragment atoms pinned to the coordinates the
        SDFs carry, minimised with those atoms tethered, and measured -- how far
        each fragment ended up from its given pose, what the strain of holding it
        costs relative to the molecule's own free minimum, and how close the
        result comes to the protein. A candidate that cannot be embedded at all
        is rejected, and that is the cheapest signal there is.

        The ``fragment_link`` tool runs this same check over a handful of
        candidates and reports them. Use this when you have hundreds -- a
        generated library to filter, a systematic linker scan to rank -- and want
        the loop in Python, where the rejects cost nothing.

        Args:
            fragment_refs: ``sdf_*`` references, one per fragment, carrying poses.
            smiles: Candidate molecules, up to 200 per call.
            pocket_ref: ``pdb_*`` reference for the pocket. Without it no clash
                check runs and a molecule threaded through the protein verifies.
            num_seeds: Restrained-embedding attempts per candidate. More is
                slower and slightly more forgiving of a hard geometry.
            strain_max: Strain ceiling in kcal/mol. Defaults to the service's
                calibrated value; pass a larger one to see near misses.
            rmsd_max: How far a fragment may drift from its given pose, in
                angstrom.
            store_poses: Store each verified pose as an ``sdf_*`` reference,
                returned as ``pose_ref``. Off by default.
            return_details: Return the whole response dict rather than the rows.

        Returns:
            A list of dicts, one per candidate in the order given, with
            ``smiles``, ``verified``, ``failed_gate``, ``reason``, ``strain``,
            ``max_fragment_rmsd``, ``min_protein_distance``, ``num_clashes`` and
            ``num_linker_atoms``. With ``return_details=True``, a dict with
            ``results``, ``num_verified``, ``pocket_atoms`` and ``thresholds``.

        Raises:
            AevomError: If a reference cannot be read or the service fails.

        Example:
            rows = aevom.constrained_pose(
                ["sdf_aaa11111", "sdf_bbb22222"], candidates, pocket_ref="pdb_ccc33333"
            )
            kept = [r for r in rows if r["verified"]]
            kept.sort(key=lambda r: r["strain"])
            for r in kept[:5]:
                print(r["smiles"], round(r["strain"], 1), r["min_protein_distance"])

            # The rejections say what to change, so count them by gate
            from collections import Counter
            print(Counter(r["failed_gate"] for r in rows if not r["verified"]))
        """
        fragment_refs = [str(r) for r in fragment_refs]
        smiles = [str(s) for s in smiles]
        if not fragment_refs:
            raise AevomError("constrained_pose needs at least one fragment reference")
        if not smiles:
            raise AevomError("constrained_pose needs at least one candidate SMILES")

        url = f"{self._backend_url}/api/sandbox/constrained-pose"
        headers = {
            "X-Sandbox-Session": self._session_token,
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "fragment_refs": fragment_refs,
            "smiles": smiles,
            "num_seeds": int(num_seeds),
            "store_poses": bool(store_poses),
        }
        if pocket_ref:
            payload["pocket_ref"] = str(pocket_ref)
        if strain_max is not None:
            payload["strain_max"] = float(strain_max)
        if rmsd_max is not None:
            payload["rmsd_max"] = float(rmsd_max)

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        try:
            # Each candidate is an embedding and two minimisations per seed, so a
            # full batch of 200 is minutes of force field with no partial result.
            with urllib.request.urlopen(request, timeout=900) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")
            except Exception:
                pass
            raise AevomError(f"constrained_pose failed: HTTP {e.code}. {detail}")
        except urllib.error.URLError as e:
            raise AevomError(f"Network error verifying poses: {e}")

        if not result.get("success"):
            raise AevomError(f"constrained_pose failed: {result.get('error')}")

        for row in result.get("results") or []:
            if row.get("pose_ref"):
                self._allowed_refs.add(row["pose_ref"])

        if return_details:
            return result
        return result.get("results") or []

    # =========================================================================
    # Visualization Methods - Trigger MolStar display
    # =========================================================================

    def visualize(self, mol_or_sdf: Any, label: Optional[str] = None) -> bool:
        """Display a molecule in the MolStar 3D viewer.

        Accepts an RDKit Mol object, SDF string, or a data reference (sdf_*).
        The visualization is triggered immediately in the frontend.

        Args:
            mol_or_sdf: Either an RDKit Mol object, SDF string, or an sdf_*
                        reference from download_pubchem_sdf() or store_sdf().
            label: Optional display label for the molecule.

        Returns:
            True if visualization was triggered successfully.

        Raises:
            VisualizationError: If visualization fails.

        Example:
            # Visualize RDKit mol
            mol = aevom.load_rdkit_mol("sdf_abc123")
            aevom.visualize(mol, label="My Molecule")

            # Visualize SDF string
            aevom.visualize(sdf_content, label="Aspirin")

            # Visualize by reference (auto-loads content)
            sdf_ref = aevom.download_pubchem_sdf(cid)
            aevom.visualize(sdf_ref, label="PubChem Molecule")
        """
        # Convert to SDF if it's an RDKit mol
        if hasattr(mol_or_sdf, "GetNumAtoms"):
            # It's an RDKit Mol object
            sdf_content = self.mol_to_sdf(mol_or_sdf)
        elif isinstance(mol_or_sdf, str):
            # Check if it's a reference string (sdf_*) and load the content
            if mol_or_sdf.startswith("sdf_") and len(mol_or_sdf) < 50:
                # It's a reference - load the actual SDF content
                sdf_content = self.load_sdf(mol_or_sdf)
            else:
                # Treat as literal SDF content
                sdf_content = mol_or_sdf
        else:
            raise VisualizationError(
                f"Expected RDKit Mol, SDF string, or sdf_* reference, got: {type(mol_or_sdf)}"
            )

        if not sdf_content or not sdf_content.strip():
            raise VisualizationError("SDF content is empty")

        url = f"{self._backend_url}/api/sandbox/visualize"
        headers = {
            "X-Sandbox-Session": self._session_token,
            "Content-Type": "application/json",
        }

        payload = {
            "sdf": sdf_content,
            "label": label or "Molecule",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("success", False)
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise VisualizationError(
                f"Failed to trigger visualization: HTTP {e.code}. {error_body}"
            )
        except urllib.error.URLError as e:
            raise VisualizationError(f"Network error triggering visualization: {e}")

    def visualize_pdb(self, pdb_content: str, label: Optional[str] = None) -> bool:
        """Display a protein structure in the MolStar 3D viewer.

        Args:
            pdb_content: PDB format string or a pdb_* reference from
                         fetch_and_clean_pdb() or store_pdb().
            label: Optional display label for the structure.

        Returns:
            True if visualization was triggered successfully.

        Raises:
            VisualizationError: If visualization fails.

        Example:
            pdb = aevom.load_pdb("pdb_abc123")
            aevom.visualize_pdb(pdb, label="My Protein")

            # Or visualize by reference (auto-loads content)
            aevom.visualize_pdb("pdb_abc123", label="My Protein")
        """
        # Check if it's a reference string (pdb_*) and load the content
        if isinstance(pdb_content, str) and pdb_content.startswith("pdb_") and len(pdb_content) < 50:
            pdb_content = self.load_pdb(pdb_content)

        if not pdb_content or not pdb_content.strip():
            raise VisualizationError("PDB content is empty")

        url = f"{self._backend_url}/api/sandbox/visualize"
        headers = {
            "X-Sandbox-Session": self._session_token,
            "Content-Type": "application/json",
        }

        payload = {
            "pdb": pdb_content,
            "label": label or "Protein Structure",
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result.get("success", False)
        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise VisualizationError(
                f"Failed to trigger visualization: HTTP {e.code}. {error_body}"
            )
        except urllib.error.URLError as e:
            raise VisualizationError(f"Network error triggering visualization: {e}")

    # =========================================================================
    # RDKit Helper Methods
    # =========================================================================

    def mol_to_sdf(self, mol: Any) -> str:
        """Convert an RDKit Mol object to SDF string.

        Args:
            mol: RDKit Mol object.

        Returns:
            SDF format string.

        Raises:
            AevomError: If conversion fails.

        Example:
            mol = Chem.MolFromSmiles("CCO")
            AllChem.EmbedMolecule(mol)
            sdf = aevom.mol_to_sdf(mol)
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise AevomError("RDKit is not available in this environment")

        if mol is None:
            raise AevomError("Cannot convert None to SDF")

        try:
            sdf = Chem.MolToMolBlock(mol)
            return sdf
        except Exception as e:
            raise AevomError(f"Failed to convert mol to SDF: {e}")

    def mol_from_sdf(self, sdf_content: str) -> Any:
        """Create an RDKit Mol object from SDF string.

        Args:
            sdf_content: SDF format string.

        Returns:
            RDKit Mol object.

        Raises:
            AevomError: If parsing fails.

        Example:
            mol = aevom.mol_from_sdf(sdf_content)
            print(f"Atoms: {mol.GetNumAtoms()}")
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise AevomError("RDKit is not available in this environment")

        mol = Chem.MolFromMolBlock(sdf_content)
        if mol is None:
            raise AevomError("Failed to parse SDF as RDKit molecule")

        return mol

    def mol_from_smiles(self, smiles: str) -> Any:
        """Create an RDKit Mol object from SMILES string.

        Note: This creates a 2D molecule. Use AllChem.EmbedMolecule()
        to generate 3D coordinates.

        Args:
            smiles: SMILES string.

        Returns:
            RDKit Mol object.

        Prefer this to ``Chem.MolFromSmiles`` for a string you wrote yourself.
        RDKit returns ``None`` for anything it cannot read, so a loop that skips
        the failures quietly loses molecules and never says how many; this
        raises, and the message names the mistake rather than restating that
        there was one.

        Raises:
            AevomError: If parsing fails, with a diagnosis of why.

        Example:
            mol = aevom.mol_from_smiles("CCO")
            AllChem.EmbedMolecule(mol)  # Add 3D coords
            aevom.visualize(mol, label="Ethanol")
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise AevomError("RDKit is not available in this environment")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise AevomError(
                f"Failed to parse SMILES: {smiles}\n{diagnose_smiles(smiles)}"
            )

        return mol

    def canonical_smiles(self, mol: Any) -> str:
        """Get canonical SMILES from an RDKit Mol object.

        Args:
            mol: RDKit Mol object.

        Returns:
            Canonical SMILES string.

        Raises:
            AevomError: If conversion fails.

        Example:
            mol = aevom.load_rdkit_mol("sdf_abc123")
            smiles = aevom.canonical_smiles(mol)
            print(f"Canonical SMILES: {smiles}")
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise AevomError("RDKit is not available in this environment")

        if mol is None:
            raise AevomError("Cannot get SMILES from None")

        try:
            smiles = Chem.MolToSmiles(mol)
            return smiles
        except Exception as e:
            raise AevomError(f"Failed to get canonical SMILES: {e}")

    def embed_mol_3d(self, mol: Any, optimize: bool = True) -> Any:
        """Generate 3D coordinates for an RDKit Mol object.

        Uses RDKit's ETKDG algorithm for conformer generation
        and optionally optimizes with MMFF force field.

        Args:
            mol: RDKit Mol object (will be modified in place).
            optimize: Whether to optimize geometry with MMFF (default True).

        Returns:
            The same mol object with 3D coordinates.

        Raises:
            AevomError: If embedding fails.

        Example:
            mol = aevom.mol_from_smiles("CCO")
            aevom.embed_mol_3d(mol)
            aevom.visualize(mol, label="Ethanol 3D")
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise AevomError("RDKit is not available in this environment")

        if mol is None:
            raise AevomError("Cannot embed None")

        try:
            # Add hydrogens for better 3D geometry
            mol = Chem.AddHs(mol)

            # Generate 3D coordinates using ETKDG
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())

            if result == -1:
                # Try with random coordinates
                result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
                if result == -1:
                    raise AevomError(
                        "Failed to generate 3D coordinates. "
                        "The molecule may be too constrained."
                    )

            if optimize:
                # Optimize with MMFF force field
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except Exception:
                    # MMFF might fail for some molecules, continue anyway
                    pass

            return mol

        except Exception as e:
            if isinstance(e, AevomError):
                raise
            raise AevomError(f"Failed to embed 3D coordinates: {e}")

    # Substituents a medicinal chemist actually reaches for on a lead: the
    # halogen scan, small alkyls and cycloalkyls, the common H-bond donors and
    # acceptors, and the polar solubilising groups. Ordered roughly by how
    # often they appear in a matched pair, so a truncated enumeration is still
    # a sensible one.
    ANALOG_R_GROUPS = (
        "[H]", "F", "Cl", "Br", "I",
        "C", "CC", "CCC", "C(C)C", "C(C)(C)C",
        "C1CC1", "C1CCC1", "C1CCCC1",
        "C(F)(F)F", "C(F)F", "OC(F)(F)F",
        "O", "OC", "OCC", "OC(C)C",
        "N", "NC", "N(C)C", "NC(C)=O",
        "C#N", "[N+](=O)[O-]", "S(C)(=O)=O", "SC",
        "C(N)=O", "C(=O)O", "C(=O)C", "C(=O)OC",
        "CO", "CCO", "CN",
        "c1ccccc1", "c1ccncc1", "N1CCOCC1", "N1CCNCC1", "N1CCCC1",
    )

    def enumerate_analogs(
        self,
        mol_or_smiles: Any,
        r_groups: Any = None,
        max_analogs: int = 2000,
        similarity_min: Optional[float] = None,
        decorate: bool = True,
        allow_unstable: bool = False,
    ) -> list:
        """Enumerate single-edit analogues of a lead.

        Two kinds of edit, which between them cover most of what a matched
        molecular pair is: swap one substituent for another, by cutting a
        single acyclic bond and re-attaching something else; and decorate an
        aromatic CH that had no substituent at all.

        This is the workhorse of lead optimisation, where the answer is
        usually a small change to a known molecule rather than a new one.
        Generative models are the wrong tool for it -- they cannot be held
        within a similarity bound, and being held there is the requirement.

        **Pass the substituents you have in mind; do not write the molecules.**
        If the forty defaults do not contain the chemistry you want, put yours
        in ``r_groups`` and let this attach them. A substituent is short, local
        and self-contained -- "C#C", "OC(F)(F)F", "N1CCOCC1" -- and splicing it
        into a ring is the part that goes wrong. Measured over 2759 structures
        written out by hand into this sandbox, **11.5% would not parse and only
        half the rest stayed inside the similarity bound: 45% unusable, and
        silently so**, because a loop that skips unparseable strings loses them
        without saying how many. Every one of those parse failures was a
        splicing error -- an "NH" written as an atom, a ring closure left open,
        a CF3 bonded a fifth time, a ring left with the wrong atom count. None
        of them is reachable from here, because the attaching is done by RDKit.

        One edit is normally enough. Across the 68 lead-optimisation tasks of
        SMDD-Bench, the analogues this produces from a single cut contain a
        molecule satisfying every stated objective and constraint in 66 of
        them, so reaching for two simultaneous changes is usually a detour.

        What it does not do is edit the scaffold itself: no ring-heteroatom
        walks (pyridine to benzene), no positional isomers, no ring
        contractions. Those are real lead-optimisation moves, so an analogue
        you have in mind and cannot find here may simply be out of range
        rather than unreachable.

        Args:
            mol_or_smiles: The lead, as an RDKit Mol or a SMILES string.
            r_groups: Substituents to try, as SMILES. Defaults to
                ``aevom.ANALOG_R_GROUPS``, forty common ones -- which are a
                starting point, not a vocabulary: the same measurement found
                712 distinct substituents in use, 584 of them only once. Supply
                your own for anything the defaults miss. ``"[H]"`` means delete
                the substituent rather than replace it.
            max_analogs: Stop after this many distinct products.
            similarity_min: If given, keep only analogues at least this
                similar to the lead (Morgan radius 2, 2048 bits) -- the usual
                bound in a lead-optimisation brief, and the one that decides
                most of them.
            decorate: Whether to also substitute aromatic CH positions. Turn
                it off to keep the heavy-atom count fixed.
                allow_unstable: Keep products carrying a group from
                ``aevom.UNSTABLE_GROUPS`` -- hypohalites, sulfenyl halides,
                N-halamines, acyl halides, acyclic peroxides, anhydrides. Off,
                because a single cut is blind to what it attaches to and makes
                these by accident at about 12% of products, and no alert
                catalogue in the stack rejects them.

        Returns:
            Canonical SMILES, without the input itself and without anything
            that came apart into disconnected fragments.

        Raises:
            AevomError: If the input is not a valid molecule.

        Example:
            analogs = aevom.enumerate_analogs(ref, similarity_min=0.7)
            print(len(analogs))          # a few hundred, typically
            # then score them in one batched call rather than one at a time

            # With chemistry of your own in mind -- substituents, not molecules:
            analogs = aevom.enumerate_analogs(
                ref, r_groups=["C#C", "C=C", "SC(F)(F)F", "CN(C)S(=O)(=O)"],
                similarity_min=0.7,
            )
        """
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem, DataStructs, rdMMPA
        except ImportError:
            raise AevomError("RDKit is not available in this environment")

        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                raise AevomError(f"Invalid SMILES: {mol_or_smiles!r}")
        elif mol_or_smiles is None:
            raise AevomError("Cannot enumerate analogues of None")
        else:
            mol = mol_or_smiles

        groups = list(r_groups) if r_groups is not None else list(self.ANALOG_R_GROUPS)
        if not groups:
            raise AevomError("No substituents to enumerate; r_groups was empty")

        # A substituent that will not attach produces nothing, for every cut, in
        # silence -- the caller gets a shorter list and no reason for it. Since
        # a malformed substituent is the one mistake this interface still
        # permits, it is worth refusing up front and saying which.
        for group in groups:
            if group == "[H]":
                continue
            if Chem.MolFromSmiles("C" + group) is None:
                raise AevomError(
                    f"r_groups contains {group!r}, which will not attach to "
                    f"anything. {diagnose_smiles('C' + group)}"
                )
        if max_analogs <= 0:
            raise AevomError(f"max_analogs must be positive, got {max_analogs}")

        reference_fp = None
        if similarity_min is not None:
            if not 0.0 <= similarity_min <= 1.0:
                raise AevomError(
                    f"similarity_min must be between 0 and 1, got {similarity_min}"
                )
            reference_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)

        seen = {Chem.MolToSmiles(mol)}
        analogs: list = []

        def keep(candidate: Any) -> None:
            if candidate is None or len(analogs) >= max_analogs:
                return
            try:
                Chem.SanitizeMol(candidate)
            except Exception:
                return
            smiles = Chem.MolToSmiles(candidate)
            # A cut that re-attached nothing leaves two pieces; the brief always
            # wants one molecule.
            if smiles in seen or "." in smiles:
                return
            # A one-cut swap does not know what it is attaching to, so cutting
            # the O-CH3 of an anisole and re-attaching "F" produces an aryl
            # hypofluorite. 12% of analogues of four ordinary motifs came back
            # carrying a group like that, and no alert catalogue here rejects
            # them: an agent optimising ADMET will happily build SAR on numbers
            # the oracle returns for a molecule that cannot be put in a bottle.
            if not allow_unstable and unstable_group(candidate):
                seen.add(smiles)
                return
            if reference_fp is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(candidate, 2, 2048)
                if DataStructs.TanimotoSimilarity(reference_fp, fp) < similarity_min:
                    seen.add(smiles)
                    return
            seen.add(smiles)
            analogs.append(smiles)

        try:
            cuts = rdMMPA.FragmentMol(mol, maxCuts=1, resultsAsMols=False)
        except Exception as e:
            raise AevomError(f"Failed to fragment the molecule: {e}")

        for _, fragments in cuts:
            if not fragments or len(analogs) >= max_analogs:
                continue
            pieces = fragments.split(".")
            if len(pieces) != 2:
                continue
            # The larger piece is the scaffold; the smaller is what gets swapped.
            core = max(pieces, key=len)
            for group in groups:
                if len(analogs) >= max_analogs:
                    break
                keep(Chem.MolFromSmiles(core.replace("[*:1]", group)))

        if decorate:
            for group in groups:
                if group == "[H]" or len(analogs) >= max_analogs:
                    continue
                try:
                    reaction = AllChem.ReactionFromSmarts(f"[cH:1]>>[c:1]{group}")
                    products = reaction.RunReactants((mol,))
                except Exception:
                    continue
                for product in products:
                    keep(product[0])

        return analogs

    def sa_score(self, mol_or_smiles: Any) -> float:
        """Score how hard a molecule looks to synthesise, from 1 (easy) to 10.

        This is Ertl and Schuffenhauer's synthetic accessibility score, the same
        one ``molecule_gates`` reports, so a value computed here is directly
        comparable to that gate. Roughly, below 4 is unremarkable, above 6 is
        worth a second look, and above 7 usually means a strained or densely
        stereogenic scaffold.

        The scorer lives in RDKit's Contrib tree rather than the package proper,
        so it is reached by path -- which the sandbox cannot do itself, since
        ``sys`` is not importable there. Hence this helper.

        Args:
            mol_or_smiles: An RDKit Mol, or a SMILES string.

        Returns:
            The score, as a float.

        Raises:
            AevomError: If the input is not a valid molecule, or if RDKit's
                Contrib tree is missing from the image.

        Example:
            print(aevom.sa_score("CC(=O)Oc1ccccc1C(=O)O"))  # aspirin, ~1.6

            # Rank enumerated analogs by how tractable they look
            ranked = sorted(candidates, key=lambda s: aevom.sa_score(s))
        """
        try:
            from rdkit import Chem
        except ImportError:
            raise AevomError("RDKit is not available in this environment")

        if mol_or_smiles is None:
            raise AevomError("Cannot score None")

        if isinstance(mol_or_smiles, str):
            mol = Chem.MolFromSmiles(mol_or_smiles)
            if mol is None:
                raise AevomError(f"Invalid SMILES: {mol_or_smiles!r}")
        else:
            mol = mol_or_smiles

        scorer = _load_sascorer()
        if scorer is None:
            raise AevomError(
                "The synthetic accessibility scorer is unavailable: RDKit's "
                "Contrib tree is missing from this image."
            )

        try:
            return float(scorer.calculateScore(mol))
        except Exception as e:
            raise AevomError(f"Failed to compute the SA score: {e}")

    # =========================================================================
    # SAFE Notation - Fragment-based molecular representation
    # =========================================================================

    def safe_from_smiles(self, smiles: str, slicer: str = "brics") -> str:
        """Encode a SMILES string as SAFE fragment notation.

        SAFE rewrites the molecule so each fragment of a decomposition is a
        contiguous substring, separated by ``.``. Generative models trained on
        SAFE (GenMol, SAFE-GPT) need this ordering.

        When the chosen slicer finds no cut points, "recap", "mmpa", "hr" and
        "rotatable" are tried in turn. Only if none of them can decompose the
        molecule -- small or highly fused rings -- is the canonical SMILES
        returned as a single fragment. Such a string round-trips fine but cannot
        be decorated; see scaffold_to_safe().

        Args:
            smiles: SMILES string of the molecule.
            slicer: Preferred fragmentation algorithm. "brics" (the default) is
                what GenMol and SAFE-GPT were trained with; "hr", "rotatable",
                "recap", "mmpa" and "attach" are also supported.

        Returns:
            SAFE string.

        Raises:
            AevomError: If the SMILES is invalid or encoding fails outright.

        Example:
            safe_str = aevom.safe_from_smiles("CC(Cc1ccc(cc1)C(C(=O)O)C)C")
            print(safe_str)  # fragments separated by '.'
            ref = aevom.store_safe(safe_str, description="Ibuprofen SAFE")

        See Also:
            smiles_from_safe() - The inverse
            scaffold_to_safe() - Encode and append a GenMol generation mask
        """
        try:
            import safe as sf
        except ImportError:
            raise AevomError("safe-mol is not available in this environment")

        if not smiles or not smiles.strip():
            raise AevomError("SMILES string cannot be empty")

        smiles = smiles.strip()

        supported = ("brics", "hr", "rotatable", "recap", "mmpa", "attach")
        if slicer not in supported:
            raise AevomError(
                f"Unknown slicer '{slicer}'. Choose one of: {', '.join(supported)}."
            )

        # Parse first so an invalid SMILES is reported as such rather than
        # surfacing as an opaque encoder failure.
        mol = self.mol_from_smiles(smiles)

        # "attach" is not in the fallback chain: without explicit [*] dummy atoms
        # it cuts every C-H bond and emits each hydrogen as its own fragment.
        for candidate in [slicer] + [
            s for s in ("recap", "mmpa", "hr", "brics", "rotatable") if s != slicer
        ]:
            try:
                encoded = sf.encode(smiles, slicer=candidate)
            except Exception:
                continue
            if encoded and "[H]" not in encoded:
                return encoded

        from rdkit import Chem

        return Chem.MolToSmiles(mol)

    def smiles_from_safe(self, safe_string: str) -> str:
        """Decode a SAFE string back to canonical SMILES.

        Args:
            safe_string: SAFE fragment notation.

        Returns:
            Canonical SMILES string.

        Raises:
            AevomError: If the SAFE string cannot be decoded.

        Example:
            safe_str = aevom.load_safe("safe_abc123")
            smiles = aevom.smiles_from_safe(safe_str)
            ref = aevom.store_smiles(smiles, description="Decoded from SAFE")

        See Also:
            safe_from_smiles() - The inverse
        """
        try:
            import safe as sf
        except ImportError:
            raise AevomError("safe-mol is not available in this environment")

        if not safe_string or not safe_string.strip():
            raise AevomError("SAFE string cannot be empty")

        try:
            smiles = sf.decode(safe_string.strip(), canonical=True)
        except Exception as e:
            raise AevomError(f"Failed to decode SAFE string: {e}")

        if not smiles:
            raise AevomError(f"SAFE string decoded to nothing: '{safe_string}'")

        return smiles

    def scaffold_to_safe(self, smiles: str, frag_min: int, frag_max: int) -> str:
        """Encode a scaffold as SAFE with its terminal fragment masked for GenMol.

        ``[*{min-max}]`` asks the model to invent a fragment of between ``min``
        and ``max`` tokens. It has to *replace* a fragment rather than be appended
        to a complete one: SAFE fragments are bonded through shared ring-bond
        indices, so dropping the terminal fragment leaves those indices dangling
        and gives the mask something to bond to. A mask tacked onto a whole
        molecule is disconnected from it, and GenMol answers by echoing the input
        back unchanged.

        SAFE orders the core first, so the fragment that goes is a peripheral
        substituent. Wider mask ranges give more diversity; tighter ranges keep
        analog size predictable.

        Args:
            smiles: SMILES of the scaffold to decorate.
            frag_min: Minimum size of the generated fragment.
            frag_max: Maximum size of the generated fragment.

        Returns:
            Masked SAFE string, e.g. "n13cnc2c1c(=O)n4c(=O)n25.C3.C5.[*{5-12}]".

        Raises:
            AevomError: If the bounds are invalid, encoding fails, or the molecule
                cannot be fragmented and so has no attachment point to decorate.

        Example:
            # Scaffold decoration input for GenMol
            masked = aevom.scaffold_to_safe("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", 5, 12)
            ref = aevom.store_safe(masked, description="Caffeine scaffold, masked")

            # De novo generation needs no scaffold at all -- just the mask:
            ref = aevom.store_safe("[*{20-30}]", description="De novo")
        """
        if frag_min < 1:
            raise AevomError(f"frag_min must be at least 1, got {frag_min}")
        if frag_max < frag_min:
            raise AevomError(
                f"frag_max ({frag_max}) must be >= frag_min ({frag_min})"
            )

        fragments = self.safe_from_smiles(smiles).split(".")
        if len(fragments) < 2:
            raise AevomError(
                f"'{smiles}' has no fragmentation points, so there is no "
                f"attachment point for a generated fragment to bond to. Masking it "
                f"would make GenMol return this same molecule unchanged. Use a "
                f"larger scaffold, or generate de novo from a bare mask such as "
                f"'[*{{20-30}}]'."
            )

        return ".".join(fragments[:-1] + [f"[*{{{frag_min}-{frag_max}}}]"])

    # =========================================================================
    # Chemical Component Dictionary (CCD) Lookup
    # =========================================================================

    def lookup_ccd_code(self, molecule_name: str) -> Optional[str]:
        """Look up the PDB Chemical Component Dictionary (CCD) code for a molecule.

        This function searches PubChem for the molecule to get its InChIKey,
        then queries the RCSB PDB to find the corresponding CCD code.

        CCD codes are used with OpenFold3's `ccd_codes` parameter for ligand
        structure prediction.

        Args:
            molecule_name: Common name of the molecule (e.g., "imatinib", "aspirin").

        Returns:
            The CCD code (e.g., "STI" for imatinib) or None if not found.

        Raises:
            AevomError: If the lookup fails due to network or API errors.

        Example:
            ccd = aevom.lookup_ccd_code("imatinib")
            print(f"CCD code: {ccd}")  # Output: CCD code: STI

            # Use with OpenFold3:
            # openfold3(molecules=[
            #     {"type": "protein", "sequence_ref": "sequence_abc123"},
            #     {"type": "ligand", "ccd_codes": ccd}
            # ])
        """
        if not molecule_name or not molecule_name.strip():
            raise AevomError("Molecule name cannot be empty")

        molecule_name = molecule_name.strip()

        # Step 1: Search PubChem for the molecule to get InChIKey
        try:
            # Use pubchempy directly (it's allowed in sandbox)
            import pubchempy as pcp

            compounds = pcp.get_compounds(molecule_name, "name")
            if not compounds:
                return None

            inchi_key = compounds[0].inchikey
            if not inchi_key:
                return None

        except Exception as e:
            raise AevomError(f"Failed to search PubChem for '{molecule_name}': {e}")

        # Step 2: Search RCSB for CCD code using InChIKey
        rcsb_search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

        search_payload = {
            "query": {
                "type": "terminal",
                "service": "text_chem",
                "parameters": {
                    "attribute": "rcsb_chem_comp_descriptor.InChIKey",
                    "operator": "exact_match",
                    "value": inchi_key,
                },
            },
            "return_type": "mol_definition",
        }

        headers = {
            "Content-Type": "application/json",
        }

        data = json.dumps(search_payload).encode("utf-8")
        req = urllib.request.Request(
            rcsb_search_url, data=data, headers=headers, method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))

                result_set = result.get("result_set", [])
                if result_set:
                    # Return the first (best match) CCD code
                    return result_set[0].get("identifier")

                return None

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass
            raise AevomError(
                f"Failed to query RCSB for CCD code: HTTP {e.code}. {error_body}"
            )
        except urllib.error.URLError as e:
            raise AevomError(f"Network error querying RCSB: {e}")

    # =========================================================================
    # PubChem Search and Download
    # =========================================================================

    def search_pubchem(self, name: str) -> Optional[dict[str, Any]]:
        """Search PubChem for a molecule by name.

        Use this to find molecules in the PubChem database by their common
        or IUPAC name. Returns identifiers including SMILES that can be used
        for visualization or further analysis.

        Args:
            name: Common or IUPAC name of the molecule (e.g., "Aspirin",
                  "Tovorafenib", "caffeine").

        Returns:
            Dictionary containing molecule identifiers:
            - cid: PubChem Compound ID (int)
            - name: IUPAC name from PubChem (str)
            - isomeric_smiles: Isomeric SMILES string (str)
            - inchi: InChI identifier (str)
            - inchi_key: InChIKey hash (str)

            Returns None if not found.

        Raises:
            AevomError: If an API error occurs.

        Example:
            result = aevom.search_pubchem("Ibuprofen")
            if result:
                print(f"CID: {result['cid']}")
                print(f"SMILES: {result['isomeric_smiles']}")

                # Visualize it
                sdf = aevom.download_pubchem_sdf(result['cid'])
                aevom.visualize(sdf, label="Ibuprofen")
        """
        if not name or not name.strip():
            raise AevomError("Molecule name cannot be empty")

        name = name.strip()

        try:
            import pubchempy as pcp

            compounds = pcp.get_compounds(name, "name")

            if not compounds:
                return None

            compound = compounds[0]

            # Get SMILES (try 'smiles' first, fallback to 'isomeric_smiles')
            smiles_value = getattr(compound, "smiles", None) or compound.isomeric_smiles

            return {
                "cid": compound.cid,
                "name": compound.iupac_name or name,
                "isomeric_smiles": smiles_value,
                "inchi": compound.inchi,
                "inchi_key": compound.inchikey,
            }

        except Exception as e:
            raise AevomError(f"Error searching PubChem for '{name}': {e}")

    def get_pubchem_properties(self, cid: int) -> Optional[dict[str, Any]]:
        """Get molecular properties from PubChem by Compound ID.

        Retrieves calculated molecular properties for a compound. Use this
        after search_pubchem() to get detailed property information.

        Args:
            cid: PubChem Compound ID (from search_pubchem result).

        Returns:
            Dictionary containing molecular properties:
            - molecular_weight: Molecular weight in g/mol (float)
            - molecular_formula: Molecular formula string (str)
            - xlogp: Computed XLogP value (float or None)
            - tpsa: Topological Polar Surface Area in Ų (float or None)
            - charge: Formal charge (int)
            - exact_mass: Exact mass in g/mol (float)
            - complexity: Molecular complexity score (float)
            - h_bond_donor_count: Number of H-bond donors (int)
            - h_bond_acceptor_count: Number of H-bond acceptors (int)
            - rotatable_bond_count: Number of rotatable bonds (int)
            - heavy_atom_count: Number of heavy (non-hydrogen) atoms (int)

            Returns None if CID not found.

        Raises:
            AevomError: If an API error occurs.

        Example:
            result = aevom.search_pubchem("Aspirin")
            props = aevom.get_pubchem_properties(result['cid'])
            print(f"MW: {props['molecular_weight']} g/mol")
            print(f"H-bond donors: {props['h_bond_donor_count']}")
        """
        if not isinstance(cid, int) or cid <= 0:
            raise AevomError(f"CID must be a positive integer, got: {cid}")

        try:
            import pubchempy as pcp

            compound = pcp.Compound.from_cid(cid)

            if compound is None:
                return None

            return {
                "molecular_weight": compound.molecular_weight,
                "molecular_formula": compound.molecular_formula,
                "xlogp": compound.xlogp,
                "tpsa": compound.tpsa,
                "charge": compound.charge or 0,
                "exact_mass": compound.exact_mass,
                "complexity": compound.complexity,
                "h_bond_donor_count": compound.h_bond_donor_count,
                "h_bond_acceptor_count": compound.h_bond_acceptor_count,
                "rotatable_bond_count": compound.rotatable_bond_count,
                "heavy_atom_count": compound.heavy_atom_count,
            }

        except Exception as e:
            raise AevomError(f"Error getting PubChem properties for CID {cid}: {e}")

    def download_pubchem_sdf(self, cid: int, name: Optional[str] = None) -> str:
        """Download 3D structure from PubChem and store it, returning a reference.

        Downloads the molecular structure in SDF format from PubChem,
        automatically stores it in Aevom's data store, and returns a
        reference ID that can be used with other tools (e.g., vina_dock).

        Attempts to get the 3D conformer first; falls back to 2D if
        3D is not available.

        Args:
            cid: PubChem Compound ID (from search_pubchem result).
            name: Optional name for the molecule (used in metadata).

        Returns:
            Reference ID (e.g., "sdf_abc12345") that can be passed to:
            - vina_dock(ligand_sdf_ref=...)
            - visualize_3d(sdf_ref=...)
            - load_sdf() or load_rdkit_mol()

        Raises:
            AevomError: If download fails or structure not available.

        Example:
            result = aevom.search_pubchem("Caffeine")
            sdf_ref = aevom.download_pubchem_sdf(result['cid'])
            print(f"Stored as: {sdf_ref}")  # e.g., "sdf_abc12345"

            # The reference can now be used directly with tools:
            # vina_dock(ligand_sdf_ref=sdf_ref, ...)

            # Or load for analysis:
            mol = aevom.load_rdkit_mol(sdf_ref)
            print(f"Atoms: {mol.GetNumAtoms()}")
        """
        if not isinstance(cid, int) or cid <= 0:
            raise AevomError(f"CID must be a positive integer, got: {cid}")

        sdf_content = None

        # Try 3D conformer first
        url_3d = (
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
            f"/record/SDF?record_type=3d"
        )

        try:
            req = urllib.request.Request(url_3d)
            with urllib.request.urlopen(req, timeout=30) as response:
                sdf_content = response.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            if e.code != 404:
                # Unexpected error
                raise AevomError(f"Failed to download 3D SDF for CID {cid}: HTTP {e.code}")
            # 404 means 3D not available, fall back to 2D

        # Fall back to 2D if 3D not available
        if sdf_content is None:
            url_2d = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
                f"/record/SDF"
            )

            try:
                req = urllib.request.Request(url_2d)
                with urllib.request.urlopen(req, timeout=30) as response:
                    sdf_content = response.read().decode("utf-8")
            except urllib.error.HTTPError as e:
                raise AevomError(
                    f"Failed to download SDF for CID {cid}: HTTP {e.code}. "
                    "Structure may not be available."
                )
            except urllib.error.URLError as e:
                raise AevomError(f"Network error downloading SDF for CID {cid}: {e}")

        # Store the SDF and return the reference
        metadata = {"pubchem_cid": cid, "source": "pubchem"}
        if name:
            metadata["name"] = name

        return self.store_sdf(sdf_content, metadata, description=f"PubChem CID {cid}")


# Singleton instance that gets configured when sandbox is initialized
_client: Optional[AevomClient] = None


def get_client() -> Optional[AevomClient]:
    """Get the global Aevom client instance."""
    return _client


def set_client(client: AevomClient) -> None:
    """Set the global Aevom client instance."""
    global _client
    _client = client


def create_client(
    session_token: str,
    backend_url: str,
    allowed_refs: Optional[set[str]] = None,
) -> AevomClient:
    """Create and set the global Aevom client.

    Args:
        session_token: Session token for backend authentication.
        backend_url: Base URL of the backend API.
        allowed_refs: Optional set of allowed reference IDs.

    Returns:
        Configured AevomClient instance.
    """
    client = AevomClient(session_token, backend_url, allowed_refs)
    set_client(client)
    return client
