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

"""Drug-likeness gates: the full pass/fail verdict on a molecule.

One call answers "is this thing a plausible drug candidate?" -- the physical
property panel, synthetic accessibility, structural alerts, and similarity to a
reference. Aevom could previously compute molecular weight and logP one molecule
at a time via ``molecule_info`` and nothing else, which meant every drug-likeness
judgment was the model eyeballing two numbers against remembered heuristics.

Three design decisions worth knowing before editing:

**Thresholds are data, not code.** Every bound lives in ``DEFAULT_THRESHOLDS``
and every one can be overridden per call. The defaults are SMDD-Bench's
hard-constraint panel (arXiv 2605.21740, Table 6), adopted because they are
published and defensible rather than invented here; a programme with different
constraints passes its own.

**Similarity gates are opt-in.** Lead optimization wants a candidate close to
its parent; scaffold hopping wants one deliberately far from it. The same number
is a pass or a fail depending on the task, so there is no honest default -- pass
``reference_smiles`` plus a bound and the gate appears.

**PAINS gates, Brenk and NIH report.** A PAINS substructure means the assay
result is probably an artifact, which disqualifies a hit outright. Brenk and NIH
are advisory by construction -- Brenk flags nitro and sulfonyl groups that
plenty of approved drugs carry -- so they are counted and named but do not fail a
molecule unless the caller asks them to.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Optional

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, QED, rdFingerprintGenerator, rdFMCS, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold

logger = logging.getLogger(__name__)

# RDKit narrates every unparseable SMILES to stderr. Callers get a structured
# invalid_smiles list instead, so the console noise is pure cost.
RDLogger.DisableLog("rdApp.error")

# SMDD-Bench Table 6. Every bound is overridable per call; these are the
# defaults because they are published, not because they are universal.
#
# Comparisons use the RAW measured value. Rounding first silently widens every
# bound by half a display unit -- a true SA of 4.502 displays as 4.50 and
# clears a "< 4.5" bar it should fail -- and the published panel this mirrors
# rounds only for its printout while comparing the exact figure. So the values
# below are gated exactly and rounded only on the way out.
DEFAULT_THRESHOLDS: dict[str, float] = {
    "mw_max": 600.0,
    "logp_min": -1.0,
    "logp_max": 5.0,
    "tpsa_max": 140.0,
    "hbd_max": 5,
    "hba_max": 10,
    "rotatable_bonds_max": 10,
    "formal_charge_min": -2,
    "formal_charge_max": 2,
    "sa_score_max": 4.5,
    "pains_max": 0,
    # Brenk sat among the optional bounds, justified by a comment saying those
    # "only make sense against a reference molecule". That is true of
    # similarity and false of a structural-alert catalogue, which needs no
    # reference at all -- it was grouped there by accident. The panel it
    # mirrors applies Brenk unconditionally, and a molecule carrying a
    # reactive-group alert is not drug-like whatever you compare it to.
    "brenk_max": 0,
}

# Bounds with no default, because they are measured AGAINST a reference and
# which side you want depends on the task: lead optimisation wants a candidate
# close to its parent, scaffold hopping wants one deliberately far from it.
# There is no default that is right for both, so these stay opt-in -- and the
# tool description has to say so, since a panel that silently omits a bound the
# caller assumed was applied is the failure this module is most prone to.
OPTIONAL_THRESHOLDS: tuple[str, ...] = (
    "similarity_min",
    "similarity_max",
    "scaffold_similarity_min",
    "scaffold_similarity_max",
    "scaffold_mcs_ratio_min",
    "scaffold_mcs_ratio_max",
    "nih_max",
)

THRESHOLD_NAMES: tuple[str, ...] = tuple(DEFAULT_THRESHOLDS) + OPTIONAL_THRESHOLDS

# Morgan radius 2 over 2048 bits -- the ECFP4-equivalent that similarity numbers
# in the medicinal chemistry literature are quoted against. Changing either
# makes our Tanimoto values incomparable to published ones.
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# A maximum common substructure search is worst-case exponential. Scaffolds are
# small enough that this bound is never reached in practice; it exists so one
# pathological input cannot hang the request.
MCS_TIMEOUT_SECONDS = 2

# Beyond this the request is better split. Descriptors are microseconds, but the
# per-molecule MCS against the reference is the term that scales badly.
MAX_MOLECULES = 256

_morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
)

_filter_catalogs: dict[str, FilterCatalog] = {}
_sascorer: Any = None
_sascorer_loaded = False


def _catalog(name: str) -> Optional[FilterCatalog]:
    """Return a lazily built structural-alert catalog, or None if unavailable.

    Catalog construction parses a bundled substructure library, so it happens
    once per process rather than once per molecule.
    """
    if name in _filter_catalogs:
        return _filter_catalogs[name]

    catalogs = {
        "pains": FilterCatalogParams.FilterCatalogs.PAINS,
        "brenk": FilterCatalogParams.FilterCatalogs.BRENK,
        "nih": FilterCatalogParams.FilterCatalogs.NIH,
    }
    if name not in catalogs:
        return None

    try:
        params = FilterCatalogParams()
        params.AddCatalog(catalogs[name])
        _filter_catalogs[name] = FilterCatalog(params)
    except Exception:
        logger.exception("Failed to build the %s filter catalog", name)
        return None
    return _filter_catalogs[name]


def _load_sascorer() -> Any:
    """Import RDKit's contributed synthetic accessibility scorer.

    ``sascorer`` ships in RDKit's Contrib tree rather than the package proper,
    so it is reached by path. Returns None when the Contrib tree is missing,
    which turns SA score into a reported gap rather than a failed request.
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
        logger.warning(
            "sascorer is unavailable (RDKit Contrib tree not found); "
            "synthetic accessibility will be omitted"
        )
        _sascorer = None
    return _sascorer


@dataclass
class Gate:
    """One threshold applied to one measured value."""

    name: str
    label: str
    value: Optional[float]
    bound: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        # The verdict was decided on the exact value; this is the printout.
        value = self.value
        if isinstance(value, float) and not value.is_integer():
            value = round(value, 4)
        return {
            "name": self.name,
            "label": self.label,
            "value": value,
            "bound": self.bound,
            "passed": self.passed,
            "detail": self.detail,
        }


@dataclass
class Reference:
    """The molecule that similarity gates are measured against."""

    smiles: str
    mol: Any
    fingerprint: Any
    scaffold: Optional[Any]
    scaffold_fingerprint: Optional[Any]
    scaffold_smiles: str = ""


def resolve_thresholds(overrides: Optional[dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    """Merge caller overrides over the defaults.

    A threshold set to None disables that gate, which is how a caller says "do
    not apply the molecular weight ceiling" without inventing an infinity.

    Returns:
        ``(thresholds, unknown)`` -- unknown names are reported rather than
        silently dropped, since a typo'd threshold would otherwise look like a
        gate that quietly did not run.
    """
    thresholds: dict[str, Any] = dict(DEFAULT_THRESHOLDS)
    unknown: list[str] = []

    for key, value in (overrides or {}).items():
        if key not in THRESHOLD_NAMES:
            unknown.append(key)
            continue
        if value is None:
            thresholds.pop(key, None)
        else:
            thresholds[key] = value

    return thresholds, unknown


def _murcko(mol: Any) -> tuple[Optional[Any], str]:
    """Return the Bemis-Murcko scaffold and its SMILES.

    An acyclic molecule has no scaffold. That is a real answer, not an error --
    scaffold comparisons against it are reported as unavailable.
    """
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    except Exception:
        return None, ""
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return None, ""
    return scaffold, Chem.MolToSmiles(scaffold)


def _build_reference(smiles: str) -> tuple[Optional[Reference], Optional[str]]:
    """Parse the reference molecule once, for reuse across the batch.

    Accepts a molfile as well as SMILES. The caller may be handing over a stored
    reference ligand verbatim rather than a transcription of one, which is the
    point -- re-typing the parent is how a run ends up measuring every
    similarity against a molecule that is not the parent.
    """
    smiles = (smiles or "").strip()
    if not smiles:
        return None, None

    if "V2000" in smiles or "V3000" in smiles or "$$$$" in smiles:
        block = smiles.split("$$$$")[0]
        mol = Chem.MolFromMolBlock(block)
        if mol is None:
            return None, "Could not read the reference molfile."
        smiles = Chem.MolToSmiles(mol)
    else:
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, f"Invalid reference_smiles: '{smiles}'"

    scaffold, scaffold_smiles = _murcko(mol)
    return (
        Reference(
            smiles=smiles,
            mol=mol,
            fingerprint=_morgan_generator.GetFingerprint(mol),
            scaffold=scaffold,
            scaffold_fingerprint=(
                _morgan_generator.GetFingerprint(scaffold) if scaffold is not None else None
            ),
            scaffold_smiles=scaffold_smiles,
        ),
        None,
    )


def _mcs_ratio(query: Any, reference: Any) -> Optional[float]:
    """Fraction of the larger scaffold covered by the two scaffolds' MCS.

    Normalizing by the larger of the two keeps the measure symmetric and makes a
    small scaffold buried inside a large one score low, which is the behaviour
    scaffold-hop novelty checks want: a fragment of the original is not a hop.
    """
    if query is None or reference is None:
        return None

    try:
        result = rdFMCS.FindMCS(
            [query, reference],
            timeout=MCS_TIMEOUT_SECONDS,
            ringMatchesRingOnly=True,
            matchValences=False,
        )
    except Exception:
        logger.exception("MCS search failed")
        return None

    if result is None or result.numAtoms == 0:
        return 0.0

    largest = max(query.GetNumAtoms(), reference.GetNumAtoms())
    if largest == 0:
        return None
    return round(result.numAtoms / largest, 3)


# Groups that do not survive contact with water. Reported, never gated.
#
# The twelve hard constraints this service reproduces are the grader's, checked
# exact against it on 100 molecules; adding a thirteenth would fail molecules the
# grader passes, which is the same silent disagreement that made this panel worth
# fixing in the first place. But an aryl hypofluorite is not a lead, and nothing
# else in the stack says so: PAINS, Brenk and NIH pass every one of them. So this
# goes beside the verdict rather than into it.
#
# The list is short deliberately. Hydrazides, hydroxylamines and cyclic
# endoperoxides are real drug chemistry -- isoniazid, artemisinin -- and are not
# here. What is here cannot be isolated.
_UNSTABLE_GROUPS = (
    ("hypohalite (O-halogen)", "[OX2]-[F,Cl,Br,I]"),
    ("sulfenyl halide (S-halogen)", "[SX2]-[F,Cl,Br,I]"),
    ("N-halamine (N-halogen)", "[NX3]-[F,Cl,Br,I]"),
    ("acyl halide", "[CX3](=O)-[F,Cl,Br,I]"),
    ("acyclic peroxide", "[OX2R0]-[OX2R0]"),
    ("anhydride", "[CX3](=O)-[OX2]-[CX3]=O"),
)

_UNSTABLE_PATTERNS: list = []


def _unstable(mol: Any) -> Optional[str]:
    """Name the non-isolable group in *mol*, if it has one."""
    global _UNSTABLE_PATTERNS
    if not _UNSTABLE_PATTERNS:
        _UNSTABLE_PATTERNS = [
            (name, Chem.MolFromSmarts(smarts)) for name, smarts in _UNSTABLE_GROUPS
        ]
    for name, pattern in _UNSTABLE_PATTERNS:
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return name
    return None


def _alerts(mol: Any) -> dict[str, list[str]]:
    """Name every structural alert the molecule trips, by catalog."""
    found: dict[str, list[str]] = {}
    for name in ("pains", "brenk", "nih"):
        catalog = _catalog(name)
        if catalog is None:
            continue
        try:
            matches = catalog.GetMatches(mol)
        except Exception:
            logger.exception("Structural alert matching failed for %s", name)
            continue
        found[name] = sorted({match.GetDescription() for match in matches})
    return found


def _descriptors(mol: Any) -> dict[str, Any]:
    """The physical property panel the gates are applied to."""
    scorer = _load_sascorer()
    sa_score = None
    if scorer is not None:
        try:
            sa_score = scorer.calculateScore(mol)
        except Exception:
            logger.exception("SA score calculation failed")

    properties: dict[str, Any] = {
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
        "sa_score": sa_score,
        "num_rings": rdMolDescriptors.CalcNumRings(mol),
        "num_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "num_stereocenters": rdMolDescriptors.CalcNumAtomStereoCenters(
            Chem.AddHs(mol)
        )
        if mol.GetNumAtoms() < 200
        else None,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "molecular_formula": rdMolDescriptors.CalcMolFormula(mol),
    }

    # QED is not a gate -- it is a single number that summarizes drug-likeness
    # and is useful for ranking molecules that all pass.
    try:
        properties["qed"] = QED.qed(mol)
    except Exception:
        properties["qed"] = None

    return properties


# (gate name, threshold key, comparison, property key, human label)
# Three of these are stated as strict inequalities in the panel this mirrors --
# "< 600 Da", "< 140 A^2", "SA < 4.5" -- and the count bounds are stated as
# "<= 5", "<= 10". Treating them all as inclusive let a molecule sitting exactly
# on a strict bound pass a gate the published check fails, which is a wrong
# answer delivered quietly. The flag says which is which.
_UPPER_BOUNDS: tuple[tuple[str, str, str, str, bool], ...] = (
    ("mw", "mw_max", "mw", "molecular weight", True),
    ("tpsa", "tpsa_max", "tpsa", "TPSA", True),
    ("hbd", "hbd_max", "hbd", "H-bond donors", False),
    ("hba", "hba_max", "hba", "H-bond acceptors", False),
    ("rotatable_bonds", "rotatable_bonds_max", "rotatable_bonds", "rotatable bonds", False),
    ("sa_score", "sa_score_max", "sa_score", "synthetic accessibility", True),
)


def _range_gate(
    name: str,
    label: str,
    value: Optional[float],
    low: Optional[float],
    high: Optional[float],
    exclusive_high: bool = False,
) -> Optional[Gate]:
    """Build a gate for a value that must sit inside a range.

    ``exclusive_high`` makes the upper bound strict, for the bounds published
    as "< x" rather than "<= x". The rendered bound says which, so a reader can
    tell a value sitting exactly on the line why it was refused.
    """
    if low is None and high is None:
        return None
    high_symbol = "<" if exclusive_high else "<="
    if low is not None and high is not None:
        bound = f"{_num(low)} to {_num(high)}"
    elif high is not None:
        bound = f"{high_symbol} {_num(high)}"
    else:
        bound = f">= {_num(low)}"

    if value is None:
        return Gate(name, label, None, bound, False, "not computed")

    under_high = True
    if high is not None:
        under_high = value < high if exclusive_high else value <= high
    passed = (low is None or value >= low) and under_high
    return Gate(name, label, value, bound, passed)


# How many decimals each measurement is worth showing. Display only -- the
# gate above has already compared the exact value.
_DISPLAY_PRECISION: dict[str, int] = {
    "mw": 2, "logp": 2, "tpsa": 2, "sa_score": 2, "qed": 3,
}


def _for_display(properties: dict[str, Any]) -> dict[str, Any]:
    """The descriptor panel, rounded for a reader."""
    shown = dict(properties)
    for name, places in _DISPLAY_PRECISION.items():
        value = shown.get(name)
        if isinstance(value, float):
            shown[name] = round(value, places)
    return shown


def _num(value: Any) -> str:
    """Render a threshold or measurement without a trailing .0, briefly."""
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(round(value, 4))
    return str(value)


def _evaluate_gates(
    properties: dict[str, Any],
    alerts: dict[str, list[str]],
    similarity: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[Gate]:
    """Apply every configured threshold, in a stable reporting order."""
    gates: list[Gate] = []

    for name, key, prop, label, exclusive in _UPPER_BOUNDS:
        if key not in thresholds:
            continue
        gate = _range_gate(
            name, label, properties.get(prop), None, thresholds[key], exclusive
        )
        if gate is not None:
            gates.append(gate)

    logp = _range_gate(
        "logp",
        "logP",
        properties.get("logp"),
        thresholds.get("logp_min"),
        thresholds.get("logp_max"),
    )
    if logp is not None:
        # logP is a window, so it belongs next to molecular weight rather than
        # at the end where the optional gates live.
        gates.insert(1, logp)

    charge = _range_gate(
        "formal_charge",
        "formal charge",
        properties.get("formal_charge"),
        thresholds.get("formal_charge_min"),
        thresholds.get("formal_charge_max"),
    )
    if charge is not None:
        gates.append(charge)

    for catalog, key, label in (
        ("pains", "pains_max", "PAINS alerts"),
        ("brenk", "brenk_max", "Brenk alerts"),
        ("nih", "nih_max", "NIH alerts"),
    ):
        if key not in thresholds or catalog not in alerts:
            continue
        hits = alerts[catalog]
        gate = _range_gate(catalog, label, len(hits), None, thresholds[key])
        if gate is not None:
            if hits and not gate.passed:
                gate.detail = ", ".join(hits[:3]) + (
                    f" (and {len(hits) - 3} more)" if len(hits) > 3 else ""
                )
            gates.append(gate)

    for name, low_key, high_key, prop, label in (
        (
            "similarity",
            "similarity_min",
            "similarity_max",
            "tanimoto",
            "Tanimoto to reference",
        ),
        (
            "scaffold_similarity",
            "scaffold_similarity_min",
            "scaffold_similarity_max",
            "scaffold_tanimoto",
            "scaffold Tanimoto",
        ),
        (
            "scaffold_mcs_ratio",
            "scaffold_mcs_ratio_min",
            "scaffold_mcs_ratio_max",
            "scaffold_mcs_ratio",
            "scaffold MCS ratio",
        ),
    ):
        gate = _range_gate(
            name,
            label,
            similarity.get(prop),
            thresholds.get(low_key),
            thresholds.get(high_key),
        )
        if gate is not None:
            if gate.value is None:
                gate.detail = "no reference molecule, or no scaffold to compare"
            gates.append(gate)

    return gates


def evaluate_molecule(
    smiles: str,
    thresholds: dict[str, Any],
    reference: Optional[Reference],
) -> Optional[dict[str, Any]]:
    """Score one molecule against the thresholds.

    Returns None when the SMILES cannot be parsed, so the caller can list it as
    invalid rather than fabricating a verdict.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    properties = _descriptors(mol)
    alerts = _alerts(mol)
    unstable = _unstable(mol)
    scaffold, scaffold_smiles = _murcko(mol)

    similarity: dict[str, Any] = {"murcko_scaffold": scaffold_smiles}
    if reference is not None:
        fingerprint = _morgan_generator.GetFingerprint(mol)
        similarity["tanimoto"] = round(
            DataStructs.TanimotoSimilarity(fingerprint, reference.fingerprint), 3
        )
        if scaffold is not None and reference.scaffold_fingerprint is not None:
            similarity["scaffold_tanimoto"] = round(
                DataStructs.TanimotoSimilarity(
                    _morgan_generator.GetFingerprint(scaffold),
                    reference.scaffold_fingerprint,
                ),
                3,
            )
        similarity["scaffold_mcs_ratio"] = _mcs_ratio(scaffold, reference.scaffold)
        similarity["same_scaffold"] = bool(
            scaffold_smiles and scaffold_smiles == reference.scaffold_smiles
        )

    gates = _evaluate_gates(properties, alerts, similarity, thresholds)
    failures = [gate for gate in gates if not gate.passed]

    return {
        "smiles": smiles,
        "canonical_smiles": Chem.MolToSmiles(mol),
        "properties": _for_display(properties),
        "alerts": alerts,
        # Not a gate: `passed` stays the grader's verdict exactly.
        "unstable_group": unstable,
        "similarity": similarity,
        "gates": [gate.to_dict() for gate in gates],
        "passed": not failures,
        "num_failed": len(failures),
        "failures": [
            f"{gate.label} {_num(gate.value) if gate.value is not None else '?'} "
            f"violates {gate.bound}"
            + (f" [{gate.detail}]" if gate.detail else "")
            for gate in failures
        ],
    }


@dataclass
class GateReport:
    """Everything one gates request produced."""

    results: list[dict[str, Any]] = field(default_factory=list)
    invalid_smiles: list[str] = field(default_factory=list)
    thresholds: dict[str, Any] = field(default_factory=dict)
    unknown_thresholds: list[str] = field(default_factory=list)
    reference_smiles: str = ""
    reference_scaffold: str = ""


def evaluate(
    smiles_list: list[str],
    thresholds: Optional[dict[str, Any]] = None,
    reference_smiles: str = "",
) -> tuple[Optional[GateReport], Optional[str]]:
    """Score a batch of molecules against the drug-likeness panel.

    Args:
        smiles_list: Molecules to score. Duplicates are kept, since a caller
            passing the same SMILES twice is usually indexing into their own
            list and expects one row per input.
        thresholds: Overrides for DEFAULT_THRESHOLDS. None values disable a gate.
        reference_smiles: Molecule that similarity gates measure against.

    Returns:
        ``(report, error)``. The error is set only for whole-request failures --
        an empty batch, too many molecules, an unparseable reference.
    """
    if not smiles_list:
        return None, "No SMILES provided"
    if len(smiles_list) > MAX_MOLECULES:
        return None, f"{len(smiles_list)} molecules exceeds the {MAX_MOLECULES} per-request limit"

    resolved, unknown = resolve_thresholds(thresholds)

    reference, error = _build_reference(reference_smiles)
    if error:
        return None, error

    # Similarity bounds without a reference would silently never run, and the
    # caller would read the resulting all-pass verdict as meaningful.
    if reference is None:
        needs_reference = [
            key
            for key in resolved
            if key.startswith(("similarity", "scaffold_"))
        ]
        if needs_reference:
            return None, (
                f"{', '.join(sorted(needs_reference))} requires reference_smiles"
            )

    report = GateReport(
        thresholds=resolved,
        unknown_thresholds=unknown,
        reference_smiles=reference.smiles if reference else "",
        reference_scaffold=reference.scaffold_smiles if reference else "",
    )

    for entry in smiles_list:
        entry = (entry or "").strip()
        if not entry:
            continue
        result = evaluate_molecule(entry, resolved, reference)
        if result is None:
            report.invalid_smiles.append(entry)
        else:
            report.results.append(result)

    return report, None
