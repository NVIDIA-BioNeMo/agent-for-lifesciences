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

"""Protein-ligand interaction profiling with PLIP.

What this adds over a distance calculation is chemistry. PLIP applies published
geometric criteria per interaction family -- angle windows for hydrogen bonds,
ring-plane offsets for pi-stacking, protonation-aware charge assignment for salt
bridges -- and reimplementing those rules badly is the main risk this module
exists to avoid. So it wraps PLIP rather than measuring anything itself.

Two ideas carry most of the value:

**Coarsening.** A raw PLIP row names atom indices, and atom indices differ
between two poses of the same ligand even when the interaction is the same. So
each interaction is reduced to ``(chain, residue number, residue name, broad
type)``. That tuple is stable across poses, which is what makes a set of them
comparable -- and comparability is the whole point, because the question is
almost never "what does this pose do" but "what do these poses agree on".

**Ligand selection is explicit.** A crystal structure carries glycans, buffer
components and ions alongside the ligand anyone cares about, and PLIP reports
every one of them as a binding site. Profiling all of them and letting the caller
sort it out invites an answer about the wrong molecule, so this module reports
every ligand, designates one, and always says which it designated and why.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# PLIP's interaction families, in the order they are reported. These are the
# attribute stems on BindingSiteReport: each has a `_features` tuple of column
# names and an `_info` list of rows.
FAMILIES: tuple[str, ...] = (
    "hydrophobic",
    "hbond",
    "waterbridge",
    "saltbridge",
    "pistacking",
    "pication",
    "halogen",
    "metal",
)

# Human labels, used in rendered output and legends.
FAMILY_LABELS: dict[str, str] = {
    "hydrophobic": "hydrophobic contact",
    "hbond": "hydrogen bond",
    "waterbridge": "water-bridged hydrogen bond",
    "saltbridge": "salt bridge",
    "pistacking": "pi-stacking",
    "pication": "pi-cation",
    "halogen": "halogen bond",
    "metal": "metal coordination",
}

# The columns every family reports first, which is what makes coarsening
# possible without special-casing each family's row layout.
_RESIDUE_COLUMNS = ("RESNR", "RESTYPE", "RESCHAIN")

# Columns dropped from the reported detail: atom indices are meaningless outside
# the file they came from, and coordinates are what the viewer is for.
_NOISE_COLUMNS = frozenset(
    {
        "RESNR",
        "RESTYPE",
        "RESCHAIN",
        "RESNR_LIG",
        "RESTYPE_LIG",
        "RESCHAIN_LIG",
        "LIGCOO",
        "PROTCOO",
        "WATERCOO",
        "METALCOO",
        "TARGETCOO",
        "LIGCARBONIDX",
        "PROTCARBONIDX",
        "DONORIDX",
        "ACCEPTORIDX",
        "DONOR_IDX",
        "ACCEPTOR_IDX",
        "WATER_IDX",
        "DON_IDX",
        "ACC_IDX",
        "PROT_IDX_LIST",
        "LIG_IDX_LIST",
        "METAL_IDX",
        "TARGET_IDX",
        "COMPLEXNUM",
    }
)

# Residue names that mean "the molecule we just placed here" rather than a
# specific compound. Docking and co-folding write these: Vina and Meeko default
# to UNL, and generated ligands commonly land as LIG or MOL.
#
# UNL is on PLIP's irrelevance list, and rightly so for a deposited crystal
# structure, where "unknown ligand" is usually junk density. In a docked complex
# it is the entire point of the calculation, so demoting it would rank a
# crystallographic cofactor above the pose that was just computed.
PIPELINE_LIGAND_NAMES = frozenset({"UNL", "LIG", "MOL", "UNK", "LG1", "DRG"})


def biolip_artifacts() -> frozenset[str]:
    """Residue names PLIP considers biologically irrelevant.

    PLIP ships the BioLiP curation of crystallization additives, cryoprotectants,
    buffers and glycans -- 400-odd names, NAG among them -- and logs "may be
    biologically irrelevant" when it profiles one. Reading its list rather than
    keeping a copy here means the judgment stays consistent with the tool making
    it, and does not go stale. These ligands are still reported; they are just
    never auto-designated while something else is present.
    """
    try:
        from plip.basic import config

        names = frozenset(str(entry).upper() for entry in config.biolip_list)
    except Exception:
        # An upstream rename would otherwise make every glycan look like the
        # ligand of interest. Designation falls back to interaction count.
        logger.warning("PLIP's biolip_list is unavailable; ligand ranking will ignore it")
        return frozenset()

    return names - PIPELINE_LIGAND_NAMES


@dataclass
class Interaction:
    """One interaction, coarsened to a residue and a family."""

    family: str
    chain: str
    residue_number: int
    residue_name: str
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """The pose-independent identity used for fingerprints and consensus."""
        return f"{self.chain}:{self.residue_number}:{self.residue_name}:{self.family}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "label": FAMILY_LABELS.get(self.family, self.family),
            "chain": self.chain,
            "residue_number": self.residue_number,
            "residue_name": self.residue_name,
            "key": self.key,
            "detail": self.detail,
        }


@dataclass
class LigandSite:
    """Every interaction PLIP found for one ligand."""

    hetid: str
    chain: str
    position: int
    identifier: str
    ligand_type: str = ""
    longname: str = ""
    smiles: str = ""
    num_heavy_atoms: int = 0
    num_rings: int = 0
    interactions: list[Interaction] = field(default_factory=list)

    @property
    def fingerprint(self) -> list[str]:
        """Sorted, de-duplicated interaction keys for this ligand."""
        return sorted({interaction.key for interaction in self.interactions})

    def counts(self) -> dict[str, int]:
        """Interaction count per family, families with none omitted."""
        counts: dict[str, int] = {}
        for interaction in self.interactions:
            counts[interaction.family] = counts.get(interaction.family, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "identifier": self.identifier,
            "hetid": self.hetid,
            "chain": self.chain,
            "position": self.position,
            "ligand_type": self.ligand_type,
            "longname": self.longname,
            "smiles": self.smiles,
            "num_heavy_atoms": self.num_heavy_atoms,
            "num_rings": self.num_rings,
            "num_interactions": len(self.interactions),
            "counts": self.counts(),
            "interactions": [i.to_dict() for i in self.interactions],
            "fingerprint": self.fingerprint,
            "residues": self.residues(),
        }

    def residues(self) -> list[dict[str, Any]]:
        """One entry per contacted residue, with the families it contributes.

        This is the shape a chemist reads: "TRP84 stacks and packs" rather than
        two rows that happen to share a residue number.
        """
        grouped: dict[tuple[str, int, str], set[str]] = {}
        for interaction in self.interactions:
            key = (interaction.chain, interaction.residue_number, interaction.residue_name)
            grouped.setdefault(key, set()).add(interaction.family)
        return [
            {
                "chain": chain,
                "residue_number": number,
                "residue_name": name,
                "families": sorted(families),
            }
            for (chain, number, name), families in sorted(
                grouped.items(), key=lambda item: (item[0][0], item[0][1])
            )
        ]


@dataclass
class Profile:
    """The result of profiling one structure."""

    ligands: list[LigandSite] = field(default_factory=list)
    designated: Optional[str] = None
    designation_reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def site(self, identifier: Optional[str] = None) -> Optional[LigandSite]:
        """Return a ligand by identifier, defaulting to the designated one."""
        target = identifier or self.designated
        for ligand in self.ligands:
            if ligand.identifier == target:
                return ligand
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ligands": [ligand.to_dict() for ligand in self.ligands],
            "designated": self.designated,
            "designation_reason": self.designation_reason,
            "warnings": self.warnings,
        }


class ProfilingError(Exception):
    """A structure PLIP could not profile at all."""


_configured = False


def configure() -> None:
    """Apply the PLIP settings this service needs. Safe to call repeatedly.

    PLIP writes a "fixed" copy of every structure it reads to a randomly named
    file under the system temp directory, and never removes it. In a CLI run that
    is a debugging aid; in a long-lived server it is one leaked file per request,
    growing without bound. NOFIXFILE keeps the fix in memory instead, and the
    interactions come out identical.

    Note that this does not remove the need to serialize analyses: PLIP also
    writes the protonated structure to a *fixed* path when reading from a string,
    so two concurrent analyses would race on the same file.
    """
    global _configured
    if _configured:
        return

    from plip.basic import config

    config.NOFIXFILE = True
    _configured = True


def _reason(exc: BaseException) -> str:
    """Describe why PLIP gave up.

    A SystemExit stringifies to its exit code, so reporting it verbatim yields
    "could not read the structure: 1" -- true, and useless to whoever has to fix
    the input.
    """
    if isinstance(exc, SystemExit):
        return (
            "the input was not recognized as a PDB or mmCIF structure "
            "(PLIP aborted while reading it)"
        )
    return str(exc) or exc.__class__.__name__


def _scalar(value: Any) -> Any:
    """Make a PLIP field JSON-serializable.

    PLIP returns distances as strings and coordinates as numpy scalars; both
    would otherwise reach the caller as something json cannot encode or as a
    string where a number is expected.
    """
    if isinstance(value, (str, bool, int)) or value is None:
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return value
        return value
    if isinstance(value, (tuple, list)):
        return [_scalar(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def _detail(features: tuple[str, ...], row: tuple[Any, ...]) -> dict[str, Any]:
    """Zip a PLIP row against its column names, dropping the noise columns.

    Zipping rather than indexing matters: the families do not share a row
    layout beyond the first three columns, and positional access is how a
    silent mis-attribution gets introduced when PLIP adds a column.
    """
    detail: dict[str, Any] = {}
    for name, value in zip(features, row):
        if name in _NOISE_COLUMNS:
            continue
        detail[name.lower()] = _scalar(value)
    return detail


def _residue(features: tuple[str, ...], row: tuple[Any, ...]) -> Optional[tuple[str, int, str]]:
    """Extract (chain, residue number, residue name) from a PLIP row."""
    lookup = dict(zip(features, row))
    if not all(column in lookup for column in _RESIDUE_COLUMNS):
        return None
    try:
        number = int(lookup["RESNR"])
    except (TypeError, ValueError):
        return None
    return str(lookup["RESCHAIN"]), number, str(lookup["RESTYPE"])


def _collect(site: Any) -> list[Interaction]:
    """Turn one PLIP binding site into coarsened interactions."""
    from plip.exchange.report import BindingSiteReport

    report = BindingSiteReport(site)
    interactions: list[Interaction] = []

    for family in FAMILIES:
        features = getattr(report, f"{family}_features", None)
        rows = getattr(report, f"{family}_info", None)
        if not features or not rows:
            continue
        for row in rows:
            residue = _residue(features, row)
            if residue is None:
                logger.warning("Skipping a %s row with no residue columns", family)
                continue
            chain, number, name = residue
            interactions.append(
                Interaction(
                    family=family,
                    chain=chain,
                    residue_number=number,
                    residue_name=name,
                    detail=_detail(features, row),
                )
            )

    return interactions


def _designate(
    ligands: list[LigandSite], requested: str = ""
) -> tuple[Optional[str], str, list[str]]:
    """Choose which ligand the fingerprint describes.

    Preference order: what the caller asked for, then the interacting ligand
    that is not a known crystallization additive, then most interactions, then
    most heavy atoms. The reason is returned so the answer can state it.
    """
    warnings: list[str] = []
    if not ligands:
        return None, "", warnings

    if requested:
        wanted = requested.strip().upper()
        matches = [
            ligand
            for ligand in ligands
            if wanted in (ligand.identifier.upper(), ligand.hetid.upper())
        ]
        if matches:
            return matches[0].identifier, f"requested ({requested})", warnings
        available = ", ".join(ligand.identifier for ligand in ligands)
        warnings.append(
            f"No ligand matching '{requested}' was found; this structure has "
            f"{available}. Profiled the largest interacting ligand instead."
        )

    artifacts = biolip_artifacts()

    def rank(ligand: LigandSite) -> tuple[int, int, int]:
        relevant = 0 if ligand.hetid.upper() in artifacts else 1
        return (relevant, len(ligand.interactions), ligand.num_heavy_atoms)

    best = max(ligands, key=rank)
    if len(ligands) == 1:
        return best.identifier, "the only ligand in the structure", warnings

    reason = (
        f"most interactions ({len(best.interactions)}) among "
        f"{len(ligands)} ligands"
    )
    if best.hetid.upper() not in artifacts and any(
        ligand.hetid.upper() in artifacts for ligand in ligands
    ):
        reason += "; the others are additives or glycans PLIP flags as likely irrelevant"
    return best.identifier, reason, warnings


def profile(pdb_text: str, ligand: str = "") -> Profile:
    """Profile every ligand binding site in a PDB structure.

    Args:
        pdb_text: The structure, as PDB-format text.
        ligand: Optional hetid or ``HETID:CHAIN:POSITION`` identifier to
            designate. When it does not match, the fallback is reported rather
            than applied silently.

    Returns:
        A Profile. Raises ProfilingError when the structure cannot be read.
    """
    from plip.structure.preparation import PDBComplex

    configure()

    complex_ = PDBComplex()
    # SystemExit is not defensive noise here: PLIP's reader calls sys.exit() when
    # it cannot recognize the input, which in a server would take down the
    # process over one malformed structure. It is a BaseException, so a plain
    # `except Exception` does not see it.
    try:
        complex_.load_pdb(pdb_text, as_string=True)
    except (Exception, SystemExit) as exc:
        raise ProfilingError(
            f"PLIP could not read the structure: {_reason(exc)}"
        ) from exc

    try:
        complex_.analyze()
    except (Exception, SystemExit) as exc:
        raise ProfilingError(
            f"PLIP failed while analyzing the structure: {_reason(exc)}"
        ) from exc

    ligands: list[LigandSite] = []
    for identifier, site in complex_.interaction_sets.items():
        ligand_info = site.ligand
        try:
            position = int(ligand_info.position)
        except (TypeError, ValueError):
            position = 0
        ligands.append(
            LigandSite(
                hetid=str(ligand_info.hetid),
                chain=str(ligand_info.chain),
                position=position,
                identifier=identifier,
                ligand_type=str(getattr(ligand_info, "type", "") or ""),
                longname=str(getattr(ligand_info, "longname", "") or ""),
                smiles=str(getattr(ligand_info, "smiles", "") or ""),
                num_heavy_atoms=int(getattr(ligand_info, "heavy_atoms", 0) or 0),
                num_rings=int(getattr(ligand_info, "num_rings", 0) or 0),
                interactions=_collect(site),
            )
        )

    ligands.sort(key=lambda entry: (-len(entry.interactions), entry.identifier))

    designated, reason, warnings = _designate(ligands, ligand)
    if not ligands:
        warnings.append(
            "No ligand binding site was found. PLIP looks for HETATM groups, so "
            "a structure with no ligand, or one whose ligand was written as "
            "ATOM records, profiles as empty."
        )

    return Profile(
        ligands=ligands,
        designated=designated,
        designation_reason=reason,
        warnings=warnings,
    )
