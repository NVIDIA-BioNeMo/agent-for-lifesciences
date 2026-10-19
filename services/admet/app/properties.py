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

"""The ADMET property catalog.

ADMET-AI emits around fifty columns per molecule, named after the TDC datasets
its ensembles were trained on: ``BBB_Martins``, ``Clearance_Hepatocyte_AZ``,
``PPBR_AZ``. Those names carry the provenance that makes a prediction citable,
so they stay as the canonical keys, but nobody asks a question in them. This
module holds the mapping in both directions -- aliases in, labels and units out
-- along with the two facts a caller cannot get from the number alone: what the
number means, and which way is better.

Direction is the part worth being careful about. "Lower hERG is better" is true
of every drug programme; "lower blood-brain-barrier penetration is better" is
true only of a peripheral target and exactly wrong for a CNS one. Properties
like that are marked ``CONTEXT``, so a caller reporting a verdict has to say
what it is optimizing for rather than assume.

Deliberately import-free beyond the standard library: the tool layer and the
tests read this catalog without paying for torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class TaskType(str, Enum):
    """Whether a prediction is a probability or a physical quantity."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class Direction(str, Enum):
    """Which way is better for a drug candidate."""

    LOWER = "lower"
    HIGHER = "higher"
    # Better depends on the programme: BBB penetration is required for a CNS
    # target and a liability for a peripheral one.
    CONTEXT = "context"


@dataclass(frozen=True)
class Property:
    """One ADMET-AI output column.

    Attributes:
        name: Canonical key, matching the ADMET-AI / TDC column name.
        label: How a chemist says it.
        abbrev: Short column header for tabular output, <= 8 characters.
        category: ADMET-AI's own grouping (Absorption, Toxicity, ...).
        task_type: Probability or physical quantity.
        units: Units of a regression value; empty for probabilities.
        direction: Which way is better, or CONTEXT when it depends.
        aliases: Spellings accepted on input.
        note: What the number means, when that is not obvious from the label.
    """

    name: str
    label: str
    abbrev: str
    category: str
    task_type: TaskType
    units: str = ""
    direction: Direction = Direction.CONTEXT
    aliases: tuple[str, ...] = ()
    note: str = ""


# The eight properties SMDD-Bench scores, and the ones a medicinal chemist asks
# about first: a toxicity liability, a permeability, a solubility, a metabolic
# fate, a clearance, an exposure and a mutagenicity.
CORE_PANEL: tuple[str, ...] = (
    "hERG",
    "BBB_Martins",
    "Solubility_AqSolDB",
    "CYP3A4_Veith",
    "Clearance_Hepatocyte_AZ",
    "Caco2_Wang",
    "PPBR_AZ",
    "AMES",
)

PROPERTIES: tuple[Property, ...] = (
    # --- Toxicity ---
    Property(
        "hERG",
        "hERG blocking",
        "hERG",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("herg", "herg_blockers", "herg_inhibition"),
        note="Probability of blocking the hERG potassium channel; cardiac "
        "liability. Below 0.3 is usually considered clean.",
    ),
    Property(
        "AMES",
        "Ames mutagenicity",
        "AMES",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("ames", "mutagenicity"),
        note="Probability of being mutagenic in the Ames assay.",
    ),
    Property(
        "DILI",
        "Drug-induced liver injury",
        "DILI",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("dili", "hepatotoxicity"),
    ),
    Property(
        "ClinTox",
        "Clinical trial toxicity",
        "ClinTox",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("clintox",),
    ),
    Property(
        "Carcinogens_Lagunin",
        "Carcinogenicity",
        "Carcino",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("carcinogenicity", "carcinogens"),
    ),
    Property(
        "Skin_Reaction",
        "Skin reaction",
        "skin",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("skin_reaction", "skin_sensitization"),
    ),
    Property(
        "LD50_Zhu",
        "Acute toxicity LD50",
        "LD50",
        "Toxicity",
        TaskType.REGRESSION,
        units="log(1/(mol/kg))",
        direction=Direction.LOWER,
        aliases=("ld50", "acute_toxicity"),
        note="Reported as log(1/(mol/kg)), so a *lower* value means a higher "
        "lethal dose and therefore less acute toxicity.",
    ),
    # --- Toxicity: the Tox21 nuclear-receptor and stress-response panel ---
    #
    # Twelve in-vitro assays rather than twelve organism-level outcomes. They are
    # useful as flags -- an estrogen-receptor hit on a molecule intended for
    # chronic dosing is worth knowing -- but several of the models are weak
    # (AUPRC below 0.5 on the published split), so a single value here is a
    # prompt to look, not a verdict. All are lower-is-better.
    Property(
        "NR-AR",
        "Androgen receptor (full length)",
        "AR",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("nr_ar", "androgen_receptor"),
    ),
    Property(
        "NR-AR-LBD",
        "Androgen receptor (ligand binding domain)",
        "AR-LBD",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("nr_ar_lbd",),
    ),
    Property(
        "NR-AhR",
        "Aryl hydrocarbon receptor",
        "AhR",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("nr_ahr", "ahr"),
    ),
    Property(
        "NR-Aromatase",
        "Aromatase",
        "Aromtase",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("nr_aromatase", "aromatase"),
    ),
    Property(
        "NR-ER",
        "Estrogen receptor (full length)",
        "ER",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("nr_er", "estrogen_receptor"),
    ),
    Property(
        "NR-ER-LBD",
        "Estrogen receptor (ligand binding domain)",
        "ER-LBD",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("nr_er_lbd",),
    ),
    Property(
        "NR-PPAR-gamma",
        "PPAR-gamma",
        "PPARg",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("nr_ppar_gamma", "ppar_gamma"),
    ),
    Property(
        "SR-ARE",
        "Antioxidant response element",
        "ARE",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("sr_are", "are"),
    ),
    Property(
        "SR-ATAD5",
        "ATAD5 (genotoxicity)",
        "ATAD5",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("sr_atad5", "atad5"),
    ),
    Property(
        "SR-HSE",
        "Heat shock response element",
        "HSE",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("sr_hse", "hse"),
    ),
    Property(
        "SR-MMP",
        "Mitochondrial membrane potential",
        "MMP",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("sr_mmp", "mitochondrial_toxicity"),
    ),
    Property(
        "SR-p53",
        "p53 stress response",
        "p53",
        "Toxicity",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("sr_p53", "p53"),
    ),
    # --- Absorption ---
    Property(
        "Caco2_Wang",
        "Caco-2 permeability",
        "Caco-2",
        "Absorption",
        TaskType.REGRESSION,
        units="log(10^-6 cm/s)",
        direction=Direction.HIGHER,
        aliases=("caco2", "caco_2", "permeability", "caco2_permeability"),
        note="Apparent permeability across a Caco-2 monolayer; a proxy for "
        "intestinal absorption.",
    ),
    Property(
        "Solubility_AqSolDB",
        "Aqueous solubility",
        "logS",
        "Absorption",
        TaskType.REGRESSION,
        units="log(mol/L)",
        direction=Direction.HIGHER,
        aliases=("solubility", "aqueous_solubility", "logs"),
    ),
    Property(
        "Lipophilicity_AstraZeneca",
        "Lipophilicity (logD)",
        "logD",
        "Absorption",
        TaskType.REGRESSION,
        units="log-ratio",
        direction=Direction.CONTEXT,
        aliases=("lipophilicity", "logd"),
    ),
    Property(
        "HydrationFreeEnergy_FreeSolv",
        "Hydration free energy",
        "dGhyd",
        "Absorption",
        TaskType.REGRESSION,
        units="kcal/mol",
        direction=Direction.CONTEXT,
        aliases=("hydration_free_energy", "freesolv"),
        note="Free energy of solvation in water. More negative means more "
        "readily hydrated, which helps solubility and hurts membrane "
        "permeability, so neither direction is uniformly better.",
    ),
    Property(
        "HIA_Hou",
        "Human intestinal absorption",
        "HIA",
        "Absorption",
        TaskType.CLASSIFICATION,
        direction=Direction.HIGHER,
        aliases=("hia", "intestinal_absorption"),
    ),
    Property(
        "Bioavailability_Ma",
        "Oral bioavailability",
        "F_oral",
        "Absorption",
        TaskType.CLASSIFICATION,
        direction=Direction.HIGHER,
        aliases=("bioavailability", "oral_bioavailability"),
    ),
    Property(
        "Pgp_Broccatelli",
        "P-glycoprotein inhibition",
        "Pgp",
        "Absorption",
        TaskType.CLASSIFICATION,
        direction=Direction.CONTEXT,
        aliases=("pgp", "p_glycoprotein"),
    ),
    Property(
        "PAMPA_NCATS",
        "PAMPA permeability",
        "PAMPA",
        "Absorption",
        TaskType.CLASSIFICATION,
        direction=Direction.HIGHER,
        aliases=("pampa",),
    ),
    # --- Distribution ---
    Property(
        "BBB_Martins",
        "Blood-brain barrier penetration",
        "BBB",
        "Distribution",
        TaskType.CLASSIFICATION,
        direction=Direction.CONTEXT,
        aliases=("bbb", "blood_brain_barrier", "bbb_penetration"),
        note="Required for a CNS target, a liability for a peripheral one -- "
        "say which you want before calling a value good.",
    ),
    Property(
        "PPBR_AZ",
        "Plasma protein binding rate",
        "PPBR",
        "Distribution",
        TaskType.REGRESSION,
        units="%",
        direction=Direction.CONTEXT,
        aliases=("ppbr", "plasma_protein_binding", "ppb"),
        note="Percent bound. High binding leaves little free drug, but the "
        "target free concentration is what matters, so direction depends on "
        "the programme.",
    ),
    Property(
        "VDss_Lombardo",
        "Volume of distribution at steady state",
        "VDss",
        "Distribution",
        TaskType.REGRESSION,
        units="L/kg",
        direction=Direction.CONTEXT,
        aliases=("vdss", "volume_of_distribution"),
    ),
    # --- Metabolism ---
    Property(
        "CYP3A4_Veith",
        "CYP3A4 inhibition",
        "3A4inh",
        "Metabolism",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("cyp3a4", "cyp3a4_inhibition"),
        note="Probability of inhibiting CYP3A4; drug-drug interaction risk.",
    ),
    Property(
        "CYP2D6_Veith",
        "CYP2D6 inhibition",
        "2D6inh",
        "Metabolism",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("cyp2d6",),
    ),
    Property(
        "CYP2C9_Veith",
        "CYP2C9 inhibition",
        "2C9inh",
        "Metabolism",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("cyp2c9",),
    ),
    Property(
        "CYP2C19_Veith",
        "CYP2C19 inhibition",
        "2C19inh",
        "Metabolism",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("cyp2c19",),
    ),
    Property(
        "CYP1A2_Veith",
        "CYP1A2 inhibition",
        "1A2inh",
        "Metabolism",
        TaskType.CLASSIFICATION,
        direction=Direction.LOWER,
        aliases=("cyp1a2",),
    ),
    Property(
        "CYP3A4_Substrate_CarbonMangels",
        "CYP3A4 substrate",
        "3A4sub",
        "Metabolism",
        TaskType.CLASSIFICATION,
        direction=Direction.CONTEXT,
        aliases=("cyp3a4_substrate",),
        note="Being a substrate is not itself a liability; it says which enzyme "
        "clears the molecule, and therefore which co-medications interact.",
    ),
    Property(
        "CYP2C9_Substrate_CarbonMangels",
        "CYP2C9 substrate",
        "2C9sub",
        "Metabolism",
        TaskType.CLASSIFICATION,
        direction=Direction.CONTEXT,
        aliases=("cyp2c9_substrate",),
    ),
    Property(
        "CYP2D6_Substrate_CarbonMangels",
        "CYP2D6 substrate",
        "2D6sub",
        "Metabolism",
        TaskType.CLASSIFICATION,
        direction=Direction.CONTEXT,
        aliases=("cyp2d6_substrate",),
    ),
    # --- Excretion ---
    Property(
        "Clearance_Hepatocyte_AZ",
        "Hepatocyte clearance",
        "CLhep",
        "Excretion",
        TaskType.REGRESSION,
        units="uL/min/10^6 cells",
        direction=Direction.LOWER,
        aliases=("clearance", "hepatocyte_clearance", "clearance_hepatocyte"),
        note="Intrinsic clearance in hepatocytes. Lower means slower "
        "metabolism and longer exposure.",
    ),
    Property(
        "Clearance_Microsome_AZ",
        "Microsomal clearance",
        "CLmic",
        "Excretion",
        TaskType.REGRESSION,
        units="uL/min/mg",
        direction=Direction.LOWER,
        aliases=("clearance_microsome", "microsomal_clearance"),
    ),
    Property(
        "Half_Life_Obach",
        "Half life",
        "t1/2",
        "Excretion",
        TaskType.REGRESSION,
        units="hr",
        direction=Direction.HIGHER,
        aliases=("half_life", "t_half"),
    ),
    # --- Physicochemical, computed by RDKit inside ADMET-AI ---
    Property(
        "molecular_weight",
        "Molecular weight",
        "MW",
        "Physicochemical",
        TaskType.REGRESSION,
        units="Dalton",
        aliases=("mw", "molecular_weight"),
    ),
    Property(
        "logP",
        "LogP",
        "logP",
        "Physicochemical",
        TaskType.REGRESSION,
        units="log-ratio",
        aliases=("logp", "clogp"),
    ),
    Property(
        "tpsa",
        "Topological polar surface area",
        "TPSA",
        "Physicochemical",
        TaskType.REGRESSION,
        units="Å^2",
        aliases=("tpsa",),
    ),
    Property(
        "QED",
        "Quantitative estimate of drug-likeness",
        "QED",
        "Physicochemical",
        TaskType.REGRESSION,
        direction=Direction.HIGHER,
        aliases=("qed",),
    ),
    Property(
        "hydrogen_bond_donors",
        "Hydrogen bond donors",
        "HBD",
        "Physicochemical",
        TaskType.REGRESSION,
        units="#",
        aliases=("hbd", "hydrogen_bond_donors"),
    ),
    Property(
        "hydrogen_bond_acceptors",
        "Hydrogen bond acceptors",
        "HBA",
        "Physicochemical",
        TaskType.REGRESSION,
        units="#",
        aliases=("hba", "hydrogen_bond_acceptors"),
    ),
    Property(
        "Lipinski",
        "Lipinski rule-of-five count",
        "Ro5",
        "Physicochemical",
        TaskType.REGRESSION,
        units="# of 4",
        direction=Direction.HIGHER,
        aliases=("lipinski", "rule_of_five"),
    ),
    Property(
        "stereo_centers",
        "Stereo centers",
        "stereo",
        "Physicochemical",
        TaskType.REGRESSION,
        units="#",
        aliases=("stereo_centers",),
    ),
    # Structural alerts. Counts of matched substructure rules, not predictions --
    # RDKit's FilterCatalog computes them, and ADMET-AI returns them alongside
    # the model outputs. They are classified REGRESSION here for the same reason
    # the package does: the value is a count, not a probability.
    Property(
        "PAINS_alert",
        "PAINS alerts",
        "PAINS",
        "Physicochemical",
        TaskType.REGRESSION,
        units="#",
        direction=Direction.LOWER,
        aliases=("pains", "pains_alert", "pains_alerts"),
        note="Pan-assay interference substructures: a non-zero count means the "
        "molecule is likely to look active against many unrelated targets.",
    ),
    Property(
        "BRENK_alert",
        "Brenk alerts",
        "BRENK",
        "Physicochemical",
        TaskType.REGRESSION,
        units="#",
        direction=Direction.LOWER,
        aliases=("brenk", "brenk_alert", "brenk_alerts"),
        note="Substructures Brenk et al. flag as unsuitable for a lead: "
        "reactive, metabolically labile or toxicophoric groups.",
    ),
    Property(
        "NIH_alert",
        "NIH alerts",
        "NIH",
        "Physicochemical",
        TaskType.REGRESSION,
        units="#",
        direction=Direction.LOWER,
        aliases=("nih", "nih_alert", "nih_alerts"),
    ),
)

BY_NAME: dict[str, Property] = {prop.name: prop for prop in PROPERTIES}

# Lookup including aliases and case-insensitive canonical names. Built once.
_BY_KEY: dict[str, Property] = {}
for _prop in PROPERTIES:
    _BY_KEY[_prop.name.lower()] = _prop
    for _alias in _prop.aliases:
        _BY_KEY[_alias.lower()] = _prop


def resolve(name: str) -> Optional[Property]:
    """Resolve a property name or alias, or None when unknown."""
    return _BY_KEY.get(str(name or "").strip().lower())


def resolve_names(names: list[str]) -> tuple[list[str], list[str]]:
    """Resolve a requested property list.

    Returns:
        ``(canonical_names, unknown)``. Order and duplicates from the request
        are preserved in the canonical list only once, so a caller asking for
        both ``herg`` and ``hERG`` gets one column.
    """
    canonical: list[str] = []
    unknown: list[str] = []
    for name in names:
        prop = resolve(name)
        if prop is None:
            unknown.append(name)
        elif prop.name not in canonical:
            canonical.append(prop.name)
    return canonical, unknown


def describe(name: str) -> dict[str, object]:
    """Render a property's metadata as a JSON-serializable dict."""
    prop = resolve(name)
    if prop is None:
        return {"name": name, "known": False}
    return {
        "name": prop.name,
        "known": True,
        "label": prop.label,
        "abbrev": prop.abbrev,
        "category": prop.category,
        "task_type": prop.task_type.value,
        "units": prop.units,
        "direction": prop.direction.value,
        "note": prop.note,
    }
