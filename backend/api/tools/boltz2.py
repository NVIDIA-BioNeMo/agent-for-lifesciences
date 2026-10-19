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

"""Boltz-2 structure and binding affinity prediction via NVIDIA's hosted MIT NIM.

Boltz-2 predicts the structure of a biomolecular complex and, uniquely among the
structure predictors Aevom wraps, can also estimate a small molecule's binding
affinity in the same call. That combination is why it earns its own tool rather
than an ``openfold3`` mode: the affinity head answers "how tightly does this
bind" from sequence alone, without needing an experimental receptor structure the
way ``vina_dock`` does.

Two things about it differ from every other structure tool here.

The output is mmCIF and nothing else -- ``output_format`` accepts no other value
-- so this is the first builtin tool to store an ``mmcif`` artifact. It is stored
as mmCIF rather than silently converted, and ``convert_structure`` produces a
``pdb`` reference for the tools that can only read PDB.

The affinity numbers are small and are what the model actually reasons over, so
they are reported inline. The large numeric outputs (PAE matrices, per-residue
confidences) are stored as ``array`` references instead, because thousands of
numbers in context are unusable.

The detailed parameter guidance lives in the upstream ``boltz2-nim`` skill; see
``api.skills.binding``.
"""

import io
import json
import logging
import os
import re
from typing import Any, Optional

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from .base import DataIOCategory, NvidiaApiError, NvidiaApiTool, ToolResult

logger = logging.getLogger(__name__)

# Every other service-backed tool reads its endpoint from the environment
# (ADMET_SERVICE_URL, VINA_SERVICE_URL, ...); this one hardcoded the hosted NIM,
# so a deployment that runs Boltz-2 itself -- on a card it already owns, behind a
# firewall, or pinned to a particular build -- had no way to say so without
# editing the source. The default is unchanged.
# ``or``, not a get() default: an orchestrator that declares the variable and
# leaves it empty means "use the default", and every compose file in this repo
# declares more variables than it sets.
BOLTZ2_API_URL = (
    os.environ.get("BOLTZ2_API_URL")
    or "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
)

# Service limits, enforced before the call so a violation reads as a sentence
# rather than as the endpoint's 422.
MAX_POLYMERS = 12
MAX_LIGANDS = 20
MAX_SEQUENCE_LENGTH = 4096

MIN_RECYCLING_STEPS, MAX_RECYCLING_STEPS, DEFAULT_RECYCLING_STEPS = 1, 10, 3
MIN_SAMPLING_STEPS, MAX_SAMPLING_STEPS, DEFAULT_SAMPLING_STEPS = 10, 1000, 50

# One structure sample by default, deliberately. Every extra sample is another
# full diffusion pass on a metered endpoint, and for the common requests -- show
# me the complex, give me a pIC50 -- it buys nothing: the affinity head averages
# DEFAULT_AFFINITY_DIFFUSION_SAMPLES draws of its own, so the reported affinity is
# already a consensus regardless of this number.
#
# What one sample does not support is reasoning about the pose, because diffusion
# is stochastic. Rather than charge every caller for reproducibility that most do
# not need, the response says when it is reporting a single draw and the schema
# says when to raise this. Whether the default should move is an empirical
# question about how much the pose actually varies, which the Track B fidelity
# check is set up to answer; it should not be guessed at here.
MIN_DIFFUSION_SAMPLES, MAX_DIFFUSION_SAMPLES, DEFAULT_DIFFUSION_SAMPLES = 1, 25, 1
MIN_STEP_SCALE, MAX_STEP_SCALE, DEFAULT_STEP_SCALE = 0.5, 5.0, 1.638
(
    MIN_AFFINITY_SAMPLING_STEPS,
    MAX_AFFINITY_SAMPLING_STEPS,
    DEFAULT_AFFINITY_SAMPLING_STEPS,
) = 10, 1000, 200
(
    MIN_AFFINITY_DIFFUSION_SAMPLES,
    MAX_AFFINITY_DIFFUSION_SAMPLES,
    DEFAULT_AFFINITY_DIFFUSION_SAMPLES,
) = 1, 10, 5

# Affinity prediction runs a second diffusion process on top of the structure, so
# a request with it enabled takes minutes rather than seconds.
#
# The client timeout is derived from this, so it is also how long Aevom is willing
# to wait -- and 300 s is not long enough for every Boltz-2 there is. A ten-sample
# co-fold on one GV100 takes a median of 527 s and up to 1084 s, so a self-hosted
# endpoint would have the request abandoned from this side *after* the GPU had
# finished the work. The hosted NIM's gateway caps what it will hold a connection
# open for, which is why the default stays at 300; a deployment pointing
# BOLTZ2_API_URL at its own service raises this to match what that service does.
POLL_SECONDS = int(os.environ.get("BOLTZ2_POLL_SECONDS") or "300")

VALID_AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
VALID_DNA_NUCLEOTIDES = set("ATCG")
VALID_RNA_NUCLEOTIDES = set("AUCG")
VALID_ALPHABETS = {
    "protein": VALID_AMINO_ACIDS,
    "dna": VALID_DNA_NUCLEOTIDES,
    "rna": VALID_RNA_NUCLEOTIDES,
}

# Names the model reaches for instead of the schema's. Every one of these was
# measured in real traces, not guessed: across two 100-task benchmark runs the
# model wrote `type` for `constraint_type` 73 times, `residues` for `contacts`
# 67 times, and `chain`/`residue` for `id`/`residue_index` 61 times between
# them -- and each one cost a call against an oracle budget of eight. The
# schema documents the right names clearly and the model writes the domain's
# names anyway, so the tool accepts both. Only unambiguous aliases are listed:
# a constraint's bare `chain` is not here, because it could mean the binder or
# a contact and guessing wrong is worse than asking.
_POLYMER_ALIASES = {
    "type": "molecule_type",
    "chain": "id",
    "chain_id": "id",
}
_CONSTRAINT_ALIASES = {
    "type": "constraint_type",
    "residues": "contacts",
    "residue_indices": "contacts",
    "ligand": "binder",
    "binder_id": "binder",
}
_RESIDUE_ALIASES = {
    "chain": "id",
    "chain_id": "id",
    "residue": "residue_index",
    "residue_number": "residue_index",
    "resi": "residue_index",
    "seq_id": "residue_index",
}

# Letters that appear in proteins and in no nucleic acid, so a sequence holding
# one cannot be DNA or RNA. Used only to fill in an omitted molecule_type; a
# sequence of nothing but ACGTU stays ambiguous and is still refused.
_PROTEIN_ONLY_LETTERS = VALID_AMINO_ACIDS - VALID_DNA_NUCLEOTIDES - VALID_RNA_NUCLEOTIDES


def _apply_aliases(entry: Any, aliases: dict[str, str]) -> Any:
    """Rename known synonyms onto the schema's names, without overwriting.

    A caller that gave both the alias and the real name meant the real one, so
    the alias is dropped rather than allowed to win.
    """
    if not isinstance(entry, dict):
        return entry
    renamed = dict(entry)
    for alias, canonical in aliases.items():
        if alias in renamed and not renamed.get(canonical):
            renamed[canonical] = renamed.pop(alias)
    return renamed


def _unwrap_binder(binder: Any) -> Any:
    """A binder given as an object rather than a chain id.

    `{"ligand": "L1"}` and `{"id": "L1"}` both name chain L1, and both appear in
    traces. Anything else is left alone so the existing error still fires.
    """
    if isinstance(binder, dict):
        for key in ("id", "ligand", "chain", "chain_id", "binder"):
            value = binder.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return binder


def _normalise_constraint(entry: Any, ligand_ids: set[str] | None = None) -> Any:
    """One constraint in whichever of its shapes the caller used.

    Boltz's own YAML nests the kind as a key -- `{"pocket": {...}}` -- and the
    model copies that shape from the documentation it has seen. Flattening it is
    unambiguous: the key names the constraint type and its value holds the rest.
    """
    if not isinstance(entry, dict):
        return entry
    for kind in ("pocket", "bond"):
        nested = entry.get(kind)
        if isinstance(nested, dict):
            entry = {**{k: v for k, v in entry.items() if k != kind}, **nested, "constraint_type": kind}
            break
    entry = _apply_aliases(entry, _CONSTRAINT_ALIASES)
    if "binder" in entry:
        entry["binder"] = _unwrap_binder(entry["binder"])
    elif entry.get("constraint_type") == "pocket" and ligand_ids and len(ligand_ids) == 1:
        # A pocket constraint steers a ligand at a pocket, and with one ligand
        # in the request there is only one thing it can be steering. 38 calls in
        # the traces omitted the binder for exactly this reason. With more than
        # one ligand it stays ambiguous and the error still fires.
        entry["binder"] = next(iter(ligand_ids))
    return entry


_CHAIN_AND_INDEX = re.compile(r"^([A-Za-z][A-Za-z0-9]*?)(\d+)$")


def _normalise_residue_ref(
    entry: Any, known_ids: set[str] | None = None, default_chain: str | None = None
) -> Any:
    """One contact or bonded atom, in whichever of its forms the caller used.

    Beyond the schema's ``{"id", "residue_index"}``:

    - ``["A", 22]`` -- the pair form Boltz's YAML uses.
    - ``"A40"`` -- chain letter glued to the index, 33 occurrences in the
      traces. Split against the declared chain ids so that ``L140`` resolves to
      chain L1 residue 40 when L1 exists, rather than chain L residue 140.
    - ``22`` -- a bare index, when the constraint named the chain once beside
      the list. 38 occurrences.
    """
    if isinstance(entry, (list, tuple)) and len(entry) == 2:
        return {"id": entry[0], "residue_index": entry[1]}
    if isinstance(entry, int) and not isinstance(entry, bool) and default_chain:
        return {"id": default_chain, "residue_index": entry}
    if isinstance(entry, str):
        match = _CHAIN_AND_INDEX.match(entry.strip())
        if match:
            chain, index = match.group(1), match.group(2)
            # Prefer the longest declared id that prefixes the string, so a
            # two-character chain id is not read as one character plus a digit.
            for candidate in sorted(known_ids or (), key=len, reverse=True):
                remainder = entry.strip()[len(candidate):]
                if entry.strip().startswith(candidate) and remainder.isdigit():
                    return {"id": candidate, "residue_index": int(remainder)}
            return {"id": chain, "residue_index": int(index)}
        if default_chain and entry.strip().isdigit():
            return {"id": default_chain, "residue_index": int(entry.strip())}
    return _apply_aliases(entry, _RESIDUE_ALIASES)


def infer_molecule_type(sequence: str) -> str | None:
    """The molecule type a sequence can only be, or None if it is ambiguous.

    Two-thirds of the polymers in the benchmark traces named a sequence and no
    molecule_type, and every one of those cost an oracle call. A sequence
    containing a letter no nucleic acid uses can only be a protein. One made
    only of ACGTU could be either a short peptide or a nucleic acid, and is
    still refused rather than guessed.
    """
    letters = set((sequence or "").strip().upper())
    if not letters:
        return None
    if letters & _PROTEIN_ONLY_LETTERS:
        return "protein"
    return None

# Chain ids are short alphanumeric labels: 'A' for polymers, 'L1' for ligands in
# every upstream example. The documented polymer rule is "single letter or 4-char
# alphanumeric", but the ligand examples are two characters and the service
# accepts them, so only the shape both agree on is enforced here. Rejecting a
# three-character id on the strength of the polymer wording would refuse ids the
# endpoint takes.
_CHAIN_ID_RE = re.compile(r"^[A-Za-z0-9]{1,4}$")
_CCD_RE = re.compile(r"^[A-Za-z0-9]{1,3}$")
_SMILES_RE = re.compile(r"^[A-Za-z0-9\(\)\[\]=#@+\-\.\\/%\*:]+$")

# Response keys the tool reads by name. Anything else that is numeric and
# array-shaped is stored as an `array` artifact rather than dropped.
_CLAIMED_RESPONSE_KEYS = frozenset(
    {"structures", "confidence_scores", "affinities", "metrics"}
)
_CLAIMED_STRUCTURE_KEYS = frozenset({"structure", "format", "name", "source"})

# Below this, a numeric list is small enough to be worth reporting inline (or
# ignoring) rather than costing the model a reference to chase.
_MIN_ARRAY_ELEMENTS = 16


class Boltz2Tool(NvidiaApiTool):
    """Predict a complex structure, and optionally ligand affinity, with Boltz-2.

    Data Flow:
        Input: polymer sequences (raw or ``sequence_ref``), optional ``msa_ref``,
            ligand SMILES/CCD codes, optional template ``pdb_ref``s
        Output: Stores the predicted structure as an ``mmcif`` reference, extra
            diffusion samples as further ``mmcif`` references, and any large
            numeric output as ``array`` references. Affinity is inline.
    """

    @property
    def name(self) -> str:
        return "boltz2"

    @property
    def summary_template(self) -> str:
        return "Predicting structure with Boltz-2..."

    def format_summary(self, arguments: dict) -> str:
        ligands = arguments.get("ligands") or []
        polymers = arguments.get("polymers") or []
        if isinstance(ligands, list) and any(
            isinstance(lig, dict) and lig.get("predict_affinity") for lig in ligands
        ):
            return "Predicting complex structure and binding affinity with Boltz-2..."
        if isinstance(ligands, list) and ligands:
            return "Predicting protein-ligand complex with Boltz-2..."
        if isinstance(polymers, list) and len(polymers) > 1:
            return "Predicting multi-chain complex with Boltz-2..."
        return "Predicting structure with Boltz-2..."

    @property
    def api_endpoint(self) -> str:
        # Read per call, like every other service-backed tool, so a test or a
        # redeploy can point this somewhere else without reimporting the module.
        return os.environ.get("BOLTZ2_API_URL") or BOLTZ2_API_URL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.BOTH

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "boltz2",
                "description": (
                    "Predict the 3D structure of a biomolecular complex with "
                    "Boltz-2, and optionally estimate the binding affinity of one "
                    "small molecule in the same call.\n\n"
                    "READ THE SKILL: call "
                    'load_skill(skill_id="boltz2-nim") for parameter guidance, '
                    "constraint syntax, and how to judge whether a prediction is "
                    "trustworthy.\n\n"
                    "WHEN TO USE THIS RATHER THAN THE ALTERNATIVES:\n"
                    "- boltz2: you have a sequence and a ligand and want both a "
                    "complex structure and an affinity estimate. Takes SMILES "
                    "directly, so it works on novel molecules.\n"
                    "- openfold3: you only need the structure, and the ligand has "
                    "a PDB CCD code.\n"
                    "- vina_dock: you already have an experimental receptor "
                    "structure and want a physics-based binding energy in "
                    "kcal/mol.\n\n"
                    "USE CASES:\n"
                    "- Protein-ligand complex prediction plus affinity from "
                    "sequence alone\n"
                    "- Single protein, multi-chain, and protein-DNA/RNA complexes\n"
                    "- Aiming a ligand at a known pocket with a pocket constraint\n"
                    "- Modified residues (phosphorylation and similar) and cyclic "
                    "peptides\n\n"
                    "INPUT: use 'polymers' for the chains and 'ligands' for small "
                    "molecules. For a single protein you can instead pass "
                    "'sequence_ref' on its own.\n\n"
                    "AFFINITY: set predict_affinity=true on exactly ONE ligand. "
                    "The service accepts no more than one per request, and "
                    "affinity adds substantially to the runtime.\n\n"
                    "MSA: pass 'msa_ref' from msa_search to improve quality. The "
                    "tool builds the nested payload the API wants.\n\n"
                    "OUTPUT: stores an mmCIF structure reference (Boltz-2 emits no "
                    "other format). Pass it to visualize_pdb to display it, or to "
                    "convert_structure to get a PDB reference for vina_dock or "
                    "proteinmpnn_design_sequence.\n\n"
                    "Note: takes 30-120 seconds, and several minutes with affinity "
                    "prediction enabled."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "polymers": {
                            "type": "array",
                            "description": (
                                "Protein, DNA and RNA chains in the complex. "
                                f"1-{MAX_POLYMERS} entries."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "molecule_type": {
                                        "type": "string",
                                        "enum": ["protein", "dna", "rna"],
                                        "description": "Type of this chain.",
                                    },
                                    "sequence": {
                                        "type": "string",
                                        "description": (
                                            "Amino acid sequence, or nucleotides "
                                            "(DNA: ATCG, RNA: AUCG). Max "
                                            f"{MAX_SEQUENCE_LENGTH} residues. "
                                            "Alternative to sequence_ref."
                                        ),
                                    },
                                    "sequence_ref": {
                                        "type": "string",
                                        "description": (
                                            "Reference to a stored sequence from "
                                            "fetch_protein_sequence."
                                        ),
                                        "x-aevom-reference": {
                                            "data_type": "sequence"
                                        },
                                    },
                                    "id": {
                                        "type": "string",
                                        "description": (
                                            "Chain identifier, 1-4 alphanumeric "
                                            "characters (e.g. 'A'). Auto-assigned "
                                            "if omitted."
                                        ),
                                    },
                                    "cyclic": {
                                        "type": "boolean",
                                        "description": (
                                            "Treat the chain as cyclic (head-to-tail "
                                            "bonded), for cyclic peptides. Default "
                                            "false."
                                        ),
                                    },
                                    "msa_ref": {
                                        "type": "string",
                                        "description": (
                                            "Per-chain MSA reference from "
                                            "msa_search. Proteins only. Use the "
                                            "top-level msa_ref instead when there "
                                            "is only one protein chain."
                                        ),
                                        "x-aevom-reference": {"data_type": "msa"},
                                    },
                                    "modifications": {
                                        "type": "array",
                                        "description": (
                                            "Chemically modified residues, e.g. "
                                            '[{"ccd": "SEP", "position": 15}] for a '
                                            "phosphoserine at residue 15."
                                        ),
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "ccd": {
                                                    "type": "string",
                                                    "description": (
                                                        "CCD code of the modified "
                                                        "residue, 1-3 characters "
                                                        "(e.g. 'SEP', 'TPO')."
                                                    ),
                                                },
                                                "position": {
                                                    "type": "integer",
                                                    "description": (
                                                        "1-based residue index in "
                                                        "this chain."
                                                    ),
                                                },
                                            },
                                            "required": ["ccd", "position"],
                                        },
                                    },
                                    "structural_templates": {
                                        "type": "array",
                                        "description": (
                                            "Template structures to guide the "
                                            "prediction of this chain."
                                        ),
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "pdb_ref": {
                                                    "type": "string",
                                                    "description": (
                                                        "Reference to a stored "
                                                        "template structure. A PDB "
                                                        "or mmCIF reference both "
                                                        "work."
                                                    ),
                                                    "x-aevom-reference": {
                                                        "data_type": "pdb"
                                                    },
                                                },
                                                "name": {
                                                    "type": "string",
                                                    "description": (
                                                        "Optional label for the "
                                                        "template."
                                                    ),
                                                },
                                            },
                                            "required": ["pdb_ref"],
                                        },
                                    },
                                },
                                "required": ["molecule_type"],
                            },
                        },
                        "ligands": {
                            "type": "array",
                            "description": (
                                "Small molecules in the complex, up to "
                                f"{MAX_LIGANDS}. Give each either 'smiles' or "
                                "'ccd', never both."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "smiles": {
                                        "type": "string",
                                        "description": (
                                            "SMILES string, e.g. "
                                            "'CC(=O)OC1=CC=CC=C1C(=O)O' for "
                                            "aspirin. Use this for novel molecules."
                                        ),
                                    },
                                    "ccd": {
                                        "type": "string",
                                        "description": (
                                            "PDB Chemical Component Dictionary "
                                            "code, 1-3 characters (e.g. 'ATP', "
                                            "'HEM')."
                                        ),
                                    },
                                    "id": {
                                        "type": "string",
                                        "description": (
                                            "Chain identifier for this ligand, e.g. "
                                            "'L1'. Referenced by pocket and bond "
                                            "constraints."
                                        ),
                                    },
                                    "predict_affinity": {
                                        "type": "boolean",
                                        "description": (
                                            "Predict this ligand's binding "
                                            "affinity. Allowed on at most ONE "
                                            "ligand per request."
                                        ),
                                    },
                                },
                            },
                        },
                        "constraints": {
                            "type": "array",
                            "description": (
                                "Structural constraints. A pocket constraint aims a "
                                "ligand at known contact residues; a bond "
                                "constraint enforces a covalent link (for covalent "
                                "inhibitors)."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "constraint_type": {
                                        "type": "string",
                                        "enum": ["pocket", "bond"],
                                        "description": "Which kind of constraint.",
                                    },
                                    "binder": {
                                        "type": "string",
                                        "description": (
                                            "Pocket constraints only: the chain id "
                                            "of the ligand to steer, e.g. 'L1'."
                                        ),
                                    },
                                    "contacts": {
                                        "type": "array",
                                        "description": (
                                            "Pocket constraints only: residues "
                                            "lining the pocket, e.g. "
                                            '[{"id": "A", "residue_index": 25}].'
                                        ),
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {
                                                    "type": "string",
                                                    "description": "Chain id.",
                                                },
                                                "residue_index": {
                                                    "type": "integer",
                                                    "description": (
                                                        "1-based residue index."
                                                    ),
                                                },
                                            },
                                            "required": ["id", "residue_index"],
                                        },
                                    },
                                    "max_distance": {
                                        "type": "number",
                                        "description": (
                                            "Pocket constraints only: how close, "
                                            "in angstroms, the ligand must come to "
                                            "the contact residues. 6.0 is the "
                                            "usual choice and is what a pocket "
                                            "defined by a crystal contact list "
                                            "means."
                                        ),
                                    },
                                    "force": {
                                        "type": "boolean",
                                        "description": (
                                            "Pocket constraints only. False (the "
                                            "default) makes the pocket a hint the "
                                            "model may ignore; true makes it a "
                                            "requirement. When you already know "
                                            "where the ligand binds and want the "
                                            "affinity for binding THERE, set this "
                                            "and max_distance together -- "
                                            "otherwise the prediction may be for "
                                            "some other site and will not compare "
                                            "with anything that pinned the pocket."
                                        ),
                                    },
                                    "atoms": {
                                        "type": "array",
                                        "description": (
                                            "Bond constraints only: exactly two "
                                            "atoms to bond, e.g. "
                                            '[{"id": "A", "residue_index": 15, '
                                            '"atom_name": "SG"}, '
                                            '{"id": "L1", "residue_index": 1, '
                                            '"atom_name": "C1"}].'
                                        ),
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {
                                                    "type": "string",
                                                    "description": "Chain id.",
                                                },
                                                "residue_index": {
                                                    "type": "integer",
                                                    "description": (
                                                        "1-based residue index."
                                                    ),
                                                },
                                                "atom_name": {
                                                    "type": "string",
                                                    "description": (
                                                        "Atom name within the "
                                                        "residue, e.g. 'SG'."
                                                    ),
                                                },
                                            },
                                            "required": [
                                                "id",
                                                "residue_index",
                                                "atom_name",
                                            ],
                                        },
                                    },
                                },
                                "required": ["constraint_type"],
                            },
                        },
                        "sequence_ref": {
                            "type": "string",
                            "description": (
                                "Shorthand for a single protein chain: a stored "
                                "sequence reference, used instead of the polymers "
                                "array."
                            ),
                            "x-aevom-reference": {"data_type": "sequence"},
                        },
                        "sequence": {
                            "type": "string",
                            "description": (
                                "Shorthand for a single protein chain: a raw amino "
                                "acid sequence, used instead of the polymers array."
                            ),
                        },
                        "msa_ref": {
                            "type": "string",
                            "description": (
                                "MSA reference from msa_search, applied to the "
                                "first protein chain. Improves prediction quality "
                                "substantially."
                            ),
                            "x-aevom-reference": {"data_type": "msa"},
                        },
                        "recycling_steps": {
                            "type": "integer",
                            "description": (
                                f"Trunk recycles, {MIN_RECYCLING_STEPS}-"
                                f"{MAX_RECYCLING_STEPS} (default "
                                f"{DEFAULT_RECYCLING_STEPS}). More is slower and "
                                "usually better."
                            ),
                        },
                        "sampling_steps": {
                            "type": "integer",
                            "description": (
                                f"Diffusion denoising steps, {MIN_SAMPLING_STEPS}-"
                                f"{MAX_SAMPLING_STEPS} (default "
                                f"{DEFAULT_SAMPLING_STEPS})."
                            ),
                        },
                        "diffusion_samples": {
                            "type": "integer",
                            "description": (
                                "Independent structures to generate, "
                                f"{MIN_DIFFUSION_SAMPLES}-{MAX_DIFFUSION_SAMPLES} "
                                f"(default {DEFAULT_DIFFUSION_SAMPLES}). The "
                                "highest-confidence one is returned as the primary "
                                "reference; the rest are stored too. Diffusion is "
                                "stochastic, so raise this to 5 or more whenever "
                                "the pose is what you are reasoning about -- "
                                "reading off contacts, comparing binding modes, "
                                "deciding which analogue sits better -- then pass "
                                "every returned sample to plip_interactions to see "
                                "which interactions survive across them. One "
                                "sample is enough only when you just need a "
                                "structure to look at or to hand to another tool."
                            ),
                        },
                        "step_scale": {
                            "type": "number",
                            "description": (
                                f"Sampling temperature, {MIN_STEP_SCALE}-"
                                f"{MAX_STEP_SCALE} (default {DEFAULT_STEP_SCALE}). "
                                "Lower gives more diverse structures."
                            ),
                        },
                        "sampling_steps_affinity": {
                            "type": "integer",
                            "description": (
                                "Denoising steps for the affinity head, "
                                f"{MIN_AFFINITY_SAMPLING_STEPS}-"
                                f"{MAX_AFFINITY_SAMPLING_STEPS} (default "
                                f"{DEFAULT_AFFINITY_SAMPLING_STEPS})."
                            ),
                        },
                        "diffusion_samples_affinity": {
                            "type": "integer",
                            "description": (
                                "Affinity samples to average, "
                                f"{MIN_AFFINITY_DIFFUSION_SAMPLES}-"
                                f"{MAX_AFFINITY_DIFFUSION_SAMPLES} (default "
                                f"{DEFAULT_AFFINITY_DIFFUSION_SAMPLES}). Set 1 for "
                                "speed."
                            ),
                        },
                        "affinity_mw_correction": {
                            "type": "boolean",
                            "description": (
                                "Apply the molecular weight correction to the "
                                "affinity prediction. Default false."
                            ),
                        },
                        "without_potentials": {
                            "type": "boolean",
                            "description": (
                                "Skip the physical steering potentials. Faster, but "
                                "allows more steric clashes. Default false."
                            ),
                        },
                        "concatenate_msas": {
                            "type": "boolean",
                            "description": (
                                "Merge the MSAs across chains, for complexes whose "
                                "chains co-evolve. Default false."
                            ),
                        },
                        "write_full_pae": {
                            "type": "boolean",
                            "description": (
                                "Return the full predicted aligned error matrix. "
                                "Stored as an 'array' reference for python_execute; "
                                "use it to judge confidence in the interface "
                                "between two chains. Default false."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short label for the predicted structure (e.g. "
                                "'KRAS G12C with sotorasib')."
                            ),
                        },
                    },
                },
            },
        }

    # --- input resolution ---------------------------------------------------

    @staticmethod
    def _coerce_list(value: Any, field: str) -> list[Any]:
        """Accept an array that the model stringified into JSON."""
        if value is None or value == "":
            return []
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                raise ValueError(
                    f"{field} must be a JSON array of objects, not a string."
                )
        if isinstance(value, dict):
            # A single object where an array was expected is an easy slip to fix.
            return [value]
        if not isinstance(value, list):
            raise ValueError(f"{field} must be an array of objects.")
        return value

    async def _resolve_sequence(self, polymer: dict[str, Any], index: int) -> str:
        """Resolve one polymer's sequence from a reference or a raw string."""
        sequence_ref = polymer.get("sequence_ref") or ""
        raw = polymer.get("sequence") or ""

        if sequence_ref:
            if self._storage is None:
                raise ValueError(
                    "Storage service not available. Cannot retrieve sequence data."
                )
            try:
                content = await self._storage.retrieve(sequence_ref)
            except InvalidReferenceError:
                raise ValueError(
                    f"Invalid sequence reference: '{sequence_ref}'. Use "
                    "fetch_protein_sequence first, which returns a reference like "
                    "'sequence_abc12345'."
                )
            except ArtifactNotFoundError:
                raise ValueError(
                    f"Sequence not found for reference '{sequence_ref}'. The data "
                    "may have expired; re-run fetch_protein_sequence to get a "
                    "fresh reference."
                )
            return content.decode("utf-8").strip().upper()

        if raw:
            return raw.strip().upper()

        raise ValueError(
            f"Polymer {index + 1}: either 'sequence' or 'sequence_ref' is required."
        )

    async def _resolve_msa(self, msa_ref: str) -> dict[str, Any]:
        """Wrap a stored a3m alignment in the payload shape Boltz-2 accepts.

        The alignment field must be named ``alignment``; upstream's older examples
        use ``data``, which the validated NIM rejects with a 422.
        """
        if self._storage is None:
            raise ValueError("Storage service not available. Cannot retrieve MSA data.")
        try:
            content = await self._storage.retrieve(msa_ref)
        except InvalidReferenceError:
            raise ValueError(
                f"Invalid MSA reference: '{msa_ref}'. Use msa_search first to "
                "generate an MSA."
            )
        except ArtifactNotFoundError:
            raise ValueError(
                f"MSA not found for reference '{msa_ref}'. The data may have "
                "expired; re-run msa_search to get a fresh reference."
            )
        a3m = content.decode("utf-8")
        logger.info("Retrieved MSA from %s (%d chars)", msa_ref, len(a3m))
        return {
            "msa_search": {
                "a3m": {
                    "alignment": a3m,
                    "format": "a3m",
                    "rank": 0,
                }
            }
        }

    async def _resolve_template(
        self, template: dict[str, Any], index: int
    ) -> dict[str, Any]:
        """Turn a stored structure reference into a template entry."""
        pdb_ref = template.get("pdb_ref") or ""
        if not pdb_ref:
            raise ValueError(
                f"Structural template {index + 1}: 'pdb_ref' is required."
            )
        if self._storage is None:
            raise ValueError(
                "Storage service not available. Cannot retrieve template structure."
            )
        try:
            content = await self._storage.retrieve(pdb_ref)
        except InvalidReferenceError:
            raise ValueError(
                f"Invalid template reference: '{pdb_ref}'. Pass a stored structure "
                "from fetch_pdb_structure, openfold3 or another structure tool."
            )
        except ArtifactNotFoundError:
            raise ValueError(
                f"Template structure not found for reference '{pdb_ref}'. The data "
                "may have expired or the reference is invalid."
            )

        data_type = (self._storage.get_data_type(pdb_ref) or "pdb").lower()
        entry: dict[str, Any] = {
            "structure": content.decode("utf-8", errors="replace"),
            "format": "cif" if data_type in ("mmcif", "cif") else "pdb",
        }
        if template.get("name"):
            entry["name"] = str(template["name"])
        return entry

    # --- validation ---------------------------------------------------------

    @staticmethod
    def _validate_int(value: Any, field: str, low: int, high: int, default: int) -> int:
        if value is None or value == "":
            return default
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be an integer between {low} and {high}.")
        if not low <= number <= high:
            raise ValueError(
                f"{field} must be between {low} and {high}, got {number}."
            )
        return number

    @staticmethod
    def _validate_float(
        value: Any, field: str, low: float, high: float, default: float
    ) -> float:
        if value is None or value == "":
            return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{field} must be a number between {low} and {high}.")
        if not low <= number <= high:
            raise ValueError(
                f"{field} must be between {low} and {high}, got {number}."
            )
        return number

    @staticmethod
    def _validate_chain_id(chain_id: str, what: str) -> str:
        if not _CHAIN_ID_RE.match(chain_id):
            raise ValueError(
                f"{what}: invalid chain id '{chain_id}'. A chain id is 1-4 "
                "alphanumeric characters, e.g. 'A' for a polymer or 'L1' for a "
                "ligand."
            )
        return chain_id

    @staticmethod
    def _validate_alphabet(sequence: str, molecule_type: str, what: str) -> None:
        alphabet = VALID_ALPHABETS[molecule_type]
        invalid = sorted(set(sequence) - alphabet)
        if invalid:
            expected = {
                "protein": "standard amino acid letters",
                "dna": "A, T, C, G",
                "rna": "A, U, C, G",
            }[molecule_type]
            raise ValueError(
                f"{what}: invalid characters in {molecule_type} sequence: "
                f"{''.join(invalid)}. Use {expected} only."
            )

    def _validate_modifications(
        self, modifications: Any, sequence_length: int, what: str
    ) -> list[dict[str, Any]]:
        entries = self._coerce_list(modifications, f"{what} modifications")
        result = []
        for i, mod in enumerate(entries):
            if not isinstance(mod, dict):
                raise ValueError(f"{what}: modification {i + 1} must be an object.")
            ccd = str(mod.get("ccd") or "").strip().upper()
            if not _CCD_RE.match(ccd):
                raise ValueError(
                    f"{what}: modification {i + 1} has invalid ccd '{ccd}'. A "
                    "residue modification CCD code is 1-3 characters, e.g. 'SEP'."
                )
            try:
                position = int(mod.get("position"))
            except (TypeError, ValueError):
                raise ValueError(
                    f"{what}: modification {i + 1} needs a 1-based integer "
                    "'position'."
                )
            if not 1 <= position <= sequence_length:
                raise ValueError(
                    f"{what}: modification {i + 1} is at position {position}, but "
                    f"the chain has {sequence_length} residues. Positions are "
                    "1-based."
                )
            result.append({"ccd": ccd, "position": position})
        return result

    def _validate_constraints(
        self, constraints: Any, known_ids: set[str], ligand_ids: set[str] | None = None
    ) -> list[dict[str, Any]]:
        """Check constraints reference declared chains before sending them.

        A typo'd chain id comes back as a 422 that does not say which id was
        wrong, and the model then has to guess which of several it got wrong.
        """
        entries = self._coerce_list(constraints, "constraints")
        known_list = ", ".join(sorted(known_ids)) or "none"
        result = []

        for i, raw in enumerate(entries):
            what = f"Constraint {i + 1}"
            raw = _normalise_constraint(raw, ligand_ids)
            if not isinstance(raw, dict):
                raise ValueError(f"{what} must be an object.")
            kind = str(raw.get("constraint_type") or "").strip().lower()

            if kind == "pocket":
                binder = str(raw.get("binder") or "").strip()
                if not binder:
                    raise ValueError(
                        f"{what}: a pocket constraint needs 'binder', the chain id "
                        "of the ligand to steer."
                    )
                if binder not in known_ids:
                    raise ValueError(
                        f"{what}: binder '{binder}' is not one of the declared "
                        f"chain ids ({known_list}). Give the ligand an explicit "
                        "'id' and use that."
                    )
                contacts = self._coerce_list(raw.get("contacts"), f"{what} contacts")
                if not contacts:
                    raise ValueError(
                        f"{what}: a pocket constraint needs at least one contact "
                        "residue."
                    )
                # A chain named once beside the residue list applies to all of
                # them: `{"chain": "A", "residues": [22, 23]}` is how the model
                # writes a pocket in one chain, 38 times in the traces. With no
                # chain named at all and a single polymer in the request, the
                # residues can only be in that one -- another 25 calls.
                default_chain = str(raw.get("chain") or raw.get("chain_id") or "").strip() or None
                if default_chain is None:
                    polymer_ids = known_ids - (ligand_ids or set())
                    if len(polymer_ids) == 1:
                        default_chain = next(iter(polymer_ids))
                pocket: dict[str, Any] = {
                    "constraint_type": "pocket",
                    "binder": binder,
                    "contacts": [
                        self._validate_residue_ref(
                            c, j, what, known_ids, default_chain=default_chain
                        )
                        for j, c in enumerate(contacts)
                    ],
                }
                # A pocket without these is advisory, and the difference is
                # measurable: forcing a decoy pocket on the hosted service
                # moved the predicted affinity by +0.42 log units against a
                # run-to-run spread of 0.08, with no overlap over three
                # repeats each -- so the service honours them rather than
                # accepting and ignoring them.
                max_distance = raw.get("max_distance")
                if max_distance is not None:
                    pocket["max_distance"] = self._validate_float(
                        max_distance, f"{what} max_distance", 0.0, 100.0, 6.0
                    )
                if raw.get("force") is not None:
                    pocket["force"] = bool(raw.get("force"))
                result.append(pocket)

            elif kind == "bond":
                atoms = self._coerce_list(raw.get("atoms"), f"{what} atoms")
                if len(atoms) != 2:
                    raise ValueError(
                        f"{what}: a bond constraint needs exactly two atoms, got "
                        f"{len(atoms)}."
                    )
                bonded = []
                for j, atom in enumerate(atoms):
                    entry = self._validate_residue_ref(atom, j, what, known_ids)
                    atom_name = str((atom or {}).get("atom_name") or "").strip()
                    if not atom_name:
                        raise ValueError(
                            f"{what}: atom {j + 1} needs 'atom_name', e.g. 'SG'."
                        )
                    entry["atom_name"] = atom_name
                    bonded.append(entry)
                result.append({"constraint_type": "bond", "atoms": bonded})

            else:
                raise ValueError(
                    f"{what}: constraint_type must be 'pocket' or 'bond', got "
                    f"'{kind}'."
                )

        return result

    @staticmethod
    def _validate_residue_ref(
        entry: Any,
        index: int,
        what: str,
        known_ids: set[str],
        default_chain: str | None = None,
    ) -> dict[str, Any]:
        entry = _normalise_residue_ref(entry, known_ids, default_chain)
        if not isinstance(entry, dict):
            raise ValueError(f"{what}: entry {index + 1} must be an object.")
        chain_id = str(entry.get("id") or "").strip()
        if chain_id not in known_ids:
            known_list = ", ".join(sorted(known_ids)) or "none"
            raise ValueError(
                f"{what}: entry {index + 1} refers to chain '{chain_id}', which is "
                f"not one of the declared chain ids ({known_list})."
            )
        try:
            residue_index = int(entry.get("residue_index"))
        except (TypeError, ValueError):
            raise ValueError(
                f"{what}: entry {index + 1} needs a 1-based integer "
                "'residue_index'."
            )
        if residue_index < 1:
            raise ValueError(
                f"{what}: entry {index + 1} has residue_index {residue_index}; "
                "indices are 1-based."
            )
        return {"id": chain_id, "residue_index": residue_index}

    # --- request assembly ---------------------------------------------------

    async def _build_polymers(
        self, polymers: list[Any], msa_ref: str
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Validate and convert the polymers array into API form."""
        if not polymers:
            raise ValueError(
                "At least one polymer is required. Pass 'polymers', or "
                "'sequence_ref' for a single protein."
            )
        if len(polymers) > MAX_POLYMERS:
            raise ValueError(
                f"Boltz-2 accepts at most {MAX_POLYMERS} polymer chains, got "
                f"{len(polymers)}."
            )

        api_polymers: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {
            "molecule_types": [],
            "chain_ids": [],
            "protein_count": 0,
        }
        # Ids the caller named explicitly are reserved up front, so an
        # auto-assigned id never lands on one that a later chain claims.
        used_ids = self._explicit_ids(polymers)
        assigned_ids: set[str] = set()
        next_id = ord("A")
        top_level_msa_available = bool(msa_ref)

        for index, raw in enumerate(polymers):
            what = f"Polymer {index + 1}"
            raw = _apply_aliases(raw, _POLYMER_ALIASES)
            if not isinstance(raw, dict):
                raise ValueError(f"{what} must be an object.")

            # Resolved first so an omitted molecule_type can be read off the
            # sequence: two-thirds of the polymers in the benchmark traces named
            # a sequence and no type, and each one cost an oracle call.
            sequence = await self._resolve_sequence(raw, index)
            if not sequence:
                raise ValueError(f"{what}: the sequence is empty.")

            molecule_type = str(raw.get("molecule_type") or "").strip().lower()
            if not molecule_type:
                molecule_type = infer_molecule_type(sequence) or ""
            if molecule_type not in VALID_ALPHABETS:
                raise ValueError(
                    f"{what}: molecule_type must be 'protein', 'dna' or 'rna', got "
                    f"'{molecule_type}'. This sequence could be more than one of "
                    "them, so it has to be stated."
                )
            if len(sequence) > MAX_SEQUENCE_LENGTH:
                raise ValueError(
                    f"{what}: sequence is {len(sequence)} residues, over the "
                    f"{MAX_SEQUENCE_LENGTH} limit. Predict a domain rather than the "
                    "full-length chain."
                )
            self._validate_alphabet(sequence, molecule_type, what)

            if raw.get("id"):
                chain_id = self._validate_chain_id(str(raw["id"]).strip(), what)
                if chain_id in assigned_ids:
                    raise ValueError(
                        f"{what}: chain id '{chain_id}' is used more than once. "
                        "Every chain needs a distinct id."
                    )
            else:
                while chr(next_id) in used_ids or chr(next_id) in assigned_ids:
                    next_id += 1
                chain_id = chr(next_id)
                next_id += 1
            assigned_ids.add(chain_id)

            entry: dict[str, Any] = {
                "id": chain_id,
                "molecule_type": molecule_type,
                "sequence": sequence,
            }
            if raw.get("cyclic"):
                entry["cyclic"] = True

            chain_msa_ref = raw.get("msa_ref") or ""
            if chain_msa_ref and molecule_type != "protein":
                raise ValueError(
                    f"{what}: an MSA can only be supplied for a protein chain."
                )
            # The top-level msa_ref is a convenience for the common single-protein
            # case, so it applies to the first protein chain and is then spent.
            if not chain_msa_ref and molecule_type == "protein" and top_level_msa_available:
                chain_msa_ref = msa_ref
                top_level_msa_available = False
            if chain_msa_ref:
                entry["msa"] = await self._resolve_msa(chain_msa_ref)
                metadata["msa_ref"] = chain_msa_ref

            modifications = self._validate_modifications(
                raw.get("modifications"), len(sequence), what
            )
            if modifications:
                entry["modifications"] = modifications

            templates = self._coerce_list(
                raw.get("structural_templates"), f"{what} structural_templates"
            )
            if templates:
                entry["structural_templates"] = [
                    await self._resolve_template(t, i) for i, t in enumerate(templates)
                ]

            api_polymers.append(entry)
            metadata["molecule_types"].append(molecule_type)
            metadata["chain_ids"].append(chain_id)
            metadata[f"{molecule_type}_{chain_id}_length"] = len(sequence)
            if molecule_type == "protein":
                metadata["protein_count"] += 1

        return api_polymers, metadata

    @staticmethod
    def _explicit_ids(entries: list[Any]) -> set[str]:
        """Chain ids the caller named, gathered before any are auto-assigned.

        Aliases are resolved here too. A chain named with ``chain`` rather than
        ``id`` would otherwise be reserved by nobody, and a later chain could be
        auto-assigned the same letter.
        """
        named = (_apply_aliases(entry, _POLYMER_ALIASES) for entry in entries)
        return {
            str(entry["id"]).strip()
            for entry in named
            if isinstance(entry, dict) and entry.get("id")
        }

    def _build_ligands(
        self, ligands: list[Any], polymer_ids: set[str]
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Validate and convert the ligands array into API form."""
        if len(ligands) > MAX_LIGANDS:
            raise ValueError(
                f"Boltz-2 accepts at most {MAX_LIGANDS} ligands, got "
                f"{len(ligands)}."
            )

        api_ligands: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {
            "ligand_count": 0,
            "affinity_ligand_id": "",
            "ligand_ids": [],
        }
        reserved = polymer_ids | self._explicit_ids(ligands)
        assigned_ids: set[str] = set()
        next_index = 1

        for index, raw in enumerate(ligands):
            what = f"Ligand {index + 1}"
            if not isinstance(raw, dict):
                raise ValueError(f"{what} must be an object.")

            smiles = str(raw.get("smiles") or "").strip()
            ccd = str(raw.get("ccd") or "").strip().upper()
            if smiles and ccd:
                raise ValueError(
                    f"{what}: give either 'smiles' or 'ccd', not both."
                )
            if not smiles and not ccd:
                raise ValueError(f"{what}: needs either 'smiles' or 'ccd'.")

            if raw.get("id"):
                chain_id = self._validate_chain_id(str(raw["id"]).strip(), what)
                if chain_id in polymer_ids or chain_id in assigned_ids:
                    raise ValueError(
                        f"{what}: chain id '{chain_id}' is used more than once. "
                        "Every chain needs a distinct id."
                    )
            else:
                while f"L{next_index}" in reserved or f"L{next_index}" in assigned_ids:
                    next_index += 1
                chain_id = f"L{next_index}"
                next_index += 1
            assigned_ids.add(chain_id)

            entry: dict[str, Any] = {"id": chain_id}
            if smiles:
                if not _SMILES_RE.match(smiles):
                    raise ValueError(
                        f"{what}: invalid SMILES '{smiles}'. Check for invalid "
                        "characters."
                    )
                entry["smiles"] = smiles
                metadata[f"ligand_{chain_id}_smiles"] = smiles
            else:
                if not _CCD_RE.match(ccd):
                    raise ValueError(
                        f"{what}: invalid CCD code '{ccd}'. A ligand CCD code is "
                        "1-3 alphanumeric characters, e.g. 'ATP'. Use "
                        "lookup_ccd_code to find one, or pass 'smiles' instead."
                    )
                entry["ccd"] = ccd
                metadata[f"ligand_{chain_id}_ccd"] = ccd

            if raw.get("predict_affinity"):
                if metadata["affinity_ligand_id"]:
                    raise ValueError(
                        "Boltz-2 predicts affinity for only one ligand per "
                        f"request, but both '{metadata['affinity_ligand_id']}' and "
                        f"'{chain_id}' set predict_affinity. Make one call per "
                        "ligand to rank several."
                    )
                entry["predict_affinity"] = True
                metadata["affinity_ligand_id"] = chain_id

            api_ligands.append(entry)
            metadata["ligand_ids"].append(chain_id)
            metadata["ligand_count"] += 1

        return api_ligands, metadata

    # --- response handling --------------------------------------------------

    @staticmethod
    def _first_number(values: Any) -> Optional[float]:
        """The consensus value from an affinity field, which arrives as a list."""
        if isinstance(values, (int, float)) and not isinstance(values, bool):
            return float(values)
        if isinstance(values, list) and values:
            try:
                return float(values[0])
            except (TypeError, ValueError):
                return None
        return None

    @classmethod
    def _summarize_affinity(cls, entry: Any) -> dict[str, float]:
        """Pull the reportable affinity numbers out of one ligand's entry.

        ``affinity_pred_value`` is log10 of IC50 in micromolar. That is the
        only affinity number here with an unambiguous definition, every other
        one is derived from it, and it is the quantity anything comparing two
        molecules should compare.

        The service's own ``affinity_pic50`` field is not a pIC50 and is not
        used. Measured across 700 affinity read-outs in two full benchmark
        runs, the field equals ``(6 - affinity_pred_value) * 1.364`` to within
        display rounding on 681 of them -- a binding free energy in kcal/mol,
        about 1.364x a pIC50. Reporting it as a pIC50 put a number in front of
        the model that disagreed with the log10 printed beside it by 2.14 on
        median, in the direction of looking far more potent than predicted. So
        pIC50 is derived, always, and the two numbers now satisfy
        ``pic50 + log_ic50_um == 6`` by construction.
        """
        if not isinstance(entry, dict):
            return {}

        summary: dict[str, float] = {}
        pred = cls._first_number(entry.get("affinity_pred_value"))
        probability = cls._first_number(entry.get("affinity_probability_binary"))

        if pred is not None:
            summary["log_ic50_um"] = pred
            summary["pic50"] = 6.0 - pred
            try:
                summary["ic50_um"] = 10.0**pred
            except OverflowError:
                pass
        if probability is not None:
            summary["binding_probability"] = probability
        return summary

    @staticmethod
    def _npz_bytes(value: Any) -> Optional[tuple[bytes, tuple[int, ...]]]:
        """Pack a numeric, array-shaped response field as an npz.

        Returns None for anything that is not a numeric array of interesting
        size, so the caller never advertises a reference to an empty artifact.
        """
        if not isinstance(value, list) or not value:
            return None
        try:
            import numpy as np

            array = np.asarray(value, dtype=np.float32)
        except Exception:  # noqa: BLE001 - a ragged or non-numeric field is not fatal
            return None
        if array.size < _MIN_ARRAY_ELEMENTS or not np.isfinite(array).any():
            return None

        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=array)
        return buffer.getvalue(), tuple(array.shape)

    @classmethod
    def _collect_arrays(cls, result: dict[str, Any]) -> list[tuple[str, bytes, tuple]]:
        """Find numeric matrices in the response that no named field claimed.

        PAE matrices (when ``write_full_pae`` is set) and per-residue confidences
        are the reason this exists: discarding them loses the only copy, and
        putting thousands of numbers in context is worse than useless. The
        response schema does not name these fields, so they are discovered rather
        than looked up.
        """
        found: list[tuple[str, bytes, tuple]] = []

        for key, value in result.items():
            if key in _CLAIMED_RESPONSE_KEYS:
                continue
            packed = cls._npz_bytes(value)
            if packed is not None:
                found.append((key, packed[0], packed[1]))

        for index, structure in enumerate(result.get("structures") or []):
            if not isinstance(structure, dict):
                continue
            for key, value in structure.items():
                if key in _CLAIMED_STRUCTURE_KEYS:
                    continue
                packed = cls._npz_bytes(value)
                if packed is not None:
                    found.append((f"{key}_{index}", packed[0], packed[1]))

        return found

    @staticmethod
    def _rank_structures(
        structures: list[Any], scores: list[Any]
    ) -> list[tuple[int, str, Optional[float]]]:
        """Order returned structures best-confidence first.

        ``confidence_scores`` is positional against ``structures``. When the two
        disagree in length the pairing is untrustworthy, so the original order is
        kept and no score is attributed to any structure rather than attaching
        one structure's confidence to another.
        """
        paired: list[tuple[int, str, Optional[float]]] = []
        aligned = len(scores) == len(structures)

        for index, structure in enumerate(structures):
            if not isinstance(structure, dict):
                continue
            content = structure.get("structure") or ""
            if not content:
                continue
            score: Optional[float] = None
            if aligned:
                try:
                    score = float(scores[index])
                except (TypeError, ValueError):
                    score = None
            paired.append((index, content, score))

        paired.sort(key=lambda item: (item[2] is None, -(item[2] or 0.0), item[0]))
        return paired

    # --- execution ----------------------------------------------------------

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Run a Boltz-2 prediction and store the results."""
        description = arguments.get("description")

        try:
            polymers = self._coerce_list(arguments.get("polymers"), "polymers")
            ligands = self._coerce_list(arguments.get("ligands"), "ligands")
            msa_ref = arguments.get("msa_ref") or ""

            # Single-protein shorthand, for parity with openfold3's simple mode.
            if not polymers:
                sequence_ref = arguments.get("sequence_ref") or ""
                raw_sequence = arguments.get("sequence") or ""
                if not sequence_ref and not raw_sequence:
                    return ToolResult.error(
                        "Provide 'polymers' for the chains in the complex, or "
                        "'sequence_ref' for a single protein."
                    )
                polymers = [
                    {
                        "molecule_type": "protein",
                        "sequence_ref": sequence_ref,
                        "sequence": raw_sequence,
                    }
                ]

            api_polymers, poly_metadata = await self._build_polymers(polymers, msa_ref)
            polymer_ids = set(poly_metadata["chain_ids"])
            api_ligands, ligand_metadata = self._build_ligands(ligands, polymer_ids)
            # Constraints address ligands as well as polymers, so they are checked
            # against every id in the request.
            api_constraints = self._validate_constraints(
                arguments.get("constraints"),
                polymer_ids | set(ligand_metadata["ligand_ids"]),
                set(ligand_metadata["ligand_ids"]),
            )

            recycling_steps = self._validate_int(
                arguments.get("recycling_steps"),
                "recycling_steps",
                MIN_RECYCLING_STEPS,
                MAX_RECYCLING_STEPS,
                DEFAULT_RECYCLING_STEPS,
            )
            sampling_steps = self._validate_int(
                arguments.get("sampling_steps"),
                "sampling_steps",
                MIN_SAMPLING_STEPS,
                MAX_SAMPLING_STEPS,
                DEFAULT_SAMPLING_STEPS,
            )
            diffusion_samples = self._validate_int(
                arguments.get("diffusion_samples"),
                "diffusion_samples",
                MIN_DIFFUSION_SAMPLES,
                MAX_DIFFUSION_SAMPLES,
                DEFAULT_DIFFUSION_SAMPLES,
            )
            step_scale = self._validate_float(
                arguments.get("step_scale"),
                "step_scale",
                MIN_STEP_SCALE,
                MAX_STEP_SCALE,
                DEFAULT_STEP_SCALE,
            )
            affinity_sampling_steps = self._validate_int(
                arguments.get("sampling_steps_affinity"),
                "sampling_steps_affinity",
                MIN_AFFINITY_SAMPLING_STEPS,
                MAX_AFFINITY_SAMPLING_STEPS,
                DEFAULT_AFFINITY_SAMPLING_STEPS,
            )
            affinity_diffusion_samples = self._validate_int(
                arguments.get("diffusion_samples_affinity"),
                "diffusion_samples_affinity",
                MIN_AFFINITY_DIFFUSION_SAMPLES,
                MAX_AFFINITY_DIFFUSION_SAMPLES,
                DEFAULT_AFFINITY_DIFFUSION_SAMPLES,
            )

            affinity_ligand_id = ligand_metadata["affinity_ligand_id"]

            request_data: dict[str, Any] = {
                "polymers": api_polymers,
                "recycling_steps": recycling_steps,
                "sampling_steps": sampling_steps,
                "diffusion_samples": diffusion_samples,
                "step_scale": step_scale,
                # mmCIF is the only format the service produces, so it is set
                # here rather than exposed as a parameter that can only be wrong.
                "output_format": "mmcif",
            }
            if api_ligands:
                request_data["ligands"] = api_ligands
            if api_constraints:
                request_data["constraints"] = api_constraints
            if affinity_ligand_id:
                request_data["sampling_steps_affinity"] = affinity_sampling_steps
                request_data["diffusion_samples_affinity"] = affinity_diffusion_samples
            if arguments.get("affinity_mw_correction"):
                request_data["affinity_mw_correction"] = True
            if arguments.get("without_potentials"):
                request_data["without_potentials"] = True
            if arguments.get("concatenate_msas"):
                request_data["concatenate_msas"] = True
            if arguments.get("write_full_pae"):
                request_data["write_full_pae"] = True

            logger.info(
                "Calling Boltz-2: %d polymers, %d ligands, affinity=%s, samples=%d",
                len(api_polymers),
                len(api_ligands),
                affinity_ligand_id or "no",
                diffusion_samples,
            )

            result = await self._call_nvidia_api(
                request_data,
                poll_seconds=POLL_SECONDS,
                service_name="Boltz-2",
            )

            structures = result.get("structures") or []
            if not structures:
                return ToolResult.error(
                    "Boltz-2 returned no structures. The prediction may have "
                    "failed; try again, or reduce the size of the request."
                )

            ranked = self._rank_structures(
                structures, list(result.get("confidence_scores") or [])
            )
            if not ranked:
                return ToolResult.error(
                    "Boltz-2 returned structures with no coordinate data."
                )

            if self._storage is None:
                return ToolResult.error(
                    "Storage service not available. Cannot store structure data."
                )

            return await self._store_and_report(
                ranked=ranked,
                result=result,
                description=description,
                poly_metadata=poly_metadata,
                ligand_metadata=ligand_metadata,
                request_summary={
                    "recycling_steps": recycling_steps,
                    "sampling_steps": sampling_steps,
                    "diffusion_samples": diffusion_samples,
                    "step_scale": step_scale,
                },
                constraint_count=len(api_constraints),
            )

        except ValueError as exc:
            logger.error("Boltz-2 validation error: %s", exc)
            # Nothing was sent to the endpoint, so the call is not charged.
            return ToolResult.invalid_arguments(str(exc))
        except NvidiaApiError as exc:
            logger.error("Boltz-2 API error: %s", exc)
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool must not break the turn
            logger.exception("Failed to predict structure with Boltz-2")
            return ToolResult.error(f"Failed to predict structure: {exc}")

    async def _store_and_report(
        self,
        *,
        ranked: list[tuple[int, str, Optional[float]]],
        result: dict[str, Any],
        description: Optional[str],
        poly_metadata: dict[str, Any],
        ligand_metadata: dict[str, Any],
        request_summary: dict[str, Any],
        constraint_count: int,
    ) -> ToolResult:
        """Store every artifact the response carries and describe them."""
        _, best_structure, best_score = ranked[0]

        has_ligand = ligand_metadata["ligand_count"] > 0
        if not description:
            if has_ligand:
                description = "Boltz-2 protein-ligand complex"
            elif poly_metadata["protein_count"] > 1:
                description = (
                    f"Boltz-2 {poly_metadata['protein_count']}-chain complex"
                )
            else:
                description = "Boltz-2 predicted structure"

        affinity_ligand_id = ligand_metadata["affinity_ligand_id"]
        affinities = result.get("affinities") or {}
        affinity = self._summarize_affinity(affinities.get(affinity_ligand_id))

        # Anything the service wanted to say about how it produced this answer.
        # A deployment serving Boltz itself knows things the caller cannot see --
        # that it folded from a single sequence because it had no alignment, or
        # that it answered from a cheap tier whose ranking is trustworthy and
        # whose value is not. Dropping that was a measurable mistake: a run
        # screened at one diffusion sample, read a binding probability of 0.38,
        # and was scored at ten samples as 0.89. The service said "confirm a
        # finalist at diffusion_samples=10 before relying on the value" and
        # nothing carried it to the model.
        service_notes = [
            str(note) for note in (result.get("warnings") or []) if str(note).strip()
        ]

        storage_metadata: dict[str, Any] = {
            "source": "boltz2",
            "output_format": "mmcif",
            "confidence_score": best_score,
            "num_structures": len(ranked),
            "constraint_count": constraint_count,
            **poly_metadata,
            **ligand_metadata,
            **request_summary,
        }
        if affinity:
            storage_metadata.update(
                {f"affinity_{key}": value for key, value in affinity.items()}
            )

        reference = await self._storage.store(
            data_type="mmcif",
            content=best_structure.encode("utf-8"),
            metadata=storage_metadata,
            description=description,
        )
        logger.info("Stored Boltz-2 structure as %s", reference)

        outputs: dict[str, str] = {"structure": reference}
        extra_refs: list[tuple[str, Optional[float]]] = []
        for rank, (index, content, score) in enumerate(ranked[1:], start=2):
            extra_ref = await self._storage.store(
                data_type="mmcif",
                content=content.encode("utf-8"),
                metadata={
                    "source": "boltz2",
                    "output_format": "mmcif",
                    "confidence_score": score,
                    "sample_index": index,
                    "primary_ref": reference,
                },
                description=f"{description} (sample {rank} of {len(ranked)})",
            )
            outputs[f"structure_{rank}"] = extra_ref
            extra_refs.append((extra_ref, score))

        array_refs: list[tuple[str, str, tuple]] = []
        for field, payload, shape in self._collect_arrays(result):
            array_ref = await self._storage.store(
                data_type="array",
                content=payload,
                metadata={
                    "source": "boltz2",
                    "field": field,
                    "shape": str(shape),
                    "structure_ref": reference,
                },
                description=f"Boltz-2 {field} for {reference}",
            )
            outputs[field] = array_ref
            array_refs.append((field, array_ref, shape))

        storage_metadata["outputs"] = outputs
        storage_metadata["all_references"] = list(outputs.values())

        message = format_storage_message(
            data_type="mmcif",
            reference=reference,
            source_description=description,
            accepting_tools=["visualize_pdb", "convert_structure"],
        )

        chains = ", ".join(
            f"{chain}:{mol_type}"
            for chain, mol_type in zip(
                poly_metadata["chain_ids"], poly_metadata["molecule_types"]
            )
        )
        message += "\n\nPrediction summary:\n"
        message += f"- Chains: {chains}\n"
        if has_ligand:
            message += f"- Ligands: {ligand_metadata['ligand_count']}\n"
        requested_samples = request_summary.get("diffusion_samples")
        if best_score is not None:
            if len(ranked) > 1:
                message += (
                    f"- Confidence: {best_score:.3f} (0-1, higher is better; this "
                    f"is the best of {len(ranked)} samples)\n"
                )
            else:
                message += (
                    f"- Confidence: {best_score:.3f} (0-1, higher is better)\n"
                )
        else:
            message += "- Confidence: not reported for this prediction\n"
        if best_score is not None:
            # Pre-empt a common failure: asked for "mean confidence / pLDDT", the
            # model ignores this score and tries to recover a per-residue profile
            # from the mmCIF with biotite, which misfires (e.g. AttributeError on
            # biotite.structure.io.pdbx). This score IS the summary answer; the
            # per-residue profile, if actually needed, is the B-factor column.
            message += (
                "- This is Boltz-2's overall confidence for the structure (its "
                "mean-pLDDT equivalent) -- report it directly for a 'mean "
                "confidence' question. The per-residue profile lives in the mmCIF "
                "B-factor column; to read it, convert to PDB with convert_structure "
                "then aevom.load_pdb_structure(ref) (each atom carries "
                "'temp_factor'). Do not hand-parse the mmCIF with biotite.\n"
            )
        if requested_samples == 1:
            # Diffusion is stochastic, so one sample is one draw. Saying so is
            # the difference between a reader treating the pose as settled and
            # treating it as provisional -- and the confidence score does not
            # carry that information, since a single sample can score well and
            # still not be reproducible.
            message += (
                "- Sampling: 1 diffusion sample, so this pose is a single draw "
                "from a stochastic process. Where the pose itself is the answer "
                "-- reading off interactions, comparing binding modes -- pass "
                "diffusion_samples=5 or more and give every returned sample to "
                "plip_interactions, which reports which contacts hold across "
                "them.\n"
            )
        if constraint_count:
            message += f"- Constraints applied: {constraint_count}\n"

        if affinity:
            message += f"\nPredicted affinity for ligand {affinity_ligand_id}:\n"
            # log10(IC50) leads, because it is the number to compare molecules
            # on and the one every other line here is derived from. pIC50 is a
            # convenience, and putting it first invited reasoning about a
            # quantity that is only ever restating this one.
            if "log_ic50_um" in affinity:
                message += (
                    f"- log10(IC50 in uM): {affinity['log_ic50_um']:.2f} "
                    "(lower binds tighter; this is the number to compare "
                    "candidates on)\n"
                )
            if "ic50_um" in affinity:
                message += f"- Implied IC50: {affinity['ic50_um']:.3g} uM\n"
            if "pic50" in affinity:
                message += f"- pIC50: {affinity['pic50']:.2f} (higher binds tighter)\n"
            if "binding_probability" in affinity:
                message += (
                    f"- Probability of binding: "
                    f"{affinity['binding_probability']:.2f}\n"
                )
            message += (
                "These are model estimates for ranking and triage, not measured "
                "IC50 values. Confirm a promising result with vina_dock or an "
                "orthogonal method before drawing conclusions.\n"
            )
        elif affinity_ligand_id:
            message += (
                f"\nAffinity was requested for ligand {affinity_ligand_id} but the "
                "response carried no affinity values.\n"
            )

        if extra_refs:
            message += (
                f"\n{len(extra_refs)} further diffusion "
                f"{'sample' if len(extra_refs) == 1 else 'samples'} stored:\n"
            )
            for extra_ref, score in extra_refs:
                shown = f"{score:.3f}" if score is not None else "n/a"
                message += f"- '{extra_ref}' (confidence {shown})\n"

        if array_refs:
            message += "\nNumeric outputs, too large to show:\n"
            for field, array_ref, shape in array_refs:
                message += f"- {field}, shape {shape}, stored as '{array_ref}'\n"
            message += (
                "Read these in python_execute with "
                f"aevom.load_array('{array_refs[0][1]}').\n"
            )

        # Placed before the format footer so it is not the thing most likely to
        # be truncated: what the service says about how it produced the number
        # changes whether the number can be used.
        if service_notes:
            message += "\n\nFrom the prediction service:\n"
            for note in service_notes:
                message += f"- {note}\n"

        message += (
            f"\nThis structure is mmCIF, the only format Boltz-2 emits. "
            f"visualize_pdb('{reference}') displays it directly; for vina_dock or "
            f"proteinmpnn_design_sequence, run "
            f"convert_structure(structure_ref='{reference}', target_format='pdb') "
            "first."
        )

        return ToolResult.data_reference(
            reference=reference,
            data_type="mmcif",
            message=message,
            metadata=storage_metadata,
        )
