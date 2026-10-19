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

"""Heavy data type detector for MCP tool schemas.

This module detects chem/bio heavy data types (SMILES, SDF, PDB, sequences, etc.)
in MCP tool schemas by examining parameter names and descriptions. It also provides
runtime content detection for tool outputs.

The detector is designed with a pluggable interface so that regex-based detection
(v1) can later be supplemented or replaced by LLM-based soft detection.
"""

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from api.storage.datatypes import resolve as resolve_data_type

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result of detecting a heavy data type in a tool schema parameter or output.

    Attributes:
        param_name: Name of the parameter that matched (for inputs) or a
            descriptive label like "output" (for outputs).
        detected_type: The Aevom storage data type (e.g., "smiles", "sdf", "pdb").
        confidence: Detection confidence score between 0.0 and 1.0.
        direction: Whether this detection applies to "input" or "output".
    """

    param_name: str
    detected_type: str
    confidence: float
    direction: str  # "input" or "output"

    def __repr__(self) -> str:
        return (
            f"DetectionResult({self.direction} '{self.param_name}' -> "
            f"{self.detected_type} @ {self.confidence:.2f})"
        )


@dataclass
class HeavyTypePattern:
    """Pattern definition for a single heavy data type.

    Attributes:
        data_type: The Aevom storage type identifier (e.g., "smiles", "sdf").
        name_patterns: Compiled regex patterns to match against parameter names.
        desc_patterns: Compiled regex patterns to match against parameter descriptions.
        output_desc_patterns: Compiled regex patterns to match tool descriptions
            for output detection (e.g., "returns SDF", "outputs PDB").
        content_patterns: Compiled regex patterns for runtime content sniffing
            of tool output to detect heavy data in the actual response.
        name_score: Confidence score awarded for a name match.
        desc_score: Confidence score awarded for a description match.
        confidence_threshold: Minimum confidence to consider a detection valid.
    """

    data_type: str
    name_patterns: list[re.Pattern] = field(default_factory=list)
    desc_patterns: list[re.Pattern] = field(default_factory=list)
    output_desc_patterns: list[re.Pattern] = field(default_factory=list)
    content_patterns: list[re.Pattern] = field(default_factory=list)
    name_score: float = 0.8
    desc_score: float = 0.5
    confidence_threshold: float = 0.5


def _compile_patterns(
    patterns: list[str], flags: int = re.IGNORECASE
) -> list[re.Pattern]:
    """Compile a list of regex pattern strings."""
    return [re.compile(p, flags) for p in patterns]


# ---------------------------------------------------------------------------
# Pattern library
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# SMILES content detection helpers
# ---------------------------------------------------------------------------

# A single SMILES token: atom letters + structural chars, no spaces.
# Must contain at least one common atom letter AND at least one structural
# character (parentheses, digits, =, #, brackets, @, /, \) to avoid matching
# plain English words. Minimum 5 chars to skip trivial molecules.
_SMILES_TOKEN_RE = re.compile(
    r"""
    (?:^|(?<=[\s,\[\]"']))         # preceded by whitespace, comma, bracket, or quote
    (
        [A-Za-z0-9@+\-\[\]\(\)\\\/=\#\$.\%\:\~]{5,}   # SMILES char class, 5+ chars
    )
    (?=$|[\s,\[\]"'])              # followed by whitespace, comma, bracket, or quote
    """,
    re.VERBOSE | re.MULTILINE,
)

# Primary structural characters that distinguish SMILES from plain words.
# Parentheses (branches), square brackets (bracket atoms), and digits (ring
# closures) are highly characteristic of SMILES and rarely appear together
# in ordinary English/scientific prose.
_SMILES_PRIMARY_STRUCTURAL = set("()[]0123456789")

# Secondary structural characters that, combined with other evidence,
# strengthen a SMILES classification.
_SMILES_SECONDARY_STRUCTURAL = set("=#@/\\")

# Valid SMILES atom letters (both aliphatic uppercase and aromatic lowercase).
# Used for atom ratio calculations.
_SMILES_ATOM_CHARS = set("CcNnOoSsPpFBIH")

# Characters that are valid in SMILES outside of bracket atoms.
# Anything else (commas, semicolons, most lowercase letters) is a red flag.
_SMILES_VALID_CHARS = set(
    "BCFHINOPSbcnops"  # atom letters
    "0123456789"  # ring closures
    "()[]"  # branches, bracket atoms
    "-=:#~"  # bonds
    "@"  # chirality
    "/\\"  # stereo
    ".+%"  # disconnection, charge, 2-digit rings
    "$"  # quadruple bond (rare)
)

# Regex to reject tokens that look like numbers (possibly signed, with
# decimal point, scientific notation). These are common in tool outputs
# (scores, energies, etc.) and should never be classified as SMILES.
_NUMERIC_RE = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def _looks_like_smiles(token: str) -> bool:
    """Heuristic: does this token look like a SMILES string?

    Requirements:
    - At least 5 characters
    - No whitespace (spaces, tabs, newlines)
    - Not a numeric value (score, energy, etc.)
    - Contains only characters valid in SMILES notation
    - Contains at least one common organic atom letter (C, c, N, n, O, o, S, s, etc.)
    - Contains at least one *primary* structural character: parentheses,
      brackets, or ring-closure digits
    - Atom letters make up at least 25% of the token (rejects score-heavy tokens)
    - Parentheses are balanced (basic SMILES syntax validity)
    """
    # Basic length and whitespace checks
    if len(token) < 5:
        return False
    if any(ch in token for ch in " \t\r\n"):
        return False

    # Reject pure numeric tokens (scores like -36.545, energies like 1.23e-4)
    if _NUMERIC_RE.match(token):
        return False

    # Check that all characters are valid SMILES chars (outside brackets).
    # This rejects tokens with commas, semicolons, non-SMILES lowercase letters.
    # We do allow bracket atoms like [Fe], [Cu2+] -- skip bracket contents.
    in_bracket = False
    for ch in token:
        if ch == "[":
            in_bracket = True
        elif ch == "]":
            in_bracket = False
        elif not in_bracket and ch not in _SMILES_VALID_CHARS:
            return False

    # Must have atom letters
    atom_count = sum(1 for ch in token if ch in _SMILES_ATOM_CHARS)
    if atom_count == 0:
        return False

    # Atom ratio: at least 25% of characters should be atom letters.
    # Real SMILES have high atom ratios (caffeine ~49%, aspirin ~57%).
    # Garbage tokens from scores/labels typically have very low ratios.
    atom_ratio = atom_count / len(token)
    if atom_ratio < 0.25:
        return False

    # Must have primary structural characters (parens, brackets, digits)
    has_primary = bool(set(token) & _SMILES_PRIMARY_STRUCTURAL)
    if not has_primary:
        return False

    # Balanced parentheses check (basic SMILES syntax validity)
    paren_depth = 0
    bracket_depth = 0
    for ch in token:
        if ch == "(":
            paren_depth += 1
        elif ch == ")":
            paren_depth -= 1
            if paren_depth < 0:
                return False
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth -= 1
            if bracket_depth < 0:
                return False
    if paren_depth != 0 or bracket_depth != 0:
        return False

    return True


def extract_smiles_from_content(content: str) -> list[str]:
    """Extract individual SMILES strings from tool output.

    Handles:
    - One SMILES per line (plain text output)
    - JSON arrays of SMILES strings: ["CCO", "c1ccccc1", ...]
    - SMILES embedded in labeled lines: "aspirin: CC(=O)Oc1ccccc1C(=O)O"
    - Mixed content where some lines are SMILES and others are prose

    Args:
        content: Raw text output from a tool.

    Returns:
        List of extracted SMILES strings (deduplicated, order preserved).
    """
    found: list[str] = []
    seen: set[str] = set()

    def _add(smi: str) -> None:
        smi = smi.strip()
        if smi and smi not in seen and _looks_like_smiles(smi):
            found.append(smi)
            seen.add(smi)

    # Strategy 1: Try JSON array parsing
    # Handles: ["CC(=O)O", "CCO"] or {"results": ["CC(=O)O", ...]}
    import json as _json

    try:
        parsed = _json.loads(content)
        candidates: list[str] = []
        if isinstance(parsed, list):
            candidates = [x for x in parsed if isinstance(x, str)]
        elif isinstance(parsed, dict):
            # Grab string items from any list-valued field
            for v in parsed.values():
                if isinstance(v, list):
                    candidates.extend(x for x in v if isinstance(x, str))
        for c in candidates:
            _add(c)
        if found:
            return found
    except (ValueError, TypeError):
        pass

    # Strategy 2: Line-by-line scanning
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        # Split on tabs first -- many tools return TSV/CSV-like output
        # where SMILES sits in one column alongside scores, labels, etc.
        # Checking the whole tab-separated line would incorrectly classify
        # "Cn1c(=O)...=O\t-36.545\t417" as a single SMILES token.
        segments = line.split("\t") if "\t" in line else [line]

        for segment in segments:
            segment = segment.strip()
            if not segment:
                continue

            # Whole segment is a SMILES
            if _looks_like_smiles(segment):
                _add(segment)
                continue

            # Labeled segment: "aspirin: CC(=O)Oc1ccccc1C(=O)O" or "1. CCO"
            # Try extracting tokens
            for match in _SMILES_TOKEN_RE.finditer(segment):
                token = match.group(1)
                _add(token)

    return found


HEAVY_TYPE_PATTERNS: list[HeavyTypePattern] = [
    # SAFE fragment notation -- declared BEFORE smiles, deliberately.
    #
    # Matching stops at the first pattern that clears its threshold, not at the
    # highest-scoring one, so order decides this case. GenMol's request field is
    # literally named `smiles` while carrying SAFE, which the smiles pattern
    # would claim with a name+description score of 1.0. Declaring SAFE first
    # means its description-only match (0.5) wins, which is what we want: a
    # field documented as SAFE is SAFE regardless of what it is named.
    #
    # `\bSAFE\b` is compiled case-sensitively (flags=0). Under the module
    # default of re.IGNORECASE it would fire on ordinary prose -- "safe to
    # retry", "a safe upper bound" -- and steal every parameter in the schema.
    # The multi-word forms below carry no such ambiguity and keep IGNORECASE.
    HeavyTypePattern(
        data_type="safe",
        name_patterns=_compile_patterns(
            [
                r"^safe$",
                r"^safe[_\-]?smiles$",
                r"^safe[_\-]?str(?:ing)?$",
                r"^safe[_\-]?notation$",
                r"^safe[_\-]?encoding$",
                r"^safe[_\-]?input$",
                r"^safe[_\-]?fragments?$",
            ]
        ),
        desc_patterns=(
            _compile_patterns([r"\bSAFE\b"], flags=0)
            + _compile_patterns(
                [
                    r"\bSAFE[-_ ]SMILES\b",
                    r"\bSAFE\s+(?:notation|string|encoding|representation|format|input)\b",
                    r"\bsequential\s+attachment[-\s]based\s+fragment\s+embedding\b",
                    r"\bfragment[-\s]based\s+SMILES\b",
                ]
            )
        ),
        output_desc_patterns=(
            _compile_patterns(
                [
                    r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\bSAFE\b",
                    r"\bSAFE\b.*\b(?:returned|output|generated|produced)\b",
                ],
                flags=0,
            )
        ),
        # SAFE strings are valid SMILES, so content alone cannot tell them
        # apart. The one exception is GenMol's masked input, whose `[*{20-30}]`
        # token is not valid SMILES -- an unambiguous SAFE-only signal. Anything
        # without a mask falls through to the SMILES extractor, which is the
        # correct default.
        content_patterns=_compile_patterns([r"\[\*\{\d+\s*-\s*\d+\}\]"]),
    ),
    # SMILES strings
    #
    # NOTE: SMILES content detection uses a custom path (extract_smiles_from_content)
    # rather than the generic content_patterns regex. The content_patterns below
    # serve as a quick "is there any SMILES here?" gate; the extractor handles
    # the actual parsing.
    HeavyTypePattern(
        data_type="smiles",
        name_patterns=_compile_patterns(
            [
                r"^smiles$",
                r"^smiles[_\-]?str(?:ing)?$",
                r"^canonical[_\-]?smiles$",
                r"^mol[_\-]?smiles$",
                r"^input[_\-]?smiles$",
                r"^ligand[_\-]?smiles$",
                r"^molecule[_\-]?smiles$",
                r"^compound[_\-]?smiles$",
                r"^starting[_\-]?smiles$",
                r"^smiles[_\-]?notation$",
            ]
        ),
        desc_patterns=_compile_patterns(
            [
                r"\bSMILES\b",
                r"\bsmiles\s+(?:string|notation|representation|format)\b",
                r"\bmolecular\s+string\b",
                r"\bchemical\s+notation\b",
                r"\bcanonical\s+smiles\b",
                r"\bSMILES\s+encoded\b",
            ]
        ),
        output_desc_patterns=_compile_patterns(
            [
                r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\bSMILES\b",
                r"\bSMILES\b.*\b(?:returned|output|generated|produced)\b",
            ]
        ),
        content_patterns=[],  # handled by custom SMILES detection below
    ),
    # SDF / MOL files
    HeavyTypePattern(
        data_type="sdf",
        name_patterns=_compile_patterns(
            [
                r"^sdf$",
                r"^sdf[_\-]?(?:data|file|content|string|block)$",
                r"^mol[_\-]?(?:file|block|data)$",
                r"^molfile$",
                r"^sdf[_\-]?format$",
                r"^structure[_\-]?data[_\-]?file$",
            ]
        ),
        desc_patterns=_compile_patterns(
            [
                r"\bSDF\b",
                r"\bstructure\s+data\s+file\b",
                r"\bMOL\s+file\b",
                r"\bmolfile\b",
                r"\bV2000\b",
                r"\bV3000\b",
                r"\bMDL\s+(?:mol|sdf)\b",
            ]
        ),
        output_desc_patterns=_compile_patterns(
            [
                r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\b(?:SDF|MOL\s*file|molfile)\b",
                r"\b(?:SDF|MOL\s*file)\b.*\b(?:returned|output|generated|produced)\b",
            ]
        ),
        content_patterns=_compile_patterns(
            [
                r"V[23]000",  # Connection table marker
                r"M\s+END",  # SDF/MOL end marker
                r"\$\$\$\$",  # SDF record separator
            ]
        ),
    ),
    # mmCIF / CIF structure files.
    #
    # Declared before "pdb" because mmCIF descriptions routinely mention the
    # PDB (e.g. "mmCIF file from the PDB"), and detect_inputs takes the first
    # matching pattern per parameter.
    HeavyTypePattern(
        data_type="mmcif",
        name_patterns=_compile_patterns(
            [
                r"^m?mcif$",
                r"^cif$",
                r"^m?mcif[_\-]?(?:data|file|content|string|block|text)$",
                r"^cif[_\-]?(?:data|file|content|string|block|text)$",
                r"^structure[_\-]?m?mcif$",
            ]
        ),
        desc_patterns=_compile_patterns(
            [
                r"\bmmCIF\b",
                r"\bPDBx\b",
                r"\bcrystallographic\s+information\s+(?:file|framework)\b",
                r"\bCIF\s+(?:file|format|data|content)\b",
            ]
        ),
        output_desc_patterns=_compile_patterns(
            [
                r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\b(?:mmCIF|PDBx|CIF)\b",
                r"\b(?:mmCIF|PDBx)\b.*\b(?:returned|output|generated|produced)\b",
            ]
        ),
        content_patterns=_compile_patterns(
            [
                r"^data_\S+",  # mmCIF data block header
                r"^_atom_site\.",  # atom site category
                r"^_entry\.id",
            ],
            flags=re.MULTILINE,
        ),
    ),
    # PDB files
    HeavyTypePattern(
        data_type="pdb",
        name_patterns=_compile_patterns(
            [
                r"^pdb$",
                r"^pdb[_\-]?(?:data|file|content|string|block|text)$",
                r"^protein[_\-]?structure$",
                r"^pdb[_\-]?format$",
                r"^structure[_\-]?pdb$",
            ]
        ),
        desc_patterns=_compile_patterns(
            [
                r"\bPDB\b(?!\s*(?:ID|identifier|code|accession))",  # PDB but not "PDB ID"
                r"\bprotein\s+data\s+bank\b",
                r"\bPDB\s+(?:file|format|data|content|structure)\b",
                r"\b3D\s+(?:protein\s+)?structure\s+(?:file|data)\b",
            ]
        ),
        output_desc_patterns=_compile_patterns(
            [
                r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\bPDB\b",
                r"\bPDB\b.*\b(?:returned|output|generated|produced)\b",
            ]
        ),
        content_patterns=_compile_patterns(
            [
                r"^(?:ATOM|HETATM)\s+\d+",  # PDB atom lines
                r"^HEADER\s+",  # PDB header
                r"^REMARK\s+\d+",  # PDB remarks
                r"^CRYST1\s+",  # PDB crystal info
            ],
            flags=re.MULTILINE,
        ),
    ),
    # DNA sequences.
    #
    # Declared before the generic "sequence" pattern so a parameter named
    # `sequence` whose description says "DNA sequence" resolves to dna rather
    # than to the protein-oriented sequence type.
    HeavyTypePattern(
        data_type="dna",
        name_patterns=_compile_patterns(
            [
                r"^dna$",
                r"^dna[_\-]?(?:sequence|seq|strand|data|string)$",
                r"^nucleotide[_\-]?sequence$",
                r"^nucleic[_\-]?acid(?:[_\-]?sequence)?$",
                r"^(?:template|coding|forward|reverse)[_\-]?strand$",
            ]
        ),
        desc_patterns=_compile_patterns(
            [
                r"\bDNA\b",
                r"\bdeoxyribonucleic\b",
                r"\bnucleotide\s+sequence\b",
                r"\bnucleic\s+acid\s+sequence\b",
                r"\bbase\s+pairs?\b",
            ]
        ),
        output_desc_patterns=_compile_patterns(
            [
                r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\b(?:DNA|nucleotide)\b",
                r"\b(?:DNA|nucleotide)\s+sequence\b.*\b(?:returned|output|generated|produced)\b",
            ]
        ),
        # Requires a long nucleotide-only run: 30+ residues drawn solely from
        # ACGTN/U is effectively never a protein, even though A/C/G/T are all
        # valid amino acid letters.
        content_patterns=_compile_patterns(
            [
                r"^>[^\n]+\n[ACGTUN\s]{30,}$",
                r"^[ACGTN]{30,}$",
            ],
            flags=re.MULTILINE,
        ),
    ),
    # RNA sequences
    HeavyTypePattern(
        data_type="rna",
        name_patterns=_compile_patterns(
            [
                r"^rna$",
                r"^rna[_\-]?(?:sequence|seq|strand|data|string)$",
                r"^m?rna[_\-]?sequence$",
            ]
        ),
        desc_patterns=_compile_patterns(
            [
                r"\bRNA\b",
                r"\bribonucleic\b",
                r"\bmRNA\b",
            ]
        ),
        output_desc_patterns=_compile_patterns(
            [
                r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\bRNA\b",
                r"\bRNA\s+sequence\b.*\b(?:returned|output|generated|produced)\b",
            ]
        ),
        content_patterns=_compile_patterns(
            [
                r"^[ACGUN]{30,}$",
            ],
            flags=re.MULTILINE,
        ),
    ),
    # Multiple sequence alignments.
    #
    # Declared before "sequence" because A3M/Stockholm alignments are built
    # from FASTA-like records that the sequence patterns would otherwise claim.
    HeavyTypePattern(
        data_type="msa",
        name_patterns=_compile_patterns(
            [
                r"^msa$",
                r"^msa[_\-]?(?:data|file|content|string|alignment)$",
                r"^alignment$",
                r"^alignments?[_\-]?(?:data|file|content|string)$",
                r"^a3m$",
                r"^a3m[_\-]?(?:data|file|content|string)$",
                r"^stockholm$",
                r"^multiple[_\-]?sequence[_\-]?alignment$",
            ]
        ),
        desc_patterns=_compile_patterns(
            [
                r"\bMSA\b",
                r"\bmultiple\s+sequence\s+alignment\b",
                r"\bA3M\b",
                r"\bStockholm\b",
                r"\baligned\s+sequences\b",
                r"\balignment\s+(?:file|format|data)\b",
            ]
        ),
        output_desc_patterns=_compile_patterns(
            [
                r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\b(?:MSA|multiple\s+sequence\s+alignment|A3M)\b",
                r"\b(?:MSA|A3M)\b.*\b(?:returned|output|generated|produced)\b",
            ]
        ),
        content_patterns=_compile_patterns(
            [
                r"^#\s*STOCKHOLM\s+1\.0",
                r"^CLUSTAL\s+",
                # A3M/aligned FASTA: 3+ records, or gap characters in the body.
                r"^>[^\n]+\n[A-Z\-\.]*[\-\.][A-Z\-\.]*$",
            ],
            flags=re.MULTILINE,
        ),
    ),
    # Protein / amino acid sequences
    HeavyTypePattern(
        data_type="sequence",
        name_patterns=_compile_patterns(
            [
                r"^sequence$",
                r"^protein[_\-]?sequence$",
                r"^amino[_\-]?acid(?:[_\-]?sequence)?$",
                r"^fasta$",
                r"^fasta[_\-]?(?:sequence|data|string)$",
                r"^aa[_\-]?sequence$",
                r"^residue[_\-]?sequence$",
                r"^peptide[_\-]?sequence$",
            ]
        ),
        desc_patterns=_compile_patterns(
            [
                r"\bamino\s+acid\b",
                r"\bprotein\s+sequence\b",
                r"\bFASTA\b",
                r"\bresidue\s+sequence\b",
                r"\bpeptide\s+sequence\b",
                r"\bpolypeptide\b",
                r"\bone-letter\s+(?:amino\s+acid\s+)?code\b",
            ]
        ),
        output_desc_patterns=_compile_patterns(
            [
                r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\b(?:sequence|FASTA|amino\s+acid)\b",
                r"\b(?:sequence|FASTA)\b.*\b(?:returned|output|generated|produced)\b",
            ]
        ),
        content_patterns=_compile_patterns(
            [
                r"^>[^\n]+\n[A-Z]{20,}",  # FASTA format
                r"^[ACDEFGHIKLMNPQRSTVWY]{20,}$",  # Raw amino acid sequence
            ],
            flags=re.MULTILINE,
        ),
    ),
    # MOL2 files
    HeavyTypePattern(
        data_type="mol2",
        name_patterns=_compile_patterns(
            [
                r"^mol2$",
                r"^mol2[_\-]?(?:data|file|content|string)$",
                r"^tripos[_\-]?(?:mol2|file)$",
            ]
        ),
        desc_patterns=_compile_patterns(
            [
                r"\bMOL2\b",
                r"\bTripos\b",
                r"\bmol2\s+(?:file|format|data)\b",
            ]
        ),
        output_desc_patterns=_compile_patterns(
            [
                r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\bMOL2\b",
                r"\bMOL2\b.*\b(?:returned|output|generated|produced)\b",
            ]
        ),
        content_patterns=_compile_patterns(
            [
                r"@<TRIPOS>MOLECULE",
                r"@<TRIPOS>ATOM",
            ]
        ),
    ),
    # PDBQT files (AutoDock)
    HeavyTypePattern(
        data_type="pdbqt",
        name_patterns=_compile_patterns(
            [
                r"^pdbqt$",
                r"^pdbqt[_\-]?(?:data|file|content|string)$",
                r"^autodock[_\-]?(?:file|input)$",
            ]
        ),
        desc_patterns=_compile_patterns(
            [
                r"\bPDBQT\b",
                r"\bAutoDock\b",
                r"\bpdbqt\s+(?:file|format|data)\b",
            ]
        ),
        output_desc_patterns=_compile_patterns(
            [
                r"\b(?:returns?|outputs?|generates?|produces?)\s+.*\bPDBQT\b",
                r"\bPDBQT\b.*\b(?:returned|output|generated|produced)\b",
            ]
        ),
        content_patterns=_compile_patterns(
            [
                r"^(?:ATOM|HETATM)\s+.*\s+[A-Z]{1,2}\s*$",  # PDBQT atom lines with partial charges
                r"^ROOT$",  # PDBQT torsion markers
                r"^BRANCH\s+\d+\s+\d+",
            ],
            flags=re.MULTILINE,
        ),
    ),
]


def unknown_pattern_types(
    patterns: list[HeavyTypePattern] | None = None,
) -> list[str]:
    """Return pattern data types not declared in ``api.storage.datatypes``.

    Detection and storage must share one vocabulary: a pattern naming a type
    the storage layer does not know produces artifacts with no MIME type and
    no text/binary classification. Asserted empty by the test suite; logged as
    a warning at import so a drifted type degrades rather than breaking boot.
    """
    return [
        pattern.data_type
        for pattern in (patterns if patterns is not None else HEAVY_TYPE_PATTERNS)
        if resolve_data_type(pattern.data_type) is None
    ]


_unknown = unknown_pattern_types()
if _unknown:
    logger.warning(
        "Heavy-type patterns reference data types unknown to api.storage.datatypes: %s",
        _unknown,
    )


class BaseDetector(ABC):
    """Abstract base for heavy data type detectors.

    Subclasses implement the detection logic. This interface ensures that
    regex-based (v1) and LLM-based (future) detectors produce compatible
    output and can be composed via CompositeDetector.
    """

    @abstractmethod
    async def detect_inputs(
        self, input_schema: dict[str, Any], tool_name: str = ""
    ) -> list[DetectionResult]:
        """Detect heavy data types in tool input parameters.

        Args:
            input_schema: JSON Schema for the tool's input parameters.
            tool_name: Optional tool name for contextual hints.

        Returns:
            List of DetectionResult for each detected heavy parameter.
        """
        ...

    @abstractmethod
    async def detect_outputs(
        self, tool_description: str, tool_name: str = ""
    ) -> list[DetectionResult]:
        """Detect heavy data types expected in tool output.

        Uses the tool description (since MCP lacks formal output schemas)
        to predict whether the tool will return heavy data.

        Args:
            tool_description: Tool's description string.
            tool_name: Optional tool name for contextual hints.

        Returns:
            List of DetectionResult for predicted heavy output types.
        """
        ...

    @abstractmethod
    def detect_content(self, content: str) -> list[DetectionResult]:
        """Detect heavy data types in actual tool output content at runtime.

        Applied post-execution to sniff for heavy data in the response.

        Args:
            content: Raw text content from tool execution.

        Returns:
            List of DetectionResult for each detected data type in the content.
        """
        ...


class RegexDetector(BaseDetector):
    """Regex-based detector for heavy data types (v1 implementation).

    Uses a library of well-selected regex patterns to match parameter names,
    descriptions, and actual content against known chem/bio data formats.
    """

    def __init__(
        self,
        patterns: list[HeavyTypePattern] | None = None,
        confidence_threshold: float = 0.5,
    ):
        """Initialize the regex detector.

        Args:
            patterns: Custom patterns to use. Defaults to HEAVY_TYPE_PATTERNS.
            confidence_threshold: Global minimum confidence threshold.
        """
        self._patterns = patterns or HEAVY_TYPE_PATTERNS
        self._confidence_threshold = confidence_threshold

    async def detect_inputs(
        self, input_schema: dict[str, Any], tool_name: str = ""
    ) -> list[DetectionResult]:
        """Detect heavy data types in input schema properties."""
        results: list[DetectionResult] = []
        properties = input_schema.get("properties", {})

        for param_name, param_schema in properties.items():
            # Only consider string-type parameters (heavy data is always text)
            param_type = param_schema.get("type", "")
            if param_type != "string":
                continue

            param_desc = param_schema.get("description", "")

            for pattern in self._patterns:
                confidence = self._score_parameter(param_name, param_desc, pattern)
                threshold = max(
                    pattern.confidence_threshold, self._confidence_threshold
                )

                if confidence >= threshold:
                    results.append(
                        DetectionResult(
                            param_name=param_name,
                            detected_type=pattern.data_type,
                            confidence=confidence,
                            direction="input",
                        )
                    )
                    # Take the first (highest-priority) match for this param
                    break

        return results

    async def detect_outputs(
        self, tool_description: str, tool_name: str = ""
    ) -> list[DetectionResult]:
        """Detect heavy data types expected in tool output from description."""
        results: list[DetectionResult] = []

        for pattern in self._patterns:
            for regex in pattern.output_desc_patterns:
                if regex.search(tool_description):
                    results.append(
                        DetectionResult(
                            param_name="output",
                            detected_type=pattern.data_type,
                            confidence=0.7,  # description-based output detection
                            direction="output",
                        )
                    )
                    break  # one match per data type is enough

        return results

    def detect_content(self, content: str) -> list[DetectionResult]:
        """Detect heavy data types by sniffing actual tool output content."""
        results: list[DetectionResult] = []

        for pattern in self._patterns:
            if pattern.data_type == "smiles":
                # SMILES uses custom multi-value detection
                smiles = extract_smiles_from_content(content)
                if smiles:
                    results.append(
                        DetectionResult(
                            param_name="output",
                            detected_type="smiles",
                            confidence=0.9,
                            direction="output",
                        )
                    )
                continue

            for regex in pattern.content_patterns:
                if regex.search(content):
                    results.append(
                        DetectionResult(
                            param_name="output",
                            detected_type=pattern.data_type,
                            confidence=0.9,  # direct content match is high confidence
                            direction="output",
                        )
                    )
                    break  # one match per data type is enough

        return results

    def _score_parameter(
        self,
        param_name: str,
        param_desc: str,
        pattern: HeavyTypePattern,
    ) -> float:
        """Score how well a parameter matches a heavy type pattern.

        Args:
            param_name: The parameter name to check.
            param_desc: The parameter description to check.
            pattern: The pattern to match against.

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        name_match = any(p.search(param_name) for p in pattern.name_patterns)
        desc_match = (
            any(p.search(param_desc) for p in pattern.desc_patterns)
            if param_desc
            else False
        )

        if name_match and desc_match:
            return 1.0
        elif name_match:
            return pattern.name_score
        elif desc_match:
            return pattern.desc_score
        return 0.0


class HeavyDataDetector:
    """High-level facade for heavy data detection.

    Wraps a BaseDetector implementation and provides a convenient API
    for the adapter layer. Currently uses RegexDetector; can be swapped
    to CompositeDetector or LLMDetector in the future.
    """

    def __init__(self, detector: BaseDetector | None = None):
        """Initialize with a specific detector or default to RegexDetector.

        Args:
            detector: Detector implementation. Defaults to RegexDetector.
        """
        self._detector = detector or RegexDetector()

    async def analyze_tool(
        self,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
    ) -> tuple[list[DetectionResult], list[DetectionResult]]:
        """Analyze an MCP tool schema for heavy data types.

        Args:
            tool_name: Name of the tool.
            tool_description: Description of the tool.
            input_schema: JSON Schema for tool input parameters.

        Returns:
            Tuple of (input_detections, output_detections).
        """
        input_detections = await self._detector.detect_inputs(input_schema, tool_name)
        output_detections = await self._detector.detect_outputs(
            tool_description, tool_name
        )

        if input_detections or output_detections:
            logger.info(
                f"Heavy data detected in MCP tool '{tool_name}': "
                f"{len(input_detections)} input(s), {len(output_detections)} output(s) "
                f"[inputs={[d.detected_type for d in input_detections]}, "
                f"outputs={[d.detected_type for d in output_detections]}]"
            )

        return input_detections, output_detections

    def detect_content(self, content: str) -> list[DetectionResult]:
        """Detect heavy data types in runtime content.

        Delegates to the underlying detector's content sniffing.

        Args:
            content: Raw text content from tool execution.

        Returns:
            List of DetectionResult for detected heavy data in the content.
        """
        return self._detector.detect_content(content)

    @staticmethod
    def extract_smiles(content: str) -> list[str]:
        """Extract individual SMILES strings from tool output content.

        Convenience wrapper around extract_smiles_from_content().

        Args:
            content: Raw text content from tool execution.

        Returns:
            List of extracted SMILES strings.
        """
        return extract_smiles_from_content(content)
