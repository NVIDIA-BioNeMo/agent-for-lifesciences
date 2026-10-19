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

"""Central vocabulary of Aevom storage data types.

Every artifact reference is ``{data_type}_{hex}``, so the ``data_type`` string
is the single most load-bearing identifier in the storage subsystem. It was
previously duplicated as bare string literals in five places that had already
drifted apart from each other:

- ``HEAVY_TYPE_PATTERNS`` in ``api.mcp.detector`` (detection)
- ``_TYPE_DESCRIPTIONS`` in ``api.mcp.adapter`` (LLM-facing wording)
- ``text_types`` in ``api.storage.service`` and its copy in ``api.main`` (export encoding)
- ``CONTENT_TYPE_MAP`` in ``api.sandbox.routes`` (sandbox MIME)

This module is the one place a data type is declared. Adding a type here gives
it detection wording, export encoding, and a sandbox MIME type at once.

``is_text`` is the important flag: text types round-trip through UTF-8 decode,
binary types (``array`` and ``png``) must be base64-encoded when exported and
cannot be stored via the sandbox's plain ``content`` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class DataType(str, Enum):
    """Aevom storage data types.

    A ``str`` enum so existing code comparing against bare strings (and JSON
    serialization of contracts) keeps working unchanged.
    """

    # --- First-class molecular types ---
    SMILES = "smiles"
    SAFE = "safe"
    SDF = "sdf"
    MOL = "mol"
    MOL2 = "mol2"
    PDB = "pdb"
    PDBQT = "pdbqt"
    MMCIF = "mmcif"
    CIF = "cif"
    ANNOTATED_CIF = "annotated_cif"
    SEQUENCE = "sequence"
    DNA = "dna"
    RNA = "rna"
    MSA = "msa"
    FASTA = "fasta"

    # --- Structural / app types ---
    SELECTION = "selection"
    DESIGN_SPEC = "design_spec"

    # --- Generic types ---
    JSON = "json"
    CSV = "csv"
    ARRAY = "array"
    PNG = "png"
    TEXT = "text"
    SKILL_FILE = "skill_file"


@dataclass(frozen=True)
class DataTypeInfo:
    """Per-type metadata driving detection, storage, and LLM-facing wording.

    Attributes:
        data_type: The canonical type identifier.
        friendly: Human/LLM-readable description used in tool schemas and
            result messages (e.g. "PDB structure data").
        mime: Content type served by the sandbox data endpoint.
        is_text: Whether the content is UTF-8 text. Binary types are
            base64-encoded on export and require the sandbox's
            ``content_base64`` field to store.
        heavy: Whether this is a "heavy" type worth ref-wrapping. Non-heavy
            types (e.g. ``skill_file``) are storage-internal.
        aliases: Alternate spellings normalized onto this type.
    """

    data_type: DataType
    friendly: str
    mime: str = "application/octet-stream"
    is_text: bool = True
    heavy: bool = True
    aliases: tuple[str, ...] = field(default_factory=tuple)


DATA_TYPE_INFO: dict[DataType, DataTypeInfo] = {
    DataType.SMILES: DataTypeInfo(
        DataType.SMILES, "SMILES string", "text/plain",
    ),
    # SAFE is itself valid SMILES, so it earns a separate type only because
    # consumers are not interchangeable: GenMol and SAFE-GPT require the
    # fragment ordering, and handing them a plain SMILES silently degrades the
    # generation rather than failing. The friendly text below is what the skill
    # contract extractor sees, so it has to make the distinction obvious.
    DataType.SAFE: DataTypeInfo(
        DataType.SAFE,
        "SAFE fragment notation (SAFE-SMILES): a dot-separated sequence of "
        "molecule fragments, itself a valid SMILES; the input format for "
        "GenMol and SAFE-GPT",
        "text/plain",
    ),
    DataType.SDF: DataTypeInfo(
        DataType.SDF, "SDF/MOL file data", "chemical/x-mdl-sdfile",
    ),
    DataType.MOL: DataTypeInfo(
        DataType.MOL, "MOL file data", "chemical/x-mdl-molfile",
    ),
    DataType.MOL2: DataTypeInfo(
        DataType.MOL2, "MOL2 file data", "chemical/x-mol2",
    ),
    DataType.PDB: DataTypeInfo(
        DataType.PDB, "PDB structure data", "chemical/x-pdb",
    ),
    DataType.PDBQT: DataTypeInfo(
        DataType.PDBQT, "PDBQT file data", "chemical/x-pdbqt",
    ),
    DataType.MMCIF: DataTypeInfo(
        DataType.MMCIF, "mmCIF structure data", "chemical/x-mmcif",
        aliases=("pdbx",),
    ),
    DataType.CIF: DataTypeInfo(
        DataType.CIF, "CIF structure data", "chemical/x-cif",
    ),
    DataType.ANNOTATED_CIF: DataTypeInfo(
        DataType.ANNOTATED_CIF,
        "annotated mmCIF structure data (mmCIF carrying RFD4/AtomWorks "
        "design-condition columns such as hotspots and fixed coordinates)",
        "chemical/x-cif",
    ),
    DataType.SEQUENCE: DataTypeInfo(
        DataType.SEQUENCE, "protein/amino acid sequence", "application/json",
    ),
    DataType.DNA: DataTypeInfo(
        DataType.DNA, "DNA nucleotide sequence", "text/plain",
    ),
    DataType.RNA: DataTypeInfo(
        DataType.RNA, "RNA nucleotide sequence", "text/plain",
    ),
    DataType.MSA: DataTypeInfo(
        DataType.MSA, "multiple sequence alignment", "text/plain",
    ),
    DataType.FASTA: DataTypeInfo(
        DataType.FASTA, "FASTA sequence data", "text/plain",
    ),
    DataType.SELECTION: DataTypeInfo(
        DataType.SELECTION, "molecular selection", "application/json",
        heavy=False,
    ),
    # Binds selections to the conditions of a design: which residues are the
    # hotspot, on which target. Like a selection it is small, immutable JSON that
    # tools pass by reference; an edit is a new artifact naming its parent.
    DataType.DESIGN_SPEC: DataTypeInfo(
        DataType.DESIGN_SPEC, "design specification", "application/json",
        heavy=False,
    ),
    DataType.JSON: DataTypeInfo(
        DataType.JSON, "JSON data", "application/json", heavy=False,
    ),
    DataType.CSV: DataTypeInfo(
        DataType.CSV, "CSV table", "text/csv", heavy=False,
    ),
    # One of two binary types: numeric arrays serialized as .npz. Cannot be
    # UTF-8 decoded, so export base64-encodes it and the sandbox must use
    # content_base64 to store it.
    DataType.ARRAY: DataTypeInfo(
        DataType.ARRAY,
        "numeric array data",
        "application/octet-stream",
        is_text=False,
        heavy=False,
    ),
    # Rendered 2D depictions from the rdkit service, served by GET
    # /api/images/{reference}. Binary like array, and not heavy: an image is a
    # terminal artifact for the UI to display, never an input another tool
    # consumes, so there is nothing for the detector to ref-wrap.
    DataType.PNG: DataTypeInfo(
        DataType.PNG,
        "PNG image",
        "image/png",
        is_text=False,
        heavy=False,
    ),
    # Overflow from a tool response that exceeded its context budget. Not
    # heavy: nothing detects it, it is only ever produced by the response
    # budget in api.tools.base and read back via aevom.load_text.
    DataType.TEXT: DataTypeInfo(
        DataType.TEXT, "plain text", "text/plain", heavy=False,
    ),
    DataType.SKILL_FILE: DataTypeInfo(
        DataType.SKILL_FILE, "skill document", "text/markdown", heavy=False,
    ),
}


# Reverse lookup including aliases, built once at import.
_BY_NAME: dict[str, DataType] = {}
for _dt, _info in DATA_TYPE_INFO.items():
    _BY_NAME[_dt.value] = _dt
    for _alias in _info.aliases:
        _BY_NAME[_alias] = _dt


def resolve(data_type: str | DataType) -> Optional[DataType]:
    """Resolve a type string (or alias) to a :class:`DataType`, or None."""
    if isinstance(data_type, DataType):
        return data_type
    return _BY_NAME.get(str(data_type or "").strip().lower())


def get_info(data_type: str | DataType) -> Optional[DataTypeInfo]:
    """Return metadata for a type, or None when the type is unknown."""
    resolved = resolve(data_type)
    return DATA_TYPE_INFO.get(resolved) if resolved else None


def is_text_type(data_type: str | DataType) -> bool:
    """Whether content of this type is UTF-8 text.

    Unknown types default to False (treated as binary) so an unrecognized
    type is base64-encoded rather than raising a ``UnicodeDecodeError``.
    """
    info = get_info(data_type)
    return info.is_text if info else False


def friendly_name(data_type: str | DataType) -> str:
    """LLM-facing description of a type, falling back to the raw string."""
    info = get_info(data_type)
    return info.friendly if info else str(data_type)


def content_type_for(data_type: str | DataType) -> str:
    """MIME type served by the sandbox data endpoint for this type."""
    info = get_info(data_type)
    return info.mime if info else "application/octet-stream"


def text_type_values() -> frozenset[str]:
    """All type strings whose content is UTF-8 text."""
    return frozenset(
        dt.value for dt, info in DATA_TYPE_INFO.items() if info.is_text
    )


def heavy_type_values() -> frozenset[str]:
    """All type strings considered heavy enough to warrant ref-wrapping."""
    return frozenset(
        dt.value for dt, info in DATA_TYPE_INFO.items() if info.heavy
    )


TEXT_TYPES: frozenset[str] = text_type_values()
