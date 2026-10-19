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

"""Work out what an uploaded file is.

Callers of the artifact endpoint know the bytes they hold but not always the
name Aevom's storage vocabulary gives them: an evaluation harness has
``ligand.sdf`` and ``receptor.cif`` on disk, a browser upload has a filename
and a MIME type nobody should be trusting. Rather than force every caller to
learn ``api.storage.datatypes``, this module maps a filename or the content
itself onto a :class:`DataType`.

The order is deliberate: an explicit ``data_type`` always wins, then the file
extension, then the content. Sniffing is last because it is the only step that
can be wrong about a file whose author knew better -- a SMILES file and a
one-line text file are indistinguishable by content, and guessing ``text``
for a ``.smi`` would cost the caller every SMILES-consuming tool.
"""

from __future__ import annotations

import json
from typing import Optional

from api.storage.annotated_cif import is_annotated_cif
from api.storage.datatypes import DataType, resolve

# Extension -> type. Several extensions land on one type; none of these are
# guesses about content, they are what the extension means by convention in
# structural biology and cheminformatics.
EXTENSION_TYPES: dict[str, DataType] = {
    "smi": DataType.SMILES,
    "smiles": DataType.SMILES,
    "sdf": DataType.SDF,
    "mol": DataType.MOL,
    "mol2": DataType.MOL2,
    "pdb": DataType.PDB,
    "ent": DataType.PDB,
    "pdbqt": DataType.PDBQT,
    # Every `.cif` Aevom deals with is an mmCIF from the PDB or from a folding
    # model, and `mmcif` is the type `convert_structure` and `boltz2` speak.
    # The narrower crystallographic `cif` type stays reachable by naming it.
    "cif": DataType.MMCIF,
    "mmcif": DataType.MMCIF,
    "fasta": DataType.FASTA,
    "fa": DataType.FASTA,
    "faa": DataType.FASTA,
    "seq": DataType.SEQUENCE,
    "a3m": DataType.MSA,
    "json": DataType.JSON,
    "csv": DataType.CSV,
    "npz": DataType.ARRAY,
    "png": DataType.PNG,
    "txt": DataType.TEXT,
    "text": DataType.TEXT,
    "md": DataType.TEXT,
    "log": DataType.TEXT,
    "yaml": DataType.TEXT,
    "yml": DataType.TEXT,
}

# PDB records that appear at the start of a coordinate file. ATOM and HETATM
# alone are not enough -- an mmCIF loop body contains both.
_PDB_RECORDS = ("HEADER", "CRYST1", "MODEL ", "ATOM  ", "HETATM", "SEQRES", "REMARK")

# How much of a payload to look at when sniffing. Format markers live in the
# first few lines of every format here, and reading a whole 100 MB structure to
# identify it would be wasteful.
_SNIFF_BYTES = 8192


def type_from_filename(filename: Optional[str]) -> Optional[DataType]:
    """Resolve a type from a filename's extension, or None."""
    if not filename:
        return None
    _, _, extension = str(filename).rpartition(".")
    if not extension or extension == filename:
        return None
    return EXTENSION_TYPES.get(extension.strip().lower())


def type_from_content(content: bytes) -> Optional[DataType]:
    """Identify a payload from its leading bytes, or None when unsure.

    Returning None is the honest answer for a payload with no format marker,
    and the caller turns that into a request for an explicit ``data_type``.
    Guessing ``text`` here would quietly strip a file of its type instead.
    """
    if content.startswith(b"\x89PNG\r\n"):
        return DataType.PNG
    # .npz is a zip archive; so is nothing else Aevom stores.
    if content.startswith(b"PK\x03\x04"):
        return DataType.ARRAY

    try:
        head = content[:_SNIFF_BYTES].decode("utf-8")
    except UnicodeDecodeError:
        return None

    stripped = head.lstrip()
    if not stripped:
        return None

    if stripped.startswith("data_"):
        # An mmCIF carrying design-condition columns is a different type to its
        # consumers: RFD4-Proteina reads the conditions, every other structure
        # tool ignores them. Content is the only place that distinction lives,
        # since both dialects use the `.cif` extension.
        # The _atom_site header block sits past the sniff window in real files,
        # so this reads beyond `head`; is_annotated_cif bounds its own scan.
        return (
            DataType.ANNOTATED_CIF
            if is_annotated_cif(content)
            else DataType.MMCIF
        )
    if stripped.startswith(">"):
        return DataType.FASTA

    lines = head.splitlines()
    if any(line.startswith(_PDB_RECORDS) for line in lines):
        return DataType.PDB
    # SDF's connection-table header block puts the counts line third, and the
    # record terminator is unique to the format.
    if "$$$$" in head or any(
        line.rstrip().endswith(("V2000", "V3000")) for line in lines[:6]
    ):
        return DataType.SDF

    if stripped[0] in "{[":
        try:
            json.loads(head)
        except ValueError:
            # A truncated head of a large valid JSON document also lands here,
            # so this is not proof it is not JSON -- only that we cannot say so
            # from the head alone.
            pass
        else:
            return DataType.JSON

    return None


def infer_data_type(
    declared: Optional[str],
    filename: Optional[str],
    content: bytes,
) -> tuple[Optional[DataType], str]:
    """Decide an artifact's type and say how the decision was reached.

    Args:
        declared: Type named by the caller, which may be an alias.
        filename: Original filename, used for its extension.
        content: The payload, sniffed only if the first two steps fail.

    Returns:
        ``(data_type, source)`` where source is one of ``declared``,
        ``filename``, ``content`` or ``unknown``. A None type means the caller
        has to name one; the source is ``unknown`` in that case, and
        ``declared`` when the caller named a type that does not exist.
    """
    if declared:
        resolved = resolve(declared)
        return resolved, "declared"

    from_name = type_from_filename(filename)
    if from_name is not None:
        # A `.cif` extension cannot tell the two mmCIF dialects apart, so the
        # content decides this one case. Every other extension is taken at its
        # word, as before.
        if from_name is DataType.MMCIF and is_annotated_cif(content):
            return DataType.ANNOTATED_CIF, "content"
        return from_name, "filename"

    from_content = type_from_content(content)
    if from_content is not None:
        return from_content, "content"

    return None, "unknown"
