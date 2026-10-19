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

"""Getting a structure into the one format PLIP reads.

PLIP parses PDB. The most interesting consumer of interaction profiling is
Boltz-2, which emits mmCIF and nothing else, and it emits ten diffusion samples
per prediction. Making the caller convert each one first would cost ten tool
calls to answer one question, so the conversion happens here.

This is deliberately different from the product's ``convert_structure`` tool,
where conversion is explicit because the output is a durable artifact the user
may need to reason about. Here it is transient -- nothing downstream ever sees
the intermediate -- so the honest choice is to do it silently and only speak up
when it fails.
"""

import logging

logger = logging.getLogger(__name__)


class ConversionError(Exception):
    """A structure that cannot be presented to PLIP as PDB."""


def sniff(content: str) -> str:
    """Identify structure text as 'mmcif' or 'pdb'.

    mmCIF opens with a ``data_`` block header and PDB has no such line, so only
    the first few lines need reading regardless of file size.
    """
    for line in content[:4096].splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return "mmcif" if stripped.startswith("data_") else "pdb"
    return "pdb"


def to_pdb(content: str) -> str:
    """Return the structure as PDB text, converting from mmCIF if needed.

    Raises:
        ConversionError: when the text is neither format, is empty, or cannot be
            expressed in PDB's fixed-column layout.
    """
    if not content.strip():
        raise ConversionError("The structure is empty.")

    if sniff(content) == "pdb":
        return content

    import gemmi

    try:
        structure = gemmi.read_structure_string(content)
    except Exception as exc:
        raise ConversionError(f"Could not parse the mmCIF: {exc}") from exc

    # gemmi accepts a file whose atom_site loop it could not understand and
    # returns a structure with no models, and shorten_chain_names() then
    # segfaults on it -- which would take the whole service down rather than
    # failing one request. This guard is the only thing standing between a
    # malformed mmCIF and that crash, since a segfault is not catchable.
    if len(structure) == 0:
        raise ConversionError(
            "The mmCIF parsed but contained no models. Its atom_site loop is "
            "most likely malformed or absent."
        )

    # setup_entities fills in the entity and subchain bookkeeping that mmCIF
    # carries implicitly and PDB output needs; without it, het groups can be
    # written without their HETATM flag, which is exactly what PLIP keys on to
    # find a ligand.
    try:
        structure.setup_entities()
        structure.shorten_chain_names()
        pdb_text = structure.make_pdb_string()
    except Exception as exc:
        raise ConversionError(
            f"Could not express this mmCIF as PDB: {exc}. PDB allows "
            f"single-character chain ids, at most 99999 atoms and residue "
            f"numbers up to 9999."
        ) from exc

    if not pdb_text.strip():
        raise ConversionError("Converting the mmCIF to PDB produced no atoms.")
    return pdb_text
