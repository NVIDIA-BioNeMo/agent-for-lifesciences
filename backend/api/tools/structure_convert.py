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

"""Convert a stored structure between mmCIF and PDB.

This bridges the one real format gap in the tool set. ``boltz2`` produces mmCIF
and nothing else, while ``vina_dock``, ``proteinmpnn_design_sequence`` and
``aevom.load_coords`` parse PDB. The viewer reads both, so this is not needed for
display -- only for handing a Boltz-2 structure to a PDB-only consumer.

The conversion is explicit rather than automatic because mmCIF to PDB is not
always possible: PDB's fixed-column record has one character for the chain id,
five digits for the atom serial and four for the residue number, and Boltz-2
allows four-character chain ids and twelve chains of up to 4,096 residues. When a
structure does not fit, this fails and says which limit it hit, rather than
writing a file that silently drops or renames part of the complex.
"""

import io
import logging
from typing import Any

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)
from api.storage.envelope import ArtifactProvenance

from .base import BaseTool, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)

# PDB fixed-column limits. Exceeding any of these makes the format unable to
# represent the structure.
MAX_PDB_ATOMS = 99999
MAX_PDB_RESIDUE_NUMBER = 9999
MIN_PDB_RESIDUE_NUMBER = -999

_MMCIF_TYPES = frozenset({"mmcif", "cif", "pdbx", "annotated_cif"})
_PDB_TYPES = frozenset({"pdb"})

# Tools that read each format, named in the success message so the model knows
# what the new reference unlocks.
_CONSUMERS = {
    "pdb": [
        "vina_dock",
        "proteinmpnn_design_sequence",
        "structure_query",
        "visualize_pdb",
    ],
    "mmcif": ["visualize_pdb", "structure_query"],
}


class ConversionError(Exception):
    """A conversion that cannot produce a valid file in the target format."""


class ConvertStructureTool(BaseTool):
    """Convert a stored structure artifact between mmCIF and PDB.

    Data Flow:
        Input: an ``mmcif`` or ``pdb`` reference
        Output: Stores the converted structure, returns a reference of the
            requested type
    """

    @property
    def name(self) -> str:
        return "convert_structure"

    @property
    def summary_template(self) -> str:
        return "Converting structure format..."

    def format_summary(self, arguments: dict) -> str:
        target = arguments.get("target_format", "")
        ref = arguments.get("structure_ref", "")
        if target and ref:
            return f"Converting {ref} to {target.upper()}..."
        if target:
            return f"Converting structure to {target.upper()}..."
        return "Converting structure format..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

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
                "name": "convert_structure",
                "description": (
                    "Convert a stored structure between mmCIF and PDB format, "
                    "storing the result as a new reference.\n\n"
                    "WHEN YOU NEED THIS: boltz2 returns mmCIF, but vina_dock, "
                    "proteinmpnn_design_sequence and aevom.load_coords read PDB. "
                    "Convert the mmCIF reference to PDB before passing it to one "
                    "of those.\n\n"
                    "WHEN YOU DO NOT NEED THIS: visualize_pdb and structure_query "
                    "already accept mmCIF references directly. Do not convert just "
                    "to display or query a structure.\n\n"
                    "mmCIF to PDB can fail: PDB allows at most "
                    f"{MAX_PDB_ATOMS} atoms, single-character chain ids, and "
                    f"residue numbers up to {MAX_PDB_RESIDUE_NUMBER}. Large "
                    "complexes will be refused with an explanation rather than "
                    "truncated; in that case keep working from the mmCIF "
                    "reference."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "structure_ref": {
                            "type": "string",
                            "description": (
                                "Reference to the structure to convert, either "
                                "'mmcif_abc123' or 'pdb_abc123'."
                            ),
                            "x-aevom-reference": {"data_type": "mmcif"},
                        },
                        "target_format": {
                            "type": "string",
                            "enum": ["pdb", "mmcif"],
                            "description": (
                                "Format to convert to. Use 'pdb' to feed a "
                                "Boltz-2 structure into vina_dock or "
                                "proteinmpnn_design_sequence."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short label for the converted structure. "
                                "Defaults to a label derived from the source."
                            ),
                        },
                    },
                    "required": ["structure_ref", "target_format"],
                },
            },
        }

    @staticmethod
    def _sniff_format(content: str) -> str:
        """Identify structure text when the reference type is not conclusive.

        mmCIF opens with a ``data_`` block header; PDB has no such line. Only
        the leading lines need looking at, so this stays cheap on large files.
        """
        for line in content[:4096].splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            return "mmcif" if stripped.startswith("data_") else "pdb"
        return "pdb"

    def _source_format(self, reference: str, content: str) -> str:
        """Decide what the stored artifact actually is.

        The reference prefix is the intended answer, but artifacts stored under a
        generic type (or an mmCIF written under ``pdb`` by an older tool) do
        occur, so the content has the final say when the prefix is not one of the
        structure types.
        """
        data_type = (self._storage.get_data_type(reference) or "").lower()
        if data_type in _MMCIF_TYPES:
            return "mmcif"
        if data_type in _PDB_TYPES:
            # An mmCIF stored under a pdb ref would otherwise fail confusingly
            # inside the PDB parser, which tolerates unknown lines silently.
            return self._sniff_format(content)
        return self._sniff_format(content)

    @staticmethod
    def _parse(content: str, source_format: str):
        """Parse structure text into a Biopython structure."""
        from Bio.PDB import MMCIFParser, PDBParser

        parser = (
            MMCIFParser(QUIET=True)
            if source_format == "mmcif"
            else PDBParser(QUIET=True)
        )
        try:
            structure = parser.get_structure("structure", io.StringIO(content))
        except Exception as exc:  # noqa: BLE001 - any parse failure is the same answer
            raise ConversionError(
                f"Could not parse the stored structure as {source_format.upper()}: "
                f"{exc}"
            )

        if not any(True for _ in structure.get_atoms()):
            raise ConversionError(
                f"The stored structure contains no atoms when read as "
                f"{source_format.upper()}."
            )
        return structure

    @staticmethod
    def _check_pdb_limits(structure) -> None:
        """Refuse a PDB write that the format cannot represent.

        Checked up front so the failure names the specific limit. Biopython
        would otherwise either raise a generic column error or, for a long chain
        id, write a file whose chains are wrong.
        """
        atom_count = sum(1 for _ in structure.get_atoms())
        if atom_count > MAX_PDB_ATOMS:
            raise ConversionError(
                f"This structure has {atom_count} atoms, more than the "
                f"{MAX_PDB_ATOMS} that PDB's 5-column atom serial can hold. "
                "PDB cannot represent it."
            )

        long_chains = sorted(
            {
                chain.id
                for chain in structure.get_chains()
                if len(str(chain.id).strip()) > 1
            }
        )
        if long_chains:
            raise ConversionError(
                "PDB allows only single-character chain ids, but this structure "
                f"uses {', '.join(repr(c) for c in long_chains)}. Re-run the "
                "prediction with single-letter chain ids if a PDB file is "
                "required."
            )

        for residue in structure.get_residues():
            number = residue.id[1]
            if not MIN_PDB_RESIDUE_NUMBER <= number <= MAX_PDB_RESIDUE_NUMBER:
                raise ConversionError(
                    f"Residue number {number} is outside the range PDB can "
                    f"represent ({MIN_PDB_RESIDUE_NUMBER} to "
                    f"{MAX_PDB_RESIDUE_NUMBER})."
                )

    @classmethod
    def _write(cls, structure, target_format: str) -> str:
        """Serialize a Biopython structure to the target format."""
        from Bio.PDB import MMCIFIO, PDBIO

        if target_format == "pdb":
            cls._check_pdb_limits(structure)
            writer = PDBIO()
        else:
            writer = MMCIFIO()

        writer.set_structure(structure)
        buffer = io.StringIO()
        try:
            writer.save(buffer)
        except Exception as exc:  # noqa: BLE001 - any write failure is the same answer
            raise ConversionError(
                f"Failed to write {target_format.upper()}: {exc}"
            )

        output = buffer.getvalue()
        if not output.strip():
            raise ConversionError(
                f"Writing {target_format.upper()} produced an empty file."
            )
        return output

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Convert a stored structure and store the result."""
        structure_ref = (arguments.get("structure_ref") or "").strip()
        target_format = (arguments.get("target_format") or "").strip().lower()
        description = arguments.get("description")

        if not structure_ref:
            return ToolResult.error(
                "The 'structure_ref' parameter is required. Pass a stored "
                "structure reference such as 'mmcif_abc12345' from boltz2."
            )
        if target_format not in ("pdb", "mmcif"):
            return ToolResult.error(
                "The 'target_format' parameter must be 'pdb' or 'mmcif', got "
                f"'{target_format}'."
            )
        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve or store "
                "structure data."
            )

        try:
            content_bytes, source_metadata = (
                await self._storage.retrieve_with_metadata(structure_ref)
            )
        except InvalidReferenceError:
            return ToolResult.error(
                f"Invalid structure reference: '{structure_ref}'. Pass a stored "
                "structure from boltz2 ('mmcif_abc12345'), openfold3 or "
                "fetch_and_clean_pdb ('pdb_abc12345')."
            )
        except ArtifactNotFoundError:
            return ToolResult.error(
                f"Structure not found for reference '{structure_ref}'. The data "
                "may have expired; re-run boltz2, openfold3 or "
                "fetch_and_clean_pdb to get a fresh reference."
            )

        content = content_bytes.decode("utf-8", errors="replace")
        if not content.strip():
            return ToolResult.error(
                f"The structure stored at '{structure_ref}' is empty."
            )

        source_format = self._source_format(structure_ref, content)
        if source_format == target_format:
            return ToolResult.error(
                f"'{structure_ref}' is already {target_format.upper()}, so no "
                "conversion is needed. Pass it to the tool you wanted to use "
                "directly."
            )

        try:
            structure = self._parse(content, source_format)
            converted = self._write(structure, target_format)
        except ConversionError as exc:
            logger.info("convert_structure refused %s: %s", structure_ref, exc)
            return ToolResult.error(
                f"{exc} The original reference '{structure_ref}' is still usable "
                "for visualize_pdb, structure_query and python_execute."
            )
        except Exception as exc:  # noqa: BLE001 - a tool must not break the turn
            logger.exception("Failed to convert %s", structure_ref)
            return ToolResult.error(f"Failed to convert structure: {exc}")

        if not description:
            source_description = source_metadata.get("description") or ""
            base = source_description or f"structure from {structure_ref}"
            description = f"{base} ({target_format.upper()})"

        atom_count = sum(1 for _ in structure.get_atoms())
        chain_ids = [str(chain.id) for chain in structure.get_chains()]

        provenance = ArtifactProvenance(
            source_tool=self.name,
            input_refs=[structure_ref],
            parameters={"target_format": target_format, "source_format": source_format},
        )

        reference = await self._storage.store(
            data_type=target_format,
            content=converted.encode("utf-8"),
            metadata={
                "source": "convert_structure",
                "source_ref": structure_ref,
                "source_format": source_format,
                "atom_count": atom_count,
                "chain_ids": chain_ids,
                # Carried forward so a converted Boltz-2 structure keeps the
                # numbers that made it worth converting.
                **{
                    key: value
                    for key, value in source_metadata.items()
                    if key.startswith("affinity_") or key == "confidence_score"
                },
                # Scientific artifact envelope (ADR 0001): structured lineage,
                # lifted into DataArtifact.provenance_json by the storage
                # repository. Additive -- readers of metadata_json are
                # unaffected.
                "provenance": provenance.to_dict(),
            },
            description=description,
        )

        logger.info(
            "Converted %s (%s) to %s as %s",
            structure_ref,
            source_format,
            target_format,
            reference,
        )

        message = format_storage_message(
            data_type=target_format,
            reference=reference,
            source_description=description,
            accepting_tools=_CONSUMERS[target_format],
        )
        message += (
            f"\n\nConverted from {source_format.upper()} ({structure_ref}): "
            f"{atom_count} atoms across {len(chain_ids)} "
            f"{'chain' if len(chain_ids) == 1 else 'chains'} "
            f"({', '.join(chain_ids)})."
        )

        return ToolResult.data_reference(
            reference=reference,
            data_type=target_format,
            message=message,
            metadata={
                "source_ref": structure_ref,
                "source_format": source_format,
                "atom_count": atom_count,
                "chain_ids": chain_ids,
            },
        )
