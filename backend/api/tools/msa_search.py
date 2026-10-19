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

"""MSA Search tool using NVIDIA's ColabFold MSA Search API.

Searches protein sequence databases (UniRef30, ColabFold EnvDB, PDB70) and
returns multiple sequence alignments in a3m format. The resulting MSA can be
passed to OpenFold3 via msa_ref for higher-quality structure predictions.
"""

import logging
from typing import Any

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from .base import DataIOCategory, NvidiaApiError, NvidiaApiTool, ToolResult

logger = logging.getLogger(__name__)

MSA_SEARCH_API_URL = (
    "https://health.api.nvidia.com/v1/biology/colabfold/msa-search/predict"
)
DEFAULT_E_VALUE = 0.0001
DEFAULT_ITERATIONS = 1
DEFAULT_DATABASES = ["all"]
DEFAULT_OUTPUT_FORMAT = "a3m"
DEFAULT_SEARCH_TYPE = "colabfold"

# A deep search over the full database set can run for minutes. The client
# timeout is derived from this by the base class, which is the point: this tool
# previously asked the gateway to poll for 300s behind a 120s client timeout, so
# any search past two minutes was abandoned while the server was still working.
POLL_SECONDS = 300


class MSASearchTool(NvidiaApiTool):
    """Search sequence databases and produce a multiple sequence alignment.

    Calls the NVIDIA-hosted ColabFold MSA Search API with a protein sequence
    and returns a merged a3m alignment that can be fed into OpenFold3 for
    improved structure prediction.

    Data Flow:
        Input: sequence_ref (from fetch_protein_sequence) or raw sequence
        Output: Stores a3m MSA text, returns reference (e.g., "msa_abc12345")
    """

    @property
    def name(self) -> str:
        return "msa_search"

    @property
    def summary_template(self) -> str:
        return "Searching sequence databases for MSA..."

    def format_summary(self, arguments: dict) -> str:
        sequence_ref = arguments.get("sequence_ref", "")
        if sequence_ref:
            return f"Running MSA search for {sequence_ref}..."
        return "Searching sequence databases for MSA..."

    @property
    def api_endpoint(self) -> str:
        return MSA_SEARCH_API_URL

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
                "name": "msa_search",
                "description": (
                    "Search protein sequence databases to generate a Multiple Sequence "
                    "Alignment (MSA). Searches UniRef30, ColabFold EnvDB, and PDB70 to "
                    "find homologous sequences and produce a merged a3m alignment.\n\n"
                    "READ THE SKILL: call "
                    'load_skill(skill_id="msa-search-nim") for what an MSA is used '
                    "for, how alignment depth affects prediction quality, and how to "
                    "judge whether an alignment is deep enough to help.\n\n"
                    "USE CASES:\n"
                    "- Improve OpenFold3 structure prediction quality by providing "
                    "evolutionary information\n"
                    "- Find homologous sequences for a protein of interest\n\n"
                    "WORKFLOW:\n"
                    "1. fetch_protein_sequence to get the sequence\n"
                    "2. msa_search to generate the MSA\n"
                    "3. openfold3 with both sequence_ref and msa_ref for best results\n\n"
                    "SCOPE: one sequence at a time. Paired MSA search for complexes "
                    "and structural template search are not available in Aevom.\n\n"
                    "Note: MSA search typically takes 30-60 seconds."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sequence_ref": {
                            "type": "string",
                            "description": (
                                "Reference to stored sequence from fetch_protein_sequence."
                            ),
                            "x-aevom-reference": {"data_type": "sequence"},
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short description for the MSA result "
                                "(e.g., 'GLP1R MSA', 'PGK1 alignment')."
                            ),
                        },
                    },
                    "required": ["sequence_ref"],
                },
            },
        }

    async def _resolve_sequence(self, sequence_ref: str) -> tuple[str, dict[str, Any]]:
        """Retrieve an amino acid sequence from storage by reference."""
        if self._storage is None:
            raise ValueError(
                "Storage service not available. Cannot retrieve sequence data."
            )
        try:
            content, metadata = await self._storage.retrieve_with_metadata(
                sequence_ref
            )
            sequence = content.decode("utf-8")
            logger.info(
                f"Retrieved sequence from {sequence_ref} ({len(sequence)} residues)"
            )
            return sequence, metadata
        except InvalidReferenceError:
            raise ValueError(
                f"Invalid sequence reference: '{sequence_ref}'. "
                "Use fetch_protein_sequence first."
            )
        except ArtifactNotFoundError:
            raise ValueError(
                f"Sequence not found for reference '{sequence_ref}'. "
                "The data may have expired or the reference is invalid."
            )

    def _count_sequences(self, a3m_text: str) -> int:
        """Count the number of sequences in an a3m alignment."""
        return sum(1 for line in a3m_text.splitlines() if line.startswith(">"))

    def _extract_alignment(self, response_data: dict) -> str:
        """Extract the best a3m alignment text from the API response.

        Prefers the merged colabfold alignment; falls back to the first
        available database alignment.
        """
        alignments = response_data.get("alignments", {})

        # Prefer the merged colabfold alignment
        colabfold = alignments.get(DEFAULT_SEARCH_TYPE, {})
        a3m_entry = colabfold.get(DEFAULT_OUTPUT_FORMAT, {})
        if a3m_entry.get("alignment"):
            return a3m_entry["alignment"]

        # Fall back to first available database with an a3m alignment
        for db_name, db_data in alignments.items():
            if isinstance(db_data, dict):
                a3m_entry = db_data.get(DEFAULT_OUTPUT_FORMAT, {})
                if isinstance(a3m_entry, dict) and a3m_entry.get("alignment"):
                    logger.info(f"Using alignment from database '{db_name}' (colabfold not available)")
                    return a3m_entry["alignment"]

        raise ValueError(
            "MSA Search API returned no alignments. "
            "The sequence may be too short or have no homologs in the databases."
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute MSA search for a protein sequence."""
        sequence_ref = arguments.get("sequence_ref", "")
        description = arguments.get("description")

        if not sequence_ref:
            return ToolResult.error(
                "sequence_ref is required. Use fetch_protein_sequence first "
                "to obtain a sequence reference."
            )

        try:
            sequence, seq_metadata = await self._resolve_sequence(sequence_ref)

            # Build API request
            request_data = {
                "sequence": sequence,
                "e_value": DEFAULT_E_VALUE,
                "iterations": DEFAULT_ITERATIONS,
                "databases": DEFAULT_DATABASES,
                "output_alignment_formats": [DEFAULT_OUTPUT_FORMAT],
            }

            logger.info(
                f"Calling MSA Search API for {len(sequence)}-residue sequence"
            )

            result = await self._call_nvidia_api(
                request_data,
                poll_seconds=POLL_SECONDS,
                service_name="MSA Search",
            )

            # Extract the merged a3m alignment
            a3m_text = self._extract_alignment(result)
            num_sequences = self._count_sequences(a3m_text)

            logger.info(
                f"MSA search complete: {num_sequences} sequences, "
                f"{len(a3m_text)} characters"
            )

            # Build per-database sequence counts from response
            db_counts: dict[str, int] = {}
            for db_name, db_data in result.get("alignments", {}).items():
                if isinstance(db_data, dict):
                    a3m_entry = db_data.get(DEFAULT_OUTPUT_FORMAT, {})
                    if isinstance(a3m_entry, dict) and a3m_entry.get("alignment"):
                        db_counts[db_name] = self._count_sequences(
                            a3m_entry["alignment"]
                        )

            # Store the MSA
            if self._storage is None:
                return ToolResult.error(
                    "Storage service not available. Cannot store MSA data."
                )

            storage_metadata = {
                "query_sequence_length": len(sequence),
                "num_sequences": num_sequences,
                "databases_searched": DEFAULT_DATABASES,
                "search_type": DEFAULT_SEARCH_TYPE,
                "e_value": DEFAULT_E_VALUE,
                "db_sequence_counts": db_counts,
            }
            if seq_metadata.get("gene_name"):
                storage_metadata["gene_name"] = seq_metadata["gene_name"]

            if not description and seq_metadata.get("gene_name"):
                description = f"{seq_metadata['gene_name']} MSA"

            reference = await self._storage.store(
                data_type="msa",
                content=a3m_text.encode("utf-8"),
                metadata=storage_metadata,
                description=description,
            )

            logger.info(f"Stored MSA as {reference}")

            source_desc = description or seq_metadata.get("gene_name", "protein")
            message = format_storage_message(
                data_type="msa",
                reference=reference,
                source_description=source_desc,
                accepting_tools=["openfold3"],
            )
            message += (
                f"\n\nMSA statistics:\n"
                f"- Sequences found: {num_sequences}\n"
                f"- Query length: {len(sequence)} residues\n"
                f"- Databases searched: {', '.join(DEFAULT_DATABASES)}"
            )

            return ToolResult.data_reference(
                reference=reference,
                data_type="msa",
                message=message,
                metadata=storage_metadata,
            )

        except ValueError as e:
            logger.error(f"MSA search validation error: {e}")
            return ToolResult.error(str(e))
        except NvidiaApiError as e:
            logger.error(f"MSA Search API error: {e}")
            return ToolResult.error(str(e))
        except Exception as e:
            logger.exception("Failed to run MSA search")
            return ToolResult.error(f"Failed to run MSA search: {str(e)}")
