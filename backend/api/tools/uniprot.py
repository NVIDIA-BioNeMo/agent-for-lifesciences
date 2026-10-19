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

"""UniProt API tool for fetching protein sequences.

This tool queries the UniProt REST API to retrieve protein sequences
by gene name and stores them using the data storage pattern.
"""

import logging
from typing import Any

import httpx

from api.storage import format_storage_message

from .base import BaseTool, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)

# UniProt REST API endpoint
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"


class UniProtFetchSequenceTool(BaseTool):
    """Fetch protein sequence from UniProt by gene name.

    This tool queries the UniProt REST API to retrieve the amino acid
    sequence for a protein identified by its gene symbol. The sequence
    is stored using the data storage pattern and a reference is returned.

    Data Flow:
        Input: Gene name (natural text), optional organism ID
        Output: Stores sequence data, returns reference (e.g., "sequence_abc123")

    Example:
        # User asks: "Get the sequence for PGK1"
        # Tool calls UniProt API, stores sequence, returns reference
        # LLM can pass reference to downstream tools
    """

    @property
    def name(self) -> str:
        return "fetch_protein_sequence"

    @property
    def summary_template(self) -> str:
        return "Fetching sequence for {gene_name} from UniProt..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        """This tool stores output data."""
        return DataIOCategory.OUTPUT_STORES

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "fetch_protein_sequence",
                "description": (
                    "Fetch a protein's amino acid sequence from UniProt by gene name. "
                    "Use this when the user asks for a protein sequence, wants to know "
                    "the amino acids in a protein, or needs sequence data for analysis. "
                    "The sequence is stored and a reference ID is returned that can be "
                    "passed to other tools.\n\n"
                    "Examples of gene names: PGK1, BRCA1, TP53, EGFR, INS (insulin)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "gene_name": {
                            "type": "string",
                            "description": (
                                "The gene symbol/name to search for (e.g., 'PGK1', 'BRCA1', 'TP53'). "
                                "This should be the official gene symbol."
                            ),
                        },
                        "organism_id": {
                            "type": "string",
                            "description": (
                                "NCBI Taxonomy ID for the organism. Default is '9606' (human). "
                                "Other examples: '10090' (mouse), '10116' (rat), '7227' (fruit fly)."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short description of the sequence for reference tracking "
                                "(e.g., 'PGK1 kinase sequence', 'Human insulin precursor'). "
                                "This helps identify the stored data later."
                            ),
                        },
                    },
                    "required": ["gene_name"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Fetch protein sequence from UniProt and store it.

        Args:
            arguments: Dict containing 'gene_name' and optionally 'organism_id'.

        Returns:
            ToolResult with reference to stored sequence data.
        """
        gene_name = arguments.get("gene_name", "").strip().upper()
        organism_id = arguments.get("organism_id", "9606").strip()
        description = arguments.get("description")

        if not gene_name:
            return ToolResult.error("Gene name is required")

        try:
            # Build UniProt query - prioritize reviewed (Swiss-Prot) entries
            query = f"gene_exact:{gene_name} AND taxonomy_id:{organism_id} AND reviewed:true"

            params = {
                "query": query,
                "format": "json",
                "fields": "accession,sequence",
                "size": 1,
            }

            logger.info(
                f"Querying UniProt for gene: {gene_name} (organism: {organism_id})"
            )

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(UNIPROT_SEARCH_URL, params=params)
                response.raise_for_status()
                data = response.json()

            # Check for results
            results = data.get("results", [])
            if not results:
                return ToolResult.error(
                    f"No protein found for gene '{gene_name}' in organism {organism_id}. "
                    "Please verify the gene name is correct and try again."
                )

            # Extract first (best) result
            entry = results[0]
            primary_accession = entry.get("primaryAccession", "")
            sequence_info = entry.get("sequence", {})
            sequence_value = sequence_info.get("value", "")
            sequence_length = sequence_info.get("length", 0)
            mol_weight = sequence_info.get("molWeight", 0)

            if not sequence_value:
                return ToolResult.error(
                    f"UniProt entry {primary_accession} found but contains no sequence data."
                )

            logger.info(
                f"Found sequence for {gene_name}: {primary_accession} "
                f"({sequence_length} residues, {mol_weight} Da)"
            )

            # Store the sequence
            if self._storage is None:
                return ToolResult.error(
                    "Storage service not available. Cannot store sequence data."
                )

            metadata = {
                "gene_name": gene_name,
                "organism_id": organism_id,
                "primary_accession": primary_accession,
                "sequence_length": sequence_length,
                "mol_weight": mol_weight,
            }

            reference = await self._storage.store(
                data_type="sequence",
                content=sequence_value.encode("utf-8"),
                metadata=metadata,
                description=description,
            )

            logger.info(f"Stored sequence for {gene_name} as {reference}")

            # Return a helpful message guiding the LLM
            message = format_storage_message(
                data_type="sequence",
                reference=reference,
                source_description=f"{gene_name} ({primary_accession}, {sequence_length} amino acids)",
                accepting_tools=None,  # No specific downstream tools yet
            )

            # Add extra context for the LLM
            message += (
                f"\n\nProtein details:\n"
                f"- Gene: {gene_name}\n"
                f"- UniProt Accession: {primary_accession}\n"
                f"- Length: {sequence_length} amino acids\n"
                f"- Molecular Weight: {mol_weight:,} Da"
            )

            return ToolResult.data_reference(
                reference=reference,
                data_type="sequence",
                message=message,
                metadata=metadata,
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"UniProt API HTTP error: {e}")
            return ToolResult.error(
                f"UniProt API request failed with status {e.response.status_code}. "
                "Please try again later."
            )
        except httpx.RequestError as e:
            logger.error(f"UniProt API request error: {e}")
            return ToolResult.error(
                f"Failed to connect to UniProt API: {str(e)}. "
                "Please check your internet connection."
            )
        except Exception as e:
            logger.exception("Failed to fetch protein sequence")
            return ToolResult.error(f"Failed to fetch protein sequence: {str(e)}")
