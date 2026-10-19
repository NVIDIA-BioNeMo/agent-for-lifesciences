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

"""ProteinMPNN inverse folding via NVIDIA's hosted IPD NIM.

Given a backbone structure, designs amino acid sequences likely to fold into it.
This is the step that turns an ``rfdiffusion_design_backbone`` result -- pure
coordinates, every residue a glycine -- into candidate proteins.

Three outputs come back and are handled differently by size: the multi-FASTA of
designed sequences is stored as a ``fasta`` reference, the per-position
probability matrix as an ``array`` reference, and the per-sequence scores go
inline because they are what the model needs to rank candidates.
"""

import io
import logging
import re
from typing import Any, Optional

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from .base import DataIOCategory, NvidiaApiError, NvidiaApiTool, ToolResult

logger = logging.getLogger(__name__)

PROTEINMPNN_API_URL = (
    "https://health.api.nvidia.com/v1/biology/ipd/proteinmpnn/predict"
)

MIN_SEQUENCES = 1
MAX_SEQUENCES = 100
DEFAULT_SEQUENCES = 8
DEFAULT_SAMPLING_TEMP = 0.1
MIN_SAMPLING_TEMP = 0.0
MAX_SAMPLING_TEMP = 1.0

# ProteinMPNN returns in about a second for a small backbone; the model is tiny.
POLL_SECONDS = 120

# How many per-sequence scores to spell out in the result. The full set is in the
# stored FASTA, and `num_seq_per_target` can be 100.
MAX_REPORTED_SCORES = 10

_SCORE_IN_HEADER = re.compile(r"\bscore=([0-9.eE+-]+)")
_SAMPLE_IN_HEADER = re.compile(r"\bsample=(\d+)")
_TEMP_IN_HEADER = re.compile(r"\bT=([0-9.]+)")

# The first record of the returned multi-FASTA describes the *input* structure,
# not a design, and the `scores` array covers only the designs. Upstream's own
# example filters headers containing "native" or "wt", which does not match the
# label the service actually emits -- so scores end up paired with the wrong
# sequences. All three labels are matched here.
_INPUT_RECORD_PREFIXES = ("input", "native", "wt")


class ProteinMPNNTool(NvidiaApiTool):
    """Design sequences for a backbone structure with ProteinMPNN.

    Data Flow:
        Input: pdb_ref (a backbone, e.g. from rfdiffusion_design_backbone)
        Output: Stores a FASTA of designed sequences (primary reference) and the
            per-position probability matrix as an array reference.
    """

    @property
    def name(self) -> str:
        return "proteinmpnn_design_sequence"

    @property
    def summary_template(self) -> str:
        return "Designing sequences with ProteinMPNN..."

    def format_summary(self, arguments: dict) -> str:
        count = arguments.get("num_seq_per_target") or DEFAULT_SEQUENCES
        pdb_ref = arguments.get("pdb_ref", "")
        if pdb_ref:
            return f"Designing {count} sequences for {pdb_ref}..."
        return f"Designing {count} sequences with ProteinMPNN..."

    @property
    def api_endpoint(self) -> str:
        return PROTEINMPNN_API_URL

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
                "name": "proteinmpnn_design_sequence",
                "description": (
                    "Design amino acid sequences that should fold into a given "
                    "backbone structure (inverse folding) with ProteinMPNN.\n\n"
                    "READ THE SKILL: call "
                    'load_skill(skill_id="proteinmpnn-nim") for guidance on '
                    "sampling temperature, fixing positions, excluding residues, "
                    "and interpreting the scores.\n\n"
                    "USE CASES:\n"
                    "- Put a sequence on a backbone from "
                    "rfdiffusion_design_backbone\n"
                    "- Redesign an existing protein's sequence for the same fold "
                    "(e.g. for stability or solubility)\n\n"
                    "WORKFLOW: rfdiffusion_design_backbone → this tool → openfold3 "
                    "on a designed sequence to confirm it folds back to the "
                    "intended shape.\n\n"
                    "SCORES: lower is better. Scores are per-residue negative "
                    "log-likelihoods, so the lowest-scoring sequence is the "
                    "model's most confident design.\n\n"
                    "Returns a FASTA reference holding every designed sequence, "
                    "plus an 'array' reference for the per-position probability "
                    "matrix — inspect that with python_execute using "
                    "aevom.load_array(). Takes a few seconds."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_ref": {
                            "type": "string",
                            "description": (
                                "Reference to the stored backbone structure to "
                                "design for, e.g. the output of "
                                "rfdiffusion_design_backbone or fetch_pdb_structure."
                            ),
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "num_seq_per_target": {
                            "type": "integer",
                            "description": (
                                f"How many sequences to design, {MIN_SEQUENCES}-"
                                f"{MAX_SEQUENCES} (default {DEFAULT_SEQUENCES})."
                            ),
                        },
                        "sampling_temp": {
                            "type": "number",
                            "description": (
                                f"Sampling temperature, {MIN_SAMPLING_TEMP}-"
                                f"{MAX_SAMPLING_TEMP} (default "
                                f"{DEFAULT_SAMPLING_TEMP}). Higher gives more "
                                "diverse but less confident sequences; 0.1-0.3 is "
                                "the useful range."
                            ),
                        },
                        "chains": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Chain IDs to redesign, e.g. [\"A\"]. Omit to "
                                "redesign every chain. Use this to redesign a "
                                "binder while leaving its target untouched."
                            ),
                        },
                        "omit_aas": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "One-letter amino acid codes to exclude entirely, "
                                "e.g. [\"C\"] to avoid cysteines and the disulfide "
                                "bonds they form."
                            ),
                        },
                        "use_soluble_model": {
                            "type": "boolean",
                            "description": (
                                "Use the model variant biased toward soluble "
                                "proteins. Default false."
                            ),
                        },
                        "ca_only": {
                            "type": "boolean",
                            "description": (
                                "Use the alpha-carbon-only model. Set true only "
                                "when the structure has CA atoms alone. Default "
                                "false."
                            ),
                        },
                        "random_seed": {
                            "type": "integer",
                            "description": (
                                "Fixed seed for reproducible designs. Omit for "
                                "fresh sequences on each call."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short label for the designed sequence set (e.g. "
                                "'PD-L1 binder designs')."
                            ),
                        },
                    },
                    "required": ["pdb_ref"],
                },
            },
        }

    async def _resolve_pdb(self, pdb_ref: str) -> str:
        """Retrieve stored structure text by reference."""
        if self._storage is None:
            raise ValueError(
                "Storage service not available. Cannot retrieve structure data."
            )
        try:
            content = await self._storage.retrieve(pdb_ref)
        except InvalidReferenceError:
            raise ValueError(
                f"Invalid structure reference: '{pdb_ref}'. Pass a stored structure "
                "from rfdiffusion_design_backbone, fetch_pdb_structure or openfold3."
            )
        except ArtifactNotFoundError:
            raise ValueError(
                f"Structure not found for reference '{pdb_ref}'. The data may have "
                "expired or the reference is invalid."
            )
        return content.decode("utf-8", errors="replace")

    @staticmethod
    def _validate_count(value: Any) -> int:
        if value is None:
            return DEFAULT_SEQUENCES
        try:
            count = int(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"num_seq_per_target must be an integer between {MIN_SEQUENCES} "
                f"and {MAX_SEQUENCES}."
            )
        if not MIN_SEQUENCES <= count <= MAX_SEQUENCES:
            raise ValueError(
                f"num_seq_per_target must be between {MIN_SEQUENCES} and "
                f"{MAX_SEQUENCES}, got {count}."
            )
        return count

    @staticmethod
    def _validate_temp(value: Any) -> float:
        if value is None:
            return DEFAULT_SAMPLING_TEMP
        try:
            temp = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"sampling_temp must be a number between {MIN_SAMPLING_TEMP} and "
                f"{MAX_SAMPLING_TEMP}."
            )
        if not MIN_SAMPLING_TEMP <= temp <= MAX_SAMPLING_TEMP:
            raise ValueError(
                f"sampling_temp must be between {MIN_SAMPLING_TEMP} and "
                f"{MAX_SAMPLING_TEMP}, got {temp}."
            )
        return temp

    @staticmethod
    def _parse_designs(mfasta: str) -> list[tuple[str, str]]:
        """Split a multi-FASTA into ``(header, sequence)`` pairs for designs only.

        Drops the leading record describing the input structure, which is not a
        design and has no entry in the ``scores`` array.
        """
        records: list[tuple[str, list[str]]] = []
        for line in mfasta.splitlines():
            if line.startswith(">"):
                records.append((line[1:].strip(), []))
            elif line.strip() and records:
                records[-1][1].append(line.strip())

        designs = []
        for header, chunks in records:
            label = header.lower().lstrip(">").strip()
            if label.startswith(_INPUT_RECORD_PREFIXES):
                continue
            designs.append((header, "".join(chunks)))
        return designs

    @classmethod
    def _score_designs(
        cls,
        designs: list[tuple[str, str]],
        scores: list[Any],
    ) -> list[Optional[float]]:
        """Line each design up with its score.

        The ``scores`` array is positional against the designs, but each header
        also carries ``score=``. Trusting the array only when the counts agree,
        and falling back to the headers otherwise, keeps a count mismatch from
        silently attributing one design's score to another.
        """
        if len(scores) == len(designs):
            try:
                return [float(s) for s in scores]
            except (TypeError, ValueError):
                logger.warning("ProteinMPNN scores were not numeric; using headers")

        parsed: list[Optional[float]] = []
        for header, _ in designs:
            match = _SCORE_IN_HEADER.search(header)
            try:
                parsed.append(float(match.group(1)) if match else None)
            except ValueError:
                parsed.append(None)
        return parsed

    @staticmethod
    def _design_label(header: str, index: int) -> str:
        """Short name for a design, for the result message.

        The raw FASTA header repeats the score and the temperature that are
        already being reported, so it is reduced to the identifying part.
        """
        sample = _SAMPLE_IN_HEADER.search(header)
        temp = _TEMP_IN_HEADER.search(header)
        label = f"design {sample.group(1) if sample else index + 1}"
        return f"{label} (T={temp.group(1)})" if temp else label

    @staticmethod
    def _probs_npz(probs: Any) -> Optional[bytes]:
        """Pack the probability matrix as an npz, matching ``aevom.load_array``.

        Returns None when the field is absent. Without that check ``np.asarray``
        turns a missing ``probs`` into a NaN scalar and the tool advertises an
        array reference holding nothing.
        """
        if not probs:
            return None
        try:
            import numpy as np

            array = np.asarray(probs, dtype=np.float32)
        except Exception:  # noqa: BLE001 - a ragged or odd matrix is not fatal
            logger.warning("Could not convert ProteinMPNN probs to an array")
            return None
        if array.size == 0:
            return None
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=array)
        return buffer.getvalue()

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Design sequences for a backbone and store the results."""
        pdb_ref = arguments.get("pdb_ref") or ""
        description = arguments.get("description")

        if not pdb_ref:
            return ToolResult.error(
                "pdb_ref is required — ProteinMPNN designs a sequence for an "
                "existing backbone. Use rfdiffusion_design_backbone or "
                "fetch_pdb_structure first."
            )

        try:
            count = self._validate_count(arguments.get("num_seq_per_target"))
            temp = self._validate_temp(arguments.get("sampling_temp"))
            input_pdb = await self._resolve_pdb(pdb_ref)

            request_data: dict[str, Any] = {
                "input_pdb": input_pdb,
                "num_seq_per_target": count,
                # The API takes a list even for a single temperature.
                "sampling_temp": [temp],
                "use_soluble_model": bool(arguments.get("use_soluble_model", False)),
                "ca_only": bool(arguments.get("ca_only", False)),
            }
            if arguments.get("chains"):
                request_data["input_pdb_chains"] = list(arguments["chains"])
            if arguments.get("omit_aas"):
                request_data["omit_AAs"] = list(arguments["omit_aas"])
            if arguments.get("random_seed") is not None:
                request_data["random_seed"] = int(arguments["random_seed"])

            logger.info(
                "Calling ProteinMPNN: %d sequences at T=%.2f for %s",
                count,
                temp,
                pdb_ref,
            )

            result = await self._call_nvidia_api(
                request_data,
                poll_seconds=POLL_SECONDS,
                service_name="ProteinMPNN",
            )

            mfasta = result.get("mfasta") or ""
            designs = self._parse_designs(mfasta)
            if not designs:
                return ToolResult.error(
                    "ProteinMPNN returned no designed sequences. Check that the "
                    "structure has valid ATOM records and, if you set chains, that "
                    "those chain IDs exist in it."
                )

            if self._storage is None:
                return ToolResult.error(
                    "Storage service not available. Cannot store designed sequences."
                )

            scores = self._score_designs(designs, list(result.get("scores") or []))
            best_index = min(
                (i for i, s in enumerate(scores) if s is not None),
                key=lambda i: scores[i],
                default=0,
            )
            best_header, best_sequence = designs[best_index]

            if not description:
                description = f"ProteinMPNN designs for {pdb_ref}"

            storage_metadata: dict[str, Any] = {
                "source": "proteinmpnn",
                "input_pdb_ref": pdb_ref,
                "num_sequences": len(designs),
                "sampling_temp": temp,
                "sequence_length": len(best_sequence),
                "scores": scores,
                "best_score": scores[best_index],
                "best_sequence": best_sequence,
            }
            if arguments.get("chains"):
                storage_metadata["chains"] = list(arguments["chains"])
            if arguments.get("omit_aas"):
                storage_metadata["omit_aas"] = list(arguments["omit_aas"])

            fasta_ref = await self._storage.store(
                data_type="fasta",
                content=mfasta.encode("utf-8"),
                metadata=storage_metadata,
                description=description,
            )
            logger.info(
                "Stored %d ProteinMPNN designs as %s", len(designs), fasta_ref
            )

            outputs = {"mfasta": fasta_ref}
            probs_ref = None
            probs_payload = self._probs_npz(result.get("probs"))
            if probs_payload is not None:
                probs_ref = await self._storage.store(
                    data_type="array",
                    content=probs_payload,
                    metadata={
                        "source": "proteinmpnn",
                        "field": "probs",
                        "shape": "(num_sequences, num_residues, 21)",
                        "input_pdb_ref": pdb_ref,
                    },
                    description=f"ProteinMPNN per-position probabilities for {pdb_ref}",
                )
                outputs["probs"] = probs_ref
                storage_metadata["probs_ref"] = probs_ref

            message = format_storage_message(
                data_type="fasta",
                reference=fasta_ref,
                source_description=description,
                accepting_tools=["openfold3"],
            )
            message += (
                f"\n\nDesigned {len(designs)} sequences of "
                f"{len(best_sequence)} residues at T={temp}.\n\n"
                "Scores (lower is better):\n"
            )
            for index, ((header, _), score) in enumerate(zip(designs, scores)):
                if index >= MAX_REPORTED_SCORES:
                    message += (
                        f"- ... and {len(designs) - MAX_REPORTED_SCORES} more in "
                        f"'{fasta_ref}'\n"
                    )
                    break
                shown = f"{score:.4f}" if score is not None else "n/a"
                message += f"- {self._design_label(header, index)}: {shown}\n"

            message += (
                f"\nBest — {self._design_label(best_header, best_index)}:\n"
                f"{best_sequence}\n\n"
                f"To validate, run openfold3 on this sequence and compare the "
                f"predicted structure to '{pdb_ref}'."
            )
            if probs_ref:
                message += (
                    f"\n\nPer-position amino acid probabilities are stored as "
                    f"'{probs_ref}', shape (num_sequences, num_residues, 21). Load "
                    f"it in python_execute with aevom.load_array('{probs_ref}') to "
                    "find positions the model was uncertain about."
                )

            storage_metadata["outputs"] = outputs
            storage_metadata["all_references"] = list(outputs.values())

            return ToolResult.data_reference(
                reference=fasta_ref,
                data_type="fasta",
                message=message,
                metadata=storage_metadata,
            )

        except ValueError as exc:
            logger.error("ProteinMPNN validation error: %s", exc)
            return ToolResult.error(str(exc))
        except NvidiaApiError as exc:
            logger.error("ProteinMPNN API error: %s", exc)
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool must not break the turn
            logger.exception("Failed to design sequences with ProteinMPNN")
            return ToolResult.error(f"Failed to design sequences: {exc}")
