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

"""GenMol molecule generation via NVIDIA's hosted NIM.

Closes the SAFE chain the rest of the codebase was already built for:
``smiles_to_safe`` produces a masked scaffold, this tool generates molecules to
fill the mask, and ``safe_to_smiles`` is no longer needed on the way out because
GenMol returns ordinary SMILES.

Three of the endpoint's quirks are absorbed here rather than left to the model.
The request field is named ``smiles`` but must carry SAFE. ``temperature`` and
``noise`` must be JSON strings. And a SAFE input with no ``[*{min-max}]`` mask is
accepted without complaint, whereupon the model returns the input molecule back
unchanged -- a failure that looks exactly like success, so it is rejected before
the call.

The generated set is stored two ways because it is consumed two ways: the whole
ranked table as one ``csv`` artifact for analysis, and the top few molecules as
individual ``smiles`` references so they chain straight into ``molecule_info``,
``generate_3d_coordinates`` and ``vina_dock``.
"""

import csv
import io
import logging
from typing import Any

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from .base import DataIOCategory, NvidiaApiError, NvidiaApiTool, ToolResult

logger = logging.getLogger(__name__)

GENMOL_API_URL = "https://health.api.nvidia.com/v1/biology/nvidia/genmol/generate"

MIN_MOLECULES = 1
MAX_MOLECULES = 1000
DEFAULT_MOLECULES = 30

MIN_TEMPERATURE = 0.01
MAX_TEMPERATURE = 10.0
DEFAULT_TEMPERATURE = 1.0

MIN_NOISE = 0.0
MAX_NOISE = 2.0
DEFAULT_NOISE = 1.0

MIN_STEP_SIZE = 1
MAX_STEP_SIZE = 10
DEFAULT_STEP_SIZE = 1

SCORING_MODES = ("QED", "LogP")
DEFAULT_SCORING = "QED"

# De novo mask bounds, in heavy atoms. 20-30 is the upstream example and lands in
# lead-like territory.
DEFAULT_MASK_MIN = 20
DEFAULT_MASK_MAX = 30

# How many of the ranked molecules get their own `smiles` reference. Each one is
# a storage write and a line of context, and the model only needs a handful to
# carry into docking; the rest stay in the CSV.
DEFAULT_STORE_TOP_N = 5
MAX_STORE_TOP_N = 25

# Upstream reports under 3 seconds for 1000 molecules on every supported GPU, so
# a two-minute ceiling fails fast when the service is unhealthy.
POLL_SECONDS = 120

# The token that makes a SAFE string generative. Without it GenMol has nothing to
# fill in and echoes the input.
_MASK_TOKEN = "[*{"


class GenMolTool(NvidiaApiTool):
    """Generate drug-like molecules with GenMol.

    Data Flow:
        Input: optional safe_ref (a masked SAFE scaffold); omit it for de novo
        Output: Stores the ranked molecule table as a CSV reference plus the top
            few molecules as individual SMILES references
    """

    @property
    def name(self) -> str:
        return "genmol_generate_molecules"

    @property
    def summary_template(self) -> str:
        return "Generating molecules with GenMol..."

    def format_summary(self, arguments: dict) -> str:
        count = arguments.get("num_molecules") or DEFAULT_MOLECULES
        if arguments.get("safe_ref"):
            return f"Generating {count} molecules from {arguments['safe_ref']}..."
        return f"Generating {count} molecules de novo with GenMol..."

    @property
    def api_endpoint(self) -> str:
        return GENMOL_API_URL

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
                "name": "genmol_generate_molecules",
                "description": (
                    "Generate novel drug-like molecules with GenMol. Use for de "
                    "novo generation, scaffold decoration, motif extension and "
                    "lead optimization.\n\n"
                    "READ THE SKILL: call "
                    'load_skill(skill_id="genmol-nim") for guidance on choosing '
                    "mask sizes, tuning diversity, and what the scores do and do "
                    "not mean.\n\n"
                    "TWO MODES:\n"
                    "- De novo: omit safe_ref. Set mask_min/mask_max for the size "
                    f"of molecule to invent (default {DEFAULT_MASK_MIN}-"
                    f"{DEFAULT_MASK_MAX} heavy atoms). No other input needed.\n"
                    "- Scaffold decoration / lead optimization: pass safe_ref from "
                    "smiles_to_safe called WITH mask_min and mask_max. The mask "
                    "marks where GenMol builds; a SAFE string without one is "
                    "rejected, because GenMol answers it by returning your input "
                    "molecule unchanged.\n\n"
                    "SCORES rank molecules for triage only. QED is drug-likeness "
                    "(0-1, higher is better); LogP is lipophilicity, where higher "
                    "is not better. Neither is a binding affinity — dock the "
                    "molecules if that is the question.\n\n"
                    "Returns a CSV reference holding every generated molecule with "
                    "its score, plus a SMILES reference for each of the top few so "
                    "they can go straight into molecule_info, "
                    "generate_3d_coordinates or vina_dock. Takes a few seconds."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "safe_ref": {
                            "type": "string",
                            "description": (
                                "Reference to a stored SAFE string carrying a "
                                "'[*{min-max}]' generation mask, from "
                                "smiles_to_safe. OMIT THIS for de novo generation "
                                "— the tool builds the bare mask itself."
                            ),
                            "x-aevom-reference": {"data_type": "safe"},
                        },
                        "mask_min": {
                            "type": "integer",
                            "description": (
                                "Smallest fragment GenMol may invent, in heavy "
                                f"atoms (default {DEFAULT_MASK_MIN}). De novo only "
                                "— when safe_ref is given, its own mask is used."
                            ),
                        },
                        "mask_max": {
                            "type": "integer",
                            "description": (
                                "Largest fragment GenMol may invent, in heavy "
                                f"atoms (default {DEFAULT_MASK_MAX}). Wider ranges "
                                "give more diversity; tighter ranges keep the size "
                                "predictable. De novo only."
                            ),
                        },
                        "num_molecules": {
                            "type": "integer",
                            "description": (
                                f"How many molecules to generate, {MIN_MOLECULES}-"
                                f"{MAX_MOLECULES} (default {DEFAULT_MOLECULES}). "
                                "Invalid and duplicate molecules are filtered "
                                "afterwards, so ask for more than you need."
                            ),
                        },
                        "temperature": {
                            "type": "number",
                            "description": (
                                f"Sampling temperature, {MIN_TEMPERATURE}-"
                                f"{MAX_TEMPERATURE} (default "
                                f"{DEFAULT_TEMPERATURE}). Higher is more diverse. "
                                "Raising it cannot fix a scaffold with no "
                                "attachment point."
                            ),
                        },
                        "noise": {
                            "type": "number",
                            "description": (
                                f"Sampling noise, {MIN_NOISE}-{MAX_NOISE} (default "
                                f"{DEFAULT_NOISE}). Higher is more stochastic."
                            ),
                        },
                        "step_size": {
                            "type": "integer",
                            "description": (
                                f"Diffusion step size, {MIN_STEP_SIZE}-"
                                f"{MAX_STEP_SIZE} (default {DEFAULT_STEP_SIZE}). "
                                "Larger is faster and lower quality."
                            ),
                        },
                        "scoring": {
                            "type": "string",
                            "enum": list(SCORING_MODES),
                            "description": (
                                f"How to score the molecules (default "
                                f"{DEFAULT_SCORING}). 'QED' for drug-likeness, "
                                "'LogP' for lipophilicity."
                            ),
                        },
                        "unique": {
                            "type": "boolean",
                            "description": (
                                "Deduplicate the output. Default false."
                            ),
                        },
                        "store_top_n": {
                            "type": "integer",
                            "description": (
                                "How many of the top-ranked molecules to store as "
                                "individual SMILES references, "
                                f"0-{MAX_STORE_TOP_N} (default "
                                f"{DEFAULT_STORE_TOP_N}). Every molecule is in the "
                                "CSV regardless; this is just which ones are ready "
                                "to chain into another tool."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short label for the generated set (e.g. "
                                "'pyrrolidinone analogs')."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    async def _resolve_safe(self, safe_ref: str) -> str:
        """Retrieve a stored SAFE string by reference."""
        if self._storage is None:
            raise ValueError(
                "Storage service not available. Cannot retrieve SAFE data."
            )
        try:
            content = await self._storage.retrieve(safe_ref)
        except InvalidReferenceError:
            raise ValueError(
                f"Invalid SAFE reference: '{safe_ref}'. Use smiles_to_safe to "
                "produce one, or omit safe_ref for de novo generation."
            )
        except ArtifactNotFoundError:
            raise ValueError(
                f"SAFE string not found for reference '{safe_ref}'. The data may "
                "have expired or the reference is invalid."
            )
        safe = content.decode("utf-8", errors="replace").strip()
        if not safe:
            raise ValueError(f"Retrieved SAFE string from '{safe_ref}' is empty.")
        return safe

    @staticmethod
    def _validate_int(value: Any, name: str, low: int, high: int, default: int) -> int:
        if value is None:
            return default
        try:
            number = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be an integer between {low} and {high}.")
        if not low <= number <= high:
            raise ValueError(
                f"{name} must be between {low} and {high}, got {number}."
            )
        return number

    @staticmethod
    def _validate_float(
        value: Any, name: str, low: float, high: float, default: float
    ) -> float:
        if value is None:
            return default
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a number between {low} and {high}.")
        if not low <= number <= high:
            raise ValueError(
                f"{name} must be between {low} and {high}, got {number}."
            )
        return number

    @classmethod
    def _build_mask(cls, arguments: dict[str, Any]) -> str:
        """The bare generation mask used when no scaffold was supplied."""
        mask_min = cls._validate_int(
            arguments.get("mask_min"), "mask_min", 1, 200, DEFAULT_MASK_MIN
        )
        mask_max = cls._validate_int(
            arguments.get("mask_max"), "mask_max", 1, 200, DEFAULT_MASK_MAX
        )
        if mask_min > mask_max:
            raise ValueError(
                f"mask_min ({mask_min}) cannot exceed mask_max ({mask_max})."
            )
        return f"[*{{{mask_min}-{mask_max}}}]"

    @staticmethod
    def _rank(molecules: list[Any]) -> list[tuple[str, float]]:
        """Well-formed ``(smiles, score)`` pairs, best score first.

        Entries missing either field are dropped rather than carried through as
        blanks: a row with no SMILES is not a molecule the model can do anything
        with, and one with no score cannot be ranked.
        """
        ranked: list[tuple[str, float]] = []
        for entry in molecules:
            if not isinstance(entry, dict):
                continue
            smiles = str(entry.get("smiles") or "").strip()
            if not smiles:
                continue
            try:
                score = float(entry.get("score"))
            except (TypeError, ValueError):
                continue
            ranked.append((smiles, score))
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked

    @staticmethod
    def _to_csv(ranked: list[tuple[str, float]]) -> bytes:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["smiles", "score"])
        for smiles, score in ranked:
            writer.writerow([smiles, f"{score:.4f}"])
        return buffer.getvalue().encode("utf-8")

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Generate molecules and store the ranked results."""
        safe_ref = arguments.get("safe_ref") or ""
        description = arguments.get("description")

        try:
            count = self._validate_int(
                arguments.get("num_molecules"),
                "num_molecules",
                MIN_MOLECULES,
                MAX_MOLECULES,
                DEFAULT_MOLECULES,
            )
            temperature = self._validate_float(
                arguments.get("temperature"),
                "temperature",
                MIN_TEMPERATURE,
                MAX_TEMPERATURE,
                DEFAULT_TEMPERATURE,
            )
            noise = self._validate_float(
                arguments.get("noise"), "noise", MIN_NOISE, MAX_NOISE, DEFAULT_NOISE
            )
            step_size = self._validate_int(
                arguments.get("step_size"),
                "step_size",
                MIN_STEP_SIZE,
                MAX_STEP_SIZE,
                DEFAULT_STEP_SIZE,
            )
            top_n = self._validate_int(
                arguments.get("store_top_n"),
                "store_top_n",
                0,
                MAX_STORE_TOP_N,
                DEFAULT_STORE_TOP_N,
            )
            scoring = arguments.get("scoring") or DEFAULT_SCORING
            if scoring not in SCORING_MODES:
                raise ValueError(
                    f"Unknown scoring '{scoring}'. Choose one of: "
                    f"{', '.join(SCORING_MODES)}."
                )

            de_novo = not safe_ref
            if de_novo:
                safe_input = self._build_mask(arguments)
            else:
                safe_input = await self._resolve_safe(safe_ref)
                if _MASK_TOKEN not in safe_input:
                    raise ValueError(
                        f"The SAFE string in '{safe_ref}' has no '[*{{min-max}}]' "
                        "generation mask, so GenMol has nothing to build and will "
                        "return your input molecule unchanged. Call smiles_to_safe "
                        "again with mask_min and mask_max to replace the "
                        "scaffold's terminal fragment with a mask, or omit "
                        "safe_ref to generate de novo."
                    )

            request_data: dict[str, Any] = {
                # Named `smiles`, but the endpoint requires SAFE in it.
                "smiles": safe_input,
                "num_molecules": count,
                # The endpoint rejects these as numbers.
                "temperature": str(temperature),
                "noise": str(noise),
                "step_size": step_size,
                "scoring": scoring,
                "unique": bool(arguments.get("unique", False)),
            }

            logger.info(
                "Calling GenMol: n=%d scoring=%s de_novo=%s T=%s noise=%s",
                count,
                scoring,
                de_novo,
                temperature,
                noise,
            )

            result = await self._call_nvidia_api(
                request_data,
                poll_seconds=POLL_SECONDS,
                service_name="GenMol",
            )

            if result.get("status") == "failed":
                return ToolResult.error(
                    "GenMol could not generate from this input: "
                    f"{result.get('error', 'no reason given')}. Check that the "
                    "SAFE string is valid and carries a generation mask."
                )

            ranked = self._rank(result.get("molecules") or [])
            if not ranked:
                return ToolResult.error(
                    "GenMol returned no usable molecules. Every generated "
                    "structure was filtered as invalid — try a wider mask range "
                    "or a different scaffold."
                )

            if self._storage is None:
                return ToolResult.error(
                    "Storage service not available. Cannot store generated molecules."
                )

            if not description:
                description = (
                    "GenMol de novo molecules"
                    if de_novo
                    else f"GenMol molecules from {safe_ref}"
                )

            storage_metadata: dict[str, Any] = {
                "source": "genmol",
                "mode": "de_novo" if de_novo else "scaffold_decoration",
                "safe_input": safe_input,
                "num_requested": count,
                "num_returned": len(ranked),
                "scoring": scoring,
                "temperature": temperature,
                "noise": noise,
                "step_size": step_size,
                "unique": bool(arguments.get("unique", False)),
                "best_score": ranked[0][1],
            }
            if not de_novo:
                storage_metadata["safe_ref"] = safe_ref

            csv_ref = await self._storage.store(
                data_type="csv",
                content=self._to_csv(ranked),
                metadata=storage_metadata,
                description=description,
            )
            logger.info(
                "Stored %d GenMol molecules as %s", len(ranked), csv_ref
            )

            outputs = {"molecules_csv": csv_ref}
            top_refs: list[tuple[str, float, str]] = []
            for rank, (smiles, score) in enumerate(ranked[:top_n], start=1):
                smiles_ref = await self._storage.store(
                    data_type="smiles",
                    content=smiles.encode("utf-8"),
                    metadata={
                        "source": "genmol",
                        "rank": rank,
                        "score": score,
                        "scoring": scoring,
                        "molecules_csv_ref": csv_ref,
                    },
                    description=f"GenMol rank {rank} ({scoring} {score:.4f})",
                )
                outputs[f"rank_{rank}"] = smiles_ref
                top_refs.append((smiles, score, smiles_ref))

            message = format_storage_message(
                data_type="csv",
                reference=csv_ref,
                source_description=description,
                accepting_tools=["python_execute"],
            )
            message += (
                f"\n\nGenerated {len(ranked)} molecules ranked by {scoring}"
                f" (requested {count})."
            )
            if len(ranked) < count:
                message += (
                    " Fewer came back than requested because invalid and duplicate "
                    "structures are filtered after generation — that is normal, "
                    "not an error."
                )
            message += (
                f"\nThe full table is in '{csv_ref}'; read it in python_execute "
                f"with aevom.load_text('{csv_ref}').\n"
            )

            if top_refs:
                message += f"\nTop {len(top_refs)} by {scoring}:\n"
                for rank, (smiles, score, smiles_ref) in enumerate(top_refs, start=1):
                    message += f"- {rank}. {score:.4f} — {smiles} → '{smiles_ref}'\n"
                message += (
                    "\nThose references are stored SMILES: pass one to "
                    "molecule_info for properties, or to generate_3d_coordinates "
                    "and then vina_dock to score it against a receptor."
                )
            else:
                message += (
                    "\nNo individual references were stored (store_top_n=0). Pick "
                    "molecules out of the CSV and store them with "
                    "aevom.store_smiles() in python_execute to use them elsewhere."
                )

            if scoring == "LogP":
                message += (
                    "\n\nLogP is lipophilicity, not quality — a high value means "
                    "greasy, which is often a liability. Rank on it only if "
                    "lipophilicity is what was asked for."
                )
            else:
                message += (
                    "\n\nQED is a heuristic drug-likeness score, not a binding "
                    "affinity. Dock the molecules if potency is the question."
                )

            if not de_novo:
                message += (
                    "\n\nBefore calling these analogs of the scaffold, confirm they "
                    "still contain it: load the CSV in python_execute and check "
                    "mol.HasSubstructMatch(scaffold). A maskless or unfragmentable "
                    "input makes GenMol drift to unrelated molecules."
                )
                if arguments.get("mask_min") or arguments.get("mask_max"):
                    # Sizing the fragment is smiles_to_safe's job once a scaffold
                    # exists. Saying so beats letting the model believe bounds it
                    # passed here shaped the result.
                    message += (
                        f"\n\nNote: mask_min/mask_max were ignored — the mask in "
                        f"'{safe_ref}' is what GenMol filled. To change its size, "
                        "call smiles_to_safe again with different bounds."
                    )

            storage_metadata["outputs"] = outputs
            storage_metadata["all_references"] = list(outputs.values())

            return ToolResult.data_reference(
                reference=csv_ref,
                data_type="csv",
                message=message,
                metadata=storage_metadata,
            )

        except ValueError as exc:
            logger.error("GenMol validation error: %s", exc)
            return ToolResult.error(str(exc))
        except NvidiaApiError as exc:
            logger.error("GenMol API error: %s", exc)
            return ToolResult.error(str(exc))
        except Exception as exc:  # noqa: BLE001 - a tool must not break the turn
            logger.exception("Failed to generate molecules with GenMol")
            return ToolResult.error(f"Failed to generate molecules: {exc}")


async def generate_from_safe(
    safe_string: str,
    num_molecules: int = DEFAULT_MOLECULES,
    temperature: float = DEFAULT_TEMPERATURE,
    noise: float = DEFAULT_NOISE,
    scoring: str = DEFAULT_SCORING,
) -> tuple[list[tuple[str, float]], str | None]:
    """Generate molecules from a masked SAFE string, for use by other tools.

    The tool above stores and formats; this is the generation on its own, so a
    tool that already has a template and its own way of judging what comes back
    -- ``fragment_link`` verifies poses -- does not have to go through storage
    and prose to get at it.

    Returns ``(ranked, error)`` where ranked is ``(smiles, score)`` best first.
    """
    if _MASK_TOKEN not in safe_string:
        return [], (
            f"The SAFE string has no '{_MASK_TOKEN}min-max}}]' mask, so GenMol has "
            "nothing to build and would echo it back unchanged."
        )

    tool = GenMolTool()
    try:
        result = await tool._call_nvidia_api(
            {
                "smiles": safe_string,
                "num_molecules": int(num_molecules),
                "temperature": str(temperature),
                "noise": str(noise),
                "step_size": DEFAULT_STEP_SIZE,
                "scoring": scoring,
                "unique": True,
            },
            poll_seconds=POLL_SECONDS,
            service_name="GenMol",
        )
    except NvidiaApiError as exc:
        return [], str(exc)
    except Exception as exc:  # noqa: BLE001 - the caller reports this as text
        logger.exception("GenMol generation failed")
        return [], f"GenMol call failed: {exc}"

    if result.get("status") == "failed":
        return [], f"GenMol failed: {result.get('error', 'no reason given')}"
    return GenMolTool._rank(result.get("molecules") or []), None
