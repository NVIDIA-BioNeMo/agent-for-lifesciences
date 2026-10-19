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

"""ADMET property prediction via the admet container service.

The question this tool answers is the one a medicinal chemist asks immediately
after "does it bind?": will it also kill the patient, reach the target tissue,
dissolve, and survive first pass. Aevom could not answer any of that before this
tool existed -- affinity was the only axis it could measure, and optimizing a
single axis is how a programme produces a potent compound nobody can dose.

Design notes worth knowing before editing:

Predictions come back as a table, not prose. A lead-optimization loop scores tens
of analogues at a time, and the useful shape of that answer is a grid the model
can scan down a column of. The full grid is also stored as CSV, because ranking
thirty candidates on a weighted sum of eight properties is python_execute's job,
not the model's.

The property catalog lives in the service, not here. This module knows the eight
core-panel names only so its schema documentation can list them; every actual
name resolution (aliases, unknown names) happens service-side, and the service
reports what it did not recognize. That keeps one copy of the domain knowledge.
"""

import csv
import io
import logging
import os
from typing import Any

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from .base import ContainerTool, DataIOCategory, ToolResult

logger = logging.getLogger(__name__)

# Documentation only -- the service owns resolution. Kept here so the schema can
# tell the model what it can ask for without a round trip.
CORE_PANEL_DOC = (
    "hERG (cardiac liability), BBB (blood-brain barrier), solubility, "
    "CYP3A4 (drug-drug interaction), clearance (hepatocyte), Caco-2 "
    "(permeability), PPBR (plasma protein binding), AMES (mutagenicity)"
)

# Beyond this many molecules the rendered table stops being something the model
# reads and starts being something it should be ranking in python_execute. The
# CSV artifact holds every row regardless.
MAX_RENDERED_ROWS = 20

# Column width for the SMILES column of the rendered table. Enough to tell
# analogues apart; the CSV has them in full.
SMILES_COLUMN_WIDTH = 34

# Matches the service's own per-request ceiling, checked here so an over-large
# batch fails with actionable advice instead of an HTTP error.
MAX_MOLECULES = 256

# Below this, the response says the batch was small. Twenty is generous: a
# screen of twenty analogues contains a satisfying molecule about three quarters
# of the time at the hit rates analogue screening actually runs at, and the
# whole point of a batched oracle is that a bigger set is free.
SMALL_BATCH = 20


class AdmetPredictTool(ContainerTool):
    """Predict ADMET properties for one or more molecules.

    Data Flow:
        Input: inline SMILES strings and/or SMILES references
        Output: a rendered table, plus a stored CSV of the full results
    """

    @property
    def name(self) -> str:
        return "admet_predict"

    @property
    def summary_template(self) -> str:
        return "Predicting ADMET properties..."

    @property
    def service_url(self) -> str:
        """URL of the ADMET service container."""
        return os.environ.get("ADMET_SERVICE_URL", "http://admet:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        """Takes references, and stores the results table."""
        return DataIOCategory.BOTH

    SMALL_BATCH = SMALL_BATCH

    @property
    def batch_argument(self) -> str:
        # One call scores the whole list for one budget unit.
        return "smiles"

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "admet_predict",
                "description": (
                    "Predict ADMET properties (absorption, distribution, "
                    "metabolism, excretion, toxicity) for one or more molecules "
                    "using the ADMET-AI ensemble.\n\n"
                    f"The default panel is eight properties: {CORE_PANEL_DOC}. "
                    "Pass `properties` to request others (CYP2D6, DILI, LD50, "
                    "half-life, VDss, QED, ...), or ['all'] for everything.\n\n"
                    "Score every candidate you are considering in ONE call -- "
                    "the ensemble batches efficiently, and comparing analogues "
                    "is the point. The full results are also stored as a CSV, "
                    "so use python_execute with aevom.load() to rank or "
                    "filter a large set rather than eyeballing the table.\n\n"
                    "These are model predictions, not measurements. Treat them "
                    "as a filter that removes clear liabilities, and say so when "
                    "reporting them. Direction is not always obvious: high "
                    "blood-brain barrier penetration is required for a CNS "
                    "target and a liability for a peripheral one, so state what "
                    "you are optimizing for before calling a value good."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "smiles": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "SMILES strings to score, passed inline. Use "
                                "this for candidates you have just enumerated."
                            ),
                        },
                        "smiles_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "References to stored SMILES artifacts (e.g. "
                                "['smiles_abc123']). An artifact containing "
                                "several lines is read as several molecules, so "
                                "a stored candidate list works here."
                            ),
                            "x-aevom-reference": {"data_type": "smiles"},
                        },
                        "properties": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Property names or aliases to return. Omit for "
                                "the eight-property core panel; pass ['all'] "
                                "for every property the ensemble predicts."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Optional label for the stored results CSV."
                            ),
                        },
                    },
                },
            },
        }

    async def _collect_smiles(
        self, inline: list[str], refs: list[str]
    ) -> tuple[list[str], dict[str, str], ToolResult | None]:
        """Gather the molecules to score from both input forms.

        Returns:
            ``(smiles, origin, error_result)`` where ``origin`` maps each SMILES
            to the reference it came from, for the results table. When
            ``error_result`` is not None the caller returns it unchanged.
        """
        smiles: list[str] = []
        origin: dict[str, str] = {}

        def add(entry: str, source: str = "") -> None:
            entry = entry.strip()
            # Duplicates score identically and waste a column of the table.
            if not entry or entry in origin or entry in smiles:
                return
            smiles.append(entry)
            if source:
                origin[entry] = source

        for entry in inline:
            if isinstance(entry, str):
                add(entry)

        for ref in refs:
            if not isinstance(ref, str) or not ref.strip():
                continue
            ref = ref.strip()
            try:
                content = await self._storage.retrieve(ref)
            except InvalidReferenceError:
                return (
                    [],
                    {},
                    ToolResult.error(
                        f"Invalid SMILES reference: '{ref}'. A reference looks "
                        f"like 'smiles_abc12345' and comes from a previous tool "
                        f"call or from aevom.store_smiles() in python_execute."
                    ),
                )
            except ArtifactNotFoundError:
                return (
                    [],
                    {},
                    ToolResult.error(
                        f"No SMILES found for reference '{ref}'. Store the "
                        f"molecule first, then pass the reference it returns."
                    ),
                )

            text = content.decode("utf-8", errors="replace")
            for line in text.splitlines():
                # Tolerate the .smi convention of "SMILES<whitespace>name".
                add(line.split()[0] if line.split() else "", ref)

        return smiles, origin, None

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Predict ADMET properties for the requested molecules.

        Args:
            arguments: Dict with 'smiles' and/or 'smiles_refs', plus optional
                'properties' and 'description'.

        Returns:
            ToolResult with the rendered table and a CSV reference, or an error.
        """
        inline = arguments.get("smiles") or []
        refs = arguments.get("smiles_refs") or []
        requested = arguments.get("properties") or None
        description = arguments.get("description")

        if isinstance(inline, str):
            inline = [inline]
        if isinstance(refs, str):
            refs = [refs]

        if not inline and not refs:
            return ToolResult.error(
                "Nothing to score. Pass SMILES strings in `smiles`, or "
                "references to stored SMILES in `smiles_refs`."
            )

        if refs and self._storage is None:
            return ToolResult.error(
                "Storage service not available, so smiles_refs cannot be "
                "resolved. Pass the SMILES strings inline instead."
            )

        smiles, origin, failure = await self._collect_smiles(inline, refs)
        if failure is not None:
            return failure

        if not smiles:
            return ToolResult.error(
                "No usable SMILES were found in the inputs. Check that the "
                "references point at non-empty SMILES artifacts."
            )

        if len(smiles) > MAX_MOLECULES:
            return ToolResult.error(
                f"{len(smiles)} molecules exceeds the {MAX_MOLECULES} per call "
                f"limit. Split them across calls, or filter with molecule_info "
                f"first and only score the survivors."
            )

        payload: dict[str, Any] = {"smiles": smiles}
        if requested:
            payload["properties"] = list(requested)

        try:
            # A full batch is seconds of compute, but the first request after a
            # cold start also waits out the ensemble load, which is tens of
            # seconds. The timeout covers the second case.
            response = await self._call_service(
                endpoint="/predict",
                method="POST",
                json_data=payload,
                timeout=300.0,
            )
        except Exception as e:
            logger.exception("ADMET prediction failed")
            return ToolResult.error(
                f"Failed to call the ADMET service: {e}. The service loads a "
                f"large model ensemble at startup and may still be warming up."
            )

        if not response.get("success"):
            error = response.get("error", "Unknown error from the ADMET service")
            unknown = response.get("unknown_properties") or []
            message = f"ADMET prediction failed: {error}"
            if unknown:
                message += f"\nUnrecognized property names: {', '.join(unknown)}"
            return ToolResult.error(message)

        results = response.get("results") or []
        if not results:
            return ToolResult.error(
                "The ADMET service scored none of the molecules. "
                + self._describe_invalid(response.get("invalid_smiles") or [])
            )

        return await self._render(
            results=results,
            property_info=response.get("property_info") or [],
            invalid=response.get("invalid_smiles") or [],
            unknown=response.get("unknown_properties") or [],
            origin=origin,
            description=description,
        )

    @staticmethod
    def _describe_invalid(invalid: list[str]) -> str:
        """Name the inputs RDKit could not parse."""
        if not invalid:
            return ""
        shown = ", ".join(entry[:60] for entry in invalid[:5])
        suffix = f" (and {len(invalid) - 5} more)" if len(invalid) > 5 else ""
        return f"{len(invalid)} SMILES could not be parsed: {shown}{suffix}"

    async def _render(
        self,
        results: list[dict[str, Any]],
        property_info: list[dict[str, Any]],
        invalid: list[str],
        unknown: list[str],
        origin: dict[str, str],
        description: str | None,
    ) -> ToolResult:
        """Turn the service response into a table, a CSV artifact and a message."""
        # Column order follows the service's property_info, which follows the
        # order the properties were requested in.
        columns = [info["name"] for info in property_info]
        by_name = {info["name"]: info for info in property_info}

        csv_ref = await self._store_csv(results, columns, description)

        lines: list[str] = []
        if len(results) == 1:
            lines.extend(self._render_single(results[0], columns, by_name))
        else:
            lines.extend(self._render_table(results, columns, by_name))
            lines.append("")
            lines.append("Legend:")
            for name in columns:
                info = by_name.get(name, {})
                units = f" [{info['units']}]" if info.get("units") else ""
                direction = info.get("direction", "")
                hint = {
                    "lower": "lower is better",
                    "higher": "higher is better",
                    "context": "depends on the programme",
                }.get(direction, "")
                hint = f" -- {hint}" if hint else ""
                lines.append(
                    f"  {info.get('abbrev', name)} = {info.get('label', name)}"
                    f"{units}{hint}"
                )

        # A call that scores a handful when it could have scored hundreds is the
        # quiet way a screen stays too small. Prose does not fix it -- told in
        # its system prompt that twenty candidates is a coin flip, the agent
        # enumerated several hundred, filtered to two hundred, and then fed
        # them to this tool five at a time, 567 calls at a median batch of 5.
        # So the point of use says so, on every call, where the number is.
        if 0 < len(results) < self.SMALL_BATCH:
            lines.append("")
            lines.append(
                f"Scored {len(results)} molecule(s). This call takes up to "
                f"{MAX_MOLECULES} for the same cost and the same wait -- if you "
                "have a screening set, pass all of it at once rather than a few "
                "at a time. Scoring a handful is how a screen ends up too small "
                "to contain an answer."
            )

        if invalid:
            lines.append("")
            lines.append(self._describe_invalid(invalid))
        if unknown:
            lines.append("")
            lines.append(
                f"Ignored unrecognized property names: {', '.join(unknown)}."
            )

        if csv_ref:
            lines.append("")
            lines.append(
                format_storage_message(
                    data_type="csv",
                    reference=csv_ref,
                    source_description=(
                        f"ADMET predictions for {len(results)} molecule(s)"
                    ),
                    accepting_tools=["python_execute"],
                )
            )
            # The table above is printed with short labels and the file is
            # written with the model's own property names, so a reader that
            # indexes by what it just read gets a KeyError. Naming the columns
            # here costs one line and removes the guess; aevom.load_table parses
            # the file, which the sandbox cannot do by hand because it has
            # neither `csv` nor `pandas`.
            lines.append(
                f"Its columns are: smiles, {', '.join(columns)}. "
                f"aevom.load_table('{csv_ref}') returns one dict per row, keyed "
                "by exactly those names."
            )
            if len(results) > MAX_RENDERED_ROWS:
                lines.append(
                    f"Only the first {MAX_RENDERED_ROWS} of {len(results)} rows "
                    f"are shown above. Rank or filter the rest in "
                    f"python_execute."
                )

        # Predictions themselves go in the metadata so the findings ledger
        # records the numbers rather than having to re-parse the table.
        metadata: dict[str, Any] = {
            "num_molecules": len(results),
            "properties": columns,
            "predictions": {
                result["smiles"]: result.get("properties", {})
                for result in results
            },
            "invalid_smiles": invalid,
        }
        if csv_ref:
            metadata["csv_ref"] = csv_ref
        if origin:
            metadata["smiles_origin"] = origin

        return ToolResult.text("\n".join(lines), metadata=metadata)

    async def _store_csv(
        self,
        results: list[dict[str, Any]],
        columns: list[str],
        description: str | None,
    ) -> str | None:
        """Store the full results as CSV, returning its reference.

        Returns None when storage is unavailable -- the table is still rendered,
        so a missing store degrades the answer rather than failing the call.
        """
        if self._storage is None:
            return None

        buffer = io.StringIO()
        # csv.writer defaults to CRLF, which is correct for the format and wrong
        # for the only consumer this file has: the sandbox has no `csv` module,
        # so the code that reads it back splits on "\n" by hand and finds a
        # stray "\r" welded to the last field of every row. That turns the last
        # column into a string and the comparison against a threshold into a
        # TypeError, a long way from here.
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(["smiles", *columns])
        for result in results:
            values = result.get("properties", {})
            writer.writerow(
                [result["smiles"], *(values.get(name, "") for name in columns)]
            )

        try:
            return await self._storage.store(
                data_type="csv",
                content=buffer.getvalue().encode("utf-8"),
                metadata={
                    "source": "admet_predict",
                    "num_molecules": len(results),
                    "properties": columns,
                },
                description=description or "ADMET predictions",
            )
        except Exception:
            logger.exception("Failed to store the ADMET results CSV")
            return None

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a prediction for display, keeping columns narrow."""
        if value is None or value == "":
            return "-"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)[:8]
        if abs(number) >= 1000 or (number != 0 and abs(number) < 0.01):
            return f"{number:.2e}"
        return f"{number:.3g}"

    @classmethod
    def _render_single(
        cls,
        result: dict[str, Any],
        columns: list[str],
        by_name: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Render one molecule as a labelled list, as molecule_info does."""
        lines = [f"ADMET predictions for {result['smiles']}:"]
        percentiles = result.get("percentiles") or {}
        for name in columns:
            info = by_name.get(name, {})
            value = result.get("properties", {}).get(name)
            if value is None:
                continue
            units = f" {info['units']}" if info.get("units") else ""
            line = f"- {info.get('label', name)}: {cls._format_value(value)}{units}"
            if info.get("direction") in ("lower", "higher"):
                line += f" ({info['direction']} is better)"
            percentile = percentiles.get(name)
            if percentile is not None:
                line += (
                    f"; {cls._format_value(percentile)}th percentile among "
                    f"approved drugs"
                )
            lines.append(line)
        return lines

    @classmethod
    def _render_table(
        cls,
        results: list[dict[str, Any]],
        columns: list[str],
        by_name: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Render many molecules as a fixed-width table."""
        headers = [by_name.get(name, {}).get("abbrev", name) for name in columns]
        shown = results[:MAX_RENDERED_ROWS]

        rows: list[list[str]] = []
        for result in shown:
            values = result.get("properties", {})
            smiles = result["smiles"]
            if len(smiles) > SMILES_COLUMN_WIDTH:
                smiles = smiles[: SMILES_COLUMN_WIDTH - 1] + "~"
            rows.append(
                [smiles, *(cls._format_value(values.get(name)) for name in columns)]
            )

        widths = [len(header) for header in ["molecule", *headers]]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def line(cells: list[str]) -> str:
            return "  ".join(
                cell.ljust(widths[index]) if index == 0 else cell.rjust(widths[index])
                for index, cell in enumerate(cells)
            )

        out = [
            f"ADMET predictions for {len(results)} molecules:",
            "",
            line(["molecule", *headers]),
            "  ".join("-" * width for width in widths),
        ]
        out.extend(line(row) for row in rows)
        return out
