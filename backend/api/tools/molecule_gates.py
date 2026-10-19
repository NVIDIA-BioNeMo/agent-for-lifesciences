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

"""Drug-likeness gates via the rdkit container service.

The verdict a medicinal chemist wants after enumerating analogues is not eight
separate numbers, it is "which of these survive?". Before this tool the model had
to call ``molecule_info`` once per molecule, remember the Lipinski-adjacent
thresholds itself, and had no access at all to synthetic accessibility, PAINS or
Brenk alerts, Tanimoto similarity, or scaffold comparison.

Design notes:

Failures are reported before passes, and named. "MW 641.2 violates <= 600" is
actionable; "failed" is not, and a bare pass/fail count invites the model to
guess which axis to fix.

Thresholds are the service's, not this module's. This tool documents the default
panel in its schema so the model knows what it is being judged against without a
round trip, but the numbers themselves live in ``services/rdkit/app/gates.py``
and come back in every response, so a threshold change lands in one place.

The full grid is stored as CSV for the same reason ``admet_predict`` stores one:
ranking twenty survivors on a weighted objective is python_execute's job.
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

# Documentation only. The service owns the numbers and returns them; this string
# exists so the schema can state the defaults the model will be judged against.
DEFAULT_PANEL_DOC = (
    "MW <= 600, logP in [-1, 5], TPSA <= 140, HBD <= 5, HBA <= 10, "
    "rotatable bonds <= 10, formal charge in [-2, +2], SA score <= 4.5, "
    "PAINS alerts = 0"
)

# Threshold names the service accepts, listed so the model can override the
# right key instead of guessing. Kept in the same order as the panel above.
THRESHOLD_DOC = (
    "mw_max, logp_min, logp_max, tpsa_max, hbd_max, hba_max, "
    "rotatable_bonds_max, formal_charge_min, formal_charge_max, sa_score_max, "
    "pains_max, brenk_max, nih_max, similarity_min, similarity_max, "
    "scaffold_similarity_min, scaffold_similarity_max, "
    "scaffold_mcs_ratio_min, scaffold_mcs_ratio_max"
)

# Columns of the rendered table, in reading order: the gated properties first,
# then the reference-relative ones.
TABLE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("MW", "properties", "mw"),
    ("logP", "properties", "logp"),
    ("TPSA", "properties", "tpsa"),
    ("HBD", "properties", "hbd"),
    ("HBA", "properties", "hba"),
    ("RB", "properties", "rotatable_bonds"),
    ("chg", "properties", "formal_charge"),
    ("SA", "properties", "sa_score"),
    ("QED", "properties", "qed"),
)

REFERENCE_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("Tan", "similarity", "tanimoto"),
    ("scafTan", "similarity", "scaffold_tanimoto"),
    ("MCS", "similarity", "scaffold_mcs_ratio"),
)

MAX_RENDERED_ROWS = 25
SMILES_COLUMN_WIDTH = 34

# Matches the service's own ceiling, checked here so an over-large batch fails
# with actionable advice instead of a 422 from FastAPI's validation.
MAX_MOLECULES = 256


class MoleculeGatesTool(ContainerTool):
    """Score molecules against a parameterized drug-likeness panel.

    Data Flow:
        Input: inline SMILES strings and/or SMILES references
        Output: a pass/fail verdict per molecule, plus a stored CSV
    """

    @property
    def name(self) -> str:
        return "molecule_gates"

    @property
    def summary_template(self) -> str:
        return "Checking drug-likeness gates..."

    @property
    def service_url(self) -> str:
        """URL of the RDKit service container."""
        return os.environ.get("RDKIT_SERVICE_URL", "http://rdkit:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        """Takes references, and stores the results table."""
        return DataIOCategory.BOTH

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "molecule_gates",
                "description": (
                    "Score one or more molecules against a drug-likeness panel "
                    "and get a per-gate pass/fail verdict.\n\n"
                    "Returns the physical property panel (MW, logP, TPSA, HBD, "
                    "HBA, rotatable bonds, formal charge), synthetic "
                    "accessibility, QED, PAINS/Brenk/NIH structural alerts, "
                    "the Bemis-Murcko scaffold, and -- when reference_smiles is "
                    "given -- Morgan Tanimoto similarity plus a scaffold MCS "
                    "ratio.\n\n"
                    f"DEFAULT THRESHOLDS: {DEFAULT_PANEL_DOC}. These are the "
                    "reference-free bounds of the standard drug-likeness "
                    "panel, applied to the exact measured value -- MW, TPSA "
                    "and SA are strict (< bound), the counts are inclusive "
                    "(<= bound). Override any via `thresholds` when the task "
                    "states different limits; pass null to disable a gate.\n\n"
                    "WHAT THIS DOES NOT CHECK unless you ask: similarity to a "
                    "reference (below). If a task states a similarity floor, "
                    "an all-pass verdict here does NOT mean the molecule "
                    "satisfies the task.\n\n"
                    "Score every candidate in ONE call so you can compare them, "
                    "and call this BEFORE spending an expensive oracle "
                    "(vina_dock, boltz2_predict) -- a molecule that fails a hard "
                    "constraint is not worth docking.\n\n"
                    "SIMILARITY IS TASK-DEPENDENT and therefore opt-in. Lead "
                    "optimization wants a candidate close to its parent "
                    "(similarity_min); scaffold hopping wants one deliberately "
                    "far from it (similarity_max plus scaffold_mcs_ratio_max). "
                    "State which you want -- there is no sensible default."
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
                                "['smiles_abc123']). A multi-line artifact is "
                                "read as several molecules."
                            ),
                            "x-aevom-reference": {"data_type": "smiles"},
                        },
                        "reference_smiles": {
                            "type": "string",
                            "description": (
                                "The parent or reference molecule, as a literal "
                                "SMILES. Required for the similarity and "
                                "scaffold gates; without it only the property "
                                "and alert gates run. Prefer reference_ref when "
                                "the molecule is already stored: a reference "
                                "typed out by hand is the one input here that "
                                "can be wrong without being invalid, and every "
                                "similarity number in the call is measured "
                                "against it."
                            ),
                        },
                        "reference_ref": {
                            "type": "string",
                            "description": (
                                "The parent molecule as a stored reference (e.g. "
                                "'sdf_abc12345' or 'smiles_abc12345'), used "
                                "instead of reference_smiles. Use this whenever "
                                "the molecule came from a file or an earlier "
                                "tool call, rather than transcribing it."
                            ),
                            "x-aevom-reference": {"data_type": ["smiles", "sdf"]},
                        },
                        "thresholds": {
                            "type": "object",
                            "description": (
                                "Overrides for the default panel. Recognized "
                                f"names: {THRESHOLD_DOC}. Null disables a gate."
                            ),
                            "additionalProperties": True,
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
            to the reference it came from. When ``error_result`` is not None the
            caller returns it unchanged.
        """
        smiles: list[str] = []
        origin: dict[str, str] = {}

        def add(entry: str, source: str = "") -> None:
            entry = entry.strip()
            # Duplicates produce identical verdicts and waste a table row.
            if not entry or entry in smiles:
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

    async def _resolve_reference(
        self, ref: str
    ) -> tuple[str, ToolResult | None]:
        """Read the parent molecule out of a stored artifact."""
        if self._storage is None:
            return "", ToolResult.error(
                "Storage service not available, so reference_ref cannot be "
                "resolved. Pass reference_smiles instead."
            )
        try:
            content = await self._storage.retrieve(ref)
        except InvalidReferenceError:
            return "", ToolResult.error(
                f"Invalid reference: '{ref}'. A reference looks like "
                f"'sdf_abc12345' or 'smiles_abc12345'."
            )
        except ArtifactNotFoundError:
            return "", ToolResult.error(
                f"No molecule found for reference '{ref}'."
            )

        text = content.decode("utf-8", errors="replace")
        if "$$$$" in text or "V2000" in text or "V3000" in text:
            # An SDF has to be read by the service, which has RDKit; this image
            # does not.
            return text, None
        for line in text.splitlines():
            parts = line.split()
            if parts:
                return parts[0], None
        return "", ToolResult.error(
            f"Reference '{ref}' holds no molecule."
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Score the requested molecules against the gate panel.

        Args:
            arguments: Dict with 'smiles' and/or 'smiles_refs', plus optional
                'reference_smiles', 'thresholds' and 'description'.

        Returns:
            ToolResult with the verdict, a table and a CSV reference, or an error.
        """
        inline = arguments.get("smiles") or []
        refs = arguments.get("smiles_refs") or []
        reference_smiles = (arguments.get("reference_smiles") or "").strip()
        reference_ref = (arguments.get("reference_ref") or "").strip()
        thresholds = arguments.get("thresholds") or None
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
                f"limit. Split them across calls."
            )

        if thresholds is not None and not isinstance(thresholds, dict):
            return ToolResult.error(
                "`thresholds` must be an object mapping threshold names to "
                f"numbers, e.g. {{'mw_max': 500}}. Recognized: {THRESHOLD_DOC}"
            )

        reference_note = ""
        if reference_ref:
            resolved, failure = await self._resolve_reference(reference_ref)
            if failure is not None:
                return failure
            if reference_smiles and reference_smiles != resolved:
                # Both given and they disagree. The stored one wins, and the
                # disagreement is reported rather than resolved silently: a
                # mistyped reference does not fail, it quietly measures every
                # similarity in the call against a molecule that is not the
                # parent, and the run reads as if it had passed.
                reference_note = (
                    f"Two references were given and they are not the same "
                    f"molecule. Used the stored one, {reference_ref}: "
                    f"{resolved}. The literal you passed was: "
                    f"{reference_smiles}"
                )
            reference_smiles = resolved

        payload: dict[str, Any] = {"smiles": smiles}
        if reference_smiles:
            payload["reference_smiles"] = reference_smiles
        if thresholds:
            payload["thresholds"] = thresholds

        try:
            response = await self._call_service(
                endpoint="/molecule-gates",
                method="POST",
                json_data=payload,
                timeout=180.0,
            )
        except Exception as e:
            logger.exception("Gate evaluation failed")
            return ToolResult.error(f"Failed to call the RDKit service: {e}")

        if not response.get("success"):
            return ToolResult.error(
                "Gate evaluation failed: "
                + response.get("error", "Unknown error from the RDKit service")
            )

        results = response.get("results") or []
        if not results:
            return ToolResult.error(
                "None of the molecules could be scored. "
                + self._describe_invalid(response.get("invalid_smiles") or [])
            )

        return await self._render(
            results=results,
            invalid=response.get("invalid_smiles") or [],
            thresholds=response.get("thresholds") or {},
            unknown=response.get("unknown_thresholds") or [],
            reference_smiles=response.get("reference_smiles") or "",
            reference_note=reference_note,
            reference_scaffold=response.get("reference_scaffold") or "",
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
        invalid: list[str],
        thresholds: dict[str, Any],
        unknown: list[str],
        reference_smiles: str,
        reference_scaffold: str,
        origin: dict[str, str],
        description: str | None,
        reference_note: str = "",
    ) -> ToolResult:
        """Turn the service response into a verdict, a table and a CSV."""
        has_reference = bool(reference_smiles)
        columns = list(TABLE_COLUMNS) + (
            list(REFERENCE_COLUMNS) if has_reference else []
        )

        csv_ref = await self._store_csv(results, columns, description)

        passed = [result for result in results if result["passed"]]
        failed = [result for result in results if not result["passed"]]

        lines: list[str] = []
        if len(results) == 1:
            lines.extend(self._render_single(results[0], thresholds))
        else:
            lines.append(
                f"{len(passed)} of {len(results)} molecules pass every gate."
            )
            if passed:
                lines.append("")
                lines.append("PASSING:")
                lines.extend(f"  {result['smiles']}" for result in passed[:10])
                if len(passed) > 10:
                    lines.append(f"  ... and {len(passed) - 10} more")
            if failed:
                lines.append("")
                lines.append("FAILING:")
                for result in failed[:10]:
                    lines.append(
                        f"  {self._truncate(result['smiles'])} -- "
                        + "; ".join(result.get("failures", []))
                    )
                if len(failed) > 10:
                    lines.append(f"  ... and {len(failed) - 10} more")
            lines.append("")
            lines.extend(self._render_table(results, columns))

        # Said beside the verdict, never folded into it: these twelve gates are
        # the published panel and match the reference scorer exactly, so a
        # thirteenth would reject molecules that score. But a molecule carrying
        # one of these cannot be made, and no alert catalogue here says so --
        # PAINS, Brenk and NIH pass every one, and the property models return
        # confident numbers for all of them.
        unstable = [r for r in results if r.get("unstable_group")]
        if unstable:
            lines.append("")
            lines.append(
                f"{len(unstable)} of these carry a group that cannot be isolated. "
                "They are not gate failures -- the panel above is the published "
                "one -- but they are not candidates either:"
            )
            for result in unstable[:8]:
                lines.append(
                    f"  {self._truncate(result['smiles'])} -- "
                    f"{result['unstable_group']}"
                )
            if len(unstable) > 8:
                lines.append(f"  ... and {len(unstable) - 8} more")

        if reference_note:
            lines.append("")
            lines.append(reference_note)

        if has_reference:
            lines.append("")
            lines.append(f"Reference: {reference_smiles}")
            if reference_scaffold:
                lines.append(f"Reference scaffold: {reference_scaffold}")

        repeats = await self._repeat_offences(results)
        if repeats:
            lines.append("")
            lines.extend(repeats)

        lines.append("")
        lines.append("Thresholds applied: " + self._describe_thresholds(thresholds))

        if invalid:
            lines.append("")
            lines.append(self._describe_invalid(invalid))
        if unknown:
            lines.append("")
            lines.append(
                f"Ignored unrecognized threshold names: {', '.join(unknown)}. "
                f"Recognized: {THRESHOLD_DOC}"
            )

        if csv_ref:
            lines.append("")
            lines.append(
                format_storage_message(
                    data_type="csv",
                    reference=csv_ref,
                    source_description=(
                        f"Drug-likeness gate results for {len(results)} molecule(s)"
                    ),
                    accepting_tools=["python_execute"],
                )
            )
            if len(results) > MAX_RENDERED_ROWS:
                lines.append(
                    f"Only the first {MAX_RENDERED_ROWS} of {len(results)} rows "
                    f"are in the table above. Rank or filter the rest in "
                    f"python_execute: aevom.load('{csv_ref}')."
                )

        # The verdict goes in the metadata so the candidate ledger and the
        # findings ledger can record per-gate outcomes without re-parsing prose.
        metadata: dict[str, Any] = {
            "num_molecules": len(results),
            "num_passed": len(passed),
            "num_failed": len(failed),
            "passing_smiles": [result["smiles"] for result in passed],
            "thresholds": thresholds,
            "invalid_smiles": invalid,
            "verdicts": {
                result["smiles"]: {
                    "passed": result["passed"],
                    "failures": result.get("failures", []),
                    "properties": result.get("properties", {}),
                    "similarity": result.get("similarity", {}),
                    "alerts": result.get("alerts", {}),
                    # Per-gate rows, not just the prose failures: the exclusion
                    # rules derived downstream need the bound each molecule
                    # missed, which the failure strings only carry as text.
                    "gates": result.get("gates", []),
                }
                for result in results
            },
        }
        if csv_ref:
            metadata["csv_ref"] = csv_ref
        if reference_smiles:
            metadata["reference_smiles"] = reference_smiles
        if origin:
            metadata["smiles_origin"] = origin

        return ToolResult.text("\n".join(lines), metadata=metadata)

    async def _repeat_offences(self, results: list[dict[str, Any]]) -> list[str]:
        """Name the features in this batch that already cost earlier candidates.

        Said here, at the moment the same feature fails again, because that is
        where it changes what the agent does next. The ``[EXCLUSIONS]`` block
        carries the same rules into the prompt for the enumeration after this
        one; this is the immediate half of the same idea.

        Only structural alerts are reported. An alert is a substructure match, so
        "this molecule failed" and "this group is disqualifying" are the same
        claim. A weight or logP violation says nothing about which fragment
        caused it, and inventing an attribution would be worse than silence.
        """
        offenders: dict[str, int] = {}
        for result in results:
            if result.get("passed"):
                continue
            for family, hits in (result.get("alerts") or {}).items():
                if not isinstance(hits, list):
                    continue
                for hit in hits:
                    offenders[str(hit)] = offenders.get(str(hit), 0) + 1

        if not offenders:
            return []

        try:
            from api.conversations.exclusions import derive
            from api.tools.candidates import load_candidates

            known = {
                rule.feature: rule.count
                for rule in derive(await load_candidates())
                if rule.kind == "alert"
            }
        except Exception:
            logger.debug("Could not read earlier exclusions", exc_info=True)
            return []

        repeated = sorted(
            (
                (feature, count, known[feature])
                for feature, count in offenders.items()
                if feature in known
            ),
            key=lambda entry: -entry[2],
        )
        if not repeated:
            return []

        lines = [
            "Already disqualifying (seen in earlier candidates too -- stop "
            "proposing these):"
        ]
        for feature, here, before in repeated[:5]:
            lines.append(
                f"  {feature}: {here} in this batch, {before} earlier "
                f"{'candidate' if before == 1 else 'candidates'}"
            )
        return lines

    @staticmethod
    def _describe_thresholds(thresholds: dict[str, Any]) -> str:
        """Render the applied bounds compactly, in a stable order."""
        if not thresholds:
            return "none"
        return ", ".join(
            f"{name}={value}" for name, value in sorted(thresholds.items())
        )

    @staticmethod
    def _truncate(smiles: str) -> str:
        """Shorten a SMILES for a table cell or a failure line."""
        if len(smiles) <= SMILES_COLUMN_WIDTH:
            return smiles
        return smiles[: SMILES_COLUMN_WIDTH - 1] + "~"

    async def _store_csv(
        self,
        results: list[dict[str, Any]],
        columns: list[tuple[str, str, str]],
        description: str | None,
    ) -> str | None:
        """Store the full results as CSV, returning its reference.

        Returns None when storage is unavailable -- the verdict is still
        rendered, so a missing store degrades the answer rather than failing.
        """
        if self._storage is None:
            return None

        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow(
            [
                "smiles",
                *(header for header, _, _ in columns),
                "murcko_scaffold",
                "pains",
                "brenk",
                "nih",
                "passed",
                "failures",
            ]
        )
        for result in results:
            alerts = result.get("alerts", {})
            writer.writerow(
                [
                    result["smiles"],
                    *(
                        result.get(section, {}).get(key, "")
                        for _, section, key in columns
                    ),
                    result.get("similarity", {}).get("murcko_scaffold", ""),
                    len(alerts.get("pains", [])),
                    len(alerts.get("brenk", [])),
                    len(alerts.get("nih", [])),
                    result["passed"],
                    "; ".join(result.get("failures", [])),
                ]
            )

        try:
            return await self._storage.store(
                data_type="csv",
                content=buffer.getvalue().encode("utf-8"),
                metadata={
                    "source": "molecule_gates",
                    "num_molecules": len(results),
                },
                description=description or "Drug-likeness gate results",
            )
        except Exception:
            logger.exception("Failed to store the gate results CSV")
            return None

    @staticmethod
    def _format_value(value: Any) -> str:
        """Format a descriptor for display, keeping columns narrow."""
        if value is None or value == "":
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)[:8]
        if float(number).is_integer() and abs(number) < 1000:
            return str(int(number))
        return f"{number:.3g}"

    @classmethod
    def _render_single(
        cls, result: dict[str, Any], thresholds: dict[str, Any]
    ) -> list[str]:
        """Render one molecule as a verdict followed by every gate."""
        verdict = "PASSES all gates" if result["passed"] else "FAILS"
        lines = [f"{result['smiles']} {verdict}."]

        if not result["passed"]:
            lines.append("")
            lines.append("Violations:")
            lines.extend(f"  - {failure}" for failure in result.get("failures", []))

        lines.append("")
        lines.append("Gates:")
        for gate in result.get("gates", []):
            mark = "pass" if gate["passed"] else "FAIL"
            value = cls._format_value(gate.get("value"))
            detail = f" [{gate['detail']}]" if gate.get("detail") else ""
            lines.append(
                f"  [{mark}] {gate['label']}: {value} (limit {gate['bound']}){detail}"
            )

        properties = result.get("properties", {})
        lines.append("")
        lines.append("Also computed:")
        lines.append(f"  Formula: {properties.get('molecular_formula', '-')}")
        lines.append(f"  QED: {cls._format_value(properties.get('qed'))}")
        lines.append(f"  Rings: {cls._format_value(properties.get('num_rings'))}")
        lines.append(
            f"  Heavy atoms: {cls._format_value(properties.get('heavy_atoms'))}"
        )

        similarity = result.get("similarity", {})
        if similarity.get("murcko_scaffold"):
            lines.append(f"  Murcko scaffold: {similarity['murcko_scaffold']}")

        if result.get("unstable_group"):
            lines.append("")
            lines.append(
                f"Not a gate failure, but not a candidate: this molecule "
                f"contains {result['unstable_group']}, which does not survive "
                "contact with water. The twelve gates above are the published "
                "panel and do not test for it."
            )

        alerts = result.get("alerts", {})
        for catalog, label in (
            ("pains", "PAINS"),
            ("brenk", "Brenk"),
            ("nih", "NIH"),
        ):
            hits = alerts.get(catalog) or []
            if hits:
                lines.append(f"  {label} alerts: {', '.join(hits[:5])}")

        return lines

    @classmethod
    def _render_table(
        cls, results: list[dict[str, Any]], columns: list[tuple[str, str, str]]
    ) -> list[str]:
        """Render many molecules as a fixed-width table."""
        headers = [header for header, _, _ in columns]
        shown = results[:MAX_RENDERED_ROWS]

        rows: list[list[str]] = []
        for result in shown:
            rows.append(
                [
                    cls._truncate(result["smiles"]),
                    *(
                        cls._format_value(result.get(section, {}).get(key))
                        for _, section, key in columns
                    ),
                    "pass" if result["passed"] else f"FAIL:{result['num_failed']}",
                ]
            )

        header_cells = ["molecule", *headers, "gates"]
        widths = [len(cell) for cell in header_cells]
        for row in rows:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        def line(cells: list[str]) -> str:
            return "  ".join(
                cell.ljust(widths[index]) if index == 0 else cell.rjust(widths[index])
                for index, cell in enumerate(cells)
            )

        out = [
            line(header_cells),
            "  ".join("-" * width for width in widths),
        ]
        out.extend(line(row) for row in rows)
        return out
