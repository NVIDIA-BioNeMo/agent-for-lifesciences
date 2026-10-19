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

"""Read and write the candidate ledger.

Three tools over one table. The ledger fills itself from ``molecule_gates`` and
``admet_predict`` results, so the write tool here exists for what those cannot
know -- where a molecule came from, why it is interesting, a score computed in
``python_execute`` -- and not for restating numbers a tool already returned.

The read tools are the point. ``query_candidates`` lists what survived, and
``best_candidate`` answers "which one" against a named objective. Both work off
stored rows rather than the transcript, so a molecule found forty iterations ago
competes on equal terms with one found in the last turn. SMDD-Bench's appendix
attributes 8 to 20 points of lost score to exactly the failure this removes:
enumerating a molecule that passes, then submitting a worse one.
"""

import logging
from typing import Any, Optional

from api.conversations.candidates import (
    canonical,
    is_lower_better,
    rank,
    score_names,
)

from .base import BaseTool, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)

# One line per candidate, so this is a row count rather than a token estimate.
# Enough to hold a full enumeration round without becoming the context problem
# the ledger exists to avoid.
DEFAULT_LIMIT = 30
MAX_LIMIT = 100

# Descriptors worth showing on every row, in reading order. Scores the caller
# asked about are shown too, so this is the always-on part rather than the whole.
_SUMMARY_PROPERTIES: tuple[tuple[str, str], ...] = (
    ("MW", "mw"),
    ("logP", "logp"),
    ("TPSA", "tpsa"),
    ("SA", "sa_score"),
    ("QED", "qed"),
)

_SMILES_WIDTH = 44


def _number(value: Any) -> str:
    """Format a value for a table cell, tersely and without lying about zero."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 1000 or (value != 0 and abs(value) < 0.01):
        return f"{value:.3g}"
    return f"{value:.2f}"


def _verdict(row: dict[str, Any]) -> str:
    passed = row.get("passed")
    if passed is None:
        return "ungated"
    return "pass" if passed else "FAIL"


def _shorten(smiles: str) -> str:
    if len(smiles) <= _SMILES_WIDTH:
        return smiles
    return smiles[: _SMILES_WIDTH - 3] + "..."


def _render_table(
    rows: list[dict[str, Any]], extra_scores: tuple[str, ...] = ()
) -> list[str]:
    """One line per candidate: verdict, molecule, descriptors, asked-for scores."""
    header = ["verdict", "smiles"]
    header.extend(label for label, _ in _SUMMARY_PROPERTIES)
    header.extend(extra_scores)
    lines = ["  ".join(header)]

    for row in rows:
        properties = row.get("properties") or {}
        scores = row.get("scores") or {}
        cells = [_verdict(row), _shorten(row.get("smiles", ""))]
        cells.extend(_number(properties.get(key)) for _, key in _SUMMARY_PROPERTIES)
        for name in extra_scores:
            value = scores.get(name, properties.get(name))
            cells.append(_number(value))
        lines.append("  ".join(cells))
    return lines


async def load_candidates() -> list[dict[str, Any]]:
    """Read the session's candidate ledger from whichever store backs it.

    Production has a recorder writing to PostgreSQL; a training rollout keeps the
    ledger in process. Both are read here so the tool is not something that only
    exists in deployment -- a tool absent during training is a tool the policy
    learns never to reach for.
    """
    from api.conversations import get_recorder
    from api.conversations.candidates import get_candidates_source

    recorder = get_recorder()
    if recorder is not None:
        try:
            return await recorder.get_candidates()
        except Exception:
            logger.warning("Could not read the candidate ledger", exc_info=True)
            return []

    source = get_candidates_source()
    if source is None:
        return []
    try:
        return list(source())
    except Exception:
        logger.warning("Could not read the in-memory candidate ledger", exc_info=True)
        return []


async def store_candidate(**fields: Any) -> bool:
    """Write one candidate to whichever store backs the session."""
    from api.conversations import get_recorder
    from api.conversations.candidates import get_candidates_source, merge_in_memory

    recorder = get_recorder()
    if recorder is not None:
        return await recorder.upsert_candidate(**fields) is not None

    source = get_candidates_source()
    if source is None:
        return False
    try:
        merge_in_memory(source(), fields)
        return True
    except Exception:
        logger.warning("Could not write the in-memory candidate ledger", exc_info=True)
        return False


class RecordCandidatesTool(BaseTool):
    """Add molecules to the candidate ledger, or annotate ones already there.

    Data Flow:
        Input: SMILES with optional provenance, scores and notes
        Output: text confirmation of what was recorded
    """

    @property
    def name(self) -> str:
        return "record_candidates"

    @property
    def summary_template(self) -> str:
        return "Recording candidates..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "record_candidates",
                "description": (
                    "Add molecules to this session's candidate ledger, or attach "
                    "scores and notes to molecules already in it.\n\n"
                    "You do NOT need to record gate or ADMET results: every "
                    "molecule_gates and admet_predict call files its own numbers "
                    "automatically, keyed by molecule. Use this tool for what "
                    "those cannot know:\n"
                    "- a molecule you have just enumerated, before scoring it, so "
                    "  the enumeration survives even if the scoring fails\n"
                    "- a score you computed yourself in python_execute\n"
                    "- an affinity or pIC50 you want ranked alongside the rest\n"
                    "- why a molecule is interesting, in notes\n\n"
                    "Record the whole batch in one call. Recording a molecule "
                    "twice merges rather than duplicating, so annotating later is "
                    "safe."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "candidates": {
                            "type": "array",
                            "description": "The molecules to record or annotate.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "smiles": {
                                        "type": "string",
                                        "description": "The molecule. Required.",
                                    },
                                    "origin": {
                                        "type": "string",
                                        "description": (
                                            "Where it came from, e.g. 'R-group "
                                            "enumeration on the parent' or "
                                            "'scaffold hop from 1EVE ligand'. "
                                            "Only recorded the first time."
                                        ),
                                    },
                                    "scores": {
                                        "type": "object",
                                        "description": (
                                            "Named numbers to rank on, e.g. "
                                            "{'affinity_kcal_mol': -9.4} or "
                                            "{'pic50': 7.8}. Merged with any "
                                            "already recorded."
                                        ),
                                        "additionalProperties": True,
                                    },
                                    "notes": {
                                        "type": "string",
                                        "description": (
                                            "Why this molecule matters. Replaces "
                                            "any earlier note."
                                        ),
                                    },
                                },
                                "required": ["smiles"],
                            },
                        },
                    },
                    "required": ["candidates"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Record or annotate candidates.

        Args:
            arguments: ``candidates``, a list of dicts each carrying ``smiles``
                and optionally ``origin``, ``scores`` and ``notes``.

        Returns:
            A ToolResult naming what was recorded, or an error.
        """
        entries = arguments.get("candidates")
        if not isinstance(entries, list) or not entries:
            return ToolResult.error(
                "record_candidates needs a non-empty `candidates` list, each "
                "entry with at least a `smiles` field."
            )

        recorded: list[str] = []
        rejected: list[str] = []

        for entry in entries:
            if not isinstance(entry, dict):
                rejected.append(str(entry))
                continue
            smiles = canonical(str(entry.get("smiles") or ""))
            if not smiles:
                rejected.append(str(entry.get("smiles") or entry))
                continue

            scores = entry.get("scores")
            numeric = (
                {
                    str(name): float(value)
                    for name, value in scores.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
                if isinstance(scores, dict)
                else {}
            )

            stored = await store_candidate(
                smiles=smiles,
                origin=str(entry.get("origin") or "") or None,
                scores=numeric or None,
                notes=str(entry.get("notes") or "") or None,
            )
            if stored:
                recorded.append(smiles)
            else:
                rejected.append(smiles)

        if not recorded:
            return ToolResult.error(
                "Nothing was recorded. This session has no candidate ledger "
                "available, so record your candidates in your reply instead and "
                "keep comparing them yourself."
            )

        message = (
            f"Recorded {len(recorded)} "
            f"{'candidate' if len(recorded) == 1 else 'candidates'} in the "
            f"session ledger. Use query_candidates to list them or "
            f"best_candidate to pick a winner."
        )
        if rejected:
            message += (
                f"\n\nNot recorded ({len(rejected)}): "
                f"{', '.join(rejected[:5])}"
                f"{' ...' if len(rejected) > 5 else ''}. "
                "Each entry needs a parseable `smiles`."
            )

        return ToolResult.text(
            message,
            metadata={
                "recorded": recorded,
                "rejected": rejected,
                "num_recorded": len(recorded),
            },
        )


class QueryCandidatesTool(BaseTool):
    """List the molecules this session has considered, filtered and ranked.

    Data Flow:
        Input: optional score to rank by, optional passing-only filter
        Output: a text table, one row per candidate
    """

    @property
    def name(self) -> str:
        return "query_candidates"

    @property
    def summary_template(self) -> str:
        return "Reviewing candidates..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "query_candidates",
                "description": (
                    "List every molecule this session has considered, with its "
                    "gate verdict, descriptors and any scores.\n\n"
                    "The ledger fills itself from molecule_gates and "
                    "admet_predict, so this works even for molecules scored many "
                    "steps ago and no longer visible in the conversation. Call it "
                    "before you commit to an answer: you may already have "
                    "something better than what you are about to propose.\n\n"
                    "Set passing_only=true to see only molecules that cleared "
                    "every gate they were judged against."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "passing_only": {
                            "type": "boolean",
                            "description": (
                                "Show only molecules that passed their gates. "
                                "Molecules never gated are still shown, since "
                                "ungated is not the same as failing."
                            ),
                        },
                        "rank_by": {
                            "type": "string",
                            "description": (
                                "Score or descriptor to order by, e.g. "
                                "'affinity_kcal_mol', 'pic50', 'qed', 'hERG'. "
                                "Direction is inferred -- affinities and "
                                "liabilities sort ascending, everything else "
                                "descending. Omit to order by gate verdict."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                f"Maximum rows (default {DEFAULT_LIMIT}, max "
                                f"{MAX_LIMIT})."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """List the ledger.

        Args:
            arguments: ``passing_only``, ``rank_by`` and ``limit``, all optional.

        Returns:
            A ToolResult with a table of candidates.
        """
        passing_only = bool(arguments.get("passing_only"))
        rank_by = (arguments.get("rank_by") or "").strip() or None

        try:
            limit = int(arguments.get("limit") or DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT
        limit = max(1, min(limit, MAX_LIMIT))

        candidates = await load_candidates()
        if not candidates:
            return ToolResult.text(
                "The candidate ledger is empty. Molecules land here "
                "automatically when you score them with molecule_gates or "
                "admet_predict, and you can add them yourself with "
                "record_candidates.",
                metadata={"num_candidates": 0},
            )

        available = score_names(candidates)
        unknown_score = rank_by is not None and rank_by not in available

        ordered = rank(
            candidates,
            score=None if unknown_score else rank_by,
            passing_only=passing_only,
        )
        shown = ordered[:limit]

        extra = () if not rank_by or unknown_score else (rank_by,)
        lines = [
            f"{len(candidates)} "
            f"{'candidate' if len(candidates) == 1 else 'candidates'} in this "
            f"session's ledger"
            + (f", {len(ordered)} passing" if passing_only else "")
            + (f", ranked by {rank_by}" if extra else "")
            + ":",
            "",
            *_render_table(shown, extra),
        ]

        if len(ordered) > len(shown):
            lines.append("")
            lines.append(
                f"... and {len(ordered) - len(shown)} more. Raise limit or "
                f"narrow with passing_only."
            )

        if unknown_score:
            lines.append("")
            lines.append(
                f"No candidate carries a score named '{rank_by}', so this is "
                f"ordered by gate verdict instead. Available: "
                f"{', '.join(available) if available else 'none yet'}."
            )

        return ToolResult.text(
            "\n".join(lines),
            metadata={
                "num_candidates": len(candidates),
                "num_shown": len(shown),
                "passing_only": passing_only,
                "rank_by": rank_by,
                "available_scores": available,
            },
        )


class BestCandidateTool(BaseTool):
    """Pick the best molecule in the ledger against a stated objective.

    Data Flow:
        Input: the score to optimize, optional passing-only constraint
        Output: text naming the winner and why, with the runners-up
    """

    @property
    def name(self) -> str:
        return "best_candidate"

    @property
    def summary_template(self) -> str:
        return "Choosing the best candidate..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "best_candidate",
                "description": (
                    "Pick the best molecule in this session's candidate ledger "
                    "against an objective, and show the runners-up.\n\n"
                    "Call this before you submit or recommend a molecule. It "
                    "compares against everything recorded, including molecules "
                    "from far earlier in the session, which is exactly the "
                    "comparison that gets lost when the conversation grows.\n\n"
                    "Defaults to requiring that the winner passed its gates, "
                    "because a molecule that violates a hard constraint is not a "
                    "candidate however good its score."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "rank_by": {
                            "type": "string",
                            "description": (
                                "The objective, e.g. 'affinity_kcal_mol', "
                                "'pic50', 'qed'. Omit to pick on gate verdict "
                                "and violation count alone, which is the right "
                                "choice when the task is only 'find something "
                                "that satisfies the constraints'."
                            ),
                        },
                        "passing_only": {
                            "type": "boolean",
                            "description": (
                                "Require that the winner passed its gates. "
                                "Default true. Set false only to see what the "
                                "best failing molecule would have been."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Choose a winner.

        Args:
            arguments: ``rank_by`` and ``passing_only``, both optional.

        Returns:
            A ToolResult naming the best candidate, or explaining why none
            qualifies.
        """
        rank_by = (arguments.get("rank_by") or "").strip() or None
        passing_only = arguments.get("passing_only")
        passing_only = True if passing_only is None else bool(passing_only)

        candidates = await load_candidates()
        if not candidates:
            return ToolResult.text(
                "The candidate ledger is empty, so there is nothing to choose "
                "between. Score some molecules with molecule_gates first.",
                metadata={"num_candidates": 0},
            )

        available = score_names(candidates)
        if rank_by is not None and rank_by not in available:
            return ToolResult.error(
                f"No candidate carries a score named '{rank_by}'. Available "
                f"scores: {', '.join(available) if available else 'none yet'}. "
                f"Either rank by one of those, record the score with "
                f"record_candidates, or omit rank_by to choose on gate verdict."
            )

        ordered = rank(candidates, score=rank_by, passing_only=passing_only)
        if not ordered:
            failing = len([row for row in candidates if row.get("passed") is False])
            return ToolResult.text(
                f"No candidate passes its gates: all {failing} judged molecules "
                f"have at least one violation. Either relax a soft constraint, "
                f"enumerate differently, or call this again with "
                f"passing_only=false to see the closest miss.",
                metadata={"num_candidates": len(candidates), "num_passing": 0},
            )

        winner = ordered[0]
        runners = ordered[1:4]

        reason = self._reason(winner, rank_by)
        lines = [f"Best candidate: {winner.get('smiles', '')}", "", reason]

        if winner.get("origin"):
            lines.append(f"Origin: {winner['origin']}")
        if winner.get("notes"):
            lines.append(f"Notes: {winner['notes']}")
        if winner.get("failures"):
            lines.append(
                "Outstanding violations: " + "; ".join(winner["failures"][:5])
            )

        if runners:
            lines.append("")
            lines.append("Next best:")
            extra = (rank_by,) if rank_by else ()
            lines.extend(_render_table(runners, extra)[1:])

        return ToolResult.text(
            "\n".join(lines),
            metadata={
                "best_smiles": winner.get("smiles"),
                "rank_by": rank_by,
                "passing_only": passing_only,
                "num_candidates": len(candidates),
                "num_considered": len(ordered),
                "best_scores": winner.get("scores") or {},
                "best_passed": winner.get("passed"),
            },
        )

    @staticmethod
    def _reason(winner: dict[str, Any], rank_by: Optional[str]) -> str:
        """Say why this one won, in the terms it was judged on."""
        verdict = winner.get("passed")
        if rank_by:
            scores = winner.get("scores") or {}
            properties = winner.get("properties") or {}
            value = scores.get(rank_by, properties.get(rank_by))
            direction = "lowest" if is_lower_better(rank_by) else "highest"
            if verdict:
                gated = "and passes every gate"
            elif verdict is None:
                gated = "and has not been gated"
            else:
                gated = "but FAILS a gate"
            return (
                f"It has the {direction} {rank_by} ({_number(value)}) of the "
                f"candidates considered, {gated}."
            )
        if verdict:
            return "It passes every gate it was judged against."
        if verdict is None:
            return (
                "Nothing in the ledger has been gated yet, so this is only the "
                "first molecule recorded. Run molecule_gates before relying on it."
            )
        return (
            f"Nothing passes, so this is the closest miss: "
            f"{len(winner.get('failures') or [])} violation(s)."
        )
