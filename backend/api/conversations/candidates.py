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

"""The candidate ledger: one row per molecule the session has considered.

The findings ledger answers "what did I do". This answers "which molecule was
best", which turns out to be a different question with a different grain. A tool
call is one event; a molecule accumulates evidence across many. Asking a reader
to reassemble a molecule from scattered tool-call rows is asking it to redo the
only part of the work that matters.

Two things make this ledger cheap enough to be worth having:

**Nothing has to be transcribed.** ``molecule_gates`` and ``admet_predict``
already return their results structurally, so :func:`capture_from_result` reads
their metadata and writes candidate rows without the agent restating a single
number. An agent asked to copy fifteen descriptors into a tool call would get
some of them wrong, and the errors would be invisible.

**Comparison happens outside the transcript.** SMDD-Bench's appendix reports
agents enumerating a molecule that satisfies every constraint, continuing to
work, and then submitting a worse one. That is not a chemistry failure; it is the
good molecule falling out of the context window. Ranking against a stored table
does not care how long ago the winner was found.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Scores where a smaller number is the better one. Docking affinities are
# negative binding free energies, so -9.5 beats -7.2; everything else here is a
# liability being minimized. Getting this wrong would rank the ledger backwards,
# which is worse than not ranking at all.
LOWER_IS_BETTER = frozenset(
    {
        "affinity_kcal_mol",
        "log_ic50_um",
        "ic50_um",
        "herg",
        "ames",
        "clearance",
        "sa_score",
    }
)

# Metadata keys the capture step reads, per tool. Explicit rather than inferred:
# a tool that changes its metadata shape should break a test here rather than
# silently stop populating the ledger.
_GATE_TOOLS = frozenset({"molecule_gates"})
_ADMET_TOOLS = frozenset({"admet_predict"})


def is_lower_better(score_name: str) -> bool:
    """Whether a smaller value of this score is the better one."""
    return score_name.lower() in LOWER_IS_BETTER


def canonical(smiles: str) -> str:
    """Canonical SMILES, or the input unchanged when RDKit is unavailable.

    Identity in this ledger is the canonical form, so that a molecule written
    two ways is one candidate rather than two. RDKit is not installed in the API
    image -- canonicalization normally happens in the rdkit service -- so this
    degrades to the input string. Two spellings then produce two rows, which is
    a worse ledger but still a correct one.
    """
    text = (smiles or "").strip()
    if not text:
        return ""
    try:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return text
        return Chem.MolToSmiles(mol)
    except ImportError:
        return text
    except Exception:
        logger.debug("Could not canonicalize %r", text, exc_info=True)
        return text


def capture_from_result(tool_name: str, result: Any) -> list[dict[str, Any]]:
    """Read candidate rows out of a tool result's metadata.

    Returns the arguments for :meth:`ConversationStore.upsert_candidate`, one per
    molecule, or an empty list for a tool that says nothing about molecules.

    Args:
        tool_name: The tool that produced the result.
        result: The ``ToolResult``.
    """
    metadata = getattr(result, "metadata", None) or {}
    if not isinstance(metadata, dict):
        return []

    if tool_name in _GATE_TOOLS:
        return _capture_gates(tool_name, metadata)
    if tool_name in _ADMET_TOOLS:
        return _capture_admet(tool_name, metadata)
    return []


def _capture_gates(tool_name: str, metadata: dict) -> list[dict[str, Any]]:
    """Turn molecule_gates verdicts into candidate rows."""
    verdicts = metadata.get("verdicts")
    if not isinstance(verdicts, dict):
        return []

    rows: list[dict[str, Any]] = []
    for smiles, verdict in verdicts.items():
        if not isinstance(verdict, dict):
            continue
        properties = verdict.get("properties")
        similarity = verdict.get("similarity")
        merged = dict(properties) if isinstance(properties, dict) else {}
        if isinstance(similarity, dict):
            # Similarity to the reference is a property of this molecule in this
            # session, so it belongs on the row rather than only in the prose.
            for key in ("tanimoto", "scaffold_tanimoto", "scaffold_mcs_ratio"):
                if key in similarity:
                    merged[key] = similarity[key]

        rows.append(
            {
                "smiles": smiles,
                "origin": tool_name,
                "properties": merged,
                "gates": _gate_rows(verdict),
                "failures": list(verdict.get("failures") or []),
                "passed": bool(verdict.get("passed")),
            }
        )
    return rows


def _gate_rows(verdict: dict) -> list[dict[str, Any]]:
    """Per-gate pass/fail, stored narrowly.

    The service returns a label and a formatted detail string per gate as well.
    Only what a later comparison needs is kept -- which gate, what bound, did it
    pass -- because the prose can always be regenerated and the row cannot.

    Structural alerts are the exception: their names are kept in ``hits``. An
    alert is a substructure match, so it is the one kind of failure attributable
    to a feature rather than to the whole molecule, and that attribution is what
    :mod:`api.conversations.exclusions` derives its rules from.
    """
    alerts = verdict.get("alerts")
    rows = [
        {
            "name": gate.get("name"),
            "value": gate.get("value"),
            "bound": gate.get("bound"),
            "passed": bool(gate.get("passed")),
        }
        for gate in (verdict.get("gates") or [])
        if isinstance(gate, dict)
    ]
    if isinstance(alerts, dict):
        for family, hits in alerts.items():
            if isinstance(hits, list) and hits:
                rows.append(
                    {
                        "name": f"{family}_alerts",
                        "value": len(hits),
                        "bound": 0,
                        "passed": False,
                        "hits": [str(hit) for hit in hits],
                    }
                )
    return rows


def _capture_admet(tool_name: str, metadata: dict) -> list[dict[str, Any]]:
    """Turn an ADMET panel into candidate scores.

    ADMET values are scores rather than properties: they are model predictions
    to rank on, not measured descriptors of the structure. Keeping them in
    ``scores`` is what lets ``best_candidate`` rank on hERG without also
    offering to rank on molecular weight.
    """
    predictions = metadata.get("predictions")
    if not isinstance(predictions, dict):
        return []

    rows: list[dict[str, Any]] = []
    for smiles, properties in predictions.items():
        if not isinstance(properties, dict):
            continue
        numeric = {
            name: float(value)
            for name, value in properties.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric:
            rows.append({"smiles": smiles, "origin": tool_name, "scores": numeric})
    return rows


def rank(
    candidates: list[dict[str, Any]],
    *,
    score: Optional[str] = None,
    passing_only: bool = False,
) -> list[dict[str, Any]]:
    """Order candidates best-first.

    With a score name, ranks on it, respecting whether smaller is better and
    putting candidates that lack the score last -- an unmeasured molecule is not
    a bad one, but it cannot win a comparison it did not enter.

    Without a score name, orders by whether they passed their gates and then by
    how few violations they had, which is the only ranking available before any
    score exists.

    Args:
        candidates: Rows from the ledger.
        score: Score to rank on.
        passing_only: Drop candidates that failed a gate. Candidates not yet
            judged are kept, since "unjudged" is not "failed".
    """
    rows = list(candidates)
    if passing_only:
        rows = [row for row in rows if row.get("passed") is not False]

    if not score:
        return sorted(
            rows,
            key=lambda row: (
                0 if row.get("passed") else 1,
                len(row.get("failures") or []),
                row.get("smiles") or "",
            ),
        )

    lower_better = is_lower_better(score)

    def sort_key(row: dict[str, Any]):
        value = _score_of(row, score)
        if value is None:
            return (1, 0.0, row.get("smiles") or "")
        return (0, value if lower_better else -value, row.get("smiles") or "")

    return sorted(rows, key=sort_key)


def _score_of(row: dict[str, Any], score: str) -> Optional[float]:
    """Look a score up in ``scores`` and then ``properties``.

    Both are searched because the split between them is about provenance, not
    about what is rankable, and a caller asking to rank by logP should not have
    to know which bucket logP landed in.
    """
    for bucket in ("scores", "properties"):
        values = row.get(bucket)
        if isinstance(values, dict) and score in values:
            value = values[score]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
    return None


def apply_update(
    existing: Optional[dict[str, Any]], update: dict[str, Any]
) -> dict[str, Any]:
    """Resolve one candidate's fields after applying an update.

    The single definition of what "record what is now known" means, shared by the
    SQL store and the in-memory ledger. Three rules, each chosen against a way
    the ledger could otherwise mislead:

    An omitted field keeps its stored value. Measurements arrive separately -- a
    gate verdict now, an ADMET panel later -- and a partial write that cleared
    what it did not mention would erase evidence, which is worse than never
    recording it because the gap would go unnoticed.

    ``properties`` and ``scores`` merge key by key, for the same reason at a
    finer grain.

    ``gates`` and ``failures`` replace wholesale, because they are a single
    judgement rather than an accumulation. Merging two judgements would produce a
    verdict neither tool issued.

    Args:
        existing: The stored candidate, or None when this is the first write.
        update: Fields to apply. None or absent means "no news", which is
            distinct from an explicit empty list.

    Returns:
        The field values to store, excluding ``smiles``.
    """
    current = existing or {}
    merged: dict[str, Any] = {
        "origin": current.get("origin") or "",
        "properties": dict(current.get("properties") or {}),
        "scores": dict(current.get("scores") or {}),
        "gates": list(current.get("gates") or []),
        "failures": list(current.get("failures") or []),
        "passed": current.get("passed"),
        "notes": current.get("notes") or "",
    }

    if update.get("properties"):
        merged["properties"].update(update["properties"])
    if update.get("scores"):
        merged["scores"].update(update["scores"])
    if update.get("gates") is not None:
        merged["gates"] = list(update["gates"])
    if update.get("failures") is not None:
        merged["failures"] = list(update["failures"])
    if update.get("passed") is not None:
        merged["passed"] = update["passed"]
    if update.get("notes"):
        merged["notes"] = update["notes"]
    # Provenance is a fact about the molecule's first appearance, so a later
    # write cannot restate it -- otherwise molecule_gates, which sees every
    # molecule, would overwrite the enumeration step that actually produced it.
    if update.get("origin") and not merged["origin"]:
        merged["origin"] = update["origin"]

    return merged


def score_names(candidates: list[dict[str, Any]]) -> list[str]:
    """Every score present on any candidate, sorted.

    Used to tell a caller what it could have ranked by when it names something
    that is not there.
    """
    names: set[str] = set()
    for row in candidates:
        for bucket in ("scores", "properties"):
            values = row.get(bucket)
            if isinstance(values, dict):
                names.update(str(key) for key in values)
    return sorted(names)


# In-memory ledger for regimes without a database -- training rollouts and the
# NeMo-Gym resource server -- mirroring the findings ledger's arrangement so
# both behave the same when there is no DB session to write to.
_candidates_source_ctx: ContextVar[
    Optional[Callable[[], list[dict[str, Any]]]]
] = ContextVar("candidates_source", default=None)


def set_candidates_source(
    source: Optional[Callable[[], list[dict[str, Any]]]],
) -> Any:
    """Provide an in-memory ledger for this request. Returns a reset token."""
    return _candidates_source_ctx.set(source)


def reset_candidates_source(token: Any) -> None:
    """Withdraw the in-memory ledger."""
    _candidates_source_ctx.reset(token)


def get_candidates_source() -> Optional[Callable[[], list[dict[str, Any]]]]:
    """This request's in-memory ledger accessor, if one was provided."""
    return _candidates_source_ctx.get()


def merge_in_memory(
    rows: list[dict[str, Any]], update: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply one upsert to an in-memory ledger.

    A thin adapter over :func:`apply_update`, which the SQL path uses too. One
    merge implementation on purpose: two that drifted would mean training and
    production remembered different things, and the divergence would surface as
    an unreproducible reward rather than as a failing test.

    Args:
        rows: The ledger, mutated in place.
        update: Upsert arguments, as :func:`capture_from_result` produces them.

    Returns:
        The same list, for convenience.
    """
    smiles = update.get("smiles")
    if not smiles:
        return rows

    existing = next((row for row in rows if row.get("smiles") == smiles), None)
    merged = apply_update(existing, update)
    if existing is None:
        rows.append({"smiles": smiles, **merged})
    else:
        existing.update(merged)
    return rows
