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

"""Extract and render findings -- the facts a tool call produced.

The advantage this domain has over general-purpose agent memory is that its
results are not prose. A dock produces a number. A fold produces pLDDT and
ipTM. Both produce references into a store that already exists. Facts like
those can be recorded once and re-rendered verbatim forever, with no model
call, no hallucination, and no per-request cost -- which is why this system
leans on a ledger instead of summarization.

The rendered ``[SESSION LOG]`` block is what makes "re-dock that ligand to
recover the -9.5 we measured twenty iterations ago" unnecessary.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Any, Callable, Optional

from api.state.format import format_session_log as _format_session_log

# Reference format is ``{type}_{8 hex}``. The type may itself contain an
# underscore ("skill_file"), so the leading segments are matched greedily and
# backtracked -- otherwise skill_file_a1b2c3d4 records as file_a1b2c3d4, and a
# ref that does not resolve is worse than no ref at all.
_REF_PATTERN = re.compile(r"\b([a-z]+(?:_[a-z]+)*_[a-f0-9]{8})\b")

# Canonical scalar name -> the metadata keys tools actually populate.
# Several names exist for the same quantity because OpenFold3's API and the
# viewer metadata disagree; the ledger normalizes on one.
SCALAR_KEYS: dict[str, tuple[str, ...]] = {
    "affinity_kcal_mol": ("affinity_kcal_mol",),
    "plddt": ("plddt_score", "complex_plddt_score"),
    "iptm": ("iptm_score",),
    "ptm": ("ptm_score",),
    "confidence": ("confidence_score",),
    "molecular_weight": ("molecular_weight",),
    "logp": ("logp",),
    "tpsa": ("tpsa",),
}

# How each scalar reads in the session log. Units matter: "-9.5" alone
# invites the model to compare it against a pLDDT.
_SCALAR_RENDERERS: dict[str, str] = {
    "affinity_kcal_mol": "{:.2f} kcal/mol",
    "plddt": "pLDDT {:.1f}",
    "iptm": "ipTM {:.3f}",
    "ptm": "pTM {:.3f}",
    "confidence": "confidence {:.3f}",
    "molecular_weight": "MW {:.1f}",
    "logp": "LogP {:.2f}",
    "tpsa": "TPSA {:.1f}",
}

# Arguments long enough to be data rather than a handle. A SMILES string or a
# block of Python in the log would drown the facts it exists to surface.
_MAX_ARG_CHARS = 40

# What a stored argument is allowed to cost. Generous next to the rendered
# line, because retrieval searches this column, but bounded -- a session's
# worth of python_execute source would otherwise dominate the table.
_MAX_STORED_ARG_CHARS = 500


def _find_refs(value: Any) -> list[str]:
    """Return artifact references appearing anywhere in a value."""
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [ref for item in value for ref in _find_refs(item)]
    if isinstance(value, dict):
        return [ref for item in value.values() for ref in _find_refs(item)]
    return _REF_PATTERN.findall(str(value))


def _extract_scalars(metadata: dict[str, Any]) -> dict[str, float]:
    """Pull known numeric results out of a tool's metadata."""
    scalars: dict[str, float] = {}
    for canonical, source_keys in SCALAR_KEYS.items():
        for key in source_keys:
            value = metadata.get(key)
            if value is None:
                continue
            try:
                scalars[canonical] = float(value)
            except (TypeError, ValueError):
                continue
            break
    return scalars


def _render_scalars(scalars: dict[str, Any]) -> str:
    """Render scalars the way a chemist would read them back."""
    parts = []
    for name, value in scalars.items():
        template = _SCALAR_RENDERERS.get(name)
        if template is None:
            parts.append(f"{name} {value}")
            continue
        try:
            parts.append(template.format(float(value)))
        except (TypeError, ValueError):
            parts.append(f"{name} {value}")
    return ", ".join(parts)


def _render_arguments(arguments: dict[str, Any]) -> str:
    """Render a call's arguments compactly, keeping handles and dropping data."""
    parts = []
    for key, value in arguments.items():
        if value is None:
            continue
        text = str(value)
        if len(text) > _MAX_ARG_CHARS:
            continue
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _truncate_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Bound what a single call's arguments cost to keep."""
    bounded: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > _MAX_STORED_ARG_CHARS:
            bounded[key] = value[:_MAX_STORED_ARG_CHARS] + "...[truncated]"
        else:
            bounded[key] = value
    return bounded


def build_summary(
    tool_name: str,
    arguments: dict[str, Any],
    scalars: dict[str, Any],
    output_refs: list[str],
    success: bool,
    fallback: str = "",
) -> str:
    """Render one finding as a single session-log line.

    Args:
        tool_name: Tool that ran.
        arguments: What it was called with.
        scalars: Numeric results extracted from its metadata.
        output_refs: Artifacts it produced.
        success: Whether it succeeded.
        fallback: Text to use when there are no scalars, e.g. the head of the
            tool's own response.

    Returns:
        A line like ``vina_dock(lig=sdf_a1b2c3d4) -> -9.50 kcal/mol [pdb_c3]``.
    """
    line = f"{tool_name}({_render_arguments(arguments)})"

    if not success:
        return f"{line} -> failed: {fallback[:120]}" if fallback else f"{line} -> failed"

    if scalars:
        line += f" -> {_render_scalars(scalars)}"
    elif fallback:
        line += f" -> {fallback[:120]}"

    if output_refs:
        line += f" [{', '.join(output_refs)}]"
    return line


def extract_finding(
    tool_name: str,
    arguments: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    """Turn a tool execution into a ledger row.

    Args:
        tool_name: Name of the tool that ran.
        arguments: Arguments it was called with.
        result: The ToolResult it returned.

    Returns:
        Dict with ``arguments``, ``input_refs``, ``output_refs``, ``scalars``,
        ``summary``, and ``success``, ready for
        ``ConversationStore.record_finding``.
    """
    from api.tools.base import ContentType

    metadata = getattr(result, "metadata", None) or {}
    content_type = getattr(result, "content_type", None)
    success = content_type is not ContentType.ERROR

    input_refs = sorted(set(_find_refs(arguments)))

    try:
        rendered = result.to_tool_response()
    except Exception:
        rendered = str(getattr(result, "data", ""))

    output_refs: list[str] = []
    if content_type is ContentType.DATA_REFERENCE and isinstance(result.data, str):
        output_refs.append(result.data)
    # Skill endpoint tools declare several outputs at once.
    output_refs.extend(_find_refs(metadata.get("outputs")))
    # python_execute announces the artifacts it created in its own output and
    # nowhere else, and it is the main way refs come into existence, so a
    # ledger that ignored its text would miss most of the session's artifacts.
    if success:
        output_refs.extend(_find_refs(rendered))
    output_refs = sorted({ref for ref in output_refs if ref not in input_refs})

    scalars = _extract_scalars(metadata)

    fallback = "" if scalars else " ".join(rendered.split())

    return {
        "arguments": _truncate_arguments(arguments),
        "input_refs": input_refs,
        "output_refs": output_refs,
        "scalars": scalars,
        "summary": build_summary(
            tool_name, arguments, scalars, output_refs, success, fallback
        ),
        "success": success,
    }


# Re-exported so callers reading the ledger can render it without reaching
# across packages. The implementation lives beside the other prompt-envelope
# formatters, which the training agent imports from a venv with no DB driver.
format_session_log = _format_session_log


# -- Searching a ledger held in memory --------------------------------------
#
# Production keeps findings in PostgreSQL and searches them there. Training
# rollouts keep them on an in-process SessionState with no database behind it.
# Both have to answer the same question, or search_session_history would exist
# only in deployment and the trained policy would never learn to reach for it.

_findings_source_ctx: ContextVar[Optional[Callable[[], list[dict[str, Any]]]]] = (
    ContextVar("findings_source", default=None)
)


def set_findings_source(source: Optional[Callable[[], list[dict[str, Any]]]]) -> Any:
    """Provide an in-memory ledger for this request. Returns a reset token."""
    return _findings_source_ctx.set(source)


def reset_findings_source(token: Any) -> None:
    """Withdraw the in-memory ledger."""
    _findings_source_ctx.reset(token)


def get_findings_source() -> Optional[Callable[[], list[dict[str, Any]]]]:
    """Return this request's in-memory ledger accessor, if one was provided."""
    return _findings_source_ctx.get()


def search_findings_in_memory(
    findings: list[dict[str, Any]],
    query: Optional[str] = None,
    tool_name: Optional[str] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Filter a ledger by tool and substring, newest first.

    Substring only. The database side ranks by trigram similarity, which this
    cannot reproduce faithfully, and an approximation that ranks differently
    would be worse than a narrower match that ranks the same way every time.

    Args:
        findings: Ledger rows, oldest first.
        query: Free text matched against the summary and tool name.
        tool_name: Restrict to one tool.
        limit: Maximum rows to return.

    Returns:
        Matching findings, newest first.
    """
    rows = list(findings)
    if tool_name:
        rows = [row for row in rows if row.get("tool_name") == tool_name]

    if query:
        needle = query.lower()
        rows = [
            row
            for row in rows
            if needle in (row.get("summary") or "").lower()
            or needle in (row.get("tool_name") or "").lower()
        ]

    return list(reversed(rows))[:limit]
