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

"""Shared formatters for state injected into the LLM system prompt.

Used by both the production LangGraph state modifier
(``backend/api/graph/state_modifier.py``) and the training-side
``AevomSimpleAgent`` (``training/agents/aevom_simple_agent/app.py``)
so that the LLM sees identical state envelopes in both regimes.

The formatters operate on plain ``dict`` / ``list`` inputs so they can
be called from any process (training does not have access to the
LangGraph ``AevomState`` typed dict).
"""

from __future__ import annotations

from typing import Any, Optional

# Cap mirrors production: bounded prompt growth even on long episodes.
MAX_STORAGE_ARTIFACTS_IN_PROMPT = 20
MAX_SELECTIONS_IN_PROMPT = 5
MAX_DESCRIPTION_LEN = 50

# The session log is one line per tool call, so it is small by construction.
# The cap only guards a pathological episode.
MAX_FINDINGS_IN_PROMPT = 40

# The candidate block exists to stop the agent submitting a worse molecule than
# one it already found, which a short list does as well as a long one. Anything
# beyond this is query_candidates' job.
MAX_CANDIDATES_IN_PROMPT = 8
MAX_CANDIDATE_SMILES_LEN = 60

# Exclusions are ordered by how many candidates each accounts for, so a short
# list holds the ones actually costing enumerations.
MAX_EXCLUSIONS_IN_PROMPT = 6

# Marks where the injected envelope begins. The training agent strips from
# here before re-injecting, so an envelope that did not start with this would
# stack a stale copy on every turn. Exported rather than duplicated for that
# reason: whichever block comes first, both sides must agree on it.
INJECTION_SENTINEL = "\n\n[VIEWER STATE:"


def format_viewer_state(viewer_state: dict) -> str:
    structures = viewer_state.get("structures", [])
    if not structures:
        return "[VIEWER STATE: Empty - no structures loaded]"

    parts = []
    for struct in structures:
        label = struct.get("label", "unknown")
        stype = struct.get("structure_type", "unknown")
        chains = struct.get("chain_ids", [])
        residues = struct.get("residue_count", 0)
        atoms = struct.get("atom_count", 0)
        pdb_ref = struct.get("pdb_ref", "")

        flags = []
        if struct.get("has_polymer"):
            flags.append("polymer")
        if struct.get("has_ligand"):
            flags.append("ligand")
        if struct.get("has_water"):
            flags.append("water")

        ref_part = f" ref={pdb_ref}" if pdb_ref else ""
        part = (
            f"{label} ({stype}{ref_part}): {len(chains)} chains, "
            f"{residues} residues, {atoms} atoms"
        )
        if flags:
            part += f" [{', '.join(flags)}]"
        parts.append(part)

    selection_info = ""
    if viewer_state.get("has_selection"):
        sel_info = viewer_state.get("selection_info", "")
        selection_info = f" | Selection: {sel_info}"

    return f"[VIEWER STATE: {'; '.join(parts)}{selection_info}]"


def format_storage_state(
    storage_state: dict,
    *,
    max_artifacts: int = MAX_STORAGE_ARTIFACTS_IN_PROMPT,
    max_description_len: int = MAX_DESCRIPTION_LEN,
) -> str:
    artifacts = storage_state.get("artifacts", [])
    if not artifacts:
        return "[STORAGE STATE: Empty - no artifacts stored]"

    parts = []
    for artifact in artifacts[:max_artifacts]:
        ref = artifact.get("reference", "unknown")
        dtype = artifact.get("data_type", "unknown")
        desc = artifact.get("description") or ""
        if desc and len(desc) > max_description_len:
            desc = desc[: max_description_len - 3] + "..."
        parts.append(f"{ref} ({dtype}): {desc}")

    # Naming the tool matters. "and N more" on its own is a dead end: the
    # model can see that something is missing but has no way to reach it, so
    # it re-runs work it already did.
    suffix = ""
    if len(artifacts) > max_artifacts:
        hidden = len(artifacts) - max_artifacts
        suffix = (
            f"; ... and {hidden} more not shown "
            f"- use search_session_history to find them"
        )

    return f"[STORAGE STATE: {'; '.join(parts)}{suffix}]"


def format_session_log(
    findings: list[dict],
    *,
    max_entries: int = MAX_FINDINGS_IN_PROMPT,
) -> str:
    """Render the findings ledger as a ``[SESSION LOG]`` block.

    The one part of the envelope that is memory rather than current state.
    Everything else here describes what exists right now; this describes what
    was measured, and it is what makes re-docking a ligand to recover a number
    from twenty iterations ago unnecessary.

    Deterministic by construction -- no model call, so no latency, no cost, and
    no opportunity to paraphrase a number into a different one. The renderer
    lives in this module, rather than beside the ledger that produces the rows,
    because the training agent imports it from a venv that has no database
    driver and must not acquire one.

    Args:
        findings: Ledger rows, oldest first, each with ``turn_seq``,
            ``iteration``, and ``summary``.
        max_entries: Keep only the most recent N.

    Returns:
        The rendered block, or an empty string when nothing has happened yet.
    """
    if not findings:
        return ""

    kept = findings[-max_entries:] if max_entries else findings
    lines = [
        f"T{finding.get('turn_seq', 1)}.{finding.get('iteration', 0)} "
        f"{finding.get('summary', '')}".rstrip()
        for finding in kept
    ]

    header = "[SESSION LOG]"
    if max_entries and len(findings) > len(kept):
        header += (
            f" (showing the most recent {len(kept)} of {len(findings)} results"
            f" - use search_session_history for the rest)"
        )

    return "\n".join([header, *lines])


def format_active_structure(
    active_pdb_ref: Optional[str],
    storage_state: Optional[dict] = None,
) -> str:
    """Format an [ACTIVE STRUCTURE] block.

    The active reference is the most recently produced/loaded PDB. We
    surface it as informational context so the agent can pick the right
    ``pdb_ref`` for ``structure_query`` without having to scroll back
    through prior tool outputs.
    """
    if not active_pdb_ref:
        return "[ACTIVE STRUCTURE: None - no PDB has been produced or loaded yet]"

    label = ""
    if storage_state:
        for artifact in storage_state.get("artifacts", []):
            if artifact.get("reference") == active_pdb_ref:
                label = artifact.get("description") or ""
                if label and len(label) > MAX_DESCRIPTION_LEN:
                    label = label[: MAX_DESCRIPTION_LEN - 3] + "..."
                break

    if label:
        return f"[ACTIVE STRUCTURE: {active_pdb_ref} -- {label}]"
    return f"[ACTIVE STRUCTURE: {active_pdb_ref}]"


def format_selections(
    selections: list[dict],
    *,
    max_selections: int = MAX_SELECTIONS_IN_PROMPT,
) -> str:
    if not selections:
        return "[SELECTION STATE: No selections available]"

    selections = _current_selections(selections)
    if not selections:
        return "[SELECTION STATE: No selections available]"

    # Selections arrive newest first. The user's own picks go ahead of the
    # agent's probes: a pick is deliberate work, and an agent that ran five
    # queries since must not push it out of a five-entry window. The sort is
    # stable, so each group stays newest first. Roles sort first of all: a role
    # is the user saying what the design is, so a long session of probes must
    # not push it out of the window.
    selections = sorted(
        selections,
        key=lambda sel: (not sel.get("role"), sel.get("provenance") != "user"),
    )

    # Roles are shown in addition to the usual window, not out of it.
    shown = max_selections + _role_count(selections)
    parts = []
    for sel in selections[:shown]:
        ref = sel.get("reference", "unknown")
        atoms = sel.get("atom_count", 0)
        residues = sel.get("residue_count", 0)
        user_made = sel.get("provenance") == "user"
        name = sel.get("name")
        listing = sel.get("residue_listing") or ""
        role = sel.get("role")

        label = f"{name} = " if name else ""
        detail = f"{atoms} atoms, {residues} residues"
        if role:
            detail += f", design role: {_role_text(role, sel.get('role_value'))}"
        if user_made:
            detail += ", picked by the user"
        # The listing says *what* was selected. It is shown for the user's
        # picks and named selections, where the model has no other way to know;
        # for an agent's own MolQL probe it would only repeat the query.
        if listing and (user_made or name or role):
            detail += f": {listing}"
        parts.append(f"{label}{ref} ({detail})")

    suffix = ""
    if len(selections) > shown:
        suffix = f"; ... and {len(selections) - shown} more"

    roles_note = ""
    if any(sel.get("role") for sel in selections):
        roles_note = (
            " | The design roles above are the user's, assigned in the viewer: "
            "pass use_my_roles=true to create_design_spec or validate_design_spec "
            "to design with them, rather than re-deriving them"
        )

    return f"[SELECTION STATE: {'; '.join(parts)}{suffix}{roles_note}]"


def _role_key(sel: dict) -> tuple[str, str]:
    """What a role selection replaces: its role, and its parameter if it has one.

    A parameter that is a string (a Float group, an H-bond kind) distinguishes
    layers; a layout (a dict) is not an identity.
    """
    value = sel.get("role_value")
    return sel["role"], value.strip() if isinstance(value, str) else ""


def _current_selections(selections: list[dict]) -> list[dict]:
    """Drop role selections that were superseded or removed.

    ``selections`` is newest first, so the first one seen for a role is the one
    in force; an empty one removes the role, and is not itself worth showing.
    Selections without a role are untouched.
    """
    seen: set[tuple[str, str]] = set()
    kept = []
    for sel in selections:
        if not sel.get("role"):
            kept.append(sel)
            continue
        key = _role_key(sel)
        if key in seen:
            continue
        seen.add(key)
        if sel.get("atom_count", 0) > 0:
            kept.append(sel)
    return kept


def _role_count(selections: list[dict]) -> int:
    return sum(1 for sel in selections if sel.get("role"))


def _role_text(role: str, value: Any) -> str:
    return f"{role}={value}" if isinstance(value, str) and value else role


def format_candidates(
    candidates: list[dict],
    *,
    max_entries: int = MAX_CANDIDATES_IN_PROMPT,
) -> str:
    """Format a ``[CANDIDATES]`` block: the best molecules found so far.

    Ranked by verdict then fewest violations, because before any score exists
    that is the only ordering available, and the block has to be useful from the
    first molecule onward.

    Kept short deliberately. The block is a reminder that a better molecule
    already exists, not a replacement for ``query_candidates`` -- its job is to
    stop the agent submitting a worse molecule than one it already found, and a
    handful of rows does that.

    Args:
        candidates: Ledger rows with ``smiles``, ``passed``, ``failures``.
        max_entries: How many rows to show.
    """
    if not candidates:
        return ""

    ordered = sorted(
        candidates,
        key=lambda row: (
            0 if row.get("passed") else 1,
            len(row.get("failures") or []),
            row.get("smiles") or "",
        ),
    )
    kept = ordered[:max_entries] if max_entries else ordered

    lines = []
    for row in kept:
        smiles = row.get("smiles") or ""
        if len(smiles) > MAX_CANDIDATE_SMILES_LEN:
            smiles = smiles[: MAX_CANDIDATE_SMILES_LEN - 3] + "..."
        passed = row.get("passed")
        if passed is None:
            verdict = "ungated"
        elif passed:
            verdict = "passes"
        else:
            failures = row.get("failures") or []
            verdict = f"fails {len(failures)}" if failures else "fails"
        scores = row.get("scores") or {}
        detail = ""
        if isinstance(scores, dict) and scores:
            shown = list(scores.items())[:2]
            detail = " | " + ", ".join(
                f"{name}={value:.2f}"
                if isinstance(value, (int, float)) and not isinstance(value, bool)
                else f"{name}={value}"
                for name, value in shown
            )
        lines.append(f"{smiles} [{verdict}]{detail}")

    passing = sum(1 for row in candidates if row.get("passed"))
    header = (
        f"[CANDIDATES: {len(candidates)} considered, {passing} passing. "
        "Call best_candidate before you commit to an answer]"
    )
    if max_entries and len(ordered) > len(kept):
        header = (
            f"[CANDIDATES: {len(candidates)} considered, {passing} passing, "
            f"showing {len(kept)}. Call query_candidates for the rest and "
            "best_candidate before you commit to an answer]"
        )

    return "\n".join([header, *lines])


def format_exclusions(
    exclusions: list[dict],
    *,
    max_entries: int = MAX_EXCLUSIONS_IN_PROMPT,
) -> str:
    """Format an ``[EXCLUSIONS]`` block: what has been disqualifying candidates.

    Phrased as instructions rather than statistics, and placed in the envelope
    rather than left in the transcript, because the point is to change the *next*
    enumeration. A rule the agent has to scroll back to find is a rule it will
    rediscover the hard way.

    Args:
        exclusions: Derived rules, most prolific first, each with ``describe``
            already rendered into ``description``.
        max_entries: How many rules to show.
    """
    if not exclusions:
        return ""

    kept = exclusions[:max_entries] if max_entries else exclusions
    lines = [
        rule.get("description") or rule.get("feature") or ""
        for rule in kept
        if rule.get("description") or rule.get("feature")
    ]
    if not lines:
        return ""

    header = (
        "[EXCLUSIONS: derived from candidates that already failed. "
        "Screen your next enumeration against these before scoring it]"
    )
    return "\n".join([header, *lines])


def format_tool_budgets(tool_budgets: list[dict]) -> str:
    """Format a ``[TOOL BUDGET]`` block from budget states.

    Spend is only useful to an agent while it still has some, so the block leads
    with what is left rather than what was used, and names exhausted tools
    explicitly instead of showing a zero the agent has to interpret.

    Args:
        tool_budgets: Dicts with ``tool_name``, ``limit`` and ``used``, most
            depleted first.
    """
    if not tool_budgets:
        return ""

    parts = []
    batched = []
    for entry in tool_budgets:
        tool = entry.get("tool_name", "unknown")
        limit = int(entry.get("limit", 0))
        used = int(entry.get("used", 0))
        remaining = max(0, limit - used)
        if remaining == 0:
            parts.append(f"{tool} exhausted ({used}/{limit})")
        else:
            parts.append(f"{tool} {remaining} of {limit} left")
        if entry.get("batch_argument") and remaining:
            batched.append((tool, entry["batch_argument"]))

    advice = "spend the remaining calls on your best candidates; an exhausted tool will refuse"
    if batched:
        # The unit is the call, not the molecule, and knowing that is worth more
        # than the count: one admet_predict scoring thirty candidates costs the
        # same one call as scoring one. Agents reliably miss this and spend a
        # fifteen-call budget on fifteen molecules.
        how = "; ".join(f"{tool} takes a list in `{argument}`" for tool, argument in batched)
        advice += (
            f". A call is the unit, not an item: {how}, so one call over every "
            "candidate costs the same as one call over a single candidate -- "
            "batch rather than looping"
        )

    return f"[TOOL BUDGET: {'; '.join(parts)} -- {advice}]"


def format_nim_notices(nim_notices: list[dict]) -> str:
    """Format a ``[NIM STATE]`` line for NIMs registered but not yet confirmed.

    An unconfirmed NIM's tools are withheld from the model, so without this the
    model cannot tell "no such tool exists" from "the user has not switched it
    on", and falls back to a different tool without saying why.

    Args:
        nim_notices: Dicts with ``name`` and ``tools`` (the tool names the NIM
            would contribute once confirmed).
    """
    if not nim_notices:
        return ""
    parts = []
    for notice in nim_notices:
        name = notice.get("name") or "NIM"
        tools = [str(t) for t in notice.get("tools") or []]
        offered = f" (would add {', '.join(tools)})" if tools else ""
        parts.append(f"{name} NIM registered but not confirmed{offered}")
    return (
        f"[NIM STATE: {'; '.join(parts)}. Its tools are unavailable to you until "
        "the user confirms it (Tool Manager, \"Review & enable\"); ask the user to "
        "confirm it rather than substituting another tool without saying so]"
    )


def inject_state_into_prompt(
    base_prompt: str,
    viewer_state: dict,
    storage_state: dict,
    selections: list[dict],
    active_pdb_ref: Optional[str] = None,
    findings: Optional[list[dict]] = None,
    tool_budgets: Optional[list[dict]] = None,
    candidates: Optional[list[dict]] = None,
    exclusions: Optional[list[dict]] = None,
    nim_notices: Optional[list[dict]] = None,
) -> str:
    """Build the augmented system prompt with dynamic state sections.

    Output is backwards-compatible with the previous production
    layout but adds an optional ``[ACTIVE STRUCTURE]`` block when
    ``active_pdb_ref`` is supplied (training only -- production
    surfaces this implicitly through ``[VIEWER STATE]``).

    The envelope must begin with ``INJECTION_SENTINEL``. The training agent
    strips from that marker before re-injecting, so a block placed ahead of it
    would survive stripping and accumulate a stale copy every turn.

    Args:
        base_prompt: The prompt to append the envelope to.
        viewer_state: What is loaded in the viewer.
        storage_state: What artifacts exist.
        selections: Available structure selections.
        active_pdb_ref: Most recently produced PDB, training only.
        findings: Ledger rows for the ``[SESSION LOG]`` block. Everything else
            here is current state; this is the only part that is memory.
        tool_budgets: Remaining call budgets, omitted entirely when no budget is
            configured -- a deployment without budgets should not carry a block
            saying so.
        candidates: Candidate ledger rows. Like the session log this is memory
            rather than current state, and for the same reason it belongs in the
            envelope: it holds comparisons the trajectory cannot be trusted to
            still contain.
        exclusions: Rules derived from candidates that already failed.
        nim_notices: NIMs registered for the session but not yet confirmed, so
            their tools are withheld. Omitted when there are none.
    """
    state_parts: list[str] = []

    if viewer_state and not viewer_state.get("is_empty", True):
        state_parts.append(format_viewer_state(viewer_state))
    else:
        state_parts.append("[VIEWER STATE: Empty - no structures loaded]")

    if storage_state and not storage_state.get("is_empty", True):
        state_parts.append(format_storage_state(storage_state))
    else:
        state_parts.append("[STORAGE STATE: Empty - no artifacts stored]")

    if findings:
        state_parts.append(format_session_log(findings))

    if candidates:
        state_parts.append(format_candidates(candidates))

    if exclusions:
        state_parts.append(format_exclusions(exclusions))

    if active_pdb_ref is not None:
        state_parts.append(format_active_structure(active_pdb_ref, storage_state))

    if selections:
        state_parts.append(format_selections(selections))
    else:
        state_parts.append("[SELECTION STATE: No selections available]")

    if tool_budgets:
        state_parts.append(format_tool_budgets(tool_budgets))

    if nim_notices:
        state_parts.append(format_nim_notices(nim_notices))

    state_section = "\n".join(state_parts)
    return f"{base_prompt}\n\n{state_section}"


__all__ = [
    "INJECTION_SENTINEL",
    "MAX_CANDIDATES_IN_PROMPT",
    "MAX_CANDIDATE_SMILES_LEN",
    "MAX_DESCRIPTION_LEN",
    "MAX_EXCLUSIONS_IN_PROMPT",
    "MAX_FINDINGS_IN_PROMPT",
    "MAX_SELECTIONS_IN_PROMPT",
    "MAX_STORAGE_ARTIFACTS_IN_PROMPT",
    "format_active_structure",
    "format_nim_notices",
    "format_selections",
    "format_session_log",
    "format_candidates",
    "format_exclusions",
    "format_storage_state",
    "format_tool_budgets",
    "format_viewer_state",
    "inject_state_into_prompt",
]
