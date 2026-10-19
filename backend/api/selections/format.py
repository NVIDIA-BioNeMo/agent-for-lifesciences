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

"""Shared selection format strings for training/production parity.

Both the LangGraph agent (production) and the NeMo-Gym resource server
(training) must return byte-identical tool response strings so the LLM
sees a consistent API regardless of execution path.

No LangGraph, Redis, or other framework dependencies -- pure functions only.
"""


def format_selection_success(
    atom_count: int,
    residue_count: int,
    reference: str,
) -> str:
    """Format a successful structure_query result."""
    return (
        f"Selection active: {atom_count} atoms in "
        f"{residue_count} residues. "
        f"Reference: {reference}. "
        f'Pass selection_ref="{reference}" to python_execute '
        f"for geometric/chemical analysis."
    )


# A selection whose query could not be run at all (parse or compile failure) is
# reported by every executor -- the browser viewer, the eval bridge and the
# training wrapper -- as a zero-atom selection whose summary starts with this
# prefix. It must be told apart from a query that ran and legitimately matched
# nothing: the former is a syntax problem the model can fix, the latter is
# information about the structure.
MOLQL_ERROR_PREFIX = "MolQL error:"

# Prefix the headless service puts on the error it returns for a failed query.
_HEADLESS_ERROR_PREFIX = "MolQL query failed:"


def molql_error_summary(detail: str) -> str:
    """Build the selection summary that marks a query as failed to execute."""
    detail = (detail or "").strip()
    if detail.startswith(_HEADLESS_ERROR_PREFIX):
        detail = detail[len(_HEADLESS_ERROR_PREFIX):].strip()
    return f"{MOLQL_ERROR_PREFIX} {detail or 'the expression could not be executed'}"


def is_molql_error_summary(summary: str | None) -> bool:
    """True when a selection summary reports a query that failed to execute."""
    return bool(summary) and summary.startswith(MOLQL_ERROR_PREFIX)


def format_selection_query_error(summary: str) -> str:
    """Format a structure_query whose MolQL expression failed to execute.

    Without this the model only ever sees "matched 0 atoms", which reads as a
    statement about the structure, and it goes on to guess at the cause.
    """
    detail = summary[len(MOLQL_ERROR_PREFIX):].strip() if summary else ""
    return (
        f"Error: MolQL query failed to execute: {detail}\n\n"
        "This is a syntax error in the expression, not an empty result. "
        "MolQL is S-expressions only: there is no `in` function and no [a, b] "
        "list syntax. To match discrete residue numbers use "
        "(set.has (set 29 30 31) atom.resno), e.g. "
        "(sel.atom.res (and (= atom.chain A) (set.has (set 29 30 31) atom.resno))); "
        "use (in-range atom.resno MIN MAX) only for a continuous range."
    )


def format_selection_empty() -> str:
    """Format a structure_query that matched zero atoms (no viewer context)."""
    return "Error: Selection query matched 0 atoms."


def format_selection_empty_with_hints(
    structures: list[dict],
) -> str:
    """Format a zero-atom result with suggestions based on loaded structures."""
    msg = "Error: Selection query matched 0 atoms."

    if not structures:
        msg += (
            " Verify the query targets entities, residues, "
            "or chains that exist in the current structure "
            "and try a different query."
        )
        return msg

    msg += "\n\nLoaded structures:"
    all_chains: list[str] = []
    any_ligand = False

    for s in structures:
        label = s.get("label", "unnamed")
        stype = s.get("structure_type", "")
        chains = s.get("chain_ids", [])
        has_lig = s.get("has_ligand", False)
        has_poly = s.get("has_polymer", False)
        contents = []
        if has_poly:
            contents.append("polymer")
        if has_lig:
            contents.append("ligand/small molecule")
            any_ligand = True
        chain_str = ", ".join(chains) if chains else "?"
        msg += f"\n  - '{label}' ({stype}): chains [{chain_str}]"
        if contents:
            msg += f", contains {', '.join(contents)}"
        all_chains.extend(chains)

    msg += "\n\nSuggestions:"
    if any_ligand:
        msg += (
            "\n  - For ligands: "
            "(sel.atom.res (= atom.entity-type non-polymer))"
        )
    if all_chains:
        first = all_chains[0]
        msg += (
            f"\n  - For a chain: "
            f"(sel.atom.chains (= atom.chain {first}))"
        )
    msg += (
        "\n  - Check get_viewer_state for available chains "
        "and entity types."
    )
    return msg


def format_selection_timeout() -> str:
    """Format a timed-out structure_query.

    A timeout only happens with no viewer attached, which is not the production
    or eval configuration (both answer selections through Molstar). An earlier
    revision inlined a full python fallback here; measurement with a viewer
    attached showed that path never fires in a realistic stack, so it was
    trimmed back. The one useful nudge that survives -- don't keep retrying a
    viewer that isn't answering -- is kept.
    """
    return (
        "Error: Selection query timed out waiting for the viewer. If a viewer is "
        "attached, confirm the structure is loaded (call visualize_pdb first) and "
        "retry at most once; do not keep re-issuing the query."
    )


def format_selection_no_data() -> str:
    """Format an acknowledged but data-less structure_query."""
    return "Selection query acknowledged but no data returned."
