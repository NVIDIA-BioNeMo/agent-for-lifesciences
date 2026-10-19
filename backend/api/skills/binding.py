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

"""Bind a skill's prose to tools Aevom already ships.

``SkillContract`` solves one half of the "prose is not callable" problem: given a
skill that documents an HTTP API, it synthesizes a tool whose parameters are
storage references. That is the right answer for a skill a user installs at
runtime, where Aevom has no prior knowledge of the endpoint.

It is the wrong answer for the NIMs. Those endpoints are first-class -- Aevom was
designed around them -- so each one gets a hand-written builtin tool that knows
the payload quirks, stores its outputs under the right data types, and is covered
by tests. What is missing is not a tool; it is the *announcement*. A ``SKILL.md``
fetched from the bionemo-agent-toolkit says to build the request yourself with
``requests`` and write the result to a file, and a model acts on the first
applicable instruction it reads.

A :class:`SkillBinding` supplies that announcement without a contract, which
matters for two reasons beyond avoiding a duplicate tool:

- ``synthesize_skill_tools`` returns nothing without a contract, so the model is
  never offered two ways to reach the same endpoint.
- Confirming a contract is also how a user authorizes outbound requests carrying
  their API key to the contract's hosts. A builtin tool already has its own
  vetted path to the NIM, so a binding must not re-open that consent gate.

Both bindings and contracts render through :class:`ToolAnnouncement`, so the
three disclosure levels in ``annotate`` stay a single implementation.

**The table below is an allowlist, not a hint.** ``SkillRegistry`` loads exactly
the skills named here. The submodule carries roughly ten NIM skill directories;
scanning that directory instead would register every one of them the moment the
submodule pin moved, including skills with no tool behind them and endpoints
known to be down.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from api.storage.datatypes import friendly_name

from .contract import SkillContract

# Where the pinned bionemo-agent-toolkit checkout lands, in each layout Aevom
# runs in. Mirrors the resolution order in ``registry.py``: the image flattens
# the submodule to /app/nim-skills, while a source checkout keeps it under
# third_party/.
_NIM_SKILLS_ENV = "AEVOM_NIM_SKILLS_DIR"
_IMAGE_NIM_SKILLS = Path(__file__).resolve().parents[2] / "nim-skills"
_REPO_NIM_SKILLS = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "bionemo-agent-toolkit"
    / "nim-skills"
)

# Library skills document the packages installed in the python_execute sandbox.
# The prose is first-party today and lives in backend/library-skills, but it is
# loaded through the same root-plus-allowlist path as the NIM skills rather than
# by scanning backend/skills. That is deliberate: when this prose is replaced by
# an upstream submodule, the swap is a new candidate path in
# ``library_skills_root`` and nothing else. The bundle layout below
# (``SKILL.md`` plus ``references/``) already matches upstream's library-skills
# tree for the same reason.
# One candidate covers both layouts: parents[2] is ``backend/`` in a checkout and
# ``/app`` in the image, and the Dockerfile copies the directory to match.
_LIBRARY_SKILLS_ENV = "AEVOM_LIBRARY_SKILLS_DIR"
_LIBRARY_SKILLS = Path(__file__).resolve().parents[2] / "library-skills"

# The default ``replaces`` for skills whose prose describes raw HTTP calls.
HAND_BUILT_HTTP = "building the HTTP request yourself"


def _resolve_root(env_var: str, candidates: tuple[Path, ...]) -> Optional[Path]:
    """First directory that exists, honouring an environment override.

    Returns None rather than raising when nothing is found: a developer who
    cloned without ``--recursive`` should get a server that starts with fewer
    skills, not one that refuses to boot. The prerequisites check is what tells
    them to run ``make submodules``.
    """
    override = os.environ.get(env_var)
    if override:
        path = Path(override)
        return path if path.is_dir() else None
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def nim_skills_root() -> Optional[Path]:
    """Directory holding the upstream NIM skill bundles, or None if absent."""
    return _resolve_root(_NIM_SKILLS_ENV, (_IMAGE_NIM_SKILLS, _REPO_NIM_SKILLS))


def library_skills_root() -> Optional[Path]:
    """Directory holding the sandbox library skill bundles, or None if absent."""
    return _resolve_root(_LIBRARY_SKILLS_ENV, (_LIBRARY_SKILLS,))


@dataclass(frozen=True)
class BoundTool:
    """A builtin tool that supersedes part of a skill's prose."""

    name: str
    summary: str
    # What the model passes in and gets back, in reference terms. Written here
    # rather than derived from the tool schema because the point is to contrast
    # with the raw request/response the prose describes.
    takes: str = ""
    returns: str = ""


@dataclass(frozen=True)
class SkillBinding:
    """Maps one upstream skill directory onto Aevom's builtin tools."""

    skill_id: str
    tools: tuple[BoundTool, ...]
    # Prose sections the builtin tool makes irrelevant. Named explicitly because
    # these skills document a local Docker deployment and a filesystem, neither
    # of which exists in Aevom, and a model told to `docker run` will try.
    supersedes: tuple[str, ...] = field(default_factory=tuple)
    # Corrections and Aevom-specific guidance that the prose gets wrong or omits.
    notes: tuple[str, ...] = field(default_factory=tuple)
    # Parts of the skill Aevom has no builtin tool for. Stated so the model does
    # not reach for a bound tool that cannot serve the request.
    not_covered: tuple[str, ...] = field(default_factory=tuple)
    # True when every bound tool exchanges storage references, which is the case
    # for all the NIM tools and lets the trailer forbid pasting raw data. Library
    # skills set it False: `molecule_gates` takes SMILES literals, so telling the
    # model never to paste a SMILES would make the tool uncallable.
    reference_only: bool = True
    # What the model would otherwise do, named in the announcement so the
    # instruction lands against the temptation the prose creates. For a NIM skill
    # that is a hand-built HTTP call; for a library skill it is hand-written
    # sandbox code. ``annotate`` names `http_request` after the HTTP default only
    # when that tool is enabled.
    replaces: str = HAND_BUILT_HTTP

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)


@dataclass(frozen=True)
class ToolAnnouncement:
    """What ``annotate`` needs to tell the model a skill has tools.

    Built from either a :class:`SkillBinding` (builtin tools) or a
    :class:`SkillContract` (synthesized tools), so the three disclosure levels
    do not need to know which kind of skill they are rendering.
    """

    # Level 1 uses the short label; level 3 uses the callable name. They differ
    # for contracts, where the catalog shows the endpoint suffix but the tool is
    # named ``{skill_id}_{suffix}``.
    short_names: tuple[str, ...]
    qualified_names: tuple[str, ...]
    body_lines: tuple[str, ...]
    trailer: str
    # Completes "Call these tools instead of ..." in the level 2 banner.
    replaces: str = HAND_BUILT_HTTP

    @property
    def is_empty(self) -> bool:
        return not self.short_names


def _binding_body_lines(binding: SkillBinding) -> list[str]:
    lines: list[str] = []
    for tool in binding.tools:
        lines.append(f"* `{tool.name}` — {tool.summary}")
        if tool.takes:
            lines.append(f"    Takes: {tool.takes}")
        if tool.returns:
            lines.append(f"    Returns: {tool.returns}")
    if binding.not_covered:
        lines.append("")
        lines.append("Not available as a tool in Aevom:")
        lines.extend(f"    - {item}" for item in binding.not_covered)
    if binding.notes:
        lines.append("")
        lines.append("Aevom-specific notes that override the instructions below:")
        lines.extend(f"    - {note}" for note in binding.notes)
    return lines


def _binding_trailer(binding: SkillBinding) -> str:
    parts: list[str] = []
    if binding.reference_only:
        parts.append(
            "These tools take and return storage references, so never paste raw "
            "sequences, SMILES, or structure text into their arguments, and do "
            "not expect raw data back."
        )
    else:
        parts.append(
            "Check each tool's schema for whether an argument is a storage "
            "reference or a literal value; they differ between these tools."
        )
    if binding.supersedes:
        parts.append(
            "Ignore the following entirely — they describe an environment Aevom "
            "does not have: " + "; ".join(binding.supersedes) + "."
        )
    parts.append(
        "Read the instructions below to understand what the model does, which "
        "settings matter, and how to interpret the results. Ignore any step that "
        "tells you to construct a request body, add an auth header, run a "
        "container, or read and write files. The tool does all of that."
    )
    return " ".join(parts)


def announcement_from_binding(
    binding: Optional[SkillBinding],
) -> Optional[ToolAnnouncement]:
    """Render a binding into an announcement, or None when there is nothing to say."""
    if binding is None or not binding.tools:
        return None
    names = binding.tool_names
    return ToolAnnouncement(
        short_names=names,
        qualified_names=names,
        body_lines=tuple(_binding_body_lines(binding)),
        trailer=_binding_trailer(binding),
        replaces=binding.replaces,
    )


def _contract_body_lines(skill_id: str, contract: SkillContract) -> list[str]:
    lines: list[str] = []
    for endpoint in contract.endpoints:
        lines.append(f"* `{endpoint.tool_name(skill_id)}` — {endpoint.summary}")
        if endpoint.inputs:
            params = ", ".join(
                f"`{p.ref_param_name}` ({friendly_name(p.data_type)})"
                for p in endpoint.inputs
            )
            lines.append(f"    Takes: {params}.")
        if endpoint.scalar_params:
            scalars = ", ".join(f"`{s.name}`" for s in endpoint.scalar_params)
            lines.append(f"    Optional settings: {scalars}.")
        if endpoint.outputs:
            lines.append("    Returns references instead of raw data:")
            for out in endpoint.outputs:
                # The prose below tells the model to read this response field;
                # say explicitly what it gets instead, or it will go looking.
                lines.append(
                    f"      - the response field `{out.json_path}` described below "
                    f"comes back as a stored {friendly_name(out.data_type)} reference "
                    f"named `{out.name}`"
                )
        if endpoint.capture_unmapped:
            lines.append(
                "    Other numeric response fields are stored as `array` references; "
                "load them with `python_execute` via `aevom.load('<ref>')`."
            )
    return lines


_CONTRACT_TRAILER = (
    "These tools take and return storage references, so never paste raw "
    "sequences, SMILES, or structure text into their arguments, and do not "
    "expect raw data back. The instructions below describe the underlying "
    "HTTP API for context — read them to understand what the endpoint does "
    "and which settings matter, but ignore any step that tells you to "
    "construct the request body, add auth headers, or parse raw response "
    "fields. The tool does all of that."
)


def announcement_from_contract(
    skill_id: str,
    contract: Optional[SkillContract],
) -> Optional[ToolAnnouncement]:
    """Render a confirmed contract into an announcement.

    Returns None for an unconfirmed or empty contract: nothing was synthesized,
    so advertising a tool would send the model after a name that does not
    resolve.
    """
    if contract is None or not contract.is_actionable:
        return None
    return ToolAnnouncement(
        short_names=tuple(e.tool_suffix for e in contract.endpoints),
        qualified_names=tuple(e.tool_name(skill_id) for e in contract.endpoints),
        body_lines=tuple(_contract_body_lines(skill_id, contract)),
        trailer=_CONTRACT_TRAILER,
    )


def resolve_announcement(
    skill_id: str,
    binding: Optional[SkillBinding],
    contract: Optional[SkillContract],
) -> Optional[ToolAnnouncement]:
    """Announcement for a skill that may carry either a binding or a contract.

    Binding wins when both are somehow present, because a binding means a
    hand-written builtin tool exists and that is the vetted path. In practice a
    skill has at most one, so the precedence is only a guard.
    """
    return announcement_from_binding(binding) or announcement_from_contract(
        skill_id, contract
    )


# --- The allowlist -----------------------------------------------------------

# Shared across every NIM skill: the prose is written for an agent with a shell
# and a filesystem, and all of it targets the same two hosted/local endpoints
# that the builtin tool already calls.
_NIM_SUPERSEDES = (
    'the "Choose Mode" section and its hosted-vs-local question',
    'the "Local Docker" section, including every `docker run`, `docker login`, '
    "readiness poll and cache mount",
    "the Python `requests` request patterns",
    "every instruction to write results to a file with `open()` or `Path.write_text()`",
)

NIM_SKILL_BINDINGS: tuple[SkillBinding, ...] = (
    SkillBinding(
        skill_id="rfdiffusion-nim",
        tools=(
            BoundTool(
                name="rfdiffusion_design_backbone",
                summary=(
                    "Generate a protein backbone with RFDiffusion — de novo, "
                    "motif scaffolding, or binder design."
                ),
                takes=(
                    "`contigs` (the DSL documented below), optional `pdb_ref` for "
                    "the input structure, optional `hotspot_res` and "
                    "`diffusion_steps`."
                ),
                returns=(
                    "a stored `pdb` reference for the designed backbone. The "
                    "`output_pdb` response field the prose tells you to save is "
                    "that reference; pass it to `visualize_pdb` to show it, or to "
                    "`proteinmpnn_design_sequence` to design a sequence for it."
                ),
            ),
        ),
        supersedes=_NIM_SUPERSEDES,
        notes=(
            "The contigs DSL section below is authoritative and worth reading — "
            "`contigs` is the one argument you must get right.",
            "Omit `pdb_ref` for de novo design. The endpoint rejects a request "
            "with no input structure, and the tool supplies the dummy PDB the "
            "prose describes, so you never need to construct it yourself.",
            "A generated backbone has no sequence — every residue comes back as "
            "glycine. That is expected, not a failure; the next step is "
            "`proteinmpnn_design_sequence`.",
        ),
    ),
    SkillBinding(
        skill_id="proteinmpnn-nim",
        tools=(
            BoundTool(
                name="proteinmpnn_design_sequence",
                summary=(
                    "Design amino acid sequences for a backbone structure with "
                    "ProteinMPNN (inverse folding)."
                ),
                takes=(
                    "`pdb_ref` for the backbone, plus optional `num_seq_per_target`, "
                    "`sampling_temp`, `chains`, `omit_aas`, `ca_only`, "
                    "`use_soluble_model`."
                ),
                returns=(
                    "a stored `fasta` reference for the `mfasta` response field, an "
                    "`array` reference for the per-position `probs` matrix, and the "
                    "per-sequence scores inline in the result message."
                ),
            ),
        ),
        supersedes=_NIM_SUPERSEDES,
        notes=(
            "Ignore the score-pairing snippet below. It filters headers "
            'containing "native" or "wt", but the input row is actually labelled '
            "`>input`, so that filter keeps it and misaligns every score against "
            "the wrong sequence. The tool pairs scores with designed sequences "
            "correctly and reports them in its result.",
            "`probs` is a per-position probability matrix that runs to tens of "
            "thousands of numbers, so it is stored rather than returned. Analyze "
            "it with `python_execute` using `aevom.load_array('<ref>')`.",
            "Lower scores are better — they are negative log-likelihoods per "
            "residue, not confidences.",
        ),
    ),
    SkillBinding(
        skill_id="genmol-nim",
        tools=(
            BoundTool(
                name="genmol_generate_molecules",
                summary=(
                    "Generate drug-like molecules with GenMol — de novo, "
                    "scaffold decoration, motif extension or lead optimization."
                ),
                takes=(
                    "an optional `safe_ref` for the masked scaffold, plus "
                    "`num_molecules`, `temperature`, `noise`, `step_size`, "
                    "`scoring` and `unique`."
                ),
                returns=(
                    "a stored `csv` reference holding every generated molecule "
                    "with its score, ranked. The `molecules` response field the "
                    "prose tells you to sort and write to a `.smi` file is that "
                    "reference; read it with `python_execute` via "
                    "`aevom.load_text('<ref>')`. The top few molecules also come "
                    "back as individual `smiles` references, ready for "
                    "`molecule_info`, `generate_3d_coordinates` or `vina_dock`."
                ),
            ),
        ),
        supersedes=_NIM_SUPERSEDES + (
            'the "Save And Report Output" section, which sorts the molecules and '
            "writes them to a `.smi` file",
            "the `safe-mol` / `sf.encode` / `scaffold_to_safe` snippets — "
            "`smiles_to_safe` is a tool and does that conversion for you",
        ),
        notes=(
            "The request field is named `smiles` but must carry SAFE, and the "
            "tool enforces that by taking a `safe_ref` rather than a molecule. "
            "Produce one with `smiles_to_safe`, or omit it entirely for de novo "
            "and set `mask_min` / `mask_max` instead.",
            "The string-typed `temperature` and `noise` quirk is handled — pass "
            "them as ordinary numbers.",
            "A generation mask replaces a fragment, it does not sit beside a "
            "whole molecule. A SAFE string with no `[*{min-max}]` is rejected "
            "rather than sent, because GenMol answers that by returning your "
            "input molecule unchanged, which looks like success.",
            "Getting fewer molecules than requested is expected — invalid and "
            "duplicate structures are filtered after generation.",
            "QED and LogP rank candidates for triage. Neither is a binding "
            "affinity, and for LogP higher is not better.",
        ),
    ),
    SkillBinding(
        skill_id="boltz2-nim",
        tools=(
            BoundTool(
                name="boltz2",
                summary=(
                    "Predict a biomolecular complex structure with Boltz-2, and "
                    "optionally the binding affinity of one small molecule in the "
                    "same call."
                ),
                takes=(
                    "`polymers` (each with `molecule_type` and a `sequence` or "
                    "`sequence_ref`), optional `ligands` with `smiles` or `ccd`, "
                    "optional `constraints`, `msa_ref` from `msa_search`, and the "
                    "sampling settings documented below. A single protein can be "
                    "given as just `sequence_ref`."
                ),
                returns=(
                    "a stored `mmcif` reference for the highest-confidence "
                    "structure, further `mmcif` references for the other "
                    "diffusion samples, and the affinity numbers inline. The "
                    "`structures[i].structure` response field the prose tells you "
                    "to write to a `.cif` file is that reference."
                ),
            ),
            BoundTool(
                name="convert_structure",
                summary=(
                    "Convert a stored structure between mmCIF and PDB, for "
                    "handing a Boltz-2 result to a PDB-only tool."
                ),
                takes="`structure_ref` and `target_format` (`pdb` or `mmcif`).",
                returns="a stored reference of the requested format.",
            ),
        ),
        supersedes=_NIM_SUPERSEDES + (
            'the "Save And Report Output" section, which writes each structure to '
            "a `.cif` file with `open()`",
            "the suggestion to visualize in PyMOL, ChimeraX or UCSF Chimera — "
            "`visualize_pdb` renders the stored structure in Aevom's own viewer",
        ),
        notes=(
            "Boltz-2 emits mmCIF and nothing else, so the structure comes back as "
            "an `mmcif` reference. `visualize_pdb` and `structure_query` accept it "
            "directly; only call `convert_structure` when you need a `pdb` "
            "reference for `vina_dock`, `proteinmpnn_design_sequence` or "
            "`aevom.load_coords`.",
            "`output_format` is not a parameter. mmCIF is the only value the "
            "endpoint accepts, and the tool sets it.",
            "Set `predict_affinity` on exactly one ligand. More than one is "
            "rejected before the request is sent, so to rank several ligands make "
            "one call per ligand and compare the pIC50 values.",
            "pIC50 and the binding probability are model estimates for ranking "
            "and triage, not measured IC50 values. Do not report them as "
            "experimental affinities; confirm a promising hit with `vina_dock` or "
            "an orthogonal method.",
            "Pass `msa_ref` from `msa_search` rather than an alignment. The tool "
            "builds the nested `msa.msa_search.a3m` payload, including the "
            "`alignment`-not-`data` field name the prose warns about.",
            "Prefer `boltz2` over `openfold3` when you need affinity as well as a "
            "structure, or when the ligand is a novel molecule given as SMILES; "
            "`openfold3` covers structure-only prediction with CCD ligands. "
            "Prefer `vina_dock` when you already have an experimental receptor "
            "structure and want a physics-based binding energy in kcal/mol.",
            "A full PAE matrix (`write_full_pae`) is thousands of numbers, so it "
            "is stored as an `array` reference. Read it with `python_execute` via "
            "`aevom.load_array('<ref>')`.",
        ),
        not_covered=(
            "Local Docker deployment — Aevom calls the hosted endpoint only.",
        ),
    ),
    SkillBinding(
        skill_id="msa-search-nim",
        tools=(
            BoundTool(
                name="msa_search",
                summary=(
                    "Search sequence databases to build a multiple sequence "
                    "alignment for a single protein."
                ),
                takes="`sequence_ref` from `fetch_protein_sequence`.",
                returns=(
                    "a stored `msa` reference. The nested "
                    "`alignments[database][format].alignment` response field the "
                    "prose tells you to save is that reference; pass it to "
                    "`openfold3` as `msa_ref`."
                ),
            ),
        ),
        supersedes=_NIM_SUPERSEDES + (
            'the "Faster Startup" and "Parallel Download" sections, which are '
            "entirely about provisioning a local database volume",
        ),
        notes=(
            "The tool selects the databases, e-value and output format, so the "
            "database names and payload fields below are background only.",
        ),
        not_covered=(
            "Paired MSA search for complexes (the `/paired/predict` endpoint) — "
            "`msa_search` handles one sequence at a time.",
            "Structural template search, which upstream reports is local-only "
            "and returns 404 on the hosted endpoint.",
        ),
    ),
)

_BINDINGS_BY_ID = {binding.skill_id: binding for binding in NIM_SKILL_BINDINGS}


def get_binding(skill_id: str) -> Optional[SkillBinding]:
    """Look up a binding by skill id."""
    return _BINDINGS_BY_ID.get(skill_id)


def bound_skill_ids() -> tuple[str, ...]:
    """Ids of every skill the registry is allowed to load from the submodule."""
    return tuple(_BINDINGS_BY_ID)


# --- Sandbox library skills --------------------------------------------------


@dataclass(frozen=True)
class LibrarySkillEntry:
    """One library skill the registry may load from ``library_skills_root()``.

    Carries an *optional* binding, which is the difference from the NIM table:
    OpenBabel and biotite have no builtin tool behind them, so their skills are
    pure prose about writing sandbox code. Where a tool does exist, the binding
    keeps the model from hand-rolling in Python what the tool already does.
    """

    skill_id: str
    binding: Optional[SkillBinding] = None


LIBRARY_SKILLS: tuple[LibrarySkillEntry, ...] = (
    LibrarySkillEntry(
        skill_id="rdkit",
        binding=SkillBinding(
            skill_id="rdkit",
            tools=(
                BoundTool(
                    name="molecule_gates",
                    summary=(
                        "Score molecules against the full drug-likeness panel "
                        "and get a per-gate pass/fail verdict."
                    ),
                    takes="a list of SMILES, and optional threshold overrides.",
                    returns=(
                        "the property panel, SA score, QED, PAINS/Brenk/NIH "
                        "alerts and the Murcko scaffold, per molecule."
                    ),
                ),
                BoundTool(
                    name="smiles_to_safe",
                    summary="Convert a SMILES to SAFE fragment notation.",
                    takes="a SMILES string.",
                    returns="the SAFE string.",
                ),
                BoundTool(
                    name="safe_to_smiles",
                    summary="Convert SAFE fragment notation back to SMILES.",
                    takes="a SAFE string.",
                    returns="the SMILES string.",
                ),
            ),
            notes=(
                "Reach for `molecule_gates` before computing descriptors by "
                "hand. Its thresholds are the ones results are judged against, "
                "so a hand-rolled panel can disagree with the verdict. Use "
                "Python when you need the intermediate values, a custom "
                "threshold, or a calculation the tool does not do.",
                "Synthetic accessibility needs `aevom.sa_score()`; the scorer "
                "lives outside the RDKit package and reaching it by path "
                "requires `sys`, which the sandbox does not allow.",
            ),
            reference_only=False,
            replaces="writing the equivalent yourself with `python_execute`",
        ),
    ),
    LibrarySkillEntry(skill_id="biopython"),
    LibrarySkillEntry(skill_id="openbabel"),
    LibrarySkillEntry(skill_id="biotite"),
    LibrarySkillEntry(
        skill_id="plip",
        binding=SkillBinding(
            skill_id="plip",
            tools=(
                BoundTool(
                    name="plip_interactions",
                    summary=(
                        "Profile protein-ligand interactions in a complex: "
                        "which residues the ligand contacts, and by what "
                        "chemistry."
                    ),
                    takes="a `structure_ref` for a complex containing the ligand.",
                    returns=(
                        "hydrogen bonds, hydrophobic contacts, pi-stacking, "
                        "pi-cation, salt bridges, halogen bonds, water bridges "
                        "and metal coordination, as residue-level records."
                    ),
                ),
            ),
            supersedes=(
                "every `import plip` and `from plip...` line, and the "
                "`PDBComplex` / `BindingSiteReport` workflow built on them",
                "every instruction to read a structure from a file path or to "
                "write a report to disk",
                "the command-line `plipcmd` invocations",
            ),
            notes=(
                "PLIP is not installed in the python_execute sandbox, so this "
                "is the only way to profile interactions. The tool runs it in "
                "its own service, which also pins the settings that keep "
                "concurrent analyses from racing on a shared protonation file.",
            ),
            replaces="writing the equivalent yourself with `python_execute`",
        ),
    ),
)

_LIBRARY_SKILLS_BY_ID = {entry.skill_id: entry for entry in LIBRARY_SKILLS}


def get_library_skill(skill_id: str) -> Optional[LibrarySkillEntry]:
    """Look up a library skill entry by skill id."""
    return _LIBRARY_SKILLS_BY_ID.get(skill_id)


def library_skill_ids() -> tuple[str, ...]:
    """Ids of every library skill the registry is allowed to load."""
    return tuple(_LIBRARY_SKILLS_BY_ID)
