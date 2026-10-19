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

"""De novo binder design against a chosen epitope, via a self-hosted
RFD4-Proteina NIM.

The tool exists to turn something the user said -- "design a binder against
this pocket" -- into the condition spec RFD4-Proteina wants, without the user
or the model having to learn that spec. What arrives here is a target
structure and a Mol\\* selection; what leaves is:

```
contig:     "50-120, /0, A1-115"
C_CRD:      sel('A') and is_protein_backbone     # hold the target's backbone
C_SEQ:      sel('A')                             # hold the target's sequence
C_HOT:      sel('A/*/40/[CG,OH]')                # engage this epitope
```

Binder design is ``task: "unconditional"``: the target is held in place by
the coordinate and sequence conditions rather than by the task, and the
binder is the expandable part of the contig. That reads oddly but is what the
upstream task set means by the name, and it is the shape that was validated
end to end against the live NIM.

The single hazard worth stating here is residue numbering. Mol* reports
author numbering; AtomWorks resolves ``res_id`` against ``label_seq_id``. On
5O45 -- the PD-L1 entry this workflow was built against -- author residue 17
is label residue 1, so an untranslated hotspot would condition a residue 16
positions away and return a confident, wrong design. Every number this module
emits goes through ``api.storage.mmcif_numbering`` first.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)
from api.selections.design_checks import hotspot_warnings
from api.selections.design_spec import SpecError, parse_design_spec
from api.selections.role_compile import MODE_BINDER, MODE_MOTIF
from api.selections.roles import ENGINE_RFD4
from api.selections.residues import STANDARD_HEAVY_ATOMS
from api.selections.design_spec import SpecLayer
from api.selections.role_compile import (
    LayerInput,
    author_identity,
    compile_rfd4,
    reported_label,
    resolve_selection,
)
from api.selections.structure import parse_structure_atoms
from api.storage.mmcif_loop import atom_site_columns, iter_atom_site_rows
from api.storage.mmcif_numbering import NumberingMap, build_numbering_map

from .base import DataIOCategory, NvidiaApiError, NvidiaApiTool, ToolResult
from .spec_plan import plan_spec

logger = logging.getLogger(__name__)

# The one inference route this NIM serves. Discovery reports it too, and the
# adapter prefers what it discovered; this is the fallback.
GENERATIVE_PATH = "/biology/ipd/rfd4-proteina/generative"

# Binder design is "unconditional" upstream: see the module docstring.
BINDER_TASK = "unconditional"

# Motif scaffolding is its own task: the contig names the residues to carry
# and the lengths of what is designed around and between them.
MOTIF_TASK = "motif_scaffolding"

# Diffusion is slow and the NIM answers synchronously, so the client has to be
# willing to wait far longer than a hosted gateway would hold a connection.
POLL_SECONDS = 1800

# Enough of the mmCIF residue vocabulary to read a designed protein chain
# back as a sequence. RFD4 writes "X" for a token that decoded to no residue
# identity, and anything outside the twenty plus selenomethionine is treated
# the same way rather than guessed at.
UNKNOWN_RESIDUE = "X"
_THREE_TO_ONE: dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M",
}

MIN_BINDER_LENGTH = 20
MAX_BINDER_LENGTH = 400
DEFAULT_BINDER_LENGTH_MIN = 60
DEFAULT_BINDER_LENGTH_MAX = 120

MAX_DESIGNS = 32
DEFAULT_NUM_DESIGNS = 4

# Flat sampling below 20 steps is rejected by the checkpoint, which samples
# the tail of its trajectory at higher density and needs that many steps to
# lay the tail out. The schema's own bound is 1; this is the real one, found
# by sending 1 to a live NIM and reading the 422 it came back with.
MIN_STEPS = 20
MAX_STEPS = 1000

# Past this many selected atoms in one residue, naming them individually says
# nothing that naming the residue does not -- the selection is the whole
# residue -- and the sel() string stops being readable.
MAX_HOTSPOT_ATOMS_PER_RESIDUE = 8

# Hotspot conditioning is the point of the tool, so its metric is requested
# whenever the deployment can compute it. Tags the NIM reports unavailable are
# dropped rather than sent, because an unavailable tag is a request error.
WANTED_METRIC_TAGS = ("design", "hotspot_conditioning", "interchain_interactions")

# The metric that ranks designs: the fraction of requested hotspots the design
# actually engages.
HOTSPOT_OBEDIENCE_KEY = "hotspot_obedience.hotspot_obedience"

# What each stable error code means for the person who asked for a design.
# The NIM publishes these precisely so a caller can say something better than
# "the request failed"; see ErrorCode in its OpenAPI document.
ERROR_ADVICE: dict[str, str] = {
    "out_of_memory": (
        "The GPU ran out of memory. Ask for fewer designs, or a shorter binder."
    ),
    "request_too_large": (
        "The request exceeded what this deployment will accept. Reduce the "
        "number of designs, or trim the target to the domain being bound."
    ),
    "service_busy": (
        "The NIM is already working. Wait and try again; nothing is queued."
    ),
    "model_unavailable": (
        "The NIM is up but its weights are not loaded. Check the container logs."
    ),
    "pipeline_unavailable": (
        "The NIM cannot run this task on its current hardware. RFD4-Proteina "
        "needs an A100, H100 or H200; it does not run on older GPUs."
    ),
    "oracle_unavailable": (
        "A requested metric needs a scoring model this deployment does not "
        "have. Retry without refolding metrics."
    ),
    "invalid_input": (
        "The target structure was rejected. It must be mmCIF, not PDB, and the "
        "chains named in the contig must exist in it."
    ),
    "schema_violation": (
        "The request did not match the NIM's schema, which is an Aevom bug "
        "rather than anything to change about the request."
    ),
    "timeout": "The design took longer than the NIM allows. Ask for fewer designs.",
    "internal_error": (
        "The NIM failed internally. The container logs will say why; a CUDA "
        "kernel error here means the GPU is too old for this model."
    ),
}


@dataclass
class HotspotSpec:
    """Hotspot ``sel()`` strings, and what had to be said about building them.

    Attributes:
        selectors: One ``sel()`` string per conditioned residue.
        chains: Label chain ids the hotspots fall on, in first-seen order.
        residue_count: Residues conditioned.
        notes: Anything the user should know -- residues dropped, numbering
            translated. Carried into the tool's reply rather than logged,
            because a silently dropped hotspot changes the result.
    """

    selectors: list[str] = field(default_factory=list)
    chains: list[str] = field(default_factory=list)
    residue_count: int = 0
    notes: list[str] = field(default_factory=list)


_author_identity = author_identity
_reported_label = reported_label


def build_hotspot_spec(
    selection: dict[str, Any], numbering: NumberingMap
) -> HotspotSpec:
    """Turn a Mol* selection into hotspot ``sel()`` strings in label numbering.

    Residues are identified by their **author** chain and number
    (``authChainId`` / ``authSeqNumber``), which is stable across file formats
    and is what ``numbering`` is keyed on; the label numbering RFD4 needs comes
    out of ``numbering``, built from the very structure being sent. The
    viewer's own ``chainId`` / ``seqNumber`` are label ids and are used only to
    notice a selection made on a different copy of the structure. Payloads
    that predate the author fields are read as author numbering throughout.

    Atoms are grouped by residue, since a ``sel()`` names one residue and a
    list of its atoms. Non-polymer atoms -- a ligand or water caught by a
    radius query -- are dropped: they have no ``label_seq_id``, so there is no
    ``res_id`` to select them by, and a hotspot is by definition a protein
    residue the binder should touch.

    Args:
        selection: The stored ``selection`` artifact, as written by
            ``structure_query``.
        numbering: Author-to-label map built from the same target structure.

    Returns:
        A :class:`HotspotSpec`. Empty ``selectors`` means nothing usable was
        found, which the caller must treat as an error rather than as "no
        hotspots".
    """
    spec = HotspotSpec()
    resolved = resolve_selection(selection, numbering)
    if resolved.empty:
        spec.notes.append("The selection contains no atoms.")
        return spec

    for residue in resolved.residues:
        names = residue.atoms
        if names and len(names) <= MAX_HOTSPOT_ATOMS_PER_RESIDUE:
            atom_part = f"/[{','.join(names)}]" if len(names) > 1 else f"/{names[0]}"
        else:
            # The whole residue is selected; naming every atom adds nothing.
            atom_part = ""
            standard = STANDARD_HEAVY_ATOMS.get(residue.comp)
            if names and standard is not None and len(names) < standard:
                # Chosen atoms, but too many to name in a sel(): RFD4 is told
                # the whole residue, which is more than was picked.
                spec.notes.append(
                    f"Residue {residue.author}: {len(names)} atoms were chosen, more "
                    f"than the {MAX_HOTSPOT_ATOMS_PER_RESIDUE} a hotspot names "
                    "individually, so RFD4 is told the whole residue."
                )
        spec.selectors.append(
            f"sel('{residue.label_chain}/*/{residue.label_seq}{atom_part}')"
        )
        if residue.label_chain not in spec.chains:
            spec.chains.append(residue.label_chain)
        if residue.author != residue.label:
            spec.notes.append(
                f"Residue {residue.author} in the viewer is "
                f"{residue.label} in the numbering RFD4 uses; "
                "the hotspot was translated."
            )

    spec.residue_count = len(resolved.residues)
    spec.notes.extend(resolved.tail_notes("hotspot"))
    if not spec.selectors:
        spec.notes.append("No protein residues remained to condition on.")

    return spec


def build_contig(
    numbering: NumberingMap,
    target_chains: list[str],
    binder_min: int,
    binder_max: int,
) -> str:
    """Compose the contig: a binder to design, then the target to keep.

    The binder comes first as a bare length range, which is what makes it the
    expandable segment. ``/0`` is the chain break between it and the target,
    and each target chain is named by its label-numbered span, because the
    contig resolves against the same numbering ``sel()`` does.
    """
    length = (
        str(binder_min) if binder_min == binder_max else f"{binder_min}-{binder_max}"
    )
    segments = [length]
    for label_chain in target_chains:
        span = _label_span(numbering, label_chain)
        if span is None:
            continue
        first, last = span
        # The model refuses two segments of one chain, and refuses to bond
        # across unmodelled residues, so a broken chain contributes its longest
        # unbroken stretch. execute() uses the role compiler instead when it can
        # tell which side holds the hotspot.
        a, b = max(
            numbering.split_span(label_chain, first, last), key=lambda piece: piece[1] - piece[0]
        )
        segments.append(f"{label_chain}{a}-{b}")
    return ", /0, ".join(segments)


def _label_span(numbering: NumberingMap, label_chain: str) -> Optional[tuple[int, int]]:
    """Polymer residue span of a chain, named in label numbering."""
    numbers = [
        label.seq_id
        for label in numbering.residues.values()
        if label.chain_id == label_chain and label.seq_id is not None
    ]
    return (min(numbers), max(numbers)) if numbers else None


def build_condition_spec(
    *,
    example_id: str,
    contig: str,
    target_chains: Optional[list[str]] = None,
    hotspot_selectors: Optional[list[str]] = None,
    target_cif: str,
    num_designs: int,
    num_steps: Optional[int] = None,
    seed: Optional[int] = None,
    metric_tags: Optional[list[str]] = None,
    filename: str = "target.cif",
    conditions: Optional[dict[str, Any]] = None,
    task: str = BINDER_TASK,
) -> dict[str, Any]:
    """Build the full request body for one binder design.

    The three conditions each do one job. ``C_CRD`` holds the target's
    backbone where it is, qualified with ``is_protein_backbone`` so side
    chains stay free to move. ``C_SEQ`` holds the target's sequence, without
    which the model would redesign the thing we are trying to bind. ``C_HOT``
    marks the epitope.

    ``conditions`` replaces all three with a block the role compiler built
    (:func:`api.selections.role_compile.compile_rfd4`), which is how a design
    with kept ranges, fixed side chains or chemistry roles is asked for.
    """
    if conditions is None:
        chain_selector = " or ".join(f"sel('{chain}')" for chain in target_chains or [])
        conditions = {
            "C_CRD": {"true": [{"select": f"({chain_selector}) and is_protein_backbone"}]},
            "C_SEQ": {"true": [{"select": chain_selector}]},
        }
        if hotspot_selectors:
            conditions["C_HOT"] = {
                "true": [{"select": selector} for selector in hotspot_selectors]
            }

    sampling: dict[str, Any] = {"num_replicates": num_designs}
    if num_steps is not None:
        sampling["num_steps"] = num_steps
    if seed is not None:
        sampling["seed"] = seed

    request: dict[str, Any] = {
        "task": task,
        "input": {
            "format": "condition_spec",
            "example_id": example_id,
            "spec": {
                "input": filename,
                "contig": contig,
                "conditions": conditions,
            },
            "files": [{"filename": filename, "content": target_cif}],
        },
        "sampling": sampling,
    }
    if metric_tags:
        request["metrics"] = {"tags": list(metric_tags)}
    return request


def advice_for_error(message: str) -> str:
    """Append actionable advice when the NIM named one of its stable codes.

    ``_describe_http_error`` keeps the code in the message as ``[code]``, so
    the mapping is a substring check rather than a second parse of the body.
    """
    for code, advice in ERROR_ADVICE.items():
        if f"[{code}]" in message:
            return f"{message}\n\n{advice}"
    return message


class DesignProteinBinderTool(NvidiaApiTool):
    """Design a protein binder against a selected epitope.

    Unlike the built-in NVIDIA tools, this one only exists while a user has an
    RFD4-Proteina NIM registered: it is constructed by
    ``api.nim.adapters.rfd4_proteina`` and bound to that endpoint.

    Data Flow:
        Input: target mmCIF reference, selection reference marking the epitope
        Output: one annotated mmCIF and one sequence per design, plus a JSON
            metrics artifact; returns references and a ranked summary
    """

    def __init__(
        self,
        endpoint_url: str,
        api_key: Optional[str] = None,
        metric_tags: Optional[list[str]] = None,
        server_name: str = "RFD4-Proteina",
    ):
        """Bind the tool to one registered NIM.

        Args:
            endpoint_url: Full URL of the generative endpoint.
            api_key: Bearer token, or None for a NIM that needs no auth.
            metric_tags: Tags this deployment can compute, already filtered by
                the adapter against ``/v1/tasks``.
            server_name: The user's name for the NIM, used in messages.
        """
        super().__init__()
        self._endpoint_url = endpoint_url
        self.self_hosted = (endpoint_url, api_key)
        self._metric_tags = list(metric_tags or [])
        self._server_name = server_name

    @property
    def name(self) -> str:
        return "design_protein_binder"

    @property
    def api_endpoint(self) -> str:
        return self._endpoint_url

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.BOTH

    @property
    def is_async(self) -> bool:
        return False

    @property
    def summary_template(self) -> str:
        return "Designing a protein binder with RFD4-Proteina..."

    def format_summary(self, arguments: dict) -> str:
        count = arguments.get("num_designs") or DEFAULT_NUM_DESIGNS
        return f"Designing {count} protein binder(s) against the selected epitope..."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Design de novo protein binders against a specific epitope "
                    "on a target structure, using a self-hosted RFD4-Proteina "
                    "NIM. Produces backbone *and* sequence together — unlike "
                    "rfdiffusion_design_backbone, this needs no separate "
                    "sequence design step.\n\n"
                    "WORKFLOW, in order:\n"
                    "1. Get the target as mmCIF (fetch_pdb_structure returns "
                    "it). Do NOT convert a PDB file for this — the NIM rejects "
                    "PDB, and converting loses bonds and formal charges.\n"
                    "2. visualize_pdb the target.\n"
                    "3. Get the epitope — the residues the binder should touch — "
                    "as a selection_* reference. **If the user selected "
                    "residues in the viewer, use their selection** (it is "
                    "listed in [SELECTION STATE] as picked by the user); do "
                    "not re-derive an epitope they already chose. Otherwise "
                    "make one: select_residues when you know the residue "
                    "numbers, structure_query when the residues are defined by "
                    "geometry or chemistry.\n"
                    "4. Optionally call validate_design_spec first. It checks "
                    "the epitope against what the model handles well, before "
                    "any GPU time is spent.\n"
                    "5. Call this tool with either design_spec_ref (from "
                    "create_design_spec), or target_ref plus "
                    "hotspot_selection_ref.\n\n"
                    "ROLES: if [SELECTION STATE] lists design roles the user "
                    "assigned in the viewer (hotspot, keep, fix_position, "
                    "fix_sequence, float, and chemistry roles), call "
                    "create_design_spec with use_my_roles=true and pass its "
                    "design_spec_ref here. Do not rebuild those roles from "
                    "residue numbers. validate_design_spec shows what the "
                    "engine will be told from them.\n\n"
                    "The epitope selection is what steers the design, so use "
                    "the actual interface: a known binding site, the residues "
                    "around a bound ligand, or whatever the user pointed at. "
                    "Keep it small — the model was trained on hotspots of a "
                    "few atoms. A whole-chain selection is not an epitope and "
                    "will produce poor designs.\n\n"
                    "Takes minutes. Returns one annotated mmCIF and one "
                    "sequence per design, ranked by how well each engages the "
                    "requested hotspots.\n\n"
                    "Each design holds the binder *and* the whole target, so "
                    "it has more residues than the binder length asked for, "
                    "and the returned sequence spans both chains. Report the "
                    "binder length this tool gives you rather than counting "
                    "the sequence."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_ref": {
                            "type": "string",
                            "description": (
                                "Reference to the target structure, as mmCIF. "
                                "Must be the same structure the selection was "
                                "made on. Omit when design_spec_ref is given."
                            ),
                            "x-aevom-reference": {"data_type": "mmcif"},
                        },
                        "hotspot_selection_ref": {
                            "type": "string",
                            "description": (
                                "Reference to a selection_* artifact marking "
                                "the epitope residues the binder should engage "
                                "— the user's own selection, or one made with "
                                "select_residues or structure_query. Omit when "
                                "design_spec_ref is given."
                            ),
                            "x-aevom-reference": {"data_type": "selection"},
                        },
                        "design_spec_ref": {
                            "type": "string",
                            "description": (
                                "Reference to a design_spec_* artifact from "
                                "create_design_spec, in place of target_ref and "
                                "hotspot_selection_ref. Preferred when the same "
                                "target and epitope are used for several runs."
                            ),
                            "x-aevom-reference": {"data_type": "design_spec"},
                        },
                        "binder_length_min": {
                            "type": "integer",
                            "description": (
                                f"Shortest acceptable binder, in residues "
                                f"(default {DEFAULT_BINDER_LENGTH_MIN}). The "
                                "model samples a length in the range. This "
                                "bounds the designed binder only — the target "
                                "is kept on top of it. Pass the length the "
                                "user asked for; do not substitute a default."
                            ),
                        },
                        "binder_length_max": {
                            "type": "integer",
                            "description": (
                                f"Longest acceptable binder, in residues "
                                f"(default {DEFAULT_BINDER_LENGTH_MAX})."
                            ),
                        },
                        "num_designs": {
                            "type": "integer",
                            "description": (
                                f"Independent designs to generate, 1-"
                                f"{MAX_DESIGNS} (default "
                                f"{DEFAULT_NUM_DESIGNS}). Binder design has a "
                                "low success rate per attempt, so several "
                                "designs are worth more than one."
                            ),
                        },
                        "num_steps": {
                            "type": "integer",
                            "description": (
                                f"Diffusion steps, {MIN_STEPS}-{MAX_STEPS}. "
                                "Omit to use the model's own preset, which is "
                                "the right choice unless iterating quickly."
                            ),
                        },
                        "seed": {
                            "type": "integer",
                            "description": (
                                "Seed for a reproducible run. Omit for "
                                "different designs each call."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short label for these designs, e.g. "
                                "'PD-L1 binder'."
                            ),
                        },
                    },
                    # Either design_spec_ref, or target_ref with
                    # hotspot_selection_ref; execute() says which is missing.
                    "required": [],
                },
            },
        }

    async def _read_spec(
        self, spec_ref: str, target_ref: str, selection_ref: str
    ) -> tuple[str, str]:
        """Resolve a design spec to the target and hotspot selection it names.

        The spec is authoritative. A ``target_ref`` or ``hotspot_selection_ref``
        passed alongside it must agree, because disagreement means the caller
        believes the spec says something it does not.
        """
        spec = parse_design_spec(await self._retrieve(spec_ref, "design spec"))
        layer = spec.layer("hotspot")
        if layer is None:
            raise ValueError(
                f"'{spec_ref}' has no hotspot layer, and a hotspot is what "
                "steers a binder design. Add one with create_design_spec."
            )
        if target_ref and target_ref != spec.target_ref:
            raise ValueError(
                f"target_ref '{target_ref}' is not the spec's target "
                f"'{spec.target_ref}'. Omit target_ref to use the spec's, or "
                "create a spec for the other structure."
            )
        if selection_ref and selection_ref != layer.selection_ref:
            raise ValueError(
                f"hotspot_selection_ref '{selection_ref}' is not the spec's "
                f"hotspot '{layer.selection_ref}'. Omit it to use the spec's, or "
                "make a new spec with create_design_spec."
            )
        return spec.target_ref, layer.selection_ref

    async def _retrieve(self, reference: str, what: str) -> bytes:
        """Fetch a stored artifact, with a message naming what was missing."""
        if self._storage is None:
            raise ValueError(
                "Storage service not available. Cannot retrieve the "
                f"{what}."
            )
        try:
            return await self._storage.retrieve(reference)
        except InvalidReferenceError:
            raise ValueError(f"Invalid {what} reference: '{reference}'.")
        except ArtifactNotFoundError:
            raise ValueError(
                f"No {what} found for reference '{reference}'. It may have "
                "expired."
            )

    @staticmethod
    def _validate_lengths(arguments: dict[str, Any]) -> tuple[int, int]:
        """Read and check the binder length range."""
        minimum = arguments.get("binder_length_min") or DEFAULT_BINDER_LENGTH_MIN
        maximum = arguments.get("binder_length_max") or DEFAULT_BINDER_LENGTH_MAX
        try:
            minimum, maximum = int(minimum), int(maximum)
        except (TypeError, ValueError):
            raise ValueError("binder_length_min and binder_length_max must be integers.")
        if minimum > maximum:
            raise ValueError(
                f"binder_length_min ({minimum}) cannot exceed "
                f"binder_length_max ({maximum})."
            )
        if minimum < MIN_BINDER_LENGTH or maximum > MAX_BINDER_LENGTH:
            raise ValueError(
                f"Binder length must be between {MIN_BINDER_LENGTH} and "
                f"{MAX_BINDER_LENGTH} residues."
            )
        return minimum, maximum

    @staticmethod
    def _validate_num_designs(value: Any) -> int:
        if value is None:
            return DEFAULT_NUM_DESIGNS
        try:
            count = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"num_designs must be an integer, 1-{MAX_DESIGNS}.")
        if not 1 <= count <= MAX_DESIGNS:
            raise ValueError(f"num_designs must be between 1 and {MAX_DESIGNS}.")
        return count

    @staticmethod
    def _validate_steps(value: Any) -> Optional[int]:
        if value is None:
            return None
        try:
            steps = int(value)
        except (TypeError, ValueError):
            raise ValueError(f"num_steps must be an integer, {MIN_STEPS}-{MAX_STEPS}.")
        if not MIN_STEPS <= steps <= MAX_STEPS:
            raise ValueError(
                f"num_steps must be between {MIN_STEPS} and {MAX_STEPS}."
            )
        return steps

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Design binders and store each one."""
        target_ref = (arguments.get("target_ref") or "").strip()
        selection_ref = (arguments.get("hotspot_selection_ref") or "").strip()
        spec_ref = (arguments.get("design_spec_ref") or "").strip()

        if spec_ref:
            try:
                target_ref, selection_ref = await self._read_spec(
                    spec_ref, target_ref, selection_ref
                )
            except (ValueError, SpecError) as exc:
                return ToolResult.error(str(exc))

        if not target_ref:
            return ToolResult.error(
                "target_ref is required: fetch the target structure as mmCIF "
                "first, with fetch_pdb_structure."
            )
        if not selection_ref:
            return ToolResult.error(
                "A hotspot is required: pass design_spec_ref, or "
                "hotspot_selection_ref with the selection_* reference of the "
                "epitope the binder should engage. If the user selected "
                "residues in the viewer, their selection is listed in "
                "[SELECTION STATE]; otherwise create one with select_residues "
                "or structure_query."
            )

        try:
            binder_min, binder_max = self._validate_lengths(arguments)
            range_was_given = (
                arguments.get("binder_length_min") is not None
                or arguments.get("binder_length_max") is not None
            )
            num_designs = self._validate_num_designs(arguments.get("num_designs"))
            num_steps = self._validate_steps(arguments.get("num_steps"))
            seed = arguments.get("seed")
            seed = int(seed) if seed is not None else None

            target_cif = (await self._retrieve(target_ref, "target structure")).decode(
                "utf-8", errors="replace"
            )
            if not _looks_like_mmcif(target_cif):
                return ToolResult.error(
                    f"'{target_ref}' is not mmCIF. RFD4-Proteina rejects PDB "
                    "input. Fetch the target as mmCIF rather than converting a "
                    "PDB file, which loses bonds and formal charges."
                )

            raw_selection = await self._retrieve(selection_ref, "selection")
            try:
                selection = json.loads(raw_selection.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return ToolResult.error(
                    f"'{selection_ref}' is not a selection artifact. Pass the "
                    "selection_* reference returned by structure_query."
                )

            numbering = build_numbering_map(target_cif)
            if not numbering:
                return ToolResult.error(
                    f"Could not read residue numbering from '{target_ref}'. The "
                    "file has no usable _atom_site loop."
                )

            hotspots = build_hotspot_spec(selection, numbering)
            if not hotspots.selectors:
                return ToolResult.error(
                    "No usable hotspot residues in the selection. "
                    + " ".join(hotspots.notes)
                )
            # Warnings go in the same notes the reply already carries, so they
            # are read together with the design they qualify.
            hotspots.notes.extend(
                hotspot_warnings(hotspots.selectors, parse_structure_atoms(target_cif))
            )

            contig = build_contig(numbering, hotspots.chains, binder_min, binder_max)
            example_id = _example_id(arguments.get("description"), target_ref)

            # A spec with more than a hotspot (kept ranges, fixed side chains,
            # chemistry roles) is compiled by the role compiler instead; a
            # hotspot-only spec compiles to exactly what the lines above built.
            role_conditions: Optional[dict[str, Any]] = None
            if not spec_ref and numbering.breaks:
                # A chain with unmodelled residues: the plain contig cannot say
                # which side to carry, but the role compiler can.
                broken = compile_rfd4(
                    [LayerInput(SpecLayer("hotspot", selection_ref, None), selection)],
                    numbering,
                    mode=MODE_BINDER,
                    binder_min=binder_min,
                    binder_max=binder_max,
                )
                if broken.errors:
                    return ToolResult.error(
                        "The target cannot be used as it is:\n"
                        + "\n".join(f"- {e}" for e in broken.errors)
                    )
                contig = broken.contig
                role_conditions = broken.conditions
                hotspots.notes.extend(broken.warnings)
            if spec_ref:
                spec = parse_design_spec(await self._retrieve(spec_ref, "design spec"))
                if (spec.engine or ENGINE_RFD4) != ENGINE_RFD4:
                    return ToolResult.error(
                        f"The spec is for {spec.engine}, not RFD4-Proteina. Pass "
                        "it to that engine's tool, or make a spec for RFD4."
                    )
                spec_plan = await plan_spec(
                    spec,
                    self._retrieve,
                    mode=MODE_BINDER,
                    engine=ENGINE_RFD4,
                    binder_min=binder_min,
                    binder_max=binder_max,
                )
                compiled = spec_plan.rfd4
                if compiled is None or compiled.errors:
                    return ToolResult.error(
                        "The design spec cannot be run:\n"
                        + "\n".join(f"- {e}" for e in (compiled.errors if compiled else []))
                    )
                contig = compiled.contig
                role_conditions = compiled.conditions
                hotspots = HotspotSpec(
                    selectors=compiled.hotspot_selectors,
                    chains=compiled.chains,
                    residue_count=compiled.hotspot_residue_count,
                    notes=list(compiled.notes) + list(compiled.warnings),
                )

            request = build_condition_spec(
                example_id=example_id,
                contig=contig,
                target_chains=hotspots.chains,
                hotspot_selectors=hotspots.selectors,
                target_cif=target_cif,
                num_designs=num_designs,
                num_steps=num_steps,
                seed=seed,
                metric_tags=self._metric_tags,
                conditions=role_conditions,
            )

            logger.info(
                "RFD4-Proteina binder design: contig=%r hotspots=%d chains=%s "
                "designs=%d tags=%s",
                contig, hotspots.residue_count, hotspots.chains, num_designs,
                self._metric_tags,
            )

            result = await self._call_nvidia_api(
                request,
                poll_seconds=POLL_SECONDS,
                service_name=self._server_name,
            )

            outcome = await self._store_designs(
                result,
                target_ref=target_ref,
                selection_ref=selection_ref,
                spec_ref=spec_ref,
                contig=contig,
                hotspots=hotspots,
                numbering=numbering,
                description=arguments.get("description"),
                binder_min=binder_min,
                binder_max=binder_max,
                range_was_given=range_was_given,
            )
            # The trace span copies these, so a design run can be traced to the
            # selection it used, who made it, and what it was derived from.
            outcome.metadata.update(
                {
                    "selection_ref": selection_ref,
                    "selection_provenance": selection.get("provenance") or "unknown",
                    "hotspot_selector_count": len(hotspots.selectors),
                    **({"selection_parent": selection["parent"]} if selection.get("parent") else {}),
                    **({"design_spec_ref": spec_ref} if spec_ref else {}),
                }
            )
            return outcome

        except ValueError as exc:
            logger.error("Binder design validation error: %s", exc)
            return ToolResult.error(str(exc))
        except NvidiaApiError as exc:
            logger.error("RFD4-Proteina API error: %s", exc)
            return ToolResult.error(advice_for_error(str(exc)))
        except Exception as exc:  # noqa: BLE001 - a tool must not break the turn
            logger.exception("Failed to design a binder")
            return ToolResult.error(f"Failed to design a binder: {exc}")

    async def _store_designs(
        self,
        result: dict[str, Any],
        *,
        target_ref: str,
        selection_ref: str,
        spec_ref: str = "",
        contig: str,
        hotspots: HotspotSpec,
        numbering: NumberingMap,
        description: Optional[str],
        binder_min: int,
        binder_max: int,
        range_was_given: bool,
    ) -> ToolResult:
        """Store each design and summarise the set for the model."""
        designs = result.get("designs") or []
        if not designs:
            warnings = result.get("warnings") or []
            return ToolResult.error(
                "The NIM returned no designs."
                + (" " + " ".join(str(w) for w in warnings) if warnings else "")
            )
        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot store the designs."
            )

        label = description or "RFD4-Proteina binder"
        ranked = sorted(
            designs,
            key=lambda d: _obedience(d) if _obedience(d) is not None else -1.0,
            reverse=True,
        )

        stored: list[dict[str, Any]] = []
        for rank, design in enumerate(ranked):
            design_id = str(design.get("design_id") or f"design_{rank}")
            entry: dict[str, Any] = {
                "design_id": design_id,
                "hotspot_obedience": _obedience(design),
                "metrics": design.get("metrics") or {},
            }

            structure = design.get("structure") or {}
            content = structure.get("content")
            if content:
                chains = read_design_chains(str(content), hotspots.chains)
                entry["chains"] = chains
                entry["structure_ref"] = await self._storage.store(
                    data_type="annotated_cif",
                    content=str(content).encode("utf-8"),
                    metadata={
                        "source": "rfd4_proteina",
                        "design_id": design_id,
                        "target_ref": target_ref,
                        "selection_ref": selection_ref,
                        **({"design_spec_ref": spec_ref} if spec_ref else {}),
                        "contig": contig,
                        "hotspot_selectors": hotspots.selectors,
                        "numbering_translated": not numbering.identical,
                        "hotspot_obedience": _obedience(design),
                        "binder_length": chains.binder_length,
                        "target_length": chains.target_length,
                        "binder_chains": list(chains.binder),
                        "target_chains": list(chains.target),
                    },
                    description=f"{label} — design {design_id}",
                )

            complex_sequence = design.get("sequence")
            if complex_sequence:
                entry["complex_length"] = len(str(complex_sequence))

            # The NIM's sequence string spans binder and target, binder first,
            # while the structure puts the binder on a *later* chain id than
            # the target it was designed against. Nothing in the string says
            # where one ends, so what is stored is the binder read from the
            # structure, which is what a downstream tool wants to fold or
            # score. Only if the structure cannot be read is the complex-wide
            # string kept, and it is labelled as such.
            chains = entry.get("chains")
            binder_sequence = chains.binder_sequence if chains else ""
            if binder_sequence:
                entry["sequence_ref"] = await self._storage.store(
                    data_type="sequence",
                    content=binder_sequence.encode("utf-8"),
                    metadata={
                        "source": "rfd4_proteina",
                        "design_id": design_id,
                        "scope": "binder",
                        "length": len(binder_sequence),
                        "chains": list(chains.binder),
                    },
                    description=(
                        f"{label} — design {design_id} binder sequence "
                        f"(chain {chains.binder_chain_ids}, "
                        f"{len(binder_sequence)} residues, target excluded)"
                    ),
                )
            elif complex_sequence:
                entry["sequence_ref"] = await self._storage.store(
                    data_type="sequence",
                    content=str(complex_sequence).encode("utf-8"),
                    metadata={
                        "source": "rfd4_proteina",
                        "design_id": design_id,
                        "scope": "complex",
                        "length": len(str(complex_sequence)),
                    },
                    description=(
                        f"{label} — design {design_id} sequence of the whole "
                        "complex (binder and target; chain boundary unknown)"
                    ),
                )

            stored.append(entry)

        metrics_ref = await self._storage.store(
            data_type="json",
            content=json.dumps(
                {
                    "designs": [
                        {
                            "design_id": entry["design_id"],
                            "metrics": entry["metrics"],
                        }
                        for entry in stored
                    ],
                    "aggregated_metrics": result.get("aggregated_metrics") or [],
                    "contig": contig,
                    "hotspot_selectors": hotspots.selectors,
                },
                indent=2,
            ).encode("utf-8"),
            metadata={"source": "rfd4_proteina", "target_ref": target_ref},
            description=f"{label} — design metrics",
        )

        return ToolResult.data_reference(
            reference=stored[0].get("structure_ref", metrics_ref),
            data_type="annotated_cif",
            message=self._summarise(
                stored,
                result=result,
                contig=contig,
                hotspots=hotspots,
                metrics_ref=metrics_ref,
                label=label,
                binder_min=binder_min,
                binder_max=binder_max,
                range_was_given=range_was_given,
            ),
            metadata={
                "source": "rfd4_proteina",
                "design_count": len(stored),
                "contig": contig,
                "metrics_ref": metrics_ref,
            },
        )

    def _summarise(
        self,
        stored: list[dict[str, Any]],
        *,
        result: dict[str, Any],
        contig: str,
        hotspots: HotspotSpec,
        metrics_ref: str,
        label: str,
        binder_min: int,
        binder_max: int,
        range_was_given: bool,
    ) -> str:
        """The reply the model reads: references and ranking, no structure text."""
        best = stored[0]
        message = format_storage_message(
            data_type="annotated_cif",
            reference=best.get("structure_ref", metrics_ref),
            source_description=f"{label} — best design ({best['design_id']})",
            accepting_tools=["visualize_pdb", "structure_query", "convert_structure"],
        )

        requested = _range_phrase(binder_min, binder_max)
        if not range_was_given:
            requested += " (tool default; none was given)"

        lines = [
            "",
            (
                f"Designed {len(stored)} binder(s) against "
                f"{hotspots.residue_count} hotspot residue(s)."
            ),
            f"- Binder length requested: {requested}",
            f"- Contig: {contig}",
            f"- Hotspots: {', '.join(hotspots.selectors)}",
            "",
            (
                "Designs, best first. Each structure holds two parts: the "
                "designed binder and the whole target it was designed "
                "against, on separate chains. The binder is NOT chain A: it "
                "takes a new chain id, and the target keeps its own. The "
                "complex is longer than the requested binder length"
                + (
                    ", and each sequence_* reference below is the binder alone:"
                    if all(e.get("chains") and e["chains"].binder for e in stored)
                    else ". Where a design's chains could not be read, its "
                    "sequence_* reference covers the whole complex:"
                )
            ),
        ]
        for entry in stored:
            obedience = entry.get("hotspot_obedience")
            score = (
                f"hotspot obedience {obedience:.2f}"
                if isinstance(obedience, (int, float))
                else "hotspot obedience not measured"
            )
            parts = [f"  - {entry['design_id']}: {score}"]
            parts.extend(_length_phrases(entry))
            chains = entry.get("chains")
            if chains and chains.binder:
                parts.append(chains.describe())
            for key in ("structure_ref", "sequence_ref"):
                if entry.get(key):
                    parts.append(entry[key])
            lines.append(", ".join(parts))

        passes = [
            row
            for row in result.get("aggregated_metrics") or []
            if str(row.get("metric", "")).endswith(".pass")
        ]
        if passes:
            lines.append("")
            lines.append("In-silico success rate:")
            for row in passes:
                lines.append(
                    f"  - {row['metric']}: {row['value']:.2f} over {row['n']} design(s)"
                )

        notes = list(hotspots.notes) + [
            str(w) for w in result.get("warnings") or []
        ]
        if notes:
            lines.append("")
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in notes)

        lines.append("")
        lines.append(f"Full per-design metrics: {metrics_ref}")
        lines.append(
            f"Display the best design with visualize_pdb('"
            f"{best.get('structure_ref', metrics_ref)}')."
        )
        return message + "\n".join(lines)


def _range_phrase(minimum: int, maximum: int) -> str:
    """``80-100 residues``, or ``80 residues`` for a fixed length."""
    return f"{minimum} residues" if minimum == maximum else f"{minimum}-{maximum} residues"


def _length_phrases(entry: dict[str, Any]) -> list[str]:
    """How long a design is: the binder first, the complex second.

    The binder is the number the request asked for, so it leads.
    """
    chains = entry.get("chains")
    binder = chains.binder_length if chains else 0
    complex_length = entry.get("complex_length")
    phrases = []
    if binder:
        phrases.append(f"{binder}-residue binder")
    if complex_length:
        phrases.append(f"{complex_length} residues with the target")
    return phrases


def _obedience(design: dict[str, Any]) -> Optional[float]:
    """A design's hotspot obedience, when the metric was computed."""
    value = (design.get("metrics") or {}).get(HOTSPOT_OBEDIENCE_KEY)
    return float(value) if isinstance(value, (int, float)) else None


def _looks_like_mmcif(text: str) -> bool:
    """Whether this is mmCIF rather than PDB, which the NIM rejects."""
    return text.lstrip().startswith("data_")


@dataclass(frozen=True)
class DesignChains:
    """A design's chains, split into the binder and the target it engages.

    Each maps a chain id to that chain's one-letter sequence, in file order.
    """

    binder: dict[str, str]
    target: dict[str, str]

    @property
    def binder_length(self) -> int:
        return sum(len(sequence) for sequence in self.binder.values())

    @property
    def target_length(self) -> int:
        return sum(len(sequence) for sequence in self.target.values())

    @property
    def binder_sequence(self) -> str:
        """The designed sequence alone, without the target."""
        return "".join(self.binder.values())

    @property
    def binder_chain_ids(self) -> str:
        return "+".join(self.binder) or "?"

    @property
    def target_chain_ids(self) -> str:
        return "+".join(self.target) or "?"

    def describe(self) -> str:
        """Which chain is which, for the model reading the result.

        This is the sentence whose absence once had a model report the binder
        as chain A: the binder is written first but lands on a later chain id.
        """
        return (
            f"binder on chain {self.binder_chain_ids} "
            f"({self.binder_length} residues), target on chain "
            f"{self.target_chain_ids} ({self.target_length} residues)"
        )


def read_design_chains(cif_text: str, target_chains: list[str]) -> DesignChains:
    """Split a design into its binder chains and the target chains it keeps.

    A design is the binder *plus* the whole target it was designed against,
    because the contig preserves the target's chains alongside the expandable
    binder segment. Upstream's length guidance is that "the length counts only
    what gets designed", so the requested range bounds the binder alone.

    Chains are told apart by ``label_asym_id``: a contig run holding a
    preserved segment keeps that segment's chain id, so the target comes back
    named as it went in, and the run with no preserved segment -- the binder
    -- is given a fresh one. That is why the binder is typically chain B while
    the target stays chain A, even though the binder is written first.

    Returns empty chains for a file with no readable ``_atom_site`` loop
    rather than guessing which part is which.
    """
    columns = atom_site_columns(cif_text)
    try:
        chain_index = columns.index("label_asym_id")
        residue_index = columns.index("label_seq_id")
        name_index = columns.index("label_comp_id")
    except ValueError:
        return DesignChains(binder={}, target={})

    # Insertion order is file order, which is what makes the binder first.
    sequences: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    for values in iter_atom_site_rows(cif_text):
        chain = values[chain_index]
        residue = (chain, values[residue_index])
        if residue in seen:
            continue
        seen.add(residue)
        letter = _THREE_TO_ONE.get(values[name_index].upper(), UNKNOWN_RESIDUE)
        sequences.setdefault(chain, []).append(letter)

    targets = set(target_chains)
    return DesignChains(
        binder={
            chain: "".join(letters)
            for chain, letters in sequences.items()
            if chain not in targets
        },
        target={
            chain: "".join(letters)
            for chain, letters in sequences.items()
            if chain in targets
        },
    )


def _example_id(description: Optional[str], target_ref: str) -> str:
    """A filesystem-safe id the NIM echoes back on every design.

    Upstream builds design ids by prefixing this, so it should read as a name.
    """
    source = description or target_ref
    cleaned = "".join(c if c.isalnum() else "_" for c in source).strip("_").lower()
    return (cleaned or "binder")[:48]


class DesignMotifScaffoldTool(DesignProteinBinderTool):
    """Design a protein that carries a motif, from a design spec's Keep.

    The motif is the user's Keep: those residues are carried into the design
    with their roles (hold position, hold sequence, or float) and everything
    around and between them is designed, at the lengths the spec's layout says.
    Where the binder tool designs a new chain against a hotspot, this designs
    one protein around residues.

    Data Flow:
        Input: a design_spec_* reference with a Keep (and any roles)
        Output: one annotated mmCIF and one sequence per design, plus a JSON
            metrics artifact; returns references and a summary
    """

    @property
    def name(self) -> str:
        return "design_motif_scaffold"

    @property
    def summary_template(self) -> str:
        return "Scaffolding a motif with RFD4-Proteina..."

    def format_summary(self, arguments: dict) -> str:
        count = arguments.get("num_designs") or DEFAULT_NUM_DESIGNS
        return f"Scaffolding the kept motif into {count} design(s)..."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Design a new protein that carries a motif — residues "
                    "taken from a structure — using a self-hosted "
                    "RFD4-Proteina NIM. The motif residues are the spec's "
                    "Keep; everything around and between them is designed. "
                    "Backbone and sequence come out together.\n\n"
                    "WORKFLOW, in order:\n"
                    "1. Get the structure as mmCIF (fetch_pdb_structure "
                    "returns it).\n"
                    "2. Get the motif as roles. **If the user assigned roles "
                    "in the viewer** (listed in [SELECTION STATE] as design "
                    "roles), call create_design_spec with use_my_roles=true. "
                    "Otherwise pass layers: keep for the motif residues, plus "
                    "fix_position, fix_sequence or float as the user asks. A "
                    "Keep with nothing else keeps positions and sequence "
                    "fixed; Float lets the design choose where the motif "
                    "sits.\n"
                    "3. Optionally validate_design_spec, which shows the "
                    "contig and conditions this will send.\n"
                    "4. Call this tool with design_spec_ref.\n\n"
                    "How long the designed pieces are comes from the keep "
                    "layer's layout (the user sets it in the viewer; to set "
                    "it yourself, give the keep layer a value of "
                    "{n_term:[min,max], linkers:[[min,max],...], "
                    "c_term:[min,max]}). Do not substitute lengths of your "
                    "own.\n\n"
                    "Takes minutes. Returns one annotated mmCIF and one "
                    "sequence per design."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "design_spec_ref": {
                            "type": "string",
                            "description": (
                                "Reference to a design_spec_* artifact from "
                                "create_design_spec, with a keep layer."
                            ),
                            "x-aevom-reference": {"data_type": "design_spec"},
                        },
                        "num_designs": {
                            "type": "integer",
                            "description": (
                                f"Independent designs to generate, 1-{MAX_DESIGNS} "
                                f"(default {DEFAULT_NUM_DESIGNS})."
                            ),
                        },
                        "num_steps": {
                            "type": "integer",
                            "description": (
                                f"Diffusion steps, {MIN_STEPS}-{MAX_STEPS}. Omit "
                                "to use the model's own preset."
                            ),
                        },
                        "seed": {
                            "type": "integer",
                            "description": "Seed for a reproducible run.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Short label for these designs.",
                        },
                    },
                    "required": ["design_spec_ref"],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Scaffold the spec's motif and store each design."""
        spec_ref = (arguments.get("design_spec_ref") or "").strip()
        if not spec_ref:
            return ToolResult.error(
                "design_spec_ref is required. If the user assigned roles in the "
                "viewer, make one with create_design_spec(use_my_roles=true); "
                "otherwise pass layers with a keep layer."
            )
        try:
            num_designs = self._validate_num_designs(arguments.get("num_designs"))
            num_steps = self._validate_steps(arguments.get("num_steps"))
            seed = arguments.get("seed")
            seed = int(seed) if seed is not None else None

            spec = parse_design_spec(await self._retrieve(spec_ref, "design spec"))
            if (spec.engine or ENGINE_RFD4) != ENGINE_RFD4:
                return ToolResult.error(
                    f"The spec is for {spec.engine}, not RFD4-Proteina. Make a spec for RFD4."
                )
            if spec.layer("keep") is None:
                return ToolResult.error(
                    f"'{spec_ref}' has no keep layer. Motif scaffolding carries "
                    "the kept residues into the design: add a keep layer with "
                    "create_design_spec."
                )
            plan = await plan_spec(spec, self._retrieve, mode=MODE_MOTIF, engine=ENGINE_RFD4)
            compiled = plan.rfd4
            if compiled is None or compiled.errors:
                return ToolResult.error(
                    "The design spec cannot be run:\n"
                    + "\n".join(f"- {e}" for e in (compiled.errors if compiled else []))
                )

            request = build_condition_spec(
                example_id=_example_id(arguments.get("description"), spec.target_ref),
                contig=compiled.contig,
                target_cif=plan.target_cif,
                num_designs=num_designs,
                num_steps=num_steps,
                seed=seed,
                conditions=compiled.conditions,
                task=MOTIF_TASK,
            )
            logger.info(
                "RFD4-Proteina motif scaffolding: contig=%r designs=%d",
                compiled.contig, num_designs,
            )
            result = await self._call_nvidia_api(
                request, poll_seconds=POLL_SECONDS, service_name=self._server_name
            )
            outcome = await self._store_scaffolds(
                result,
                spec_ref=spec_ref,
                target_ref=spec.target_ref,
                contig=compiled.contig,
                notes=list(compiled.notes) + list(compiled.warnings),
                description=arguments.get("description"),
            )
            outcome.metadata.update({"design_spec_ref": spec_ref})
            return outcome
        except (ValueError, SpecError) as exc:
            logger.error("Motif scaffolding validation error: %s", exc)
            return ToolResult.error(str(exc))
        except NvidiaApiError as exc:
            logger.error("RFD4-Proteina API error: %s", exc)
            return ToolResult.error(advice_for_error(str(exc)))
        except Exception as exc:  # noqa: BLE001 - a tool must not break the turn
            logger.exception("Failed to scaffold a motif")
            return ToolResult.error(f"Failed to scaffold a motif: {exc}")

    async def _store_scaffolds(
        self,
        result: dict[str, Any],
        *,
        spec_ref: str,
        target_ref: str,
        contig: str,
        notes: list[str],
        description: Optional[str],
    ) -> ToolResult:
        """Store each scaffold and summarise the set for the model."""
        designs = result.get("designs") or []
        if not designs:
            warnings = result.get("warnings") or []
            return ToolResult.error(
                "The NIM returned no designs."
                + (" " + " ".join(str(w) for w in warnings) if warnings else "")
            )
        if self._storage is None:
            return ToolResult.error("Storage service not available. Cannot store the designs.")

        label = description or "RFD4-Proteina motif scaffold"
        stored: list[dict[str, Any]] = []
        for rank, design in enumerate(designs):
            design_id = str(design.get("design_id") or f"design_{rank}")
            entry: dict[str, Any] = {
                "design_id": design_id,
                "metrics": design.get("metrics") or {},
            }
            content = (design.get("structure") or {}).get("content")
            if content:
                # No target: every chain is designed, so every chain is read.
                chains = read_design_chains(str(content), [])
                entry["chains"] = chains.binder
                entry["structure_ref"] = await self._storage.store(
                    data_type="annotated_cif",
                    content=str(content).encode("utf-8"),
                    metadata={
                        "source": "rfd4_proteina",
                        "design_id": design_id,
                        "target_ref": target_ref,
                        "design_spec_ref": spec_ref,
                        "contig": contig,
                        "task": MOTIF_TASK,
                        "chains": {c: len(seq) for c, seq in chains.binder.items()},
                    },
                    description=f"{label} — design {design_id}",
                )
                sequence = chains.binder_sequence
                if sequence:
                    entry["sequence_ref"] = await self._storage.store(
                        data_type="sequence",
                        content=sequence.encode("utf-8"),
                        metadata={
                            "source": "rfd4_proteina",
                            "design_id": design_id,
                            "length": len(sequence),
                        },
                        description=f"{label} — design {design_id} sequence ({len(sequence)} residues)",
                    )
            stored.append(entry)

        metrics_ref = await self._storage.store(
            data_type="json",
            content=json.dumps(
                {
                    "designs": [
                        {"design_id": e["design_id"], "metrics": e["metrics"]} for e in stored
                    ],
                    "aggregated_metrics": result.get("aggregated_metrics") or [],
                    "contig": contig,
                },
                indent=2,
            ).encode("utf-8"),
            metadata={"source": "rfd4_proteina", "target_ref": target_ref},
            description=f"{label} — design metrics",
        )

        best = stored[0]
        best_ref = best.get("structure_ref", metrics_ref)
        message = format_storage_message(
            data_type="annotated_cif",
            reference=best_ref,
            source_description=f"{label} — first design ({best['design_id']})",
            accepting_tools=["visualize_pdb", "structure_query", "convert_structure"],
        )
        lines = ["", f"Scaffolded the motif into {len(stored)} design(s).", f"- Contig: {contig}", ""]
        lines.append(
            "Designs. The kept residues keep the chain they had; chain ids and "
            "residue numbers in the output are renumbered by the engine, so do "
            "not assume the motif is at its original numbers:"
        )
        for entry in stored:
            parts = [f"  - {entry['design_id']}"]
            chains = entry.get("chains") or {}
            if chains:
                parts.append(
                    "chains " + ", ".join(f"{c} ({len(seq)} residues)" for c, seq in chains.items())
                )
            for key in ("structure_ref", "sequence_ref"):
                if entry.get(key):
                    parts.append(entry[key])
            lines.append(", ".join(parts))
        all_notes = notes + [str(w) for w in result.get("warnings") or []]
        if all_notes:
            lines += ["", "Notes:"] + [f"  - {n}" for n in all_notes]
        lines += [
            "",
            f"Full per-design metrics: {metrics_ref}",
            f"Display a design with visualize_pdb('{best_ref}').",
        ]
        return ToolResult.data_reference(
            reference=best_ref,
            data_type="annotated_cif",
            message=message + "\n".join(lines),
            metadata={
                "source": "rfd4_proteina",
                "design_count": len(stored),
                "contig": contig,
                "metrics_ref": metrics_ref,
                "task": MOTIF_TASK,
            },
        )
