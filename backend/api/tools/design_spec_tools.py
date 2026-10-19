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

"""Tools that bind selections to design roles and check them.

``create_design_spec`` records *what a design is asked to do* -- which target,
and which selections play which role (hotspot, keep, fix position, fix
sequence, float, chemistry) -- as one artifact the user, the agent and the
design tool all refer to. ``validate_design_spec`` compiles the same inputs the
design tool does and says, before any GPU time is spent, what the engine will
actually be told and whether it is likely to follow it.

A spec's layers can come from three places, and a call may mix them: explicit
``layers``, a bare ``hotspot_selection_ref``, or ``use_my_roles``, which takes
the roles the user has assigned in the viewer (their latest selection for each
role). Neither tool needs RFD4-Proteina to be registered: the checks are about
the spec, not the model.
"""

import logging
from typing import Any, Optional

from api.selections.design_checks import hotspot_warnings
from api.selections.design_spec import (
    DesignSpec,
    SpecError,
    SpecLayer,
    parse_design_spec,
    parse_layer,
)
from api.selections.ops import SelectionError
from api.selections.registry import get_role_layers
from api.selections.role_compile import MODE_BINDER, MODE_MOTIF
from api.selections.roles import ENGINE_RFD4, ENGINES, ROLES

from .base import ToolResult
from .selection_tools import _SelectionTool, _get_redis
from .spec_plan import SpecPlan, plan_spec

logger = logging.getLogger(__name__)

# The reply lists the sel() strings RFD4 will receive; past this it counts.
MAX_LISTED_SELECTORS = 12

_LAYER_SCHEMA = {
    "type": "object",
    "properties": {
        "condition": {
            "type": "string",
            "enum": list(ROLES),
            "description": (
                "The role: hotspot (binder should touch), keep (carry these "
                "residues into the design), fix_position (hold coordinates), "
                "fix_sequence (hold amino-acid identity), float (keep but let "
                "the design choose where they sit), secondary_structure, rasa "
                "(burial), hbond, pistack, pication, salt, hydrophobic, metal."
            ),
        },
        "role": {
            "type": "string",
            "enum": list(ROLES),
            "description": "Alias for 'condition'. Give one or the other.",
        },
        "selection_ref": {
            "type": "string",
            "description": "The selection_* whose residues play this role.",
            "x-aevom-reference": {"data_type": "selection"},
        },
        "value": {
            "description": (
                "The role's parameter: HELIX/SHEET/LOOP for secondary_structure; "
                "BURIED/INTERMEDIATE/EXPOSED for rasa; DONOR/ACCEPTOR/BOTH for "
                "hbond; AROMATIC/CATION for pication; POSITIVE/NEGATIVE for salt; "
                "COORDINATING for metal; a group name for float (default "
                "'motif'); for keep, an optional layout "
                "{n_term:[min,max], linkers:[[min,max],...], c_term:[min,max]}."
            ),
        },
    },
    # Either 'condition' or its alias 'role' names the role; parse_layer
    # insists on one and says so, which a schema "required" cannot express
    # portably across providers.
    "required": ["selection_ref"],
}


class _SpecTool(_SelectionTool):
    """Shared reading of a spec's layers and their compilation."""

    async def _my_role_layers(self) -> list[SpecLayer]:
        """The roles the user has assigned in the viewer, as spec layers."""
        redis = _get_redis()
        if redis is None:
            return []
        infos = await get_role_layers(redis, self._session_id or "default")
        return [SpecLayer(i.role, i.reference, i.role_value) for i in infos if i.role]

    async def _gather_layers(self, arguments: dict[str, Any]) -> list[SpecLayer]:
        """Layers from my roles, then explicit layers, then a bare hotspot.

        Later sources override earlier ones for the same role (and parameter),
        so an explicit hotspot always beats the user's.

        Raises:
            SpecError: If a layer is malformed.
        """
        gathered: dict[tuple[str, str], SpecLayer] = {}
        if arguments.get("use_my_roles"):
            for layer in await self._my_role_layers():
                gathered[layer.key] = layer
        for raw in arguments.get("layers") or []:
            layer = parse_layer(raw)
            gathered[layer.key] = layer
        hotspot_ref = (arguments.get("hotspot_selection_ref") or "").strip()
        if hotspot_ref:
            layer = SpecLayer("hotspot", hotspot_ref)
            gathered[layer.key] = layer
        return list(gathered.values())

    async def _plan(
        self, spec: DesignSpec, arguments: dict[str, Any]
    ) -> SpecPlan:
        mode = (arguments.get("mode") or "").strip().lower() or None
        engine = (arguments.get("engine") or "").strip().lower() or None
        return await plan_spec(
            spec,
            self._retrieve,
            mode=mode,
            engine=engine,
            binder_min=int(arguments.get("binder_length_min") or 60),
            binder_max=int(arguments.get("binder_length_max") or 120),
        )


class CreateDesignSpecTool(_SpecTool):
    """Record a target and the roles of its selections as a design spec."""

    @property
    def name(self) -> str:
        return "create_design_spec"

    @property
    def summary_template(self) -> str:
        return "Creating design spec..."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "create_design_spec",
                "description": (
                    "Record what a design is asked to do: the target structure "
                    "and which selection plays which role. Returns a "
                    "design_spec_* reference for validate_design_spec, "
                    "design_protein_binder and design_motif_scaffold.\n\n"
                    "Roles: hotspot (the binder should touch these), keep (carry "
                    "these residues into the design; the rest is dropped), "
                    "fix_position (hold their coordinates), fix_sequence (hold "
                    "their amino-acid identities), float (keep them but let the "
                    "design choose where they sit), plus chemistry roles "
                    "(hbond, pistack, pication, salt, hydrophobic, metal), "
                    "secondary_structure and rasa (burial).\n\n"
                    "Pass use_my_roles=true to take the roles the user assigned "
                    "in the viewer (listed in [SELECTION STATE] as roles); "
                    "anything you pass in layers or hotspot_selection_ref "
                    "overrides the user's role of the same kind. Use a spec when "
                    "the same target and roles will be designed against more "
                    "than once. Specs are never edited: to change a role, make "
                    "a new spec with based_on_spec_ref, which records the old "
                    "one as its parent."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_ref": {
                            "type": "string",
                            "description": "The target structure (mmCIF for RFD4, PDB for RFdiffusion).",
                            "x-aevom-reference": {"data_type": "mmcif"},
                        },
                        "use_my_roles": {
                            "type": "boolean",
                            "description": "Take the roles the user assigned in the viewer.",
                        },
                        "layers": {
                            "type": "array",
                            "items": _LAYER_SCHEMA,
                            "description": "Selections and the roles they play.",
                        },
                        "hotspot_selection_ref": {
                            "type": "string",
                            "description": (
                                "Shorthand for a hotspot layer: the epitope "
                                "residues, as a selection_* reference."
                            ),
                            "x-aevom-reference": {"data_type": "selection"},
                        },
                        "engine": {
                            "type": "string",
                            "enum": list(ENGINES),
                            "description": "Engine the spec is for (default rfd4).",
                        },
                        "based_on_spec_ref": {
                            "type": "string",
                            "description": (
                                "An earlier spec to derive this one from. Its "
                                "target_ref is used if you omit target_ref."
                            ),
                            "x-aevom-reference": {"data_type": "design_spec"},
                        },
                        "name": {
                            "type": "string",
                            "description": "Short name, e.g. 'PD-L1 epitope v2'.",
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        target_ref = (arguments.get("target_ref") or "").strip()
        base_ref = (arguments.get("based_on_spec_ref") or "").strip()
        name = (arguments.get("name") or "").strip()[:80] or None
        engine = (arguments.get("engine") or "").strip().lower() or None
        if engine is not None and engine not in ENGINES:
            return ToolResult.invalid_arguments(
                f"engine must be one of: {', '.join(ENGINES)}."
            )

        try:
            layers = await self._gather_layers(arguments)
            if not layers:
                return ToolResult.invalid_arguments(
                    "Nothing to record: pass layers, hotspot_selection_ref, or "
                    "use_my_roles=true when the user has assigned roles in the viewer."
                )

            base: Optional[DesignSpec] = None
            if base_ref:
                base = parse_design_spec(await self._retrieve(base_ref, "design spec"))
                target_ref = target_ref or base.target_ref
            if not target_ref:
                return ToolResult.invalid_arguments(
                    "target_ref is required (or based_on_spec_ref, which names one)."
                )
            if base is not None and target_ref != base.target_ref:
                return ToolResult.invalid_arguments(
                    f"target_ref '{target_ref}' is not the base spec's target "
                    f"'{base.target_ref}'. A spec is for one target; start a new "
                    "one without based_on_spec_ref."
                )

            # Fail now, not at design time: the references must exist and the
            # selections must be readable.
            await self._retrieve(target_ref, "target structure")
            for layer in layers:
                await self._load_selection(layer.selection_ref)

            if base is not None:
                spec = base.with_layers(layers)
                spec.parent = base_ref
                spec.name = name or base.name
                spec.engine = engine or base.engine
            else:
                spec = DesignSpec(
                    target_ref=target_ref, layers=layers, name=name, engine=engine
                )
        except (SelectionError, SpecError) as exc:
            return ToolResult.error(str(exc))

        if self._storage is None:
            return ToolResult.error("Storage service not available.")
        roles_text = ", ".join(
            f"{ROLES[l.condition].label.lower()}"
            + (f" ({l.key[1]})" if l.key[1] else "")
            + f" {l.selection_ref}"
            for l in spec.layers
        )
        reference = await self._storage.store(
            data_type="design_spec",
            content=spec.to_json(),
            metadata={
                "target_ref": spec.target_ref,
                "layer_count": len(spec.layers),
                "conditions": [l.condition for l in spec.layers],
                **({"parent": spec.parent} if spec.parent else {}),
                **({"name": spec.name} if spec.name else {}),
                **({"engine": spec.engine} if spec.engine else {}),
            },
            description=(
                f"Design spec{f' {spec.name!r}' if spec.name else ''}: "
                f"{roles_text} on {spec.target_ref}"
            ),
        )
        commit = getattr(self._storage, "commit", None)
        if commit is not None:
            await commit()
        return ToolResult.text(
            f"Design spec stored as '{reference}': {roles_text} on "
            f"{spec.target_ref}. Run validate_design_spec('{reference}') to see "
            f"what the engine will be told, or pass design_spec_ref='{reference}' "
            "to the design tool.",
            metadata={"design_spec_ref": reference},
        )


class ValidateDesignSpecTool(_SpecTool):
    """Check a spec before spending GPU time on it."""

    @property
    def name(self) -> str:
        return "validate_design_spec"

    @property
    def summary_template(self) -> str:
        return "Validating design spec..."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "validate_design_spec",
                "description": (
                    "Check a design's inputs before running it: what the engine "
                    "will be told (the contig and each condition, in the engine's "
                    "own numbering), residues that could not be mapped onto the "
                    "target, roles that do not fit together (a hotspot outside "
                    "what is kept), roles the engine cannot honour, and whether "
                    "the epitope is of a size and shape the model handles well. "
                    "Takes seconds and runs no model. Pass design_spec_ref, or "
                    "target_ref with layers, hotspot_selection_ref or "
                    "use_my_roles."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "design_spec_ref": {
                            "type": "string",
                            "x-aevom-reference": {"data_type": "design_spec"},
                        },
                        "target_ref": {
                            "type": "string",
                            "x-aevom-reference": {"data_type": "mmcif"},
                        },
                        "hotspot_selection_ref": {
                            "type": "string",
                            "x-aevom-reference": {"data_type": "selection"},
                        },
                        "layers": {"type": "array", "items": _LAYER_SCHEMA},
                        "use_my_roles": {"type": "boolean"},
                        "mode": {
                            "type": "string",
                            "enum": [MODE_BINDER, MODE_MOTIF],
                            "description": (
                                "binder (design a chain against the hotspot) or "
                                "motif (scaffold the kept residues). Default: "
                                "binder if there is a hotspot, else motif."
                            ),
                        },
                        "engine": {"type": "string", "enum": list(ENGINES)},
                        "binder_length_min": {"type": "integer"},
                        "binder_length_max": {"type": "integer"},
                    },
                    "required": [],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        spec_ref = (arguments.get("design_spec_ref") or "").strip()
        target_ref = (arguments.get("target_ref") or "").strip()

        try:
            if spec_ref:
                spec = parse_design_spec(await self._retrieve(spec_ref, "design spec"))
            else:
                layers = await self._gather_layers(arguments)
                if not (target_ref and layers):
                    return ToolResult.invalid_arguments(
                        "Pass design_spec_ref, or target_ref with layers, "
                        "hotspot_selection_ref or use_my_roles."
                    )
                spec = DesignSpec(target_ref=target_ref, layers=layers)
            plan = await self._plan(spec, arguments)
        except (SelectionError, SpecError, ValueError) as exc:
            return ToolResult.error(str(exc))

        if plan.rfd4 is not None:
            return self._report_rfd4(spec, spec_ref, plan)
        return self._report_rfdiffusion1(spec, spec_ref, plan)

    def _report_rfd4(self, spec: DesignSpec, spec_ref: str, plan: SpecPlan) -> ToolResult:
        compiled = plan.rfd4
        assert compiled is not None
        hotspot_layer = spec.layer("hotspot")

        if plan.mode == MODE_BINDER and hotspot_layer is not None and not compiled.hotspot_selectors:
            return ToolResult.error(
                "No usable hotspot residues. "
                + " ".join(compiled.notes + compiled.errors)
            )
        lines = [
            f"Design spec {spec_ref or '(unsaved)'} on {spec.target_ref}: "
            f"{plan.mode} design for RFD4.",
        ]
        lines.extend(f"  {line}" for line in compiled.summary)
        if compiled.hotspot_selectors:
            listed = compiled.hotspot_selectors[:MAX_LISTED_SELECTORS]
            more = len(compiled.hotspot_selectors) - len(listed)
            lines.append(
                f"RFD4 will be told to engage {compiled.hotspot_residue_count} residue(s) on "
                f"chain(s) {', '.join(dict.fromkeys(_chain_of(s) for s in compiled.hotspot_selectors))}"
                " (label numbering):"
            )
            lines.extend(f"  {s}" for s in listed)
            if more > 0:
                lines.append(f"  (+{more} more)")
        if compiled.contig:
            lines.append(f"Contig: {compiled.contig}")
        extra = [
            f"  {annotation} {value}: {len(entries)} selection(s)"
            for annotation, values in compiled.conditions.items()
            if annotation not in ("C_HOT",)
            for value, entries in values.items()
        ]
        if extra:
            lines.append("Conditions sent:")
            lines.extend(extra)

        problems = list(compiled.notes) + list(compiled.warnings)
        if plan.mode == MODE_BINDER and compiled.hotspot_selectors:
            problems += hotspot_warnings(compiled.hotspot_selectors, plan.target_atoms)
        if compiled.errors:
            lines.append("")
            lines.append("Cannot run as it stands:")
            lines.extend(f"  - {e}" for e in compiled.errors)
        if problems:
            lines.append("")
            lines.append("Check before running:")
            lines.extend(f"  - {p}" for p in problems)
        elif not compiled.errors:
            lines.append("")
            lines.append("No concerns: the roles map cleanly and are a typical size.")
        if compiled.errors:
            return ToolResult.error("\n".join(lines))
        return ToolResult.text("\n".join(lines))

    def _report_rfdiffusion1(self, spec: DesignSpec, spec_ref: str, plan: SpecPlan) -> ToolResult:
        compiled = plan.rfdiffusion1
        assert compiled is not None
        lines = [
            f"Design spec {spec_ref or '(unsaved)'} on {spec.target_ref}: "
            f"{plan.mode} design for RFdiffusion.",
        ]
        lines.extend(f"  {line}" for line in compiled.summary)
        if compiled.contigs:
            lines.append(f"Contig: {compiled.contigs}")
        if compiled.hotspot_res:
            lines.append(f"hotspot_res: [{', '.join(compiled.hotspot_res)}] (author numbering)")
        if compiled.fixed_positions:
            lines.append(f"ProteinMPNN fixed positions: {', '.join(compiled.fixed_positions)}")
        problems = list(compiled.notes) + list(compiled.warnings)
        if compiled.errors:
            lines.append("")
            lines.append("Cannot run as it stands:")
            lines.extend(f"  - {e}" for e in compiled.errors)
        if problems:
            lines.append("")
            lines.append("Check before running:")
            lines.extend(f"  - {p}" for p in problems)
        if compiled.errors:
            return ToolResult.error("\n".join(lines))
        return ToolResult.text("\n".join(lines))


def _chain_of(selector: str) -> str:
    """The chain a ``sel('A/*/...')`` selector names."""
    return selector.removeprefix("sel('").split("/", 1)[0]
