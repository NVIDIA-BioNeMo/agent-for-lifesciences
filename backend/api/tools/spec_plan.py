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

"""Reading a design spec's artifacts and compiling them for an engine.

The glue between a stored :class:`DesignSpec` and :mod:`role_compile`: it
fetches the target and each layer's selection, picks the engine and the mode
(binder or motif), and returns the compiled plan together with the parsed
inputs the callers also need. Shared by the spec tools and the design tools so
that *validating* a spec and *running* it read the very same plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from api.selections.design_spec import DesignSpec, SpecError
from api.selections.role_compile import (
    MODE_BINDER,
    MODE_MOTIF,
    LayerInput,
    RFdiffusion1Plan,
    RolePlan,
    compile_rfd4,
    compile_rfdiffusion1,
)
from api.selections.roles import DEFAULT_ENGINE, ENGINE_RFD4, ENGINE_RFDIFFUSION1, ENGINES
from api.selections.structure import StructureAtom, parse_structure_atoms
from api.storage.mmcif_numbering import NumberingMap, build_numbering_map

Retrieve = Callable[[str, str], Awaitable[bytes]]


def looks_like_mmcif(text: str) -> bool:
    """Whether this is mmCIF rather than PDB, which RFD4-Proteina rejects."""
    return text.lstrip().startswith("data_")


@dataclass
class SpecPlan:
    """A spec, compiled for an engine."""

    spec: DesignSpec
    engine: str
    mode: str
    layers: list[LayerInput] = field(default_factory=list)
    target_cif: str = ""
    numbering: Optional[NumberingMap] = None
    target_atoms: list[StructureAtom] = field(default_factory=list)
    rfd4: Optional[RolePlan] = None
    rfdiffusion1: Optional[RFdiffusion1Plan] = None

    @property
    def errors(self) -> list[str]:
        plan = self.rfd4 or self.rfdiffusion1
        return list(plan.errors) if plan else []

    @property
    def warnings(self) -> list[str]:
        plan = self.rfd4 or self.rfdiffusion1
        return list(plan.warnings) if plan else []

    @property
    def notes(self) -> list[str]:
        plan = self.rfd4 or self.rfdiffusion1
        return list(plan.notes) if plan else []


def infer_mode(spec: DesignSpec) -> str:
    """Binder if the spec has a hotspot; motif if it only keeps a motif."""
    if spec.layer("hotspot") is None and spec.layer("keep") is not None:
        return MODE_MOTIF
    return MODE_BINDER


async def load_layer_inputs(spec: DesignSpec, retrieve: Retrieve) -> list[LayerInput]:
    """Fetch the selection each layer names.

    Raises:
        SpecError: If a layer's artifact is not a selection.
    """
    inputs: list[LayerInput] = []
    for layer in spec.layers:
        raw = await retrieve(layer.selection_ref, "selection")
        try:
            selection = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            selection = None
        if not isinstance(selection, dict) or "atoms" not in selection:
            raise SpecError(
                f"'{layer.selection_ref}' ({layer.condition}) is not a selection artifact."
            )
        inputs.append(LayerInput(layer=layer, selection=selection))
    return inputs


async def plan_spec(
    spec: DesignSpec,
    retrieve: Retrieve,
    *,
    mode: Optional[str] = None,
    engine: Optional[str] = None,
    binder_min: int = 60,
    binder_max: int = 120,
) -> SpecPlan:
    """Fetch a spec's inputs and compile it for ``engine``.

    Compilation problems are returned on the plan (``errors`` / ``warnings``)
    rather than raised, so a caller can show them all at once.

    Raises:
        SpecError: If an input is unreadable or the target is the wrong format
            for the engine.
    """
    engine = (engine or spec.engine or DEFAULT_ENGINE).lower()
    if engine not in ENGINES:
        raise SpecError(f"Unknown engine '{engine}'. Engines: {', '.join(ENGINES)}.")
    mode = mode or infer_mode(spec)
    if mode not in (MODE_BINDER, MODE_MOTIF):
        raise SpecError("mode must be 'binder' or 'motif'.")

    target_cif = (await retrieve(spec.target_ref, "target structure")).decode(
        "utf-8", errors="replace"
    )
    layers = await load_layer_inputs(spec, retrieve)
    plan = SpecPlan(spec=spec, engine=engine, mode=mode, layers=layers, target_cif=target_cif)
    plan.target_atoms = parse_structure_atoms(target_cif)
    mmcif = looks_like_mmcif(target_cif)
    if mmcif:
        plan.numbering = build_numbering_map(target_cif) or None

    if engine == ENGINE_RFD4:
        if not mmcif:
            raise SpecError(
                f"'{spec.target_ref}' is not mmCIF. RFD4-Proteina rejects PDB input; "
                "fetch the target as mmCIF."
            )
        if plan.numbering is None:
            raise SpecError(
                f"Could not read residue numbering from '{spec.target_ref}': the file "
                "has no usable _atom_site loop."
            )
        plan.rfd4 = compile_rfd4(
            layers,
            plan.numbering,
            mode=mode,
            binder_min=binder_min,
            binder_max=binder_max,
            layout=spec.layout,
        )
    else:
        plan.rfdiffusion1 = compile_rfdiffusion1(
            layers,
            plan.numbering,
            mode=mode,
            binder_min=binder_min,
            binder_max=binder_max,
            layout=spec.layout,
        )
    return plan


__all__ = [
    "ENGINE_RFD4",
    "ENGINE_RFDIFFUSION1",
    "SpecPlan",
    "infer_mode",
    "load_layer_inputs",
    "looks_like_mmcif",
    "plan_spec",
]
