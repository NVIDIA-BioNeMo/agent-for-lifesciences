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

"""The design spec: which selections steer a design, and how.

A selection is only a set of residues; it says nothing about what to do with
them. The same residues could be a hotspot (the binder should touch them), a
region to keep fixed, or the site a centroid is computed from. The spec is
where that is said, so the meaning of a set lives with the design rather than
being baked into the selection or guessed from its name.

A spec binds *selections by reference* to *conditions*, on one target:

    {"version": 1, "target_ref": "mmcif_ab12cd34",
     "layers": [{"condition": "hotspot", "selection_ref": "selection_9f8e7d6c"}]}

Like a selection it is immutable. Changing a layer makes a new spec that names
the old one as ``parent``, so a campaign's history of what was asked is
recoverable from the artifacts alone.

A layer's condition is one of the *roles* in :mod:`api.selections.roles`:
hotspot, keep, fix_position, fix_sequence, float, and the chemistry roles. The
roles live here, on the spec, rather than on the selection, so the same
residues can play different parts in different designs. Layers that carry a
parameter (a Float group, a Burial level) may appear more than once, once per
parameter value; the others once each.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .roles import (
    ENGINES,
    ROLE_ORDER,
    ROLES,
    VALUE_ENUM,
    VALUE_GROUP,
    VALUE_LAYOUT,
    get_role,
)

SPEC_VERSION = 1

#: Spec condition name -> the RFD4 condition it compiles to.
CONDITIONS: dict[str, str] = {key: ROLES[key].annotation for key in ROLE_ORDER}

#: The Float group a layer joins when none is named: all floated residues keep
#: their relative spacing, which is what scaffolding one motif wants.
DEFAULT_FLOAT_GROUP = "motif"


class SpecError(ValueError):
    """A spec that cannot be used, phrased for the model to act on."""


@dataclass(frozen=True)
class SpecLayer:
    """One condition and the selection that supplies its residues."""

    condition: str
    selection_ref: str
    #: Condition parameters beyond the residue set; unused by ``hotspot``.
    value: Any = None

    @property
    def key(self) -> tuple[str, str]:
        """What makes a layer unique: its condition, and for roles that carry a
        parameter (a Float group, a Burial level) that parameter too."""
        return layer_key(self.condition, self.value)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "condition": self.condition,
            "selection_ref": self.selection_ref,
        }
        if self.value is not None:
            out["value"] = self.value
        return out


def layer_key(condition: str, value: Any) -> tuple[str, str]:
    """The identity of a layer: ``(condition, parameter)``."""
    role = ROLES.get(condition)
    if role is not None and role.value_kind in (VALUE_ENUM, VALUE_GROUP):
        return condition, str(value or "").strip()
    return condition, ""


@dataclass(frozen=True)
class Layout:
    """How long the designed pieces around the kept segments are.

    Not a role: it names no residues. ``n_term`` and ``c_term`` are optional
    ``(min, max)`` lengths before the first and after the last kept segment;
    ``linkers`` are the lengths between consecutive kept segments, in order.
    A missing linker takes ``DEFAULT_LINKER``.
    """

    n_term: Optional[tuple[int, int]] = None
    c_term: Optional[tuple[int, int]] = None
    linkers: tuple[tuple[int, int], ...] = ()

    def linker(self, index: int) -> tuple[int, int]:
        return self.linkers[index] if index < len(self.linkers) else DEFAULT_LINKER

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.n_term:
            out["n_term"] = list(self.n_term)
        if self.c_term:
            out["c_term"] = list(self.c_term)
        if self.linkers:
            out["linkers"] = [list(l) for l in self.linkers]
        return out


DEFAULT_LINKER = (10, 30)
MAX_PIECE_LENGTH = 200


def _length_range(raw: Any, what: str) -> tuple[int, int]:
    """A ``[min, max]`` (or a single number) as a checked pair."""
    try:
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            low = high = int(raw)
        else:
            low, high = (int(raw[0]), int(raw[1]))
    except (TypeError, ValueError, IndexError):
        raise SpecError(f"{what} must be a length or a [min, max] pair of whole numbers.")
    if not 0 <= low <= high <= MAX_PIECE_LENGTH:
        raise SpecError(
            f"{what} must satisfy 0 <= min <= max <= {MAX_PIECE_LENGTH}; got {low}-{high}."
        )
    return low, high


def parse_layout(raw: Any) -> Layout:
    """Read a layout, which is whatever the Keep layer's ``value`` carries.

    Raises:
        SpecError: If a length is not a sensible range.
    """
    if raw in (None, "", {}):
        return Layout()
    if not isinstance(raw, dict):
        raise SpecError("A Keep layer's value is a layout: {n_term, linkers, c_term}.")
    n_term = _length_range(raw["n_term"], "n_term") if raw.get("n_term") else None
    c_term = _length_range(raw["c_term"], "c_term") if raw.get("c_term") else None
    linkers = tuple(
        _length_range(item, f"linkers[{i}]") for i, item in enumerate(raw.get("linkers") or [])
    )
    return Layout(n_term=n_term, c_term=c_term, linkers=linkers)


@dataclass
class DesignSpec:
    """Selections bound to design conditions, on one target structure."""

    target_ref: str
    layers: list[SpecLayer] = field(default_factory=list)
    name: Optional[str] = None
    parent: Optional[str] = None
    #: The engine the spec is meant for; absent means RFD4.
    engine: Optional[str] = None
    version: int = SPEC_VERSION

    def layer(self, condition: str) -> Optional[SpecLayer]:
        """The first layer for ``condition``, or None."""
        return next((l for l in self.layers if l.condition == condition), None)

    def layers_for(self, condition: str) -> list[SpecLayer]:
        """Every layer for ``condition`` (several for Float groups, enums)."""
        return [l for l in self.layers if l.condition == condition]

    @property
    def layout(self) -> Layout:
        """The segment layout carried on the Keep layer (empty if none)."""
        keep = self.layer("keep")
        return parse_layout(keep.value) if keep is not None else Layout()

    def with_layers(self, layers: list[SpecLayer]) -> "DesignSpec":
        """A copy whose layers are replaced or added, matched by identity.

        A layer replaces the one with the same condition and parameter; every
        other layer is kept, so an edit to one never silently drops another.
        """
        replaced = {l.key for l in layers}
        kept = [l for l in self.layers if l.key not in replaced]
        return DesignSpec(
            target_ref=self.target_ref,
            layers=kept + list(layers),
            name=self.name,
            parent=self.parent,
            engine=self.engine,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "version": self.version,
            "target_ref": self.target_ref,
            "layers": [l.to_dict() for l in self.layers],
        }
        if self.name:
            out["name"] = self.name
        if self.parent:
            out["parent"] = self.parent
        if self.engine:
            out["engine"] = self.engine
        return out

    def to_json(self) -> bytes:
        return json.dumps(self.to_dict(), indent=2).encode("utf-8")


def parse_layer(raw: Any) -> SpecLayer:
    """Read one layer, accepting the condition by its role key or RFD4 name.

    Raises:
        SpecError: If it is not an object, names an unknown role, has no
            selection reference, or carries a parameter the role does not take.
    """
    if not isinstance(raw, dict):
        raise SpecError("Each layer must be an object with 'condition' and 'selection_ref'.")
    # The viewer calls these "roles", so a model reading the user's words will
    # often write ``role``. Accept it; ``condition`` wins if both are given.
    named = raw.get("condition") or raw.get("role")
    if not named:
        raise SpecError(
            "Each layer needs a 'condition' (also accepted as 'role'). "
            f"Roles: {', '.join(ROLE_ORDER)}."
        )
    role = get_role(named)
    if role is None:
        raise SpecError(f"Unknown condition '{named}'. Roles: {', '.join(ROLE_ORDER)}.")
    selection_ref = str(raw.get("selection_ref") or "").strip()
    if not selection_ref:
        raise SpecError(f"The '{role.key}' layer needs a selection_ref.")

    value = raw.get("value")
    if role.value_kind == VALUE_ENUM:
        variant = role.variant(value)
        if variant is None:
            allowed = ", ".join(v.value for v in role.variants)
            raise SpecError(f"The '{role.key}' layer needs a value, one of: {allowed}.")
        value = variant.value
    elif role.value_kind == VALUE_GROUP:
        value = str(value or DEFAULT_FLOAT_GROUP).strip() or DEFAULT_FLOAT_GROUP
    elif role.value_kind == VALUE_LAYOUT:
        parse_layout(value)  # raises SpecError on nonsense
        value = value or None
    else:
        value = None
    return SpecLayer(condition=role.key, selection_ref=selection_ref, value=value)


def parse_design_spec(raw: bytes | str | dict[str, Any]) -> DesignSpec:
    """Read a stored spec.

    Raises:
        SpecError: If the content is not a spec this version understands.
    """
    data = raw
    if isinstance(raw, (bytes, bytearray)):
        raw = bytes(raw).decode("utf-8", errors="replace")
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            raise SpecError("That reference is not a design spec.")
    if not isinstance(data, dict) or "layers" not in data or "target_ref" not in data:
        raise SpecError("That reference is not a design spec.")
    if int(data.get("version", SPEC_VERSION)) > SPEC_VERSION:
        raise SpecError(
            f"This spec is version {data['version']}; this server reads up to "
            f"version {SPEC_VERSION}."
        )
    layers = [parse_layer(l) for l in data.get("layers") or []]
    seen: set[tuple[str, str]] = set()
    for layer in layers:
        if layer.key in seen:
            what = layer.condition + (f" ({layer.key[1]})" if layer.key[1] else "")
            raise SpecError(f"The '{what}' condition appears more than once.")
        seen.add(layer.key)
    engine = str(data.get("engine") or "").strip().lower() or None
    if engine is not None and engine not in ENGINES:
        raise SpecError(f"Unknown engine '{engine}'. Engines: {', '.join(ENGINES)}.")
    return DesignSpec(
        target_ref=str(data["target_ref"]),
        layers=layers,
        name=data.get("name") or None,
        parent=data.get("parent") or None,
        engine=engine,
    )


def wrap_hotspot(target_ref: str, selection_ref: str) -> DesignSpec:
    """The one-layer spec a bare ``hotspot_selection_ref`` stands for."""
    return DesignSpec(
        target_ref=target_ref,
        layers=[SpecLayer("hotspot", selection_ref)],
    )
