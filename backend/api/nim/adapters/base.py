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

"""What an adapter is, and what it is handed.

A NIM announces a model, not a capability. ``BakerLab/rfd4-proteina`` says
nothing about contigs, hotspots or condition specs -- that knowledge lives in
Aevom, in an adapter keyed by the model id. Registering a NIM therefore means
recognising its model and attaching the tools we already know how to build
for it.

Two things an adapter can contribute, and it may do both:

* **New tools.** ``build_tools`` returns tool instances bound to this
  endpoint. This is how RFD4-Proteina contributes binder design.
* **Re-pointing.** ``serves_builtin_tools`` names built-in tools whose hosted
  NVIDIA endpoint this NIM can stand in for, so a self-hosted Boltz-2 serves
  the Boltz-2 tool the agent already has.

A NIM whose model matches no adapter still registers. It shows in the tool
panel marked "no adapter", which is a more useful answer than refusing the
URL: the user learns the endpoint is reachable and what model it serves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Optional

from ..client import NIMInfo

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from api.tools.base import BaseTool


@dataclass(frozen=True)
class NIMEndpoint:
    """A registered NIM, as an adapter sees it.

    Attributes:
        server_id: Aevom's id for this registration, unique per session.
        name: The user's name for it, shown in the tool panel.
        url: Base URL, no trailing slash.
        api_key: Bearer token, or None for an endpoint that wants no auth --
            which is the normal case for a self-hosted NIM.
        info: What the discovery probe found.
    """

    server_id: str
    name: str
    url: str
    api_key: Optional[str]
    info: NIMInfo

    def path(self, path: str) -> str:
        """Absolute URL for a path on this NIM."""
        return f"{self.url.rstrip('/')}/{path.lstrip('/')}"

    def supports_metric_tags(self, *tags: str) -> list[str]:
        """Filter tags down to the ones this deployment can compute.

        A deployment without an oracle or a structure database reports those
        tags unavailable, and asking for one anyway is a request error. Tags
        the NIM never mentioned are dropped too: an older build that does not
        serve ``/v1/tasks`` reports nothing, and sending speculative tags to
        it would turn a working call into a 422.
        """
        return [tag for tag in tags if self.info.metric_tags.get(tag)]


@dataclass(frozen=True)
class NIMAdapter:
    """Aevom's knowledge of what one model can be asked to do.

    Attributes:
        key: Stable identifier for the adapter, e.g. ``rfd4_proteina``.
        label: Human-readable model name for the UI.
        model_ids: Model ids this adapter claims, as reported by
            ``/v1/models``. Matched case-insensitively, and also against the
            bare name after the owner prefix, so a locally re-tagged
            ``rfd4-proteina`` still resolves to ``BakerLab/rfd4-proteina``.
        build_tools: Builds the tools this NIM contributes. Called with the
            endpoint; returns an empty list for an adapter that only
            re-points built-ins.
        serves_builtin_tools: Pairs of (built-in tool name, inference path on
            this NIM) for tools whose hosted NVIDIA endpoint this NIM can
            stand in for. The path is needed because the built-in tool holds
            a full URL, while registration only supplies a base.
        summary: One line shown under the model name in the UI.
    """

    key: str
    label: str
    model_ids: tuple[str, ...]
    build_tools: Callable[[NIMEndpoint], "list[BaseTool]"] = lambda _: []
    serves_builtin_tools: tuple[tuple[str, str], ...] = field(default=())
    summary: str = ""

    def matches(self, model_id: str) -> bool:
        """Whether this adapter claims the given model id."""
        candidate = _normalize(model_id)
        return any(
            candidate == _normalize(known)
            or candidate == _bare(_normalize(known))
            or _bare(candidate) == _bare(_normalize(known))
            for known in self.model_ids
        )

    def builtin_endpoints(self, endpoint: NIMEndpoint) -> dict[str, str]:
        """Full URLs to use for each built-in tool this NIM can serve.

        The declared path is a fallback. When the probe read an OpenAPI
        document, a discovered inference path wins: a NIM that moved or
        versioned its route still re-points correctly, and the declared path
        only has to be right for NIMs that serve no OpenAPI.
        """
        discovered = endpoint.info.inference_paths
        resolved: dict[str, str] = {}
        for tool_name, declared_path in self.serves_builtin_tools:
            path = declared_path
            if discovered and declared_path not in discovered:
                suffix = declared_path.rsplit("/", 1)[-1]
                match = next(
                    (p for p in discovered if p.rsplit("/", 1)[-1] == suffix), None
                )
                path = match or (discovered[0] if len(discovered) == 1 else declared_path)
            resolved[tool_name] = endpoint.path(path)
        return resolved


def _normalize(model_id: str) -> str:
    return (model_id or "").strip().lower()


def _bare(model_id: str) -> str:
    """The model name without its owner prefix or version suffix.

    ``bakerlab/rfd4-proteina:1.0.0`` becomes ``rfd4-proteina``.
    """
    return model_id.rsplit("/", 1)[-1].split(":", 1)[0]
