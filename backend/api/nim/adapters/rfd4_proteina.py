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

"""RFD4-Proteina: contributes the binder design tool.

There is no hosted NVIDIA equivalent of this model in Aevom, so this adapter
adds a tool rather than re-pointing one. The tool is bound to the endpoint
the user registered, and told which metrics that particular deployment can
compute -- a NIM without an oracle cannot be asked for refolding metrics, and
asking anyway is a request error rather than a silently skipped metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import NIMAdapter, NIMEndpoint

if TYPE_CHECKING:  # pragma: no cover - types only
    from api.tools.base import BaseTool


def _build_tools(endpoint: NIMEndpoint) -> "list[BaseTool]":
    """Build the binder design tool against this endpoint."""
    from api.tools.rfd4_proteina import (
        GENERATIVE_PATH,
        MOTIF_TASK,
        WANTED_METRIC_TAGS,
        DesignMotifScaffoldTool,
        DesignProteinBinderTool,
    )

    # The probe reports the route this build actually serves; the constant is
    # only the fallback for a NIM that exposed no OpenAPI document.
    path = next(
        (p for p in endpoint.info.inference_paths if "rfd4" in p or "proteina" in p),
        GENERATIVE_PATH,
    )

    binder = DesignProteinBinderTool(
        endpoint_url=endpoint.path(path),
        api_key=endpoint.api_key,
        metric_tags=endpoint.supports_metric_tags(*WANTED_METRIC_TAGS),
        server_name=endpoint.name,
    )
    tools: "list[BaseTool]" = [binder]
    # A NIM that reports its tasks is believed; one that reports none is given
    # the benefit of the doubt, as it is for the metric tags.
    tasks = endpoint.info.tasks
    if not tasks or MOTIF_TASK in tasks:
        tools.append(
            DesignMotifScaffoldTool(
                endpoint_url=endpoint.path(path),
                api_key=endpoint.api_key,
                server_name=endpoint.name,
            )
        )
    return tools


ADAPTER = NIMAdapter(
    key="rfd4_proteina",
    label="RFD4-Proteina",
    model_ids=("BakerLab/rfd4-proteina", "ipd/rfd4-proteina", "rfd4-proteina"),
    build_tools=_build_tools,
    summary=(
        "De novo protein design. Adds design_protein_binder, which designs "
        "binders against an epitope selected in the viewer, and "
        "design_motif_scaffold, which designs a protein around kept residues."
    ),
)
