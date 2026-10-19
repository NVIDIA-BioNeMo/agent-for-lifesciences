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

"""Self-hosted Boltz-2: re-points the built-in tool, contributes nothing new.

Aevom already has a ``boltz2`` tool; a self-hosted Boltz-2 NIM runs the same
model behind the same request schema. So this adapter adds no tool and
instead declares that it can serve the built-in one, which the session store
turns into a per-session endpoint override.

This is the second reason the override exists. ``BOLTZ2_API_URL`` already
re-points Boltz-2 process-wide, but that is a deployment decision made before
the server starts; this is a user pointing one session at their own GPU.

**The model ids below are unverified.** Everything the RFD4 adapter claims was
read off a running container; this one was not, because no Boltz-2 NIM has
been available to probe. If a real deployment reports an id that is not in the
list, it registers cleanly, shows "no adapter" in the tool panel, and silently
does not re-point -- which looks like success until someone notices the local
GPU is idle. The panel does print the model id it saw, so the fix is to read
it off the screen and add it here. Do that rather than guessing again.
"""

from __future__ import annotations

from .base import NIMAdapter

# The hosted endpoint is /v1/biology/mit/boltz2/predict. The self-hosted
# container is assumed to drop the /v1 prefix, as the RFD4 NIM does with its
# own route -- one observation generalised to a second service, so also
# unverified. A NIM that serves an OpenAPI document corrects this by itself:
# builtin_endpoints() prefers a discovered route whose last segment matches.
_PREDICT_PATH = "/biology/mit/boltz2/predict"

ADAPTER = NIMAdapter(
    key="boltz2",
    label="Boltz-2",
    model_ids=("mit/boltz2", "boltz2", "boltz-2"),
    serves_builtin_tools=(("boltz2", _PREDICT_PATH),),
    summary=(
        "Co-folding and binding affinity. Replaces the hosted endpoint used by "
        "the built-in boltz2 tool for this session."
    ),
)
