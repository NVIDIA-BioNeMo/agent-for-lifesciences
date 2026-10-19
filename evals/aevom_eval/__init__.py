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

"""Harbor integration for evaluating Aevom against its production stack.

Layout
------

``driver/``
    Runs *inside* the Harbor environment's ``main`` container, on the same
    Compose network as the Aevom api. Stdlib only -- ``main`` is a bare
    ``python:3.12-slim`` and installing packages per trial would be slow and
    another source of drift.

``agent.py``, ``verifier.py``
    Run on the *host*, in the Harbor process. These import ``harbor``.

The split exists because the Aevom stack publishes no host ports: fixed ports
would collide as soon as ``--n-concurrent`` exceeds one, so all HTTP happens
from inside the environment and the host talks to it via ``environment.exec``.

Why this package is not called ``harbor``
-----------------------------------------

Harbor loads the agent by import path with the repo on ``sys.path``. A local
package named ``harbor`` shadows the real one, and the failure is confusing --
``import harbor`` silently resolves to an empty namespace package.
"""
