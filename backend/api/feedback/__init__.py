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

"""Free-text user feedback, relayed to a configured inbox by email.

Gated by ``AEVOM_FEAT_FEEDBACK_EMAIL``. Nothing is persisted: the message is
handed to an SMTP relay and forgotten.
"""

from .config import FEEDBACK_ENABLED, MAX_MESSAGE_CHARS, MIN_MESSAGE_CHARS
from .routes import router as feedback_router

__all__ = [
    "FEEDBACK_ENABLED",
    "MAX_MESSAGE_CHARS",
    "MIN_MESSAGE_CHARS",
    "feedback_router",
]
