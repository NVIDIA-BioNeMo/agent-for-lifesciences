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

"""Session artifact seeding and submission read-back.

Two halves of one interface: a way to put files into a session before the
conversation starts, and a way to read out the artifact the agent nominated as
its answer when the conversation ends.
"""

from .routes import router as artifacts_router
from .submissions import (
    SUBMISSION_FLAG,
    build_submission_metadata,
    build_submission_record,
    is_submission,
)

__all__ = [
    "artifacts_router",
    "SUBMISSION_FLAG",
    "build_submission_metadata",
    "build_submission_record",
    "is_submission",
]
