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

"""Data storage module for tool artifacts.

This module provides storage capabilities for tools that need to pass
large data artifacts (like SDF files, PDB structures) between each other
without flowing the raw data through the LLM.

Example:
    from api.storage import format_storage_message, scoped_storage

    # In a tool that generates data:
    storage = scoped_storage(session, session_id)
    ref = await storage.store("sdf", sdf_bytes, {"smiles": smiles})
    message = format_storage_message("sdf", ref, smiles, ["visualize_3d"])

    # In a tool that consumes data:
    content = await storage.retrieve(ref)
"""

from .datatypes import (
    DATA_TYPE_INFO,
    DataType,
    DataTypeInfo,
    content_type_for,
    friendly_name,
    get_info,
    is_text_type,
)
from .memory import InMemoryDataArtifactRepository
from .repository import DataArtifactRepository
from .scoped import SessionScopedStorage, artifact_owner, scoped_storage
from .service import (
    ArtifactNotFoundError,
    DataStorageError,
    DataStorageService,
    InvalidReferenceError,
    format_storage_message,
    generate_reference_id,
)

__all__ = [
    "DataStorageService",
    "DataStorageError",
    "ArtifactNotFoundError",
    "InvalidReferenceError",
    "DataArtifactRepository",
    "InMemoryDataArtifactRepository",
    "SessionScopedStorage",
    "artifact_owner",
    "scoped_storage",
    "format_storage_message",
    "generate_reference_id",
    "DataType",
    "DataTypeInfo",
    "DATA_TYPE_INFO",
    "content_type_for",
    "friendly_name",
    "get_info",
    "is_text_type",
]
