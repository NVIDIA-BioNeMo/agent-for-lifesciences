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

"""Serve the React build, confined to its own directory.

The catch-all route has to be registered after every API router, or it would
shadow them.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_spa(app: FastAPI, static_path: Path) -> None:
    """Mount the SPA's assets and its index fallback on ``app``."""
    static_root = static_path.resolve()
    index = static_root / "index.html"

    assets = static_root / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    async def serve_spa(path: str):
        """Serve a file from the build, or index.html for client-side routes."""
        # An API path that no router claimed is a missing endpoint, not a page:
        # answering it with index.html and a 200 hides the mistake from clients.
        if path == "api" or path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")

        # Resolving before the containment check catches "..", absolute paths
        # (pathlib discards the root when joined with one) and symlinks out.
        candidate = (static_root / path).resolve()
        if candidate.is_relative_to(static_root) and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(index)
