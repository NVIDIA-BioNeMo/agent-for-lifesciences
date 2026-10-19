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

"""Discovery probe for a self-hosted NVIDIA NIM.

Unlike MCP, a NIM does not describe its own tools: it serves one model behind
a handful of fixed endpoints, and what you can usefully *do* with that model
is knowledge Aevom supplies (see ``api.nim.adapters``). So this client does
not "connect" in the MCP sense -- it asks a candidate URL who it is, and
returns enough for the adapter registry to recognise the model and for the UI
to show the user what they just pointed at.

Every endpoint but ``/v1/models`` is optional. NIMs differ in which of the
conventional routes they expose, and a NIM that answers ``/v1/models`` with a
model the registry recognises is usable even if it serves nothing else.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

from api.bounded_http import ResponseTooLargeError, format_limit, request_bounded

logger = logging.getLogger(__name__)

# Long enough to cross a loaded cluster, short enough that a wrong URL fails
# while the user is still looking at the dialog. Inference uses its own, far
# longer, timeout -- this is discovery only.
PROBE_TIMEOUT = 15.0

# Paths that answer "are you a NIM, and which one?". Ordered by how much they
# are relied on: the model list is required, the rest enrich the result.
_MODELS_PATH = "/v1/models"
_HEALTH_PATHS = ("/v1/health/ready", "/health/ready")
_METADATA_PATH = "/v1/metadata"
_TASKS_PATH = "/v1/tasks"
_OPENAPI_PATH = "/openapi.json"

# OpenAPI paths that are infrastructure rather than inference. Anything else
# accepting POST is taken to be the model's own endpoint.
_NON_INFERENCE_PREFIXES = ("/v1/health", "/health", "/v2/health", "/v1/metadata",
                           "/v1/license", "/v1/manifest", "/metrics")


class NIMConnectionError(Exception):
    """Raised when a NIM cannot be reached, or does not look like a NIM."""


class NIMAuthError(Exception):
    """Raised when a NIM rejects the supplied credentials (or lack of them)."""


@dataclass
class NIMInfo:
    """What a discovery probe learned about a NIM.

    Attributes:
        model_id: The model's id from ``/v1/models``, e.g.
            ``BakerLab/rfd4-proteina``. This is the key the adapter registry
            matches on, so it is the one field a probe must produce.
        model_version: Version string reported alongside the model.
        description: The model's self-description, shown in the UI.
        ready: Whether the NIM reported itself ready to serve.
        release: NIM release from ``/v1/metadata``, when exposed.
        tasks: Task names from ``/v1/tasks``. Empty for NIMs without the
            endpoint, which is most of them.
        metric_tags: Tag name -> whether this deployment can compute it.
            A deployment missing an oracle or a structure database reports
            its unavailable tags here, and asking for one anyway is an error,
            so adapters must filter their requests through this.
        inference_paths: POST paths from ``/openapi.json`` that look like
            inference rather than infrastructure.
    """

    model_id: str
    model_version: Optional[str] = None
    description: Optional[str] = None
    ready: bool = False
    release: Optional[str] = None
    tasks: list[str] = field(default_factory=list)
    metric_tags: dict[str, bool] = field(default_factory=dict)
    inference_paths: list[str] = field(default_factory=list)

    @property
    def available_metric_tags(self) -> list[str]:
        """Metric tags this deployment can actually compute."""
        return [tag for tag, available in self.metric_tags.items() if available]

    def to_dict(self) -> dict[str, Any]:
        """Serialize for Redis storage and for the API response."""
        return {
            "model_id": self.model_id,
            "model_version": self.model_version,
            "description": self.description,
            "ready": self.ready,
            "release": self.release,
            "tasks": list(self.tasks),
            "metric_tags": dict(self.metric_tags),
            "inference_paths": list(self.inference_paths),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NIMInfo":
        """Deserialize from Redis."""
        return cls(
            model_id=data["model_id"],
            model_version=data.get("model_version"),
            description=data.get("description"),
            ready=data.get("ready", False),
            release=data.get("release"),
            tasks=list(data.get("tasks") or []),
            metric_tags=dict(data.get("metric_tags") or {}),
            inference_paths=list(data.get("inference_paths") or []),
        )


class NIMClient:
    """Probes a candidate NIM endpoint and reports what it is.

    Example:
        client = NIMClient("http://localhost:8000")
        info = await client.probe()
        info.model_id  # "BakerLab/rfd4-proteina"
    """

    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        timeout: float = PROBE_TIMEOUT,
    ):
        """Initialize the probe client.

        Args:
            url: Base URL of the NIM, e.g. ``http://localhost:8000``.
            api_key: Optional bearer token. A self-hosted NIM usually needs
                none, so this stays unset in the common case and no
                ``Authorization`` header is sent at all.
            timeout: Per-request timeout in seconds.
        """
        self.url = url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def headers(self) -> dict[str, str]:
        """Request headers, carrying auth only when a key was supplied.

        Sending ``Authorization: Bearer None`` to a NIM that wants no auth is
        how the hosted-API client would behave; some NIMs reject it outright.
        """
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _get_json(self, client: httpx.AsyncClient, path: str) -> Optional[Any]:
        """GET one path, returning None for anything that is not a 2xx JSON body.

        Optional endpoints are the norm here, so a miss is not exceptional.
        Auth failures are, because they mean the whole probe is invalid rather
        than this one route being absent. So are redirects and oversized
        bodies: either would otherwise surface as a misleading "No NIM found".
        """
        try:
            response = await request_bounded(
                client, "GET", f"{self.url}{path}", headers=self.headers()
            )
        except ResponseTooLargeError as exc:
            raise NIMConnectionError(
                f"{self.url}{path} returned a response larger than Aevom's "
                f"{format_limit(exc.limit)} limit, so it was discarded."
            ) from exc
        except httpx.HTTPError as exc:
            logger.debug("NIM probe: %s%s failed: %s", self.url, path, exc)
            return None

        if response.is_redirect:
            raise NIMConnectionError(
                f"{self.url}{path} redirected to "
                f"{response.headers.get('location', 'another URL')}. Aevom does "
                "not follow redirects; register the final URL instead."
            )
        if response.status_code in (401, 403):
            raise NIMAuthError(
                f"{self.url} rejected the request to {path} "
                f"({response.status_code}). "
                + ("The API key may be wrong." if self.api_key
                   else "This endpoint requires an API key.")
            )
        if response.status_code >= 400:
            return None

        try:
            return response.json()
        except ValueError:
            return None

    async def probe(self) -> NIMInfo:
        """Interrogate the endpoint and describe the NIM behind it.

        Raises:
            NIMAuthError: The endpoint demanded credentials we do not have.
            NIMConnectionError: Unreachable, or reachable but not a NIM.
        """
        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=False) as client:
            models = await self._get_json(client, _MODELS_PATH)
            if models is None:
                raise NIMConnectionError(
                    f"No NIM found at {self.url}: {_MODELS_PATH} did not return "
                    "a model list. Check the URL and that the container is running."
                )

            model = _first_model(models)
            if model is None:
                raise NIMConnectionError(
                    f"{self.url} answered {_MODELS_PATH} but listed no models."
                )

            info = NIMInfo(
                model_id=str(model.get("id") or model.get("name") or "").strip(),
                model_version=_opt_str(model.get("version")),
                description=_opt_str(model.get("description")),
            )
            if not info.model_id:
                raise NIMConnectionError(
                    f"{self.url} listed a model with no id, so it cannot be "
                    "matched to an adapter."
                )

            info.ready = await self._probe_health(client)

            metadata = await self._get_json(client, _METADATA_PATH)
            if isinstance(metadata, dict):
                info.release = _opt_str(metadata.get("version"))

            tasks = await self._get_json(client, _TASKS_PATH)
            if isinstance(tasks, dict):
                info.tasks = [
                    str(task["name"])
                    for task in tasks.get("tasks") or []
                    if isinstance(task, dict) and task.get("name")
                ]
                info.metric_tags = {
                    str(tag): bool((spec or {}).get("available"))
                    for tag, spec in (tasks.get("metric_tags") or {}).items()
                    if isinstance(spec, dict)
                }

            openapi = await self._get_json(client, _OPENAPI_PATH)
            if isinstance(openapi, dict):
                info.inference_paths = _inference_paths(openapi)

            logger.info(
                "Probed NIM at %s: model=%s ready=%s tasks=%d paths=%s",
                self.url, info.model_id, info.ready, len(info.tasks),
                info.inference_paths,
            )
            return info

    async def _probe_health(self, client: httpx.AsyncClient) -> bool:
        """Whether the NIM says it is ready to serve.

        The readiness route has two spellings and two body shapes across NIM
        versions: this deployment answers ``/v1/health/ready`` with a bare
        ``true``, while the OpenAPI document describes an object. Reaching the
        route at all is the signal; the body only refines it.
        """
        for path in _HEALTH_PATHS:
            body = await self._get_json(client, path)
            if body is None:
                continue
            if isinstance(body, bool):
                return body
            if isinstance(body, dict):
                status = str(body.get("status", "")).lower()
                return status in ("", "ready", "ok", "healthy", "success")
            return True
        return False


def _first_model(models: Any) -> Optional[dict[str, Any]]:
    """Pull the single served model out of an OpenAI-shaped model list.

    A NIM serves exactly one model, so the first entry is the model.
    """
    if isinstance(models, dict):
        data = models.get("data")
        if isinstance(data, list) and data:
            first = data[0]
            return first if isinstance(first, dict) else None
        if models.get("id") or models.get("name"):
            return models
    if isinstance(models, list) and models and isinstance(models[0], dict):
        return models[0]
    return None


def _inference_paths(openapi: dict[str, Any]) -> list[str]:
    """POST paths from an OpenAPI document, minus the infrastructure routes."""
    paths = openapi.get("paths")
    if not isinstance(paths, dict):
        return []
    return [
        path
        for path, spec in paths.items()
        if isinstance(spec, dict)
        and "post" in spec
        and not path.startswith(_NON_INFERENCE_PREFIXES)
    ]


def _opt_str(value: Any) -> Optional[str]:
    """Coerce to a non-empty string, or None."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None
