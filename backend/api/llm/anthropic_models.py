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

"""Discover the Claude models an Anthropic API key can reach.

``GET /v1/models`` reports, per model, its family (``line``), context and output
limits, and which thinking modes and effort levels it accepts -- everything a
ModelConfig needs. So nothing about individual Claude models is hardcoded: the
dropdown follows whatever the key can access, reduced to the newest model of
each family.
"""

import hashlib
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import (
    ANTHROPIC_BASE_URL,
    ANTHROPIC_ID_PREFIX,
    ANTHROPIC_KEY_FILE,
    PROVIDER_ANTHROPIC,
    ModelConfig,
    anthropic_line_rank,
    key_file_present,
    register_models,
)

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"

# Output cap per call. The API allows 128k on current models, but the agent
# loop never needs that much in one turn and the context assembler reserves
# max_tokens out of the window on every call.
MAX_OUTPUT_TOKENS_CAP = 32000

# The 5.x models report a 1M-token window. Budgeting against all of it would
# let a session grow to a million billed input tokens per call before the
# context assembler degrades anything, so the window is capped by default.
CONTEXT_CAP = int(os.getenv("AEVOM_ANTHROPIC_CONTEXT_CAP", "200000"))

# Optional output_config.effort override (low, medium, high, xhigh, max). Unset
# leaves the API default; a level a model does not support is dropped for it.
EFFORT = os.getenv("AEVOM_ANTHROPIC_EFFORT") or None

REFRESH_TTL_S = float(os.getenv("AEVOM_ANTHROPIC_MODELS_TTL", "600"))
# A failed listing is retried sooner than a successful one is refreshed, so a
# brief outage at boot does not hide the models for the whole TTL.
FAILURE_RETRY_S = 60.0

_lock = threading.Lock()
_last_refresh: float = 0.0
_last_ok: bool = False
_last_key: Optional[str] = None


def _read_key(key_file: str) -> str:
    return Path(key_file).read_text().strip()


def list_models(api_key: str, timeout: float = 10.0) -> list[dict[str, Any]]:
    """Every model the key can access, following pagination."""
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    params: dict[str, Any] = {"limit": 100}
    models: list[dict[str, Any]] = []
    with httpx.Client(base_url=ANTHROPIC_BASE_URL, headers=headers, timeout=timeout) as client:
        while True:
            response = client.get("/v1/models", params=params)
            response.raise_for_status()
            body = response.json()
            models.extend(body.get("data", []))
            if not body.get("has_more") or not body.get("last_id"):
                return models
            params["after_id"] = body["last_id"]


def _supported(capabilities: dict[str, Any], *path: str) -> bool:
    node: Any = capabilities
    for key in path:
        if not isinstance(node, dict):
            return False
        node = node.get(key)
    return isinstance(node, dict) and bool(node.get("supported"))


def latest_per_line(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The newest active model of each family, newest family first.

    ``created_at`` is ISO-8601 UTC throughout, so string order is time order.
    """
    newest: dict[str, dict[str, Any]] = {}
    for model in models:
        if model.get("lifecycle", "active") != "active":
            continue
        line = model.get("line") or model["id"]
        current = newest.get(line)
        if current is None or model.get("created_at", "") > current.get("created_at", ""):
            newest[line] = model
    return sorted(newest.values(), key=lambda m: m.get("created_at", ""), reverse=True)


def to_model_config(model: dict[str, Any], key_file: str = ANTHROPIC_KEY_FILE) -> ModelConfig:
    """Build a ModelConfig from one ``/v1/models`` entry."""
    capabilities = model.get("capabilities") or {}

    if _supported(capabilities, "thinking", "types", "adaptive"):
        thinking_mode: Optional[str] = "adaptive"
    elif _supported(capabilities, "thinking", "types", "enabled"):
        thinking_mode = "enabled"
    else:
        thinking_mode = None

    effort = None
    if EFFORT and _supported(capabilities, "effort", EFFORT):
        effort = EFFORT

    max_tokens = min(int(model.get("max_tokens") or MAX_OUTPUT_TOKENS_CAP), MAX_OUTPUT_TOKENS_CAP)
    context_window = min(int(model.get("max_input_tokens") or CONTEXT_CAP), CONTEXT_CAP)

    return ModelConfig(
        id=f"{ANTHROPIC_ID_PREFIX}{model['id']}",
        name=model["id"],
        base_url=ANTHROPIC_BASE_URL,
        api_key_file=key_file,
        provider=PROVIDER_ANTHROPIC,
        pretty_name=model.get("display_name") or model["id"],
        supports_reasoning=thinking_mode is not None,
        retains_reasoning=thinking_mode is not None,
        # Thinking requires the default temperature, and newer models reject
        # the parameter outright, so it is never sent.
        supports_temperature=False,
        default_temperature=1.0,
        max_tokens=max_tokens,
        context_window=context_window,
        template_name="base",
        thinking_param="",
        thinking_mode=thinking_mode,
        thinking_budget_tokens=min(8192, max_tokens // 2),
        effort=effort,
    )


def discover(key_file: str = ANTHROPIC_KEY_FILE) -> list[ModelConfig]:
    """ModelConfigs for the newest model of each family; empty on any failure."""
    if not key_file_present(key_file):
        return []
    try:
        models = list_models(_read_key(key_file))
    except (httpx.HTTPError, ValueError, OSError) as e:
        logger.warning(f"Anthropic model discovery failed: {e}")
        return []
    configs = sorted(
        (to_model_config(m, key_file) for m in latest_per_line(models)),
        key=anthropic_line_rank,
    )
    logger.info(f"Discovered Anthropic models: {', '.join(c.name for c in configs) or 'none'}")
    return configs


def refresh_anthropic_models(force: bool = False, key_file: str = ANTHROPIC_KEY_FILE) -> None:
    """Re-run discovery when the TTL has passed or the key has changed.

    A failed refresh keeps the previous list, so a transient outage does not
    empty the dropdown under a user who is mid-session.
    """
    global _last_refresh, _last_ok, _last_key

    with _lock:
        # A digest, not the key: only change detection needs it.
        key = (
            hashlib.sha256(_read_key(key_file).encode()).hexdigest()
            if key_file_present(key_file)
            else None
        )
        if key is None:
            register_models(PROVIDER_ANTHROPIC, [])
            _last_key, _last_ok, _last_refresh = None, False, time.monotonic()
            return

        age = time.monotonic() - _last_refresh
        ttl = REFRESH_TTL_S if _last_ok else FAILURE_RETRY_S
        if not force and key == _last_key and _last_refresh and age < ttl:
            return

        configs = discover(key_file)
        _last_refresh = time.monotonic()
        _last_ok = bool(configs)
        if configs or key != _last_key:
            register_models(PROVIDER_ANTHROPIC, configs)
        _last_key = key


def reset_discovery_state() -> None:
    """Forget the last refresh (tests)."""
    global _last_refresh, _last_ok, _last_key
    with _lock:
        _last_refresh, _last_ok, _last_key = 0.0, False, None
        register_models(PROVIDER_ANTHROPIC, [])
