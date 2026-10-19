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

"""Discover the OpenAI models an OpenAI API key can reach.

``GET /v1/models`` lists ids and creation times and nothing else: no family,
no limits, no capabilities. So the family comes from the id and the limits
from a small per-family table, and the dropdown gets the newest model of each
family the key can access. Rosalind, the life-sciences family, is listed only
when FEAT_ROSALIND is true.
"""

import datetime
import hashlib
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from .config import (
    OPENAI_BASE_URL,
    OPENAI_ID_PREFIX,
    OPENAI_KEY_FILE,
    PROVIDER_OPENAI,
    ModelConfig,
    family_rank,
    key_file_present,
    register_models,
)

logger = logging.getLogger(__name__)

CORE_FAMILIES = ("astra", "sol", "luna")
ROSALIND = "rosalind"

# gpt-6-astra, gpt-6.1-sol. Excludes dated snapshots, -pro and -chat-latest.
_CORE_ID = re.compile(r"^gpt-(?P<version>\d+(?:\.\d+)?)-(?P<family>astra|sol|luna)$")
_VERSION = re.compile(r"^gpt-(?P<version>\d+(?:\.\d+)?)-")
_SNAPSHOT_SUFFIX = re.compile(r"-(\d{4}-\d{2}-\d{2}|\d{6})$")

# Output cap per call. The models allow 128k, but the agent loop never needs
# that much in one turn and the context assembler reserves max_tokens out of
# the window on every call.
MAX_OUTPUT_TOKENS_CAP = 32000

# The current models report a 1.05M-token window. Budgeting against all of it
# would let a session grow to a million billed input tokens per call before
# the context assembler degrades anything, so the window is capped by default.
CONTEXT_CAP = int(os.getenv("AEVOM_OPENAI_CONTEXT_CAP", "200000"))

# Optional reasoning.effort override (low, medium, high, xhigh, max). Unset
# leaves the model default (medium). "none" and "minimal" are refused because
# Astra and Sol return a 400 for them.
_REJECTED_EFFORTS = {"none", "minimal"}
EFFORT = (os.getenv("AEVOM_OPENAI_EFFORT") or "").strip().lower() or None
if EFFORT in _REJECTED_EFFORTS:
    logger.warning(f"Ignoring AEVOM_OPENAI_EFFORT={EFFORT}: the current models reject it")
    EFFORT = None

REFRESH_TTL_S = float(os.getenv("AEVOM_OPENAI_MODELS_TTL", "600"))
# A failed listing is retried sooner than a successful one is refreshed, so a
# brief outage at boot does not hide the models for the whole TTL.
FAILURE_RETRY_S = 60.0

_lock = threading.Lock()
_last_refresh: float = 0.0
_last_ok: bool = False
_last_key: Optional[str] = None


def rosalind_enabled() -> bool:
    return os.getenv("FEAT_ROSALIND", "false").strip().lower() == "true"


def _read_key(key_file: str) -> str:
    return Path(key_file).read_text().strip()


def list_models(api_key: str, timeout: float = 10.0) -> list[dict[str, Any]]:
    """Every model the key can access. The endpoint is not paginated."""
    headers = {"Authorization": f"Bearer {api_key}"}
    with httpx.Client(base_url=OPENAI_BASE_URL, headers=headers, timeout=timeout) as client:
        response = client.get("/models")
        response.raise_for_status()
        return list(response.json().get("data", []))


def _shut_down(model: dict[str, Any]) -> bool:
    """Whether the model's announced shutdown date has passed."""
    value = model.get("shutdown_date")
    if not value:
        return False
    if isinstance(value, (int, float)):
        return value <= time.time()
    try:
        return datetime.date.fromisoformat(str(value)[:10]) <= datetime.date.today()
    except ValueError:
        return False


def family_of(model_id: str) -> Optional[str]:
    """The family a model id belongs to, or None if it is not offered."""
    match = _CORE_ID.match(model_id)
    if match:
        return match.group("family")
    if (
        ROSALIND in model_id
        and "preview" not in model_id
        and not _SNAPSHOT_SUFFIX.search(model_id)
    ):
        return ROSALIND
    return None


def _version(model_id: str) -> tuple[int, ...]:
    match = _VERSION.match(model_id)
    if not match:
        return ()
    return tuple(int(part) for part in match.group("version").split("."))


def latest_per_family(
    models: list[dict[str, Any]], include_rosalind: bool = False
) -> list[dict[str, Any]]:
    """The newest model of each offered family.

    Newest by ``created``, then by the version in the id. ``created`` decides
    for Rosalind, whose newest model carries no version (gpt-rosalind-research
    supersedes gpt-5.5-rosalind).
    """
    families = CORE_FAMILIES + ((ROSALIND,) if include_rosalind else ())
    newest: dict[str, dict[str, Any]] = {}
    for model in models:
        family = family_of(model.get("id", ""))
        if family not in families or _shut_down(model):
            continue
        key = (model.get("created", 0), _version(model["id"]))
        current = newest.get(family)
        if current is None or key > (current.get("created", 0), _version(current["id"])):
            newest[family] = model
    return list(newest.values())


def pretty_name_for(model_id: str) -> str:
    """gpt-6.1-sol -> GPT-6.1 Sol; gpt-rosalind-research -> GPT Rosalind Research."""
    parts = model_id.split("-")
    if parts and parts[0] == "gpt":
        head, rest = "GPT", parts[1:]
        if rest and re.fullmatch(r"\d+(?:\.\d+)?", rest[0]):
            head, rest = f"GPT-{rest[0]}", rest[1:]
        return " ".join([head] + [p.capitalize() for p in rest])
    return model_id


def to_model_config(model: dict[str, Any], key_file: str = OPENAI_KEY_FILE) -> ModelConfig:
    """Build a ModelConfig from one ``/v1/models`` entry."""
    return ModelConfig(
        id=f"{OPENAI_ID_PREFIX}{model['id']}",
        name=model["id"],
        base_url=OPENAI_BASE_URL,
        api_key_file=key_file,
        provider=PROVIDER_OPENAI,
        pretty_name=pretty_name_for(model["id"]),
        # Every offered family reasons, and reasoning items are replayed.
        supports_reasoning=True,
        retains_reasoning=True,
        # The reasoning models reject the parameter outright with a 400.
        supports_temperature=False,
        default_temperature=1.0,
        max_tokens=MAX_OUTPUT_TOKENS_CAP,
        context_window=CONTEXT_CAP,
        template_name="base",
        thinking_param="",
        effort=EFFORT,
    )


def discover(key_file: str = OPENAI_KEY_FILE) -> list[ModelConfig]:
    """ModelConfigs for the newest model of each family; empty on any failure."""
    if not key_file_present(key_file):
        return []
    try:
        models = list_models(_read_key(key_file))
    except (httpx.HTTPError, ValueError, OSError) as e:
        logger.warning(f"OpenAI model discovery failed: {e}")
        return []
    configs = sorted(
        (to_model_config(m, key_file) for m in latest_per_family(models, rosalind_enabled())),
        key=family_rank,
    )
    logger.info(f"Discovered OpenAI models: {', '.join(c.name for c in configs) or 'none'}")
    return configs


def refresh_openai_models(force: bool = False, key_file: str = OPENAI_KEY_FILE) -> None:
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
            register_models(PROVIDER_OPENAI, [])
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
            register_models(PROVIDER_OPENAI, configs)
        _last_key = key


def reset_discovery_state() -> None:
    """Forget the last refresh (tests)."""
    global _last_refresh, _last_ok, _last_key
    with _lock:
        _last_refresh, _last_ok, _last_key = 0.0, False, None
        register_models(PROVIDER_OPENAI, [])
