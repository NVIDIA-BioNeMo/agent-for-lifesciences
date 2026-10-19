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

"""Model configuration for LLM integration.

This module defines available models and their configurations,
enabling easy model switching via environment variables.
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# NVIDIA API base URL (OpenAI-compatible)
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Anthropic Messages API. Not OpenAI-compatible in the way that matters: the
# compatibility endpoint rejects adaptive thinking, which is the only thinking
# mode the 5.x models accept, so these models go through AnthropicLLMClient.
ANTHROPIC_BASE_URL = "https://api.anthropic.com"

# OpenAI Responses API. Chat Completions refuses function tools on the current
# reasoning models, so these go through OpenAIResponsesClient, not LLMClient.
OPENAI_BASE_URL = "https://api.openai.com/v1"

# Mounted key files, one per provider. Every one of them is optional; a missing
# or empty file means the provider is not configured, and at least one LLM key
# must be present for the backend to start (see require_llm_key()).
NVIDIA_KEY_FILE = "/run/secrets/nvidia_api_key"
NVINF_KEY_FILE = "/run/secrets/nvidia_api_key_nvinf"
ANTHROPIC_KEY_FILE = "/run/secrets/anthropic_api_key"
OPENAI_KEY_FILE = "/run/secrets/openai_api_key"
LLM_KEY_FILES = (NVIDIA_KEY_FILE, NVINF_KEY_FILE, ANTHROPIC_KEY_FILE, OPENAI_KEY_FILE)

PROVIDER_OPENAI_COMPAT = "openai_compat"
PROVIDER_ANTHROPIC = "anthropic"
PROVIDER_OPENAI = "openai"

# Prefixes of discovered model ids. They keep discovered models apart from the
# static registry -- "claude-opus-4-6" and "gpt-6-astra" there are
# inference-api proxies of the same models.
ANTHROPIC_ID_PREFIX = "anthropic/"
OPENAI_ID_PREFIX = "openai/"


def key_file_present(path: str) -> bool:
    """Whether ``path`` is a regular file holding a non-blank key.

    Empty counts as absent: preflight creates empty placeholders so that the
    compose bind mounts (``create_host_path: false``) succeed for providers the
    user has not configured, and k8s optional secrets leave no file at all.
    """
    try:
        p = Path(path)
        return p.is_file() and bool(p.read_text().strip())
    except OSError:
        return False


@dataclass
class ModelConfig:
    """Configuration for an LLM model.

    Attributes:
        id: Short identifier for the model (used in env vars).
        name: Full model path for the API (e.g., 'nvidia/nemotron-3-nano-30b-a3b').
        base_url: Base URL for the API endpoint.
        api_key_file: Path to the API key file (allows per-model keys).
        supports_reasoning: Whether model supports reasoning/thinking mode.
        default_temperature: Default temperature for generation.
        supports_temperature: Whether the endpoint accepts a ``temperature`` at
            all. Some proxied models reject the parameter outright with a 400
            rather than clamping it, so when this is ``False`` the parameter is
            omitted from the request instead of being pinned to a value.
        top_p: Top-p (nucleus) sampling parameter.
        presence_penalty: Penalty for token repetition (0-2 typical for Qwen3).
        max_tokens: Maximum tokens for generation.
        context_window: Total prompt+generation token capacity. Deliberately a
            conservative floor rather than a spec sheet: it only drives the
            budget warning, and under-stating it warns early while over-stating
            it warns too late (or never). ``context.tokens.estimate_error``
            telemetry is what calibrates these.
        template_name: Name of the Jinja2 template for this model.
        thinking_param: Parameter name for enabling thinking mode in extra_body.
        thinking_level: Thinking level for Gemini 3 ("low", "medium", "high").
        requires_thought_signatures: Whether model requires thought signature preservation (Gemini 3).
        content_has_think_tags: Whether model embeds reasoning in content with </think> delimiter.
        retains_reasoning: Whether the model's own reasoning is sent back to it
            on subsequent turns. Training already does this -- the Responses API
            loop appends the whole output array, reasoning items included -- so
            leaving it off in production means the deployed policy sees less
            than the trained one. Enable per model only where the reasoning
            round-trips cleanly.
        extra_body: Additional parameters to pass in request body.
        provider: Which client speaks to this model: ``openai_compat``
            (LLMClient, chat completions) or ``anthropic`` (AnthropicLLMClient,
            native Messages API).
        pretty_name: UI label. Static entries use MODEL_PRETTY_NAMES instead;
            discovered entries carry the provider's display name here.
        thinking_mode: Anthropic only. ``adaptive`` (the model decides when
            and how much to think, depth set by ``effort``), ``enabled`` (a
            fixed ``thinking_budget_tokens``; the only mode on the 4.5
            generation), or None for no thinking.
        thinking_budget_tokens: Anthropic only, for ``thinking_mode="enabled"``.
        effort: Anthropic only. ``output_config.effort``; None leaves the API
            default.
    """

    id: str
    name: str
    base_url: str = NVIDIA_BASE_URL
    api_key_file: str = NVIDIA_KEY_FILE
    supports_reasoning: bool = False
    default_temperature: float = 0.15
    supports_temperature: bool = True
    top_p: float = 0.95
    presence_penalty: Optional[float] = None
    max_tokens: int = 16384
    context_window: int = 131072
    template_name: str = "base"
    thinking_param: str = "enable_thinking"  # Some models use "thinking" instead
    thinking_level: Optional[str] = None  # "low", "medium", "high" for Gemini 3
    requires_thought_signatures: bool = False  # True for Gemini 3 strict validation
    content_has_think_tags: bool = False  # True for models that embed reasoning in content with </think>
    retains_reasoning: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    provider: str = PROVIDER_OPENAI_COMPAT
    pretty_name: Optional[str] = None
    thinking_mode: Optional[str] = None
    thinking_budget_tokens: int = 8192
    effort: Optional[str] = None

    def get_extra_body(self) -> dict[str, Any]:
        """Get extra_body with reasoning enabled if supported."""
        body = dict(self.extra_body)
        
        # Add thinking mode for models that need an explicit param to enable it.
        # Models that reason by default (e.g. Nemotron Super) set thinking_param=""
        # to skip this injection.
        if self.supports_reasoning and self.thinking_param:
            existing = body.get("chat_template_kwargs", {})
            body["chat_template_kwargs"] = {**existing, self.thinking_param: True}
        
        # Add thinking_level for Gemini 3 models
        if self.thinking_level:
            body["thinking_level"] = self.thinking_level
        
        return body


# Registry of available models
MODELS: dict[str, ModelConfig] = {
    # Nemotron 3 Super 120B on build.nvidia.com. Distinct from the two Super
    # entries further down, which are the same family served by the internal
    # inference-api, hence the size in the id rather than a bare "super".
    "nemotron-3-super-120b": ModelConfig(
        id="nemotron-3-super-120b",
        name="nvidia/nemotron-3-super-120b-a12b",
        supports_reasoning=True,
        default_temperature=0.7,
        top_p=0.95,
        max_tokens=16384,
        context_window=131072,
        template_name="base",
        thinking_param="",  # Reasons by default; no extra_body param needed
        retains_reasoning=True,
    ),
    # Nemotron 3 Nano Omni 30B - the reasoning variant, on build.nvidia.com
    "nemotron-3-nano-omni": ModelConfig(
        id="nemotron-3-nano-omni",
        name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        supports_reasoning=True,
        default_temperature=0.7,
        top_p=0.95,
        max_tokens=16384,
        context_window=131072,
        template_name="base",
        thinking_param="",  # Reasons by default; no extra_body param needed
        retains_reasoning=True,
    ),
    # Nemotron 3.5 Super VL 120B on build.nvidia.com. The compose default: one of
    # the quickest public entries, answering tool calls in about a second.
    # Reasons by default via reasoning_content.
    "nemotron-3-5-super-vl": ModelConfig(
        id="nemotron-3-5-super-vl",
        name="nvidia/nemotron-3.5-super-vl-120b-a12b",
        supports_reasoning=True,
        default_temperature=0.7,
        top_p=0.95,
        max_tokens=16384,
        context_window=131072,
        template_name="base",
        thinking_param="",  # Reasons by default; no extra_body param needed
        retains_reasoning=True,
    ),
    # Nemotron 3.5 Lightning 30B on build.nvidia.com. Despite the name, the
    # slowest public entry when it was added: 76-153s per call, with the first
    # streamed token at 128s even on a one-line prompt.
    "nemotron-3-5-lightning": ModelConfig(
        id="nemotron-3-5-lightning",
        name="nvidia/nemotron-3.5-lightning-30b-a3b",
        supports_reasoning=True,
        default_temperature=0.7,
        top_p=0.95,
        max_tokens=16384,
        context_window=131072,
        template_name="base",
        thinking_param="",  # Reasons by default; no extra_body param needed
        retains_reasoning=True,
    ),
    # Muse Glimmer 30B on build.nvidia.com. Reasons by default. Do not give it a
    # thinking_param: a request carrying chat_template_kwargs.enable_thinking
    # hung for 180s without a byte, while the same request without it answered
    # in under 10s.
    "muse-glimmer": ModelConfig(
        id="muse-glimmer",
        name="meta/muse-glimmer-30b",
        supports_reasoning=True,
        default_temperature=0.7,
        top_p=0.95,
        max_tokens=16384,
        context_window=131072,
        template_name="base",
        thinking_param="",
        retains_reasoning=True,
    ),
    # GPT-OSS 20B on build.nvidia.com. Returns its reasoning twice, as both
    # `reasoning` and `reasoning_content`; the client reads the latter.
    "gpt-oss-20b": ModelConfig(
        id="gpt-oss-20b",
        name="openai/gpt-oss-20b",
        supports_reasoning=True,
        default_temperature=1.0,
        top_p=1.0,
        max_tokens=16384,
        context_window=131072,
        template_name="base",
        thinking_param="",  # Reasons by default; no extra_body param needed
        retains_reasoning=True,
    ),
    # GLM-5.3 on build.nvidia.com. Reasoning cannot be switched off:
    # enable_thinking=False still returns reasoning_content.
    "glm-5-3": ModelConfig(
        id="glm-5-3",
        name="z-ai/glm-5.3",
        supports_reasoning=True,
        default_temperature=1.0,
        top_p=1.0,
        max_tokens=16384,
        context_window=131072,
        template_name="base",
        thinking_param="",
        retains_reasoning=True,
    ),
    # GLM-5.3 Flash on build.nvidia.com. Same reasoning behaviour as GLM-5.3.
    "glm-5-3-flash": ModelConfig(
        id="glm-5-3-flash",
        name="z-ai/glm-5.3-flash",
        supports_reasoning=True,
        default_temperature=1.0,
        top_p=1.0,
        max_tokens=16384,
        context_window=131072,
        template_name="base",
        thinking_param="",
        retains_reasoning=True,
    ),
    # Gemini 3 Pro - via NVIDIA inference API with thought signature support
    "gemini-3-pro": ModelConfig(
        id="gemini-3-pro",
        name="gcp/google/gemini-3-pro",
        base_url="https://inference-api.nvidia.com/v1",
        api_key_file="/run/secrets/nvidia_api_key_nvinf",
        supports_reasoning=False,  # No readable reasoning_content (encoded in thought_signatures)
        default_temperature=1.0,  # LOCKED at 1.0 for reasoning engine stability
        top_p=0.95,
        max_tokens=8192,
        context_window=1048576,
        template_name="base",
        thinking_level="medium",  # Default thinking level for Gemini 3
        requires_thought_signatures=True,  # Strict validation - must preserve signatures
    ),
    # Claude Opus 4.6 - via NVIDIA inference API (Bedrock proxy), no reasoning via OpenAI SDK
    "claude-opus-4-6": ModelConfig(
        id="claude-opus-4-6",
        name="us/aws/anthropic/bedrock-claude-opus-4-6-v1",
        base_url="https://inference-api.nvidia.com/v1",
        api_key_file="/run/secrets/nvidia_api_key_nvinf",
        supports_reasoning=False,
        default_temperature=0.7,
        top_p=0.95,
        max_tokens=8192,
        context_window=200000,
        template_name="base",
    ),
    # GPT 5.1 - via NVIDIA inference API
    "gpt-5-1": ModelConfig(
        id="gpt-5-1",
        name="openai/openai/gpt-5.1",
        base_url="https://inference-api.nvidia.com/v1",
        api_key_file="/run/secrets/nvidia_api_key_nvinf",
        supports_reasoning=False,
        default_temperature=0.9,
        top_p=0.95,
        max_tokens=8192,
        context_window=272000,
        template_name="base",
    ),
    # GPT-6 Astra - via NVIDIA inference API (Azure proxy).
    # Rejects `temperature` at any value but 1.0 and `top_p` at any value, so the
    # parameter is omitted rather than pinned. Reasons internally (reasoning_tokens
    # > 0) but returns empty reasoning_content, so nothing is displayable.
    "gpt-6-astra": ModelConfig(
        id="gpt-6-astra",
        name="azure/openai/gpt-6-astra",
        base_url="https://inference-api.nvidia.com/v1",
        api_key_file="/run/secrets/nvidia_api_key_nvinf",
        supports_reasoning=False,
        supports_temperature=False,
        default_temperature=1.0,
        max_tokens=16384,
        context_window=272000,
        template_name="base",
    ),
    # Nemotron 3 Super Dynamo - via NVIDIA inference API
    # Model returns reasoning via standard reasoning_content field (reasons by default)
    "nemotron-3-super-dynamo": ModelConfig(
        id="nemotron-3-super-dynamo",
        name="nvidia/nvidia/nemotron-3-super-preview",
        base_url="https://inference-api.nvidia.com/v1",
        api_key_file="/run/secrets/nvidia_api_key_nvinf",
        supports_reasoning=True,
        default_temperature=0.7,
        top_p=0.95,
        max_tokens=1024,
        context_window=131072,
        template_name="base",
        thinking_param="",  # Reasons by default; no extra_body param needed
        retains_reasoning=True,
    ),
    # Nemotron 3 Super V3 (GA) - via NVIDIA inference API
    # Model returns reasoning via standard reasoning_content field (reasons by default)
    "nemotron-3-super-rc-nim": ModelConfig(
        id="nemotron-3-super-rc-nim",
        name="nvidia/nvidia/nemotron-3-super-v3",
        base_url="https://inference-api.nvidia.com/v1",
        api_key_file="/run/secrets/nvidia_api_key_nvinf",
        supports_reasoning=True,
        default_temperature=0.7,
        top_p=0.95,
        max_tokens=32768,
        context_window=131072,
        template_name="base",
        thinking_param="",  # Reasons by default; no extra_body param needed
        retains_reasoning=True,
    ),
    # Nemotron 3 Ultra on build.nvidia.com. The Helm chart default, and the only
    # Ultra an external user can reach: inference-api.nvidia.com is NVIDIA-internal,
    # integrate.api.nvidia.com is public. Not interchangeable with the -nvinf
    # entry below -- this endpoint takes thinking switched on explicitly, so the
    # two sample differently and their eval scores are not comparable.
    #
    # No reasoning_budget. This endpoint rejects the parameter by failing the
    # request, and dresses the refusal up as a capacity problem: every call
    # carrying it came back with an in-stream 503 "Service temporarily
    # overloaded" (0/6), against 5/6 for the same request without it. Retrying
    # cannot help because nothing is actually overloaded -- prompt size makes no
    # difference either, a 47k-token request succeeded 6/6 once the budget was
    # dropped. The sibling nemotron-3-super-120b accepts it fine, so this is
    # specific to the Ultra deployment.
    "nemotron-3-ultra": ModelConfig(
        id="nemotron-3-ultra",
        name="nvidia/nemotron-3-ultra-550b-a55b",
        # base_url and api_key_file default to integrate.api + nvidia_api_key
        supports_reasoning=True,
        default_temperature=1.0,
        top_p=0.95,
        max_tokens=16384,
        context_window=131072,
        template_name="base",
        thinking_param="enable_thinking",
        retains_reasoning=True,
    ),
    # Nemotron 3 Ultra (GA) - via NVIDIA inference API (internal only)
    # Model returns reasoning via standard reasoning_content field (reasons by default)
    "nemotron-3-ultra-nvinf": ModelConfig(
        id="nemotron-3-ultra-nvinf",
        name="nvidia/nvidia/nemotron-3-ultra",
        base_url="https://inference-api.nvidia.com/v1",
        api_key_file="/run/secrets/nvidia_api_key_nvinf",
        supports_reasoning=True,
        default_temperature=0.7,
        top_p=0.95,
        max_tokens=32768,
        context_window=131072,
        template_name="base",
        thinking_param="",  # Reasons by default; no extra_body param needed
        retains_reasoning=True,
    ),
}

# Default model if none specified (matches the docker-compose AEVOM_MODEL
# default, and is what preflight validates when nothing sets AEVOM_MODEL).
# Deliberately a public build.nvidia.com model, so a checkout with a single
# build.nvidia.com key runs without any model configuration. The Helm chart
# and k8s base set nemotron-3-ultra explicitly, and the internal cluster and
# the eval stack pin nemotron-3-ultra-nvinf.
DEFAULT_MODEL_ID = "nemotron-3-5-super-vl"

# Pretty-printed display names for the UI dropdown
MODEL_PRETTY_NAMES: dict[str, str] = {
    "nemotron-3-super-120b": "Nemotron 3 Super 120B",
    "nemotron-3-nano-omni": "Nemotron 3 Nano Omni 30B",
    "nemotron-3-5-super-vl": "Nemotron 3.5 Super VL 120B",
    "nemotron-3-5-lightning": "Nemotron 3.5 Lightning 30B",
    "muse-glimmer": "Muse Glimmer 30B",
    "gpt-oss-20b": "GPT-OSS 20B",
    "glm-5-3": "GLM-5.3",
    "glm-5-3-flash": "GLM-5.3 Flash",
    "gemini-3-pro": "Gemini 3 Pro",
    "claude-opus-4-6": "Claude Opus 4.6",
    "gpt-5-1": "GPT 5.1",
    "gpt-6-astra": "GPT-6 Astra",
    "nemotron-3-super-dynamo": "Nemotron 3 Super Dynamo",
    "nemotron-3-super-rc-nim": "Nemotron 3 Super V3",
    "nemotron-3-ultra": "Nemotron 3 Ultra",
    "nemotron-3-ultra-nvinf": "Nemotron 3 Ultra (inference-api)",
}

MODELS_HIDDEN_FROM_UI: set[str] = {
    "claude-opus-4-6",
}


# Models found at runtime, keyed by id. Kept out of MODELS so the static
# registry stays exactly what the source declares -- preflight and CI read it on
# the host, where discovery never runs.
_DISCOVERED_MODELS: dict[str, ModelConfig] = {}

# Family preference per discovered provider: the dropdown order, and the order
# a fallback picks in.
_FAMILY_ORDER = {
    PROVIDER_ANTHROPIC: ("opus", "sonnet", "fable", "haiku"),
    PROVIDER_OPENAI: ("astra", "sol", "luna", "rosalind"),
}


def family_rank(config: ModelConfig) -> int:
    """Position of a discovered model's family in its provider's order."""
    order = _FAMILY_ORDER.get(config.provider, ())
    for i, family in enumerate(order):
        if f"-{family}-" in config.name or config.name.endswith(f"-{family}"):
            return i
    return len(order)


def anthropic_line_rank(config: ModelConfig) -> int:
    """Position of a Claude model's family in the Anthropic order."""
    order = _FAMILY_ORDER[PROVIDER_ANTHROPIC]
    for i, line in enumerate(order):
        if f"-{line}-" in config.name or config.name.endswith(f"-{line}"):
            return i
    return len(order)


def register_models(provider: str, configs: list[ModelConfig]) -> None:
    """Replace every discovered model of ``provider`` with ``configs``."""
    for model_id in [k for k, v in _DISCOVERED_MODELS.items() if v.provider == provider]:
        del _DISCOVERED_MODELS[model_id]
    for config in configs:
        _DISCOVERED_MODELS[config.id] = config


def all_models() -> dict[str, ModelConfig]:
    """Static plus discovered models, static entries first."""
    return {**MODELS, **_DISCOVERED_MODELS}


def available_models() -> list[ModelConfig]:
    """Models whose key file is present, hidden ones excluded."""
    return [
        config
        for model_id, config in all_models().items()
        if model_id not in MODELS_HIDDEN_FROM_UI and key_file_present(config.api_key_file)
    ]


def pretty_name(config: ModelConfig) -> str:
    return config.pretty_name or MODEL_PRETTY_NAMES.get(config.id, config.id)


def configured_key_files() -> list[str]:
    return [path for path in LLM_KEY_FILES if key_file_present(path)]


def require_llm_key() -> None:
    """Fail startup when no LLM provider key is configured.

    Every key mount is optional, so nothing upstream of the backend can enforce
    "at least one": compose bind-mounts empty placeholders and k8s skips missing
    optional secrets. Without this the pod would come up and fail every chat.
    """
    if not configured_key_files():
        raise RuntimeError(
            "No LLM API key found. Provide at least one of API_KEY_NV, "
            "API_KEY_NVINF, API_KEY_ANTHROPIC or API_KEY_OPENAI (compose), or "
            "create the nvidia-api-key, anthropic-api-key or openai-api-key "
            "secret (k8s, see deploy/helm/README.md)."
        )


def _discover_anthropic_now() -> None:
    """Populate discovered Anthropic models on demand.

    Imported lazily: this module is also loaded standalone on the host by
    preflight, where the backend package and httpx are not available.
    """
    try:
        from api.llm.anthropic_models import refresh_anthropic_models
    except ImportError:
        return
    refresh_anthropic_models()


def _discover_openai_now() -> None:
    """Populate discovered OpenAI models on demand. Lazy for the same reason."""
    try:
        from api.llm.openai_models import refresh_openai_models
    except ImportError:
        return
    refresh_openai_models()


# Discovered providers in fallback order: (provider, id prefix, key file,
# name of the module-level discovery function). The function is looked up by
# name at call time so tests can replace it.
_DISCOVERED_PROVIDERS = (
    (PROVIDER_ANTHROPIC, ANTHROPIC_ID_PREFIX, ANTHROPIC_KEY_FILE, "_discover_anthropic_now"),
    (PROVIDER_OPENAI, OPENAI_ID_PREFIX, OPENAI_KEY_FILE, "_discover_openai_now"),
)


def _discover_now(function_name: str) -> None:
    globals()[function_name]()


def get_model_config(model_id: str) -> ModelConfig:
    """Get configuration for a model by ID.

    Args:
        model_id: Short identifier for the model.

    Returns:
        ModelConfig for the requested model.

    Raises:
        ValueError: If model_id is not found in registry.
    """
    if model_id not in MODELS and model_id not in _DISCOVERED_MODELS:
        for _provider, prefix, _key_file, discover in _DISCOVERED_PROVIDERS:
            if model_id.startswith(prefix):
                _discover_now(discover)
    models = all_models()
    if model_id not in models:
        available = ", ".join(models.keys())
        raise ValueError(f"Unknown model: {model_id}. Available: {available}")
    return models[model_id]


def _fallback_model() -> Optional[ModelConfig]:
    """The model to use when the requested one has no key."""
    if key_file_present(MODELS[DEFAULT_MODEL_ID].api_key_file):
        return MODELS[DEFAULT_MODEL_ID]
    for provider, _prefix, key_file, discover in _DISCOVERED_PROVIDERS:
        if not key_file_present(key_file):
            continue
        if not any(c.provider == provider for c in _DISCOVERED_MODELS.values()):
            _discover_now(discover)
        discovered = [c for c in available_models() if c.provider == provider]
        if discovered:
            return min(discovered, key=family_rank)
    available = available_models()
    return available[0] if available else None


def get_active_model() -> ModelConfig:
    """Get the currently active model from environment.

    Reads AEVOM_MODEL (else DEFAULT_MODEL_ID). When that model's key is not
    configured, falls back to DEFAULT_MODEL_ID, then the best discovered
    Anthropic model, then the best discovered OpenAI model, then any model
    with a key -- so a checkout holding only an Anthropic or OpenAI key runs
    without setting AEVOM_MODEL. With no key at all the
    requested model is returned unchanged and require_llm_key() reports it.

    Returns:
        ModelConfig for the active model.
    """
    model_id = os.environ.get("AEVOM_MODEL", DEFAULT_MODEL_ID)
    config = get_model_config(model_id)
    if not key_file_present(config.api_key_file):
        fallback = _fallback_model()
        if fallback is not None:
            logger.warning(
                f"No API key for model {model_id} ({config.api_key_file}); "
                f"using {fallback.id}"
            )
            config = fallback
    logger.info(f"Active model: {config.id}")
    return config
