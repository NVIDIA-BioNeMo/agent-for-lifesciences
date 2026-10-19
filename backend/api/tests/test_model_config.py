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

"""Tests for LLM model configuration and registry.

Verifies that model configs are correctly registered, looked up by ID,
and that environment-driven model selection works as expected.
"""

from pathlib import Path

import pytest

from api.llm import config as config_module
from api.llm.config import (
    ANTHROPIC_ID_PREFIX,
    ANTHROPIC_KEY_FILE,
    DEFAULT_MODEL_ID,
    MODELS,
    MODELS_HIDDEN_FROM_UI,
    NVIDIA_KEY_FILE,
    NVINF_KEY_FILE,
    OPENAI_ID_PREFIX,
    OPENAI_KEY_FILE,
    PROVIDER_ANTHROPIC,
    PROVIDER_OPENAI,
    ModelConfig,
    available_models,
    get_active_model,
    get_model_config,
    key_file_present,
    pretty_name,
    register_models,
    require_llm_key,
)


class TestModelRegistry:
    """Tests for the MODELS registry and get_model_config()."""

    def test_all_registered_models_have_matching_ids(self):
        """Every entry's dict key must match its ModelConfig.id."""
        for key, config in MODELS.items():
            assert key == config.id, f"Key '{key}' != config.id '{config.id}'"

    def test_default_model_exists(self):
        """DEFAULT_MODEL_ID must reference a registered model."""
        assert DEFAULT_MODEL_ID in MODELS

    def test_get_model_config_returns_correct_instance(self):
        for model_id in MODELS:
            config = get_model_config(model_id)
            assert isinstance(config, ModelConfig)
            assert config.id == model_id

    def test_get_model_config_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_config("nonexistent-model-xyz")

    def test_error_message_lists_available_models(self):
        with pytest.raises(ValueError) as exc_info:
            get_model_config("bad-id")
        for model_id in MODELS:
            assert model_id in str(exc_info.value)


class TestClaudeOpus46Config:
    """Tests specific to the Claude Opus 4.6 model entry."""

    def test_registered(self):
        assert "claude-opus-4-6" in MODELS

    def test_config_fields(self):
        config = get_model_config("claude-opus-4-6")
        assert config.name == "us/aws/anthropic/bedrock-claude-opus-4-6-v1"
        assert config.base_url == "https://inference-api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key_nvinf"
        assert config.supports_reasoning is False
        assert config.requires_thought_signatures is False
        assert config.template_name == "base"
        assert config.default_temperature == 0.7
        assert config.max_tokens == 8192

    def test_no_extra_body(self):
        """Claude should not inject thinking params into extra_body."""
        config = get_model_config("claude-opus-4-6")
        extra = config.get_extra_body()
        assert "chat_template_kwargs" not in extra
        assert "thinking_level" not in extra


class TestGpt51Config:
    """Tests specific to the GPT 5.1 model entry."""

    def test_registered(self):
        assert "gpt-5-1" in MODELS

    def test_config_fields(self):
        config = get_model_config("gpt-5-1")
        assert config.name == "openai/openai/gpt-5.1"
        assert config.base_url == "https://inference-api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key_nvinf"
        assert config.supports_reasoning is False
        assert config.requires_thought_signatures is False
        assert config.template_name == "base"
        assert config.default_temperature == 0.9
        assert config.max_tokens == 8192

    def test_no_extra_body(self):
        """GPT 5.1 should not inject thinking params into extra_body."""
        config = get_model_config("gpt-5-1")
        extra = config.get_extra_body()
        assert "chat_template_kwargs" not in extra
        assert "thinking_level" not in extra


class TestGpt6AstraConfig:
    """Tests specific to the GPT-6 Astra model entry."""

    def test_registered(self):
        assert "gpt-6-astra" in MODELS

    def test_config_fields(self):
        config = get_model_config("gpt-6-astra")
        assert config.name == "azure/openai/gpt-6-astra"
        assert config.base_url == "https://inference-api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key_nvinf"
        assert config.supports_reasoning is False
        assert config.requires_thought_signatures is False
        assert config.template_name == "base"
        assert config.default_temperature == 1.0
        assert config.max_tokens == 16384

    def test_temperature_is_unsupported(self):
        """The endpoint 400s on any temperature but 1.0, so none is sent at all."""
        config = get_model_config("gpt-6-astra")
        assert config.supports_temperature is False

    def test_no_extra_body(self):
        """GPT-6 Astra should not inject thinking params into extra_body."""
        config = get_model_config("gpt-6-astra")
        extra = config.get_extra_body()
        assert "chat_template_kwargs" not in extra
        assert "thinking_level" not in extra


class TestNemotron3SuperDynamoConfig:
    """Tests specific to the Nemotron 3 Super Dynamo model entry."""

    def test_registered(self):
        assert "nemotron-3-super-dynamo" in MODELS

    def test_config_fields(self):
        config = get_model_config("nemotron-3-super-dynamo")
        assert config.name == "nvidia/nvidia/nemotron-3-super-preview"
        assert config.base_url == "https://inference-api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key_nvinf"
        assert config.supports_reasoning is True
        assert config.content_has_think_tags is False
        assert config.thinking_param == ""
        assert config.requires_thought_signatures is False
        assert config.template_name == "base"
        assert config.default_temperature == 0.7
        assert config.top_p == 0.95
        assert config.max_tokens == 1024

    def test_no_extra_body(self):
        """Nemotron 3 Super Dynamo should not inject thinking params into extra_body."""
        config = get_model_config("nemotron-3-super-dynamo")
        extra = config.get_extra_body()
        assert "chat_template_kwargs" not in extra
        assert "thinking_level" not in extra


class TestNemotron3UltraConfig:
    """Tests specific to the public (build.nvidia.com) Nemotron 3 Ultra entry."""

    def test_registered(self):
        assert "nemotron-3-ultra" in MODELS

    def test_config_fields(self):
        config = get_model_config("nemotron-3-ultra")
        assert config.name == "nvidia/nemotron-3-ultra-550b-a55b"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key"
        assert config.supports_reasoning is True
        assert config.content_has_think_tags is False
        assert config.requires_thought_signatures is False
        assert config.template_name == "base"
        assert config.default_temperature == 1.0
        assert config.top_p == 0.95
        assert config.max_tokens == 16384

    def test_extra_body_has_thinking_but_no_reasoning_budget(self):
        """Thinking on, budget off.

        The budget is the regression this guards. Sending `reasoning_budget`
        to this endpoint fails the request and reports it as an in-stream 503
        "Service temporarily overloaded" -- 0/6 calls carrying it succeeded,
        against 5/6 without. It reads as a capacity problem that no amount of
        retrying fixes, so the parameter must not come back.
        """
        config = get_model_config("nemotron-3-ultra")
        extra = config.get_extra_body()
        assert extra["chat_template_kwargs"]["enable_thinking"] is True
        assert "reasoning_budget" not in extra


class TestNemotron3UltraNvinfConfig:
    """Tests specific to the NVIDIA-internal (inference-api) Nemotron 3 Ultra entry.

    Production and the eval suite pin this model, and the eval gate thresholds are
    calibrated against it, so these assertions are a guard against a sampling
    change silently invalidating the measured baseline.
    """

    def test_registered(self):
        assert "nemotron-3-ultra-nvinf" in MODELS

    def test_config_fields(self):
        config = get_model_config("nemotron-3-ultra-nvinf")
        assert config.name == "nvidia/nvidia/nemotron-3-ultra"
        assert config.base_url == "https://inference-api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key_nvinf"
        assert config.supports_reasoning is True
        assert config.content_has_think_tags is False
        assert config.thinking_param == ""
        assert config.requires_thought_signatures is False
        assert config.template_name == "base"
        assert config.default_temperature == 0.7
        assert config.top_p == 0.95
        assert config.max_tokens == 32768

    def test_no_extra_body(self):
        """This endpoint reasons by default, so nothing needs injecting."""
        config = get_model_config("nemotron-3-ultra-nvinf")
        extra = config.get_extra_body()
        assert "chat_template_kwargs" not in extra
        assert "thinking_level" not in extra


class TestNemotron3Super120bConfig:
    """Tests specific to the build.nvidia.com Nemotron 3 Super 120B entry."""

    def test_registered(self):
        assert "nemotron-3-super-120b" in MODELS

    def test_config_fields(self):
        config = get_model_config("nemotron-3-super-120b")
        assert config.name == "nvidia/nemotron-3-super-120b-a12b"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key"
        assert config.supports_reasoning is True
        assert config.retains_reasoning is True
        assert config.template_name == "base"
        assert config.default_temperature == 0.7

    def test_reasons_without_being_asked(self):
        """It returns reasoning_content unprompted, so nothing is injected."""
        config = get_model_config("nemotron-3-super-120b")
        assert config.thinking_param == ""
        assert config.get_extra_body() == {}


class TestNemotron3NanoOmniConfig:
    """Tests specific to the Nemotron 3 Nano Omni entry."""

    def test_registered(self):
        assert "nemotron-3-nano-omni" in MODELS

    def test_config_fields(self):
        config = get_model_config("nemotron-3-nano-omni")
        assert config.name == "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key"
        assert config.supports_reasoning is True
        assert config.retains_reasoning is True
        assert config.template_name == "base"

    def test_reasons_without_being_asked(self):
        config = get_model_config("nemotron-3-nano-omni")
        assert config.thinking_param == ""
        assert config.get_extra_body() == {}


class TestNemotron35SuperVlConfig:
    """Tests specific to the build.nvidia.com Nemotron 3.5 Super VL entry."""

    def test_registered(self):
        assert "nemotron-3-5-super-vl" in MODELS

    def test_is_the_default(self):
        """The default has to be an endpoint an external user can actually reach."""
        assert DEFAULT_MODEL_ID == "nemotron-3-5-super-vl"

    def test_config_fields(self):
        config = get_model_config("nemotron-3-5-super-vl")
        assert config.name == "nvidia/nemotron-3.5-super-vl-120b-a12b"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key"
        assert config.supports_reasoning is True
        assert config.retains_reasoning is True
        assert config.template_name == "base"
        assert config.default_temperature == 0.7

    def test_reasons_without_being_asked(self):
        config = get_model_config("nemotron-3-5-super-vl")
        assert config.thinking_param == ""
        assert config.get_extra_body() == {}


class TestNemotron35LightningConfig:
    """Tests specific to the build.nvidia.com Nemotron 3.5 Lightning entry."""

    def test_registered(self):
        assert "nemotron-3-5-lightning" in MODELS

    def test_config_fields(self):
        config = get_model_config("nemotron-3-5-lightning")
        assert config.name == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key"
        assert config.supports_reasoning is True
        assert config.retains_reasoning is True
        assert config.template_name == "base"
        assert config.default_temperature == 0.7

    def test_reasons_without_being_asked(self):
        config = get_model_config("nemotron-3-5-lightning")
        assert config.thinking_param == ""
        assert config.get_extra_body() == {}


class TestMuseGlimmerConfig:
    """Tests specific to the Muse Glimmer entry."""

    def test_registered(self):
        assert "muse-glimmer" in MODELS

    def test_config_fields(self):
        config = get_model_config("muse-glimmer")
        assert config.name == "meta/muse-glimmer-30b"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key"
        assert config.supports_reasoning is True
        assert config.retains_reasoning is True
        assert config.template_name == "base"
        assert config.default_temperature == 0.7

    def test_reasons_without_being_asked(self):
        """No thinking switch is sent.

        This guards a hang, not just a redundancy: a request carrying
        chat_template_kwargs.enable_thinking sat for 180s without a byte, while
        the same request without it answered in under 10s.
        """
        config = get_model_config("muse-glimmer")
        assert config.thinking_param == ""
        assert config.get_extra_body() == {}


class TestGptOss20bConfig:
    """Tests specific to the build.nvidia.com GPT-OSS 20B entry."""

    def test_registered(self):
        assert "gpt-oss-20b" in MODELS

    def test_config_fields(self):
        config = get_model_config("gpt-oss-20b")
        assert config.name == "openai/gpt-oss-20b"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key"
        assert config.supports_reasoning is True
        assert config.retains_reasoning is True
        assert config.template_name == "base"
        assert config.default_temperature == 1.0
        assert config.top_p == 1.0

    def test_reasons_without_being_asked(self):
        config = get_model_config("gpt-oss-20b")
        assert config.thinking_param == ""
        assert config.get_extra_body() == {}


class TestGlm53Config:
    """Tests specific to the GLM-5.3 entry."""

    def test_registered(self):
        assert "glm-5-3" in MODELS

    def test_config_fields(self):
        config = get_model_config("glm-5-3")
        assert config.name == "z-ai/glm-5.3"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key"
        assert config.supports_reasoning is True
        assert config.retains_reasoning is True
        assert config.template_name == "base"
        assert config.default_temperature == 1.0
        assert config.top_p == 1.0

    def test_reasons_without_being_asked(self):
        """Reasoning cannot be switched off here, so no switch is sent at all."""
        config = get_model_config("glm-5-3")
        assert config.thinking_param == ""
        assert config.get_extra_body() == {}


class TestGlm53FlashConfig:
    """Tests specific to the GLM-5.3 Flash entry."""

    def test_registered(self):
        assert "glm-5-3-flash" in MODELS

    def test_config_fields(self):
        config = get_model_config("glm-5-3-flash")
        assert config.name == "z-ai/glm-5.3-flash"
        assert config.base_url == "https://integrate.api.nvidia.com/v1"
        assert config.api_key_file == "/run/secrets/nvidia_api_key"
        assert config.supports_reasoning is True
        assert config.retains_reasoning is True
        assert config.template_name == "base"
        assert config.default_temperature == 1.0
        assert config.top_p == 1.0

    def test_reasons_without_being_asked(self):
        """Reasoning cannot be switched off here, so no switch is sent at all."""
        config = get_model_config("glm-5-3-flash")
        assert config.thinking_param == ""
        assert config.get_extra_body() == {}


class TestGetActiveModel:
    """Tests for environment-driven model selection."""

    def test_defaults_to_default_model(self, monkeypatch):
        monkeypatch.delenv("AEVOM_MODEL", raising=False)
        config = get_active_model()
        assert config.id == DEFAULT_MODEL_ID

    def test_respects_env_var(self, monkeypatch):
        monkeypatch.setenv("AEVOM_MODEL", "claude-opus-4-6")
        config = get_active_model()
        assert config.id == "claude-opus-4-6"
        assert config.name == "us/aws/anthropic/bedrock-claude-opus-4-6-v1"

    def test_invalid_env_var_raises(self, monkeypatch):
        monkeypatch.setenv("AEVOM_MODEL", "does-not-exist")
        with pytest.raises(ValueError, match="Unknown model"):
            get_active_model()


def claude(name: str) -> ModelConfig:
    return ModelConfig(
        id=f"{ANTHROPIC_ID_PREFIX}{name}",
        name=name,
        base_url="https://api.anthropic.com",
        api_key_file=ANTHROPIC_KEY_FILE,
        provider=PROVIDER_ANTHROPIC,
    )


def gpt(name: str) -> ModelConfig:
    return ModelConfig(
        id=f"{OPENAI_ID_PREFIX}{name}",
        name=name,
        base_url="https://api.openai.com/v1",
        api_key_file=OPENAI_KEY_FILE,
        provider=PROVIDER_OPENAI,
    )


@pytest.fixture
def keys(monkeypatch):
    """Key files to treat as present; every other key file is absent."""
    present: set[str] = set()
    monkeypatch.setattr(config_module, "key_file_present", lambda path: path in present)
    yield present
    register_models(PROVIDER_ANTHROPIC, [])
    register_models(PROVIDER_OPENAI, [])


@pytest.fixture
def discovered(monkeypatch):
    """Stand in for Anthropic discovery, which would otherwise call the API."""
    models = [claude("claude-haiku-5-5"), claude("claude-opus-5-5"), claude("claude-sonnet-5-5")]
    calls = []

    def fake_discover():
        calls.append(1)
        register_models(PROVIDER_ANTHROPIC, models)

    monkeypatch.setattr(config_module, "_discover_anthropic_now", fake_discover)
    return calls


@pytest.fixture
def discovered_openai(monkeypatch):
    """Stand in for OpenAI discovery, which would otherwise call the API."""
    models = [gpt("gpt-6-luna"), gpt("gpt-6.1-sol"), gpt("gpt-6-astra")]
    calls = []

    def fake_discover():
        calls.append(1)
        register_models(PROVIDER_OPENAI, models)

    monkeypatch.setattr(config_module, "_discover_openai_now", fake_discover)
    return calls


class TestKeyFilePresent:
    def test_populated_file(self, tmp_path):
        key = tmp_path / "key"
        key.write_text("nvapi-xyz\n")
        assert key_file_present(str(key))

    def test_empty_placeholder_missing_file_and_directory_are_absent(self, tmp_path):
        empty = tmp_path / "empty"
        empty.write_text("\n  \n")
        assert not key_file_present(str(empty))
        assert not key_file_present(str(tmp_path / "missing"))
        assert not key_file_present(str(tmp_path))


class TestActiveModelFallback:
    def test_requested_model_with_key_is_used(self, keys, monkeypatch):
        keys.add(NVINF_KEY_FILE)
        monkeypatch.setenv("AEVOM_MODEL", "gpt-5-1")
        assert get_active_model().id == "gpt-5-1"

    def test_falls_back_to_default_model_when_its_key_exists(self, keys, monkeypatch):
        keys.add(NVIDIA_KEY_FILE)
        monkeypatch.setenv("AEVOM_MODEL", "gpt-5-1")
        assert get_active_model().id == DEFAULT_MODEL_ID

    def test_anthropic_only_falls_back_to_best_claude(self, keys, discovered, monkeypatch):
        keys.add(ANTHROPIC_KEY_FILE)
        monkeypatch.delenv("AEVOM_MODEL", raising=False)
        assert get_active_model().id == "anthropic/claude-opus-5-5"
        assert discovered == [1]

    def test_requested_claude_model_is_discovered_on_demand(self, keys, discovered, monkeypatch):
        keys.add(ANTHROPIC_KEY_FILE)
        monkeypatch.setenv("AEVOM_MODEL", "anthropic/claude-sonnet-5-5")
        assert get_active_model().id == "anthropic/claude-sonnet-5-5"

    def test_unknown_claude_model_still_raises(self, keys, discovered):
        keys.add(ANTHROPIC_KEY_FILE)
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_config("anthropic/claude-nonexistent")

    def test_openai_only_falls_back_to_best_gpt(self, keys, discovered_openai, monkeypatch):
        keys.add(OPENAI_KEY_FILE)
        monkeypatch.delenv("AEVOM_MODEL", raising=False)
        assert get_active_model().id == "openai/gpt-6-astra"
        assert discovered_openai == [1]

    def test_anthropic_is_preferred_over_openai(self, keys, discovered, discovered_openai, monkeypatch):
        keys.update({ANTHROPIC_KEY_FILE, OPENAI_KEY_FILE})
        monkeypatch.delenv("AEVOM_MODEL", raising=False)
        assert get_active_model().id == "anthropic/claude-opus-5-5"
        assert discovered_openai == []

    def test_requested_gpt_model_is_discovered_on_demand(self, keys, discovered, discovered_openai, monkeypatch):
        keys.add(OPENAI_KEY_FILE)
        monkeypatch.setenv("AEVOM_MODEL", "openai/gpt-6.1-sol")
        assert get_active_model().id == "openai/gpt-6.1-sol"
        assert discovered == [] and discovered_openai == [1]

    def test_requested_gpt_without_its_key_falls_back(self, keys, discovered_openai, monkeypatch):
        keys.add(NVIDIA_KEY_FILE)
        monkeypatch.setenv("AEVOM_MODEL", "openai/gpt-6.1-sol")
        assert get_active_model().id == DEFAULT_MODEL_ID

    def test_no_key_at_all_returns_the_requested_model(self, keys, monkeypatch):
        monkeypatch.setenv("AEVOM_MODEL", "gpt-5-1")
        assert get_active_model().id == "gpt-5-1"


class TestAvailableModels:
    def test_only_models_with_a_key_are_listed(self, keys, discovered):
        keys.add(ANTHROPIC_KEY_FILE)
        config_module._discover_anthropic_now()
        ids = [m.id for m in available_models()]
        assert ids == ["anthropic/claude-haiku-5-5", "anthropic/claude-opus-5-5", "anthropic/claude-sonnet-5-5"]

    def test_openai_models_are_listed_with_their_key(self, keys, discovered_openai):
        keys.add(OPENAI_KEY_FILE)
        config_module._discover_openai_now()
        assert [m.id for m in available_models()] == ["openai/gpt-6-luna", "openai/gpt-6.1-sol", "openai/gpt-6-astra"]

    def test_hidden_models_stay_hidden(self, keys):
        keys.add(NVINF_KEY_FILE)
        ids = {m.id for m in available_models()}
        assert "gpt-5-1" in ids
        assert ids.isdisjoint(MODELS_HIDDEN_FROM_UI)

    def test_pretty_name_prefers_the_discovered_display_name(self):
        assert pretty_name(MODELS["nemotron-3-ultra"]) == "Nemotron 3 Ultra"
        config = claude("claude-opus-5-5")
        config.pretty_name = "Claude Opus 5.5"
        assert pretty_name(config) == "Claude Opus 5.5"


class TestRequireLlmKey:
    def test_raises_without_any_key(self, keys):
        with pytest.raises(RuntimeError, match="No LLM API key found"):
            require_llm_key()

    @pytest.mark.parametrize("key_file", [NVIDIA_KEY_FILE, NVINF_KEY_FILE, ANTHROPIC_KEY_FILE, OPENAI_KEY_FILE])
    def test_any_single_key_is_enough(self, keys, key_file):
        keys.add(key_file)
        require_llm_key()


def test_production_model_is_registered():
    """The chart's production AEVOM_MODEL must exist in the MODELS registry.

    Deploying an unknown model id does not fail at rollout -- the api pod starts,
    then every chat request raises. Line-parsed rather than loaded with yaml so
    this keeps working in the bare CI image, same reason as
    test_eval_compose_parity.py.
    """
    repo_root = Path(__file__).resolve().parents[3]
    values = repo_root / "deploy" / "helm" / "aevom" / "values-production.yaml"
    if not values.exists():
        pytest.skip("values-production.yaml is not published with the public release")
    content = values.read_text(encoding="utf-8")

    model_id = None
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("AEVOM_MODEL:"):
            model_id = stripped.split(":", 1)[1].strip().strip("'\"")
            break

    assert model_id, f"AEVOM_MODEL not found in {values.name}"
    assert model_id in MODELS, f"Production model '{model_id}' is not in MODELS registry"
