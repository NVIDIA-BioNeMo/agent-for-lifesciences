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

"""Assert the eval Compose stack still matches the production Kubernetes config.

The eval system's central claim is that it measures production. That claim is only
as good as the agreement between ``docker-compose.eval.yml`` and what the Helm
chart deploys, and nothing else in the repo would notice if they drifted -- a new
env var in the chart would simply be absent from every eval run, silently
changing what is being measured.

The production config is read from the chart's ``values.yaml`` and
``values-production.yaml`` rather than by shelling out to ``helm template``, for
two reasons: this suite runs in the bare ``python:3.11-slim`` CI image, and
.devcontainer/devcontainer.json sets ``"helm": "none"`` so there is no local helm
binary either. Both value files are plain YAML, which is exactly why the chart
keeps ``config``, ``apiEnv`` and ``pyexecEnv`` as literal maps.

That the chart *renders* is a separate question, covered by the chart-lint job in
.gitlab-ci.yml so a template error cannot hide behind a passing test here.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[3]
CHART = REPO_ROOT / "deploy" / "helm" / "aevom"
VALUES = CHART / "values.yaml"
VALUES_PRODUCTION = CHART / "values-production.yaml"
EVAL_COMPOSE = REPO_ROOT / "docker-compose.eval.yml"

# Env vars the eval stack adds on purpose. Anything outside this set must be
# justified by the chart's values.
EVAL_ONLY_ENV = {
    "AEVOM_TOOL_CACHE_MODE",
    "AEVOM_TOOL_CACHE_DIR",
}

# Secret-backed vars: production sources them from a k8s Secret, the eval stack
# from a local password or bind-mounted key file. The values cannot match and
# should not.
SECRET_BACKED_ENV = {"POSTGRES_PASSWORD"}

# ConfigMap keys the eval stack takes from values.yaml instead of production:
# production's feedback relay and distribution list are NVIDIA-internal, so the
# eval stack runs with the relay off.
PUBLIC_DEFAULT_CONFIG = {
    "AEVOM_FEAT_FEEDBACK_EMAIL",
    "FEEDBACK_EMAIL_ADDRESS",
    "FEEDBACK_FROM_ADDRESS",
    "SMTP_HOST",
    "SMTP_PORT",
}

_INTERPOLATION = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::?-([^}]*))?\}")


def _interpolate(value: str) -> str:
    """Resolve ``${VAR:-default}`` to its default, as Compose would with no env.

    Deliberately ignores the ambient environment: the test asserts what a clean
    CI runner gets, not what the developer's shell happens to export.
    """

    def replace(match: re.Match[str]) -> str:
        return match.group(2) or ""

    return _INTERPOLATION.sub(replace, value)


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def _production_values() -> dict:
    """values.yaml overlaid with values-production.yaml, as helm -f would.

    Only the top level is merged, which is all the chart needs: production
    overrides whole sections, and the sections this test reads (``config``,
    ``apiEnv``, ``pyexecEnv``) are flat maps. If production starts overriding
    something nested, this needs to become a recursive merge.
    """
    base = _load(VALUES)
    overrides = _load(VALUES_PRODUCTION)
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = {**base[key], **value}
        else:
            merged[key] = value
    return merged


def _compose_env(service: str) -> dict[str, str]:
    compose = _load(EVAL_COMPOSE)
    raw = compose["services"][service].get("environment", {})
    assert isinstance(raw, dict), (
        f"{service}.environment must use mapping form so this test can compare "
        "keys; list form would need parsing"
    )
    return {k: _interpolate(str(v)) for k, v in raw.items()}


def _normalise_db_url(url: str) -> str:
    """Blank the password so the k8s and Compose URLs become comparable."""
    return re.sub(r"//aevom:[^@]*@", "//aevom:<password>@", url)


@pytest.fixture(scope="module")
def values() -> dict:
    if not VALUES_PRODUCTION.exists():
        pytest.skip("values-production.yaml is not published with the public release")
    return _production_values()


@pytest.fixture(scope="module")
def configmap(values) -> dict[str, str]:
    """The ``aevom-config`` ConfigMap as the production values render it."""
    return {k: str(v) for k, v in values["config"].items()}


@pytest.fixture(scope="module")
def api_env() -> dict[str, str]:
    return _compose_env("api")


def test_every_configmap_key_reaches_the_eval_api(configmap, api_env):
    """A key added to the ConfigMap must also be set on the eval api service."""
    missing = sorted(set(configmap) - set(api_env))
    assert not missing, (
        f"chart config keys absent from docker-compose.eval.yml api service: {missing}. "
        "Add them, or the eval stack is running with different config than production."
    )


def test_configmap_values_match(configmap, api_env):
    public = _load(VALUES)["config"]
    expected_config = {**configmap, **{key: str(public[key]) for key in PUBLIC_DEFAULT_CONFIG}}
    mismatched = {
        key: (expected, api_env[key])
        for key, expected in expected_config.items()
        if key in api_env and api_env[key] != expected
    }
    assert not mismatched, (
        "docker-compose.eval.yml disagrees with the production ConfigMap "
        f"(key: (production, eval)): {mismatched}"
    )


def test_deployment_literal_env_matches(values, api_env):
    """The chart's ``apiEnv`` must be reproduced by the eval stack.

    Covers the vars that bypass the ConfigMap -- DATABASE_URL, the API key file
    path, and the Phoenix trio -- which are just as load-bearing.
    """
    problems = []
    for name, value in values["apiEnv"].items():
        if name in SECRET_BACKED_ENV:
            continue
        if name not in api_env:
            problems.append(f"{name}: missing from eval compose")
            continue

        expected, actual = str(value), api_env[name]
        if name == "DATABASE_URL":
            # k8s interpolates $(POSTGRES_PASSWORD) at pod start.
            expected = _normalise_db_url(expected.replace("$(POSTGRES_PASSWORD)", ""))
            actual = _normalise_db_url(actual)
        if expected != actual:
            problems.append(f"{name}: production={expected!r} eval={actual!r}")

    assert not problems, "api literal env drift: " + "; ".join(problems)


def test_pyexec_callback_url_matches_production(values):
    """pyexec's callback must resolve the api the same way in both stacks.

    Production hardcodes the ``aevom-api`` Service name in both this var and the
    ConfigMap's SANDBOX_BACKEND_URL, which is why the eval api service carries
    an ``aevom-api`` network alias.
    """
    expected = values["pyexecEnv"]["BACKEND_URL"]
    assert _compose_env("pyexec")["BACKEND_URL"] == expected

    aliases = _load(EVAL_COMPOSE)["services"]["api"]["networks"]["default"]["aliases"]
    host = expected.removeprefix("http://").split(":")[0]
    assert host in aliases, (
        f"pyexec calls back to {host!r} but the eval api service has no such "
        f"network alias (has {aliases})"
    )


def test_no_unexplained_env_in_eval_api(values, configmap, api_env):
    """The eval api service may not invent config beyond EVAL_ONLY_ENV."""
    declared = set(configmap) | set(values["apiEnv"])
    unexplained = sorted(set(api_env) - declared - EVAL_ONLY_ENV)
    assert not unexplained, (
        f"eval-only env vars not declared in EVAL_ONLY_ENV: {unexplained}. "
        "Either they belong in the chart's values or they need a note."
    )


def test_eval_api_runs_the_production_entrypoint():
    """No `command:` override, so the image's uvicorn CMD (no --reload) runs."""
    api = _load(EVAL_COMPOSE)["services"]["api"]
    assert "command" not in api and "entrypoint" not in api, (
        "docker-compose.eval.yml overrides the api entrypoint; production runs "
        "the image default, so the eval stack would measure a different server"
    )


def test_eval_api_does_not_mount_source():
    """Live-reload mounts would let the eval stack run uncommitted code."""
    api = _load(EVAL_COMPOSE)["services"]["api"]
    source_mounts = []
    for volume in api.get("volumes", []):
        target = volume["target"] if isinstance(volume, dict) else volume.split(":")[1]
        if target.rstrip("/") in {"/app/api", "/app", "/app/skills", "/app/static"}:
            source_mounts.append(target)

    assert not source_mounts, (
        f"eval api service mounts application source at {source_mounts}; the "
        "eval stack must run only what is baked into the pinned image"
    )


def test_eval_image_defaults_are_images_something_builds():
    """The default tags must be the ones `docker compose build` actually produces.

    The eval stack never builds; it pins images. On a merge request there is no
    registry tag to pin -- the build stage is main-only -- so CI builds from
    docker-compose.yml and relies on these defaults matching. A rename on either
    side would leave the eval stack pulling a tag that does not exist, or worse
    silently using a stale one that does.
    """
    dev = _load(REPO_ROOT / "docker-compose.yml")["services"]
    buildable = {
        service["image"]
        for service in dev.values()
        if "build" in service and "image" in service
    }
    unbuilt = {
        service["build"].get("dockerfile", "Dockerfile")
        for name, service in dev.items()
        if "build" in service and "image" not in service
    }
    assert not unbuilt, (
        f"docker-compose.yml services build without an explicit `image:` "
        f"({sorted(unbuilt)}); their tag then depends on the compose project "
        "name, which is the checkout directory name"
    )

    dangling = []
    for name, service in _load(EVAL_COMPOSE)["services"].items():
        image = _interpolate(str(service.get("image", "")))
        # Third-party images (postgres, redis, phoenix) are pulled, not built.
        if not image or "aevom" not in image:
            continue
        if image not in buildable:
            dangling.append(f"{name} -> {image}")

    assert not dangling, (
        "docker-compose.eval.yml defaults to image tags that docker-compose.yml "
        f"does not build: {dangling}. Buildable tags are {sorted(buildable)}."
    )


def _chart_workloads(values: dict) -> dict[str, dict]:
    """The chart's workload sections, keyed by name.

    A workload is a top-level values section carrying both ``enabled`` and
    ``resources``. That is what separates them from ``ingress`` and
    ``certManager``, which also have ``enabled`` but schedule no pods -- and it
    means a workload added to the chart is picked up here automatically.
    """
    return {
        key: value
        for key, value in values.items()
        if isinstance(value, dict) and "enabled" in value and "resources" in value
    }


def test_workload_discovery_still_works(values):
    """Guard the heuristic above: a silent miss would make the next test vacuous."""
    workloads = _chart_workloads(values)
    assert "api" in workloads and len(workloads) >= 7, (
        f"only found workloads {sorted(workloads)} in values.yaml; the "
        "enabled+resources convention _chart_workloads relies on has changed"
    )


def test_eval_stack_covers_every_production_workload(values):
    """Each enabled chart workload needs a counterpart service in the eval stack.

    The chart's workload keys are deliberately the eval Compose service names, so
    this is a straight set comparison. The k8s objects are named ``aevom-api``
    while the section and the Compose service are both ``api``.
    """
    compose_services = set(_load(EVAL_COMPOSE)["services"])
    missing = sorted(
        name
        for name, workload in _chart_workloads(values).items()
        if workload["enabled"] and name not in compose_services
    )

    assert not missing, (
        f"chart workloads with no eval Compose service: {missing}"
    )
