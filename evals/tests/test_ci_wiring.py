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

"""Pin the parts of the eval pipeline that only fail once CI runs them.

An import path typo in ``run_suite.sh``, or a new task the gate's trial count
doesn't know about, costs a full pipeline to discover. These are cheap here.

Import paths are checked by parsing rather than importing: the agent and verifier
subclass Harbor's base classes, and Harbor is deliberately not a dependency of
the backend test environment.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_SUITE = REPO_ROOT / "evals" / "scripts" / "run_suite.sh"
GITLAB_CI = REPO_ROOT / ".gitlab-ci.yml"
MAKEFILE = REPO_ROOT / "Makefile"
TASKS_DIR = REPO_ROOT / "evals" / "tasks"
EVAL_COMPOSE = REPO_ROOT / "docker-compose.eval.yml"


class _Reference(list):
    """A `!reference [job, key]` node, held until the whole file is loaded."""


class _GitLabLoader(yaml.SafeLoader):
    """SafeLoader that accepts GitLab's `!reference` tag.

    `!reference` is a GitLab extension, not YAML, so a plain `safe_load` raises
    ConstructorError on any file that uses it -- which turned every test taking
    the `ci` fixture into a setup error once the build jobs began pulling in the
    `.docker-prune` script that way.
    """


_GitLabLoader.add_constructor(
    "!reference", lambda loader, node: _Reference(loader.construct_sequence(node))
)


def _splice_references(root: dict, node):
    """Replace each `!reference` in a list with the lines it points at.

    GitLab does the same when it builds a job, so the tests see `before_script`
    and `script` as the flat lists of commands the runner would execute.
    """
    if isinstance(node, dict):
        return {key: _splice_references(root, value) for key, value in node.items()}
    if isinstance(node, list) and not isinstance(node, _Reference):
        spliced = []
        for item in node:
            if isinstance(item, _Reference):
                target = root
                for key in item:
                    target = target[key]
                spliced.extend(_splice_references(root, target))
            else:
                spliced.append(_splice_references(root, item))
        return spliced
    return node


def load_gitlab_ci(text: str) -> dict:
    document = yaml.load(text, Loader=_GitLabLoader)
    return _splice_references(document, document)


@pytest.fixture(scope="module")
def ci() -> dict:
    if not GITLAB_CI.exists():
        pytest.skip(".gitlab-ci.yml is not published with the public release")
    return load_gitlab_ci(GITLAB_CI.read_text())


def test_references_are_spliced_into_the_job_scripts(ci):
    """The loader must hand back runnable lines, not tag objects.

    Several tests join or scan a job's `script`/`before_script`; a leftover
    `!reference` placeholder there would pass loading and then fail confusingly.
    """
    before_script = ci["build-api"]["before_script"]
    assert all(isinstance(line, str) for line in before_script)
    assert before_script[: len(ci[".docker-prune"]["script"])] == ci[".docker-prune"]["script"]


def test_the_loader_resolves_a_reference_it_has_not_seen_before():
    loaded = load_gitlab_ci(
        "\n".join(
            [
                ".shared:",
                "  script:",
                "    - echo one",
                "    - echo two",
                "job:",
                "  before_script:",
                "    - !reference [.shared, script]",
                "    - echo three",
            ]
        )
    )
    assert loaded["job"]["before_script"] == ["echo one", "echo two", "echo three"]


@pytest.fixture(scope="module")
def makefile() -> str:
    return MAKEFILE.read_text()


@pytest.fixture(scope="module")
def suite_script() -> str:
    return RUN_SUITE.read_text()


def n_tasks() -> int:
    return sum(1 for path in TASKS_DIR.iterdir() if (path / "task.toml").exists())


def defines_class(module_path: Path, class_name: str) -> bool:
    tree = ast.parse(module_path.read_text())
    return any(
        isinstance(node, ast.ClassDef) and node.name == class_name
        for node in tree.body
    )


@pytest.mark.parametrize("flag", ["--agent", "--verifier"])
def test_import_paths_resolve(suite_script, flag):
    """`module.path:ClassName` handed to harbor must name something real."""
    match = re.search(rf"{flag} (\S+)", suite_script)
    assert match, f"run_suite.sh no longer passes {flag} to harbor"

    dotted, _, class_name = match.group(1).partition(":")
    assert class_name, f"{flag} value {match.group(1)!r} has no :ClassName"

    module_path = REPO_ROOT / "evals" / (dotted.replace(".", "/") + ".py")
    assert module_path.is_file(), f"{flag} points at {module_path}, which does not exist"
    assert defines_class(module_path, class_name), (
        f"{module_path} does not define {class_name}"
    )


def test_the_suite_runs_the_task_directory(suite_script):
    default = re.search(r"AEVOM_EVAL_TASKS:-(\S+)\}", suite_script)
    assert default, "run_suite.sh has no default for AEVOM_EVAL_TASKS"
    assert (REPO_ROOT / default.group(1)).is_dir()


def test_gate_is_invoked_after_harbor(suite_script):
    """Harbor's exit code does not reflect the rewards, so the gate must run."""
    assert "gate.py" in suite_script
    assert suite_script.index("harbor \"${harbor_args[@]}\"") < suite_script.index(
        "gate.py"
    )


def test_the_gate_is_not_exec_d(suite_script):
    """exec would replace the shell, taking the EXIT trap with it.

    That trap deletes the staging directory, which holds copies of the API keys on
    the Docker host. Observed: `exec evals/scripts/gate.py` left them there after
    every run, on a host whose /tmp nothing else cleans.
    """
    assert "exec evals/scripts/gate.py" not in suite_script


@pytest.mark.parametrize("job", ["eval-smoke", "eval-live"])
def test_min_trials_matches_the_task_count(ci, job):
    """Otherwise adding a task silently loosens the gate.

    ``--min-trials`` is what stops a job that ran almost nothing from passing on
    the strength of the one task it managed, so it has to track tasks x attempts.
    """
    variables = ci[job]["variables"]
    attempts = int(variables["AEVOM_EVAL_ATTEMPTS"])
    expected = n_tasks() * attempts
    assert int(variables["AEVOM_EVAL_MIN_TRIALS"]) == expected, (
        f"{job} expects {variables['AEVOM_EVAL_MIN_TRIALS']} trials but there are "
        f"{n_tasks()} task(s) x {attempts} attempt(s) = {expected}"
    )


def test_smoke_gates_merge_requests_without_needing_a_schedule(ci):
    sources = [rule.get("if", "") for rule in ci["eval-smoke"]["rules"]]
    assert any("merge_request_event" in rule for rule in sources), (
        "eval-smoke must run on merge requests; that is the tier that gates merges"
    )
    assert not any("schedule" in rule for rule in sources)


def test_live_runs_only_on_a_schedule(ci):
    """Live tools and 3 attempts per task is not a per-push cost."""
    rules = ci["eval-live"]["rules"]
    assert len(rules) == 1
    assert "schedule" in rules[0]["if"]


def resolved_rules(ci: dict, name: str) -> list[dict]:
    """A job's rules, following `extends` one level as GitLab does."""
    job = ci[name]
    if "rules" in job:
        return job["rules"]
    parents = job.get("extends") or []
    if isinstance(parents, str):
        parents = [parents]
    for parent in parents:
        if "rules" in ci.get(parent, {}):
            return ci[parent]["rules"]
    return []


def test_the_nightly_schedule_runs_nothing_but_the_evals(ci):
    """A scheduled pipeline on main also matches CI_COMMIT_BRANCH == "main".

    Which is the whole rule for the build jobs and for deploy-production, so
    without an explicit guard the nightly would rebuild and push every image and
    then roll production -- grading images other than the deployed ones, while
    replacing them mid-run.
    """
    unguarded = []
    for name, job in ci.items():
        if not isinstance(job, dict) or job.get("stage") not in {"build", "deploy"}:
            continue
        guard = next(
            (
                rule
                for rule in resolved_rules(ci, name)
                if "schedule" in str(rule.get("if", ""))
            ),
            None,
        )
        if not guard or guard.get("when") != "never":
            unguarded.append(name)

    assert not unguarded, (
        f"{unguarded} would run on the nightly eval schedule; give them "
        '`- if: $CI_PIPELINE_SOURCE == "schedule"` with `when: never` first, or '
        "extend .publish-rules"
    )


def test_smoke_replays_and_live_does_not(ci):
    assert ci["eval-smoke"]["variables"]["AEVOM_EVAL_MODE"] == "mixed"
    assert ci["eval-live"]["variables"]["AEVOM_EVAL_MODE"] == "off"


def test_live_pins_every_aevom_image(ci):
    """A live run is only attributable if every image is pinned to the registry."""
    compose = yaml.safe_load(EVAL_COMPOSE.read_text())["services"]
    needed = set()
    for service in compose.values():
        match = re.match(r"\$\{([A-Z_]+):-aevom", str(service.get("image", "")))
        if match:
            needed.add(match.group(1))

    pinned = set(ci["eval-live"]["variables"]) & needed
    assert pinned == needed, (
        f"eval-live does not pin {sorted(needed - pinned)}, so those services "
        "would fall back to whatever :latest tag is on the runner"
    )


def test_every_pinned_image_is_built_by_some_job(ci):
    """A pin to a registry path nothing pushes fails at image pull."""
    pushed = set()
    for name, job in ci.items():
        if not isinstance(job, dict) or job.get("stage") != "build":
            continue
        for line in job.get("script", []):
            pushed.update(re.findall(r"\$CI_REGISTRY_IMAGE/([\w.-]+):", str(line)))

    for key, value in ci["eval-live"]["variables"].items():
        if not key.startswith("AEVOM_") or not key.endswith("_IMAGE"):
            continue
        match = re.match(r"\$CI_REGISTRY_IMAGE/([\w.-]+):", str(value))
        assert match, f"{key}={value} is not a registry path this repo builds"
        assert match.group(1) in pushed, (
            f"{key} pins {value} but no build job pushes {match.group(1)}; "
            f"the build stage pushes {sorted(pushed)}"
        )


def test_eval_stage_exists_and_sits_after_build(ci):
    stages = ci["stages"]
    assert "eval" in stages
    assert stages.index("eval") > stages.index("build")
    assert stages.index("eval") < stages.index("deploy")


def test_the_eval_jobs_run_on_a_glibc_image(ci):
    """An Alpine image cannot install harbor, and fails deep inside pip.

    harbor depends on litellm, which publishes no musl wheel; pip then tries to
    build its Rust extension against an image with no C toolchain. The observed
    failure was `linker \\`cc\\` not found` after a two-minute download, nowhere
    near anything eval-shaped.
    """
    image = ci[".eval-base"]["image"]
    assert "alpine" not in image and not image.startswith("docker:"), (
        f"the eval jobs run on {image!r}; a musl image cannot install harbor"
    )


def test_the_eval_jobs_install_the_docker_cli(ci):
    """The image supplies Python, not docker, and the suite shells out to both."""
    before_script = "\n".join(ci[".eval-base"]["before_script"])
    assert "docker-ce-cli" in before_script
    # harbor and run_suite.sh both invoke `docker compose`, which is a plugin
    # rather than part of the CLI package.
    assert "docker-compose-plugin" in before_script


def test_harbor_version_is_pinned(ci):
    """gate.py parses harbor's JobResult schema; a float version would drift."""
    version = ci["variables"]["HARBOR_VERSION"]
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(version)), (
        f"HARBOR_VERSION={version!r} is not an exact version"
    )
    before_script = "\n".join(ci[".eval-base"]["before_script"])
    assert "harbor==${HARBOR_VERSION}" in before_script, (
        "the eval jobs must install the pinned version, not whatever pip resolves"
    )


def test_every_harbor_pin_agrees(ci, makefile):
    """Four files name a version, and a local run must grade like CI grades.

    The devcontainer bakes harbor into a venv of its own, so a developer running
    `make eval` uses whatever that image installed -- which is only the same
    harbor CI uses if the pins were kept together by something. The preflight
    counts too: it prints an install command, and one that names a stale version
    is worse than no advice.
    """
    dockerfile = (REPO_ROOT / ".devcontainer" / "Dockerfile").read_text()
    preflight = (REPO_ROOT / "scripts" / "check-eval-prerequisites.sh").read_text()
    pins = {
        ".gitlab-ci.yml": str(ci["variables"]["HARBOR_VERSION"]),
        "Makefile": _pin(makefile, r"^HARBOR_VERSION\s*:=\s*(\S+)", "Makefile"),
        ".devcontainer/Dockerfile": _pin(
            dockerfile, r"^ARG HARBOR_VERSION=(\S+)", "the devcontainer Dockerfile"
        ),
        "scripts/check-eval-prerequisites.sh": _pin(
            preflight, r'^HARBOR_VERSION="(\S+)"', "the eval preflight"
        ),
    }
    assert len(set(pins.values())) == 1, f"harbor pins disagree: {pins}"


def test_the_eval_targets_preflight_before_building(makefile):
    """The point is to fail in a second, not after a multi-minute image build."""
    for target in ("eval", "eval-smoke"):
        prerequisites = re.search(
            rf"^{re.escape(target)}:(.*)$", makefile, re.MULTILINE
        ).group(1)
        assert "check-eval" in prerequisites, (
            f"{target} does not depend on check-eval, so a missing harbor would "
            "only surface once run_suite.sh is reached"
        )


def _pin(text: str, pattern: str, where: str) -> str:
    match = re.search(pattern, text, re.MULTILINE)
    assert match, f"{where} no longer declares HARBOR_VERSION"
    return match.group(1)


@pytest.mark.parametrize(
    ("target", "script"),
    [
        ("eval-smoke", "evals/scripts/run_suite.sh"),
        ("eval", "evals/scripts/run_suite.sh"),
        ("eval-record", "evals/scripts/record_fixtures.py"),
    ],
)
def test_make_targets_delegate_to_the_scripts(makefile, target, script):
    """Not reimplemented in make, so a tier has one definition CI shares."""
    body = re.search(
        rf"^{re.escape(target)}:.*?\n((?:\t.*\n|\n)*)", makefile, re.MULTILINE
    )
    assert body, f"the Makefile has no {target} target"
    assert script in body.group(1), (
        f"{target} does not run {script}; a make-only copy of the tier will drift "
        "from what CI grades"
    )


@pytest.mark.parametrize(
    ("target", "job"),
    [("eval-smoke", "eval-smoke"), ("eval", "eval-live")],
)
def test_make_tiers_match_their_ci_counterparts(ci, makefile, target, job):
    """A local run that passes where CI fails wastes a pipeline to discover."""
    body = re.search(
        rf"^{re.escape(target)}:.*?\n((?:\t.*\n|\n)*)", makefile, re.MULTILINE
    ).group(1)
    variables = ci[job]["variables"]

    # MIN_TRIALS is deliberately absent from the make targets: TASK= narrows the
    # suite, which makes a suite-wide floor wrong.
    for key in ("AEVOM_EVAL_MODE", "AEVOM_EVAL_MIN_MEAN", "AEVOM_EVAL_MIN_TASK"):
        expected = str(variables[key])
        assert f"{key}={expected}" in body, (
            f"{target} does not set {key}={expected}, which is what {job} uses"
        )
