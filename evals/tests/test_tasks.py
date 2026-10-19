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

"""Structural checks on the eval task definitions.

A malformed task fails at job start, after Harbor has already built images and
brought a stack up, so these run in the fast suite instead. They also pin the
things that are easy to get subtly wrong: the relative ``include`` path back to
the shared Compose file, and the expectation keys the scorer actually reads.
"""

from __future__ import annotations

import stat
import tomllib
from pathlib import Path

import pytest

from aevom_eval import scoring

REPO_ROOT = Path(__file__).resolve().parents[2]
TASKS_DIR = REPO_ROOT / "evals" / "tasks"
EVAL_COMPOSE = REPO_ROOT / "docker-compose.eval.yml"

# Derived from the scorer rather than restated here. A hand-maintained copy
# would have to be updated in lockstep with the scorer, and the failure mode of
# forgetting is silent: a task declares a key, the scorer ignores it, and the
# assertion the author wrote never runs. This still catches the case the check
# exists for -- a typo in a task.toml naming a key nothing reads.
KNOWN_EXPECT_KEYS = set(scoring.EXPECT_KEY_DIMENSIONS)


def task_dirs() -> list[Path]:
    return sorted(path for path in TASKS_DIR.iterdir() if (path / "task.toml").exists())


def task_ids() -> list[str]:
    return [path.name for path in task_dirs()]


@pytest.fixture(params=task_dirs(), ids=task_ids())
def task_dir(request) -> Path:
    return request.param


@pytest.fixture
def config(task_dir) -> dict:
    return tomllib.loads((task_dir / "task.toml").read_text())


def test_there_are_tasks():
    assert task_dirs(), f"no task definitions found under {TASKS_DIR}"


def test_required_files_exist(task_dir):
    """Harbor validates these at job start; catching it here is much cheaper."""
    for relative in (
        "task.toml",
        "instruction.md",
        "environment/docker-compose.yaml",
        "tests/test.sh",
    ):
        assert (task_dir / relative).exists(), f"{task_dir.name} is missing {relative}"


def test_instruction_is_not_empty(task_dir):
    assert (task_dir / "instruction.md").read_text().strip()


def test_compose_include_resolves_to_the_shared_stack(task_dir):
    """The include path is relative and four levels deep -- easy to get wrong."""
    compose_path = task_dir / "environment" / "docker-compose.yaml"
    text = compose_path.read_text()
    assert "include:" in text, (
        f"{task_dir.name} does not include the shared eval stack; it would run "
        "with only Harbor's main container and no Aevom"
    )

    included = [
        line.split("path:", 1)[1].strip()
        for line in text.splitlines()
        if "path:" in line
    ]
    assert included, f"{task_dir.name} has an include block with no path"
    for relative in included:
        resolved = (compose_path.parent / relative).resolve()
        assert resolved == EVAL_COMPOSE.resolve(), (
            f"{task_dir.name} includes {resolved}, expected {EVAL_COMPOSE}"
        )


def test_task_compose_does_not_define_main(task_dir):
    """Harbor owns `main` via [environment].docker_image; redefining it collides."""
    text = (task_dir / "environment" / "docker-compose.yaml").read_text()
    assert "  main:" not in text, (
        f"{task_dir.name} defines a `main` service, which would override the "
        "image Harbor builds for the harness container"
    )


def test_environment_declares_a_python_image(config, task_dir):
    """The driver runs `python -m ...` inside `main`."""
    image = (config.get("environment") or {}).get("docker_image")
    assert image, f"{task_dir.name} must set [environment].docker_image"
    assert "python" in image, (
        f"{task_dir.name} uses {image!r}; the harness driver needs a Python "
        "interpreter in the main container"
    )


def test_expectations_exist_and_are_understood(config, task_dir):
    expect = (config.get("metadata") or {}).get("expect")
    assert expect, f"{task_dir.name} has no [metadata.expect]; it would grade nothing"

    unknown = sorted(set(expect) - KNOWN_EXPECT_KEYS)
    assert not unknown, (
        f"{task_dir.name} declares expectation keys the scorer ignores: {unknown}"
    )


def test_expectations_produce_a_scoreable_task(config, task_dir):
    """Every task must declare something the scorer will actually grade."""
    expect = config["metadata"]["expect"]
    dimensions = scoring._declared_dimensions(expect)
    assert dimensions, (
        f"{task_dir.name} declares expectations that map to no reward dimension"
    )


def test_answer_numbers_are_well_formed(config, task_dir):
    for spec in config["metadata"]["expect"].get("answer_numbers") or []:
        if isinstance(spec, dict):
            assert "value" in spec, f"{task_dir.name}: answer_numbers entry needs value"
            assert float(spec.get("tolerance", 0)) >= 0
        else:
            assert isinstance(spec, (int, float))


def test_tool_argument_specs_are_well_formed(config, task_dir):
    known = {"tool", "arg", "example", "present", "equals", "matches", "min_items"}
    for spec in config["metadata"]["expect"].get("required_tool_arguments") or []:
        assert "tool" in spec and "arg" in spec, (
            f"{task_dir.name}: a required_tool_arguments entry needs tool and arg"
        )
        unknown = sorted(set(spec) - known)
        assert not unknown, (
            f"{task_dir.name}: {spec['tool']}.{spec['arg']} declares {unknown}, "
            "which the scorer ignores"
        )
        predicates = sorted(set(spec) & {"present", "equals", "matches", "min_items"})
        assert predicates, (
            f"{task_dir.name}: {spec['tool']}.{spec['arg']} declares no predicate, "
            "so it asserts nothing beyond the tool having been called"
        )


def test_tool_argument_examples_satisfy_their_own_predicates(config, task_dir):
    """A regex that does not accept the call the task is asking for grades nothing.

    ``matches`` is the one predicate that can be silently wrong in a direction
    nothing else catches: a typo makes it reject every trajectory, and the task
    then fails every run for a reason that looks like a model regression. Making
    the author write the value they have in mind turns that into a unit failure.
    """
    for spec in config["metadata"]["expect"].get("required_tool_arguments") or []:
        label = f"{task_dir.name}: {spec['tool']}.{spec['arg']}"
        if spec.get("present") is False:
            assert "example" not in spec, (
                f"{label} expects the argument to be omitted, so an example value "
                "would contradict it"
            )
            continue
        assert "example" in spec, (
            f"{label} needs an `example` of a call that should pass, both as "
            "documentation and so this check can verify the predicates accept it"
        )
        passed, reason = scoring._argument_matches(spec["example"], spec)
        assert passed, f"{label}: its own example is rejected -- {reason}"


def test_expectations_score_a_perfect_trajectory(config, task_dir):
    """A task whose own expectations cannot all pass is a broken task.

    Guards against contradictions like requiring a tool and forbidding it, or a
    numeric tolerance that no value can satisfy.
    """
    expect = config["metadata"]["expect"]
    perfect = _synthesise_passing_trajectory(expect)
    score = scoring.score_trajectory(perfect, expect)
    assert score.rewards[scoring.OVERALL] == 1.0, (
        f"{task_dir.name} cannot be satisfied even by a constructed ideal run: "
        + "; ".join(f"{c.name}: {c.detail}" for c in score.failures)
    )


def test_test_scripts_are_identical_across_tasks():
    """They are copies by necessity (Harbor uploads each task's own tests dir).

    Pinning them together means a fix to one is a fix to all, instead of three
    stale copies nobody notices.
    """
    scripts = {
        path.name: (path / "tests" / "test.sh").read_text() for path in task_dirs()
    }
    reference_name = sorted(scripts)[0]
    reference = scripts[reference_name]
    drifted = sorted(name for name, text in scripts.items() if text != reference)
    assert not drifted, (
        f"tests/test.sh in {drifted} differs from {reference_name}'s; copy one "
        "over the others rather than maintaining variants"
    )


def test_test_scripts_do_not_report_the_gated_reward_key():
    """The liveness check must not be able to satisfy the CI gate.

    It cannot grade correctness -- expectations live in task.toml, which is not
    in the container -- so reporting `reward` would let a misconfigured job pass.
    """
    for path in task_dirs():
        text = (path / "tests" / "test.sh").read_text()
        assert '"harness_ok"' in text
        assert '"reward"' not in text, (
            f"{path.name}/tests/test.sh writes a `reward` key; a job that "
            "skipped --verifier would then satisfy gate.py"
        )


def test_test_scripts_are_executable():
    """Harbor chmods when needed, but a non-executable script is a surprise."""
    for path in task_dirs():
        script = path / "tests" / "test.sh"
        mode = script.stat().st_mode
        assert mode & stat.S_IXUSR, f"{script} is not executable"


def _synthesise_passing_trajectory(expect: dict) -> dict:
    """Build the run a task is asking for, so the task can be checked for sense."""
    tools = list(expect.get("required_tool_sequence") or []) + list(
        expect.get("required_tools") or []
    )
    if expect.get("structure_query_must_succeed") and "structure_query" not in tools:
        tools.append("structure_query")

    # Arguments the ideal run would have passed, keyed by tool. A spec that names
    # a tool no other expectation requires still has to be reachable, or the
    # sense-check would fail on a task that is merely under-specified.
    arguments: dict[str, dict] = {}
    for spec in expect.get("required_tool_arguments") or []:
        if spec["tool"] not in tools:
            tools.append(spec["tool"])
        arguments.setdefault(spec["tool"], {})
        if "example" in spec:
            arguments[spec["tool"]][spec["arg"]] = spec["example"]

    answer_parts = list(expect.get("answer_contains") or [])
    for group in expect.get("answer_contains_any") or []:
        answer_parts.append(str(group[0]))
    for spec in expect.get("answer_numbers") or []:
        value = spec["value"] if isinstance(spec, dict) else spec
        answer_parts.append(str(value))
    answer = " ".join(str(part) for part in answer_parts)
    answer += " " * max(0, int(expect.get("answer_min_chars") or 0) - len(answer))

    residues = int(expect.get("min_selection_residues") or 0)
    artifact_types = list(expect.get("required_artifact_types") or [])
    artifacts = [{"data_type": t, "reference": f"{t}_0"} for t in artifact_types]
    while len(artifacts) < int(expect.get("min_artifacts") or 0):
        artifacts.append({"data_type": "pdb", "reference": f"pdb_{len(artifacts)}"})

    return {
        "final_content": answer,
        "tool_calls": [
            {"name": name, "success": True, "args": arguments.get(name, {})}
            for name in tools
        ],
        "artifacts": artifacts,
        "bridge_events": [
            {
                "kind": "structure_query",
                "molql": "(sel.atom.all)",
                "atom_count": 50,
                "residue_count": max(residues, 1),
                "error": None,
            }
        ],
        "errors": [],
        "hit_iteration_limit": False,
    }
