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

"""Score an Aevom trajectory against a task's declared expectations.

Kept separate from ``verifier.py`` so it imports nothing from Harbor: the
scoring rules are the part worth unit-testing, and they should be testable
without a Harbor install or a running stack.

Expectations live in the task's ``[metadata.expect]`` block in ``task.toml``,
which keeps one statement of what a task means next to the instruction itself.

Only *declared* dimensions are scored. A task that says nothing about artifacts
gets no ``artifacts`` key, rather than a vacuous 1.0 that would inflate the mean
and make an unrelated task look better than it is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# Reward dimension names, in the order they are reported.
TOOL_SELECTION = "tool_selection"
ARTIFACTS = "artifacts"
ANSWER = "answer"
NO_ERRORS = "no_errors"
# Scores the job and policy outcomes a call reported (ADR 0001). Fed by
# api.telemetry_contract.ToolInvocationEvent, which reaches the trajectory via
# the SSE tool_call_end payload; vina_dock is the tool that exercises it, being
# the one wired through JobManager.
JOB_AND_POLICY = "job_and_policy"
OVERALL = "reward"

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Check:
    """One assertion and how it turned out.

    ``detail`` is what a human reads when a task starts failing, so it names the
    observed value and not just the verdict.
    """

    dimension: str
    name: str
    passed: bool
    detail: str
    evaluable: bool = True
    """Whether the harness could observe what this check asks about.

    A check the trajectory carries no evidence for is not a failure -- it is a
    measurement gap. Scoring it as 0 would report missing plumbing as a quality
    regression, which is exactly what happened when the job_and_policy
    dimension shipped before the fields it reads were on the wire.
    """


@dataclass
class Score:
    rewards: dict[str, float] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rewards": self.rewards,
            "checks": [
                {
                    "dimension": check.dimension,
                    "name": check.name,
                    "passed": check.passed,
                    "detail": check.detail,
                }
                for check in self.checks
            ],
        }

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if not check.passed]


def reward_details(score: Score) -> dict:
    """Project a score into the shape Harbor's viewer reward-details panel parses.

    The viewer wants a mapping of name to ``{kind, score, criteria}``, each
    criterion being ``{name, value, weight}`` plus an optional ``description``. It
    validates that shape and quietly falls back to a raw JSON dump when anything
    is off, so a typo here costs a panel with no error to explain it. That is what
    test_reward_details.py checks.
    """
    details: dict[str, dict] = {}
    for dimension, reward in score.rewards.items():
        if dimension == OVERALL:
            # The overall reward has no checks of its own, being the mean over the
            # declared dimensions, so its criteria are those dimensions -- which is
            # also the most useful thing to show against it.
            criteria = [
                {"name": name, "value": value, "weight": 1.0}
                for name, value in score.rewards.items()
                if name != OVERALL
            ]
        else:
            criteria = [
                {
                    "name": check.name,
                    "value": 1.0 if check.passed else 0.0,
                    "weight": 1.0,
                    "description": check.detail,
                }
                for check in score.checks
                if check.dimension == dimension
            ]
        details[dimension] = {
            "kind": "aevom_expectations",
            "score": reward,
            "criteria": criteria,
        }
    return details


def score_trajectory(trajectory: dict | None, expect: dict[str, Any]) -> Score:
    """Grade one trajectory. A missing trajectory scores zero on every dimension."""
    checks: list[Check] = []

    if not trajectory:
        # Distinguishable from a bad answer: the harness never produced evidence.
        dimensions = _declared_dimensions(expect) or {NO_ERRORS}
        return Score(
            rewards={**{name: 0.0 for name in dimensions}, OVERALL: 0.0},
            checks=[
                Check(
                    NO_ERRORS,
                    "trajectory_present",
                    False,
                    "no trajectory was recorded; the harness failed to drive Aevom",
                )
            ],
        )

    checks += _check_tools(trajectory, expect)
    checks += _check_artifacts(trajectory, expect)
    checks += _check_answer(trajectory, expect)
    checks += _check_errors(trajectory, expect)
    checks += _check_job_and_policy(trajectory, expect)

    rewards: dict[str, float] = {}
    for dimension in (TOOL_SELECTION, ARTIFACTS, ANSWER, NO_ERRORS, JOB_AND_POLICY):
        relevant = [check for check in checks if check.dimension == dimension]
        if not relevant:
            continue
        scorable = [check for check in relevant if check.evaluable]
        if not scorable:
            # Declared, but nothing in the trajectory could answer it. Treated
            # like an undeclared dimension -- absent from the mean rather than
            # scored 0 -- so a harness gap reads as "not measured" instead of
            # silently deflating the run's reward.
            continue
        rewards[dimension] = sum(1.0 for c in scorable if c.passed) / len(scorable)

    # Mean over declared dimensions. Undeclared ones are absent rather than 1.0,
    # so this number means "of what this task asked for, how much held".
    rewards[OVERALL] = (
        round(sum(rewards.values()) / len(rewards), 6) if rewards else 0.0
    )
    return Score(rewards=rewards, checks=checks)


# Every `[metadata.expect]` key the scorer understands, and the dimension it
# contributes to. Module-level so `evals/tests/test_tasks.py` can check task
# files against the real mapping rather than keeping its own copy -- two lists
# that must agree are one list that eventually won't.
EXPECT_KEY_DIMENSIONS: dict[str, str] = {
    "required_tools": TOOL_SELECTION,
    "forbidden_tools": TOOL_SELECTION,
    "required_tool_sequence": TOOL_SELECTION,
    "required_tool_arguments": TOOL_SELECTION,
    "required_artifact_types": ARTIFACTS,
    "min_artifacts": ARTIFACTS,
    "answer_contains": ANSWER,
    "answer_contains_any": ANSWER,
    "answer_numbers": ANSWER,
    "answer_min_chars": ANSWER,
    "answer_excludes": ANSWER,
    "structure_query_must_succeed": ANSWER,
    "min_selection_residues": ANSWER,
    "max_failed_tool_calls": NO_ERRORS,
    "allow_iteration_limit": NO_ERRORS,
    "max_driver_errors": NO_ERRORS,
    "required_job_status": JOB_AND_POLICY,
    "policy_decision_expected": JOB_AND_POLICY,
}


def _declared_dimensions(expect: dict[str, Any]) -> set[str]:
    return {
        EXPECT_KEY_DIMENSIONS[key] for key in expect if key in EXPECT_KEY_DIMENSIONS
    }


# -- tool selection --------------------------------------------------------


def _check_tools(trajectory: dict, expect: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    calls = trajectory.get("tool_calls") or []
    names = [call.get("name") for call in calls if call.get("name")]
    used = set(names)

    for tool in expect.get("required_tools") or []:
        checks.append(
            Check(
                TOOL_SELECTION,
                f"required_tool:{tool}",
                tool in used,
                f"{tool} {'was called' if tool in used else 'was never called'}; "
                f"called: {sorted(used) or 'nothing'}",
            )
        )

    for tool in expect.get("forbidden_tools") or []:
        checks.append(
            Check(
                TOOL_SELECTION,
                f"forbidden_tool:{tool}",
                tool not in used,
                f"{tool} {'was called' if tool in used else 'was not called'}",
            )
        )

    for spec in expect.get("required_tool_arguments") or []:
        checks.append(_check_argument(calls, spec))

    sequence = expect.get("required_tool_sequence") or []
    if sequence:
        # Subsequence, not contiguous: the model is free to interleave other
        # calls, but the declared ones must happen in the declared order.
        remaining = list(sequence)
        for name in names:
            if remaining and name == remaining[0]:
                remaining.pop(0)
        checks.append(
            Check(
                TOOL_SELECTION,
                "tool_sequence",
                not remaining,
                f"expected order {sequence}; saw {names}"
                + (f"; never reached {remaining}" if remaining else ""),
            )
        )

    return checks


def _check_argument(calls: list[dict], spec: dict[str, Any]) -> Check:
    """Assert some call to a tool passed an argument the task cares about.

    Which tool got called is a weak signal for the design tools, because their
    difficulty is entirely in the arguments: ``rfdiffusion_design_backbone``
    takes the same three modes through one ``contigs`` string, and calling it at
    all says nothing about whether the model scaffolded a motif or ignored the
    input structure. The trajectory has recorded ``args`` all along.

    Satisfied by *any* call to the tool, so a model that gets the arguments wrong
    and then corrects itself passes here and pays for the retry under
    ``max_failed_tool_calls`` instead.
    """
    tool = spec["tool"]
    argument = spec["arg"]
    label = f"tool_argument:{tool}.{argument}"

    matching = [call for call in calls if call.get("name") == tool]
    if not matching:
        return Check(TOOL_SELECTION, label, False, f"{tool} was never called")

    observed = [(call.get("args") or {}).get(argument) for call in matching]
    for value in observed:
        passed, reason = _argument_matches(value, spec)
        if passed:
            return Check(TOOL_SELECTION, label, True, reason)

    _, reason = _argument_matches(observed[0], spec)
    return Check(
        TOOL_SELECTION,
        label,
        False,
        f"{reason}; over {len(matching)} call(s) {argument} was {observed!r}",
    )


def _argument_matches(value: Any, spec: dict[str, Any]) -> tuple[bool, str]:
    """Evaluate one argument against the predicates a spec declares.

    Every declared predicate must hold. An empty string or list counts as absent:
    a model that passes ``hotspot_res=[]`` has not named any hotspots.
    """
    present = value is not None and value != "" and value != []

    if "present" in spec:
        wanted = bool(spec["present"])
        if present is not wanted:
            return False, (
                f"expected the argument to be {'supplied' if wanted else 'omitted'}, "
                f"but it was {value!r}"
            )

    if "equals" in spec:
        if value != spec["equals"]:
            return False, f"expected {spec['equals']!r}, saw {value!r}"

    if "matches" in spec:
        pattern = spec["matches"]
        if not present or not re.search(pattern, str(value)):
            return False, f"{value!r} does not match /{pattern}/"

    if "min_items" in spec:
        count = len(value) if isinstance(value, (list, tuple, dict)) else 0
        if count < int(spec["min_items"]):
            return False, f"expected at least {spec['min_items']} item(s), saw {value!r}"

    return True, f"{value!r} satisfied the expectation"


# -- artifacts -------------------------------------------------------------


def _check_artifacts(trajectory: dict, expect: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    artifacts = trajectory.get("artifacts") or []
    types = [artifact.get("data_type") for artifact in artifacts]

    for data_type in expect.get("required_artifact_types") or []:
        checks.append(
            Check(
                ARTIFACTS,
                f"artifact_type:{data_type}",
                data_type in types,
                f"artifact types present: {sorted(set(t for t in types if t))}",
            )
        )

    minimum = expect.get("min_artifacts")
    if minimum is not None:
        checks.append(
            Check(
                ARTIFACTS,
                "min_artifacts",
                len(artifacts) >= int(minimum),
                f"{len(artifacts)} artifact(s) stored, wanted at least {minimum}",
            )
        )

    return checks


# -- answer ----------------------------------------------------------------


def _check_answer(trajectory: dict, expect: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    answer = trajectory.get("final_content") or ""
    folded = answer.lower()

    for needle in expect.get("answer_contains") or []:
        checks.append(
            Check(
                ANSWER,
                f"contains:{needle}",
                str(needle).lower() in folded,
                f"{needle!r} {'found' if str(needle).lower() in folded else 'absent'}",
            )
        )

    for group in expect.get("answer_contains_any") or []:
        options = [str(option) for option in group]
        hit = next((o for o in options if o.lower() in folded), None)
        checks.append(
            Check(
                ANSWER,
                f"contains_any:{'|'.join(options)}",
                hit is not None,
                f"matched {hit!r}" if hit else f"none of {options} appeared",
            )
        )

    for needle in expect.get("answer_excludes") or []:
        checks.append(
            Check(
                ANSWER,
                f"excludes:{needle}",
                str(needle).lower() not in folded,
                f"{needle!r} {'appeared' if str(needle).lower() in folded else 'absent'}",
            )
        )

    for spec in expect.get("answer_numbers") or []:
        checks.append(_check_number(answer, spec))

    minimum_chars = expect.get("answer_min_chars")
    if minimum_chars is not None:
        checks.append(
            Check(
                ANSWER,
                "answer_min_chars",
                len(answer) >= int(minimum_chars),
                f"answer is {len(answer)} chars, wanted at least {minimum_chars}",
            )
        )

    if expect.get("structure_query_must_succeed"):
        checks.append(_check_structure_query(trajectory))

    minimum_residues = expect.get("min_selection_residues")
    if minimum_residues is not None:
        best = _best_selection(trajectory)
        observed = best.get("residue_count", 0) if best else 0
        checks.append(
            Check(
                ANSWER,
                "min_selection_residues",
                observed >= int(minimum_residues),
                f"largest selection covered {observed} residue(s), "
                f"wanted at least {minimum_residues}",
            )
        )

    return checks


def _check_number(answer: str, spec: Any) -> Check:
    """Assert a numeric value appears in the answer within a tolerance.

    Substring matching is wrong for numbers: a molecular weight of 194.19 should
    accept "194.2" and reject "1194.19", and a docking affinity should accept
    -7.3 when the expectation is -7.5 +/- 0.5.
    """
    if isinstance(spec, (int, float)):
        wanted, tolerance = float(spec), 0.0
    else:
        wanted = float(spec["value"])
        tolerance = float(spec.get("tolerance", 0.0))

    found = [float(match.group()) for match in _NUMBER.finditer(answer)]
    hit = next((value for value in found if abs(value - wanted) <= tolerance), None)
    return Check(
        ANSWER,
        f"number:{wanted}",
        hit is not None,
        f"found {hit}" if hit is not None else f"no number within {tolerance} of {wanted}",
    )


def _check_structure_query(trajectory: dict) -> Check:
    """Did a viewer query actually resolve, with atoms?

    This is the regression test for the browser bridge. Before it existed every
    ``structure_query`` timed out, so the distinction that matters is not "was
    the tool called" but "did it come back with a selection".
    """
    queries = [
        event
        for event in (trajectory.get("bridge_events") or [])
        if event.get("kind") == "structure_query"
    ]
    successful = [
        query
        for query in queries
        if not query.get("error") and (query.get("atom_count") or 0) > 0
    ]
    if successful:
        best = max(successful, key=lambda q: q.get("atom_count") or 0)
        return Check(
            ANSWER,
            "structure_query_succeeded",
            True,
            f"selected {best.get('atom_count')} atoms in "
            f"{best.get('residue_count')} residue(s)",
        )
    if not queries:
        return Check(
            ANSWER,
            "structure_query_succeeded",
            False,
            "structure_query was never issued",
        )
    return Check(
        ANSWER,
        "structure_query_succeeded",
        False,
        f"{len(queries)} query/queries issued, none selected any atoms: "
        + "; ".join(
            f"{q.get('molql')!r} -> {q.get('error') or q.get('summary')}"
            for q in queries[:3]
        ),
    )


def _best_selection(trajectory: dict) -> dict | None:
    queries = [
        event
        for event in (trajectory.get("bridge_events") or [])
        if event.get("kind") == "structure_query" and not event.get("error")
    ]
    return max(queries, key=lambda q: q.get("residue_count") or 0, default=None)


# -- errors ----------------------------------------------------------------


def _check_errors(trajectory: dict, expect: dict[str, Any]) -> list[Check]:
    checks: list[Check] = []
    calls = trajectory.get("tool_calls") or []
    failed = [call for call in calls if call.get("success") is False]

    maximum = expect.get("max_failed_tool_calls")
    if maximum is not None:
        checks.append(
            Check(
                NO_ERRORS,
                "max_failed_tool_calls",
                len(failed) <= int(maximum),
                f"{len(failed)} tool call(s) failed "
                f"({sorted({c.get('name') for c in failed})}), "
                f"allowed {maximum}",
            )
        )

    if "allow_iteration_limit" in expect and not expect["allow_iteration_limit"]:
        hit = bool(trajectory.get("hit_iteration_limit"))
        checks.append(
            Check(
                NO_ERRORS,
                "iteration_limit",
                not hit,
                "hit the iteration limit" if hit else "finished within the limit",
            )
        )

    maximum_driver = expect.get("max_driver_errors")
    if maximum_driver is not None:
        errors = trajectory.get("errors") or []
        checks.append(
            Check(
                NO_ERRORS,
                "max_driver_errors",
                len(errors) <= int(maximum_driver),
                f"harness errors: {errors}" if errors else "no harness errors",
            )
        )

    return checks


# -- job and policy (P0 architecture-contracts work) ------------------------


def _check_job_and_policy(trajectory: dict, expect: dict[str, Any]) -> list[Check]:
    """Score a call's recorded ``job_id``/``policy_decision``.

    Only declared dimensions are scored -- same rule as every other check
    here. As of this writing, ``Trajectory.record()``
    (``evals/aevom_eval/driver/trajectory.py``) does not yet copy ``job_id``/
    ``policy_decision`` from the SSE ``tool_call_end`` event onto each
    ``tool_calls`` entry, because that event does not yet carry them (see
    ``api.telemetry_contract`` and ADR 0001's "Future phases" -- the
    underlying data exists in the ``jobs`` table and OTel spans today, just
    not threaded through the eval trajectory). A task that declares
    ``required_job_status``/``policy_decision_expected`` will fail every
    check until that follow-up lands; this function is implemented ahead of
    it so the dimension activates with no further scoring-code changes once
    it does.
    """
    checks: list[Check] = []
    calls = trajectory.get("tool_calls") or []

    for spec in expect.get("required_job_status") or []:
        checks.append(_check_call_field(calls, "job_status", spec["tool"], spec["status"]))

    for spec in expect.get("policy_decision_expected") or []:
        checks.append(
            _check_call_field(calls, "policy_decision", spec["tool"], spec["decision"])
        )

    return checks


def _check_call_field(
    calls: list[dict], field_name: str, tool: str, expected: Any
) -> Check:
    """Assert some call to ``tool`` reported ``field_name == expected``.

    Distinguishes three outcomes rather than two:

    - the tool ran and reported the expected value -> pass
    - the tool never ran, or ran and reported something else -> fail
    - the tool ran but no call carries the field -> *not evaluable*

    Only the third is a measurement gap: it means the backend never put the
    field on the wire (an older trajectory, or a tool that ran outside a
    context that produces invocation events), so calling it a failure would
    grade the harness rather than the agent. A tool that was never called is a
    real failure -- the agent didn't do the work -- and stays scored.
    """
    label = f"{field_name}:{tool}"
    matching = [c for c in calls if c.get("name") == tool]

    if not matching:
        return Check(JOB_AND_POLICY, label, False, f"{tool} was never called")

    if not any(field_name in c for c in matching):
        return Check(
            JOB_AND_POLICY,
            label,
            False,
            f"{tool} ran but no call reported {field_name}; the backend did not "
            "send it (trajectory predates the field, or the tool ran outside an "
            "invocation-event context)",
            evaluable=False,
        )

    observed = [c.get(field_name) for c in matching if field_name in c]
    return Check(
        JOB_AND_POLICY,
        label,
        expected in observed,
        f"{tool} {field_name} observed: {observed}",
    )
