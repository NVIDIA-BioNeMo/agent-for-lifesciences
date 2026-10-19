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

"""Tests for the [SESSION LOG] block in the injected prompt envelope.

The block has to strip and re-inject cleanly. The training agent removes
everything from ``INJECTION_SENTINEL`` onward before injecting fresh state, so
a block emitted ahead of that marker would survive stripping and accumulate one
stale copy per iteration -- which is both a slow context leak and a set of
contradictory session logs in the same prompt.
"""

from api.state.format import (
    INJECTION_SENTINEL,
    format_session_log,
    inject_state_into_prompt,
)

FINDINGS = [
    {
        "turn_seq": 1,
        "iteration": 2,
        "summary": "vina_dock(lig=sdf_a1b2c3d4) -> -9.50 kcal/mol [pdb_c3c3c3c3]",
    },
    {
        "turn_seq": 1,
        "iteration": 4,
        "summary": "vina_dock(lig=sdf_d4d4d4d4) -> -6.10 kcal/mol [pdb_e5e5e5e5]",
    },
]

EMPTY_VIEWER = {"is_empty": True}
EMPTY_STORAGE = {"is_empty": True}


def _strip(content: str) -> str:
    """Mirror _strip_previous_injection in the training agent."""
    index = content.find(INJECTION_SENTINEL)
    return content if index == -1 else content[:index]


class TestRendering:
    def test_measurements_appear_verbatim(self):
        out = format_session_log(FINDINGS)
        assert "-9.50 kcal/mol" in out
        assert "-6.10 kcal/mol" in out

    def test_empty_ledger_renders_nothing(self):
        assert format_session_log([]) == ""

    def test_truncation_states_what_it_hid(self):
        out = format_session_log(FINDINGS, max_entries=1)
        assert "most recent 1 of 2" in out


class TestEnvelopeIntegration:
    def test_the_block_lands_in_the_prompt(self):
        prompt = inject_state_into_prompt(
            "BASE", EMPTY_VIEWER, EMPTY_STORAGE, [], findings=FINDINGS
        )
        assert "[SESSION LOG]" in prompt
        assert "-9.50 kcal/mol" in prompt

    def test_no_block_when_nothing_has_happened(self):
        prompt = inject_state_into_prompt("BASE", EMPTY_VIEWER, EMPTY_STORAGE, [])
        assert "[SESSION LOG]" not in prompt

    def test_the_envelope_still_begins_at_the_sentinel(self):
        prompt = inject_state_into_prompt(
            "BASE", EMPTY_VIEWER, EMPTY_STORAGE, [], findings=FINDINGS
        )
        assert INJECTION_SENTINEL in prompt
        assert prompt.index(INJECTION_SENTINEL) == len("BASE")


class TestStripAndReinject:
    def test_stripping_removes_the_session_log_too(self):
        prompt = inject_state_into_prompt(
            "BASE", EMPTY_VIEWER, EMPTY_STORAGE, [], findings=FINDINGS
        )
        assert _strip(prompt) == "BASE"

    def test_re_injecting_does_not_stack_envelopes(self):
        """The training agent injects once per iteration into the same message."""
        prompt = "BASE"
        for _ in range(5):
            prompt = inject_state_into_prompt(
                _strip(prompt),
                EMPTY_VIEWER,
                EMPTY_STORAGE,
                [],
                findings=FINDINGS,
            )
        assert prompt.count("[SESSION LOG]") == 1
        assert prompt.count("[VIEWER STATE:") == 1

    def test_a_later_iteration_shows_the_newer_ledger(self):
        prompt = inject_state_into_prompt(
            "BASE", EMPTY_VIEWER, EMPTY_STORAGE, [], findings=FINDINGS[:1]
        )
        prompt = inject_state_into_prompt(
            _strip(prompt), EMPTY_VIEWER, EMPTY_STORAGE, [], findings=FINDINGS
        )
        assert prompt.count("-9.50 kcal/mol") == 1
        assert "-6.10 kcal/mol" in prompt

    def test_a_prompt_that_never_had_an_envelope_is_untouched(self):
        assert _strip("BASE") == "BASE"
