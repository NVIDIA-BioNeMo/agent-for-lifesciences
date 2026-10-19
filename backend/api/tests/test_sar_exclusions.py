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

"""Tests for derived SAR exclusion rules.

The rules exist to stop a disqualifying feature being rediscovered: an agent that
loses eight analogues to a catechol should not enumerate six more. Two claims have
to hold for that to work.

**The attribution is honest.** A structural alert names a substructure, so
excluding it is sound. A weight or logP violation names nothing structural, and a
rule that pretended otherwise would confidently delete good chemistry. These
tests hold that line.

**The rules follow the evidence.** Nothing is stored separately, so a rule cannot
outlive the candidate that justified it or disagree with the ledger.
"""

import pytest

from api.conversations import exclusions
from api.state.format import format_exclusions, inject_state_into_prompt


def candidate(smiles: str, *, passed: bool = False, gates=()) -> dict:
    return {"smiles": smiles, "passed": passed, "gates": list(gates)}


def alert_gate(family: str, *hits: str) -> dict:
    return {
        "name": f"{family}_alerts",
        "value": len(hits),
        "bound": 0,
        "passed": False,
        "hits": list(hits),
    }


def property_gate(name: str, value, bound, passed: bool = False) -> dict:
    return {"name": name, "value": value, "bound": bound, "passed": passed}


class TestAlertRules:
    def test_a_tripped_alert_becomes_an_exclusion(self):
        rules = exclusions.derive(
            [candidate("CCO", gates=[alert_gate("pains", "catechol_A")])]
        )

        assert len(rules) == 1
        assert rules[0].kind == "alert"
        assert rules[0].feature == "catechol_A"

    def test_offenders_accumulate_across_candidates(self):
        rules = exclusions.derive(
            [
                candidate("MOL1", gates=[alert_gate("pains", "catechol_A")]),
                candidate("MOL2", gates=[alert_gate("pains", "catechol_A")]),
            ]
        )

        assert rules[0].count == 2
        assert rules[0].offenders == ["MOL1", "MOL2"]

    def test_the_same_molecule_is_counted_once(self):
        """Re-gating a molecule must not inflate the rule that excludes it."""
        rules = exclusions.derive(
            [
                candidate("MOL1", gates=[alert_gate("pains", "catechol_A")]),
                candidate("MOL1", gates=[alert_gate("pains", "catechol_A")]),
            ]
        )

        assert rules[0].count == 1

    def test_each_alert_family_is_recognized(self):
        rules = exclusions.derive(
            [
                candidate("A", gates=[alert_gate("pains", "p")]),
                candidate("B", gates=[alert_gate("brenk", "b")]),
                candidate("C", gates=[alert_gate("nih", "n")]),
            ]
        )

        assert {rule.feature for rule in rules} == {"p", "b", "n"}

    def test_several_hits_on_one_molecule_become_several_rules(self):
        rules = exclusions.derive(
            [candidate("CCO", gates=[alert_gate("pains", "one", "two")])]
        )

        assert {rule.feature for rule in rules} == {"one", "two"}

    def test_a_gate_that_merely_looks_like_an_alert_is_not_one(self):
        # Guards against a future gate named *_alerts silently producing
        # substructure exclusions it cannot justify.
        rules = exclusions.derive(
            [
                candidate(
                    "CCO",
                    gates=[
                        {
                            "name": "made_up_alerts",
                            "passed": False,
                            "hits": ["nonsense"],
                        }
                    ],
                )
            ]
        )

        assert [rule.feature for rule in rules] == ["made_up_alerts"]
        assert rules[0].kind == "property"

    def test_the_description_says_what_to_do(self):
        rules = exclusions.derive(
            [candidate("CCO", gates=[alert_gate("pains", "catechol_A")])]
        )

        assert rules[0].describe() == (
            "Avoid the catechol_A substructure (1 candidate disqualified)"
        )


class TestPropertyRules:
    def test_a_violated_bound_becomes_an_exclusion(self):
        rules = exclusions.derive(
            [candidate("CCO", gates=[property_gate("mw", 720.0, "<= 600")])]
        )

        assert rules[0].kind == "property"
        assert rules[0].feature == "mw"
        assert rules[0].bound == "<= 600"

    def test_a_property_rule_is_never_phrased_as_a_substructure(self):
        """Molecular weight says nothing about which fragment carried the mass."""
        rules = exclusions.derive(
            [candidate("CCO", gates=[property_gate("mw", 720.0, "<= 600")])]
        )

        described = rules[0].describe()
        assert "substructure" not in described
        assert "Keep mw within <= 600" in described

    def test_the_worst_miss_is_kept_for_an_upper_bound(self):
        rules = exclusions.derive(
            [
                candidate("A", gates=[property_gate("mw", 650.0, "<= 600")]),
                candidate("B", gates=[property_gate("mw", 900.0, "<= 600")]),
            ]
        )

        assert rules[0].worst == 900.0

    def test_the_worst_miss_is_kept_for_a_lower_bound(self):
        # Direction is read off the bound rather than assumed, so a minimum
        # threshold does not report the largest value as the worst.
        rules = exclusions.derive(
            [
                candidate("A", gates=[property_gate("qed", 0.20, ">= 0.3")]),
                candidate("B", gates=[property_gate("qed", 0.05, ">= 0.3")]),
            ]
        )

        assert rules[0].worst == 0.05

    def test_a_non_numeric_value_does_not_break_the_rule(self):
        rules = exclusions.derive(
            [candidate("CCO", gates=[property_gate("charge", None, "in [-2, 2]")])]
        )

        assert rules[0].count == 1
        assert rules[0].worst is None


class TestWhatIsExcluded:
    def test_passing_candidates_contribute_nothing(self):
        assert exclusions.derive([candidate("CCO", passed=True)]) == []

    def test_ungated_candidates_contribute_nothing(self):
        assert exclusions.derive([{"smiles": "CCO", "passed": None}]) == []

    def test_a_gate_that_passed_is_not_an_exclusion(self):
        rules = exclusions.derive(
            [
                candidate(
                    "CCO",
                    gates=[
                        property_gate("mw", 350.0, "<= 600", passed=True),
                        property_gate("logp", 6.0, "<= 5"),
                    ],
                )
            ]
        )

        assert [rule.feature for rule in rules] == ["logp"]

    def test_an_empty_ledger_derives_nothing(self):
        assert exclusions.derive([]) == []


class TestOrdering:
    def test_the_costliest_rule_reads_first(self):
        rules = exclusions.derive(
            [
                candidate("A", gates=[alert_gate("pains", "rare")]),
                candidate("B", gates=[alert_gate("pains", "common")]),
                candidate("C", gates=[alert_gate("pains", "common")]),
            ]
        )

        assert [rule.feature for rule in rules] == ["common", "rare"]

    def test_alerts_outrank_properties_at_equal_cost(self):
        # An alert is actionable -- remove the group. A bound only says the
        # molecule was too something, so it is the less useful line to read.
        rules = exclusions.derive(
            [
                candidate(
                    "A",
                    gates=[
                        property_gate("mw", 720.0, "<= 600"),
                        alert_gate("pains", "catechol_A"),
                    ],
                )
            ]
        )

        assert [rule.kind for rule in rules] == ["alert", "property"]


class TestScreen:
    def test_a_known_offender_is_reported(self):
        ledger = [candidate("BAD", gates=[alert_gate("pains", "catechol_A")])]

        assert [rule.feature for rule in exclusions.screen(ledger, "BAD")] == [
            "catechol_A"
        ]

    def test_an_unseen_molecule_reports_nothing(self):
        ledger = [candidate("BAD", gates=[alert_gate("pains", "catechol_A")])]

        assert exclusions.screen(ledger, "UNSEEN") == []


class TestPromptBlock:
    def rendered(self, candidates):
        from api.graph.state import derive_exclusions

        return derive_exclusions(candidates)

    def test_no_rules_produce_no_block(self):
        assert format_exclusions([]) == ""

    def test_the_block_instructs_rather_than_reports(self):
        block = format_exclusions(
            self.rendered([candidate("A", gates=[alert_gate("pains", "catechol_A")])])
        )

        assert "Avoid the catechol_A substructure" in block
        assert "before scoring it" in block

    def test_the_list_is_capped(self):
        from api.state.format import MAX_EXCLUSIONS_IN_PROMPT

        many = [
            candidate(f"MOL{index}", gates=[alert_gate("pains", f"alert{index}")])
            for index in range(MAX_EXCLUSIONS_IN_PROMPT + 4)
        ]

        block = format_exclusions(self.rendered(many))

        assert len(block.splitlines()) == MAX_EXCLUSIONS_IN_PROMPT + 1

    def test_the_envelope_still_begins_at_the_sentinel(self):
        from api.state.format import INJECTION_SENTINEL

        prompt = inject_state_into_prompt(
            "BASE",
            {"is_empty": True},
            {"is_empty": True},
            [],
            exclusions=self.rendered(
                [candidate("A", gates=[alert_gate("pains", "catechol_A")])]
            ),
        )

        assert prompt.split(INJECTION_SENTINEL)[0] == "BASE"
        assert "[EXCLUSIONS" in prompt

    def test_the_block_is_in_the_never_degraded_floor(self):
        from langchain_core.messages import HumanMessage

        from api.context.assembler import ContextAssembler

        messages = ContextAssembler("BASE").assemble(
            {
                "messages": [HumanMessage(content="hi")],
                "exclusions": self.rendered(
                    [candidate("A", gates=[alert_gate("pains", "catechol_A")])]
                ),
            }
        )

        assert "[EXCLUSIONS" in messages[0].content

    def test_derivation_survives_malformed_rows(self):
        """A bad row must not cost the agent its whole exclusion block."""
        from api.graph.state import derive_exclusions

        assert derive_exclusions([{"passed": False, "gates": "not a list"}]) == []


@pytest.mark.asyncio
class TestGatesReportsRepeats:
    """molecule_gates names features that already cost earlier candidates.

    This is the half that changes behaviour immediately; the prompt block is the
    half that changes the next enumeration.
    """

    @pytest.fixture
    def ledger(self):
        from api.conversations.candidates import (
            reset_candidates_source,
            set_candidates_source,
        )

        rows = [candidate("EARLIER", gates=[alert_gate("pains", "catechol_A")])]
        token = set_candidates_source(lambda: rows)
        yield rows
        reset_candidates_source(token)

    async def test_a_repeated_alert_is_called_out(self, ledger):
        from api.tools.molecule_gates import MoleculeGatesTool

        lines = await MoleculeGatesTool()._repeat_offences(
            [{"passed": False, "alerts": {"pains": ["catechol_A"]}}]
        )

        assert any("stop" in line for line in lines)
        assert any("catechol_A" in line for line in lines)

    async def test_a_new_alert_is_not_called_out(self, ledger):
        from api.tools.molecule_gates import MoleculeGatesTool

        lines = await MoleculeGatesTool()._repeat_offences(
            [{"passed": False, "alerts": {"pains": ["brand_new"]}}]
        )

        assert lines == []

    async def test_a_passing_molecule_is_not_an_offender(self, ledger):
        from api.tools.molecule_gates import MoleculeGatesTool

        lines = await MoleculeGatesTool()._repeat_offences(
            [{"passed": True, "alerts": {"pains": ["catechol_A"]}}]
        )

        assert lines == []

    async def test_property_failures_are_never_called_out(self, ledger):
        """No substructure claim from a whole-molecule property violation."""
        from api.tools.molecule_gates import MoleculeGatesTool

        lines = await MoleculeGatesTool()._repeat_offences(
            [{"passed": False, "failures": ["MW 720 violates <= 600"], "alerts": {}}]
        )

        assert lines == []
