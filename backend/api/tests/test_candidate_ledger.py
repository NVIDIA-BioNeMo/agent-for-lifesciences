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

"""Tests for the candidate ledger.

The ledger exists to answer one question reliably: of everything this session has
considered, which molecule is best? Its value depends on three properties, and
these tests are organized around them.

**Nothing is lost.** A later partial measurement must not erase an earlier one,
because a ledger that forgets is worse than no ledger -- it would be trusted.

**Nothing is transcribed.** Gate and ADMET results are captured from tool
metadata, so the agent never restates a number it could restate wrongly.

**Ranking is right.** Affinities and liabilities sort ascending; everything else
descending. Ranking backwards would confidently recommend the worst molecule.
"""

import pytest

from api.conversations import candidates as ledger
from api.tools.base import ContentType, ToolResult
from api.tools.candidates import (
    BestCandidateTool,
    QueryCandidatesTool,
    RecordCandidatesTool,
)


def gates_result(*molecules: dict) -> ToolResult:
    """A molecule_gates-shaped result for the molecules described."""
    return ToolResult.text(
        "verdicts",
        metadata={
            "verdicts": {
                molecule["smiles"]: {
                    "passed": molecule.get("passed", True),
                    "failures": molecule.get("failures", []),
                    "properties": molecule.get("properties", {}),
                    "similarity": molecule.get("similarity", {}),
                    "alerts": molecule.get("alerts", {}),
                    "gates": molecule.get("gates", []),
                }
                for molecule in molecules
            }
        },
    )


def admet_result(predictions: dict) -> ToolResult:
    return ToolResult.text("panel", metadata={"predictions": predictions})


@pytest.fixture
def rows():
    """An in-memory ledger, as training uses."""
    return []


@pytest.fixture
def bound(rows):
    """Bind the in-memory ledger for the duration of a test."""
    token = ledger.set_candidates_source(lambda: rows)
    yield rows
    ledger.reset_candidates_source(token)


def store(rows, **fields):
    return ledger.merge_in_memory(rows, fields)


class TestCaptureFromGates:
    def test_a_gate_verdict_becomes_a_candidate(self):
        captured = ledger.capture_from_result(
            "molecule_gates",
            gates_result({"smiles": "CCO", "properties": {"mw": 46.1, "logp": -0.3}}),
        )

        assert len(captured) == 1
        assert captured[0]["smiles"] == "CCO"
        assert captured[0]["properties"]["mw"] == 46.1
        assert captured[0]["passed"] is True

    def test_failures_are_kept_verbatim(self):
        # The named violation is what tells the agent which feature to change,
        # so it has to survive into the row rather than becoming a count.
        captured = ledger.capture_from_result(
            "molecule_gates",
            gates_result(
                {
                    "smiles": "CCCCCCCC",
                    "passed": False,
                    "failures": ["logP 5.2 violates <= 5"],
                }
            ),
        )

        assert captured[0]["failures"] == ["logP 5.2 violates <= 5"]
        assert captured[0]["passed"] is False

    def test_similarity_to_the_reference_lands_on_the_row(self):
        captured = ledger.capture_from_result(
            "molecule_gates",
            gates_result(
                {
                    "smiles": "CCO",
                    "similarity": {"tanimoto": 0.42, "scaffold_mcs_ratio": 0.9},
                }
            ),
        )

        assert captured[0]["properties"]["tanimoto"] == 0.42
        assert captured[0]["properties"]["scaffold_mcs_ratio"] == 0.9

    def test_per_gate_pass_fail_is_recorded(self):
        captured = ledger.capture_from_result(
            "molecule_gates",
            gates_result(
                {
                    "smiles": "CCO",
                    "gates": [
                        {"name": "mw", "value": 46.1, "bound": 600, "passed": True},
                        {"name": "logp", "value": 5.2, "bound": 5, "passed": False},
                    ],
                }
            ),
        )

        gates = {gate["name"]: gate["passed"] for gate in captured[0]["gates"]}
        assert gates == {"mw": True, "logp": False}

    def test_structural_alerts_become_a_failed_gate(self):
        captured = ledger.capture_from_result(
            "molecule_gates",
            gates_result({"smiles": "CCO", "alerts": {"pains": ["quinone_A"]}}),
        )

        alert_gates = [
            gate for gate in captured[0]["gates"] if gate["name"] == "pains_alerts"
        ]
        assert alert_gates and alert_gates[0]["passed"] is False

    def test_a_clean_alert_family_adds_no_gate(self):
        captured = ledger.capture_from_result(
            "molecule_gates",
            gates_result({"smiles": "CCO", "alerts": {"pains": [], "brenk": []}}),
        )

        assert captured[0]["gates"] == []

    def test_a_tool_that_says_nothing_about_molecules_is_ignored(self):
        assert ledger.capture_from_result("vina_dock", gates_result()) == []
        assert (
            ledger.capture_from_result(
                "molecule_gates", ToolResult.text("no metadata")
            )
            == []
        )


class TestCaptureFromAdmet:
    def test_predictions_become_scores_not_properties(self):
        """ADMET values are predictions to rank on, not measured descriptors.

        The split is what lets best_candidate offer to rank on hERG without also
        offering to rank on molecular weight.
        """
        captured = ledger.capture_from_result(
            "admet_predict", admet_result({"CCO": {"hERG": 0.12, "BBB": 0.8}})
        )

        assert captured[0]["scores"] == {"hERG": 0.12, "BBB": 0.8}
        assert "properties" not in captured[0]

    def test_non_numeric_predictions_are_dropped(self):
        captured = ledger.capture_from_result(
            "admet_predict", admet_result({"CCO": {"hERG": 0.12, "note": "n/a"}})
        )

        assert captured[0]["scores"] == {"hERG": 0.12}

    def test_a_molecule_with_no_numbers_is_not_recorded(self):
        captured = ledger.capture_from_result(
            "admet_predict", admet_result({"CCO": {"note": "failed"}})
        )

        assert captured == []


class TestMergeSemantics:
    def test_a_second_measurement_adds_rather_than_replaces(self, rows):
        store(rows, smiles="CCO", properties={"mw": 46.1})
        store(rows, smiles="CCO", scores={"hERG": 0.12})

        assert len(rows) == 1
        assert rows[0]["properties"]["mw"] == 46.1
        assert rows[0]["scores"]["hERG"] == 0.12

    def test_omitting_a_field_does_not_clear_it(self, rows):
        # The failure that would make the ledger untrustworthy: an ADMET call
        # arriving after a gate call and wiping the gate verdict.
        store(rows, smiles="CCO", passed=True, failures=[])
        store(rows, smiles="CCO", scores={"hERG": 0.12})

        assert rows[0]["passed"] is True

    def test_a_new_judgement_replaces_the_old_one(self, rows):
        # Two gate verdicts do not merge: that would produce a verdict neither
        # tool issued.
        store(rows, smiles="CCO", passed=False, failures=["logP too high"])
        store(rows, smiles="CCO", passed=True, failures=[])

        assert rows[0]["passed"] is True
        assert rows[0]["failures"] == []

    def test_origin_is_only_set_once(self, rows):
        # Where a molecule came from does not change when it is measured again.
        store(rows, smiles="CCO", origin="enumeration round 1")
        store(rows, smiles="CCO", origin="molecule_gates")

        assert rows[0]["origin"] == "enumeration round 1"

    def test_scores_merge_key_by_key(self, rows):
        store(rows, smiles="CCO", scores={"hERG": 0.12})
        store(rows, smiles="CCO", scores={"affinity_kcal_mol": -9.4})

        assert rows[0]["scores"] == {"hERG": 0.12, "affinity_kcal_mol": -9.4}

    def test_an_entry_without_a_molecule_is_dropped(self, rows):
        store(rows, smiles="", properties={"mw": 1})

        assert rows == []


class TestStoreContract:
    """The SQL store and the shared merge have to agree on field names.

    ``upsert_candidate`` passes ``apply_update``'s output straight into the ORM
    model, both as constructor keywords and as attributes, and reads the stored
    row back through ``to_dict()``. Nothing in the type system holds those three
    together, and no test in this repo exercises a real database, so the
    agreement is checked directly. A rename that broke it would otherwise fail
    only in deployment, and silently -- the recorder swallows its own errors.
    """

    def test_merged_fields_are_all_columns_on_the_model(self):
        from api.db.models import Candidate

        merged = ledger.apply_update(None, {})
        columns = {column.key for column in Candidate.__table__.columns}

        assert set(merged) <= columns

    def test_merged_fields_all_survive_to_dict(self):
        from api.db.models import Candidate

        merged = ledger.apply_update(None, {})
        serialized = Candidate(session_id="s1", smiles="CCO", **merged).to_dict()

        assert set(merged) <= set(serialized)

    def test_a_stored_row_can_be_merged_again(self):
        """to_dict() output has to be valid input to the next merge."""
        from api.db.models import Candidate

        first = ledger.apply_update(None, {"passed": True, "properties": {"mw": 46.1}})
        stored = Candidate(session_id="s1", smiles="CCO", **first).to_dict()

        second = ledger.apply_update(stored, {"scores": {"hERG": 0.1}})

        assert second["passed"] is True
        assert second["properties"]["mw"] == 46.1
        assert second["scores"]["hERG"] == 0.1


class TestRanking:
    def test_lower_is_better_for_affinities(self):
        assert ledger.is_lower_better("affinity_kcal_mol")
        assert ledger.is_lower_better("log_ic50_um")

    def test_higher_is_better_for_everything_else(self):
        assert not ledger.is_lower_better("qed")
        assert not ledger.is_lower_better("pic50")

    def test_affinity_ranks_most_negative_first(self):
        rows = [
            {"smiles": "weak", "scores": {"affinity_kcal_mol": -6.1}},
            {"smiles": "strong", "scores": {"affinity_kcal_mol": -9.8}},
        ]

        assert ledger.rank(rows, score="affinity_kcal_mol")[0]["smiles"] == "strong"

    def test_qed_ranks_highest_first(self):
        rows = [
            {"smiles": "dull", "properties": {"qed": 0.3}},
            {"smiles": "nice", "properties": {"qed": 0.9}},
        ]

        assert ledger.rank(rows, score="qed")[0]["smiles"] == "nice"

    def test_a_candidate_without_the_score_ranks_last(self):
        """Unmeasured is not bad, but it cannot win a comparison it did not enter."""
        rows = [
            {"smiles": "unmeasured", "scores": {}},
            {"smiles": "measured", "scores": {"pic50": 5.0}},
        ]

        ordered = ledger.rank(rows, score="pic50")
        assert [row["smiles"] for row in ordered] == ["measured", "unmeasured"]

    def test_without_a_score_passing_molecules_come_first(self):
        rows = [
            {"smiles": "bad", "passed": False, "failures": ["a", "b"]},
            {"smiles": "good", "passed": True, "failures": []},
            {"smiles": "close", "passed": False, "failures": ["a"]},
        ]

        ordered = ledger.rank(rows)
        assert [row["smiles"] for row in ordered] == ["good", "close", "bad"]

    def test_passing_only_keeps_ungated_molecules(self):
        """Ungated is not failing, and dropping it would hide fresh enumerations."""
        rows = [
            {"smiles": "failed", "passed": False},
            {"smiles": "ungated", "passed": None},
        ]

        ordered = ledger.rank(rows, passing_only=True)
        assert [row["smiles"] for row in ordered] == ["ungated"]

    def test_a_score_is_found_in_properties_too(self):
        # The scores/properties split is about provenance, and a caller ranking
        # by logP should not have to know which bucket it landed in.
        rows = [{"smiles": "CCO", "properties": {"logp": 2.0}}]

        assert ledger.rank(rows, score="logp")[0]["smiles"] == "CCO"

    def test_score_names_lists_what_is_rankable(self):
        rows = [
            {"scores": {"pic50": 1}, "properties": {"mw": 2}},
            {"scores": {"hERG": 3}},
        ]

        assert ledger.score_names(rows) == ["hERG", "mw", "pic50"]


class TestPromptBlock:
    def test_an_empty_ledger_produces_no_block(self):
        from api.state.format import format_candidates, inject_state_into_prompt

        assert format_candidates([]) == ""
        prompt = inject_state_into_prompt(
            "BASE", {"is_empty": True}, {"is_empty": True}, []
        )
        assert "[CANDIDATES" not in prompt

    def test_the_header_counts_what_passed(self):
        from api.state.format import format_candidates

        block = format_candidates(
            [
                {"smiles": "a", "passed": True},
                {"smiles": "b", "passed": False, "failures": ["x"]},
            ]
        )

        assert "2 considered, 1 passing" in block

    def test_passing_molecules_are_listed_first(self):
        from api.state.format import format_candidates

        block = format_candidates(
            [
                {"smiles": "FAILING", "passed": False, "failures": ["x"]},
                {"smiles": "PASSING", "passed": True},
            ]
        )

        assert block.index("PASSING") < block.index("FAILING")

    def test_the_block_tells_the_agent_to_check_before_answering(self):
        # The block's whole job is preventing a worse submission than a molecule
        # already found, so it has to name the tool that prevents it.
        from api.state.format import format_candidates

        block = format_candidates([{"smiles": "a", "passed": True}])

        assert "best_candidate" in block

    def test_the_list_is_capped_and_says_so(self):
        from api.state.format import MAX_CANDIDATES_IN_PROMPT, format_candidates

        many = [
            {"smiles": f"C{index}", "passed": True}
            for index in range(MAX_CANDIDATES_IN_PROMPT + 5)
        ]

        block = format_candidates(many)

        assert len(block.splitlines()) == MAX_CANDIDATES_IN_PROMPT + 1
        assert "query_candidates" in block

    def test_a_long_smiles_is_shortened(self):
        from api.state.format import MAX_CANDIDATE_SMILES_LEN, format_candidates

        block = format_candidates([{"smiles": "C" * 200, "passed": True}])

        assert "..." in block
        assert len(block.splitlines()[1]) <= MAX_CANDIDATE_SMILES_LEN + 20

    def test_the_envelope_still_begins_at_the_sentinel(self):
        from api.state.format import INJECTION_SENTINEL, inject_state_into_prompt

        prompt = inject_state_into_prompt(
            "BASE",
            {"is_empty": True},
            {"is_empty": True},
            [],
            candidates=[{"smiles": "CCO", "passed": True}],
        )

        assert prompt.split(INJECTION_SENTINEL)[0] == "BASE"

    def test_the_block_is_in_the_never_degraded_floor(self):
        """A ledger summarized away under context pressure is not a ledger."""
        from langchain_core.messages import HumanMessage

        from api.context.assembler import ContextAssembler

        assembler = ContextAssembler("BASE")
        messages = assembler.assemble(
            {
                "messages": [HumanMessage(content="hi")],
                "candidates": [{"smiles": "CCO", "passed": True}],
            }
        )

        assert "[CANDIDATES" in messages[0].content


@pytest.mark.asyncio
class TestRecordTool:
    async def test_a_molecule_is_recorded(self, bound):
        result = await RecordCandidatesTool().execute(
            {"candidates": [{"smiles": "CCO", "origin": "enumeration"}]}
        )

        assert result.content_type != ContentType.ERROR
        assert bound[0]["smiles"] == "CCO"
        assert bound[0]["origin"] == "enumeration"

    async def test_agent_supplied_scores_are_kept(self, bound):
        await RecordCandidatesTool().execute(
            {"candidates": [{"smiles": "CCO", "scores": {"affinity_kcal_mol": -9.4}}]}
        )

        assert bound[0]["scores"]["affinity_kcal_mol"] == -9.4

    async def test_non_numeric_scores_are_ignored(self, bound):
        await RecordCandidatesTool().execute(
            {"candidates": [{"smiles": "CCO", "scores": {"affinity": "strong"}}]}
        )

        assert bound[0]["scores"] == {}

    async def test_an_empty_list_is_an_error(self, bound):
        result = await RecordCandidatesTool().execute({"candidates": []})

        assert result.content_type == ContentType.ERROR

    async def test_a_bad_entry_is_reported_without_losing_the_good_ones(self, bound):
        result = await RecordCandidatesTool().execute(
            {"candidates": [{"smiles": "CCO"}, {"smiles": ""}]}
        )

        assert result.content_type != ContentType.ERROR
        assert result.metadata["num_recorded"] == 1
        assert result.metadata["rejected"]

    async def test_recording_the_same_molecule_twice_merges(self, bound):
        tool = RecordCandidatesTool()
        await tool.execute({"candidates": [{"smiles": "CCO", "notes": "first"}]})
        await tool.execute(
            {"candidates": [{"smiles": "CCO", "scores": {"pic50": 6.0}}]}
        )

        assert len(bound) == 1
        assert bound[0]["notes"] == "first"
        assert bound[0]["scores"]["pic50"] == 6.0

    async def test_no_ledger_available_says_what_to_do_instead(self):
        result = await RecordCandidatesTool().execute(
            {"candidates": [{"smiles": "CCO"}]}
        )

        assert result.content_type == ContentType.ERROR
        assert "in your reply" in result.to_tool_response()


@pytest.mark.asyncio
class TestQueryTool:
    async def test_an_empty_ledger_explains_how_it_fills(self, bound):
        result = await QueryCandidatesTool().execute({})

        text = result.to_tool_response()
        assert "empty" in text
        assert "molecule_gates" in text

    async def test_candidates_are_listed_with_their_verdict(self, bound):
        store(bound, smiles="CCO", passed=True, properties={"mw": 46.1})
        store(bound, smiles="CCC", passed=False, failures=["logP too high"])

        result = await QueryCandidatesTool().execute({})

        text = result.to_tool_response()
        assert "pass" in text and "FAIL" in text
        assert "46.10" in text

    async def test_passing_only_filters(self, bound):
        store(bound, smiles="GOOD", passed=True)
        store(bound, smiles="BAD", passed=False, failures=["x"])

        result = await QueryCandidatesTool().execute({"passing_only": True})

        assert "GOOD" in result.to_tool_response()
        assert "BAD" not in result.to_tool_response()

    async def test_ranking_adds_the_score_column(self, bound):
        store(bound, smiles="CCO", scores={"pic50": 7.5})

        result = await QueryCandidatesTool().execute({"rank_by": "pic50"})

        assert "pic50" in result.to_tool_response()
        assert "7.50" in result.to_tool_response()

    async def test_an_unknown_score_falls_back_and_says_what_exists(self, bound):
        store(bound, smiles="CCO", scores={"pic50": 7.5})

        result = await QueryCandidatesTool().execute({"rank_by": "nonsense"})

        text = result.to_tool_response()
        assert "No candidate carries a score named 'nonsense'" in text
        assert "pic50" in text

    async def test_the_limit_is_reported_when_it_bites(self, bound):
        for index in range(5):
            store(bound, smiles=f"C{index}", passed=True)

        result = await QueryCandidatesTool().execute({"limit": 2})

        assert "and 3 more" in result.to_tool_response()

    async def test_ungated_molecules_read_as_ungated(self, bound):
        store(bound, smiles="CCO")

        assert "ungated" in (await QueryCandidatesTool().execute({})).to_tool_response()


@pytest.mark.asyncio
class TestBestCandidateTool:
    async def test_an_empty_ledger_is_reported_plainly(self, bound):
        result = await BestCandidateTool().execute({})

        assert "empty" in result.to_tool_response()

    async def test_the_best_affinity_wins(self, bound):
        store(bound, smiles="WEAK", passed=True, scores={"affinity_kcal_mol": -6.0})
        store(bound, smiles="STRONG", passed=True, scores={"affinity_kcal_mol": -9.7})

        result = await BestCandidateTool().execute({"rank_by": "affinity_kcal_mol"})

        assert result.metadata["best_smiles"] == "STRONG"
        assert "lowest affinity_kcal_mol" in result.to_tool_response()

    async def test_a_failing_molecule_cannot_win_by_default(self, bound):
        """A molecule that violates a hard constraint is not a candidate."""
        store(
            bound,
            smiles="CHEAT",
            passed=False,
            failures=["MW too high"],
            scores={"pic50": 9.9},
        )
        store(bound, smiles="HONEST", passed=True, scores={"pic50": 6.0})

        result = await BestCandidateTool().execute({"rank_by": "pic50"})

        assert result.metadata["best_smiles"] == "HONEST"

    async def test_passing_only_false_reveals_the_best_failure(self, bound):
        store(
            bound, smiles="CHEAT", passed=False, failures=["MW"], scores={"pic50": 9.9}
        )

        result = await BestCandidateTool().execute(
            {"rank_by": "pic50", "passing_only": False}
        )

        assert result.metadata["best_smiles"] == "CHEAT"
        assert "FAILS a gate" in result.to_tool_response()

    async def test_nothing_passing_suggests_a_way_forward(self, bound):
        store(bound, smiles="BAD", passed=False, failures=["MW too high"])

        result = await BestCandidateTool().execute({})

        text = result.to_tool_response()
        assert "No candidate passes" in text
        assert "passing_only=false" in text

    async def test_an_unknown_objective_is_an_error_naming_the_options(self, bound):
        store(bound, smiles="CCO", passed=True, scores={"pic50": 6.0})

        result = await BestCandidateTool().execute({"rank_by": "affinity"})

        assert result.content_type == ContentType.ERROR
        assert "pic50" in result.to_tool_response()

    async def test_without_an_objective_it_picks_on_the_gate_verdict(self, bound):
        store(bound, smiles="GOOD", passed=True)

        result = await BestCandidateTool().execute({})

        assert result.metadata["best_smiles"] == "GOOD"
        assert "passes every gate" in result.to_tool_response()

    async def test_the_runners_up_are_shown(self, bound):
        for index, affinity in enumerate([-9.5, -8.5, -7.5]):
            store(
                bound,
                smiles=f"MOL{index}",
                passed=True,
                scores={"affinity_kcal_mol": affinity},
            )

        result = await BestCandidateTool().execute({"rank_by": "affinity_kcal_mol"})

        text = result.to_tool_response()
        assert "Next best:" in text
        assert "MOL1" in text

    async def test_provenance_and_notes_are_surfaced(self, bound):
        store(
            bound,
            smiles="CCO",
            passed=True,
            origin="scaffold hop",
            notes="keeps the hinge contact",
        )

        text = (await BestCandidateTool().execute({})).to_tool_response()

        assert "scaffold hop" in text
        assert "keeps the hinge contact" in text


@pytest.mark.asyncio
class TestEndToEnd:
    async def test_gates_then_admet_then_pick(self, bound):
        """The path the agent actually walks, with nothing transcribed."""
        for row in ledger.capture_from_result(
            "molecule_gates",
            gates_result(
                {"smiles": "GOOD", "passed": True, "properties": {"mw": 350.0}},
                {
                    "smiles": "BAD",
                    "passed": False,
                    "failures": ["MW 720 violates <= 600"],
                    "properties": {"mw": 720.0},
                },
            ),
        ):
            store(bound, **row)

        for row in ledger.capture_from_result(
            "admet_predict",
            admet_result({"GOOD": {"hERG": 0.05}, "BAD": {"hERG": 0.9}}),
        ):
            store(bound, **row)

        result = await BestCandidateTool().execute({"rank_by": "hERG"})

        # BAD has the worse hERG anyway, but it is excluded on the gate, which is
        # the ordering that matters: constraints before preferences.
        assert result.metadata["best_smiles"] == "GOOD"
        assert result.metadata["num_candidates"] == 2

        # And the gate verdict survived the later ADMET write.
        good = next(row for row in bound if row["smiles"] == "GOOD")
        assert good["passed"] is True
        assert good["properties"]["mw"] == 350.0
