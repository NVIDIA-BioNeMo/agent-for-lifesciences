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

"""Tests for the drug-likeness gate panel.

These run inside the rdkit image, which is where the descriptor arithmetic,
sascorer's data files and the PAINS/Brenk/NIH catalogs actually exist.

Molecules chosen so a wrong answer is visible: ibuprofen and imatinib clear the
whole panel, cyclosporine fails on size and flexibility, p-terphenyl fails on
lipophilicity alone (logP 5.02, a fraction over the bound), hexachlorobenzene
fails on lipophilicity AND the halogenated-ring alerts, and a quinone is the
canonical PAINS hit. Caffeine is
here as the instructive near-miss -- its Crippen logP is -1.03, a hundredth below
the panel's floor, which is why every threshold is overridable.
"""

import pytest
from fastapi.testclient import TestClient

from app.gates import (
    DEFAULT_THRESHOLDS,
    OPTIONAL_THRESHOLDS,
    _UPPER_BOUNDS,
    _range_gate,
    MAX_MOLECULES,
    evaluate,
    evaluate_molecule,
    resolve_thresholds,
)
from app.main import app

ETHANOL = "CCO"
CAFFEINE = "Cn1cnc2c1c(=O)n(C)c(=O)n2C"
IBUPROFEN = "CC(C)Cc1ccc(cc1)C(C)C(=O)O"
IMATINIB = (
    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
)
# Fails the molecular weight and rotatable bond ceilings by a wide margin.
CYCLOSPORINE = (
    "CC[C@H]1NC(=O)[C@@H]([C@@H](O)[C@H](C)C/C=C/C)N(C)C(=O)[C@H](C(C)C)"
    "N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](C)"
    "NC(=O)[C@H](C)NC(=O)[C@@H](CC(C)C)N(C)C(=O)[C@@H](NC(=O)[C@@H](CC(C)C)"
    "N(C)C(=O)CN(C)C1=O)C(C)C"
)
# logP 5.61, and comfortably inside every other bound -- so it isolates the
# lipophilicity gate. A long alkane would do too, except it also busts the
# rotatable-bond ceiling.
# p-terphenyl: logP 5.02 against a bound of 5, and nothing else. The single
# clean logP failure in the set, which is what the "fails only on X" tests need.
LIPOPHILE = "c1ccc(-c2ccc(-c3ccccc3)cc2)cc1"
# Hexachlorobenzene: lipophilic AND carrying two halogenated-ring Brenk alerts.
HALOGENATED = "Clc1c(Cl)c(Cl)c(Cl)c(Cl)c1Cl"
# 1,4-benzoquinone: the textbook PAINS substructure.
QUINONE = "O=C1C=CC(=O)C=C1"


def one(smiles: str, **kwargs):
    """Score a single molecule, asserting the request itself succeeded."""
    report, error = evaluate([smiles], **kwargs)
    assert error is None, error
    assert report is not None
    assert report.results, f"{smiles} was not scored"
    return report.results[0]


class TestWhereTheBoundLies:
    """Gating decisions are made on the exact value, against the right operator.

    Both halves of this were wrong, and both failed the same way: the gate said
    pass where the published panel says fail, quietly. Descriptors were rounded
    to two decimals BEFORE being compared, so a true SA of 4.502 became 4.50 and
    cleared a "< 4.5" bar; and every upper bound was inclusive, so a value
    sitting exactly on a strict bound passed as well. A caller reading the
    all-pass verdict had no way to see either.
    """

    def test_a_strict_bound_refuses_a_value_sitting_on_it(self):
        gate = _range_gate("mw", "molecular weight", 600.0, None, 600.0, exclusive_high=True)

        assert gate.passed is False
        assert gate.bound == "< 600"

    def test_an_inclusive_bound_accepts_a_value_sitting_on_it(self):
        gate = _range_gate("hbd", "H-bond donors", 5, None, 5, exclusive_high=False)

        assert gate.passed is True
        assert gate.bound == "<= 5"

    def test_a_value_just_over_a_strict_bound_is_refused_not_rounded_in(self):
        """4.502 displays as 4.50. The verdict must not be made on the printout."""
        gate = _range_gate("sa_score", "synthetic accessibility", 4.502, None, 4.5, exclusive_high=True)

        assert gate.passed is False

    def test_a_value_just_under_a_strict_bound_still_passes(self):
        gate = _range_gate("sa_score", "synthetic accessibility", 4.498, None, 4.5, exclusive_high=True)

        assert gate.passed is True

    def test_the_rendered_value_is_readable_without_deciding_anything(self):
        gate = _range_gate("mw", "molecular weight", 341.76999999999998, None, 600.0, exclusive_high=True)

        assert gate.passed is True
        assert gate.to_dict()["value"] == 341.77

    def test_the_published_strict_bounds_are_the_strict_ones(self):
        exclusive = {name for name, _, _, _, strict in _UPPER_BOUNDS if strict}

        assert exclusive == {"mw", "tpsa", "sa_score"}


class TestBrenkIsNotOptional:
    """It was grouped with the similarity bounds, whose comment says they only
    make sense against a reference. A structural-alert catalogue needs no
    reference, and the published panel applies it unconditionally."""

    def test_brenk_is_applied_by_default(self):
        thresholds, _ = resolve_thresholds(None)

        assert thresholds["brenk_max"] == 0

    def test_brenk_is_no_longer_listed_as_needing_a_reference(self):
        assert "brenk_max" not in OPTIONAL_THRESHOLDS

    def test_the_bounds_that_really_need_a_reference_stay_optional(self):
        """Similarity has no right default: lead optimisation wants a close
        analogue, scaffold hopping wants a distant one."""
        assert "similarity_min" in OPTIONAL_THRESHOLDS
        assert "similarity_max" in OPTIONAL_THRESHOLDS
        assert "similarity_min" not in DEFAULT_THRESHOLDS


class TestThresholdResolution:
    def test_defaults_are_the_published_panel(self):
        thresholds, unknown = resolve_thresholds(None)
        assert unknown == []
        assert thresholds == DEFAULT_THRESHOLDS
        assert thresholds["mw_max"] == 600.0
        assert thresholds["sa_score_max"] == 4.5

    def test_override_replaces_one_bound_and_leaves_the_rest(self):
        thresholds, _ = resolve_thresholds({"mw_max": 500})
        assert thresholds["mw_max"] == 500
        assert thresholds["logp_max"] == DEFAULT_THRESHOLDS["logp_max"]

    def test_none_disables_a_gate(self):
        thresholds, _ = resolve_thresholds({"mw_max": None})
        assert "mw_max" not in thresholds

    def test_unknown_names_are_reported_not_swallowed(self):
        # A typo'd threshold silently doing nothing is the failure mode this
        # guards: the caller would read the all-pass verdict as meaningful.
        _, unknown = resolve_thresholds({"mw_maximum": 500})
        assert unknown == ["mw_maximum"]

    def test_optional_bounds_are_accepted(self):
        thresholds, unknown = resolve_thresholds({"similarity_min": 0.4})
        assert unknown == []
        assert thresholds["similarity_min"] == 0.4


class TestDescriptors:
    def test_computes_the_full_panel(self):
        result = one(CAFFEINE)
        properties = result["properties"]
        assert properties["mw"] == pytest.approx(194.19, abs=0.05)
        assert properties["hbd"] == 0
        assert properties["formal_charge"] == 0
        assert properties["num_aromatic_rings"] == 2
        assert properties["molecular_formula"] == "C8H10N4O2"
        assert 0.0 <= properties["qed"] <= 1.0

    def test_synthetic_accessibility_is_available(self):
        # sascorer lives in RDKit's Contrib tree rather than the package, so a
        # None here means the image lost the Contrib symlink.
        assert one(ETHANOL)["properties"]["sa_score"] is not None

    def test_simple_molecules_score_easier_to_make_than_complex_ones(self):
        assert (
            one(ETHANOL)["properties"]["sa_score"]
            < one(CYCLOSPORINE)["properties"]["sa_score"]
        )

    def test_canonicalizes_the_input(self):
        assert one("C(C)O")["canonical_smiles"] == "CCO"


class TestVerdicts:
    def test_a_clean_molecule_passes_every_gate(self):
        result = one(IBUPROFEN)
        assert result["passed"]
        assert result["failures"] == []

    def test_a_marketed_drug_inside_the_panel_passes(self):
        # 494 Da with seven rotatable bonds: near the ceilings, under all of them.
        assert one(IMATINIB)["passed"]

    def test_a_polar_drug_fails_the_panels_logp_floor(self):
        # Caffeine at -1.03 is a hundredth outside a published bound. Not a bug:
        # the reason every threshold is overridable, and worth pinning so the
        # sharp edge is documented rather than discovered.
        assert not one(CAFFEINE)["passed"]
        assert one(CAFFEINE, thresholds={"logp_min": -2})["passed"]

    def test_an_oversized_molecule_fails_and_says_which_gates(self):
        result = one(CYCLOSPORINE)
        assert not result["passed"]
        failures = " ".join(result["failures"])
        assert "molecular weight" in failures
        assert "rotatable bonds" in failures

    def test_a_lipophilic_molecule_fails_only_on_logp(self):
        result = one(LIPOPHILE)
        assert not result["passed"]
        assert [gate["name"] for gate in result["gates"] if not gate["passed"]] == [
            "logp"
        ]

    def test_a_molecule_can_fail_two_gates_and_both_are_named(self):
        # Hexachlorobenzene is lipophilic and carries halogenated-ring alerts.
        # Reporting only the first would hide half the reason it is unusable.
        failed = [g["name"] for g in one(HALOGENATED)["gates"] if not g["passed"]]

        assert failed == ["logp", "brenk"]

    def test_failures_name_the_violated_bound(self):
        # "failed" is not actionable; "logP 5.61 violates -1 to 5" is.
        failure = one(LIPOPHILE)["failures"][0]
        assert "logP" in failure
        assert "-1 to 5" in failure

    def test_a_loosened_threshold_turns_a_failure_into_a_pass(self):
        assert not one(LIPOPHILE)["passed"]
        assert one(LIPOPHILE, thresholds={"logp_max": 12})["passed"]

    def test_a_disabled_gate_does_not_appear(self):
        result = one(LIPOPHILE, thresholds={"logp_min": None, "logp_max": None})
        assert result["passed"]
        assert "logp" not in [gate["name"] for gate in result["gates"]]

    def test_every_gate_carries_its_bound_and_measured_value(self):
        for gate in one(IBUPROFEN)["gates"]:
            assert gate["bound"]
            assert gate["value"] is not None


class TestStructuralAlerts:
    def test_flags_a_pains_substructure(self):
        result = one(QUINONE)
        assert result["alerts"]["pains"]
        assert not result["passed"]

    def test_a_clean_molecule_trips_nothing(self):
        assert one(IBUPROFEN)["alerts"]["pains"] == []

    def test_brenk_gates_by_default_and_nih_does_not(self):
        # Brenk is a structural-alert catalogue: it needs no reference molecule
        # and the published panel applies it unconditionally. NIH stays
        # advisory -- it is reported so a caller can see it, and gates only
        # when asked for.
        result = one(HALOGENATED)

        assert result["alerts"]["brenk"]
        gated = [gate["name"] for gate in result["gates"]]
        assert "brenk" in gated
        assert "nih" not in gated

    def test_brenk_gates_when_the_caller_asks(self):
        result = one("O=[N+]([O-])c1ccccc1", thresholds={"brenk_max": 0})
        assert not result["passed"]
        assert "Brenk" in " ".join(result["failures"])

    def test_a_failing_alert_gate_names_the_substructure(self):
        gate = next(
            gate for gate in one(QUINONE)["gates"] if gate["name"] == "pains"
        )
        assert gate["detail"]


class TestScaffoldsAndSimilarity:
    def test_reports_the_murcko_scaffold(self):
        assert one(CAFFEINE)["similarity"]["murcko_scaffold"]

    def test_an_acyclic_molecule_has_no_scaffold(self):
        assert one(ETHANOL)["similarity"]["murcko_scaffold"] == ""

    def test_a_molecule_is_maximally_similar_to_itself(self):
        result = one(CAFFEINE, reference_smiles=CAFFEINE)
        assert result["similarity"]["tanimoto"] == 1.0
        assert result["similarity"]["scaffold_mcs_ratio"] == 1.0
        assert result["similarity"]["same_scaffold"] is True

    def test_an_unrelated_molecule_scores_low(self):
        assert one(LIPOPHILE, reference_smiles=CAFFEINE)["similarity"]["tanimoto"] < 0.2

    def test_a_close_analog_scores_between(self):
        # Removing one N-methyl from caffeine gives theobromine: similar, not
        # identical, and on the same scaffold.
        theobromine = "Cn1cnc2c1c(=O)[nH]c(=O)n2C"
        similarity = one(theobromine, reference_smiles=CAFFEINE)["similarity"]
        assert 0.3 < similarity["tanimoto"] < 1.0
        assert similarity["scaffold_mcs_ratio"] > 0.8

    def test_a_similarity_floor_rejects_a_distant_molecule(self):
        # The lead-optimization direction: a molecule that clears every physical
        # gate is still wrong if it has stopped resembling the parent.
        result = one(
            LIPOPHILE,
            reference_smiles=CAFFEINE,
            thresholds={"similarity_min": 0.4, "logp_max": None},
        )
        assert not result["passed"]
        # One failure, and it is the similarity one -- the physical gates were
        # all cleared. The measured value is not pinned here: it belongs to
        # whichever molecule LIPOPHILE is, and pinning it makes this test fail
        # for a reason that has nothing to do with what it is checking.
        assert len(result["failures"]) == 1
        assert result["failures"][0].startswith("Tanimoto to reference ")
        assert result["failures"][0].endswith("violates >= 0.4")

    def test_a_similarity_ceiling_rejects_a_near_duplicate(self):
        # The scaffold-hop direction: too close to the parent is the failure.
        result = one(
            CAFFEINE, reference_smiles=CAFFEINE, thresholds={"similarity_max": 0.6}
        )
        assert not result["passed"]

    def test_a_scaffold_mcs_ceiling_rejects_the_same_scaffold(self):
        result = one(
            CAFFEINE,
            reference_smiles=CAFFEINE,
            thresholds={"scaffold_mcs_ratio_max": 0.6},
        )
        assert not result["passed"]

    def test_similarity_gates_are_absent_without_a_reference(self):
        names = [gate["name"] for gate in one(IBUPROFEN)["gates"]]
        assert "similarity" not in names

    def test_the_reference_scaffold_is_reported_once(self):
        report, error = evaluate([CAFFEINE, ETHANOL], reference_smiles=CAFFEINE)
        assert error is None
        assert report.reference_scaffold


class TestRequestFailures:
    def test_rejects_an_empty_batch(self):
        _, error = evaluate([])
        assert "No SMILES" in error

    def test_rejects_an_oversized_batch(self):
        _, error = evaluate([ETHANOL] * (MAX_MOLECULES + 1))
        assert str(MAX_MOLECULES) in error

    def test_rejects_an_unparseable_reference(self):
        _, error = evaluate([ETHANOL], reference_smiles="not-a-molecule")
        assert "reference_smiles" in error

    def test_refuses_a_similarity_gate_with_no_reference(self):
        # Silently skipping it would produce an all-pass verdict the caller
        # would read as the molecule having cleared a bound it never met.
        _, error = evaluate([ETHANOL], thresholds={"similarity_min": 0.4})
        assert "requires reference_smiles" in error

    def test_lists_unparseable_molecules_without_failing_the_batch(self):
        report, error = evaluate([ETHANOL, "not-a-molecule", CAFFEINE])
        assert error is None
        assert report.invalid_smiles == ["not-a-molecule"]
        assert [result["smiles"] for result in report.results] == [ETHANOL, CAFFEINE]

    def test_skips_blank_entries(self):
        report, error = evaluate([ETHANOL, "", "   "])
        assert error is None
        assert len(report.results) == 1
        assert report.invalid_smiles == []


class TestBatch:
    def test_scores_every_molecule_in_input_order(self):
        report, error = evaluate([CAFFEINE, LIPOPHILE, ETHANOL])
        assert error is None
        assert [result["smiles"] for result in report.results] == [
            CAFFEINE,
            LIPOPHILE,
            ETHANOL,
        ]

    def test_reports_the_thresholds_it_applied(self):
        report, _ = evaluate([ETHANOL], thresholds={"mw_max": 300})
        assert report.thresholds["mw_max"] == 300

    def test_verdicts_are_independent(self):
        report, _ = evaluate([IBUPROFEN, LIPOPHILE])
        assert [result["passed"] for result in report.results] == [True, False]


class TestEndpoints:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_publishes_the_default_panel(self, client):
        # The tool documents these bounds in its schema; this is what makes
        # drift between the two visible instead of silent.
        body = client.get("/gate-defaults").json()
        assert body["defaults"] == DEFAULT_THRESHOLDS
        assert body["max_molecules"] == MAX_MOLECULES
        assert "similarity_min" in body["threshold_names"]

    def test_scores_a_batch(self, client):
        response = client.post(
            "/molecule-gates", json={"smiles": [IBUPROFEN, LIPOPHILE]}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"]
        assert [result["passed"] for result in body["results"]] == [True, False]
        assert body["thresholds"]["mw_max"] == 600.0

    def test_applies_overrides_and_names_the_ones_it_did_not_know(self, client):
        body = client.post(
            "/molecule-gates",
            json={
                "smiles": [IBUPROFEN],
                "thresholds": {"mw_max": 100, "mw_maximum": 100},
            },
        ).json()
        assert not body["results"][0]["passed"]
        assert body["unknown_thresholds"] == ["mw_maximum"]

    def test_reports_the_reference_and_its_scaffold(self, client):
        body = client.post(
            "/molecule-gates",
            json={"smiles": [IBUPROFEN], "reference_smiles": CAFFEINE},
        ).json()
        assert body["reference_smiles"] == CAFFEINE
        assert body["reference_scaffold"]
        assert body["results"][0]["similarity"]["tanimoto"] < 0.3

    def test_a_similarity_gate_without_a_reference_is_a_failed_request(self, client):
        body = client.post(
            "/molecule-gates",
            json={"smiles": [IBUPROFEN], "thresholds": {"similarity_min": 0.4}},
        ).json()
        assert not body["success"]
        assert "requires reference_smiles" in body["error"]

    def test_an_unparseable_reference_is_a_failed_request(self, client):
        body = client.post(
            "/molecule-gates",
            json={"smiles": [IBUPROFEN], "reference_smiles": "not-a-molecule"},
        ).json()
        assert not body["success"]
        assert "reference_smiles" in body["error"]

    def test_the_schema_rejects_an_empty_batch(self, client):
        response = client.post("/molecule-gates", json={"smiles": []})
        assert response.status_code == 422

    def test_the_schema_rejects_an_oversized_batch(self, client):
        response = client.post(
            "/molecule-gates", json={"smiles": [ETHANOL] * (MAX_MOLECULES + 1)}
        )
        assert response.status_code == 422


class TestStabilityIsReportedNotGated:
    """An aryl hypofluorite is not a lead, and no catalogue here says so.

    The twelve gates reproduce the published panel exactly -- verified against
    the reference scorer on 100 molecules with zero disagreements -- so a
    thirteenth would fail molecules that score, which is the silent
    disagreement this panel was fixed to remove. The information still has to
    reach the caller, so it goes beside the verdict.

    It is not hypothetical: a single-cut analogue sweep makes these by accident
    at about 12% of products, PAINS/Brenk/NIH pass every one, and eight of 122
    benchmark submissions shipped one as the final answer.
    """

    @pytest.mark.parametrize(
        "smiles,expected",
        [
            ("FOc1ccccc1", "hypohalite"),
            ("FSc1ccccc1", "sulfenyl halide"),
            ("O=C(NF)c1ccccc1", "N-halamine"),
            ("O=C(F)c1ccccc1", "acyl halide"),
            ("OOc1ccccc1", "acyclic peroxide"),
        ],
    )
    def test_the_group_is_named(self, smiles, expected):
        result = evaluate_molecule(smiles, DEFAULT_THRESHOLDS, None)

        assert result is not None
        assert expected in (result.get("unstable_group") or "")

    def test_it_does_not_change_the_verdict(self):
        # The molecule passes every published gate, and must go on doing so.
        result = evaluate_molecule("FOc1ccccc1", DEFAULT_THRESHOLDS, None)

        assert result["unstable_group"] is not None
        assert result["passed"] is True
        assert result["failures"] == []

    @pytest.mark.parametrize(
        "smiles",
        [
            "COc1ccccc1",
            "CSc1ccccc1",
            "NNC(=O)c1ccncc1",     # isoniazid: a hydrazide is real chemistry
            "O=C(NO)c1ccccc1",     # hydroxamic acid, likewise
            "OC(F)(F)F",
            "CC(=O)Oc1ccccc1C(=O)O",
        ],
    )
    def test_real_chemistry_is_not_flagged(self, smiles):
        result = evaluate_molecule(smiles, DEFAULT_THRESHOLDS, None)

        assert result is not None
        assert result["unstable_group"] is None
