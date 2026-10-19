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

"""Tests for the plip_interactions tool.

The service is stubbed throughout: PLIP and OpenBabel live only in the plip image,
and the interaction geometry is that suite's business. What matters here is the
handling of an ensemble -- that the same reference passed twice cannot inflate the
consensus denominator, that a structure which failed to parse is excluded from the
denominator rather than counted as a pose that made no contacts, and that the
answer always says which ligand it was about.
"""

import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.base import ContentType
from api.tools.plip import MAX_STRUCTURES, PlipInteractionsTool

TRP84 = "A:84:TRP:pistacking"
PHE288 = "A:288:PHE:hbond"
TYR121 = "A:121:TYR:waterbridge"


def interaction(
    family: str = "hbond",
    chain: str = "A",
    number: int = 288,
    name: str = "PHE",
    detail: dict | None = None,
) -> dict:
    """One interaction as the service would report it."""
    return {
        "family": family,
        "label": {"hbond": "hydrogen bond", "pistacking": "pi-stacking"}.get(
            family, family
        ),
        "chain": chain,
        "residue_number": number,
        "residue_name": name,
        "key": f"{chain}:{number}:{name}:{family}",
        "detail": detail if detail is not None else {"dist_d-a": 3.2, "don_angle": 138.6},
    }


def structure(
    name: str,
    fingerprint: list[str] | None = None,
    interactions: list[dict] | None = None,
    designated: str = "E20:A:2001",
    reason: str = "the only ligand in the structure",
    warnings: list[str] | None = None,
    smiles: str = "COc1ccccc1",
    cofactors: list[str] | None = None,
) -> dict:
    """One profiled structure as the service would report it."""
    fingerprint = [PHE288] if fingerprint is None else fingerprint
    return {
        "name": name,
        "cofactors": cofactors or [],
        "designated": designated,
        "designation_reason": reason,
        "warnings": warnings or [],
        "fingerprint": fingerprint,
        "ligands": [
            {
                "identifier": designated,
                "hetid": (designated or "LIG").split(":")[0],
                "chain": "A",
                "position": 2001,
                "ligand_type": "SMALLMOLECULE",
                "longname": "E20",
                "smiles": smiles,
                "num_heavy_atoms": 28,
                "num_rings": 2,
                "num_interactions": len(
                    interactions if interactions is not None else [interaction()]
                ),
                "counts": {"hbond": 1},
                "interactions": (
                    interactions if interactions is not None else [interaction()]
                ),
                "fingerprint": fingerprint,
                "residues": [
                    {
                        "chain": "A",
                        "residue_number": 288,
                        "residue_name": "PHE",
                        "families": ["hbond"],
                    }
                ],
            }
        ]
        if designated
        else [],
    }


def consensus_entry(key: str, count: int, total: int) -> dict:
    chain, number, name, family = key.split(":")
    return {
        "key": key,
        "chain": chain,
        "residue_number": int(number),
        "residue_name": name,
        "family": family,
        "count": count,
        "total": total,
        "fraction": round(count / total, 3),
        "sources": [f"pose{index}" for index in range(count)],
    }


def response(
    structures: list[dict],
    conserved: list[dict] | None = None,
    variable: list[dict] | None = None,
    residues: list[dict] | None = None,
    failed: list[dict] | None = None,
    threshold: float = 0.5,
    total: int | None = None,
) -> dict:
    """Build a successful service response."""
    total = len(structures) if total is None else total
    conserved = (
        [consensus_entry(PHE288, total, total)] if conserved is None else conserved
    )
    return {
        "success": True,
        "structures": structures,
        "consensus": {
            "total_structures": total,
            "threshold": threshold,
            "conserved": conserved,
            "variable": variable or [],
            "num_variable": len(variable or []),
        },
        "consensus_residues": residues
        if residues is not None
        else [
            {
                "chain": "A",
                "residue_number": 288,
                "residue_name": "PHE",
                "families": ["hbond"],
                "min_fraction": 1.0,
            }
        ],
        "failed": failed or [],
        "families": {
            "hbond": "hydrogen bond",
            "pistacking": "pi-stacking",
            "waterbridge": "water-bridged hydrogen bond",
        },
        "error": None,
    }


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.fixture
def tool(storage):
    instance = PlipInteractionsTool()
    instance._storage = storage
    return instance


def stub(tool, payload, calls: list | None = None):
    """Replace the tool's service call with a canned response."""

    async def _call(endpoint, method="POST", json_data=None, timeout=30.0, retries=0):
        if calls is not None:
            calls.append(json_data)
        if isinstance(payload, Exception):
            raise payload
        return payload

    tool._call_service = _call


async def complex_ref(storage, name: str = "complex") -> str:
    return await storage.store("pdb", b"ATOM      1  N   ALA A   1\nEND\n", description=name)


class TestInputHandling:
    @pytest.mark.asyncio
    async def test_requires_a_structure(self, tool):
        result = await tool.execute({})
        assert result.content_type == ContentType.ERROR
        assert "structure_refs" in result.data

    @pytest.mark.asyncio
    async def test_the_error_says_to_pass_every_pose(self, tool):
        result = await tool.execute({})
        assert "consensus" in result.data

    @pytest.mark.asyncio
    async def test_accepts_a_bare_string_where_a_list_is_declared(
        self, tool, storage
    ):
        ref = await complex_ref(storage)
        calls: list = []
        stub(tool, response([structure(ref)]), calls)

        await tool.execute({"structure_refs": ref})

        assert [entry["name"] for entry in calls[0]["structures"]] == [ref]

    @pytest.mark.asyncio
    async def test_sends_each_structure_with_its_reference_as_the_name(
        self, tool, storage
    ):
        first = await complex_ref(storage, "pose1")
        second = await complex_ref(storage, "pose2")
        calls: list = []
        stub(tool, response([structure(first), structure(second)]), calls)

        await tool.execute({"structure_refs": [first, second]})

        assert [entry["name"] for entry in calls[0]["structures"]] == [first, second]

    @pytest.mark.asyncio
    async def test_deduplicates_repeated_references(self, tool, storage):
        # The same pose twice would corroborate itself in the consensus.
        ref = await complex_ref(storage)
        calls: list = []
        stub(tool, response([structure(ref)]), calls)

        await tool.execute({"structure_refs": [ref, ref, ref]})

        assert len(calls[0]["structures"]) == 1

    @pytest.mark.asyncio
    async def test_an_invalid_reference_is_an_error(self, tool):
        result = await tool.execute({"structure_refs": ["not-a-reference"]})
        assert result.content_type == ContentType.ERROR
        assert "boltz2_predict" in result.data

    @pytest.mark.asyncio
    async def test_a_missing_artifact_is_an_error(self, tool):
        result = await tool.execute({"structure_refs": ["pdb_deadbeef"]})
        assert result.content_type == ContentType.ERROR
        assert "expired" in result.data

    @pytest.mark.asyncio
    async def test_an_empty_artifact_is_an_error(self, tool, storage):
        ref = await storage.store("pdb", b"   ")
        result = await tool.execute({"structure_refs": [ref]})
        assert result.content_type == ContentType.ERROR
        assert "empty" in result.data

    @pytest.mark.asyncio
    async def test_an_oversized_ensemble_is_refused_with_advice(self, tool, storage):
        refs = [await complex_ref(storage, str(i)) for i in range(MAX_STRUCTURES + 1)]
        result = await tool.execute({"structure_refs": refs})
        assert result.content_type == ContentType.ERROR
        assert str(MAX_STRUCTURES) in result.data

    @pytest.mark.asyncio
    async def test_requires_storage(self, storage):
        instance = PlipInteractionsTool()
        instance._storage = None
        result = await instance.execute({"structure_refs": ["pdb_abc12345"]})
        assert result.content_type == ContentType.ERROR
        assert "Storage" in result.data


class TestThreshold:
    @pytest.mark.asyncio
    async def test_passes_the_threshold_through(self, tool, storage):
        ref = await complex_ref(storage)
        calls: list = []
        stub(tool, response([structure(ref)]), calls)

        await tool.execute({"structure_refs": [ref], "consensus_threshold": 0.8})

        assert calls[0]["consensus_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_is_omitted_when_not_given(self, tool, storage):
        ref = await complex_ref(storage)
        calls: list = []
        stub(tool, response([structure(ref)]), calls)

        await tool.execute({"structure_refs": [ref]})

        assert "consensus_threshold" not in calls[0]

    @pytest.mark.asyncio
    async def test_rejects_a_threshold_above_one(self, tool, storage):
        ref = await complex_ref(storage)
        result = await tool.execute(
            {"structure_refs": [ref], "consensus_threshold": 1.5}
        )
        assert result.content_type == ContentType.ERROR
        assert "between 0 and 1" in result.data

    @pytest.mark.asyncio
    async def test_rejects_a_non_numeric_threshold(self, tool, storage):
        ref = await complex_ref(storage)
        result = await tool.execute(
            {"structure_refs": [ref], "consensus_threshold": "high"}
        )
        assert result.content_type == ContentType.ERROR

    @pytest.mark.asyncio
    async def test_passes_a_requested_ligand_through(self, tool, storage):
        ref = await complex_ref(storage)
        calls: list = []
        stub(tool, response([structure(ref)]), calls)

        await tool.execute({"structure_refs": [ref], "ligand": "E20"})

        assert calls[0]["ligand"] == "E20"


class TestSingleStructureRendering:
    @pytest.mark.asyncio
    async def test_lists_every_interaction_by_family(self, tool, storage):
        ref = await complex_ref(storage)
        stub(
            tool,
            response(
                [
                    structure(
                        ref,
                        fingerprint=[PHE288, TRP84],
                        interactions=[
                            interaction(),
                            interaction(
                                "pistacking",
                                number=84,
                                name="TRP",
                                detail={"centdist": 3.88, "angle": 15.0, "offset": 0.59},
                            ),
                        ],
                    )
                ],
                conserved=[
                    consensus_entry(PHE288, 1, 1),
                    consensus_entry(TRP84, 1, 1),
                ],
            ),
        )

        result = await tool.execute({"structure_refs": [ref]})

        assert "Hydrogen Bond:" in result.data
        assert "Pi-Stacking:" in result.data
        assert "PHE288 (chain A)" in result.data
        assert "TRP84 (chain A)" in result.data

    @pytest.mark.asyncio
    async def test_reports_the_geometry(self, tool, storage):
        ref = await complex_ref(storage)
        stub(tool, response([structure(ref)]))

        result = await tool.execute({"structure_refs": [ref]})

        assert "donor-acceptor 3.20 A" in result.data
        assert "angle 139 deg" in result.data

    @pytest.mark.asyncio
    async def test_says_a_single_structure_cannot_corroborate_itself(
        self, tool, storage
    ):
        # Presenting one pose's contacts as a consensus is the misreading this
        # sentence exists to prevent.
        ref = await complex_ref(storage)
        stub(tool, response([structure(ref)]))

        result = await tool.execute({"structure_refs": [ref]})

        assert "one structure" in result.data
        assert "sampling noise" in result.data

    @pytest.mark.asyncio
    async def test_names_the_ligand_and_why_it_was_chosen(self, tool, storage):
        ref = await complex_ref(storage)
        stub(tool, response([structure(ref)]))

        result = await tool.execute({"structure_refs": [ref]})

        assert "Ligand profiled: E20:A:2001" in result.data
        assert "only ligand" in result.data

    @pytest.mark.asyncio
    async def test_reports_the_perceived_smiles(self, tool, storage):
        ref = await complex_ref(storage)
        stub(tool, response([structure(ref, smiles="COc1ccccc1CN")]))

        result = await tool.execute({"structure_refs": [ref]})

        assert "COc1ccccc1CN" in result.data

    @pytest.mark.asyncio
    async def test_a_structure_with_no_ligand_says_so(self, tool, storage):
        ref = await complex_ref(storage)
        stub(
            tool,
            response(
                [structure(ref, fingerprint=[], designated="", reason="")],
                conserved=[],
                residues=[],
            ),
        )

        result = await tool.execute({"structure_refs": [ref]})

        assert "No protein-ligand interactions" in result.data


class TestEnsembleRendering:
    @pytest.mark.asyncio
    async def test_leads_with_the_conserved_count(self, tool, storage):
        refs = [await complex_ref(storage, str(i)) for i in range(4)]
        stub(
            tool,
            response(
                [structure(ref) for ref in refs],
                conserved=[
                    consensus_entry(PHE288, 4, 4),
                    consensus_entry(TRP84, 3, 4),
                ],
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert result.data.startswith("2 interaction(s) are conserved across 4")

    @pytest.mark.asyncio
    async def test_tabulates_support_as_a_fraction_of_poses(self, tool, storage):
        refs = [await complex_ref(storage, str(i)) for i in range(4)]
        stub(
            tool,
            response(
                [structure(ref) for ref in refs],
                conserved=[consensus_entry(TRP84, 3, 4)],
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert "TRP84" in result.data
        assert "3/4" in result.data
        assert "pi-stack" in result.data

    @pytest.mark.asyncio
    async def test_shows_what_fell_below_the_threshold(self, tool, storage):
        # The poses that disagree are evidence about the sampler, and hiding
        # them would make a stochastic contact look absent rather than weak.
        refs = [await complex_ref(storage, str(i)) for i in range(4)]
        stub(
            tool,
            response(
                [structure(ref) for ref in refs],
                conserved=[consensus_entry(PHE288, 4, 4)],
                variable=[consensus_entry(TYR121, 1, 4)],
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert "Below the threshold" in result.data
        assert "TYR121" in result.data
        assert "1/4" in result.data

    @pytest.mark.asyncio
    async def test_no_consensus_is_reported_as_a_finding(self, tool, storage):
        refs = [await complex_ref(storage, str(i)) for i in range(3)]
        stub(
            tool,
            response(
                [structure(ref) for ref in refs],
                conserved=[],
                variable=[consensus_entry(TYR121, 1, 3)],
                residues=[],
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert "poses disagree" in result.data
        assert "consensus_threshold" in result.data

    @pytest.mark.asyncio
    async def test_lists_the_per_structure_totals(self, tool, storage):
        refs = [await complex_ref(storage, str(i)) for i in range(2)]
        stub(
            tool,
            response(
                [
                    structure(refs[0], fingerprint=[PHE288, TRP84]),
                    structure(refs[1], fingerprint=[PHE288]),
                ]
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert f"{refs[0]}: 2 interaction(s)" in result.data
        assert f"{refs[1]}: 1 interaction(s)" in result.data

    @pytest.mark.asyncio
    async def test_a_pose_with_no_ligand_reads_as_such(self, tool, storage):
        refs = [await complex_ref(storage, str(i)) for i in range(2)]
        stub(
            tool,
            response(
                [
                    structure(refs[0]),
                    structure(refs[1], fingerprint=[], designated="", reason=""),
                ]
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert "no ligand found" in result.data

    @pytest.mark.asyncio
    async def test_flags_poses_that_bound_nothing(self, tool, storage):
        refs = [await complex_ref(storage, str(i)) for i in range(2)]
        stub(
            tool,
            response(
                [
                    structure(refs[0]),
                    structure(refs[1], fingerprint=[], designated="", reason=""),
                ]
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert "no interactions at all" in result.data
        assert "still count in the denominator" in result.data

    @pytest.mark.asyncio
    async def test_warns_when_different_ligands_were_profiled(self, tool, storage):
        # Comparing contacts across two different het groups is meaningless, and
        # the fingerprints would look plausible either way.
        refs = [await complex_ref(storage, str(i)) for i in range(2)]
        stub(
            tool,
            response(
                [
                    structure(refs[0], designated="E20:A:2001"),
                    structure(refs[1], designated="NAG:A:3001"),
                ]
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert "Different ligands" in result.data
        assert "Pass `ligand`" in result.data

    @pytest.mark.asyncio
    async def test_surfaces_service_warnings_once(self, tool, storage):
        refs = [await complex_ref(storage, str(i)) for i in range(2)]
        warning = "No ligand matching 'XYZ' was found; profiled E20:A:2001 instead."
        stub(
            tool,
            response([structure(ref, warnings=[warning]) for ref in refs]),
        )

        result = await tool.execute({"structure_refs": refs})

        assert result.data.count(warning) == 1

    @pytest.mark.asyncio
    async def test_prints_the_family_legend(self, tool, storage):
        refs = [await complex_ref(storage, str(i)) for i in range(2)]
        stub(tool, response([structure(ref) for ref in refs]))

        result = await tool.execute({"structure_refs": refs})

        assert "H-bond = hydrogen bond" in result.data


class TestPartialFailure:
    @pytest.mark.asyncio
    async def test_names_the_structures_that_could_not_be_profiled(
        self, tool, storage
    ):
        refs = [await complex_ref(storage, str(i)) for i in range(2)]
        stub(
            tool,
            response(
                [structure(refs[0])],
                total=1,
                failed=[{"name": refs[1], "error": "the input was not recognized"}],
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert "could not be profiled" in result.data
        assert refs[1] in result.data
        assert result.metadata["num_failed"] == 1

    @pytest.mark.asyncio
    async def test_the_denominator_excludes_them(self, tool, storage):
        refs = [await complex_ref(storage, str(i)) for i in range(3)]
        stub(
            tool,
            response(
                [structure(refs[0]), structure(refs[1])],
                conserved=[consensus_entry(PHE288, 2, 2)],
                total=2,
                failed=[{"name": refs[2], "error": "bad"}],
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert result.metadata["num_structures"] == 2
        assert "2/2" in result.data


class TestErrors:
    @pytest.mark.asyncio
    async def test_a_service_error_is_reported(self, tool, storage):
        ref = await complex_ref(storage)
        stub(tool, {"success": False, "error": "None of the structures could be profiled."})

        result = await tool.execute({"structure_refs": [ref]})

        assert result.content_type == ContentType.ERROR
        assert "None of the structures" in result.data

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_reported(self, tool, storage):
        ref = await complex_ref(storage)
        stub(tool, RuntimeError("connection refused"))

        result = await tool.execute({"structure_refs": [ref]})

        assert result.content_type == ContentType.ERROR
        assert "connection refused" in result.data


class TestMetadata:
    @pytest.mark.asyncio
    async def test_carries_the_fingerprint_for_later_comparison(self, tool, storage):
        # An analog that keeps these keys has kept the binding mode; this is what
        # a later call compares against.
        refs = [await complex_ref(storage, str(i)) for i in range(2)]
        stub(
            tool,
            response(
                [structure(ref) for ref in refs],
                conserved=[
                    consensus_entry(PHE288, 2, 2),
                    consensus_entry(TRP84, 2, 2),
                ],
            ),
        )

        result = await tool.execute({"structure_refs": refs})

        assert result.metadata["consensus_fingerprint"] == [PHE288, TRP84]
        assert result.metadata["consensus_residues"] == ["PHE288"]
        assert result.metadata["fingerprints"][refs[0]] == [PHE288]
        assert result.metadata["ligands_profiled"] == ["E20:A:2001"]

    @pytest.mark.asyncio
    async def test_records_the_threshold_that_was_applied(self, tool, storage):
        ref = await complex_ref(storage)
        stub(tool, response([structure(ref)], threshold=0.75))

        result = await tool.execute({"structure_refs": [ref]})

        assert result.metadata["consensus_threshold"] == 0.75


class TestSchema:
    def test_declares_the_reference_parameter(self, tool):
        schema = tool.get_schema()
        params = schema["function"]["parameters"]["properties"]
        assert params["structure_refs"]["x-aevom-reference"]["data_type"] == "pdb"
        assert schema["function"]["parameters"]["required"] == ["structure_refs"]

    def test_tells_the_model_to_pass_every_pose(self, tool):
        description = tool.get_schema()["function"]["description"]
        assert "PASS EVERY POSE IN ONE CALL" in description

    def test_says_mmcif_needs_no_conversion(self, tool):
        # Otherwise the model spends a turn per diffusion sample on
        # convert_structure before it can ask this question at all.
        description = tool.get_schema()["function"]["description"]
        assert "do NOT" in description
        assert "convert_structure" in description

    def test_the_summary_mentions_the_ensemble_size(self, tool):
        summary = tool.format_summary({"structure_refs": ["a", "b", "c"]})
        assert "3 structures" in summary


class TestCofactorsAreReportedByAbsence:
    """PLIP's metal channel can only fire when a metal is in the file.

    A predicted complex usually has none, so a zinc metalloenzyme is profiled as
    its apo site: the sulfonamide or carboxylate that actually coordinates the
    zinc comes back as ordinary hydrogen bonds or a salt bridge to the
    histidines that would hold it. Measured across 120 benchmark runs on 30
    metal- or cofactor-dependent targets, the string "Metal Complex" appears
    zero times and "zinc" appears zero times -- and runs went on to enumerate
    analogues that delete the coordinating group.

    Naming the absence needs no knowledge of what the target is, which matters:
    the identity is deliberately withheld.
    """

    @pytest.mark.asyncio
    async def test_absence_is_stated(self, tool, storage):
        ref = await complex_ref(storage)
        stub(tool, response([structure(ref, cofactors=[])]))

        result = await tool.execute({"structure_refs": [ref]})

        assert "No metal ion or cofactor is present" in result.data
        assert "histidines" in result.data

    @pytest.mark.asyncio
    async def test_presence_is_stated_briefly(self, tool, storage):
        ref = await complex_ref(storage)
        stub(tool, response([structure(ref, cofactors=["ZN"])]))

        result = await tool.execute({"structure_refs": [ref]})

        assert "Cofactors present: ZN." in result.data
        assert "No metal ion" not in result.data
