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

"""Tests for the Boltz-2 builtin tool.

Follows the fixture pattern established in ``test_protein_design_nims.py``: stub
``_get_api_key`` (there is no key in CI) and intercept the wire with a mock
transport, so request construction, response parsing and the storage writes all
stay under test.

Two things get asserted harder than the rest, because they are the ways this tool
can fail silently rather than loudly. The MSA payload has to nest as
``msa.msa_search.a3m.alignment`` -- upstream's older ``data`` field 422s on the
validated NIM, and a request without an MSA still succeeds, just worse. And the
structure has to land as an ``mmcif`` artifact rather than a ``pdb`` one, since
storing mmCIF text under a ``pdb_`` ref is exactly the papering-over this tool
exists to avoid.
"""

import io
import json

import httpx
import numpy as np
import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.base import ContentType
from api.tools.boltz2 import Boltz2Tool

SEQUENCE = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"

# Minimal mmCIF, enough to be recognizable as the format and to differ between
# samples so ranking assertions mean something.
def _mmcif(tag: str) -> str:
    return f"data_{tag}\n#\n_atom_site.group_PDB\nATOM\n#\n"


A3M = f">query\n{SEQUENCE}\n>hit1\n{SEQUENCE}\n"


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


def json_response(payload: dict, status: int = 200):
    """A transport handler returning a fixed JSON payload.

    Records the request it saw, so tests can assert on the body the tool built.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        handler.request = request
        return httpx.Response(status, json=payload)

    handler.request = None
    return handler


@pytest.fixture
def patch_httpx(monkeypatch):
    """Swap in a mock transport, leaving the rest of the request path real."""

    def apply(handler):
        real_client = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs.pop("transport", None)
            return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return apply


@pytest.fixture
def tool(storage):
    instance = Boltz2Tool()
    instance._storage = storage
    instance._get_api_key = lambda: "test-key"
    return instance


def one_structure(confidence: float = 0.87) -> dict:
    return {
        "structures": [
            {"structure": _mmcif("primary"), "format": "mmcif", "name": "model_0"}
        ],
        "confidence_scores": [confidence],
        "metrics": {"elapsed_ms": 41000},
    }


def sent_body(handler) -> dict:
    return json.loads(handler.request.content)


@pytest.mark.asyncio
class TestRequestConstruction:
    async def test_a_single_sequence_becomes_one_protein_polymer(
        self, tool, patch_httpx
    ):
        handler = json_response(one_structure())
        patch_httpx(handler)

        result = await tool.execute({"sequence": SEQUENCE})

        assert result.content_type == ContentType.DATA_REFERENCE
        body = sent_body(handler)
        assert body["polymers"] == [
            {"id": "A", "molecule_type": "protein", "sequence": SEQUENCE}
        ]
        assert "ligands" not in body

    async def test_the_output_format_is_pinned_to_mmcif(self, tool, patch_httpx):
        # mmcif is the only accepted value. Exposing it as a parameter would only
        # let the model choose a 422.
        handler = json_response(one_structure())
        patch_httpx(handler)

        await tool.execute({"sequence": SEQUENCE, "output_format": "pdb"})

        assert sent_body(handler)["output_format"] == "mmcif"

    async def test_the_poll_header_covers_affinity_runtime(self, tool, patch_httpx):
        handler = json_response(one_structure())
        patch_httpx(handler)

        await tool.execute({"sequence": SEQUENCE})

        assert handler.request.headers["NVCF-POLL-SECONDS"] == "300"
        assert handler.request.headers["Authorization"] == "Bearer test-key"

    async def test_a_sequence_reference_is_resolved(
        self, tool, storage, patch_httpx
    ):
        handler = json_response(one_structure())
        patch_httpx(handler)
        ref = await storage.store(data_type="sequence", content=SEQUENCE.encode())

        await tool.execute({"sequence_ref": ref})

        assert sent_body(handler)["polymers"][0]["sequence"] == SEQUENCE

    async def test_the_msa_nests_under_alignment_not_data(
        self, tool, storage, patch_httpx
    ):
        # `data` is the stale field name from upstream's older examples and 422s
        # on the validated NIM. This is the one payload shape worth pinning.
        handler = json_response(one_structure())
        patch_httpx(handler)
        msa_ref = await storage.store(data_type="msa", content=A3M.encode())

        await tool.execute({"sequence": SEQUENCE, "msa_ref": msa_ref})

        msa = sent_body(handler)["polymers"][0]["msa"]
        assert msa == {
            "msa_search": {"a3m": {"alignment": A3M, "format": "a3m", "rank": 0}}
        }

    async def test_the_top_level_msa_lands_on_the_first_protein_chain(
        self, tool, storage, patch_httpx
    ):
        handler = json_response(one_structure())
        patch_httpx(handler)
        msa_ref = await storage.store(data_type="msa", content=A3M.encode())

        await tool.execute(
            {
                "polymers": [
                    {"molecule_type": "dna", "sequence": "ATCGATCG"},
                    {"molecule_type": "protein", "sequence": SEQUENCE},
                ],
                "msa_ref": msa_ref,
            }
        )

        polymers = sent_body(handler)["polymers"]
        assert "msa" not in polymers[0]
        assert polymers[1]["msa"]["msa_search"]["a3m"]["alignment"] == A3M

    async def test_chain_ids_are_assigned_around_explicit_ones(
        self, tool, patch_httpx
    ):
        # An auto-assigned id must not collide with an id a later chain claims.
        handler = json_response(one_structure())
        patch_httpx(handler)

        await tool.execute(
            {
                "polymers": [
                    {"molecule_type": "protein", "sequence": SEQUENCE},
                    {"molecule_type": "protein", "sequence": SEQUENCE, "id": "A"},
                ]
            }
        )

        ids = [p["id"] for p in sent_body(handler)["polymers"]]
        assert ids == ["B", "A"]

    async def test_ligand_ids_avoid_the_polymer_ids(self, tool, patch_httpx):
        handler = json_response(one_structure())
        patch_httpx(handler)

        await tool.execute(
            {
                "polymers": [{"molecule_type": "protein", "sequence": SEQUENCE}],
                "ligands": [{"smiles": ASPIRIN}, {"ccd": "ATP", "id": "L1"}],
            }
        )

        body = sent_body(handler)
        assert [lig["id"] for lig in body["ligands"]] == ["L2", "L1"]
        assert body["ligands"][1]["ccd"] == "ATP"

    async def test_affinity_settings_are_only_sent_when_asked_for(
        self, tool, patch_httpx
    ):
        handler = json_response(one_structure())
        patch_httpx(handler)

        await tool.execute(
            {"sequence": SEQUENCE, "ligands": [{"smiles": ASPIRIN}]}
        )

        body = sent_body(handler)
        assert "sampling_steps_affinity" not in body
        assert "diffusion_samples_affinity" not in body

    async def test_affinity_settings_ride_along_with_the_flag(
        self, tool, patch_httpx
    ):
        handler = json_response(one_structure())
        patch_httpx(handler)

        await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "predict_affinity": True}],
                "diffusion_samples_affinity": 1,
            }
        )

        body = sent_body(handler)
        assert body["ligands"][0]["predict_affinity"] is True
        assert body["sampling_steps_affinity"] == 200
        assert body["diffusion_samples_affinity"] == 1

    async def test_modifications_templates_and_cyclic_are_forwarded(
        self, tool, storage, patch_httpx
    ):
        handler = json_response(one_structure())
        patch_httpx(handler)
        template_ref = await storage.store(
            data_type="pdb", content=b"ATOM      1  CA  GLY A   1\nEND\n"
        )

        await tool.execute(
            {
                "polymers": [
                    {
                        "molecule_type": "protein",
                        "sequence": SEQUENCE,
                        "cyclic": True,
                        "modifications": [{"ccd": "sep", "position": 3}],
                        "structural_templates": [
                            {"pdb_ref": template_ref, "name": "ref1"}
                        ],
                    }
                ],
            }
        )

        polymer = sent_body(handler)["polymers"][0]
        assert polymer["cyclic"] is True
        assert polymer["modifications"] == [{"ccd": "SEP", "position": 3}]
        template = polymer["structural_templates"][0]
        assert template["format"] == "pdb"
        assert template["name"] == "ref1"
        assert "ATOM" in template["structure"]

    async def test_an_mmcif_template_is_declared_as_cif(
        self, tool, storage, patch_httpx
    ):
        handler = json_response(one_structure())
        patch_httpx(handler)
        template_ref = await storage.store(
            data_type="mmcif", content=_mmcif("tpl").encode()
        )

        await tool.execute(
            {
                "polymers": [
                    {
                        "molecule_type": "protein",
                        "sequence": SEQUENCE,
                        "structural_templates": [{"pdb_ref": template_ref}],
                    }
                ]
            }
        )

        templates = sent_body(handler)["polymers"][0]["structural_templates"]
        assert templates[0]["format"] == "cif"

    async def test_pocket_and_bond_constraints_are_forwarded(
        self, tool, patch_httpx
    ):
        handler = json_response(one_structure())
        patch_httpx(handler)

        await tool.execute(
            {
                "polymers": [
                    {"molecule_type": "protein", "sequence": SEQUENCE, "id": "A"}
                ],
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}],
                "constraints": [
                    {
                        "constraint_type": "pocket",
                        "binder": "L1",
                        "contacts": [{"id": "A", "residue_index": 12}],
                    },
                    {
                        "constraint_type": "bond",
                        "atoms": [
                            {"id": "A", "residue_index": 12, "atom_name": "SG"},
                            {"id": "L1", "residue_index": 1, "atom_name": "C1"},
                        ],
                    },
                ],
            }
        )

        constraints = sent_body(handler)["constraints"]
        assert constraints[0]["binder"] == "L1"
        assert constraints[0]["contacts"] == [{"id": "A", "residue_index": 12}]
        assert constraints[1]["atoms"][0]["atom_name"] == "SG"

    async def test_stringified_arrays_are_accepted(self, tool, patch_httpx):
        # LLMs regularly JSON-encode array arguments; a bare failure here would
        # look like the tool being broken.
        handler = json_response(one_structure())
        patch_httpx(handler)

        result = await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": json.dumps([{"smiles": ASPIRIN}]),
            }
        )

        assert result.content_type == ContentType.DATA_REFERENCE
        assert sent_body(handler)["ligands"][0]["smiles"] == ASPIRIN


@pytest.mark.asyncio
class TestValidation:
    async def _error(self, tool, arguments) -> str:
        result = await tool.execute(arguments)
        assert result.content_type == ContentType.ERROR
        return result.data

    async def _accepted(self, tool, arguments) -> None:
        """Assert the tool did not refuse these arguments.

        Not "did not error": with no API key the call still comes back a 401
        from the endpoint, and that is the point -- it got as far as the wire.
        """
        from api.tools.base import INVALID_ARGUMENTS

        result = await tool.execute(arguments)
        assert not (result.metadata or {}).get(INVALID_ARGUMENTS), result.data

    async def test_two_affinity_ligands_are_rejected_before_the_wire(self, tool):
        # The endpoint 422s on this, and the message does not say the limit is
        # one, so the model would retry the same request.
        message = await self._error(
            tool,
            {
                "sequence": SEQUENCE,
                "ligands": [
                    {"smiles": ASPIRIN, "predict_affinity": True},
                    {"ccd": "ATP", "predict_affinity": True},
                ],
            },
        )
        assert "one ligand per" in message
        assert "one call per ligand" in message

    async def test_the_names_the_model_reaches_for_are_accepted(self, tool):
        """Measured, not guessed: this is the shape traces actually contain.

        Across two 100-task benchmark runs the model wrote `type` for
        `constraint_type` 73 times, `residues` for `contacts` 67 times, and
        `chain`/`residue` for `id`/`residue_index` 61 times. The schema names
        them correctly and the model writes the domain's names anyway, and each
        rejection cost one of eight oracle calls.
        """
        await self._accepted(
            tool,
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}],
                "constraints": [
                    {
                        "type": "pocket",
                        "ligand": "L1",
                        "residues": [{"chain": "A", "residue": 5}],
                    }
                ],
            },
        )

    async def test_boltz_yaml_shapes_are_accepted_too(self, tool):
        # Boltz's own YAML nests the kind and writes contacts as pairs, which is
        # the documentation the model has most likely seen.
        await self._accepted(
            tool,
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}],
                "constraints": [{"pocket": {"binder": "L1", "contacts": [["A", 5]]}}],
            },
        )

    async def test_a_binder_given_as_an_object_names_the_same_chain(self, tool):
        await self._accepted(
            tool,
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}],
                "constraints": [
                    {
                        "constraint_type": "pocket",
                        "binder": {"ligand": "L1"},
                        "contacts": [{"id": "A", "residue_index": 5}],
                    }
                ],
            },
        )

    async def test_an_omitted_molecule_type_is_read_off_an_unambiguous_sequence(self, tool):
        # Two-thirds of the polymers in the traces named a sequence and no type.
        await self._accepted(
            tool, {"polymers": [{"sequence": SEQUENCE}], "ligands": [{"smiles": ASPIRIN}]}
        )

    async def test_an_ambiguous_sequence_is_still_refused_rather_than_guessed(self, tool):
        # ACGT is a nucleic acid or a four-residue peptide, and guessing wrong
        # would silently fold the wrong molecule.
        message = await self._error(
            tool, {"polymers": [{"sequence": "ACGTACGTACGT"}], "ligands": [{"smiles": ASPIRIN}]}
        )

        assert "could be more than one" in message

    async def test_an_explicit_type_wins_over_its_alias(self, tool):
        message = await self._error(
            tool,
            {
                "polymers": [{"sequence": SEQUENCE, "molecule_type": "dna", "type": "protein"}],
                "ligands": [{"smiles": ASPIRIN}],
            },
        )
        # Validated as DNA, which an amino-acid sequence is not.
        assert "dna" in message.lower() or "alphabet" in message.lower()

    async def test_bare_residue_indices_take_the_chain_named_beside_them(self, tool):
        # `{"chain": "A", "residues": [22, 23]}` -- 38 calls in the traces.
        await self._accepted(
            tool,
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}],
                "constraints": [
                    {"type": "pocket", "binder": "L1", "chain": "A", "residues": [3, 5]}
                ],
            },
        )

    async def test_bare_indices_with_no_chain_named_fall_to_the_only_polymer(self, tool):
        # `{"pocket": {"residue_indices": [22, 23]}}` -- 25 calls, and with one
        # protein in the request the residues cannot be anywhere else.
        await self._accepted(
            tool,
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN}],
                "constraints": [{"pocket": {"residue_indices": [3, 5]}}],
            },
        )

    async def test_a_chain_letter_glued_to_an_index_is_split(self, tool):
        await self._accepted(
            tool,
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}],
                "constraints": [
                    {"constraint_type": "pocket", "binder": "L1", "contacts": ["A3", "A5"]}
                ],
            },
        )

    async def test_a_glued_index_prefers_the_longest_declared_chain_id(self):
        from api.tools.boltz2 import _normalise_residue_ref

        # L140 is chain L1 residue 40 when L1 exists, not chain L residue 140.
        assert _normalise_residue_ref("L140", {"A", "L1"}) == {"id": "L1", "residue_index": 40}
        assert _normalise_residue_ref("L140", {"A", "L"}) == {"id": "L", "residue_index": 140}

    async def test_a_single_ligand_is_assumed_to_be_the_binder(self, tool):
        # A pocket constraint steers a ligand, and with one in the request there
        # is only one thing it can be steering. 38 calls omitted it.
        await self._accepted(
            tool,
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN}],
                "constraints": [{"type": "pocket", "chain": "A", "residues": [3, 5]}],
            },
        )

    async def test_with_two_ligands_the_binder_is_still_required(self, tool):
        message = await self._error(
            tool,
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}, {"ccd": "ATP", "id": "L2"}],
                "constraints": [{"type": "pocket", "chain": "A", "residues": [3, 5]}],
            },
        )

        assert "needs 'binder'" in message

    async def test_a_rejected_call_is_marked_so_the_budget_can_be_refunded(self, tool):
        from api.tools.base import INVALID_ARGUMENTS

        result = await tool.execute(
            {"sequence": SEQUENCE, "ligands": [{"smiles": ASPIRIN, "ccd": "ATP"}]}
        )

        assert result.content_type == ContentType.ERROR
        assert result.metadata.get(INVALID_ARGUMENTS) is True

    async def test_a_ligand_needs_exactly_one_of_smiles_and_ccd(self, tool):
        both = await self._error(
            tool,
            {"sequence": SEQUENCE, "ligands": [{"smiles": ASPIRIN, "ccd": "ATP"}]},
        )
        assert "not both" in both

        neither = await self._error(
            tool, {"sequence": SEQUENCE, "ligands": [{"id": "L1"}]}
        )
        assert "either 'smiles' or 'ccd'" in neither

    async def test_a_constraint_id_must_match_a_declared_chain(self, tool):
        message = await self._error(
            tool,
            {
                "polymers": [
                    {"molecule_type": "protein", "sequence": SEQUENCE, "id": "A"}
                ],
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}],
                "constraints": [
                    {
                        "constraint_type": "pocket",
                        "binder": "LIG",
                        "contacts": [{"id": "A", "residue_index": 3}],
                    }
                ],
            },
        )
        assert "'LIG'" in message
        # The message names what is available, so the retry can be right.
        assert "A" in message and "L1" in message

    async def test_a_bond_constraint_needs_two_atoms(self, tool):
        message = await self._error(
            tool,
            {
                "polymers": [
                    {"molecule_type": "protein", "sequence": SEQUENCE, "id": "A"}
                ],
                "constraints": [
                    {
                        "constraint_type": "bond",
                        "atoms": [
                            {"id": "A", "residue_index": 1, "atom_name": "SG"}
                        ],
                    }
                ],
            },
        )
        assert "exactly two atoms" in message

    @pytest.mark.parametrize("chain_id", ["TOOLONG", "A-1", "A 1"])
    async def test_a_malformed_chain_id_is_rejected(self, tool, chain_id):
        message = await self._error(
            tool,
            {
                "polymers": [
                    {
                        "molecule_type": "protein",
                        "sequence": SEQUENCE,
                        "id": chain_id,
                    }
                ]
            },
        )
        assert "1-4 alphanumeric" in message

    @pytest.mark.parametrize("chain_id", ["A", "L1", "LIG1"])
    async def test_the_chain_id_shapes_upstream_uses_are_accepted(
        self, tool, patch_httpx, chain_id
    ):
        # The polymer docs say "single letter or 4-char alphanumeric", but every
        # upstream ligand example is two characters and the service takes them.
        # A stricter client-side rule would refuse valid requests.
        handler = json_response(one_structure())
        patch_httpx(handler)

        result = await tool.execute(
            {
                "polymers": [
                    {
                        "molecule_type": "protein",
                        "sequence": SEQUENCE,
                        "id": chain_id,
                    }
                ]
            }
        )

        assert result.content_type == ContentType.DATA_REFERENCE
        assert sent_body(handler)["polymers"][0]["id"] == chain_id

    async def test_a_duplicate_chain_id_is_rejected(self, tool):
        message = await self._error(
            tool,
            {
                "polymers": [
                    {"molecule_type": "protein", "sequence": SEQUENCE, "id": "A"},
                    {"molecule_type": "protein", "sequence": SEQUENCE, "id": "A"},
                ]
            },
        )
        assert "more than once" in message

    @pytest.mark.parametrize(
        "molecule_type,sequence,expected",
        [
            ("protein", "MKTXZ", "invalid characters"),
            ("dna", "ATCGU", "A, T, C, G"),
            ("rna", "AUCGT", "A, U, C, G"),
        ],
    )
    async def test_alphabets_are_checked_per_molecule_type(
        self, tool, molecule_type, sequence, expected
    ):
        message = await self._error(
            tool,
            {"polymers": [{"molecule_type": molecule_type, "sequence": sequence}]},
        )
        assert expected in message

    async def test_an_unknown_molecule_type_is_rejected(self, tool):
        message = await self._error(
            tool, {"polymers": [{"molecule_type": "peptide", "sequence": SEQUENCE}]}
        )
        assert "'protein', 'dna' or 'rna'" in message

    async def test_an_oversized_sequence_suggests_a_domain(self, tool):
        message = await self._error(
            tool, {"sequence": "A" * 4097, "polymers": []}
        )
        assert "4096" in message
        assert "domain" in message

    async def test_too_many_polymers_is_rejected(self, tool):
        message = await self._error(
            tool,
            {
                "polymers": [
                    {"molecule_type": "protein", "sequence": SEQUENCE}
                    for _ in range(13)
                ]
            },
        )
        assert "at most 12" in message

    @pytest.mark.parametrize(
        "field,value",
        [
            ("recycling_steps", 11),
            ("sampling_steps", 5),
            ("diffusion_samples", 26),
            ("step_scale", 9.0),
            ("sampling_steps_affinity", 1001),
            ("diffusion_samples_affinity", 11),
        ],
    )
    async def test_out_of_range_settings_name_the_field(self, tool, field, value):
        message = await self._error(
            tool, {"sequence": SEQUENCE, field: value}
        )
        assert field in message

    async def test_an_msa_on_a_dna_chain_is_rejected(self, tool, storage):
        msa_ref = await storage.store(data_type="msa", content=A3M.encode())
        message = await self._error(
            tool,
            {
                "polymers": [
                    {
                        "molecule_type": "dna",
                        "sequence": "ATCG",
                        "msa_ref": msa_ref,
                    }
                ]
            },
        )
        assert "protein chain" in message

    async def test_no_input_at_all_asks_for_the_right_thing(self, tool):
        message = await self._error(tool, {})
        assert "polymers" in message
        assert "sequence_ref" in message

    @pytest.mark.parametrize("ref", ["sequence_missing1", "not-a-ref"])
    async def test_a_bad_sequence_reference_points_at_the_producer(self, tool, ref):
        message = await self._error(tool, {"sequence_ref": ref})
        assert "fetch_protein_sequence" in message

    @pytest.mark.parametrize("ref", ["msa_missing1", "not-a-ref"])
    async def test_a_bad_msa_reference_points_at_msa_search(self, tool, ref):
        message = await self._error(tool, {"sequence": SEQUENCE, "msa_ref": ref})
        assert "msa_search" in message


@pytest.mark.asyncio
class TestOutputs:
    async def test_the_structure_is_stored_as_mmcif_not_pdb(
        self, tool, storage, patch_httpx
    ):
        # Storing mmCIF under a pdb_ ref is the shortcut this tool exists to
        # avoid; the ref prefix is what every downstream consumer dispatches on.
        patch_httpx(json_response(one_structure()))

        result = await tool.execute({"sequence": SEQUENCE})

        assert result.data.startswith("mmcif_")
        assert result.metadata["data_type"] == "mmcif"
        assert (await storage.retrieve(result.data)).decode() == _mmcif("primary")

    async def test_confidence_is_framed_to_pre_empt_biotite_mmcif_parsing(
        self, tool, storage, patch_httpx
    ):
        # A "mean confidence / pLDDT" question was answered by trying to parse the
        # mmCIF with biotite.structure.io.pdbx (wrong API) instead of reporting
        # this score. The message now says the score is the summary answer and
        # points per-residue reads at load_pdb_structure's temp_factor.
        patch_httpx(json_response(one_structure(confidence=0.77)))

        result = await tool.execute({"sequence": SEQUENCE})

        msg = result.metadata["message"]
        assert "mean-pLDDT equivalent" in msg
        assert "temp_factor" in msg
        assert "biotite" in msg

    async def test_the_highest_confidence_sample_is_primary(
        self, tool, storage, patch_httpx
    ):
        patch_httpx(
            json_response(
                {
                    "structures": [
                        {"structure": _mmcif("low")},
                        {"structure": _mmcif("best")},
                        {"structure": _mmcif("mid")},
                    ],
                    "confidence_scores": [0.41, 0.93, 0.66],
                }
            )
        )

        result = await tool.execute(
            {"sequence": SEQUENCE, "diffusion_samples": 3}
        )

        assert (await storage.retrieve(result.data)).decode() == _mmcif("best")
        assert result.metadata["confidence_score"] == 0.93
        # The others are kept, ranked, and named so the model can compare them.
        extras = result.metadata["outputs"]
        assert (await storage.retrieve(extras["structure_2"])).decode() == _mmcif(
            "mid"
        )
        assert (await storage.retrieve(extras["structure_3"])).decode() == _mmcif(
            "low"
        )
        assert extras["structure_2"] in result.metadata["message"]

    async def test_mismatched_confidence_scores_do_not_reorder_samples(
        self, tool, storage, patch_httpx
    ):
        # A scores list of a different length cannot be trusted positionally.
        # Keeping the response order beats attaching one sample's score to
        # another and promoting the wrong structure.
        patch_httpx(
            json_response(
                {
                    "structures": [
                        {"structure": _mmcif("first")},
                        {"structure": _mmcif("second")},
                    ],
                    "confidence_scores": [0.9],
                }
            )
        )

        result = await tool.execute({"sequence": SEQUENCE})

        assert (await storage.retrieve(result.data)).decode() == _mmcif("first")
        assert result.metadata["confidence_score"] is None

    async def test_affinity_appears_inline_and_in_metadata(
        self, tool, patch_httpx
    ):
        # Inline because the numbers are what the model reasons over; in
        # metadata because the findings ledger should not have to re-parse prose.
        patch_httpx(
            json_response(
                {
                    **one_structure(),
                    "affinities": {
                        "L1": {
                            # What the service really sends here is not a
                            # pIC50; see TestAffinityUnits below.
                            "affinity_pic50": [8.73],
                            "affinity_pred_value": [-0.4],
                            "affinity_probability_binary": [0.82],
                        }
                    },
                }
            )
        )

        result = await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [
                    {"smiles": ASPIRIN, "id": "L1", "predict_affinity": True}
                ],
            }
        )

        assert result.metadata["affinity_pic50"] == 6.4
        assert result.metadata["affinity_log_ic50_um"] == -0.4
        assert result.metadata["affinity_binding_probability"] == 0.82
        message = result.metadata["message"]
        assert "pIC50: 6.40" in message
        assert "log10(IC50 in uM): -0.40" in message
        assert "0.82" in message

    async def test_pic50_is_derived_when_only_the_raw_value_comes_back(
        self, tool, patch_httpx
    ):
        # affinity_pred_value is log10(IC50 in uM), so pIC50 = 6 - value.
        patch_httpx(
            json_response(
                {
                    **one_structure(),
                    "affinities": {"L1": {"affinity_pred_value": [1.0]}},
                }
            )
        )

        result = await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [
                    {"smiles": ASPIRIN, "id": "L1", "predict_affinity": True}
                ],
            }
        )

        assert result.metadata["affinity_pic50"] == pytest.approx(5.0)
        assert result.metadata["affinity_ic50_um"] == pytest.approx(10.0)

    async def test_affinity_is_labelled_as_an_estimate(self, tool, patch_httpx):
        # Reported as a measured IC50 it is actively misleading, and the model
        # will repeat whatever wording it is given.
        patch_httpx(
            json_response(
                {
                    **one_structure(),
                    "affinities": {"L1": {"affinity_pred_value": [-1.1]}},
                }
            )
        )

        result = await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [
                    {"smiles": ASPIRIN, "id": "L1", "predict_affinity": True}
                ],
            }
        )

        message = result.metadata["message"]
        assert "not measured IC50 values" in message
        assert "ranking" in message

    async def test_a_missing_affinity_response_is_stated_not_silent(
        self, tool, patch_httpx
    ):
        patch_httpx(json_response(one_structure()))

        result = await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [
                    {"smiles": ASPIRIN, "id": "L1", "predict_affinity": True}
                ],
            }
        )

        assert "no affinity values" in result.metadata["message"]

    async def test_a_pae_matrix_is_stored_as_an_array(
        self, tool, storage, patch_httpx
    ):
        pae = np.random.default_rng(0).random((8, 8)).tolist()
        patch_httpx(json_response({**one_structure(), "pae": pae}))

        result = await tool.execute(
            {"sequence": SEQUENCE, "write_full_pae": True}
        )

        pae_ref = result.metadata["outputs"]["pae"]
        assert pae_ref.startswith("array_")
        loaded = np.load(io.BytesIO(await storage.retrieve(pae_ref)))["data"]
        assert loaded.shape == (8, 8)
        assert np.allclose(loaded, np.asarray(pae, dtype=np.float32))
        assert "aevom.load_array" in result.metadata["message"]

    async def test_every_stored_reference_is_listed_in_metadata(
        self, tool, patch_httpx
    ):
        # The convention the storage strip and findings ledger read.
        pae = np.random.default_rng(1).random((8, 8)).tolist()
        patch_httpx(
            json_response(
                {
                    "structures": [
                        {"structure": _mmcif("a")},
                        {"structure": _mmcif("b")},
                    ],
                    "confidence_scores": [0.7, 0.6],
                    "pae": pae,
                }
            )
        )

        result = await tool.execute(
            {"sequence": SEQUENCE, "diffusion_samples": 2, "write_full_pae": True}
        )

        outputs = result.metadata["outputs"]
        assert set(outputs) == {"structure", "structure_2", "pae"}
        assert result.metadata["all_references"] == list(outputs.values())

    async def test_small_numeric_fields_do_not_become_references(
        self, tool, patch_httpx
    ):
        # A reference the model has to chase is worse than three numbers it can
        # read, so only genuinely large arrays are stored.
        patch_httpx(
            json_response({**one_structure(), "ptm": [0.81, 0.79, 0.80]})
        )

        result = await tool.execute({"sequence": SEQUENCE})

        assert "ptm" not in result.metadata["outputs"]

    async def test_the_message_routes_the_mmcif_to_the_right_next_tools(
        self, tool, patch_httpx
    ):
        # Without this the model either fails to display the structure or
        # converts it needlessly before viewing.
        patch_httpx(json_response(one_structure()))

        result = await tool.execute({"sequence": SEQUENCE})

        message = result.metadata["message"]
        assert "visualize_pdb" in message
        assert "convert_structure" in message
        assert "vina_dock" in message

    async def test_the_chain_layout_is_recorded_for_downstream_queries(
        self, tool, patch_httpx
    ):
        patch_httpx(json_response(one_structure()))

        result = await tool.execute(
            {
                "polymers": [
                    {"molecule_type": "protein", "sequence": SEQUENCE},
                    {"molecule_type": "dna", "sequence": "ATCGATCG"},
                ],
                "ligands": [{"smiles": ASPIRIN}],
            }
        )

        assert result.metadata["chain_ids"] == ["A", "B"]
        assert result.metadata["molecule_types"] == ["protein", "dna"]
        assert result.metadata["ligand_ids"] == ["L1"]
        assert result.metadata["protein_count"] == 1


@pytest.mark.asyncio
class TestTheServiceGetsToSpeak:
    """What the endpoint says about how it produced a number reaches the model.

    A deployment serving Boltz itself knows things the caller cannot see: that it
    folded from a single sequence because it had no alignment for that protein,
    or that it answered from a cheap tier whose ranking is trustworthy and whose
    absolute value is not.

    Dropping that was measurable. A benchmark run screened a molecule at one
    diffusion sample, read a binding probability of 0.38, and was later scored at
    ten samples as 0.89 -- failing on a gate it had actually cleared. The service
    returned "confirm a finalist at diffusion_samples=10 before relying on the
    value" with that very response, and nothing carried it to the model.
    """

    async def test_a_service_note_is_reported(self, tool, patch_httpx):
        payload = one_structure()
        payload["warnings"] = [
            "Co-folded at diffusion_samples=1 (screening tier).",
            "No multiple sequence alignment was available for chain A.",
        ]
        patch_httpx(json_response(payload))

        result = await tool.execute({"sequence": SEQUENCE})

        message = result.metadata["message"]
        assert "From the prediction service:" in message
        assert "screening tier" in message
        assert "No multiple sequence alignment" in message

    async def test_silence_stays_silent(self, tool, patch_httpx):
        patch_httpx(json_response(one_structure()))

        result = await tool.execute({"sequence": SEQUENCE})

        assert "From the prediction service" not in result.metadata["message"]

    async def test_empty_notes_are_not_rendered(self, tool, patch_httpx):
        payload = one_structure()
        payload["warnings"] = ["", "   "]
        patch_httpx(json_response(payload))

        result = await tool.execute({"sequence": SEQUENCE})

        assert "From the prediction service" not in result.metadata["message"]


class TestDeployment:
    """Where Boltz-2 lives, and how long Aevom will wait for it.

    Every other service-backed tool reads its endpoint from the environment;
    this one hardcoded the hosted NIM, so a deployment running Boltz-2 on a
    card it already owns had nowhere to say so. The wait matters for the same
    reason: the client timeout is derived from the poll, and a ten-sample
    co-fold takes a median of 527 s on a GV100 against a 300 s default -- long
    enough that Aevom would abandon the request *after* the GPU finished it.

    Both default to what they were, and both treat an empty value as unset,
    because a compose file declares more variables than it sets.
    """

    def test_the_hosted_nim_is_still_the_default(self, tool, monkeypatch):
        monkeypatch.delenv("BOLTZ2_API_URL", raising=False)

        assert tool.api_endpoint == (
            "https://health.api.nvidia.com/v1/biology/mit/boltz2/predict"
        )

    def test_an_endpoint_can_be_pointed_elsewhere(self, tool, monkeypatch):
        monkeypatch.setenv("BOLTZ2_API_URL", "http://boltz:8000/predict")

        assert tool.api_endpoint == "http://boltz:8000/predict"

    def test_an_empty_endpoint_means_the_default(self, tool, monkeypatch):
        monkeypatch.setenv("BOLTZ2_API_URL", "")

        assert "health.api.nvidia.com" in tool.api_endpoint

    @pytest.mark.asyncio
    async def test_the_wait_is_configurable(self, tool, patch_httpx, monkeypatch):
        import importlib

        from api.tools import boltz2

        monkeypatch.setenv("BOLTZ2_POLL_SECONDS", "1800")
        importlib.reload(boltz2)
        try:
            assert boltz2.POLL_SECONDS == 1800
        finally:
            monkeypatch.delenv("BOLTZ2_POLL_SECONDS", raising=False)
            importlib.reload(boltz2)
        assert boltz2.POLL_SECONDS == 300


@pytest.mark.asyncio
class TestPocketStrength:
    """A pocket constraint is a hint unless it is told to be a requirement.

    The hosted service rejects unknown fields outright -- `seed` comes back
    422 "Extra inputs are not permitted" -- so max_distance and force being
    accepted proves they are supported, and a decoy-pocket probe proves they
    are honoured: forcing the ligand at residues 1-5 moved the predicted
    affinity from +1.891 to +2.312 over three repeats each, against a
    within-condition spread of 0.08 and with no overlap.
    """

    async def test_a_pocket_is_advisory_unless_told_otherwise(self, tool, patch_httpx):
        handler = json_response(one_structure())
        patch_httpx(handler)

        await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}],
                "constraints": [
                    {
                        "constraint_type": "pocket",
                        "binder": "L1",
                        "contacts": [{"id": "A", "residue_index": 3}],
                    }
                ],
            }
        )

        pocket = sent_body(handler)["constraints"][0]
        assert "max_distance" not in pocket
        assert "force" not in pocket

    async def test_a_forced_pocket_reaches_the_wire(self, tool, patch_httpx):
        handler = json_response(one_structure())
        patch_httpx(handler)

        await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}],
                "constraints": [
                    {
                        "constraint_type": "pocket",
                        "binder": "L1",
                        "contacts": [{"id": "A", "residue_index": 3}],
                        "max_distance": 6.0,
                        "force": True,
                    }
                ],
            }
        )

        pocket = sent_body(handler)["constraints"][0]
        assert pocket["max_distance"] == 6.0
        assert pocket["force"] is True

    async def test_a_bond_constraint_gains_neither(self, tool, patch_httpx):
        handler = json_response(one_structure())
        patch_httpx(handler)

        await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1"}],
                "constraints": [
                    {
                        "constraint_type": "bond",
                        "atoms": [
                            {"id": "A", "residue_index": 3, "atom_name": "SG"},
                            {"id": "L1", "residue_index": 1, "atom_name": "C1"},
                        ],
                        "max_distance": 6.0,
                        "force": True,
                    }
                ],
            }
        )

        bond = sent_body(handler)["constraints"][0]
        assert set(bond) == {"constraint_type", "atoms"}


@pytest.mark.asyncio
class TestAffinityUnits:
    """The two affinity numbers must agree, and they did not.

    pIC50 is -log10(IC50 in molar) and affinity_pred_value is log10(IC50 in
    micromolar), so the two sum to exactly 6 by definition. Measured over 700
    affinity read-outs in two full SMDD-Bench runs, the pIC50 this tool printed
    and the log10 it printed beside it summed to 8.14 on median and to 6.00
    zero times: the service's affinity_pic50 field is (6 - pred) * 1.364, a
    binding free energy in kcal/mol, and it was being relabelled. The model
    read the first number it was given, so it was reasoning about potency
    roughly 2.1 log units better than the model had predicted.
    """

    async def test_the_two_reported_numbers_sum_to_six(self, tool, patch_httpx):
        patch_httpx(
            json_response(
                {
                    **one_structure(),
                    # The kcal/mol-scaled value the service actually sends for
                    # a pred of -1.77: (6 + 1.77) * 1.364 = 10.59.
                    "affinities": {
                        "L1": {"affinity_pic50": [10.59], "affinity_pred_value": [-1.77]}
                    },
                }
            )
        )

        result = await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1", "predict_affinity": True}],
            }
        )

        pic50 = result.metadata["affinity_pic50"]
        log_ic50 = result.metadata["affinity_log_ic50_um"]
        assert pic50 + log_ic50 == pytest.approx(6.0)
        assert pic50 == pytest.approx(7.77)

    async def test_the_service_field_is_not_echoed_as_a_pic50(self, tool, patch_httpx):
        patch_httpx(
            json_response(
                {
                    **one_structure(),
                    "affinities": {
                        "L1": {"affinity_pic50": [10.59], "affinity_pred_value": [-1.77]}
                    },
                }
            )
        )

        result = await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1", "predict_affinity": True}],
            }
        )

        assert "10.59" not in result.metadata["message"]

    async def test_the_comparable_number_is_stated_first(self, tool, patch_httpx):
        """Whichever number leads is the one that gets reasoned about."""
        patch_httpx(
            json_response(
                {
                    **one_structure(),
                    "affinities": {"L1": {"affinity_pred_value": [-1.77]}},
                }
            )
        )

        result = await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1", "predict_affinity": True}],
            }
        )

        message = result.metadata["message"]
        assert message.index("log10(IC50 in uM)") < message.index("pIC50")

    async def test_an_affinity_with_no_comparable_value_is_stated_not_silent(
        self, tool, patch_httpx
    ):
        """A response carrying only the service's own field says so.

        Every response observed in practice carries affinity_pred_value. If one
        does not, the honest answer is that nothing comparable came back --
        not a number in units this tool cannot vouch for.
        """
        patch_httpx(
            json_response(
                {
                    **one_structure(),
                    "affinities": {"L1": {"affinity_pic50": [10.59]}},
                }
            )
        )

        result = await tool.execute(
            {
                "sequence": SEQUENCE,
                "ligands": [{"smiles": ASPIRIN, "id": "L1", "predict_affinity": True}],
            }
        )

        message = result.metadata["message"]
        assert "no affinity values" in message
        assert "10.59" not in message


@pytest.mark.asyncio
class TestSamplingTransparency:
    """Whether a single-sample pose is presented as provisional.

    diffusion_samples defaults to 1 for cost, which is defensible, but diffusion
    is stochastic: one sample is one draw. The confidence score does not convey
    that -- a lone sample can score 0.9 and still not reproduce -- so the
    response has to say it, or the pose gets read as settled fact.
    """

    async def test_a_single_sample_is_flagged_as_one_draw(self, tool, patch_httpx):
        patch_httpx(json_response(one_structure()))

        result = await tool.execute({"sequence": SEQUENCE})

        message = result.metadata["message"]
        assert "single draw" in message
        assert "stochastic" in message

    async def test_the_note_says_how_to_get_a_consensus(self, tool, patch_httpx):
        # A caveat the reader cannot act on is just hedging.
        patch_httpx(json_response(one_structure()))

        result = await tool.execute({"sequence": SEQUENCE})

        message = result.metadata["message"]
        assert "diffusion_samples=5" in message
        assert "plip_interactions" in message

    async def test_asking_for_several_samples_drops_the_note(self, tool, patch_httpx):
        patch_httpx(
            json_response(
                {
                    "structures": [
                        {"structure": _mmcif("a")},
                        {"structure": _mmcif("b")},
                    ],
                    "confidence_scores": [0.9, 0.8],
                }
            )
        )

        result = await tool.execute({"sequence": SEQUENCE, "diffusion_samples": 2})

        assert "single draw" not in result.metadata["message"]

    async def test_the_confidence_line_does_not_claim_a_best_of_one(
        self, tool, patch_httpx
    ):
        # "the best of the returned samples" is a strange thing to say about one
        # sample, and it implies a selection that did not happen.
        patch_httpx(json_response(one_structure(confidence=0.87)))

        result = await tool.execute({"sequence": SEQUENCE})

        message = result.metadata["message"]
        assert "Confidence: 0.870" in message
        assert "best of" not in message

    async def test_the_confidence_line_counts_the_samples_when_there_are_several(
        self, tool, patch_httpx
    ):
        patch_httpx(
            json_response(
                {
                    "structures": [
                        {"structure": _mmcif("a")},
                        {"structure": _mmcif("b")},
                        {"structure": _mmcif("c")},
                    ],
                    "confidence_scores": [0.5, 0.93, 0.7],
                }
            )
        )

        result = await tool.execute({"sequence": SEQUENCE, "diffusion_samples": 3})

        assert "best of 3 samples" in result.metadata["message"]

    async def test_the_schema_explains_when_to_raise_the_count(self, tool):
        described = tool.get_schema()["function"]["parameters"]["properties"]
        description = described["diffusion_samples"]["description"]
        assert "stochastic" in description
        assert "plip_interactions" in description

    async def test_affinity_sampling_still_defaults_above_one(self, tool):
        # The affinity head averages several samples already, which is why the
        # note above is about the pose rather than about the pIC50.
        from api.tools.boltz2 import DEFAULT_AFFINITY_DIFFUSION_SAMPLES

        assert DEFAULT_AFFINITY_DIFFUSION_SAMPLES > 3


@pytest.mark.asyncio
class TestFailures:
    async def test_a_422_surfaces_the_endpoint_complaint(self, tool, patch_httpx):
        patch_httpx(
            json_response(
                {"error": "polymers.0.sequence: invalid residue at position 4"},
                status=422,
            )
        )

        result = await tool.execute({"sequence": SEQUENCE})

        assert result.content_type == ContentType.ERROR
        assert "invalid residue at position 4" in result.data
        assert "422" in result.data

    async def test_a_queued_job_is_reported_rather_than_parsed(
        self, tool, patch_httpx
    ):
        patch_httpx(json_response({"id": "job-1"}, status=202))

        result = await tool.execute({"sequence": SEQUENCE})

        assert result.content_type == ContentType.ERROR
        assert "queued" in result.data

    async def test_an_empty_structures_list_is_an_error(self, tool, patch_httpx):
        patch_httpx(json_response({"structures": [], "confidence_scores": []}))

        result = await tool.execute({"sequence": SEQUENCE})

        assert result.content_type == ContentType.ERROR
        assert "no structures" in result.data

    async def test_structures_without_coordinates_are_an_error(
        self, tool, patch_httpx
    ):
        patch_httpx(
            json_response({"structures": [{"format": "mmcif", "structure": ""}]})
        )

        result = await tool.execute({"sequence": SEQUENCE})

        assert result.content_type == ContentType.ERROR
        assert "no coordinate data" in result.data


class TestSchema:
    def test_output_format_is_not_offered_as_a_parameter(self):
        properties = Boltz2Tool().get_schema()["function"]["parameters"][
            "properties"
        ]
        assert "output_format" not in properties

    def test_the_schema_points_at_the_bound_skill(self):
        description = Boltz2Tool().get_schema()["function"]["description"]
        assert 'load_skill(skill_id="boltz2-nim")' in description

    def test_the_description_distinguishes_the_three_affinity_routes(self):
        description = Boltz2Tool().get_schema()["function"]["description"]
        assert "openfold3" in description
        assert "vina_dock" in description

    def test_reference_parameters_are_annotated_for_the_ref_resolver(self):
        properties = Boltz2Tool().get_schema()["function"]["parameters"][
            "properties"
        ]
        assert properties["msa_ref"]["x-aevom-reference"]["data_type"] == "msa"
        assert (
            properties["sequence_ref"]["x-aevom-reference"]["data_type"]
            == "sequence"
        )
