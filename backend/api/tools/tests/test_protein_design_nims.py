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

"""Tests for the RFDiffusion and ProteinMPNN builtin tools.

These are the first ``NvidiaApiTool`` subclasses with coverage, so the fixtures
here establish the pattern: stub ``_get_api_key`` (there is no key in CI) and
intercept the HTTP call with a transport, leaving the real request construction,
response parsing and storage writes under test.

The payloads are the shapes the live endpoints actually returned, which differ
from the upstream reference docs in two ways that matter -- the multi-FASTA opens
with an ``>input`` record that has no entry in ``scores``, and a validation
failure arrives as ``{"error": ...}`` rather than FastAPI's ``detail``.
"""

import io
import json

import httpx
import numpy as np
import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.base import ContentType
from api.tools.proteinmpnn import ProteinMPNNTool
from api.tools.rfdiffusion import DUMMY_INPUT_PDB, RFDiffusionTool

# A three-residue backbone, enough for the CA-counting and chaining assertions.
BACKBONE_PDB = "\n".join(
    [
        "ATOM      1  N   GLY A   1      -9.464  -7.089 -12.401  1.00  0.00",
        "ATOM      2  CA  GLY A   1      -8.596  -6.009 -11.949  1.00  0.00",
        "ATOM      3  C   GLY A   1      -7.130  -6.340 -12.201  1.00  0.00",
        "ATOM      4  CA  GLY A   2      -6.596  -5.009 -10.949  1.00  0.00",
        "ATOM      5  CA  GLY A   3      -5.596  -4.009  -9.949  1.00  0.00",
        "END",
    ]
)

# Verbatim response shape from the live ProteinMPNN endpoint: an `>input` record
# for the supplied structure, then one record per design.
MFASTA = (
    ">input, score=2.6361, global_score=2.6361, fixed_chains=[], "
    "designed_chains=['A'], model_name=v_48_002\n"
    "GGG\n"
    ">T=0.1, sample=1, score=1.2134, global_score=1.2134, seq_recovery=0.0000\n"
    "MVE\n"
    ">T=0.1, sample=2, score=1.1795, global_score=1.1795, seq_recovery=0.0000\n"
    "MVK\n"
)
DESIGN_SCORES = [1.2133920192718506, 1.1794562339782715]
PROBS = np.random.default_rng(0).random((2, 3, 21)).tolist()


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


def json_response(payload: dict, status: int = 200):
    """A transport handler returning a fixed JSON payload.

    Records the request it saw on the handler, so tests can assert on the body
    and headers the tool actually built.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        handler.request = request
        return httpx.Response(status, json=payload)

    handler.request = None
    return handler


@pytest.fixture
def patch_httpx(monkeypatch):
    """Swap in a mock transport, leaving the rest of the request path real.

    Only the wire is stubbed, so header construction, the poll-seconds value and
    the error translation in ``NvidiaApiTool._call_nvidia_api`` stay under test.
    """

    def apply(handler):
        real_client = httpx.AsyncClient

        def factory(*args, **kwargs):
            kwargs.pop("transport", None)
            return real_client(*args, transport=httpx.MockTransport(handler), **kwargs)

        monkeypatch.setattr(httpx, "AsyncClient", factory)

    return apply


@pytest.mark.asyncio
class TestRFDiffusion:
    async def test_de_novo_injects_the_dummy_structure(self, storage, patch_httpx):
        # The endpoint 422s a de novo request with no input_pdb, so the tool has
        # to supply one. If it stops doing so, every de novo design breaks.
        handler = json_response({"output_pdb": BACKBONE_PDB, "elapsed_ms": 7613})
        patch_httpx(handler)
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute({"contigs": "80-120"})

        assert result.content_type == ContentType.DATA_REFERENCE
        sent = json.loads(handler.request.content)
        assert sent["input_pdb"] == DUMMY_INPUT_PDB
        assert sent["contigs"] == "80-120"
        assert result.metadata["design_mode"] == "de_novo"

    async def test_the_poll_header_is_set(self, storage, patch_httpx):
        handler = json_response({"output_pdb": BACKBONE_PDB})
        patch_httpx(handler)
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        await tool.execute({"contigs": "100"})

        assert handler.request.headers["NVCF-POLL-SECONDS"] == "120"
        assert handler.request.headers["Authorization"] == "Bearer test-key"

    async def test_a_supplied_structure_is_forwarded(self, storage, patch_httpx):
        handler = json_response({"output_pdb": BACKBONE_PDB})
        patch_httpx(handler)
        target = await storage.store(data_type="pdb", content=b"ATOM  target\nEND\n")
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute(
            {"contigs": "A1-10/0 50-80", "pdb_ref": target}
        )

        sent = json.loads(handler.request.content)
        assert sent["input_pdb"] == "ATOM  target\nEND\n"
        assert result.metadata["design_mode"] == "motif_scaffolding"
        assert result.metadata["input_pdb_ref"] == target

    async def test_hotspots_mark_it_as_binder_design(self, storage, patch_httpx):
        patch_httpx(json_response({"output_pdb": BACKBONE_PDB}))
        target = await storage.store(data_type="pdb", content=b"ATOM  x\n")
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute(
            {"contigs": "A1-10/0 50-80", "pdb_ref": target, "hotspot_res": ["A5"]}
        )

        assert result.metadata["design_mode"] == "binder_design"
        assert result.metadata["hotspot_res"] == ["A5"]

    async def test_stores_the_backbone_as_a_pdb_reference(self, storage, patch_httpx):
        patch_httpx(json_response({"output_pdb": BACKBONE_PDB}))
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute({"contigs": "3"})

        assert result.data.startswith("pdb_")
        assert (await storage.retrieve(result.data)).decode() == BACKBONE_PDB
        # Counted from CA atoms, not ATOM lines: the first residue has N and C too.
        assert result.metadata["num_residues"] == 3

    async def test_the_result_points_at_the_next_tool(self, storage, patch_httpx):
        # A backbone alone is not a protein. The model has to be told the next
        # step, or it reports a design that is entirely glycine as the answer.
        patch_httpx(json_response({"output_pdb": BACKBONE_PDB}))
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute({"contigs": "3"})

        assert "proteinmpnn_design_sequence" in result.metadata["message"]
        assert "no sequence" in result.metadata["message"]

    async def test_a_422_surfaces_the_endpoint_complaint(self, storage, patch_httpx):
        # The live 422 body is {"error": "..."} and names the field to fix, which
        # is far more useful to the model than the status code alone.
        patch_httpx(
            json_response(
                {"error": "Either input_pdb or input_pdb_asset must be specified"},
                status=422,
            )
        )
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute({"contigs": "80-120"})

        assert result.content_type == ContentType.ERROR
        assert "Either input_pdb or input_pdb_asset must be specified" in result.data
        assert "422" in result.data

    async def test_a_queued_job_is_reported_rather_than_parsed(
        self, storage, patch_httpx
    ):
        # A 202 carries a job id, not results. Aevom cannot resume it, so saying
        # so beats returning a reference to nothing.
        patch_httpx(json_response({"id": "job-1"}, status=202))
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute({"contigs": "80-120"})

        assert result.content_type == ContentType.ERROR
        assert "queued" in result.data

    async def test_an_oversized_response_is_reported_with_the_limit(
        self, storage, patch_httpx, monkeypatch
    ):
        monkeypatch.setattr("api.bounded_http.MAX_RESPONSE_BYTES", 1024)
        patch_httpx(json_response({"output_pdb": "ATOM" * 1000}))
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute({"contigs": "80-120"})

        assert result.content_type == ContentType.ERROR
        assert "larger than Aevom's 1 KiB limit" in result.data
        assert "smaller request" in result.data

    async def test_a_redirect_is_reported_rather_than_followed(
        self, storage, patch_httpx
    ):
        patch_httpx(
            lambda request: httpx.Response(
                307, headers={"location": "http://elsewhere.example/"}
            )
        )
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute({"contigs": "80-120"})

        assert result.content_type == ContentType.ERROR
        assert "http://elsewhere.example/" in result.data
        assert "does not follow redirects" in result.data

    async def test_missing_contigs_is_rejected_before_the_call(self, storage):
        tool = RFDiffusionTool()
        tool._storage = storage
        result = await tool.execute({})
        assert result.content_type == ContentType.ERROR
        assert "contigs is required" in result.data

    async def test_hotspots_without_a_target_are_rejected(self, storage):
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"
        result = await tool.execute({"contigs": "100", "hotspot_res": ["A5"]})
        assert result.content_type == ContentType.ERROR
        assert "requires pdb_ref" in result.data

    @pytest.mark.parametrize("steps", [0, 51, "many"])
    async def test_out_of_range_diffusion_steps_are_rejected(self, storage, steps):
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"
        result = await tool.execute({"contigs": "100", "diffusion_steps": steps})
        assert result.content_type == ContentType.ERROR
        assert "diffusion_steps" in result.data

    async def test_an_unknown_reference_is_a_clean_error(self, storage, patch_httpx):
        tool = RFDiffusionTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"
        result = await tool.execute({"contigs": "100", "pdb_ref": "pdb_missing1"})
        assert result.content_type == ContentType.ERROR


@pytest.mark.asyncio
class TestProteinMPNN:
    async def _run(self, storage, patch_httpx, arguments, payload=None):
        payload = payload or {
            "mfasta": MFASTA,
            "scores": DESIGN_SCORES,
            "probs": PROBS,
        }
        handler = json_response(payload)
        patch_httpx(handler)
        backbone = await storage.store(
            data_type="pdb", content=BACKBONE_PDB.encode()
        )
        tool = ProteinMPNNTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"
        result = await tool.execute({"pdb_ref": backbone, **arguments})
        return result, handler, backbone

    async def test_the_input_record_is_excluded_from_the_designs(
        self, storage, patch_httpx
    ):
        # The mfasta opens with an `>input` record for the supplied structure.
        # Counting it as a design shifts every score onto the wrong sequence.
        result, _, _ = await self._run(storage, patch_httpx, {})

        assert result.metadata["num_sequences"] == 2
        assert result.metadata["scores"] == DESIGN_SCORES
        assert "GGG" not in result.metadata["message"]

    async def test_the_lowest_scoring_design_is_reported_as_best(
        self, storage, patch_httpx
    ):
        # Scores are per-residue negative log-likelihoods, so lower is better --
        # the opposite of what the upstream API reference claims.
        result, _, _ = await self._run(storage, patch_httpx, {})

        assert result.metadata["best_score"] == min(DESIGN_SCORES)
        assert result.metadata["best_sequence"] == "MVK"
        assert "lower is better" in result.metadata["message"]
        assert "Best — design 2 (T=0.1)" in result.metadata["message"]

    async def test_scores_fall_back_to_the_headers_on_a_count_mismatch(
        self, storage, patch_httpx
    ):
        # Rather than zip a short scores array against the designs and mislabel
        # them, parse the score each header already carries.
        result, _, _ = await self._run(
            storage,
            patch_httpx,
            {},
            payload={"mfasta": MFASTA, "scores": [0.5], "probs": []},
        )

        assert result.metadata["scores"] == [1.2134, 1.1795]

    async def test_the_fasta_is_stored_verbatim(self, storage, patch_httpx):
        result, _, _ = await self._run(storage, patch_httpx, {})

        assert result.data.startswith("fasta_")
        assert (await storage.retrieve(result.data)).decode() == MFASTA

    async def test_probs_are_stored_as_a_loadable_array(self, storage, patch_httpx):
        # (num_seq, num_residues, 21) runs to hundreds of thousands of numbers on
        # a real protein, so it must never reach the model's context.
        result, _, _ = await self._run(storage, patch_httpx, {})

        probs_ref = result.metadata["probs_ref"]
        assert probs_ref.startswith("array_")
        with np.load(io.BytesIO(await storage.retrieve(probs_ref))) as archive:
            assert archive["data"].shape == (2, 3, 21)
        assert "aevom.load_array" in result.metadata["message"]
        # Not a single probability in the message itself.
        assert str(PROBS[0][0][0])[:8] not in result.metadata["message"]

    async def test_both_references_are_listed_in_metadata(self, storage, patch_httpx):
        result, _, _ = await self._run(storage, patch_httpx, {})

        assert set(result.metadata["outputs"]) == {"mfasta", "probs"}
        assert len(result.metadata["all_references"]) == 2

    async def test_a_missing_probs_field_is_not_fatal(self, storage, patch_httpx):
        result, _, _ = await self._run(
            storage,
            patch_httpx,
            {},
            payload={"mfasta": MFASTA, "scores": DESIGN_SCORES},
        )

        assert result.content_type == ContentType.DATA_REFERENCE
        assert "probs_ref" not in result.metadata

    async def test_sampling_temp_is_sent_as_a_list(self, storage, patch_httpx):
        # The API rejects a bare number; it wants a list even for one value.
        _, handler, _ = await self._run(storage, patch_httpx, {"sampling_temp": 0.3})

        assert json.loads(handler.request.content)["sampling_temp"] == [0.3]

    async def test_optional_controls_are_forwarded_under_api_names(
        self, storage, patch_httpx
    ):
        _, handler, _ = await self._run(
            storage,
            patch_httpx,
            {"chains": ["A"], "omit_aas": ["C"], "num_seq_per_target": 4},
        )

        sent = json.loads(handler.request.content)
        assert sent["input_pdb_chains"] == ["A"]
        assert sent["omit_AAs"] == ["C"]
        assert sent["num_seq_per_target"] == 4

    async def test_omitted_controls_are_left_out_of_the_payload(
        self, storage, patch_httpx
    ):
        _, handler, _ = await self._run(storage, patch_httpx, {})

        sent = json.loads(handler.request.content)
        assert "input_pdb_chains" not in sent
        assert "omit_AAs" not in sent
        assert "random_seed" not in sent

    async def test_the_backbone_is_sent_inline(self, storage, patch_httpx):
        _, handler, _ = await self._run(storage, patch_httpx, {})

        assert json.loads(handler.request.content)["input_pdb"] == BACKBONE_PDB

    async def test_a_422_surfaces_the_endpoint_complaint(self, storage, patch_httpx):
        patch_httpx(
            json_response(
                {"error": "Validation Errors: contigs: Extra inputs are not permitted"},
                status=422,
            )
        )
        backbone = await storage.store(data_type="pdb", content=BACKBONE_PDB.encode())
        tool = ProteinMPNNTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute({"pdb_ref": backbone})

        assert result.content_type == ContentType.ERROR
        assert "Extra inputs are not permitted" in result.data

    async def test_an_empty_mfasta_is_a_clean_error(self, storage, patch_httpx):
        result, _, _ = await self._run(
            storage, patch_httpx, {}, payload={"mfasta": "", "scores": []}
        )

        assert result.content_type == ContentType.ERROR
        assert "no designed sequences" in result.data

    async def test_missing_pdb_ref_is_rejected_before_the_call(self, storage):
        tool = ProteinMPNNTool()
        tool._storage = storage
        result = await tool.execute({})
        assert result.content_type == ContentType.ERROR
        assert "pdb_ref is required" in result.data

    @pytest.mark.parametrize(
        "arguments,expected",
        [
            ({"num_seq_per_target": 0}, "num_seq_per_target"),
            ({"num_seq_per_target": 101}, "num_seq_per_target"),
            ({"sampling_temp": 1.5}, "sampling_temp"),
            ({"sampling_temp": "hot"}, "sampling_temp"),
        ],
    )
    async def test_out_of_range_settings_are_rejected(
        self, storage, arguments, expected
    ):
        backbone = await storage.store(data_type="pdb", content=BACKBONE_PDB.encode())
        tool = ProteinMPNNTool()
        tool._storage = storage
        tool._get_api_key = lambda: "test-key"

        result = await tool.execute({"pdb_ref": backbone, **arguments})

        assert result.content_type == ContentType.ERROR
        assert expected in result.data

    async def test_many_designs_do_not_all_reach_the_message(
        self, storage, patch_httpx
    ):
        # num_seq_per_target goes to 100; listing every score would flood context.
        records = "".join(
            f">T=0.1, sample={i}, score=1.{i:04d}\nMVE\n" for i in range(1, 41)
        )
        result, _, _ = await self._run(
            storage,
            patch_httpx,
            {},
            payload={"mfasta": records, "scores": []},
        )

        assert result.metadata["num_sequences"] == 40
        assert result.metadata["message"].count("- design ") == 10
        assert "30 more" in result.metadata["message"]
