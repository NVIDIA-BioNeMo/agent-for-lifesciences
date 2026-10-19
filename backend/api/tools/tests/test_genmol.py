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

"""Tests for the GenMol builtin tool.

Follows the fixture pattern established in ``test_protein_design_nims.py``: stub
``_get_api_key`` and intercept only the wire, so request construction, response
parsing and the storage writes stay under test.

The three behaviours worth guarding are all silent failures if they regress. A
SAFE input with no mask makes GenMol echo the input back rather than erroring.
``temperature`` and ``noise`` sent as numbers are rejected by the endpoint. And
de novo generation has to work with no prior artifact at all, or every "invent
some molecules" request needs a python_execute round trip first.
"""

import json

import httpx
import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.base import ContentType
from api.tools.genmol import (
    DEFAULT_MASK_MAX,
    DEFAULT_MASK_MIN,
    GenMolTool,
)

# A masked scaffold as smiles_to_safe would produce it: the pyrrolidinone from
# the upstream examples with its terminal fragment replaced by a mask.
MASKED_SAFE = "C1CC(=O)NC1.[*{10-15}]"

MOLECULES = [
    {"smiles": "CCOc1ccccc1", "score": 0.512},
    {"smiles": "OC[C@@H]1O[CH][C@@H](O)[C@H]1O", "score": 0.844},
    {"smiles": "CC(=O)Nc1ccccc1", "score": 0.731},
]


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


def json_response(payload: dict, status: int = 200):
    """A transport handler returning a fixed JSON payload."""

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


def make_tool(storage):
    tool = GenMolTool()
    tool._storage = storage
    tool._get_api_key = lambda: "test-key"
    return tool


@pytest.mark.asyncio
class TestGenMol:
    async def _run(self, storage, patch_httpx, arguments, payload=None):
        payload = payload or {"status": "success", "molecules": MOLECULES}
        handler = json_response(payload)
        patch_httpx(handler)
        result = await make_tool(storage).execute(arguments)
        return result, handler

    async def test_de_novo_builds_the_mask_itself(self, storage, patch_httpx):
        # With no safe_ref there is no artifact to read, so the tool has to
        # construct the bare mask or de novo generation needs a python_execute
        # round trip before every call.
        result, handler = await self._run(storage, patch_httpx, {})

        assert result.content_type == ContentType.DATA_REFERENCE
        sent = json.loads(handler.request.content)
        assert sent["smiles"] == f"[*{{{DEFAULT_MASK_MIN}-{DEFAULT_MASK_MAX}}}]"
        assert result.metadata["mode"] == "de_novo"

    async def test_de_novo_mask_bounds_are_honoured(self, storage, patch_httpx):
        _, handler = await self._run(
            storage, patch_httpx, {"mask_min": 5, "mask_max": 9}
        )

        assert json.loads(handler.request.content)["smiles"] == "[*{5-9}]"

    async def test_an_inverted_mask_range_is_rejected(self, storage):
        result = await make_tool(storage).execute({"mask_min": 30, "mask_max": 10})

        assert result.content_type == ContentType.ERROR
        assert "cannot exceed mask_max" in result.data

    async def test_the_safe_string_is_sent_in_the_smiles_field(
        self, storage, patch_httpx
    ):
        # The endpoint's field is named `smiles` but requires SAFE. Sending a
        # plain SMILES is accepted and quietly degrades the generation.
        safe_ref = await storage.store(data_type="safe", content=MASKED_SAFE.encode())
        _, handler = await self._run(storage, patch_httpx, {"safe_ref": safe_ref})

        assert json.loads(handler.request.content)["smiles"] == MASKED_SAFE

    async def test_a_maskless_safe_string_is_rejected_before_the_call(
        self, storage, patch_httpx
    ):
        # GenMol answers a maskless input by returning that same molecule, which
        # is indistinguishable from a successful generation downstream.
        handler = json_response({"status": "success", "molecules": MOLECULES})
        patch_httpx(handler)
        safe_ref = await storage.store(
            data_type="safe", content=b"C1CC(=O)NC1.c1ccccc1"
        )

        result = await make_tool(storage).execute({"safe_ref": safe_ref})

        assert result.content_type == ContentType.ERROR
        assert "no '[*{min-max}]' generation mask" in result.data
        assert "smiles_to_safe" in result.data
        assert handler.request is None

    async def test_temperature_and_noise_are_sent_as_strings(
        self, storage, patch_httpx
    ):
        # An API quirk: numbers are rejected. The schema takes numbers so the
        # model never has to know.
        _, handler = await self._run(
            storage, patch_httpx, {"temperature": 1.5, "noise": 0.5}
        )

        sent = json.loads(handler.request.content)
        assert sent["temperature"] == "1.5"
        assert sent["noise"] == "0.5"

    async def test_the_poll_header_is_set(self, storage, patch_httpx):
        _, handler = await self._run(storage, patch_httpx, {})

        assert handler.request.headers["NVCF-POLL-SECONDS"] == "120"
        assert handler.request.headers["Authorization"] == "Bearer test-key"

    async def test_the_full_set_is_stored_as_a_ranked_csv(self, storage, patch_httpx):
        result, _ = await self._run(storage, patch_httpx, {})

        assert result.data.startswith("csv_")
        rows = (await storage.retrieve(result.data)).decode().strip().splitlines()
        assert rows[0] == "smiles,score"
        # Best score first, not the order the endpoint returned them in.
        assert rows[1].startswith("OC[C@@H]1O[CH][C@@H](O)[C@H]1O,0.8440")
        assert len(rows) == 1 + len(MOLECULES)
        assert result.metadata["best_score"] == 0.844

    async def test_the_top_molecules_get_their_own_smiles_references(
        self, storage, patch_httpx
    ):
        # Downstream tools take one reference each, so the candidates worth
        # docking have to be individually addressable.
        result, _ = await self._run(storage, patch_httpx, {"store_top_n": 2})

        outputs = result.metadata["outputs"]
        assert set(outputs) == {"molecules_csv", "rank_1", "rank_2"}
        assert outputs["rank_1"].startswith("smiles_")
        best = (await storage.retrieve(outputs["rank_1"])).decode()
        assert best == "OC[C@@H]1O[CH][C@@H](O)[C@H]1O"
        assert len(result.metadata["all_references"]) == 3
        assert "generate_3d_coordinates" in result.metadata["message"]

    async def test_store_top_n_zero_stores_only_the_csv(self, storage, patch_httpx):
        result, _ = await self._run(storage, patch_httpx, {"store_top_n": 0})

        assert set(result.metadata["outputs"]) == {"molecules_csv"}
        assert "aevom.store_smiles()" in result.metadata["message"]

    async def test_more_molecules_than_stored_refs_stay_in_the_csv(
        self, storage, patch_httpx
    ):
        # num_molecules goes to 1000; storing a reference per molecule would
        # flood both the artifact list and the context.
        many = [{"smiles": f"C{'C' * i}O", "score": i / 100} for i in range(1, 41)]
        result, _ = await self._run(
            storage,
            patch_httpx,
            {"num_molecules": 40},
            payload={"status": "success", "molecules": many},
        )

        assert result.metadata["num_returned"] == 40
        assert len(result.metadata["all_references"]) == 1 + 5

    async def test_a_short_return_is_explained_rather_than_failed(
        self, storage, patch_httpx
    ):
        # Post-generation filtering routinely yields fewer than requested. A
        # model told nothing will report it as a partial failure.
        result, _ = await self._run(storage, patch_httpx, {"num_molecules": 100})

        assert result.content_type == ContentType.DATA_REFERENCE
        assert "Fewer came back than requested" in result.metadata["message"]
        assert result.metadata["num_requested"] == 100

    async def test_malformed_entries_are_dropped(self, storage, patch_httpx):
        result, _ = await self._run(
            storage,
            patch_httpx,
            {},
            payload={
                "status": "success",
                "molecules": [
                    {"smiles": "CCO", "score": 0.4},
                    {"smiles": "", "score": 0.9},
                    {"smiles": "CCC"},
                    "not-a-dict",
                ],
            },
        )

        assert result.metadata["num_returned"] == 1

    async def test_a_failed_status_is_surfaced_with_its_reason(
        self, storage, patch_httpx
    ):
        # A 200 carrying status: "failed" is the endpoint's normal way of
        # reporting an invalid SAFE string.
        result, _ = await self._run(
            storage,
            patch_httpx,
            {},
            payload={"status": "failed", "error": "invalid SAFE string"},
        )

        assert result.content_type == ContentType.ERROR
        assert "invalid SAFE string" in result.data

    async def test_an_empty_molecule_list_is_a_clean_error(self, storage, patch_httpx):
        result, _ = await self._run(
            storage,
            patch_httpx,
            {},
            payload={"status": "success", "molecules": []},
        )

        assert result.content_type == ContentType.ERROR
        assert "no usable molecules" in result.data

    async def test_a_422_surfaces_the_endpoint_complaint(self, storage, patch_httpx):
        patch_httpx(
            json_response(
                {"error": "Validation Errors: temperature: not a valid string"},
                status=422,
            )
        )

        result = await make_tool(storage).execute({})

        assert result.content_type == ContentType.ERROR
        assert "not a valid string" in result.data
        assert "422" in result.data

    async def test_logp_ranking_carries_its_caveat(self, storage, patch_httpx):
        result, _ = await self._run(storage, patch_httpx, {"scoring": "LogP"})

        assert result.metadata["scoring"] == "LogP"
        assert "lipophilicity, not quality" in result.metadata["message"]

    async def test_scaffold_mode_warns_about_containment(self, storage, patch_httpx):
        safe_ref = await storage.store(data_type="safe", content=MASKED_SAFE.encode())
        result, _ = await self._run(storage, patch_httpx, {"safe_ref": safe_ref})

        assert result.metadata["mode"] == "scaffold_decoration"
        assert result.metadata["safe_ref"] == safe_ref
        assert "HasSubstructMatch" in result.metadata["message"]

    async def test_mask_bounds_alongside_a_scaffold_are_reported_as_ignored(
        self, storage, patch_httpx
    ):
        # The scaffold carries its own mask, so bounds passed here do nothing.
        # Silently dropping them leaves the model believing it sized the result.
        safe_ref = await storage.store(data_type="safe", content=MASKED_SAFE.encode())
        result, handler = await self._run(
            storage, patch_httpx, {"safe_ref": safe_ref, "mask_min": 3, "mask_max": 4}
        )

        assert json.loads(handler.request.content)["smiles"] == MASKED_SAFE
        assert "mask_min/mask_max were ignored" in result.metadata["message"]

    async def test_an_unknown_scoring_mode_is_rejected(self, storage):
        result = await make_tool(storage).execute({"scoring": "SA"})

        assert result.content_type == ContentType.ERROR
        assert "Unknown scoring 'SA'" in result.data

    async def test_an_unknown_reference_is_a_clean_error(self, storage):
        result = await make_tool(storage).execute({"safe_ref": "safe_missing1"})

        assert result.content_type == ContentType.ERROR

    @pytest.mark.parametrize(
        "arguments,expected",
        [
            ({"num_molecules": 0}, "num_molecules"),
            ({"num_molecules": 1001}, "num_molecules"),
            ({"temperature": 0}, "temperature"),
            ({"temperature": "hot"}, "temperature"),
            ({"noise": 5}, "noise"),
            ({"step_size": 11}, "step_size"),
            ({"store_top_n": 100}, "store_top_n"),
        ],
    )
    async def test_out_of_range_settings_are_rejected(
        self, storage, arguments, expected
    ):
        result = await make_tool(storage).execute(arguments)

        assert result.content_type == ContentType.ERROR
        assert expected in result.data

    async def test_the_schema_marks_safe_ref_as_a_reference_and_optional(self):
        # A required safe_ref would break de novo, and a plain string parameter
        # would invite the model to paste a molecule into it.
        schema = GenMolTool().get_schema()["function"]

        assert schema["parameters"]["required"] == []
        assert schema["parameters"]["properties"]["safe_ref"][
            "x-aevom-reference"
        ] == {"data_type": "safe"}
        assert "genmol-nim" in schema["description"]
