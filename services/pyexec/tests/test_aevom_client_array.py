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

"""Tests for `array` artifact handling in the sandbox's aevom client.

Numeric outputs a tool cannot summarize -- PAE matrices, per-residue pLDDT --
are stored as `array_*` artifacts precisely so they can be analyzed here with
numpy instead of being pushed through the model's context as text. The backend
writes them as compressed .npz; these tests pin both halves of that round trip
and the refusals that keep non-numeric data out.
"""

import base64
import io
import json

import numpy as np
import pytest

from app.aevom_client import AevomClient, AevomError, StorageError


class FakeTransport:
    """Stands in for the backend's /api/sandbox endpoints."""

    def __init__(self, artifacts=None):
        self.artifacts = dict(artifacts or {})
        self.stored: list[dict] = []
        self._counter = 0

    def fetch(self, reference):
        return self.artifacts[reference]

    def store(self, payload):
        self._counter += 1
        reference = f"{payload['data_type']}_stored{self._counter}"
        self.stored.append(payload)
        if "content_base64" in payload:
            content = base64.b64decode(payload["content_base64"])
        else:
            content = payload["content"].encode("utf-8")
        self.artifacts[reference] = content
        return {"reference": reference}


@pytest.fixture
def client(monkeypatch):
    """An AevomClient whose HTTP calls are redirected to a fake transport."""
    transport = FakeTransport()
    instance = AevomClient(session_token="tok", backend_url="http://backend:8000")

    monkeypatch.setattr(
        AevomClient, "_fetch_data", lambda self, ref: transport.fetch(ref)
    )

    def _store(self, data_type, content, metadata=None, description=None):
        payload = {"data_type": data_type, "metadata": metadata or {}}
        if isinstance(content, (bytes, bytearray)):
            payload["content_base64"] = base64.b64encode(bytes(content)).decode("ascii")
        else:
            payload["content"] = content
        if description:
            payload["description"] = description
        return transport.store(payload)["reference"]

    monkeypatch.setattr(AevomClient, "store", _store)
    instance.transport = transport
    return instance


def npz_bytes(array, key="data"):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: np.asarray(array)})
    return buffer.getvalue()


class TestLoadArray:
    def test_round_trips_a_matrix(self, client):
        matrix = np.arange(24, dtype=np.float32).reshape(4, 6)
        client.transport.artifacts["array_pae"] = npz_bytes(matrix)

        loaded = client.load_array("array_pae")
        assert loaded.shape == (4, 6)
        assert loaded.dtype == np.float32
        assert np.allclose(loaded, matrix)

    def test_generic_load_detects_the_array_prefix(self, client):
        client.transport.artifacts["array_plddt"] = npz_bytes([0.9] * 32)

        loaded = client.load("array_plddt")
        assert isinstance(loaded, np.ndarray)
        assert loaded.shape == (32,)

    def test_falls_back_to_the_first_key(self, client):
        # Archives written elsewhere may not use the "data" key.
        client.transport.artifacts["array_other"] = npz_bytes([1, 2, 3], key="scores")
        assert np.allclose(client.load_array("array_other"), [1, 2, 3])

    def test_preserves_integer_dtype(self, client):
        client.transport.artifacts["array_counts"] = npz_bytes(
            np.array([1, 2, 3], dtype=np.int64)
        )
        assert client.load_array("array_counts").dtype == np.int64

    def test_a_corrupt_artifact_raises_a_clear_error(self, client):
        client.transport.artifacts["array_bad"] = b"not an npz file"
        with pytest.raises(AevomError, match="Failed to decode array"):
            client.load_array("array_bad")

    def test_an_empty_archive_raises(self, client):
        buffer = io.BytesIO()
        np.savez(buffer)
        client.transport.artifacts["array_empty"] = buffer.getvalue()
        with pytest.raises(AevomError):
            client.load_array("array_empty")

    def test_a_pickled_payload_is_refused(self, client):
        # allow_pickle=False is a security boundary: an object array would
        # otherwise execute arbitrary code on load.
        buffer = io.BytesIO()
        np.savez(buffer, data=np.array([{"x": 1}], dtype=object))
        client.transport.artifacts["array_pickled"] = buffer.getvalue()
        with pytest.raises(AevomError):
            client.load_array("array_pickled")


class TestStoreArray:
    def test_returns_an_array_reference(self, client):
        reference = client.store_array(np.zeros((3, 3)))
        assert reference.startswith("array_")

    def test_sends_binary_content_base64_encoded(self, client):
        # The store endpoint is JSON, so npz bytes cannot go through `content`.
        client.store_array(np.zeros((2, 2)))
        payload = client.transport.stored[-1]
        assert "content_base64" in payload
        assert "content" not in payload

    def test_records_shape_and_dtype_in_metadata(self, client):
        client.store_array(np.zeros((5, 7), dtype=np.float64))
        metadata = client.transport.stored[-1]["metadata"]
        assert metadata["shape"] == [5, 7]
        assert metadata["dtype"] == "float64"

    def test_accepts_a_nested_list(self, client):
        reference = client.store_array([[1, 2], [3, 4]])
        assert np.allclose(client.load_array(reference), [[1, 2], [3, 4]])

    def test_round_trips_through_the_store(self, client):
        original = np.random.default_rng(0).normal(size=(8, 8))
        reference = client.store_array(original, description="Interface PAE")
        assert np.allclose(client.load_array(reference), original)

    def test_user_metadata_is_preserved_alongside_shape(self, client):
        client.store_array([1, 2, 3], metadata={"chain": "A"})
        metadata = client.transport.stored[-1]["metadata"]
        assert metadata["chain"] == "A"
        assert metadata["shape"] == [3]

    @pytest.mark.parametrize(
        "value", [["a", "b"], [{"x": 1}], np.array(["x", "y"])]
    )
    def test_non_numeric_data_is_refused(self, client, value):
        with pytest.raises(StorageError, match="numeric"):
            client.store_array(value)

    def test_the_error_points_at_the_json_alternative(self, client):
        with pytest.raises(StorageError, match="store\\('json'"):
            client.store_array(["a"])


class TestTextTypesAreUnaffected:
    def test_a_pdb_reference_still_decodes_as_text(self, client):
        client.transport.artifacts["pdb_x"] = b"ATOM      1  N   MET A   1\n"
        assert client.load("pdb_x").startswith("ATOM")

    def test_a_json_reference_still_parses(self, client):
        client.transport.artifacts["json_x"] = json.dumps({"a": 1}).encode()
        assert client.load("json_x") == {"a": 1}

    def test_a_csv_reference_decodes_as_text(self, client):
        # genmol_generate_molecules stores its ranked table as csv, so a load()
        # that fell through to raw bytes would hand the model a bytes repr.
        client.transport.artifacts["csv_x"] = b"smiles,score\nCCO,0.4000\n"
        assert client.load("csv_x").splitlines()[0] == "smiles,score"

    def test_text_content_is_not_base64_encoded(self, client):
        client.store("smiles", "CCO")
        payload = client.transport.stored[-1]
        assert payload["content"] == "CCO"
        assert "content_base64" not in payload
