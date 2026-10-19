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

"""aevom.load() must return a ``sequence`` artifact as text.

Every tool that stores a sequence (UniProt lookup, RFD4-Proteina binder design)
writes the bare residue string. ``load()`` and ``load_sequence()`` both ran it
through ``json.loads``, so loading any real sequence raised
``JSONDecodeError: Expecting value: line 1 column 1 (char 0)``. Seen live in a
GLP-1R binder-design session, where the model burned several turns before
falling back to ``load_text()``.
"""

import pytest

from app.aevom_client import AevomClient


@pytest.fixture
def client(monkeypatch):
    inst = AevomClient(session_token="tok", backend_url="http://backend:8000")
    store: dict[str, bytes] = {}
    monkeypatch.setattr(AevomClient, "_fetch_data", lambda self, ref: store[ref])
    inst._store = store  # type: ignore[attr-defined]
    return inst


def test_load_returns_raw_sequence_as_text(client):
    client._store["sequence_59231645"] = b"AAAAELRGVLLVAGLVAVHLAAL"
    assert client.load("sequence_59231645") == "AAAAELRGVLLVAGLVAVHLAAL"


def test_load_sequence_wraps_raw_text(client):
    client._store["sequence_59231645"] = b"MKTAYIAKQR\n"
    assert client.load_sequence("sequence_59231645") == {
        "sequence": "MKTAYIAKQR",
        "length": 10,
    }


def test_load_sequence_still_parses_a_json_payload(client):
    client._store["sequence_abcd1234"] = b'{"sequence": "MKV", "length": 3, "source": "x"}'
    assert client.load_sequence("sequence_abcd1234")["source"] == "x"


def test_design_spec_prefix_is_recognised(client):
    assert client._get_data_type("design_spec_abcd1234") == "design_spec"
