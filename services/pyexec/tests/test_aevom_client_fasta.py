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

"""aevom.load() must decode a ``fasta`` artifact to text, not raw bytes.

ProteinMPNN stores its designed sequences as a ``fasta`` reference. ``load()``
only decoded a fixed list of text types and ``fasta`` was missing, so the
payload came back as ``bytes`` -- and the first string operation on it
(``.splitlines()`` against a str separator, an f-string, ...) raised
``TypeError: a bytes-like object is required, not 'str'``. Seen live when the
model tried to read ProteinMPNN designs after a binder-design run.
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


def test_load_decodes_fasta_as_text(client):
    client._store["fasta_abcd1234"] = b">design_0 score=2.7\nMKTAYIAKQR\n"
    out = client.load("fasta_abcd1234")
    assert isinstance(out, str)
    assert out.startswith(">design_0")


def test_fasta_is_str_iterable_without_bytes_errors(client):
    client._store["fasta_abcd1234"] = b">a\nMKT\n>b\nGGS\n"
    fasta = client.load("fasta_abcd1234")
    seqs = [line for line in fasta.splitlines() if not line.startswith(">")]
    assert seqs == ["MKT", "GGS"]
