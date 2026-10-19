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

"""Tests for reading a stored CSV back inside the sandbox.

admet_predict tells the caller to "rank or filter the rest in python_execute".
The sandbox has neither `csv` nor `pandas` on its import allowlist, so until now
that meant splitting on commas by hand -- which breaks on the first quoted SMILES
containing one, and leaves a carriage return welded to the last field of every
row when the file was written with the csv module's default CRLF.

Measured over 120 benchmark runs: 64 tried `import pandas` or `import csv` and
were refused; 20 went on to raise a KeyError or TypeError on a column.
"""

import pytest

from app.aevom_client import AevomClient, AevomError

CSV = (
    "smiles,CYP3A4_Veith,Clearance_Hepatocyte_AZ\n"
    "CCO,0.12,15.4\n"
    "c1ccccc1,0.88,3.2\n"
)


class _Stub(AevomClient):
    def __init__(self, payload):
        super().__init__(session_token="tok", backend_url="http://backend:8000")
        self._payload = payload

    def load(self, reference):  # noqa: D102 - test double
        return self._payload


class TestReadingItBack:
    def test_rows_are_keyed_by_the_file_s_own_headers(self):
        rows = _Stub(CSV).load_table("csv_x")

        assert [r["smiles"] for r in rows] == ["CCO", "c1ccccc1"]
        assert set(rows[0]) == {"smiles", "CYP3A4_Veith", "Clearance_Hepatocyte_AZ"}

    def test_numbers_come_back_as_numbers(self):
        rows = _Stub(CSV).load_table("csv_x")

        # The point: a threshold comparison must work without a cast.
        assert [r for r in rows if r["CYP3A4_Veith"] < 0.5] == [rows[0]]
        assert rows[1]["Clearance_Hepatocyte_AZ"] == pytest.approx(3.2)

    def test_a_smiles_with_a_comma_survives(self):
        # The hand-rolled split this replaces gets this wrong every time.
        payload = 'smiles,logp\n"CC(=O)N(C)C,x",1.5\n'
        rows = _Stub(payload).load_table("csv_x")

        assert rows[0]["smiles"] == "CC(=O)N(C)C,x"
        assert rows[0]["logp"] == pytest.approx(1.5)

    def test_crlf_does_not_stick_to_the_last_column(self):
        rows = _Stub("smiles,logp\r\nCCO,1.5\r\n").load_table("csv_x")

        assert rows[0]["logp"] == pytest.approx(1.5)

    def test_an_empty_table_is_empty_not_an_error(self):
        assert _Stub("smiles,logp\n").load_table("csv_x") == []

    def test_bytes_are_decoded(self):
        rows = _Stub(CSV.encode("utf-8")).load_table("csv_x")

        assert rows[0]["smiles"] == "CCO"

    def test_a_non_text_reference_says_so(self):
        with pytest.raises(AevomError) as caught:
            _Stub({"not": "csv"}).load_table("array_x")

        assert "not text" in str(caught.value)
