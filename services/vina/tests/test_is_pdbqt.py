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

"""Tests for is_pdbqt() receptor-format detection.

The docking pipeline skips receptor preparation when the input already looks
like a PDBQT. The detector used to rely on the atom-type token in columns
77-79, but a bare element symbol (C/N/O/S) lives in the same columns of a raw
PDB and is also a valid Vina atom type. A biological-assembly PDB -- wrapped in
MODEL/ENDMDL and carrying COMPND headers -- was therefore misread as PDBQT,
preparation was skipped, and Vina rejected the raw multi-model receptor
("Unexpected multi-MODEL tag" / "inappropriate tag ... COMPND"). These tests
pin the fix: raw-PDB-only records force detection to "not PDBQT" (prepare it).
"""

from app.preparation import is_pdbqt


def _atom(charge=None, element="N"):
    """Build one ATOM line with the atom type/element in columns 77-79."""
    line = "ATOM      1  N   LEU A   1      11.104  13.207  10.567  1.00  0.00".ljust(66)
    line += (f"{charge:>10.3f}" if charge is not None else " " * 10)
    line += f" {element:<2}"
    return line


class TestIsPdbqtPositives:
    def test_meeko_torsions_header_marks_pdbqt(self):
        content = "REMARK  4 active torsions:\n" + _atom(charge=-0.348)
        assert is_pdbqt(content) is True

    def test_atom_type_columns_alone_mark_pdbqt(self):
        # No raw-PDB records, atom types present -> prepared PDBQT.
        content = "\n".join(_atom(charge=-0.1, element=e) for e in ["N", "C", "O", "C"])
        assert is_pdbqt(content) is True


class TestIsPdbqtNegatives:
    def test_biological_assembly_with_compnd_and_models_is_not_pdbqt(self):
        content = (
            "HEADER    BIOTIN BINDING PROTEIN\n"
            "COMPND    STREPTAVIDIN\n"
            "CRYST1    1.000    1.000    1.000  90.00  90.00  90.00 P 1\n"
            "MODEL        1\n"
            + "\n".join(_atom(element=e) for e in ["N", "C", "C", "O"]) + "\n"
            "ENDMDL\n"
            "MODEL        2\n"
            + "\n".join(_atom(element=e) for e in ["N", "C", "C", "O"]) + "\n"
            "ENDMDL\n"
        )
        assert is_pdbqt(content) is False

    def test_a_single_raw_header_forces_preparation(self):
        # Even without MODEL wrappers, a COMPND record means raw PDB.
        content = "COMPND    SOME PROTEIN\n" + "\n".join(
            _atom(element=e) for e in ["C", "N", "O", "C"]
        )
        assert is_pdbqt(content) is False
