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

"""Tests for the `safe` module's exposure in the pyexec sandbox.

`import safe as sf` has to work, because that is what SAFE-based skills write.
But `safe.converter` carries a module-level `dm` (datamol) whose fsspec-backed
IO -- `dm.read_csv()`, `dm.to_sdf()` -- accepts URLs and filesystem paths. That
is the surface `open(` and `socket.` are already blocked for, arriving as a
transitive re-export, so both ways of reaching it are closed and tested here.
"""

import pytest

from app.sandbox import execute_code

pytest.importorskip("safe", reason="safe-mol lives in the RDKit base image")

IBUPROFEN = "CC(Cc1ccc(cc1)C(C(=O)O)C)C"


class TestSafeIsUsable:
    def test_import_safe_is_allowed(self):
        result = execute_code("import safe as sf\nresult = callable(sf.encode)")
        assert result["success"] is True, result["error"]
        assert result["result"] is True

    def test_encodes_and_decodes(self):
        result = execute_code(
            "import safe as sf\n"
            f'encoded = sf.encode("{IBUPROFEN}")\n'
            "result = (len(encoded.split(\".\")), sf.decode(encoded, canonical=True))"
        )
        assert result["success"] is True, result["error"]
        fragments, decoded = result["result"]
        assert fragments > 1
        assert decoded

    def test_the_fragmentation_error_is_catchable(self):
        # Skills are told to fall back to the raw SMILES on this error, which
        # requires the exception class to be reachable from the sandbox.
        result = execute_code(
            "import safe as sf\n"
            "try:\n"
            '    out = sf.encode("c1ccccc1")\n'
            "except sf.SAFEFragmentationError:\n"
            '    out = "fell back"\n'
            "result = out"
        )
        assert result["success"] is True, result["error"]
        assert result["result"] == "fell back"


class TestDatamolIsNotReachable:
    def test_importing_datamol_directly_is_blocked(self):
        result = execute_code("import datamol as dm\nresult = 1")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_importing_dm_from_safe_converter_is_blocked(self):
        # The re-export path. RestrictedPython does not guard IMPORT_FROM, so
        # this has to be caught in the import hook rather than by _getattr_.
        result = execute_code("from safe.converter import dm\nresult = 1")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_traversing_to_dm_through_safe_is_blocked(self):
        result = execute_code("import safe as sf\nresult = sf.converter.dm")
        assert result["success"] is False
        assert "not allowed" in result["error"]

    def test_the_error_names_the_supported_alternative(self):
        # A blocked call the model cannot recover from is a dead end; point it at
        # rdkit.Chem and the aevom client instead.
        result = execute_code("import safe as sf\nresult = sf.converter.dm")
        assert "rdkit.Chem" in result["error"]

    def test_a_users_own_dm_attribute_still_works(self):
        # The block is scoped to modules; shadowing every attribute named `dm`
        # would break ordinary code for no security gain.
        result = execute_code(
            "class Holder:\n"
            "    dm = 42\n"
            "result = Holder().dm"
        )
        assert result["success"] is True, result["error"]
        assert result["result"] == 42
