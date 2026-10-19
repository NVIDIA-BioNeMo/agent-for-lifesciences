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

"""Tests for the central data type vocabulary.

Every artifact reference is `{data_type}_{hex}`, and three subsystems read this
table: the MCP detector, artifact export encoding, and the sandbox MIME type.
The tests that matter are the invariants those subsystems depend on -- chiefly
that `is_text` is right, since getting it wrong means either a
UnicodeDecodeError on export or base64 shown to the user as if it were a
structure file.
"""

import pytest

from api.storage.datatypes import (
    DATA_TYPE_INFO,
    TEXT_TYPES,
    DataType,
    content_type_for,
    friendly_name,
    get_info,
    heavy_type_values,
    is_text_type,
    resolve,
    text_type_values,
)


class TestCompleteness:
    def test_every_enum_member_has_metadata(self):
        # A type without metadata silently gets treated as binary with no MIME.
        assert set(DATA_TYPE_INFO) == set(DataType)

    def test_metadata_keys_match_their_declared_type(self):
        for data_type, info in DATA_TYPE_INFO.items():
            assert info.data_type is data_type

    def test_every_type_has_a_friendly_name_and_mime(self):
        for info in DATA_TYPE_INFO.values():
            assert info.friendly.strip()
            assert "/" in info.mime

    def test_the_first_class_molecular_types_are_all_present(self):
        for value in (
            "smiles",
            "safe",
            "sdf",
            "pdb",
            "mmcif",
            "sequence",
            "dna",
            "rna",
            "msa",
            "array",
        ):
            assert resolve(value) is not None


class TestResolve:
    @pytest.mark.parametrize("raw", ["pdb", "PDB", "  Pdb  "])
    def test_is_case_and_whitespace_insensitive(self, raw):
        assert resolve(raw) is DataType.PDB

    def test_passes_an_enum_member_through(self):
        assert resolve(DataType.SMILES) is DataType.SMILES

    def test_resolves_an_alias(self):
        assert resolve("pdbx") is DataType.MMCIF

    @pytest.mark.parametrize("raw", ["", None, "protein_blob", "pdb_id"])
    def test_unknown_values_resolve_to_none(self, raw):
        assert resolve(raw) is None

    def test_get_info_returns_none_for_unknown(self):
        assert get_info("protein_blob") is None


class TestTextVersusBinary:
    def test_the_binary_types_are_exactly_array_and_png(self):
        binary = {dt.value for dt, info in DATA_TYPE_INFO.items() if not info.is_text}
        assert binary == {"array", "png"}

    @pytest.mark.parametrize(
        "value",
        ["pdb", "sdf", "smiles", "safe", "mmcif", "sequence", "dna", "msa", "json"],
    )
    def test_molecular_and_structured_types_are_text(self, value):
        assert is_text_type(value)

    def test_an_unknown_type_is_treated_as_binary(self):
        # Failing closed here turns a mystery type into base64 rather than a
        # UnicodeDecodeError partway through an export.
        assert is_text_type("protein_blob") is False

    def test_text_type_values_matches_the_table(self):
        assert text_type_values() == {
            dt.value for dt, info in DATA_TYPE_INFO.items() if info.is_text
        }

    def test_the_module_level_constant_agrees(self):
        assert TEXT_TYPES == text_type_values()


class TestMimeTypes:
    @pytest.mark.parametrize(
        "value,expected",
        [
            ("pdb", "chemical/x-pdb"),
            ("sdf", "chemical/x-mdl-sdfile"),
            ("mmcif", "chemical/x-mmcif"),
            ("smiles", "text/plain"),
            ("safe", "text/plain"),
            ("json", "application/json"),
            ("array", "application/octet-stream"),
            # GET /api/images/{reference} serves this verbatim as the response
            # media type, so a wrong value here shows the user a broken image.
            ("png", "image/png"),
        ],
    )
    def test_known_types_get_their_mime(self, value, expected):
        assert content_type_for(value) == expected

    def test_unknown_types_fall_back_to_octet_stream(self):
        assert content_type_for("protein_blob") == "application/octet-stream"


class TestFriendlyNames:
    @pytest.mark.parametrize(
        "value,fragment",
        [
            ("smiles", "SMILES"),
            ("safe", "SAFE"),
            ("pdb", "PDB"),
            ("mmcif", "mmCIF"),
            ("sequence", "sequence"),
            ("array", "array"),
        ],
    )
    def test_names_describe_the_format(self, value, fragment):
        assert fragment in friendly_name(value)

    def test_unknown_types_echo_the_raw_string(self):
        assert friendly_name("protein_blob") == "protein_blob"

    def test_safe_names_its_consumers(self):
        # This string is not display-only: _data_type_help() in skills/extract.py
        # puts it in the contract-extraction prompt, and it is how the model
        # learns to pick `safe` over `smiles` for a GenMol-style field. Without
        # naming a consumer, "fragment notation" reads as a SMILES variant.
        name = friendly_name("safe")
        assert "GenMol" in name
        assert "fragment" in name.lower()


class TestHeavyTypes:
    def test_molecular_types_are_heavy(self):
        heavy = heavy_type_values()
        for value in (
            "smiles",
            "safe",
            "sdf",
            "pdb",
            "mmcif",
            "sequence",
            "dna",
            "rna",
            "msa",
        ):
            assert value in heavy

    def test_storage_internal_types_are_not_heavy(self):
        heavy = heavy_type_values()
        for value in ("skill_file", "json", "array", "selection", "png"):
            assert value not in heavy
