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

"""Telling an annotated mmCIF from an ordinary one.

Both dialects use the ``.cif`` extension and both start ``data_``, so content
is the only thing that separates them. The distinction earns its keep because
the consumers differ: RFD4-Proteina reads the condition columns, and every
other structure tool ignores them.
"""

from api.artifacts.infer import infer_data_type
from api.storage.annotated_cif import (
    condition_columns,
    describe,
    is_annotated_cif,
    summarize_conditions,
)
from api.storage.datatypes import DataType

PLAIN_CIF = """\
data_1ABC
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
ATOM 1 CA GLY A 1 1.000 1.000 1.000
ATOM 2 CA ALA A 2 2.000 1.000 1.000
#
"""

# Column names and value conventions follow what AtomWorks writes: masks are
# booleans, and integer annotation columns use -1 where they do not apply.
ANNOTATED_CIF = """\
data_DESIGN
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.mask_hotspot_1_atom
_atom_site.mask_coordinate_1_atom
_atom_site.annotation_expsegmin_1_residue
ATOM 1 CA GLY A 1 1.000 1.000 1.000 True  True -1
ATOM 2 CA ALA A 2 2.000 1.000 1.000 False True -1
ATOM 3 CB ALA A 2 3.000 1.000 1.000 True  True 40
#
"""


class TestDetection:
    def test_plain_mmcif_is_not_annotated(self):
        assert is_annotated_cif(PLAIN_CIF) is False

    def test_condition_columns_make_it_annotated(self):
        assert is_annotated_cif(ANNOTATED_CIF) is True

    def test_bytes_are_accepted(self):
        assert is_annotated_cif(ANNOTATED_CIF.encode()) is True

    def test_empty_content_is_not_annotated(self):
        assert is_annotated_cif("") is False

    def test_undecodable_content_is_not_annotated(self):
        assert is_annotated_cif(b"\xff\xfe\x00binary") is False

    def test_top_level_condition_category_counts(self):
        # Vector-valued conditions live in their own category rather than in
        # the atom_site loop.
        text = PLAIN_CIF + "_condition_coordinate_1_atom.val0 1.0\n"
        assert is_annotated_cif(text) is True

    def test_columns_are_reported_by_name(self):
        assert "mask_hotspot_1_atom" in condition_columns(ANNOTATED_CIF)


class TestInference:
    def test_annotated_content_infers_the_annotated_type(self):
        data_type, _ = infer_data_type(None, "x.cif", ANNOTATED_CIF.encode())
        assert data_type == DataType.ANNOTATED_CIF

    def test_plain_content_still_infers_mmcif(self):
        data_type, _ = infer_data_type(None, "x.cif", PLAIN_CIF.encode())
        assert data_type == DataType.MMCIF

    def test_extension_alone_cannot_decide_so_content_wins(self):
        # Both dialects are called .cif, which is why this one extension is
        # the exception to trusting the filename.
        _, source = infer_data_type(None, "x.cif", ANNOTATED_CIF.encode())
        assert source == "content"

    def test_type_is_inferred_without_a_filename(self):
        data_type, _ = infer_data_type(None, None, ANNOTATED_CIF.encode())
        assert data_type == DataType.ANNOTATED_CIF

    def test_a_declared_type_is_still_respected(self):
        # Sniffing must not override a caller who named the type.
        data_type, source = infer_data_type("mmcif", "x.cif", ANNOTATED_CIF.encode())
        assert (data_type, source) == (DataType.MMCIF, "declared")


class TestSummary:
    def test_set_atoms_are_counted_per_condition(self):
        summaries = {s.name: s.set_count for s in summarize_conditions(ANNOTATED_CIF)}
        assert summaries["hotspot"] == 2

    def test_sentinel_values_are_not_counted_as_set(self):
        # -1 marks "does not apply"; counting it would report every atom.
        summaries = {s.name: s.set_count for s in summarize_conditions(ANNOTATED_CIF)}
        assert summaries["expsegmin"] == 1

    def test_description_names_the_conditions_in_use(self):
        text = describe(ANNOTATED_CIF)
        assert "hotspot" in text and "2 atoms" in text

    def test_description_of_a_plain_mmcif_says_so(self):
        assert "No design conditions" in describe(PLAIN_CIF)

    def test_conditions_are_ordered_by_coverage(self):
        # coordinate covers 3 atoms, hotspot 2, so coordinate leads.
        assert describe(ANNOTATED_CIF).index("coordinate") < describe(
            ANNOTATED_CIF
        ).index("hotspot")
