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

"""Argument coercion in the JSON-Schema-to-Pydantic bridge.

Tool schemas are hand-written in OpenAI format and turned into Pydantic models to
give LangChain a signature. Models do not always send the JSON type the schema
declares, and the resulting validation error names Pydantic rather than the fix,
so the model is left guessing at what to change.

Two directions are tolerated, and only those two: an array or object sent as a
JSON string literal, and a number sent where a string belongs. Anything else
still fails, because a genuinely wrong argument should be reported as one.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from api.graph.tools import _json_schema_to_pydantic, _make_tool_wrapper
from api.tools.registry import ToolRegistry, _register_default_tools

_PROPERTIES = {
    "gene_name": {"type": "string", "description": "gene symbol"},
    "organism_id": {"type": "string", "description": "NCBI taxonomy id"},
    "chains": {"type": "array", "description": "chain ids"},
    "options": {"type": "object", "description": "extra settings"},
    "steps": {"type": "integer", "description": "iterations"},
    "soluble": {"type": "boolean", "description": "use the soluble model"},
}


@pytest.fixture
def model():
    return _json_schema_to_pydantic("demo", _PROPERTIES, ["gene_name"])


class TestNumbersWhereStringsAreDeclared:
    """`organism_id: 9606` is the natural thing for a model to emit.

    An eval baseline burned three consecutive fetch_protein_sequence calls
    discovering the taxonomy id had to be quoted, which is three calls of an
    error budget spent on a type annotation.
    """

    def test_integer_becomes_a_string(self, model):
        assert model(gene_name="PGK1", organism_id=9606).organism_id == "9606"

    def test_float_becomes_a_string(self, model):
        assert model(gene_name="PGK1", organism_id=1.5).organism_id == "1.5"

    def test_a_string_is_untouched(self, model):
        assert model(gene_name="PGK1", organism_id="9606").organism_id == "9606"

    def test_bool_is_not_stringified(self, model):
        """bool subclasses int, and "True" is not what a string field wants."""
        with pytest.raises(ValidationError):
            model(gene_name="PGK1", organism_id=True)

    def test_a_list_still_fails(self, model):
        """Coercing this would hand the tool the literal "['A']"."""
        with pytest.raises(ValidationError):
            model(gene_name="PGK1", organism_id=["9606"])

    def test_required_string_is_coerced_too(self, model):
        assert model(gene_name=42).gene_name == "42"


class TestJsonStringsWhereStructuresAreDeclared:
    def test_array_sent_as_a_string_literal(self, model):
        assert model(gene_name="PGK1", chains='["A", "B"]').chains == ["A", "B"]

    def test_object_sent_as_a_string_literal(self, model):
        assert model(gene_name="PGK1", options='{"a": 1}').options == {"a": 1}

    def test_a_native_list_is_untouched(self, model):
        assert model(gene_name="PGK1", chains=["A"]).chains == ["A"]


class TestOtherTypesAreUnaffected:
    def test_integer_field(self, model):
        assert model(gene_name="PGK1", steps=50).steps == 50

    def test_boolean_field(self, model):
        assert model(gene_name="PGK1", soluble=True).soluble is True

    def test_optional_fields_default_to_none(self, model):
        parsed = model(gene_name="PGK1")
        assert parsed.organism_id is None
        assert parsed.chains is None


class TestTypeLists:
    """JSON Schema lets `type` be a list, as select_residues' `atoms` does.

    The converter once used the type as a dict key, so a list raised and the
    whole tool was silently left out of every chat.
    """

    @pytest.fixture
    def atoms_model(self):
        return _json_schema_to_pydantic(
            "select",
            {
                "atoms": {
                    "type": ["array", "string"],
                    "items": {"type": "string"},
                    "description": "atom names or a keyword",
                },
                "label": {"type": ["string", "null"], "description": "name"},
            },
            [],
        )

    def test_a_list_is_untouched(self, atoms_model):
        assert atoms_model(atoms=["CA", "OH"]).atoms == ["CA", "OH"]

    def test_a_keyword_stays_a_string(self, atoms_model):
        assert atoms_model(atoms="backbone").atoms == "backbone"

    def test_a_json_string_literal_takes_the_array_branch(self, atoms_model):
        assert atoms_model(atoms='["CA"]').atoms == ["CA"]

    def test_a_number_takes_the_string_branch(self, atoms_model):
        assert atoms_model(atoms=5).atoms == "5"

    def test_a_dict_still_fails(self, atoms_model):
        with pytest.raises(ValidationError):
            atoms_model(atoms={"name": "CA"})

    def test_null_in_the_list_is_a_plain_optional_string(self, atoms_model):
        assert atoms_model().label is None
        assert atoms_model(label=7).label == "7"

    def test_the_model_is_shown_both_types(self, atoms_model):
        prop = atoms_model.model_json_schema()["properties"]["atoms"]
        types = {branch.get("type") for branch in prop["anyOf"]}
        assert {"array", "string"} <= types


class TestEveryRegisteredToolGetsAWrapper:
    """build_langchain_tools logs and skips a tool whose wrapper fails.

    That keeps one bad schema from taking down chat, but it also means a broken
    schema shows up only as a smaller count in the logs. Building each wrapper
    directly makes the failure name the tool.
    """

    @pytest.mark.asyncio
    async def test_no_tool_is_dropped(self):
        registry = ToolRegistry()
        _register_default_tools(registry)
        built = []
        for schema in await registry.get_all_schemas():
            name = schema["function"]["name"]
            built.append(_make_tool_wrapper(name, schema).name)
        assert "select_residues" in built
        assert set(registry.tool_names) <= set(built)
