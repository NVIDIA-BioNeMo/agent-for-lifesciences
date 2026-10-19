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

"""Tests for the skill data-type contract models.

Contracts are LLM-generated or hand-edited in the UI, so validation is the only
thing standing between a bad guess and a synthesized tool that calls the wrong
host with the wrong body. These tests pin the parts that carry that weight:
path parsing, name collisions, and the confirmed/host gates.
"""

import pytest

from api.skills.contract import (
    SkillContract,
    SkillContractError,
    SkillEndpoint,
    SkillInputParam,
    SkillOutputField,
    parse_contract,
    split_json_path,
    validate_json_path,
)
from api.storage.datatypes import DataType


class TestJsonPath:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("$.structures[0].structure", "structures[0].structure"),
            (".name", "name"),
            ("name", "name"),
            ("$.a.b.c", "a.b.c"),
            ("  $.results[2]  ", "results[2]"),
            ("$.pae_scores", "pae_scores"),
            ("$.my-field", "my-field"),
        ],
    )
    def test_accepts_dot_and_index_paths(self, raw, expected):
        assert validate_json_path(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "",
            "   ",
            "$.items[*]",
            "$..name",
            "$.items[?(@.x)]",
            "$[0]",
            "$.9field",
            "$.a b",
        ],
    )
    def test_rejects_unsupported_syntax(self, raw):
        with pytest.raises(ValueError):
            validate_json_path(raw)

    def test_split_produces_key_and_index_tokens(self):
        assert split_json_path("$.structures[0].structure") == [
            "structures",
            0,
            "structure",
        ]

    def test_split_handles_consecutive_indices(self):
        assert split_json_path("$.grid[1][2]") == ["grid", 1, 2]


class TestSkillInputParam:
    def test_json_path_defaults_to_name(self):
        param = SkillInputParam(name="sequence", data_type=DataType.SEQUENCE)
        assert param.json_path == "sequence"

    def test_ref_param_name_avoids_stuttering(self):
        # A field already named for its type gets a bare _ref suffix, so the LLM
        # sees `smiles_ref` rather than `smiles_smiles_ref`.
        param = SkillInputParam(name="smiles", data_type=DataType.SMILES)
        assert param.ref_param_name == "smiles_ref"

    def test_ref_param_name_includes_type_otherwise(self):
        param = SkillInputParam(name="target", data_type=DataType.PDB)
        assert param.ref_param_name == "target_pdb_ref"

    def test_empty_name_rejected(self):
        with pytest.raises(ValueError):
            SkillInputParam(name="  ", data_type=DataType.SMILES)

    def test_unknown_data_type_rejected(self):
        with pytest.raises(ValueError):
            SkillInputParam(name="thing", data_type="protein_blob")

    def test_a_genmol_style_safe_param_keeps_its_field_name(self):
        # GenMol's request field is named `smiles` but carries SAFE. The contract
        # has to record both: `name` is what goes on the wire, `data_type` is
        # what the ref must resolve to, and the ref param name splices the type
        # in so the model can see which of the two it is being asked for.
        param = SkillInputParam(name="smiles", data_type="safe")
        assert param.json_path == "smiles"
        assert param.ref_param_name == "smiles_safe_ref"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValueError):
            SkillInputParam(name="x", data_type=DataType.SMILES, sneaky=True)


class TestSkillOutputField:
    def test_json_path_is_required(self):
        with pytest.raises(ValueError):
            SkillOutputField(name="structure", data_type=DataType.PDB)

    def test_normalizes_json_path(self):
        output = SkillOutputField(
            name="structure",
            json_path="$.structures[0].structure",
            data_type=DataType.MMCIF,
        )
        assert output.json_path == "structures[0].structure"


def _endpoint(**overrides) -> SkillEndpoint:
    payload = {
        "tool_suffix": "predict",
        "summary": "Predict a complex structure.",
        "url": "https://api.example.com/predict",
        "inputs": [{"name": "sequence", "data_type": "sequence"}],
        "outputs": [
            {"name": "structure", "json_path": "$.structure", "data_type": "mmcif"}
        ],
    }
    payload.update(overrides)
    return SkillEndpoint.model_validate(payload)


class TestSkillEndpoint:
    def test_tool_suffix_is_normalized(self):
        endpoint = _endpoint(tool_suffix="Predict-Structure")
        assert endpoint.tool_suffix == "predict_structure"

    @pytest.mark.parametrize("suffix", ["", "9predict", "predict structure", "a" * 41])
    def test_invalid_tool_suffix_rejected(self, suffix):
        with pytest.raises(ValueError):
            _endpoint(tool_suffix=suffix)

    def test_tool_name_is_namespaced_by_skill(self):
        assert _endpoint().tool_name("skill_ab12cd34") == "skill_ab12cd34_predict"

    @pytest.mark.parametrize(
        "url", ["ftp://example.com/x", "example.com/predict", "", "file:///etc/passwd"]
    )
    def test_non_http_url_rejected(self, url):
        with pytest.raises(ValueError):
            _endpoint(url=url)

    def test_empty_summary_rejected(self):
        with pytest.raises(ValueError):
            _endpoint(summary="   ")

    def test_host_is_lowercased(self):
        assert _endpoint(url="https://API.Example.COM/predict").host == "api.example.com"

    def test_colliding_ref_and_scalar_names_rejected(self):
        with pytest.raises(ValueError, match="colliding parameter names"):
            _endpoint(
                inputs=[{"name": "smiles", "data_type": "smiles"}],
                scalar_params=[{"name": "smiles_ref"}],
            )

    def test_duplicate_output_names_rejected(self):
        with pytest.raises(ValueError, match="duplicate output names"):
            _endpoint(
                outputs=[
                    {"name": "structure", "json_path": "$.a", "data_type": "pdb"},
                    {"name": "structure", "json_path": "$.b", "data_type": "pdb"},
                ]
            )

    def test_timeout_is_bounded(self):
        with pytest.raises(ValueError):
            _endpoint(timeout_seconds=6000)
        with pytest.raises(ValueError):
            _endpoint(timeout_seconds=0)

    def test_capture_unmapped_defaults_on(self):
        # Auxiliary scores are the main reason contracts exist, so the default
        # must not silently drop them.
        assert _endpoint().capture_unmapped is True


class TestSkillContract:
    def test_hosts_are_deduplicated_in_order(self):
        contract = SkillContract(
            endpoints=[
                _endpoint(tool_suffix="predict", url="https://a.example.com/1"),
                _endpoint(tool_suffix="score", url="https://b.example.com/2"),
                _endpoint(tool_suffix="rescore", url="https://a.example.com/3"),
            ]
        )
        assert contract.hosts == ["a.example.com", "b.example.com"]

    def test_duplicate_suffixes_rejected(self):
        with pytest.raises(ValueError, match="Duplicate endpoint tool_suffix"):
            SkillContract(endpoints=[_endpoint(), _endpoint()])

    def test_is_actionable_requires_confirmation_and_endpoints(self):
        assert not SkillContract(endpoints=[_endpoint()]).is_actionable
        assert not SkillContract(endpoints=[], confirmed=True).is_actionable
        assert SkillContract(endpoints=[_endpoint()], confirmed=True).is_actionable

    def test_to_dict_is_json_safe(self):
        data = SkillContract(endpoints=[_endpoint()], source="declared").to_dict()
        assert data["endpoints"][0]["inputs"][0]["data_type"] == "sequence"
        assert data["source"] == "declared"


class TestParseContract:
    def test_none_yields_empty_unconfirmed_contract(self):
        contract = parse_contract(None)
        assert contract.endpoints == []
        assert contract.source == "none"
        assert not contract.confirmed

    def test_bare_endpoint_list_is_accepted(self):
        # Frontmatter usually declares just the list under `endpoints:`.
        contract = parse_contract(
            [
                {
                    "tool_suffix": "predict",
                    "summary": "Predict.",
                    "url": "https://api.example.com/predict",
                }
            ],
            source="declared",
        )
        assert contract.source == "declared"
        assert contract.endpoints[0].tool_suffix == "predict"

    def test_source_override_wins(self):
        contract = parse_contract({"endpoints": [], "source": "llm"}, source="declared")
        assert contract.source == "declared"

    def test_scalar_input_rejected(self):
        with pytest.raises(SkillContractError):
            parse_contract("https://api.example.com/predict")

    def test_validation_failure_becomes_contract_error(self):
        with pytest.raises(SkillContractError):
            parse_contract({"endpoints": [{"tool_suffix": "predict"}]})

    def test_round_trips_through_to_dict(self):
        original = parse_contract(
            {"endpoints": [_endpoint().model_dump(mode="json")], "source": "llm"}
        )
        assert parse_contract(original.to_dict()) == original
