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

"""Tests for the SAFE notation conversion tools.

The RDKit service is mocked here; the real conversion is covered by
`services/rdkit/tests/test_safe_notation.py`, which runs in the image where
`safe-mol` is installed. What matters at this layer is the type plumbing: the
declared reference types, which data type each tool writes, and whether the
messages the model reads carry the SAFE string and the not-fragmented warning.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from api.storage import ArtifactNotFoundError, InvalidReferenceError
from api.tools.safe_notation import SafeToSmilesTool, SmilesToSafeTool

IBUPROFEN = "CC(Cc1ccc(cc1)C(C(=O)O)C)C"
IBUPROFEN_SAFE = "CC(C)C1CCC1.C1CCC1C(C)C(=O)O"


def make_storage(content: bytes, metadata: dict | None = None, reference="safe_new1"):
    """A storage double that returns `content` and records what it is given."""
    storage = MagicMock()
    storage.retrieve_with_metadata = AsyncMock(return_value=(content, metadata or {}))
    storage.store = AsyncMock(return_value=reference)
    return storage


class TestSchemas:
    def test_smiles_to_safe_declares_a_smiles_reference(self):
        # The x-aevom-reference annotation is what makes the ref resolvable and
        # what the UI uses to offer the right artifacts, so it has to name the
        # *input* type, not the output.
        params = SmilesToSafeTool().get_schema()["function"]["parameters"]
        assert params["properties"]["smiles_ref"]["x-aevom-reference"] == {
            "data_type": "smiles"
        }
        assert params["required"] == ["smiles_ref"]

    def test_safe_to_smiles_declares_a_safe_reference(self):
        params = SafeToSmilesTool().get_schema()["function"]["parameters"]
        assert params["properties"]["safe_ref"]["x-aevom-reference"] == {
            "data_type": "safe"
        }
        assert params["required"] == ["safe_ref"]

    def test_the_slicer_enum_is_constrained(self):
        slicer = SmilesToSafeTool().get_schema()["function"]["parameters"][
            "properties"
        ]["slicer"]
        assert "brics" in slicer["enum"]

    def test_both_tools_store_and_retrieve(self):
        for tool in (SmilesToSafeTool(), SafeToSmilesTool()):
            assert tool.data_io_category.value == "both"

    def test_the_description_tells_the_model_that_genmol_needs_safe(self):
        # Without this the model has no way to know a field named `smiles` wants
        # SAFE, and the mistake is silent.
        description = SmilesToSafeTool().get_schema()["function"]["description"]
        assert "GenMol" in description


class TestSmilesToSafeExecution:
    @pytest.fixture
    def tool(self):
        instance = SmilesToSafeTool()
        instance._call_service = AsyncMock(
            return_value={
                "success": True,
                "safe": IBUPROFEN_SAFE,
                "num_fragments": 2,
                "fragmented": True,
                "masked": False,
            }
        )
        return instance

    @pytest.mark.asyncio
    async def test_stores_a_safe_artifact(self, tool):
        storage = make_storage(IBUPROFEN.encode())
        tool.set_storage(storage)

        result = await tool.execute({"smiles_ref": "smiles_abc123"})

        assert result.content_type.value == "data_reference"
        assert storage.store.call_args.kwargs["data_type"] == "safe"
        assert storage.store.call_args.kwargs["content"] == IBUPROFEN_SAFE.encode()

    @pytest.mark.asyncio
    async def test_records_the_source_smiles_and_slicer(self, tool):
        storage = make_storage(IBUPROFEN.encode())
        tool.set_storage(storage)

        await tool.execute({"smiles_ref": "smiles_abc123"})

        metadata = storage.store.call_args.kwargs["metadata"]
        assert metadata["smiles"] == IBUPROFEN
        assert metadata["smiles_ref"] == "smiles_abc123"
        assert metadata["slicer"] == "brics"

    @pytest.mark.asyncio
    async def test_carries_forward_the_source_name(self, tool):
        storage = make_storage(IBUPROFEN.encode(), {"name": "ibuprofen"})
        tool.set_storage(storage)

        await tool.execute({"smiles_ref": "smiles_abc123"})

        assert storage.store.call_args.kwargs["metadata"]["name"] == "ibuprofen"

    @pytest.mark.asyncio
    async def test_the_message_includes_the_safe_string(self, tool):
        # The model needs to see the fragments to reason about them without
        # spending another tool call loading the artifact.
        tool.set_storage(make_storage(IBUPROFEN.encode()))
        result = await tool.execute({"smiles_ref": "smiles_abc123"})
        assert IBUPROFEN_SAFE in result.metadata["message"]

    @pytest.mark.asyncio
    async def test_forwards_the_mask_bounds_to_the_service(self, tool):
        tool.set_storage(make_storage(IBUPROFEN.encode()))

        await tool.execute(
            {"smiles_ref": "smiles_abc123", "mask_min": 10, "mask_max": 15}
        )

        payload = tool._call_service.call_args.kwargs["json_data"]
        assert payload["mask_min"] == 10
        assert payload["mask_max"] == 15

    @pytest.mark.asyncio
    async def test_one_mask_bound_alone_is_rejected_before_any_call(self, tool):
        tool.set_storage(make_storage(IBUPROFEN.encode()))

        result = await tool.execute({"smiles_ref": "smiles_abc123", "mask_min": 10})

        assert result.content_type.value == "error"
        assert "together" in result.data
        tool._call_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unknown_slicer_is_rejected_before_any_call(self, tool):
        tool.set_storage(make_storage(IBUPROFEN.encode()))

        result = await tool.execute(
            {"smiles_ref": "smiles_abc123", "slicer": "nonsense"}
        )

        assert result.content_type.value == "error"
        tool._call_service.assert_not_called()

    @pytest.mark.asyncio
    async def test_an_unfragmented_result_warns_that_it_cannot_be_decorated(self, tool):
        # A single-fragment SAFE string round-trips fine, but a mask on it has no
        # attachment point, so GenMol would echo the input back. The previous
        # wording ("still a valid model input") read as reassurance and led the
        # model to keep pushing a scaffold that could never be decorated.
        tool._call_service = AsyncMock(
            return_value={
                "success": True,
                "safe": "c1ccccc1",
                "num_fragments": 1,
                "fragmented": False,
                "masked": False,
            }
        )
        tool.set_storage(make_storage(b"c1ccccc1"))

        result = await tool.execute({"smiles_ref": "smiles_benzene"})

        assert result.content_type.value == "data_reference"
        message = result.metadata["message"]
        assert "CANNOT be used for scaffold decoration" in message
        assert "unchanged" in message

    @pytest.mark.asyncio
    async def test_a_masked_result_reports_which_fragment_it_replaced(self, tool):
        tool._call_service = AsyncMock(
            return_value={
                "success": True,
                "safe": "n13cnc2c1c(=O)n4c(=O)n25.C3.C5.[*{5-12}]",
                "num_fragments": 3,
                "fragmented": True,
                "masked": True,
                "slicer_used": "recap",
                "replaced_fragment": "C4",
            }
        )
        tool.set_storage(make_storage(b"CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))

        result = await tool.execute(
            {"smiles_ref": "smiles_caffeine", "mask_min": 5, "mask_max": 12}
        )

        assert result.content_type.value == "data_reference"
        message = result.metadata["message"]
        assert "replaced the terminal fragment 'C4'" in message
        assert "open attachment point" in message
        assert result.metadata["replaced_fragment"] == "C4"

    @pytest.mark.asyncio
    async def test_a_slicer_substitution_is_reported(self, tool):
        # BRICS cannot cut caffeine; saying so explains the unfamiliar cut points.
        tool._call_service = AsyncMock(
            return_value={
                "success": True,
                "safe": "n13cnc2c1c(=O)n4c(=O)n25.C3.C5.C4",
                "num_fragments": 4,
                "fragmented": True,
                "masked": False,
                "slicer_used": "recap",
            }
        )
        tool.set_storage(make_storage(b"CN1C=NC2=C1C(=O)N(C(=O)N2C)C"))

        result = await tool.execute({"smiles_ref": "smiles_caffeine"})

        assert "'brics' slicer found no cut points" in result.metadata["message"]
        assert "'recap' was used instead" in result.metadata["message"]
        assert result.metadata["slicer"] == "recap"

    @pytest.mark.asyncio
    async def test_a_missing_ref_is_an_error(self, tool):
        tool.set_storage(make_storage(b""))
        result = await tool.execute({})
        assert result.content_type.value == "error"
        assert "smiles_ref is required" in result.data

    @pytest.mark.asyncio
    async def test_an_invalid_reference_names_the_expected_shape(self, tool):
        storage = MagicMock()
        storage.retrieve_with_metadata = AsyncMock(side_effect=InvalidReferenceError())
        tool.set_storage(storage)

        result = await tool.execute({"smiles_ref": "not-a-ref"})

        assert result.content_type.value == "error"
        assert "smiles_abc12345" in result.data

    @pytest.mark.asyncio
    async def test_a_missing_artifact_is_an_error(self, tool):
        storage = MagicMock()
        storage.retrieve_with_metadata = AsyncMock(side_effect=ArtifactNotFoundError())
        tool.set_storage(storage)

        result = await tool.execute({"smiles_ref": "smiles_gone"})

        assert result.content_type.value == "error"
        assert "No smiles found" in result.data

    @pytest.mark.asyncio
    async def test_an_empty_stored_smiles_is_an_error(self, tool):
        tool.set_storage(make_storage(b"   "))
        result = await tool.execute({"smiles_ref": "smiles_empty"})
        assert result.content_type.value == "error"
        assert "empty" in result.data

    @pytest.mark.asyncio
    async def test_a_service_failure_is_surfaced(self, tool):
        tool._call_service = AsyncMock(
            return_value={"success": False, "error": "SAFE encoding failed: boom"}
        )
        tool.set_storage(make_storage(IBUPROFEN.encode()))

        result = await tool.execute({"smiles_ref": "smiles_abc123"})

        assert result.content_type.value == "error"
        assert "boom" in result.data


class TestSafeToSmilesExecution:
    @pytest.fixture
    def tool(self):
        instance = SafeToSmilesTool()
        instance._call_service = AsyncMock(
            return_value={"success": True, "smiles": IBUPROFEN}
        )
        return instance

    @pytest.mark.asyncio
    async def test_stores_a_smiles_artifact(self, tool):
        storage = make_storage(IBUPROFEN_SAFE.encode(), reference="smiles_new1")
        tool.set_storage(storage)

        result = await tool.execute({"safe_ref": "safe_abc123"})

        assert result.content_type.value == "data_reference"
        assert storage.store.call_args.kwargs["data_type"] == "smiles"
        assert storage.store.call_args.kwargs["content"] == IBUPROFEN.encode()

    @pytest.mark.asyncio
    async def test_records_the_source_safe_string(self, tool):
        storage = make_storage(IBUPROFEN_SAFE.encode(), reference="smiles_new1")
        tool.set_storage(storage)

        await tool.execute({"safe_ref": "safe_abc123"})

        metadata = storage.store.call_args.kwargs["metadata"]
        assert metadata["safe"] == IBUPROFEN_SAFE
        assert metadata["safe_ref"] == "safe_abc123"

    @pytest.mark.asyncio
    async def test_a_missing_ref_is_an_error(self, tool):
        tool.set_storage(make_storage(b""))
        result = await tool.execute({})
        assert result.content_type.value == "error"
        assert "safe_ref is required" in result.data

    @pytest.mark.asyncio
    async def test_a_missing_artifact_is_an_error(self, tool):
        storage = MagicMock()
        storage.retrieve_with_metadata = AsyncMock(side_effect=ArtifactNotFoundError())
        tool.set_storage(storage)

        result = await tool.execute({"safe_ref": "safe_gone"})

        assert result.content_type.value == "error"
        assert "No safe found" in result.data

    @pytest.mark.asyncio
    async def test_a_service_failure_is_surfaced(self, tool):
        tool._call_service = AsyncMock(
            return_value={"success": False, "error": "Invalid SAFE string"}
        )
        tool.set_storage(make_storage(b"garbage"))

        result = await tool.execute({"safe_ref": "safe_bad"})

        assert result.content_type.value == "error"
        assert "Invalid SAFE string" in result.data


class TestRegistration:
    def test_both_tools_are_registered_by_default(self):
        from api.tools.registry import get_registry

        names = get_registry().tool_names
        assert "smiles_to_safe" in names
        assert "safe_to_smiles" in names
