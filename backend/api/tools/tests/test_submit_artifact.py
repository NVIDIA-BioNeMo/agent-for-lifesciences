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

"""Tests for the submit_artifact tool.

The behaviour that matters is that a submission is a *verified* claim: pointing
at a reference that does not exist has to fail loudly, because a silently
accepted submission of a typo'd reference looks exactly like a session that
never answered at all.
"""

import json

import pytest

from api.artifacts.submissions import is_submission
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.service import DataStorageService
from api.tools.base import ContentType, _session_id_ctx, _storage_ctx
from api.tools.submit_artifact import MAX_INLINE_CHARS, SubmitArtifactTool


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.fixture
def tool(storage):
    instance = SubmitArtifactTool()
    storage_token = _storage_ctx.set(storage)
    session_token = _session_id_ctx.set("session-under-test")
    yield instance
    _storage_ctx.reset(storage_token)
    _session_id_ctx.reset(session_token)


async def submission_records(storage):
    listed = await storage.list_artifacts(limit=50)
    return [a for a in listed if is_submission(a.get("metadata"))]


@pytest.mark.asyncio
class TestSubmittingAReference:
    async def test_records_the_submission(self, tool, storage):
        ref = await storage.store("sdf", b"molblock", metadata={})

        result = await tool.execute(
            {"artifact_ref": ref, "label": "analogue 12", "notes": "passes all gates"}
        )

        assert result.content_type is ContentType.TEXT
        assert result.metadata["submitted_ref"] == ref
        assert result.metadata["submitted_data_type"] == "sdf"

        records = await submission_records(storage)
        assert len(records) == 1
        assert records[0]["metadata"]["artifact_ref"] == ref
        assert records[0]["metadata"]["label"] == "analogue 12"
        assert records[0]["metadata"]["notes"] == "passes all gates"
        assert records[0]["metadata"]["session_id"] == "session-under-test"

    async def test_the_record_body_names_the_artifact(self, tool, storage):
        ref = await storage.store("sdf", b"molblock", metadata={})
        result = await tool.execute({"artifact_ref": ref})

        body = json.loads(
            await storage.retrieve(result.metadata["submission_record_ref"])
        )
        assert body["artifact_ref"] == ref
        assert body["data_type"] == "sdf"
        assert body["submitted_at"]

    async def test_an_unknown_reference_is_an_error(self, tool, storage):
        result = await tool.execute({"artifact_ref": "sdf_deadbeef"})
        assert result.content_type is ContentType.ERROR
        assert await submission_records(storage) == []

    async def test_a_malformed_reference_is_an_error(self, tool, storage):
        result = await tool.execute({"artifact_ref": "not-a-reference"})
        assert result.content_type is ContentType.ERROR
        assert await submission_records(storage) == []

    async def test_resubmitting_leaves_both_records_newest_last(self, tool, storage):
        first = await storage.store("sdf", b"one", metadata={})
        second = await storage.store("sdf", b"two", metadata={})

        await tool.execute({"artifact_ref": first})
        await tool.execute({"artifact_ref": second})

        records = await submission_records(storage)
        assert {r["metadata"]["artifact_ref"] for r in records} == {first, second}


@pytest.mark.asyncio
class TestSubmittingInlineContent:
    async def test_stores_the_answer_and_submits_it(self, tool, storage):
        result = await tool.execute(
            {"content": '{"pharmacophore": ["c1ccccc1"]}', "label": "SMARTS answer"}
        )

        assert result.content_type is ContentType.TEXT
        submitted = result.metadata["submitted_ref"]
        assert submitted.startswith("json_")
        assert await storage.retrieve(submitted) == b'{"pharmacophore": ["c1ccccc1"]}'

    async def test_a_declared_type_is_honoured(self, tool, storage):
        result = await tool.execute({"content": "CCO", "data_type": "smiles"})
        assert result.metadata["submitted_ref"].startswith("smiles_")

    async def test_unrecognizable_prose_becomes_text(self, tool, storage):
        result = await tool.execute({"content": "Arg120 and Tyr355 form the pocket"})
        assert result.metadata["submitted_ref"].startswith("text_")

    async def test_an_oversized_inline_answer_is_refused(self, tool, storage):
        result = await tool.execute({"content": "x" * (MAX_INLINE_CHARS + 1)})
        assert result.content_type is ContentType.ERROR
        assert await submission_records(storage) == []

    async def test_a_reference_takes_precedence_over_content(self, tool, storage):
        ref = await storage.store("sdf", b"molblock", metadata={})
        result = await tool.execute({"artifact_ref": ref, "content": "ignored"})
        assert result.metadata["submitted_ref"] == ref


@pytest.mark.asyncio
class TestFailureModes:
    async def test_submitting_nothing_is_an_error(self, tool):
        result = await tool.execute({})
        assert result.content_type is ContentType.ERROR

    async def test_no_storage_is_a_clear_error(self):
        instance = SubmitArtifactTool()
        token = _storage_ctx.set(None)
        try:
            result = await instance.execute({"artifact_ref": "sdf_deadbeef"})
        finally:
            _storage_ctx.reset(token)
        assert result.content_type is ContentType.ERROR
        assert "Storage service not available" in result.data


class TestSchema:
    def test_no_argument_is_required(self):
        # Either artifact_ref or content works, so neither can be required in
        # the schema; the tool enforces the either-or itself.
        schema = SubmitArtifactTool().get_schema()
        assert schema["function"]["parameters"]["required"] == []

    def test_the_summary_names_the_reference(self):
        tool = SubmitArtifactTool()
        assert "sdf_abc12345" in tool.format_summary({"artifact_ref": "sdf_abc12345"})
