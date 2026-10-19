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

"""Tests for session artifact seeding and submission read-back.

Two things are worth pinning down here. First, type inference: a caller that
uploads ``receptor.cif`` and gets back a ``cif_`` reference has silently lost
access to ``convert_structure``, so the extension-to-type mapping is part of the
contract rather than an implementation detail. Second, all-or-nothing writes: a
batch with one bad entry must store nothing, because a harness that seeds three
files and gets two has no way to tell which task it is now running.
"""

import base64

import pytest
from fastapi import HTTPException

from api.artifacts import routes
from api.artifacts.infer import infer_data_type, type_from_content, type_from_filename
from api.artifacts.routes import ArtifactUpload, SeedArtifactsRequest
from api.artifacts.submissions import build_submission_metadata
from api.storage.datatypes import DataType
from api.storage.memory import InMemoryDataArtifactRepository
from api.storage.scoped import SessionScopedStorage
from api.storage.service import DataStorageService

SESSION = "11111111-2222-3333-4444-555555555555"

PDB_TEXT = """HEADER    HYDROLASE                               01-JAN-00   1ABC
ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00 20.00           N
END
"""

MMCIF_TEXT = """data_1ABC
#
_entry.id 1ABC
"""

SDF_TEXT = """ethanol
     RDKit          2D

  3  2  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0
$$$$
"""


class FakeDbSession:
    """Stands in for the AsyncSession the route commits."""

    def __init__(self):
        self.commits = 0

    async def commit(self):
        self.commits += 1


@pytest.fixture
def storage(monkeypatch):
    """Point the routes at in-memory storage."""
    service = DataStorageService(repository=InMemoryDataArtifactRepository())
    monkeypatch.setattr(
        routes,
        "scoped_storage",
        lambda _session, session_id: SessionScopedStorage(service, session_id),
    )
    return service


def upload(**kwargs) -> SeedArtifactsRequest:
    return SeedArtifactsRequest(artifacts=[ArtifactUpload(**kwargs)])


async def seed(request, session_id=SESSION, db=None):
    return await routes.create_session_artifacts(
        request, session_id=session_id, session=db or FakeDbSession()
    )


class TestTypeInference:
    def test_extensions_map_to_storage_types(self):
        assert type_from_filename("ligand.sdf") is DataType.SDF
        assert type_from_filename("receptor.pdb") is DataType.PDB
        assert type_from_filename("actives.smi") is DataType.SMILES
        assert type_from_filename("notes.md") is DataType.TEXT

    def test_cif_is_treated_as_mmcif(self):
        # The narrower `cif` type would cut the file off from convert_structure
        # and from every tool that names mmcif.
        assert type_from_filename("7xyz.cif") is DataType.MMCIF

    def test_unknown_and_missing_extensions_are_not_guessed(self):
        assert type_from_filename("data.frobnicate") is None
        assert type_from_filename("Makefile") is None
        assert type_from_filename(None) is None

    def test_content_sniffing_identifies_structure_formats(self):
        assert type_from_content(PDB_TEXT.encode()) is DataType.PDB
        assert type_from_content(MMCIF_TEXT.encode()) is DataType.MMCIF
        assert type_from_content(SDF_TEXT.encode()) is DataType.SDF
        assert type_from_content(b">seq1\nMKV\n") is DataType.FASTA
        assert type_from_content(b'{"a": 1}') is DataType.JSON

    def test_binary_signatures_are_recognized(self):
        assert type_from_content(b"\x89PNG\r\n\x1a\n\x00") is DataType.PNG
        assert type_from_content(b"PK\x03\x04rest") is DataType.ARRAY

    def test_featureless_content_yields_no_guess(self):
        assert type_from_content(b"CCO") is None
        assert type_from_content(b"") is None

    def test_declared_type_wins_over_the_filename(self):
        data_type, source = infer_data_type("smiles", "answer.txt", b"CCO")
        assert data_type is DataType.SMILES
        assert source == "declared"

    def test_filename_wins_over_content(self):
        # A .smi holding one line is indistinguishable from prose by content.
        data_type, source = infer_data_type(None, "actives.smi", b"CCO")
        assert data_type is DataType.SMILES
        assert source == "filename"

    def test_content_is_the_last_resort(self):
        data_type, source = infer_data_type(None, "download", PDB_TEXT.encode())
        assert data_type is DataType.PDB
        assert source == "content"

    def test_an_unresolvable_upload_reports_unknown(self):
        assert infer_data_type(None, "x.frob", b"CCO") == (None, "unknown")

    def test_aliases_resolve(self):
        data_type, _ = infer_data_type("pdbx", None, b"")
        assert data_type is DataType.MMCIF


@pytest.mark.asyncio
class TestSeeding:
    async def test_stores_content_and_returns_a_typed_reference(self, storage):
        response = await seed(upload(filename="ligand.sdf", content=SDF_TEXT))

        item = response.artifacts[0]
        assert item.reference.startswith("sdf_")
        assert item.data_type == "sdf"
        assert item.data_type_source == "filename"
        assert await storage.retrieve(item.reference) == SDF_TEXT.encode()

    async def test_the_artifact_is_scoped_to_the_session(self, storage):
        response = await seed(upload(filename="r.pdb", content=PDB_TEXT))
        listed = await storage.list_artifacts(session_id=SESSION)
        assert [a["reference"] for a in listed] == [response.artifacts[0].reference]
        assert await storage.list_artifacts(session_id="other-session-id") == []

    async def test_filenames_index_the_returned_references(self, storage):
        response = await seed(
            SeedArtifactsRequest(
                artifacts=[
                    ArtifactUpload(filename="receptor.pdb", content=PDB_TEXT),
                    ArtifactUpload(filename="ligand.sdf", content=SDF_TEXT),
                ]
            )
        )
        assert set(response.references) == {"receptor.pdb", "ligand.sdf"}
        assert response.references["ligand.sdf"].startswith("sdf_")

    async def test_uploads_without_a_filename_are_absent_from_the_index(self, storage):
        response = await seed(upload(data_type="smiles", content="CCO"))
        assert response.references == {}
        assert len(response.artifacts) == 1

    async def test_binary_content_round_trips_through_base64(self, storage):
        payload = b"\x89PNG\r\n\x1a\n" + bytes(range(256))
        response = await seed(
            upload(
                filename="depiction.png",
                content_base64=base64.b64encode(payload).decode(),
            )
        )
        assert response.artifacts[0].data_type == "png"
        assert await storage.retrieve(response.artifacts[0].reference) == payload

    async def test_a_binary_type_sent_as_text_is_refused(self, storage):
        with pytest.raises(HTTPException) as exc:
            await seed(upload(data_type="png", content="not really a png"))
        assert exc.value.status_code == 400
        assert "content_base64" in exc.value.detail

    async def test_the_description_defaults_to_the_filename(self, storage):
        response = await seed(upload(filename="ligand.sdf", content=SDF_TEXT))
        assert response.artifacts[0].description == "Uploaded ligand.sdf"

    async def test_caller_metadata_is_preserved_alongside_the_session_id(self, storage):
        response = await seed(
            upload(
                filename="ligand.sdf",
                content=SDF_TEXT,
                metadata={"task_id": "smdd_042"},
            )
        )
        _, metadata = await storage.retrieve_with_metadata(
            response.artifacts[0].reference
        )
        assert metadata["task_id"] == "smdd_042"
        assert metadata["session_id"] == SESSION
        assert metadata["filename"] == "ligand.sdf"

    async def test_the_write_is_committed(self, storage):
        db = FakeDbSession()
        await seed(upload(filename="ligand.sdf", content=SDF_TEXT), db=db)
        assert db.commits == 1

    async def test_an_unknown_declared_type_is_a_400(self, storage):
        with pytest.raises(HTTPException) as exc:
            await seed(upload(data_type="protein_blob", content="x"))
        assert exc.value.status_code == 400
        assert "not a known Aevom data type" in exc.value.detail

    async def test_an_unidentifiable_upload_is_a_400(self, storage):
        with pytest.raises(HTTPException) as exc:
            await seed(upload(filename="answer.frob", content="CCO"))
        assert exc.value.status_code == 400
        assert "data_type" in exc.value.detail

    async def test_missing_content_is_a_400(self, storage):
        with pytest.raises(HTTPException) as exc:
            await seed(upload(filename="ligand.sdf"))
        assert exc.value.status_code == 400

    async def test_malformed_base64_is_a_400(self, storage):
        with pytest.raises(HTTPException) as exc:
            await seed(upload(filename="a.npz", content_base64="not base64!!"))
        assert exc.value.status_code == 400

    async def test_an_empty_batch_is_a_400(self, storage):
        with pytest.raises(HTTPException) as exc:
            await seed(SeedArtifactsRequest(artifacts=[]))
        assert exc.value.status_code == 400

    async def test_an_empty_payload_is_a_400(self, storage):
        with pytest.raises(HTTPException) as exc:
            await seed(upload(filename="ligand.sdf", content=""))
        assert exc.value.status_code == 400

    async def test_an_oversized_artifact_is_a_413(self, storage, monkeypatch):
        monkeypatch.setattr(routes, "MAX_ARTIFACT_BYTES", 16)
        with pytest.raises(HTTPException) as exc:
            await seed(upload(data_type="text", content="x" * 32))
        assert exc.value.status_code == 413

    async def test_an_oversized_batch_is_a_413(self, storage, monkeypatch):
        monkeypatch.setattr(routes, "MAX_REQUEST_BYTES", 24)
        with pytest.raises(HTTPException) as exc:
            await seed(
                SeedArtifactsRequest(
                    artifacts=[
                        ArtifactUpload(data_type="text", content="x" * 16),
                        ArtifactUpload(data_type="text", content="y" * 16),
                    ]
                )
            )
        assert exc.value.status_code == 413

    async def test_too_many_artifacts_is_a_400(self, storage, monkeypatch):
        monkeypatch.setattr(routes, "MAX_ARTIFACTS_PER_REQUEST", 2)
        with pytest.raises(HTTPException) as exc:
            await seed(
                SeedArtifactsRequest(
                    artifacts=[
                        ArtifactUpload(data_type="text", content="a"),
                        ArtifactUpload(data_type="text", content="b"),
                        ArtifactUpload(data_type="text", content="c"),
                    ]
                )
            )
        assert exc.value.status_code == 400

    async def test_a_rejected_batch_stores_nothing(self, storage):
        # The valid entry comes first, so a per-entry write loop would have
        # persisted it before reaching the bad one.
        with pytest.raises(HTTPException):
            await seed(
                SeedArtifactsRequest(
                    artifacts=[
                        ArtifactUpload(filename="receptor.pdb", content=PDB_TEXT),
                        ArtifactUpload(data_type="nonsense", content="x"),
                    ]
                )
            )
        assert await storage.list_artifacts(session_id=SESSION) == []


class TestSessionIdValidation:
    def test_a_uuid_is_accepted(self):
        assert routes.require_valid_session_id(SESSION) == SESSION

    def test_training_style_ids_are_accepted(self):
        assert routes.require_valid_session_id("env_a1b2c3d4e5f6") is not None

    @pytest.mark.parametrize("bad", ["", "short", "has spaces in it", "../../etc/passwd"])
    def test_junk_is_rejected(self, bad):
        with pytest.raises(HTTPException) as exc:
            routes.require_valid_session_id(bad)
        assert exc.value.status_code == 400


@pytest.mark.asyncio
class TestSubmissionReadBack:
    async def record_submission(self, storage, artifact_ref, data_type, **kwargs):
        return await storage.store(
            data_type="json",
            content=b"{}",
            metadata=build_submission_metadata(
                artifact_ref, data_type, session_id=SESSION, **kwargs
            ),
            description="Submitted answer",
        )

    async def test_lists_only_submission_records(self, storage):
        await storage.store(
            data_type="sdf",
            content=SDF_TEXT.encode(),
            metadata={"session_id": SESSION},
        )
        record_ref = await self.record_submission(
            storage, "sdf_deadbeef", "sdf", label="analogue 12", notes="best gate pass"
        )

        response = await routes.list_session_submissions(
            session_id=SESSION, session=FakeDbSession()
        )

        assert len(response.submissions) == 1
        submission = response.submissions[0]
        assert submission.artifact_ref == "sdf_deadbeef"
        assert submission.data_type == "sdf"
        assert submission.label == "analogue 12"
        assert submission.notes == "best gate pass"
        assert submission.record_ref == record_ref

    async def test_a_session_with_no_submission_returns_an_empty_list(self, storage):
        response = await routes.list_session_submissions(
            session_id=SESSION, session=FakeDbSession()
        )
        assert response.submissions == []

    async def test_other_sessions_submissions_are_invisible(self, storage):
        await storage.store(
            data_type="json",
            content=b"{}",
            metadata=build_submission_metadata(
                "sdf_deadbeef", "sdf", session_id="another-session-id"
            ),
        )
        response = await routes.list_session_submissions(
            session_id=SESSION, session=FakeDbSession()
        )
        assert response.submissions == []
