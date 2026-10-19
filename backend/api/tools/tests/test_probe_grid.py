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

"""Tests for the probe_grid tool.

The service is stubbed throughout: the scoring function, the atom typing and the
agreement with ``vina --write_maps`` are the vina suite's business
(``services/vina/tests/test_probe_grid.py``). What matters here is the handoff --
that a 69,000-point grid per probe becomes a reference rather than context, that
the stored bytes are the npz the sandbox array loader expects, and that the
inline summary says enough to act on without the array.
"""

import base64
import io

import numpy as np
import pytest

from api.storage import DataStorageService, InMemoryDataArtifactRepository
from api.tools.base import ContentType
from api.tools.probe_grid import PROBE_TYPES, ProbeGridTool

CENTER = [31.575, -1.590, 25.599]
SHAPE = [5, 5, 5]
ORIGIN = [30.575, -2.590, 24.599]


def grid_array(probes: int = 3, fill: float = -0.4) -> np.ndarray:
    """A grid shaped the way the service returns one: (probe, nx, ny, nz)."""
    return np.full((probes, *SHAPE), fill, dtype=np.float32)


def npz_base64(array: np.ndarray) -> str:
    buffer = io.BytesIO()
    # The archive key has to be "data": that is what _decode_array in the
    # sandbox client reads, and a different key fails only at load time.
    np.savez_compressed(buffer, data=array)
    return base64.b64encode(buffer.getvalue()).decode()


def minimum(
    probe: str = "O_A",
    family: str = "Acceptor",
    x: float = 31.6,
    y: float = -1.6,
    z: float = 25.6,
    energy: float = -0.812,
) -> dict:
    return {
        "probe": probe,
        "family": family,
        "x": x,
        "y": y,
        "z": z,
        "energy": energy,
    }


def response(
    array: np.ndarray | None = None,
    probes: list[str] | None = None,
    minima: list[dict] | None = None,
) -> dict:
    """A successful service response."""
    probes = probes if probes is not None else ["C_H", "N_D", "O_A"]
    array = grid_array(len(probes)) if array is None else array
    return {
        "success": True,
        "npz_base64": npz_base64(array),
        "probes": probes,
        "origin": ORIGIN,
        "spacing": 0.5,
        "shape": SHAPE,
        "minima": minima if minima is not None else [minimum()],
        "num_receptor_atoms": 871,
        "num_template_typed": 864,
        "error": None,
    }


@pytest.fixture
def storage():
    return DataStorageService(repository=InMemoryDataArtifactRepository())


@pytest.fixture
def tool(storage):
    instance = ProbeGridTool()
    instance._storage = storage
    return instance


def stub(tool, payload, calls: list | None = None):
    """Replace the tool's service call with a canned response."""

    async def _call(endpoint, method="POST", json_data=None, timeout=30.0, retries=0):
        if calls is not None:
            calls.append(json_data)
        if isinstance(payload, Exception):
            raise payload
        return payload

    tool._call_service = _call


async def receptor_ref(storage) -> str:
    return await storage.store(
        "pdb", b"ATOM      1  N   ALA A   1      11.104   6.134  -6.504\nEND\n"
    )


class TestInputHandling:
    @pytest.mark.asyncio
    async def test_requires_a_receptor(self, tool):
        result = await tool.execute({"center": CENTER})
        assert result.content_type == ContentType.ERROR
        assert "receptor_pdb_ref" in result.data

    @pytest.mark.asyncio
    async def test_requires_a_centre(self, tool, storage):
        ref = await receptor_ref(storage)
        result = await tool.execute({"receptor_pdb_ref": ref})
        assert result.content_type == ContentType.ERROR
        assert "center" in result.data

    @pytest.mark.asyncio
    async def test_a_two_component_centre_is_refused(self, tool, storage):
        ref = await receptor_ref(storage)
        result = await tool.execute({"receptor_pdb_ref": ref, "center": [1.0, 2.0]})
        assert result.content_type == ContentType.ERROR

    @pytest.mark.asyncio
    async def test_an_unknown_probe_lists_the_available_ones(self, tool, storage):
        # Silently dropping it would return a grid missing the axis the caller
        # asked for, which reads as "no site for this type".
        ref = await receptor_ref(storage)
        result = await tool.execute(
            {"receptor_pdb_ref": ref, "center": CENTER, "probes": ["C_H", "Zn"]}
        )
        assert result.content_type == ContentType.ERROR
        assert "Zn" in result.data
        assert "O_A" in result.data

    @pytest.mark.asyncio
    async def test_an_invalid_reference_says_where_to_get_one(self, tool):
        result = await tool.execute(
            {"receptor_pdb_ref": "not-a-reference", "center": CENTER}
        )
        assert result.content_type == ContentType.ERROR
        assert "fetch_and_clean_pdb" in result.data

    @pytest.mark.asyncio
    async def test_a_missing_artifact_is_an_error(self, tool):
        result = await tool.execute(
            {"receptor_pdb_ref": "pdb_deadbeef", "center": CENTER}
        )
        assert result.content_type == ContentType.ERROR
        assert "not found" in result.data

    @pytest.mark.asyncio
    async def test_requires_storage(self):
        instance = ProbeGridTool()
        instance._storage = None
        result = await instance.execute(
            {"receptor_pdb_ref": "pdb_abc12345", "center": CENTER}
        )
        assert result.content_type == ContentType.ERROR
        assert "Storage" in result.data


class TestServiceRequest:
    @pytest.mark.asyncio
    async def test_sends_the_receptor_contents_not_the_reference(self, tool, storage):
        ref = await receptor_ref(storage)
        calls: list = []
        stub(tool, response(), calls)

        await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})

        assert calls[0]["pdb_content"].startswith("ATOM")

    @pytest.mark.asyncio
    async def test_defaults_to_a_twenty_angstrom_box_and_three_probes(
        self, tool, storage
    ):
        ref = await receptor_ref(storage)
        calls: list = []
        stub(tool, response(), calls)

        await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})

        assert calls[0]["size"] == [20.0, 20.0, 20.0]
        assert calls[0]["spacing"] == 0.5
        assert calls[0]["probes"] == ["C_H", "N_D", "O_A"]

    @pytest.mark.asyncio
    async def test_passes_the_box_and_spacing_through(self, tool, storage):
        ref = await receptor_ref(storage)
        calls: list = []
        stub(tool, response(), calls)

        await tool.execute(
            {
                "receptor_pdb_ref": ref,
                "center": CENTER,
                "size": [24, 24, 24],
                "spacing": 0.375,
                "max_minima": 8,
            }
        )

        assert calls[0]["size"] == [24.0, 24.0, 24.0]
        assert calls[0]["spacing"] == 0.375
        assert calls[0]["max_minima"] == 8

    @pytest.mark.asyncio
    async def test_integer_coordinates_are_sent_as_floats(self, tool, storage):
        # The service validates floats; a JSON integer would be coerced, but
        # sending 31 for 31.575 would be a silent half-angstrom shift if the
        # model wrote the centre as an int in the first place.
        ref = await receptor_ref(storage)
        calls: list = []
        stub(tool, response(), calls)

        await tool.execute({"receptor_pdb_ref": ref, "center": [31, -2, 26]})

        assert calls[0]["center"] == [31.0, -2.0, 26.0]


class TestStoredGrid:
    @pytest.mark.asyncio
    async def test_returns_an_array_reference(self, tool, storage):
        ref = await receptor_ref(storage)
        stub(tool, response())

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})

        assert result.content_type == ContentType.DATA_REFERENCE
        assert result.data.startswith("array_")

    @pytest.mark.asyncio
    async def test_the_stored_bytes_load_as_numpy_under_the_key_data(
        self, tool, storage
    ):
        # aevom.load_array() reads archive["data"]. If the archive key or the
        # base64 decoding were wrong, this tool would still return a plausible
        # reference and fail only inside the sandbox.
        array = grid_array()
        ref = await receptor_ref(storage)
        stub(tool, response(array))

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})

        content = await storage.retrieve(result.data)
        with np.load(io.BytesIO(content)) as archive:
            assert np.allclose(archive["data"], array)

    @pytest.mark.asyncio
    async def test_the_probe_axis_leads_the_stored_shape(self, tool, storage):
        ref = await receptor_ref(storage)
        stub(tool, response(grid_array(probes=2), probes=["C_H", "O_A"]))

        result = await tool.execute(
            {"receptor_pdb_ref": ref, "center": CENTER, "probes": ["C_H", "O_A"]}
        )

        assert result.metadata["grid_shape"] == [2, *SHAPE]
        assert result.metadata["shape"] == SHAPE

    @pytest.mark.asyncio
    async def test_metadata_carries_the_geometry_needed_to_index_the_grid(
        self, tool, storage
    ):
        # Without origin and spacing the array is a block of numbers with no
        # position, so it cannot be turned back into coordinates.
        ref = await receptor_ref(storage)
        stub(tool, response())

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})

        assert result.metadata["origin"] == ORIGIN
        assert result.metadata["spacing"] == 0.5
        assert result.metadata["center"] == CENTER
        assert result.metadata["probes"] == ["C_H", "N_D", "O_A"]
        assert result.metadata["receptor_ref"] == ref

    @pytest.mark.asyncio
    async def test_the_minima_travel_in_metadata_as_well_as_prose(self, tool, storage):
        # Downstream code should not have to parse the message back.
        ref = await receptor_ref(storage)
        stub(tool, response(minima=[minimum(), minimum(probe="C_H")]))

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})

        assert [m["probe"] for m in result.metadata["minima"]] == ["O_A", "C_H"]


class TestMessage:
    @pytest.mark.asyncio
    async def test_ranks_the_minima_with_coordinates_and_energies(self, tool, storage):
        ref = await receptor_ref(storage)
        stub(tool, response(minima=[minimum(energy=-0.812)]))

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})
        message = result.to_tool_response()

        assert "Acceptor" in message
        assert "(31.60, -1.60, 25.60)" in message
        assert "-0.812" in message

    @pytest.mark.asyncio
    async def test_explains_what_a_family_means(self, tool, storage):
        # "Acceptor site" is ambiguous on its own: it is where a *ligand*
        # acceptor goes, which is next to a receptor donor. Reading it the other
        # way inverts every polar suggestion the grid makes.
        ref = await receptor_ref(storage)
        stub(tool, response())

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})
        message = result.to_tool_response()

        assert "ligand atom that belongs at that point" in message
        assert "receptor donor" in message

    @pytest.mark.asyncio
    async def test_says_how_to_index_the_array(self, tool, storage):
        ref = await receptor_ref(storage)
        stub(tool, response())

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})
        message = result.to_tool_response()

        assert "origin + spacing * (i, j, k)" in message
        assert "125 points" in message

    @pytest.mark.asyncio
    async def test_points_at_python_execute_for_the_array(self, tool, storage):
        ref = await receptor_ref(storage)
        stub(tool, response())

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})
        message = result.to_tool_response()

        assert "python_execute" in message

    @pytest.mark.asyncio
    async def test_the_grid_itself_never_reaches_the_message(self, tool, storage):
        # The whole point of the reference. A 20 A box at 0.5 A spacing is
        # 69,000 numbers per probe; even summarising them inline would crowd out
        # the rest of the turn.
        ref = await receptor_ref(storage)
        stub(tool, response(np.full((3, 51, 51, 51), -0.4, dtype=np.float32)))

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})
        message = result.to_tool_response()

        assert len(message) < 2000

    @pytest.mark.asyncio
    async def test_an_empty_box_says_what_to_check(self, tool, storage):
        ref = await receptor_ref(storage)
        stub(tool, response(np.zeros((3, *SHAPE), dtype=np.float32), minima=[]))

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})
        message = result.to_tool_response()

        assert "No favourable minima" in message
        assert "centred off the pocket" in message

    @pytest.mark.asyncio
    async def test_reports_how_much_typing_came_from_templates(self, tool, storage):
        # The typing path is where this beats a geometric score, so how much of
        # the receptor went through it is worth stating.
        ref = await receptor_ref(storage)
        stub(tool, response())

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})
        message = result.to_tool_response()

        assert "864 of 871" in message


class TestErrors:
    @pytest.mark.asyncio
    async def test_a_service_error_is_reported(self, tool, storage):
        ref = await receptor_ref(storage)
        stub(tool, {"success": False, "error": "Receptor preparation failed."})

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})

        assert result.content_type == ContentType.ERROR
        assert "Receptor preparation failed." in result.data

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_reported(self, tool, storage):
        ref = await receptor_ref(storage)
        stub(tool, RuntimeError("connection refused"))

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})

        assert result.content_type == ContentType.ERROR
        assert "connection refused" in result.data

    @pytest.mark.asyncio
    async def test_an_unreadable_payload_is_an_error_not_a_reference(
        self, tool, storage
    ):
        # Storing a corrupt archive would fail later, in the sandbox, with
        # nothing pointing back here.
        ref = await receptor_ref(storage)
        stub(tool, {"success": True, "npz_base64": "not base64!!", "probes": ["C_H"]})

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})

        assert result.content_type == ContentType.ERROR
        assert "unreadable" in result.data

    @pytest.mark.asyncio
    async def test_a_missing_payload_is_an_error(self, tool, storage):
        ref = await receptor_ref(storage)
        stub(tool, {"success": True, "probes": ["C_H"]})

        result = await tool.execute({"receptor_pdb_ref": ref, "center": CENTER})

        assert result.content_type == ContentType.ERROR


class TestSchema:
    def test_declares_the_receptor_as_a_pdb_reference(self, tool):
        params = tool.get_schema()["function"]["parameters"]
        assert params["properties"]["receptor_pdb_ref"]["x-aevom-reference"] == {
            "data_type": "pdb"
        }
        assert params["required"] == ["receptor_pdb_ref", "center"]

    def test_the_probe_enum_matches_the_implementation(self, tool):
        params = tool.get_schema()["function"]["parameters"]
        assert params["properties"]["probes"]["items"]["enum"] == list(PROBE_TYPES)

    def test_separates_itself_from_docking_and_profiling(self, tool):
        # The failure this prevents is reaching for probe_grid to score a
        # molecule, or for vina_dock to find a pharmacophore point.
        description = tool.get_schema()["function"]["description"]
        assert "vina_dock" in description
        assert "plip_interactions" in description

    def test_says_what_an_energy_means(self, tool):
        # These are single-atom energies around -1 kcal/mol. Read as affinities
        # they look like failures.
        description = tool.get_schema()["function"]["description"]
        assert "not binding affinities" in description

    def test_the_summary_names_the_centre(self, tool):
        assert "31.6, -1.6, 25.6" in tool.format_summary({"center": CENTER})

    def test_the_summary_survives_a_malformed_centre(self, tool):
        assert tool.format_summary({"center": "middle"})
