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

"""Tests for SAFE notation conversion in the RDKit service.

These run in the RDKit image, which is where `safe-mol` is installed with a
torch-free `__init__` shim. Two things are worth pinning here beyond the happy
path:

- **The fragmentation fallback.** Small or highly-fused molecules genuinely have
  no cut points, and a single-fragment SAFE string is still a valid model input.
  Reporting that as an error would break legitimate scaffolds.
- **The shim.** `import safe` must not drag in torch. If upstream's eager
  `__init__` ever comes back the image grows by gigabytes, which is easy to miss
  and expensive to ship.
"""

import sys

import pytest
from fastapi.testclient import TestClient
from rdkit import Chem

from app.main import app
from app.safe_notation import DEFAULT_SLICER, apply_mask, safe_to_smiles, smiles_to_safe

# Ibuprofen: several BRICS cut points, so it encodes to multiple fragments.
IBUPROFEN = "CC(Cc1ccc(cc1)C(C(=O)O)C)C"
# Benzene has no cut points under any slicer -- the fragmentation fallback case.
BENZENE = "c1ccccc1"
# Caffeine defeats BRICS but not recap/mmpa/hr, so it exercises the slicer cascade.
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


@pytest.fixture
def client():
    return TestClient(app)


class TestSafeMolInstall:
    def test_importing_safe_does_not_pull_in_torch(self):
        import safe as sf

        assert callable(sf.encode)
        assert callable(sf.decode)
        assert "torch" not in sys.modules

    def test_a_deferred_attribute_reports_its_missing_dependency(self):
        # Lazy must mean deferred, not absent: an AttributeError here would mean
        # the shim dropped part of the upstream API rather than deferring it.
        import safe as sf

        try:
            sf.SAFETokenizer
        except ImportError:
            pass
        except AttributeError as e:
            pytest.fail(f"lazy attribute lost from the safe shim: {e}")

    def test_an_unknown_attribute_still_raises_attribute_error(self):
        import safe as sf

        with pytest.raises(AttributeError):
            sf.no_such_thing  # noqa: B018

    def test_rdkit_is_the_source_build_not_a_pip_wheel(self):
        # The dist-info stub in the Dockerfile exists to stop pip installing a
        # second RDKit over the source build. If this path moves out of
        # /opt/rdkit, the stub is lying and the wheel won.
        import rdkit

        assert "/opt/rdkit" in rdkit.__file__


class TestSmilesToSafeFunction:
    def test_produces_multiple_fragments(self):
        result, error, fragmented, _ = smiles_to_safe(IBUPROFEN)
        assert error is None
        assert fragmented is True
        assert len(result.split(".")) > 1

    def test_round_trips_back_to_the_same_molecule(self):
        encoded, _, _, _ = smiles_to_safe(IBUPROFEN)
        decoded, error = safe_to_smiles(encoded)
        assert error is None
        assert Chem.CanonSmiles(decoded) == Chem.CanonSmiles(IBUPROFEN)

    def test_an_unfragmentable_molecule_reports_fragmented_false(self):
        result, error, fragmented, _ = smiles_to_safe(BENZENE)
        assert error is None
        assert fragmented is False
        assert Chem.CanonSmiles(result) == Chem.CanonSmiles(BENZENE)

    def test_an_invalid_smiles_is_an_error(self):
        result, error, _, _ = smiles_to_safe("not-a-molecule")
        assert result is None
        assert "Invalid SMILES" in error

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_smiles_is_an_error(self, value):
        result, error, _, _ = smiles_to_safe(value)
        assert result is None
        assert "empty" in error

    def test_an_unsupported_slicer_is_an_error(self):
        result, error, _, _ = smiles_to_safe(IBUPROFEN, slicer="nonsense")
        assert result is None
        assert "Unsupported slicer" in error

    @pytest.mark.parametrize("slicer", ["brics", "hr", "recap", "attach"])
    def test_the_supported_slicers_all_work(self, slicer):
        result, error, _, _ = smiles_to_safe(IBUPROFEN, slicer=slicer)
        assert error is None
        assert result


class TestSlicerCascade:
    """BRICS finds no cut points in caffeine, which is why generation silently
    degraded to echoing the input back. Another slicer has to be tried."""

    def test_caffeine_is_fragmented_even_though_brics_cannot(self):
        result, error, fragmented, slicer_used = smiles_to_safe(CAFFEINE)
        assert error is None
        assert fragmented is True
        assert slicer_used != "brics"
        assert len(result.split(".")) > 1

    def test_the_fragments_still_describe_caffeine(self):
        result, _, _, _ = smiles_to_safe(CAFFEINE)
        decoded, error = safe_to_smiles(result)
        assert error is None
        assert Chem.CanonSmiles(decoded) == Chem.CanonSmiles(CAFFEINE)

    def test_the_requested_slicer_wins_when_it_works(self):
        _, error, _, slicer_used = smiles_to_safe(IBUPROFEN, slicer="brics")
        assert error is None
        assert slicer_used == "brics"

    def test_no_slicer_helps_a_single_ring(self):
        _, error, fragmented, _ = smiles_to_safe(BENZENE)
        assert error is None
        assert fragmented is False

    def test_explicit_hydrogen_fragments_are_never_returned(self):
        # The 'attach' slicer cuts every C-H bond on a molecule with no [*] dummy
        # atoms, emitting each hydrogen separately. GenMol answers that with junk.
        result, error, _, _ = smiles_to_safe(CAFFEINE, slicer="attach")
        assert error is None
        assert "[H]" not in result


class TestSafeToSmilesFunction:
    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_string_is_an_error(self, value):
        result, error = safe_to_smiles(value)
        assert result is None
        assert "empty" in error

    def test_an_undecodable_string_is_an_error(self):
        result, error = safe_to_smiles("this is not safe notation")
        assert result is None
        assert error


class TestApplyMask:
    """The mask has to *replace* the terminal fragment. Appending it to a complete
    molecule leaves it bonded to nothing, and GenMol answers such an input by
    returning the input molecule unchanged."""

    def test_the_mask_replaces_the_terminal_fragment(self):
        result, replaced, error = apply_mask("core12.C1.C2", 10, 15)
        assert error is None
        assert result == "core12.C1.[*{10-15}]"
        assert replaced == "C2"

    def test_the_masked_string_has_an_open_attachment_point(self):
        safe_str, _, _, _ = smiles_to_safe(CAFFEINE)
        result, _, error = apply_mask(safe_str, 5, 12)
        assert error is None
        # A dangling ring-bond index means this no longer parses as a molecule --
        # that dangling index is precisely what the generated fragment bonds to.
        assert Chem.MolFromSmiles(result) is None

    def test_masking_a_single_fragment_scaffold_is_an_error(self):
        result, replaced, error = apply_mask("c1ccccc1", 10, 15)
        assert result is None
        assert replaced is None
        assert "no attachment point" in error

    def test_equal_bounds_are_allowed(self):
        result, _, error = apply_mask("core12.C1.C2", 8, 8)
        assert error is None
        assert result.endswith(".[*{8-8}]")

    @pytest.mark.parametrize("frag_min", [0, -1])
    def test_a_non_positive_minimum_is_an_error(self, frag_min):
        result, _, error = apply_mask("core12.C1.C2", frag_min, 10)
        assert result is None
        assert "frag_min" in error

    def test_an_inverted_range_is_an_error(self):
        result, _, error = apply_mask("core12.C1.C2", 15, 10)
        assert result is None
        assert "frag_max" in error


class TestSmilesToSafeEndpoint:
    def test_encodes_a_molecule(self, client):
        response = client.post("/smiles-to-safe", json={"smiles": IBUPROFEN})
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert "." in body["safe"]
        assert body["num_fragments"] == len(body["safe"].split("."))
        assert body["masked"] is False

    def test_defaults_to_brics(self, client):
        explicit = client.post(
            "/smiles-to-safe", json={"smiles": IBUPROFEN, "slicer": DEFAULT_SLICER}
        ).json()
        implicit = client.post("/smiles-to-safe", json={"smiles": IBUPROFEN}).json()
        assert implicit["safe"] == explicit["safe"]

    def test_applies_a_generation_mask(self, client):
        response = client.post(
            "/smiles-to-safe",
            json={"smiles": IBUPROFEN, "mask_min": 10, "mask_max": 15},
        )
        body = response.json()
        assert body["success"] is True
        assert body["safe"].endswith(".[*{10-15}]")
        assert body["masked"] is True
        assert body["replaced_fragment"]

    def test_the_mask_takes_the_place_of_a_fragment(self, client):
        unmasked = client.post("/smiles-to-safe", json={"smiles": IBUPROFEN}).json()
        masked = client.post(
            "/smiles-to-safe",
            json={"smiles": IBUPROFEN, "mask_min": 5, "mask_max": 10},
        ).json()
        # One real fragment gave way to the mask, and the mask is not counted.
        assert masked["num_fragments"] == unmasked["num_fragments"] - 1
        assert masked["replaced_fragment"] == unmasked["safe"].split(".")[-1]

    def test_masking_an_unfragmentable_molecule_is_rejected(self, client):
        body = client.post(
            "/smiles-to-safe",
            json={"smiles": BENZENE, "mask_min": 5, "mask_max": 10},
        ).json()
        assert body["success"] is False
        assert "no attachment point" in body["error"]

    def test_reports_which_slicer_actually_ran(self, client):
        body = client.post("/smiles-to-safe", json={"smiles": CAFFEINE}).json()
        assert body["success"] is True
        assert body["fragmented"] is True
        assert body["slicer_used"] != "brics"

    def test_one_mask_bound_alone_is_rejected(self, client):
        body = client.post(
            "/smiles-to-safe", json={"smiles": IBUPROFEN, "mask_min": 10}
        ).json()
        assert body["success"] is False
        assert "together" in body["error"]

    def test_an_unfragmentable_molecule_still_succeeds(self, client):
        body = client.post("/smiles-to-safe", json={"smiles": BENZENE}).json()
        assert body["success"] is True
        assert body["fragmented"] is False

    def test_an_invalid_smiles_is_reported_as_a_failure(self, client):
        body = client.post("/smiles-to-safe", json={"smiles": "xyz123!!"}).json()
        assert body["success"] is False
        assert body["error"]

    def test_an_unsupported_slicer_is_reported_as_a_failure(self, client):
        body = client.post(
            "/smiles-to-safe", json={"smiles": IBUPROFEN, "slicer": "nonsense"}
        ).json()
        assert body["success"] is False
        assert "Unsupported slicer" in body["error"]


class TestSafeToSmilesEndpoint:
    def test_decodes_back_to_the_original_molecule(self, client):
        encoded = client.post("/smiles-to-safe", json={"smiles": IBUPROFEN}).json()[
            "safe"
        ]
        body = client.post("/safe-to-smiles", json={"safe": encoded}).json()
        assert body["success"] is True
        assert Chem.CanonSmiles(body["smiles"]) == Chem.CanonSmiles(IBUPROFEN)

    def test_a_plain_smiles_decodes_unchanged(self, client):
        # A SAFE string is valid SMILES and the converse holds for a single
        # fragment, so this must not be treated as malformed input.
        body = client.post("/safe-to-smiles", json={"safe": BENZENE}).json()
        assert body["success"] is True
        assert Chem.CanonSmiles(body["smiles"]) == Chem.CanonSmiles(BENZENE)

    def test_garbage_is_reported_as_a_failure(self, client):
        body = client.post(
            "/safe-to-smiles", json={"safe": "this is not safe notation"}
        ).json()
        assert body["success"] is False
        assert body["error"]

    def test_an_empty_string_is_reported_as_a_failure(self, client):
        body = client.post("/safe-to-smiles", json={"safe": "   "}).json()
        assert body["success"] is False
        assert "empty" in body["error"]


class TestServiceInfo:
    def test_health_still_responds(self, client):
        assert client.get("/health").json()["status"] == "healthy"
