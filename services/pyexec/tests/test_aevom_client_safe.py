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

"""Tests for SAFE notation handling in the sandbox's aevom client.

SAFE is the one storage type whose payload is indistinguishable from another
type's: a SAFE string is valid SMILES. So the tests worth having are the ones
that pin the *type* boundary -- that `store_safe` writes `safe` and not `smiles`,
that `load()` decodes a `safe_*` reference as text -- plus the fragmentation
fallback, since a molecule with no cut points is a normal input, not an error.

The conversion helpers need `safe-mol`, which lives in the RDKit base image, so
those tests skip outside the container.
"""

import base64

import pytest

from app.aevom_client import AevomClient, AevomError, StorageError

safe_lib = pytest.importorskip("safe", reason="safe-mol lives in the RDKit base image")

# Ibuprofen: several BRICS cut points, so it encodes to multiple fragments.
IBUPROFEN = "CC(Cc1ccc(cc1)C(C(=O)O)C)C"
# Benzene has no cut points under any slicer -- the fragmentation fallback case.
BENZENE = "c1ccccc1"
# Caffeine defeats BRICS but not recap/mmpa/hr, so it exercises the slicer cascade.
CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"


class FakeTransport:
    """Stands in for the backend's /api/sandbox endpoints."""

    def __init__(self, artifacts=None):
        self.artifacts = dict(artifacts or {})
        self.stored: list[dict] = []
        self._counter = 0

    def fetch(self, reference):
        return self.artifacts[reference]

    def store(self, payload):
        self._counter += 1
        reference = f"{payload['data_type']}_stored{self._counter}"
        self.stored.append(payload)
        if "content_base64" in payload:
            content = base64.b64decode(payload["content_base64"])
        else:
            content = payload["content"].encode("utf-8")
        self.artifacts[reference] = content
        return {"reference": reference}


@pytest.fixture
def client(monkeypatch):
    """An AevomClient whose HTTP calls are redirected to a fake transport."""
    transport = FakeTransport()
    instance = AevomClient(session_token="tok", backend_url="http://backend:8000")

    monkeypatch.setattr(
        AevomClient, "_fetch_data", lambda self, ref: transport.fetch(ref)
    )

    def _store(self, data_type, content, metadata=None, description=None):
        payload = {"data_type": data_type, "metadata": metadata or {}}
        if isinstance(content, (bytes, bytearray)):
            payload["content_base64"] = base64.b64encode(bytes(content)).decode("ascii")
        else:
            payload["content"] = content
        if description:
            payload["description"] = description
        return transport.store(payload)["reference"]

    monkeypatch.setattr(AevomClient, "store", _store)
    instance.transport = transport
    return instance


class TestStoreSafe:
    def test_returns_a_safe_reference(self, client):
        reference = client.store_safe("c1ccccc1.CC(=O)O")
        assert reference.startswith("safe_")

    def test_stores_under_the_safe_type_not_smiles(self, client):
        # The whole point of the separate type: a SAFE string would be accepted
        # as `smiles` without complaint, and downstream consumers could not tell.
        client.store_safe("c1ccccc1.CC(=O)O")
        assert client.transport.stored[-1]["data_type"] == "safe"

    def test_sends_content_as_text(self, client):
        client.store_safe("c1ccccc1.CC(=O)O")
        payload = client.transport.stored[-1]
        assert payload["content"] == "c1ccccc1.CC(=O)O"
        assert "content_base64" not in payload

    def test_strips_surrounding_whitespace(self, client):
        client.store_safe("  c1ccccc1.CC(=O)O\n")
        assert client.transport.stored[-1]["content"] == "c1ccccc1.CC(=O)O"

    def test_preserves_metadata_and_description(self, client):
        client.store_safe(
            "c1ccccc1",
            metadata={"slicer": "brics", "name": "benzene"},
            description="Benzene in SAFE",
        )
        payload = client.transport.stored[-1]
        assert payload["metadata"]["slicer"] == "brics"
        assert payload["description"] == "Benzene in SAFE"

    def test_keeps_a_generation_mask_intact(self, client):
        # `[*{10-15}]` is what makes the string a GenMol input rather than just a
        # molecule; mangling it would be silently accepted by the model.
        client.store_safe("C1CC(=O)NC1.[*{10-15}]")
        assert client.transport.stored[-1]["content"] == "C1CC(=O)NC1.[*{10-15}]"

    @pytest.mark.parametrize("value", ["", "   ", "\n"])
    def test_an_empty_string_is_refused(self, client, value):
        with pytest.raises(StorageError, match="cannot be empty"):
            client.store_safe(value)


class TestLoadSafe:
    def test_returns_the_stored_string(self, client):
        client.transport.artifacts["safe_abc"] = b"c1ccccc1.CC(=O)O"
        assert client.load_safe("safe_abc") == "c1ccccc1.CC(=O)O"

    def test_generic_load_decodes_a_safe_reference_as_text(self, client):
        # Without "safe" in load()'s text-type tuple this would fall through to
        # the binary branch and hand the user bytes.
        client.transport.artifacts["safe_abc"] = b"C1CC(=O)NC1.[*{10-15}]"
        loaded = client.load("safe_abc")
        assert isinstance(loaded, str)
        assert loaded == "C1CC(=O)NC1.[*{10-15}]"

    def test_round_trips_through_the_store(self, client):
        reference = client.store_safe("c1ccccc1.CC(=O)O")
        assert client.load_safe(reference) == "c1ccccc1.CC(=O)O"


class TestSafeFromSmiles:
    def test_produces_multiple_fragments(self, client):
        result = client.safe_from_smiles(IBUPROFEN)
        assert "." in result
        assert len(result.split(".")) > 1

    def test_the_result_decodes_back_to_the_same_molecule(self, client):
        from rdkit import Chem

        encoded = client.safe_from_smiles(IBUPROFEN)
        assert Chem.CanonSmiles(client.smiles_from_safe(encoded)) == Chem.CanonSmiles(
            IBUPROFEN
        )

    def test_an_unfragmentable_molecule_falls_back_to_canonical_smiles(self, client):
        from rdkit import Chem

        # No slicer finds a bond to cut in benzene. A single-fragment string is
        # still valid for round-tripping, so this must not surface as a failure.
        result = client.safe_from_smiles(BENZENE)
        assert "." not in result
        assert Chem.CanonSmiles(result) == Chem.CanonSmiles(BENZENE)

    def test_another_slicer_is_tried_when_brics_finds_no_cut_points(self, client):
        from rdkit import Chem

        # BRICS cannot cut caffeine, but recap/mmpa/hr can. Silently returning a
        # single fragment here is what made GenMol echo its input back.
        result = client.safe_from_smiles(CAFFEINE)
        assert len(result.split(".")) > 1
        assert Chem.CanonSmiles(client.smiles_from_safe(result)) == Chem.CanonSmiles(
            CAFFEINE
        )

    def test_explicit_hydrogen_fragments_are_never_returned(self, client):
        # 'attach' cuts every C-H bond when there are no [*] dummy atoms.
        assert "[H]" not in client.safe_from_smiles(CAFFEINE, slicer="attach")

    def test_an_invalid_smiles_raises(self, client):
        with pytest.raises(AevomError):
            client.safe_from_smiles("not-a-molecule")

    def test_an_empty_smiles_raises(self, client):
        with pytest.raises(AevomError, match="cannot be empty"):
            client.safe_from_smiles("")

    def test_an_unknown_slicer_raises(self, client):
        with pytest.raises(AevomError, match="Unknown slicer"):
            client.safe_from_smiles(IBUPROFEN, slicer="nonsense")


class TestSmilesFromSafe:
    def test_decodes_a_fragment_sequence(self, client):
        from rdkit import Chem

        decoded = client.smiles_from_safe(client.safe_from_smiles(IBUPROFEN))
        assert Chem.MolFromSmiles(decoded) is not None

    def test_an_empty_string_raises(self, client):
        with pytest.raises(AevomError, match="cannot be empty"):
            client.smiles_from_safe("")

    def test_an_undecodable_string_raises(self, client):
        with pytest.raises(AevomError):
            client.smiles_from_safe("this is not safe notation")


class TestScaffoldToSafe:
    def test_the_mask_ends_the_string(self, client):
        assert client.scaffold_to_safe(CAFFEINE, 10, 15).endswith(".[*{10-15}]")

    def test_the_mask_replaces_a_fragment_rather_than_joining_them(self, client):
        plain = client.safe_from_smiles(CAFFEINE)
        masked = client.scaffold_to_safe(CAFFEINE, 5, 12)
        # Same number of dot-separated parts: one real fragment made way for the
        # mask. Appending instead would leave the mask bonded to nothing, and
        # GenMol answers that by returning the input molecule unchanged.
        assert len(masked.split(".")) == len(plain.split("."))
        assert masked.split(".")[:-1] == plain.split(".")[:-1]

    def test_the_masked_string_has_an_open_attachment_point(self, client):
        from rdkit import Chem

        masked = client.scaffold_to_safe(CAFFEINE, 5, 12)
        # A dangling ring-bond index no longer parses as a molecule, and that
        # dangling index is exactly what the generated fragment bonds to.
        assert Chem.MolFromSmiles(masked) is None

    def test_an_unfragmentable_scaffold_raises(self, client):
        with pytest.raises(AevomError, match="no attachment point"):
            client.scaffold_to_safe(BENZENE, 10, 15)

    def test_equal_bounds_are_allowed(self, client):
        assert client.scaffold_to_safe(CAFFEINE, 8, 8).endswith(".[*{8-8}]")

    @pytest.mark.parametrize("frag_min", [0, -1])
    def test_a_non_positive_minimum_raises(self, client, frag_min):
        with pytest.raises(AevomError, match="frag_min"):
            client.scaffold_to_safe(CAFFEINE, frag_min, 10)

    def test_an_inverted_range_raises(self, client):
        with pytest.raises(AevomError, match="frag_max"):
            client.scaffold_to_safe(CAFFEINE, 15, 10)

    def test_the_result_is_storable_as_safe(self, client):
        reference = client.store_safe(client.scaffold_to_safe(IBUPROFEN, 10, 15))
        assert client.transport.stored[-1]["data_type"] == "safe"
        assert client.load_safe(reference).endswith(".[*{10-15}]")


class TestSafeImportIsTorchFree:
    def test_importing_safe_does_not_pull_in_torch(self):
        # The RDKit image replaces safe/__init__.py with a lazy one precisely so
        # `import safe` costs nothing. If upstream's eager __init__ ever comes
        # back, the image grows by gigabytes and cold start regresses -- worth
        # failing a test over rather than noticing in a deploy.
        import sys

        assert "safe" in sys.modules
        assert "torch" not in sys.modules

    def test_the_conversion_api_is_present(self):
        assert callable(safe_lib.encode)
        assert callable(safe_lib.decode)
        assert safe_lib.SAFEFragmentationError is not None

    def test_a_deferred_attribute_reports_its_missing_dependency(self):
        # Lazy must mean deferred, not absent. Asking for SAFETokenizer should
        # either work (if torch is somehow present) or name the missing
        # dependency -- an AttributeError would wrongly suggest the shim dropped
        # the API rather than deferring it.
        try:
            safe_lib.SAFETokenizer
        except ImportError:
            pass
        except AttributeError as e:
            pytest.fail(f"lazy attribute lost from the safe shim: {e}")

    def test_an_unknown_attribute_still_raises_attribute_error(self):
        with pytest.raises(AttributeError):
            safe_lib.no_such_thing  # noqa: B018
