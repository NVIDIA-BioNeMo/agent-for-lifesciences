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

"""Tests for the training-only vina_dock result cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from api.training.vina_cache import (
    VinaCacheEntry,
    build_cached_vina_result,
    build_vina_cache,
)


# Locate the aevom-data-gen dataset if present. Tests that need real fixtures
# are skipped when the dataset isn't available (e.g. CI / isolated dev boxes).
DATASET_BASE = Path(
    os.environ.get(
        "AEVOM_DATASET_BASE_PATH",
        "/workspaces/aevom-data-gen/data/output/pdb_v27",
    )
)
COMPLEX_0001 = DATASET_BASE / "complex_0001"

_has_dataset = COMPLEX_0001.is_dir() and (COMPLEX_0001 / "metadata.json").exists()
requires_dataset = pytest.mark.skipif(
    not _has_dataset,
    reason=f"pdb_v27 dataset not available at {DATASET_BASE}",
)


# ---------------------------------------------------------------------------
# build_vina_cache
# ---------------------------------------------------------------------------


@requires_dataset
class TestBuildVinaCache:
    """Verify cache construction from the real pdb_v27 complex_0001 fixture."""

    def _preload_refs(self) -> dict[str, str]:
        meta = json.loads((COMPLEX_0001 / "metadata.json").read_text())
        refs: dict[str, str] = {
            "sdf_winner": "complex_0001/winner.sdf",
            "pdb_receptor": "complex_0001/receptor.pdb",
        }
        for i, pair in enumerate(meta.get("pairs", [])):
            loser_rel = pair["loser_filename"]
            refs[f"sdf_loser_{i}"] = f"complex_0001/{loser_rel}"
        return refs

    def test_build_cache_from_preload_refs(self):
        preload_refs = self._preload_refs()
        cache = build_vina_cache(preload_refs, dataset_base_path=DATASET_BASE)

        # Winner indexed with correct affinity
        assert "sdf_winner" in cache
        meta = json.loads((COMPLEX_0001 / "metadata.json").read_text())
        expected_winner_affinity = meta["winner_metrics"]["vina_score"]
        assert cache["sdf_winner"].cached_affinity == pytest.approx(
            expected_winner_affinity
        )
        assert cache["sdf_winner"].docked_pdbqt_path == COMPLEX_0001 / "docked.pdbqt"

        # All four losers indexed with correct affinities
        for i, pair in enumerate(meta["pairs"]):
            ref_name = f"sdf_loser_{i}"
            assert ref_name in cache, f"missing {ref_name}"
            assert cache[ref_name].cached_affinity == pytest.approx(
                pair["metrics"]["vina_score"]
            )
            assert cache[ref_name].docked_pdbqt_path.exists()

        # PDB receptor ref isn't cached (not an SDF)
        assert "pdb_receptor" not in cache

    def test_build_cache_missing_docked_file(self, tmp_path: Path, monkeypatch):
        """Missing docked.pdbqt -> entry silently omitted, not raised."""
        fake_base = tmp_path / "pdb"
        complex_dir = fake_base / "complex_9999"
        (complex_dir / "losers").mkdir(parents=True)

        (complex_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "winner_metrics": {"vina_score": -8.1},
                    "pairs": [
                        {
                            "loser_filename": "losers/loser_01_steric_clash.sdf",
                            "metrics": {"vina_score": -5.5},
                        }
                    ],
                }
            )
        )
        (complex_dir / "winner.sdf").write_text("fake winner sdf")
        (complex_dir / "losers" / "loser_01_steric_clash.sdf").write_text(
            "fake loser sdf"
        )

        preload_refs = {
            "sdf_winner": "complex_9999/winner.sdf",
            "sdf_loser_1": "complex_9999/losers/loser_01_steric_clash.sdf",
        }

        cache = build_vina_cache(preload_refs, dataset_base_path=fake_base)
        assert cache == {}

    def test_build_cache_partial_docked_files(self, tmp_path: Path):
        """Winner present, loser missing -> winner indexed, loser omitted."""
        fake_base = tmp_path / "pdb"
        complex_dir = fake_base / "complex_7777"
        (complex_dir / "losers").mkdir(parents=True)

        (complex_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "winner_metrics": {"vina_score": -7.0},
                    "pairs": [
                        {
                            "loser_filename": "losers/loser_01_clash.sdf",
                            "metrics": {"vina_score": -4.0},
                        }
                    ],
                }
            )
        )
        (complex_dir / "winner.sdf").write_text("x")
        (complex_dir / "docked.pdbqt").write_text("pdbqt winner")
        (complex_dir / "losers" / "loser_01_clash.sdf").write_text("y")
        # No loser_01_docked.pdbqt

        preload_refs = {
            "sdf_winner": "complex_7777/winner.sdf",
            "sdf_loser_1": "complex_7777/losers/loser_01_clash.sdf",
        }
        cache = build_vina_cache(preload_refs, dataset_base_path=fake_base)

        assert set(cache.keys()) == {"sdf_winner"}
        assert cache["sdf_winner"].cached_affinity == pytest.approx(-7.0)

    def test_build_cache_ignores_non_sdf_refs(self, tmp_path: Path):
        cache = build_vina_cache(
            {"pdb_receptor": "any/receptor.pdb"},
            dataset_base_path=tmp_path,
        )
        assert cache == {}


# ---------------------------------------------------------------------------
# build_cached_vina_result
# ---------------------------------------------------------------------------


@requires_dataset
class TestBuildCachedVinaResult:
    """Verify the cache-hit ToolResult matches production shape."""

    @pytest.fixture
    def entry(self) -> VinaCacheEntry:
        meta = json.loads((COMPLEX_0001 / "metadata.json").read_text())
        return VinaCacheEntry(
            cached_affinity=meta["winner_metrics"]["vina_score"],
            docked_pdbqt_path=COMPLEX_0001 / "docked.pdbqt",
            source_sdf_path=COMPLEX_0001 / "winner.sdf",
        )

    @pytest.fixture
    def mock_storage(self):
        receptor_pdb = (COMPLEX_0001 / "receptor.pdb").read_bytes()
        storage = MagicMock()
        storage.retrieve = AsyncMock(return_value=receptor_pdb)
        storage.store = AsyncMock(return_value="pdb_cachedresult")
        return storage

    @pytest.mark.asyncio
    async def test_result_shape_matches_production(
        self, entry: VinaCacheEntry, mock_storage
    ):
        arguments = {
            "receptor_pdb_ref": "pdb_receptor",
            "ligand_sdf_ref": "sdf_winner",
            "center_x": 10.0,
            "center_y": 20.0,
            "center_z": 30.0,
            "size_x": 20.0,
            "size_y": 20.0,
            "size_z": 20.0,
            "exhaustiveness": 16,
        }

        result = await build_cached_vina_result(
            entry=entry, arguments=arguments, storage=mock_storage
        )

        from api.tools.base import ContentType

        assert result.content_type == ContentType.DATA_REFERENCE
        assert result.data == "pdb_cachedresult"
        assert result.metadata["data_type"] == "pdb"
        assert result.metadata["source"] == "vina_dock"
        assert result.metadata["is_complex"] is True
        assert result.metadata["cached"] is True
        assert result.metadata["affinity_kcal_mol"] == pytest.approx(
            entry.cached_affinity
        )
        assert result.metadata["num_poses"] == 1
        assert result.metadata["ligand_chain"] == "Z"
        assert result.metadata["center"] == [10.0, 20.0, 30.0]
        assert result.metadata["size"] == [20.0, 20.0, 20.0]
        assert result.metadata["exhaustiveness"] == 16
        assert result.metadata["receptor_ref"] == "pdb_receptor"
        assert result.metadata["ligand_ref"] == "sdf_winner"
        assert result.metadata["docking_mode"] == "cached"

        msg = result.metadata["message"]
        assert "Best binding affinity" in msg
        assert f"{entry.cached_affinity:.2f} kcal/mol" in msg
        assert "Number of poses found: 1" in msg
        assert "Ligand residue name in complex" in msg
        # One of the three interpretation bands must be present
        interpretations = (
            "Weak binding",
            "Moderate binding",
            "Strong binding",
        )
        assert any(b in msg for b in interpretations)

        # Complex PDB was stored
        mock_storage.store.assert_awaited_once()
        store_kwargs = mock_storage.store.await_args.kwargs
        assert store_kwargs["data_type"] == "pdb"
        pdb_content = store_kwargs["content"].decode("utf-8")
        assert "ATOM" in pdb_content
        assert "HETATM" in pdb_content

    @pytest.mark.asyncio
    async def test_retrieve_failure_propagates_for_fallback(
        self, entry: VinaCacheEntry
    ):
        """Storage errors must raise so the caller can fall back to real Vina."""
        storage = MagicMock()
        storage.retrieve = AsyncMock(side_effect=KeyError("missing"))
        storage.store = AsyncMock()

        with pytest.raises(KeyError):
            await build_cached_vina_result(
                entry=entry,
                arguments={"receptor_pdb_ref": "pdb_missing"},
                storage=storage,
            )
        storage.store.assert_not_called()
