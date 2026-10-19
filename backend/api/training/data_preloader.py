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

"""Data preloader for training sessions.

This module handles loading molecules and receptors from disk into
the storage system at the start of each training episode.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class PreloadResult:
    """Result of preloading data for a training session."""
    
    success: bool
    loaded_refs: dict[str, str]  # Requested ref -> actual storage ref
    errors: list[str]
    metadata: dict[str, Any]


class TrainingDataPreloader:
    """Preloads training data into storage for episodes.
    
    This class handles:
    - Loading SDF files into molecule storage
    - Loading PDBQT receptor files
    - Mapping requested reference IDs to actual storage references
    
    Example usage:
        preloader = TrainingDataPreloader(
            storage_service=storage,
            dataset_base_path="/data/dataset_cdd_v1",
        )
        
        # At session start
        result = await preloader.preload_session(
            preload_refs={
                "sdf_winner_001": "complex_001/winner.sdf",
                "sdf_loser_001": "complex_001/losers/loser_01_steric_clash.sdf",
                "receptor_001": "complex_001/receptor.pdbqt",
            }
        )
        
        # Use loaded references
        winner_ref = result.loaded_refs["sdf_winner_001"]
    """
    
    def __init__(
        self,
        storage_service: Any,  # DataStorageService
        dataset_base_path: Optional[str | Path] = None,
    ):
        """Initialize the preloader.
        
        Args:
            storage_service: The DataStorageService for storing data.
            dataset_base_path: Base path for resolving relative paths.
        """
        self.storage = storage_service
        self.dataset_base_path = Path(dataset_base_path) if dataset_base_path else None
    
    def _resolve_path(self, path: str) -> Path:
        """Resolve a path, potentially relative to dataset base."""
        path_obj = Path(path)
        
        if path_obj.is_absolute():
            return path_obj
        
        if self.dataset_base_path:
            return self.dataset_base_path / path
        
        return path_obj
    
    def _detect_data_type(self, path: Path) -> str:
        """Detect data type from file extension."""
        suffix = path.suffix.lower()
        
        if suffix in (".sdf", ".mol"):
            return "sdf"
        elif suffix == ".pdbqt":
            return "pdbqt"
        elif suffix == ".pdb":
            return "pdb"
        elif suffix == ".mol2":
            return "mol2"
        elif suffix == ".smiles":
            return "smiles"
        elif suffix == ".fasta":
            return "sequence"
        else:
            return "unknown"
    
    async def preload_file(
        self,
        ref_name: str,
        file_path: str,
        session_id: Optional[str] = None,
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Preload a single file into storage.
        
        Args:
            ref_name: Requested reference name.
            file_path: Path to the file.
            session_id: Optional session ID for scoping.
            
        Returns:
            Tuple of (success, storage_ref, error_message).
        """
        resolved_path = self._resolve_path(file_path)
        
        if not resolved_path.exists():
            return False, None, f"File not found: {resolved_path}"
        
        try:
            # Read file content
            content = resolved_path.read_bytes()
            data_type = self._detect_data_type(resolved_path)
            
            # Determine storage method based on type
            if data_type == "sdf":
                # Store as SDF
                storage_ref = await self.storage.store(
                    data_type="sdf",
                    content=content,
                    metadata={
                        "source_file": str(resolved_path),
                        "requested_ref": ref_name,
                    },
                    session_id=session_id,
                )
            elif data_type in ("pdb", "pdbqt"):
                # Store as PDB/PDBQT
                storage_ref = await self.storage.store(
                    data_type=data_type,
                    content=content,
                    metadata={
                        "source_file": str(resolved_path),
                        "requested_ref": ref_name,
                    },
                    session_id=session_id,
                )
            else:
                # Generic storage
                storage_ref = await self.storage.store(
                    data_type=data_type,
                    content=content,
                    metadata={
                        "source_file": str(resolved_path),
                        "requested_ref": ref_name,
                    },
                    session_id=session_id,
                )
            
            logger.debug(f"Preloaded {ref_name} -> {storage_ref}")
            return True, storage_ref, None
            
        except Exception as e:
            logger.error(f"Failed to preload {ref_name}: {e}")
            return False, None, str(e)
    
    async def preload_session(
        self,
        preload_refs: dict[str, str],
        session_id: Optional[str] = None,
    ) -> PreloadResult:
        """Preload all data for a training session.
        
        Args:
            preload_refs: Mapping of reference names to file paths.
            session_id: Optional session ID for scoping.
            
        Returns:
            PreloadResult with loaded references.
        """
        loaded_refs = {}
        errors = []
        
        for ref_name, file_path in preload_refs.items():
            success, storage_ref, error = await self.preload_file(
                ref_name=ref_name,
                file_path=file_path,
                session_id=session_id,
            )
            
            if success and storage_ref:
                loaded_refs[ref_name] = storage_ref
            else:
                errors.append(f"{ref_name}: {error}")
        
        return PreloadResult(
            success=len(errors) == 0,
            loaded_refs=loaded_refs,
            errors=errors,
            metadata={
                "total_requested": len(preload_refs),
                "total_loaded": len(loaded_refs),
                "session_id": session_id,
            },
        )
    
    async def preload_from_task(
        self,
        task: dict,
        session_id: Optional[str] = None,
    ) -> PreloadResult:
        """Preload data from a training task dictionary.
        
        Args:
            task: Task dictionary with preload_refs field.
            session_id: Optional session ID.
            
        Returns:
            PreloadResult with loaded references.
        """
        preload_refs = task.get("preload_refs", {})
        return await self.preload_session(preload_refs, session_id)


class MockStorageService:
    """Mock storage service for testing without database."""
    
    def __init__(self):
        self._counter = 0
        self._storage: dict[str, bytes] = {}
    
    async def store(
        self,
        data_type: str,
        content: bytes,
        metadata: Optional[dict] = None,
        session_id: Optional[str] = None,
    ) -> str:
        """Store data and return reference."""
        self._counter += 1
        ref = f"{data_type}_{self._counter:08x}"
        self._storage[ref] = content
        return ref
    
    async def retrieve(self, ref: str) -> Optional[bytes]:
        """Retrieve data by reference."""
        return self._storage.get(ref)


def load_training_task(task_file: str | Path, task_idx: int = 0) -> dict:
    """Load a single training task from a JSONL file.
    
    Args:
        task_file: Path to JSONL file.
        task_idx: Index of task to load (0-based).
        
    Returns:
        Task dictionary.
    """
    task_file = Path(task_file)
    
    with open(task_file) as f:
        for idx, line in enumerate(f):
            if idx == task_idx:
                return json.loads(line)
    
    raise ValueError(f"Task index {task_idx} not found in {task_file}")


def iter_training_tasks(task_file: str | Path) -> Any:
    """Iterate over training tasks from a JSONL file.
    
    Args:
        task_file: Path to JSONL file.
        
    Yields:
        Task dictionaries.
    """
    task_file = Path(task_file)
    
    with open(task_file) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


async def preload_task_data(
    task: dict,
    storage_service: Any,
    dataset_base_path: Optional[str] = None,
    session_id: Optional[str] = None,
) -> PreloadResult:
    """Convenience function to preload data for a task.
    
    Args:
        task: Training task dictionary.
        storage_service: Storage service instance.
        dataset_base_path: Base path for dataset files.
        session_id: Optional session ID.
        
    Returns:
        PreloadResult with loaded references.
    """
    preloader = TrainingDataPreloader(
        storage_service=storage_service,
        dataset_base_path=dataset_base_path,
    )
    
    return await preloader.preload_from_task(task, session_id)
