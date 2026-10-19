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

"""Viewer state tool for querying the current Molstar viewer state.

This tool enables the LLM to understand what is currently displayed in the
molecular viewer, allowing it to answer context-aware questions about
structures the user is viewing.

The viewer state is passed from the frontend with each chat request and
stored in Redis keyed by request_id for this tool to access. This allows
horizontal scaling across multiple pods.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .base import BaseTool, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)

# Redis key TTL for viewer state (5 minutes -- well beyond any request lifetime)
VIEWER_STATE_TTL = 300


@dataclass
class StructureInfo:
    """Information about a single loaded structure."""
    
    label: str = ""
    structure_type: str = ""  # "protein", "small_molecule", "complex"
    format: str = ""  # "pdb", "mmcif", "sdf", etc.

    # Source artifact reference (e.g. "pdb_abc12345") -- populated when the
    # structure was loaded via a ref-based tool such as ``visualize_pdb``.
    # Used by ``structure_query`` to validate that the requested target
    # structure is actually loaded in the viewer.
    pdb_ref: str = ""

    # Composition
    chain_ids: list[str] = field(default_factory=list)
    residue_count: int = 0
    atom_count: int = 0
    
    # Entity breakdown
    has_polymer: bool = False  # protein/nucleic acid
    has_ligand: bool = False  # non-polymer small molecules
    has_water: bool = False
    
    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ViewerState:
    """Complete state of the Molstar viewer.
    
    This is serialized from the frontend and passed with chat requests.
    """
    
    is_empty: bool = True
    structures: list[StructureInfo] = field(default_factory=list)
    
    # Current visual state
    has_selection: bool = False
    selection_info: Optional[str] = None
    
    def to_description(self, selections: Optional[list] = None) -> str:
        """Generate a human-readable description of the viewer state.
        
        Args:
            selections: Optional list of SelectionInfo objects to include
                       in the description.
        """
        if self.is_empty:
            return "The 3D viewer is currently empty. No structures are loaded."
        
        lines = []
        
        for i, struct in enumerate(self.structures):
            struct_desc = []
            
            # Main identifier
            if struct.label:
                struct_desc.append(f"**{struct.label}**")
            else:
                struct_desc.append(f"Structure {i + 1}")
            
            # Type
            if struct.structure_type:
                struct_desc.append(f"({struct.structure_type})")
            
            lines.append(" ".join(struct_desc))
            
            # Composition details
            details = []
            
            if struct.chain_ids:
                chain_str = ", ".join(struct.chain_ids)
                details.append(f"Chains: {chain_str}")
            
            if struct.residue_count > 0:
                details.append(f"{struct.residue_count} residues")
            
            if struct.atom_count > 0:
                details.append(f"{struct.atom_count} atoms")
            
            if details:
                lines.append("  - " + ", ".join(details))
            
            # Entity types
            entity_types = []
            if struct.has_polymer:
                entity_types.append("polymer (protein/nucleic acid)")
            if struct.has_ligand:
                entity_types.append("ligand/small molecule")
            if struct.has_water:
                entity_types.append("water")
            
            if entity_types:
                lines.append(f"  - Contains: {', '.join(entity_types)}")
        
        # Selection info
        if self.has_selection and self.selection_info:
            lines.append(f"\nCurrent selection: {self.selection_info}")
        
        # Available selections from registry
        if selections:
            lines.append("\n**Available Selections for Analysis:**")
            for sel in selections[:5]:  # Limit to 5
                lines.append(
                    f"  - `{sel.reference}`: {sel.atom_count} atoms, "
                    f"{sel.residue_count} residues"
                )
            if len(selections) > 5:
                lines.append(f"  ... and {len(selections) - 5} more")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for observation serialization.
        
        Returns:
            Dict representation suitable for RL training.
        """
        return {
            "is_empty": self.is_empty,
            "structures": [
                {
                    "label": s.label,
                    "structure_type": s.structure_type,
                    "format": s.format,
                    "pdb_ref": s.pdb_ref,
                    "chain_ids": s.chain_ids,
                    "residue_count": s.residue_count,
                    "atom_count": s.atom_count,
                    "has_polymer": s.has_polymer,
                    "has_ligand": s.has_ligand,
                    "has_water": s.has_water,
                }
                for s in self.structures
            ],
            "has_selection": self.has_selection,
            "selection_info": self.selection_info,
        }
    
    def to_condensed(self, selections: Optional[list] = None) -> str:
        """Generate a condensed one-line summary for system prompt injection.
        
        Args:
            selections: Optional list of SelectionInfo objects.
        
        Returns a compact representation like:
        - Empty: "[VIEWER STATE: Empty - no structures loaded]"
        - Protein: "[VIEWER STATE: PGK1 (protein) chains: A, 417 residues]"
        - Complex: "[VIEWER STATE: CDK2 (complex) chains: A,B, 298 residues, has ligand]"
        - With selections: "[VIEWER STATE: PGK1 (protein); SELECTIONS: selection_abc123 (24 atoms)]"
        """
        if self.is_empty:
            # Even if empty, check for available selections
            if selections:
                sel_parts = [s.to_brief() for s in selections[:3]]  # Limit to 3
                return f"[VIEWER STATE: Empty; AVAILABLE SELECTIONS: {', '.join(sel_parts)}]"
            return "[VIEWER STATE: Empty - no structures loaded]"
        
        parts = []
        for struct in self.structures:
            struct_parts = []
            
            # Label and type
            if struct.label:
                struct_parts.append(struct.label)
            if struct.structure_type:
                struct_parts.append(f"({struct.structure_type})")
            
            # Chains
            if struct.chain_ids:
                struct_parts.append(f"chains: {','.join(struct.chain_ids)}")
            
            # Residue count
            if struct.residue_count > 0:
                struct_parts.append(f"{struct.residue_count} residues")
            
            # Ligand indicator
            if struct.has_ligand:
                struct_parts.append("has ligand")
            
            parts.append(" ".join(struct_parts))
        
        result = f"[VIEWER STATE: {'; '.join(parts)}"
        
        # Add available selections
        if selections:
            sel_parts = [s.to_brief() for s in selections[:3]]  # Limit to 3
            result += f"; AVAILABLE SELECTIONS: {', '.join(sel_parts)}"
        
        result += "]"
        return result


def _redis_key(request_id: str) -> str:
    """Build Redis key for viewer state."""
    return f"viewer:{request_id}"


def _parse_state_data(state_data: dict[str, Any]) -> ViewerState:
    """Parse raw frontend state_data dict into a ViewerState object."""
    structures = []
    for struct_data in state_data.get("structures", []):
        structures.append(StructureInfo(
            label=struct_data.get("label", ""),
            structure_type=struct_data.get("structureType", struct_data.get("structure_type", "")),
            format=struct_data.get("format", ""),
            pdb_ref=struct_data.get("pdbRef", struct_data.get("pdb_ref", "")),
            chain_ids=struct_data.get("chainIds", struct_data.get("chain_ids", [])),
            residue_count=struct_data.get("residueCount", struct_data.get("residue_count", 0)),
            atom_count=struct_data.get("atomCount", struct_data.get("atom_count", 0)),
            has_polymer=struct_data.get("hasPolymer", struct_data.get("has_polymer", False)),
            has_ligand=struct_data.get("hasLigand", struct_data.get("has_ligand", False)),
            has_water=struct_data.get("hasWater", struct_data.get("has_water", False)),
            metadata=struct_data.get("metadata", {}),
        ))

    return ViewerState(
        is_empty=state_data.get("isEmpty", state_data.get("is_empty", True)),
        structures=structures,
        has_selection=state_data.get("hasSelection", state_data.get("has_selection", False)),
        selection_info=state_data.get("selectionInfo", state_data.get("selection_info")),
    )


async def set_viewer_state(request_id: str, state_data: Optional[dict[str, Any]]) -> None:
    """Set the current viewer state from frontend data.
    
    Called by the chat endpoint when receiving a request with viewer state.
    Stores the state in Redis keyed by request_id.
    
    Args:
        request_id: Unique identifier for the current request.
        state_data: Serialized viewer state from frontend, or None if not provided.
    """
    from api.redis import get_redis
    redis = await get_redis()
    
    if state_data is None:
        logger.debug(f"[set_viewer_state] No viewer state provided (None) for {request_id}")
        # Store an explicit empty state so get still works
        empty = ViewerState(is_empty=True)
        await redis.set(_redis_key(request_id), json.dumps(empty.to_dict()), ex=VIEWER_STATE_TTL)
        return
    
    try:
        vs = _parse_state_data(state_data)
        
        await redis.set(_redis_key(request_id), json.dumps(vs.to_dict()), ex=VIEWER_STATE_TTL)
        
        logger.info(f"[set_viewer_state] Stored state for {request_id}: is_empty={vs.is_empty}, "
                   f"{len(vs.structures)} structures")
        for s in vs.structures:
            logger.info(f"[set_viewer_state]   - {s.label} ({s.structure_type}): "
                       f"chains={s.chain_ids}, residues={s.residue_count}, atoms={s.atom_count}")
        
    except Exception as e:
        logger.warning(f"Failed to parse/store viewer state: {e}")
        empty = ViewerState(is_empty=True)
        await redis.set(_redis_key(request_id), json.dumps(empty.to_dict()), ex=VIEWER_STATE_TTL)


async def get_current_viewer_state(request_id: str) -> ViewerState:
    """Get the current viewer state from Redis.
    
    Args:
        request_id: Unique identifier for the current request.
    
    Returns:
        Current ViewerState, or an empty state if none is set.
    """
    from api.redis import get_redis
    redis = await get_redis()
    
    raw = await redis.get(_redis_key(request_id))
    if raw is None:
        return ViewerState(is_empty=True)
    
    try:
        data = json.loads(raw)
        return _parse_state_data(data)
    except Exception as e:
        logger.warning(f"Failed to deserialize viewer state for {request_id}: {e}")
        return ViewerState(is_empty=True)


async def add_visualization_to_state(
    request_id: str,
    label: str,
    structure_type: str = "protein",
    format: str = "pdb",
    metadata: Optional[dict[str, Any]] = None,
    pdb_ref: str = "",
) -> None:
    """Update viewer state when a visualization tool sends data to the frontend.
    
    This should be called when visualize_pdb, visualize_3d, or python_execute
    sends visualization data via SSE. It updates the Redis-backed viewer state so
    that subsequent get_viewer_state calls reflect what's actually displayed.
    
    Args:
        request_id: Unique identifier for the current request.
        label: Display label for the structure.
        structure_type: Type of structure ("protein", "small_molecule", "complex").
        format: Data format ("pdb", "mmcif", "sdf").
        metadata: Optional metadata about the structure.
        pdb_ref: Source artifact reference (e.g. "pdb_abc12345"). Empty when
            the structure is not ref-addressable. Used by structure_query to
            validate that its target is actually loaded.
    """
    from api.redis import get_redis
    redis = await get_redis()
    
    # Get current state
    current = await get_current_viewer_state(request_id)
    
    # Create new structure info
    new_structure = StructureInfo(
        label=label,
        structure_type=structure_type,
        format=format,
        pdb_ref=pdb_ref,
        has_polymer=(structure_type == "protein"),
        has_ligand=(structure_type == "small_molecule"),
        metadata=metadata or {},
    )
    
    current.structures.append(new_structure)
    current.is_empty = False
    
    # Save back
    await redis.set(_redis_key(request_id), json.dumps(current.to_dict()), ex=VIEWER_STATE_TTL)

    logger.info(
        f"[add_visualization_to_state] Added {label} ({structure_type}, {format}, "
        f"pdb_ref={pdb_ref or '-'}) for {request_id} - "
        f"now {len(current.structures)} structures"
    )


class GetViewerStateTool(BaseTool):
    """Query the current state of the Molstar 3D viewer.
    
    This tool returns information about what is currently displayed in the
    molecular viewer, enabling the LLM to understand context when users
    ask questions about structures they're viewing.
    
    Use Cases:
        - User asks "what chains are in this structure?"
        - User asks "is there a ligand?"
        - User references "the protein" without specifying which one
        - Before using structure_query to ensure something is loaded
    """
    
    @property
    def name(self) -> str:
        return "get_viewer_state"
    
    @property
    def summary_template(self) -> str:
        return "Checking viewer state..."
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL
    
    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY
    
    @property
    def is_async(self) -> bool:
        return False
    
    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "get_viewer_state",
                "description": """Query the current state of the Molstar 3D molecular viewer.

Use this tool when:
- The user asks about "this structure" or "the protein" without specifying
- You need to know what is currently displayed before answering questions
- The user asks about structural features (chains, residues, ligands)
- You want to verify something is loaded before using structure_query
- You need to find available selection references for python_execute

Returns information about:
- What structure(s) are loaded (label, type)
- Chain IDs present in the structure
- Residue and atom counts
- Whether ligands/small molecules are present
- Current selection state
- Available selections for analysis (references you can pass to python_execute)

Example scenarios:
- User: "What chains are in this structure?" → Call get_viewer_state first
- User: "Is there a ligand?" → Call get_viewer_state to check
- User: "Highlight the binding site" → Call get_viewer_state to confirm structure is loaded
- User: "Compute the center of mass" → Call get_viewer_state to find available selections
""",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
    
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Get the current viewer state.
        
        Args:
            arguments: Empty dict (no parameters needed).
            
        Returns:
            ToolResult with description of current viewer state.
        """
        # request_id and session_id are injected by the tool registry
        request_id = getattr(self, "_request_id", None) or "unknown"
        session_id = getattr(self, "_session_id", None) or "default"
        
        state = await get_current_viewer_state(request_id)
        
        # Load selections for the description
        from api.selections import get_available_selections
        from api.redis import get_redis as _get_redis
        redis_client = await _get_redis()
        selections = await get_available_selections(redis_client, session_id)
        
        description = state.to_description(selections=selections)
        
        logger.info(f"[get_viewer_state] Queried state for {request_id}: is_empty={state.is_empty}, "
                   f"structures={len(state.structures)}")
        logger.info(f"[get_viewer_state] Returning to LLM:\n{description}")
        
        return ToolResult.text(description)
