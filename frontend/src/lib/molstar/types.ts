/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Type definitions for Molstar structure queries.
 */

import type { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';
import type { StructureSelection } from 'molstar/lib/mol-model/structure';

/**
 * Result of a structure query containing matched elements.
 */
export interface QueryResult {
  /** Number of atoms in the selection */
  atomCount: number;
  /** Number of residues in the selection */
  residueCount: number;
  /** The underlying Molstar selection object */
  selection: StructureSelection;
  /** Human-readable summary of the selection */
  summary: string;
}

/**
 * Options for highlighting selections.
 */
export interface HighlightOptions {
  /** Color for highlighting (hex number, e.g., 0xFF0000 for red) */
  color?: number;
  /** Transparency (0 = opaque, 1 = fully transparent) */
  transparency?: number;
  /** Whether to focus camera on the selection */
  focus?: boolean;
}

/**
 * Options for distance-based queries.
 */
export interface DistanceQueryOptions {
  /** Distance threshold in Angstroms */
  radius: number;
  /** Whether to include whole residues when any atom matches */
  wholeResidues?: boolean;
  /** Entity types to include ('polymer', 'non-polymer', 'water', 'branched') */
  entityTypes?: EntityType[];
}

/**
 * Entity types in molecular structures.
 */
export type EntityType = 'polymer' | 'non-polymer' | 'water' | 'branched';

/**
 * Chain selection criteria.
 */
export interface ChainSelector {
  /** Chain ID (e.g., 'A', 'B') */
  chainId?: string;
  /** Auth chain ID */
  authChainId?: string;
}

/**
 * Residue selection criteria.
 */
export interface ResidueSelector {
  /** Residue name (e.g., 'ALA', 'GLY') */
  name?: string;
  /** Residue sequence number */
  seqNumber?: number;
  /** Range of sequence numbers [start, end] */
  seqRange?: [number, number];
  /** Insertion code */
  insCode?: string;
}

/**
 * Atom selection criteria.
 */
export interface AtomSelector {
  /** Atom name (e.g., 'CA', 'CB', 'N') */
  name?: string;
  /** Element symbol (e.g., 'C', 'N', 'O') */
  element?: string;
  /** Atom ID */
  id?: number;
}

/**
 * Combined selector for building complex queries.
 */
export interface StructureSelector {
  chain?: ChainSelector;
  residue?: ResidueSelector;
  atom?: AtomSelector;
}

/**
 * Information about a residue in a selection.
 */
export interface ResidueInfo {
  /** Residue name (3-letter code) */
  name: string;
  /**
   * Sequence number in LABEL numbering (label_seq_id), 1..n per polymer
   * entity. This is the numbering AtomWorks/RFD4 use for ``res_id``.
   */
  seqNumber: number;
  /** Chain ID in LABEL numbering (label_asym_id). */
  chainId: string;
  /**
   * Sequence number as the depositor wrote it (auth_seq_id). This is what the
   * user sees in the viewer and what MolQL ``atom.resno`` matches.
   */
  authSeqNumber: number;
  /** Chain ID as the depositor wrote it (auth_asym_id); MolQL ``atom.chain``. */
  authChainId: string;
  /** Insertion code (if any) */
  insCode: string;
  /** Entity type */
  entityType: EntityType;
}

/**
 * Information about an atom in a selection.
 */
export interface AtomInfo {
  /** Atom name */
  name: string;
  /** Element symbol */
  element: string;
  /** Atom ID */
  id: number;
  /** Coordinates [x, y, z] in Angstroms */
  coords: [number, number, number];
  /** Residue this atom belongs to */
  residue: ResidueInfo;
}

/**
 * Geometric measurement results.
 */
export interface GeometricMeasurement {
  /** Distance in Angstroms (for 2 atoms) */
  distance?: number;
  /** Angle in degrees (for 3 atoms) */
  angle?: number;
  /** Dihedral angle in degrees (for 4 atoms) */
  dihedral?: number;
}

/**
 * Plugin reference type for internal use.
 */
export type PluginRef = PluginUIContext;

/**
 * Information about a loaded structure in the viewer.
 */
export interface StructureStateInfo {
  /** Display label for the structure */
  label: string;
  /** Type of structure (protein, small_molecule, complex) */
  structureType: string;
  /** File format used (pdb, sdf, etc.) */
  format: string;
  /** Chain IDs present in the structure */
  chainIds: string[];
  /** Total number of residues */
  residueCount: number;
  /** Total number of atoms */
  atomCount: number;
  /** Whether the structure contains polymer entities */
  hasPolymer: boolean;
  /** Whether the structure contains ligand/small molecule entities */
  hasLigand: boolean;
  /** Whether the structure contains water molecules */
  hasWater: boolean;
  /** Additional metadata */
  metadata: Record<string, unknown>;
}

/**
 * Complete state of the Molstar viewer for LLM context.
 */
export interface ViewerStateInfo {
  /** Whether the viewer is empty (no structures loaded) */
  isEmpty: boolean;
  /** Information about loaded structures */
  structures: StructureStateInfo[];
  /** Whether there is an active selection */
  hasSelection: boolean;
  /** Description of current selection if any */
  selectionInfo: string | null;
}

