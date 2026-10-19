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
 * Selection utilities for working with Molstar structures.
 * 
 * Provides functions for:
 * - Executing queries and getting results
 * - Extracting information from selections
 * - Visual highlighting
 * - Geometric measurements
 */

import { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';
import { Structure, StructureElement, StructureSelection, StructureProperties as SP } from 'molstar/lib/mol-model/structure';
import { Loci } from 'molstar/lib/mol-model/loci';
import { Vec3 } from 'molstar/lib/mol-math/linear-algebra';
import type { Expression } from 'molstar/lib/mol-script/language/expression';

import type {
  QueryResult,
  HighlightOptions,
  ResidueInfo,
  AtomInfo,
  EntityType,
  ViewerStateInfo,
  StructureStateInfo,
} from './types';
import { runQuery, ligandQuery, polymerQuery, waterQuery } from './queries';

/**
 * Get the first structure from the plugin state.
 */
export function getStructure(plugin: PluginUIContext): Structure | undefined {
  const structures = plugin.managers.structure.hierarchy.current.structures;
  if (structures.length === 0) return undefined;
  
  const structureRef = structures[0];
  const cell = structureRef.cell;
  if (!cell?.obj) return undefined;
  
  return cell.obj.data;
}

/**
 * Get all structures from the plugin state.
 */
export function getAllStructures(plugin: PluginUIContext): Structure[] {
  const structures = plugin.managers.structure.hierarchy.current.structures;
  return structures
    .map(ref => ref.cell?.obj?.data)
    .filter((s): s is Structure => s !== undefined);
}

/**
 * Execute a query expression and return structured results.
 */
export function queryStructure(
  plugin: PluginUIContext,
  expression: Expression
): QueryResult | null {
  const structure = getStructure(plugin);
  if (!structure) return null;

  const selection = runQuery(expression, structure);
  
  // Count atoms and residues
  const loci = StructureSelection.toLociWithSourceUnits(selection);
  
  let atomCount = 0;
  const residueSet = new Set<string>();
  
  if (Loci.isEmpty(loci)) {
    return {
      atomCount: 0,
      residueCount: 0,
      selection,
      summary: 'No atoms matched the query',
    };
  }

  // Iterate through the selection to count
  StructureElement.Loci.forEachLocation(loci, (loc) => {
    atomCount++;
    const residueKey = `${SP.chain.label_asym_id(loc)}:${SP.residue.label_seq_id(loc)}:${SP.residue.pdbx_PDB_ins_code(loc)}`;
    residueSet.add(residueKey);
  });

  const residueCount = residueSet.size;
  const summary = `Selected ${atomCount} atoms in ${residueCount} residues`;

  return {
    atomCount,
    residueCount,
    selection,
    summary,
  };
}

/**
 * Get detailed information about residues in a selection.
 */
export function getResidueInfo(
  plugin: PluginUIContext,
  expression: Expression
): ResidueInfo[] {
  const structure = getStructure(plugin);
  if (!structure) return [];

  const selection = runQuery(expression, structure);
  const loci = StructureSelection.toLociWithSourceUnits(selection);
  
  if (Loci.isEmpty(loci)) return [];

  const residueMap = new Map<string, ResidueInfo>();

  StructureElement.Loci.forEachLocation(loci, (loc) => {
    const chainId = SP.chain.label_asym_id(loc);
    const seqNumber = SP.residue.label_seq_id(loc);
    const insCode = SP.residue.pdbx_PDB_ins_code(loc) || '';
    const key = `${chainId}:${seqNumber}:${insCode}`;
    
    if (!residueMap.has(key)) {
      residueMap.set(key, {
        name: SP.atom.label_comp_id(loc),
        seqNumber,
        chainId,
        authSeqNumber: SP.residue.auth_seq_id(loc),
        authChainId: SP.chain.auth_asym_id(loc),
        insCode,
        entityType: SP.entity.type(loc) as EntityType,
      });
    }
  });

  return Array.from(residueMap.values());
}

/**
 * Get detailed information about atoms in a selection.
 */
export function getAtomInfo(
  plugin: PluginUIContext,
  expression: Expression
): AtomInfo[] {
  const structure = getStructure(plugin);
  if (!structure) return [];

  const selection = runQuery(expression, structure);
  const loci = StructureSelection.toLociWithSourceUnits(selection);
  
  if (Loci.isEmpty(loci)) return [];

  const atoms: AtomInfo[] = [];

  StructureElement.Loci.forEachLocation(loci, (loc) => {
    const pos = loc.unit.conformation.position(loc.element, Vec3());
    
    atoms.push({
      name: SP.atom.label_atom_id(loc),
      element: SP.atom.type_symbol(loc),
      id: SP.atom.id(loc),
      coords: [pos[0], pos[1], pos[2]],
      residue: {
        name: SP.atom.label_comp_id(loc),
        seqNumber: SP.residue.label_seq_id(loc),
        chainId: SP.chain.label_asym_id(loc),
        authSeqNumber: SP.residue.auth_seq_id(loc),
        authChainId: SP.chain.auth_asym_id(loc),
        insCode: SP.residue.pdbx_PDB_ins_code(loc) || '',
        entityType: SP.entity.type(loc) as EntityType,
      },
    });
  });

  return atoms;
}

/**
 * Highlight atoms matching a query expression.
 */
export async function highlightSelection(
  plugin: PluginUIContext,
  expression: Expression,
  options: HighlightOptions = {}
): Promise<void> {
  const structure = getStructure(plugin);
  if (!structure) return;

  const selection = runQuery(expression, structure);
  const loci = StructureSelection.toLociWithSourceUnits(selection);
  
  if (Loci.isEmpty(loci)) return;

  // Use the plugin's highlight manager
  plugin.managers.interactivity.lociHighlights.highlight({ loci });
  
  // Focus camera if requested
  if (options.focus) {
    plugin.managers.camera.focusLoci(loci);
  }
}

/**
 * Clear all highlights.
 */
export function clearHighlights(plugin: PluginUIContext): void {
  plugin.managers.interactivity.lociHighlights.clearHighlights();
}

/**
 * Select atoms matching a query (for more persistent selection than highlight).
 */
export async function selectAtoms(
  plugin: PluginUIContext,
  expression: Expression,
  options: { add?: boolean } = {}
): Promise<void> {
  const structure = getStructure(plugin);
  if (!structure) return;

  const selection = runQuery(expression, structure);
  const loci = StructureSelection.toLociWithSourceUnits(selection);
  
  if (Loci.isEmpty(loci)) return;

  if (options.add) {
    plugin.managers.interactivity.lociSelects.select({ loci });
  } else {
    plugin.managers.interactivity.lociSelects.deselectAll();
    plugin.managers.interactivity.lociSelects.select({ loci });
  }
}

/**
 * Clear the current selection.
 */
export function clearSelection(plugin: PluginUIContext): void {
  plugin.managers.interactivity.lociSelects.deselectAll();
}

/**
 * Focus the camera on atoms matching a query.
 */
export async function focusOnSelection(
  plugin: PluginUIContext,
  expression: Expression
): Promise<void> {
  const structure = getStructure(plugin);
  if (!structure) return;

  const selection = runQuery(expression, structure);
  const loci = StructureSelection.toLociWithSourceUnits(selection);
  
  if (Loci.isEmpty(loci)) return;

  plugin.managers.camera.focusLoci(loci);
}

/**
 * Calculate distance between two atoms.
 */
export function measureDistance(
  atom1: AtomInfo,
  atom2: AtomInfo
): number {
  const [x1, y1, z1] = atom1.coords;
  const [x2, y2, z2] = atom2.coords;
  
  return Math.sqrt(
    Math.pow(x2 - x1, 2) +
    Math.pow(y2 - y1, 2) +
    Math.pow(z2 - z1, 2)
  );
}

/**
 * Calculate angle between three atoms (in degrees).
 */
export function measureAngle(
  atom1: AtomInfo,
  atom2: AtomInfo,  // vertex
  atom3: AtomInfo
): number {
  const v1 = Vec3.create(
    atom1.coords[0] - atom2.coords[0],
    atom1.coords[1] - atom2.coords[1],
    atom1.coords[2] - atom2.coords[2]
  );
  const v2 = Vec3.create(
    atom3.coords[0] - atom2.coords[0],
    atom3.coords[1] - atom2.coords[1],
    atom3.coords[2] - atom2.coords[2]
  );
  
  const dot = Vec3.dot(v1, v2);
  const mag1 = Vec3.magnitude(v1);
  const mag2 = Vec3.magnitude(v2);
  
  const cosAngle = dot / (mag1 * mag2);
  const angleRad = Math.acos(Math.max(-1, Math.min(1, cosAngle)));
  
  return angleRad * (180 / Math.PI);
}

/**
 * Calculate dihedral angle between four atoms (in degrees).
 */
export function measureDihedral(
  atom1: AtomInfo,
  atom2: AtomInfo,
  atom3: AtomInfo,
  atom4: AtomInfo
): number {
  // Vectors between consecutive atoms
  const b1 = Vec3.create(
    atom2.coords[0] - atom1.coords[0],
    atom2.coords[1] - atom1.coords[1],
    atom2.coords[2] - atom1.coords[2]
  );
  const b2 = Vec3.create(
    atom3.coords[0] - atom2.coords[0],
    atom3.coords[1] - atom2.coords[1],
    atom3.coords[2] - atom2.coords[2]
  );
  const b3 = Vec3.create(
    atom4.coords[0] - atom3.coords[0],
    atom4.coords[1] - atom3.coords[1],
    atom4.coords[2] - atom3.coords[2]
  );
  
  // Normal vectors to planes
  const n1 = Vec3.cross(Vec3(), b1, b2);
  const n2 = Vec3.cross(Vec3(), b2, b3);
  
  // Normalize b2 for the calculation
  const b2Norm = Vec3.normalize(Vec3(), b2);
  
  // m1 is perpendicular to n1 in the plane defined by b2
  const m1 = Vec3.cross(Vec3(), n1, b2Norm);
  
  const x = Vec3.dot(n1, n2);
  const y = Vec3.dot(m1, n2);
  
  const angleRad = Math.atan2(y, x);
  return angleRad * (180 / Math.PI);
}

/**
 * Get the center of mass of a selection.
 */
export function getCenterOfMass(
  plugin: PluginUIContext,
  expression: Expression
): [number, number, number] | null {
  const atoms = getAtomInfo(plugin, expression);
  if (atoms.length === 0) return null;
  
  let sumX = 0, sumY = 0, sumZ = 0;
  for (const atom of atoms) {
    sumX += atom.coords[0];
    sumY += atom.coords[1];
    sumZ += atom.coords[2];
  }
  
  const n = atoms.length;
  return [sumX / n, sumY / n, sumZ / n];
}

/**
 * Calculate the bounding box of a selection.
 */
export function getBoundingBox(
  plugin: PluginUIContext,
  expression: Expression
): { min: [number, number, number]; max: [number, number, number] } | null {
  const atoms = getAtomInfo(plugin, expression);
  if (atoms.length === 0) return null;
  
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  
  for (const atom of atoms) {
    const [x, y, z] = atom.coords;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    minZ = Math.min(minZ, z);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
    maxZ = Math.max(maxZ, z);
  }
  
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
  };
}

/**
 * Get the current state of the viewer for LLM context.
 * 
 * Collects information about all loaded structures including:
 * - Chain composition
 * - Entity types (polymer, ligand, water)
 * - Atom and residue counts
 */
export function getViewerState(
  plugin: PluginUIContext,
  structureLabels: Map<string, { label: string; format: string }>,
  picks?: { residueCount: number; description: string }
): ViewerStateInfo {
  const structures = getAllStructures(plugin);
  
  if (structures.length === 0) {
    return {
      isEmpty: true,
      structures: [],
      hasSelection: false,
      selectionInfo: null,
    };
  }
  
  const structureInfos: StructureStateInfo[] = [];
  
  for (const structure of structures) {
    // Get unique chain IDs
    const chainIds = new Set<string>();
    const residueKeys = new Set<string>();
    let atomCount = 0;
    
    // Count atoms and collect chain info by iterating through units
    for (const unit of structure.units) {
      const elements = unit.elements;
      atomCount += elements.length;
      
      for (let i = 0; i < elements.length; i++) {
        const element = elements[i];
        const loc = StructureElement.Location.create(structure, unit, element);
        const chainId = SP.chain.label_asym_id(loc);
        chainIds.add(chainId);
        
        const residueKey = `${chainId}:${SP.residue.label_seq_id(loc)}:${SP.residue.pdbx_PDB_ins_code(loc)}`;
        residueKeys.add(residueKey);
      }
    }
    
    // Check entity types by running queries
    const hasPolymer = !Loci.isEmpty(
      StructureSelection.toLociWithSourceUnits(runQuery(polymerQuery(), structure))
    );
    const hasLigand = !Loci.isEmpty(
      StructureSelection.toLociWithSourceUnits(runQuery(ligandQuery(), structure))
    );
    const hasWater = !Loci.isEmpty(
      StructureSelection.toLociWithSourceUnits(runQuery(waterQuery(), structure))
    );
    
    // Determine structure type
    let structureType = 'unknown';
    if (hasPolymer && hasLigand) {
      structureType = 'complex';
    } else if (hasPolymer) {
      structureType = 'protein';
    } else if (hasLigand) {
      structureType = 'small_molecule';
    }
    
    // Get label from the stored map or use default
    const structureId = structure.hashCode.toString();
    const labelInfo = structureLabels.get(structureId) || { label: 'Structure', format: '' };
    
    structureInfos.push({
      label: labelInfo.label,
      structureType,
      format: labelInfo.format,
      chainIds: Array.from(chainIds).sort(),
      residueCount: residueKeys.size,
      atomCount,
      hasPolymer,
      hasLigand,
      hasWater,
      // Queries, selections and picks all address the first structure only
      // (see getStructure). Say so, so the model does not assume otherwise.
      metadata: { selectable: structureInfos.length === 0 },
    });
  }

  // What the user has marked, in order of how deliberate it is: residues
  // picked by hand for a design, then whatever Mol* itself has selected (a
  // structure_query result, or a left-click in selection mode).
  const stats = plugin.managers.structure.selection.stats;
  const molstarSelected = stats.elementCount > 0;
  const pickCount = picks?.residueCount ?? 0;

  const descriptions: string[] = [];
  if (picks && pickCount > 0) descriptions.push(picks.description);
  if (molstarSelected) descriptions.push(`Selected in viewer: ${stats.label}`);

  return {
    isEmpty: false,
    structures: structureInfos,
    hasSelection: pickCount > 0 || molstarSelected,
    selectionInfo: descriptions.length > 0 ? descriptions.join('; ') : null,
  };
}

