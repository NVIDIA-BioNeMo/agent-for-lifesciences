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
 * Showing and colouring residues given by label id.
 *
 * The viewer's side of a selection that did not come from the viewer: the
 * agent's `select_residues`, a user's committed draft being re-shown, a
 * layer's colour. Everything is addressed by label ids and built into loci
 * directly, with no MolQL text in between.
 */

import { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';
import {
  Structure,
  StructureElement,
  StructureSelection,
  StructureProperties as SP,
} from 'molstar/lib/mol-model/structure';
import { Loci } from 'molstar/lib/mol-model/loci';
import { Color } from 'molstar/lib/mol-util/color';
import {
  clearStructureOverpaint,
  setStructureOverpaint,
} from 'molstar/lib/mol-plugin-state/helpers/structure-overpaint';

import { LabelAtom, LabelResidue, picksQuery, runQuery } from './queries';
import { getStructure } from './selections';

/** Parse `#RRGGBB`; null for anything else. */
export function parseHexColour(value: string | undefined): Color | null {
  if (!value || !/^#[0-9a-fA-F]{6}$/.test(value)) return null;
  return Color(parseInt(value.slice(1), 16));
}

function lociOf(
  structure: Structure,
  residues: LabelResidue[],
  atoms: LabelAtom[] = []
): StructureElement.Loci | null {
  if (residues.length === 0 && atoms.length === 0) return null;
  const selection = runQuery(picksQuery(residues, atoms), structure);
  const loci = StructureSelection.toLociWithSourceUnits(selection);
  return Loci.isEmpty(loci) ? null : loci;
}

/** What {@link showAtoms} found of what it was asked to show. */
export interface ShownCounts {
  residues: number;
  atoms: number;
}

/**
 * Select residues and individual atoms in the viewer, optionally focusing the
 * camera on them.
 *
 * Replaces the current viewer selection, as every other select path does.
 * ``residues`` are shown whole; ``atoms`` are shown alone, for residues where
 * only some atoms were chosen.
 *
 * @returns How many of each were found in the loaded structure; fewer than
 *   asked for means the selection was made on a different structure.
 */
export function showAtoms(
  plugin: PluginUIContext,
  residues: LabelResidue[],
  atoms: LabelAtom[],
  options: { focus?: boolean } = {}
): ShownCounts {
  const none = { residues: 0, atoms: 0 };
  const structure = getStructure(plugin);
  if (!structure) return none;
  plugin.managers.interactivity.lociSelects.deselectAll();
  const loci = lociOf(structure, residues, atoms);
  if (!loci) return none;

  plugin.managers.interactivity.lociSelects.select({ loci });
  if (options.focus) plugin.managers.camera.focusLoci(loci);

  const residueKeys = new Set<string>();
  const atomKeys = new Set<string>();
  StructureElement.Loci.forEachLocation(loci, (location) => {
    const residue = `${SP.chain.label_asym_id(location)}:${SP.residue.label_seq_id(location)}`;
    residueKeys.add(residue);
    atomKeys.add(`${residue}:${SP.atom.label_atom_id(location)}`);
  });
  return {
    residues: residues.filter((r) => residueKeys.has(`${r.chain}:${r.seq}`)).length,
    atoms: atoms.filter((a) => atomKeys.has(`${a.chain}:${a.seq}:${a.name}`)).length,
  };
}

/**
 * Select whole residues in the viewer.
 *
 * @returns How many were found in the loaded structure.
 */
export function showResidues(
  plugin: PluginUIContext,
  residues: LabelResidue[],
  options: { focus?: boolean } = {}
): number {
  return showAtoms(plugin, residues, [], options).residues;
}

/** Remove every colour applied with {@link paintResidues}. */
export async function clearPaint(plugin: PluginUIContext): Promise<void> {
  const components = plugin.managers.structure.hierarchy.current.structures[0]?.components ?? [];
  await clearStructureOverpaint(plugin, components);
}

/** One colour layer: whole residues, chosen atoms, and the colour they are painted. */
export interface PaintLayer {
  residues: LabelResidue[];
  /** Atoms of residues that are only partly in the layer. */
  atoms?: LabelAtom[];
  colour: Color;
}

/**
 * Paint colour layers over the structure, replacing whatever was painted.
 *
 * Overpaint is additive in Mol*, so each call clears first; callers pass the
 * complete set of layers they want to see rather than a delta.
 */
export async function paintLayers(plugin: PluginUIContext, layers: PaintLayer[]): Promise<void> {
  await clearPaint(plugin);
  const components = plugin.managers.structure.hierarchy.current.structures[0]?.components ?? [];
  if (components.length === 0) return;
  for (const layer of layers) {
    const atoms = layer.atoms ?? [];
    if (layer.residues.length === 0 && atoms.length === 0) continue;
    await setStructureOverpaint(plugin, components, layer.colour, async (structure) => {
      const loci = lociOf(structure, layer.residues, atoms);
      return loci ?? StructureElement.Loci.none(structure);
    });
  }
}
