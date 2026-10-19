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


import { Structure, StructureSelection } from 'molstar/lib/mol-model/structure';
import { trajectoryFromMmCIF } from 'molstar/lib/mol-model-formats/structure/mmcif';
import { CIF } from 'molstar/lib/mol-io/reader/cif';
import { Task } from 'molstar/lib/mol-task';
import { Subject } from 'rxjs';
import type { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';
import { ButtonsType, ModifiersKeys } from 'molstar/lib/mol-util/input/input-observer';
import type { Expression } from 'molstar/lib/mol-script/language/expression';
import type { Loci } from 'molstar/lib/mol-model/loci';

import { runQuery } from '../src/lib/molstar/queries';

export async function loadStructure(cif: string): Promise<Structure> {
  const parsed = await Task.resolveInContext(CIF.parse(cif));
  if (parsed.isError) throw new Error(parsed.message);
  const file = parsed.result;
  const trajectory = await Task.resolveInContext(trajectoryFromMmCIF(file.blocks[0], file));
  const model = await Task.resolveInContext(trajectory.getFrameAtIndex(0));
  return Structure.ofModel(model);
}

/**
 * Just enough of a plugin for the picker: one structure in the hierarchy and
 * the click stream. Everything the picker computes goes through real Mol*
 * queries on the real structure; only the UI shell is stubbed.
 */
export function fakePlugin(structure: Structure) {
  const click = new Subject<unknown>();
  const shown: { selected: unknown[]; focused: unknown[] } = { selected: [], focused: [] };
  const plugin = {
    behaviors: { interaction: { click } },
    managers: {
      interactivity: {
        lociSelects: {
          deselectAll: () => {
            shown.selected = [];
          },
          select: (entry: unknown) => {
            shown.selected.push(entry);
          },
        },
      },
      camera: {
        focusLoci: (loci: unknown) => {
          shown.focused.push(loci);
        },
      },
      structure: {
        hierarchy: {
          // No components, so painting has nothing to paint and returns early.
          current: { structures: [{ cell: { obj: { data: structure } }, components: [] }] },
        },
      },
    },
  } as unknown as PluginUIContext;
  return { plugin, click, shown };
}

/** Emit a click on one residue the way Mol* does. */
export function clickResidue(
  click: Subject<unknown>,
  structure: Structure,
  expression: Expression,
  opts: { button?: number; modifiers?: Partial<Record<'shift' | 'control' | 'meta' | 'alt', boolean>> } = {}
): void {
  const loci = StructureSelection.toLociWithSourceUnits(runQuery(expression, structure));
  click.next({
    current: { loci },
    button: opts.button ?? ButtonsType.Flag.Secondary,
    buttons: opts.button ?? ButtonsType.Flag.Secondary,
    modifiers: ModifiersKeys.create(opts.modifiers ?? {}),
  });
}

/** Emit a click carrying loci you built yourself (a bond, say). */
export function clickLoci(
  click: Subject<unknown>,
  loci: Loci,
  opts: { button?: number; modifiers?: Partial<Record<'shift' | 'control' | 'meta' | 'alt', boolean>> } = {}
): void {
  click.next({
    current: { loci },
    button: opts.button ?? ButtonsType.Flag.Secondary,
    buttons: opts.button ?? ButtonsType.Flag.Secondary,
    modifiers: ModifiersKeys.create(opts.modifiers ?? {}),
  });
}
