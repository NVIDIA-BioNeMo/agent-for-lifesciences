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

import test from 'node:test';
import assert from 'node:assert/strict';
import { StructureElement, StructureProperties, type Structure } from 'molstar/lib/mol-model/structure';
import type { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';

import {
  AevomChainColorThemeProvider,
  applyViewerStyle,
  chainColour,
  roleColours,
  sortChainIds,
  STYLES,
  STYLE_IDS,
  ROLE_COLOURS,
} from '../src/lib/molstar/styles';
import { CIF } from './fixture';
import { loadStructure } from './harness';

/** The fixture with chain B's rows written before chain A's, as RFD4 writes a design. */
function binderFirst(cif: string): string {
  const lines = cif.split('\n');
  const isRow = (l: string, chain: string) => new RegExp(`^(ATOM|HETATM) \\d+ \\w+ \\w+ \\. \\w+ ${chain} `).test(l);
  const b = lines.filter((l) => isRow(l, 'B'));
  const rest = lines.filter((l) => !isRow(l, 'B'));
  const firstA = rest.findIndex((l) => isRow(l, 'A'));
  return [...rest.slice(0, firstA), ...b, ...rest.slice(firstA)].join('\n');
}

function coloursByChain(structure: Structure, props: { palette: number[]; roles: Record<string, number> }) {
  const theme = AevomChainColorThemeProvider.factory({ structure }, props);
  const loc = StructureElement.Location.create(structure);
  const out: Record<string, number> = {};
  for (const unit of structure.units) {
    loc.unit = unit;
    loc.element = unit.elements[0];
    out[StructureProperties.chain.label_asym_id(loc)] = theme.color(loc, false);
  }
  return out;
}

const PALETTE = [0x111111, 0x222222, 0x333333];

test('chains sort by letter then length, not by arrival', () => {
  assert.deepEqual(sortChainIds(['B', 'AA', 'A', 'C', 'B']), ['A', 'B', 'C', 'AA']);
});

test('a chain takes the palette entry for its place, wrapping round', () => {
  const chains = ['A', 'B', 'C', 'D'];
  assert.equal(chainColour('A', chains, PALETTE, {}), PALETTE[0]);
  assert.equal(chainColour('C', chains, PALETTE, {}), PALETTE[2]);
  assert.equal(chainColour('D', chains, PALETTE, {}), PALETTE[0]);
});

test('a role colour beats the palette, and an unknown chain is grey', () => {
  assert.equal(chainColour('A', ['A', 'B'], PALETTE, { A: 0xabcdef }), 0xabcdef);
  assert.equal(chainColour('Z', ['A', 'B'], PALETTE, {}), 0xbbbbbb);
  assert.equal(chainColour('A', ['A'], [], {}), 0xbbbbbb);
});

test('roles map target and binder chains to their own colours', () => {
  const colours = roleColours({ target: ['A'], binder: ['B'] });
  assert.equal(colours.A, ROLE_COLOURS.target);
  assert.equal(colours.B, ROLE_COLOURS.binder);
  assert.deepEqual(roleColours(null), {});
});

test('the theme colours chains by letter whichever is written first', async () => {
  const props = { palette: PALETTE, roles: {} };
  const normal = coloursByChain(await loadStructure(CIF), props);
  const swapped = coloursByChain(await loadStructure(binderFirst(CIF)), props);
  assert.equal(normal.A, PALETTE[0]);
  assert.equal(normal.B, PALETTE[1]);
  assert.deepEqual(swapped, normal);
});

test('the theme gives target and binder their role colours', async () => {
  const props = { palette: PALETTE, roles: roleColours({ target: ['A'], binder: ['B'] }) };
  const colours = coloursByChain(await loadStructure(binderFirst(CIF)), props);
  assert.equal(colours.A, ROLE_COLOURS.target);
  assert.equal(colours.B, ROLE_COLOURS.binder);
});

test('every style has a usable palette, and none leads with the pick colour', () => {
  for (const id of STYLE_IDS) {
    const palette = STYLES[id].palette;
    if (id === 'default') {
      assert.equal(palette, null);
      continue;
    }
    assert.ok(palette && palette.length >= 6, id);
    assert.equal(new Set(palette).size, palette.length, `${id} repeats a colour`);
  }
});

// ---------------------------------------------------------------------------
// applyViewerStyle against a stand-in plugin
// ---------------------------------------------------------------------------

function fakeStylePlugin() {
  const cartoon = {
    type: { name: 'cartoon', params: { sizeFactor: 0.2, tubularHelices: false, roundCap: false, aspectRatio: 5 } },
    colorTheme: { name: 'chain-id', params: { asymId: 'auth' } },
  };
  const setProps: any[] = [];
  const options = { ignoreLight: false };
  const plugin = {
    canvas3d: { props: { renderer: { backgroundColor: 0x0a0e14 } }, setProps: (p: unknown) => setProps.push(p) },
    managers: {
      structure: {
        component: {
          state: { options },
          setOptions: async (o: { ignoreLight: boolean }) => {
            options.ignoreLight = o.ignoreLight;
          },
        },
        hierarchy: {
          current: {
            structures: [{ components: [{ representations: [{ cell: { transform: { ref: 'cartoon-1', params: cartoon } } }] }] }],
          },
        },
      },
    },
    representation: { structure: { themes: { colorThemeRegistry: { has: () => true, add: () => undefined } } } },
    state: {
      data: {
        build() {
          const edits: Array<(old: any) => any> = [];
          const builder = {
            to: () => ({
              update: (_t: unknown, fn: (old: any) => any) => {
                edits.push(fn);
                return builder;
              },
            }),
            commit: async () => {
              for (const fn of edits) {
                const next = fn(cartoon);
                cartoon.type = next.type;
                cartoon.colorTheme = next.colorTheme;
              }
            },
          };
          return builder;
        },
      },
    },
  };
  return { plugin: plugin as unknown as PluginUIContext, cartoon, setProps, options };
}

test('Studio turns on strong occlusion, softens the light and colours chains by letter', async () => {
  const { plugin, cartoon, setProps } = fakeStylePlugin();
  await applyViewerStyle(plugin, 'studio', 'chains', null);
  const props = setProps.at(-1);
  assert.equal(props.postprocessing.occlusion.name, 'on');
  assert.equal(props.postprocessing.occlusion.params.samples, 64);
  assert.equal(props.postprocessing.outline.name, 'off');
  assert.equal(props.renderer.ambientIntensity, 0.65);
  assert.equal(props.renderer.backgroundColor, 0x0a0e14, 'keeps our background');
  assert.equal(cartoon.colorTheme.name, 'aevom-chain');
  assert.deepEqual((cartoon.colorTheme.params as any).palette, STYLES.studio.palette);
});

test('Illustrative draws outlines in flat light', async () => {
  const { plugin, setProps, options } = fakeStylePlugin();
  await applyViewerStyle(plugin, 'illustrative', 'chains', null);
  assert.equal(setProps.at(-1).postprocessing.outline.name, 'on');
  assert.equal(options.ignoreLight, true);
});

test('Tubes makes helices tubular', async () => {
  const { plugin, cartoon } = fakeStylePlugin();
  await applyViewerStyle(plugin, 'tubes', 'chains', null);
  const params = cartoon.type.params as any;
  assert.equal(params.tubularHelices, true);
  assert.equal(params.roundCap, true);
  assert.equal(params.aspectRatio, 5, 'untouched parameters survive');
});

test('Roles mode paints the target and binder their own colours', async () => {
  const { plugin, cartoon } = fakeStylePlugin();
  await applyViewerStyle(plugin, 'studio', 'roles', { target: ['A'], binder: ['B'] });
  assert.deepEqual((cartoon.colorTheme.params as any).roles, { A: ROLE_COLOURS.target, B: ROLE_COLOURS.binder });
});

test('Default puts everything back exactly as Mol* had it', async () => {
  const { plugin, cartoon, setProps, options } = fakeStylePlugin();
  await applyViewerStyle(plugin, 'tubes', 'roles', { target: ['A'], binder: ['B'] });
  await applyViewerStyle(plugin, 'illustrative', 'chains', null);
  await applyViewerStyle(plugin, 'default', 'chains', null);
  assert.deepEqual(cartoon.colorTheme, { name: 'chain-id', params: { asymId: 'auth' } });
  assert.deepEqual(cartoon.type.params, { sizeFactor: 0.2, tubularHelices: false, roundCap: false, aspectRatio: 5 });
  assert.equal(options.ignoreLight, false);
  const props = setProps.at(-1);
  assert.equal(props.postprocessing.outline.name, 'off');
  assert.equal(props.renderer.ambientIntensity, 0.4);
});

test('Roles with no roles known falls back to Mol*\'s own colours on the default style', async () => {
  const { plugin, cartoon } = fakeStylePlugin();
  await applyViewerStyle(plugin, 'default', 'roles', null);
  assert.equal(cartoon.colorTheme.name, 'chain-id');
});
