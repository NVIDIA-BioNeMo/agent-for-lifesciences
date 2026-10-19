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
 * Viewer looks: how the structure is lit and drawn, and how chains are coloured.
 *
 * Two independent choices:
 *
 * - a **style** sets the lighting, the post-processing (ambient occlusion,
 *   outlines), the cartoon shape and a chain palette;
 * - a **colour mode** says whether chains take colours from that palette
 *   ("chains") or from their job in a design ("roles": target, binder).
 *
 * The built-in Mol* look is the `default` style and is left exactly as Mol*
 * draws it. The others colour chains with a theme of our own, which assigns
 * palette colours by chain *letter* (A first, then B, ...) rather than by the
 * order chains are written in the file. Mol*'s own chain colouring follows file
 * order, so a design that lists its binder first gave the binder the colour the
 * target had before the design.
 */

import { Color } from 'molstar/lib/mol-util/color';
import { ParamDefinition as PD } from 'molstar/lib/mol-util/param-definition';
import { Bond, StructureElement, StructureProperties, Unit } from 'molstar/lib/mol-model/structure';
import type { Structure } from 'molstar/lib/mol-model/structure';
import { ColorTheme } from 'molstar/lib/mol-theme/color';
import { ColorThemeCategory } from 'molstar/lib/mol-theme/color/categories';
import type { ThemeDataContext } from 'molstar/lib/mol-theme/theme';
import type { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';
import { StructureRepresentation3D } from 'molstar/lib/mol-plugin-state/transforms/representation';
import { PostprocessingParams } from 'molstar/lib/mol-canvas3d/passes/postprocessing';
import { RendererParams } from 'molstar/lib/mol-gl/renderer';

export type StyleId = 'default' | 'studio' | 'illustrative' | 'tubes';
export type ColourMode = 'chains' | 'roles';

/** Which label chains are the target and which are the designed binder. */
export interface ChainRoles {
  target: string[];
  binder: string[];
}

export const STYLE_IDS: readonly StyleId[] = ['default', 'studio', 'illustrative', 'tubes'];

/**
 * The look the viewer uses: soft light and strong occlusion, with the target
 * and the designed binder told apart by colour. The other styles stay defined
 * (and tested) so one can be offered again, but nothing selects them now.
 */
export const VIEWER_STYLE: StyleId = 'studio';
export const VIEWER_COLOUR_MODE: ColourMode = 'roles';

/** Colour of picked hotspot residues. Magenta, because no style uses it for a chain first. */
export const PICK_COLOUR = 0xff2e93;

/** Neutral slate for the target and a vivid green for what was designed. */
export const ROLE_COLOURS = { target: 0x8fa3b8, binder: 0x3ddc97 } as const;

const FALLBACK_COLOUR = 0xbbbbbb;

interface StyleSpec {
  label: string;
  description: string;
  /** Chain colours, A first. Absent for the Mol* default. */
  palette: readonly number[] | null;
  ignoreLight: boolean;
  occlusion: { samples: number } | null;
  outline: { scale: number; threshold: number } | null;
  /** Key and ambient light, as Mol* intensities. Absent leaves Mol*'s own. */
  light: { key: number; ambient: number } | null;
  cartoon: { tubularHelices: boolean; roundCap: boolean; sizeFactor: number };
}

const CARTOON_DEFAULT = { tubularHelices: false, roundCap: false, sizeFactor: 0.2 };

export const STYLES: Record<StyleId, StyleSpec> = {
  default: {
    label: 'Default',
    description: "Mol*'s own look",
    palette: null,
    ignoreLight: false,
    occlusion: null,
    outline: null,
    light: null,
    cartoon: CARTOON_DEFAULT,
  },
  studio: {
    label: 'Studio',
    description: 'Soft ambient light, strong occlusion, colour-blind-safe chain colours (Paul Tol "bright")',
    palette: [0x4477aa, 0xee6677, 0x228833, 0xccbb44, 0x66ccee, 0xaa3377, 0xbbbbbb],
    ignoreLight: false,
    occlusion: { samples: 64 },
    outline: null,
    light: { key: 0.5, ambient: 0.65 },
    cartoon: CARTOON_DEFAULT,
  },
  illustrative: {
    label: 'Illustrative',
    description: 'Flat light, black outlines and pastel chains, after David Goodsell',
    palette: [0x8fd3f4, 0xf7a08f, 0xa8e6a1, 0xd8a7e0, 0xf2d16b, 0xd9c2a0, 0xf29bc6, 0xcfcfcf],
    ignoreLight: true,
    occlusion: { samples: 32 },
    outline: { scale: 1, threshold: 0.33 },
    light: null,
    cartoon: CARTOON_DEFAULT,
  },
  tubes: {
    label: 'Tubes',
    description: 'Round helices, a thin silhouette and Paul Tol "vibrant" chain colours',
    palette: [0x0077bb, 0xee7733, 0x33bbee, 0xee3377, 0xcc3311, 0x009988, 0xbbbbbb],
    ignoreLight: false,
    occlusion: { samples: 64 },
    outline: { scale: 1, threshold: 0.5 },
    light: { key: 0.5, ambient: 0.65 },
    cartoon: { tubularHelices: true, roundCap: true, sizeFactor: 0.24 },
  },
};

// ---------------------------------------------------------------------------
// Chain colour theme
// ---------------------------------------------------------------------------

/** `A, B, ..., Z, AA, AB`: letters first, then by length, never by file order. */
export function sortChainIds(ids: Iterable<string>): string[] {
  return Array.from(new Set(ids)).sort((a, b) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0));
}

/**
 * The colour a chain gets: its role's colour if it has one, otherwise the
 * palette entry for its place among the sorted chain ids.
 */
export function chainColour(
  chain: string,
  sortedChains: readonly string[],
  palette: readonly number[],
  roles: Readonly<Record<string, number>>
): number {
  const role = roles[chain];
  if (role !== undefined) return role;
  const index = sortedChains.indexOf(chain);
  if (index < 0 || palette.length === 0) return FALLBACK_COLOUR;
  return palette[index % palette.length];
}

/** Role colours keyed by label chain id, from the target/binder split. */
export function roleColours(roles: ChainRoles | null): Record<string, number> {
  const out: Record<string, number> = {};
  if (!roles) return out;
  for (const chain of roles.target) out[chain] = ROLE_COLOURS.target;
  for (const chain of roles.binder) out[chain] = ROLE_COLOURS.binder;
  return out;
}

const AevomChainParams = {
  palette: PD.Value<number[]>([], { isHidden: true }),
  roles: PD.Value<Record<string, number>>({}, { isHidden: true }),
};
type AevomChainParams = typeof AevomChainParams;

export const AEVOM_CHAIN_THEME = 'aevom-chain';

function labelChainIds(structure: Structure): string[] {
  const ids: string[] = [];
  for (const model of structure.models) {
    model.properties.structAsymMap.forEach((_value, label) => ids.push(label));
  }
  return sortChainIds(ids);
}

function aevomChainTheme(
  ctx: ThemeDataContext,
  props: PD.Values<AevomChainParams>
): ColorTheme<AevomChainParams> {
  let color: (location: unknown) => Color = () => Color(FALLBACK_COLOUR);
  if (ctx.structure) {
    const chains = labelChainIds(ctx.structure.root);
    const scratch = StructureElement.Location.create(ctx.structure.root);
    const chainOf = (location: StructureElement.Location): string | undefined => {
      // Coarse (sphere/Gaussian) units have no label_asym_id property of the atomic kind.
      return Unit.isAtomic(location.unit)
        ? StructureProperties.chain.label_asym_id(location)
        : StructureProperties.coarse.asym_id(location);
    };
    color = (location) => {
      let chain: string | undefined;
      if (StructureElement.Location.is(location as StructureElement.Location)) {
        chain = chainOf(location as StructureElement.Location);
      } else if (Bond.isLocation(location as Bond.Location)) {
        const bond = location as Bond.Location;
        scratch.unit = bond.aUnit;
        scratch.element = bond.aUnit.elements[bond.aIndex];
        chain = chainOf(scratch);
      }
      if (chain === undefined) return Color(FALLBACK_COLOUR);
      return Color(chainColour(chain, chains, props.palette, props.roles));
    };
  }
  return {
    factory: aevomChainTheme,
    granularity: 'group',
    color: color as ColorTheme<AevomChainParams>['color'],
    props,
    description: 'Colours chains by letter order, or by their role in a design.',
  };
}

export const AevomChainColorThemeProvider: ColorTheme.Provider<AevomChainParams, typeof AEVOM_CHAIN_THEME> = {
  name: AEVOM_CHAIN_THEME,
  label: 'Aevom Chain',
  category: ColorThemeCategory.Chain,
  factory: aevomChainTheme,
  getParams: () => AevomChainParams,
  defaultValues: PD.getDefaultValues(AevomChainParams),
  isApplicable: (ctx) => !!ctx.structure,
};

/** Make the theme available to representations. Safe to call repeatedly. */
export function registerStyleThemes(plugin: PluginUIContext): void {
  const registry = plugin.representation.structure.themes.colorThemeRegistry;
  if (!registry.has(AevomChainColorThemeProvider)) registry.add(AevomChainColorThemeProvider);
}

// ---------------------------------------------------------------------------
// Applying a style
// ---------------------------------------------------------------------------

interface Original {
  type: unknown;
  colorTheme: unknown;
}

/**
 * Remembers each cartoon's untouched parameters the first time a style changes
 * it, so "Default" can put them back exactly. Keyed by the state ref, which
 * changes whenever a structure is reloaded.
 */
const originals = new WeakMap<object, Map<string, Original>>();

function polymerCartoons(plugin: PluginUIContext) {
  const out: { ref: string; params: any }[] = [];
  for (const structure of plugin.managers.structure.hierarchy.current.structures) {
    for (const component of structure.components) {
      for (const rep of component.representations) {
        const params = rep.cell.transform.params as any;
        if (params?.type?.name === 'cartoon') out.push({ ref: rep.cell.transform.ref, params });
      }
    }
  }
  return out;
}

/**
 * Apply a style and colour mode to what is loaded. `roles` is only used in
 * "roles" mode; chains without a role (or no roles at all) take palette colours.
 */
export async function applyViewerStyle(
  plugin: PluginUIContext,
  style: StyleId,
  mode: ColourMode,
  roles: ChainRoles | null
): Promise<void> {
  registerStyleThemes(plugin);
  const spec = STYLES[style];

  // Lighting and post-processing. Everything not set here goes back to Mol*'s
  // default, so switching away from a style leaves nothing behind.
  const canvas = plugin.canvas3d;
  if (canvas) {
    const post = PD.getDefaultValues(PostprocessingParams);
    const renderer = PD.getDefaultValues(RendererParams);
    const kept = canvas.props.renderer;
    const light = spec.light
      ? [{ ...renderer.light[0], intensity: spec.light.key }]
      : renderer.light;
    canvas.setProps({
      renderer: {
        ...renderer,
        backgroundColor: kept.backgroundColor,
        light,
        ambientIntensity: spec.light ? spec.light.ambient : renderer.ambientIntensity,
      },
      postprocessing: {
        ...post,
        occlusion: spec.occlusion
          ? {
              name: 'on',
              params: {
                ...PD.getDefaultValues(PostprocessingParams).occlusion.params,
                samples: spec.occlusion.samples,
                radius: 5,
                bias: 0.8,
                blurKernelSize: 15,
                blurDepthBias: 0.5,
                resolutionScale: 1,
              },
            }
          : post.occlusion,
        outline: spec.outline
          ? {
              name: 'on',
              params: {
                scale: spec.outline.scale,
                threshold: spec.outline.threshold,
                color: Color(0x000000),
                includeTransparent: true,
              },
            }
          : { name: 'off', params: {} },
      },
    } as any);
  }

  const options = plugin.managers.structure.component.state.options;
  if (options.ignoreLight !== spec.ignoreLight) {
    await plugin.managers.structure.component.setOptions({ ...options, ignoreLight: spec.ignoreLight });
  }

  // The cartoon: its shape and its colours.
  const cartoons = polymerCartoons(plugin);
  if (cartoons.length === 0) return;
  const key = plugin.state.data;
  let saved = originals.get(key);
  if (!saved) {
    saved = new Map();
    originals.set(key, saved);
  }
  const remembered = saved;
  const palette = spec.palette ?? STYLES.studio.palette!;
  const useOurTheme = spec.palette !== null || (mode === 'roles' && roles !== null);

  const update = plugin.state.data.build();
  for (const { ref, params } of cartoons) {
    if (!remembered.has(ref)) {
      remembered.set(ref, {
        type: JSON.parse(JSON.stringify(params.type)),
        colorTheme: params.colorTheme,
      });
    }
    const original = remembered.get(ref)!;
    update.to(ref).update(StructureRepresentation3D, (old: any) => {
      old.type = {
        name: 'cartoon',
        params: {
          ...(original.type as any).params,
          tubularHelices: spec.cartoon.tubularHelices,
          roundCap: spec.cartoon.roundCap,
          sizeFactor: spec.cartoon.sizeFactor,
        },
      };
      old.colorTheme = useOurTheme
        ? {
            name: AEVOM_CHAIN_THEME,
            params: {
              palette: [...palette],
              roles: mode === 'roles' ? roleColours(roles) : {},
            },
          }
        : original.colorTheme;
      return old;
    });
  }
  await update.commit();
}
