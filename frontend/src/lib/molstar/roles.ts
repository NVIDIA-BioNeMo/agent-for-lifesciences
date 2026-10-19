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
 * Which design roles a pick can play, and what each button says.
 *
 * The catalog itself (`GET /api/design-roles`) lives on the backend, so the
 * viewer, the language model and the compiler read one definition. This module
 * only *evaluates* it against the current pick: it decides which buttons to
 * offer, which to grey out and why, and mirrors the backend's atom rules so
 * that a button is shown exactly when the backend would accept the pick.
 *
 * Pure data and logic, no Mol*, so every rule is testable on its own.
 */

import type { PickedResidue } from './pickSet';
import { residueKey } from './pickSet';

// --- The catalog, as the backend sends it ----------------------------------

export type AtomRule =
  | { kind: 'narrow'; allowed: Record<string, string[]> }
  | { kind: 'group'; groups: Record<string, string[][]> };

export interface RoleVariant {
  value: string;
  label: string;
  tooltip: string;
  atom_rule?: AtomRule;
}

export type SupportLevel = 'native' | 'residue' | 'implied' | 'downstream' | 'none';

export interface RoleDef {
  key: string;
  label: string;
  annotation: string;
  tooltip: string;
  colour: string;
  group: 'primary' | 'more';
  granularity: 'residue' | 'atom' | 'either';
  value_kind: 'none' | 'enum' | 'group' | 'layout';
  variants: RoleVariant[];
  atom_rule: AtomRule | null;
  inside_keep: boolean;
  engines: Record<string, { support: SupportLevel; note: string }>;
}

export interface EngineDef {
  key: string;
  label: string;
  structure_format: string;
  numbering: string;
  note: string;
}

export interface RolesPayload {
  roles: RoleDef[];
  engines: EngineDef[];
  default_engine: string;
}

/** The group a Float joins when none is named; matches the backend. */
export const DEFAULT_FLOAT_GROUP = 'motif';

/** Roles the Motif button assigns together. */
export const MOTIF_ROLES = ['keep', 'fix_position', 'fix_sequence', 'float'] as const;

// --- Applying an atom rule -------------------------------------------------

/**
 * The atoms a role applies to within one picked residue, or null if the
 * residue cannot play it. Mirrors `roles.apply_atom_rule` on the backend.
 */
export function applyAtomRule(
  rule: AtomRule | null | undefined,
  residueName: string,
  picked: readonly string[]
): string[] | null {
  if (!rule) return [...picked];
  const name = residueName.toUpperCase();
  const have = new Set(picked);
  if (rule.kind === 'narrow') {
    const allowed = rule.allowed[name];
    if (!allowed || allowed.length === 0) return null;
    const kept = allowed.filter((a) => have.has(a));
    return kept.length > 0 ? kept : null;
  }
  const groups = rule.groups[name];
  if (!groups || groups.length === 0) return null;
  const out: string[] = [];
  for (const group of groups) {
    if (group.every((a) => have.has(a))) {
      for (const a of group) if (!out.includes(a)) out.push(a);
    }
  }
  return out.length > 0 ? out : null;
}

// --- Buttons ---------------------------------------------------------------

/** One residue of the current pick, with the atoms it contributes. */
export interface PickedAtoms {
  residue: PickedResidue;
  atoms: string[];
}

/** One button the user may press to give the current pick a role. */
export interface RoleOption {
  /** `hotspot`, `rasa:BURIED`, `float:g2`, or `motif` for the composite. */
  id: string;
  role: string;
  value: string | null;
  label: string;
  tooltip: string;
  colour: string;
  group: 'primary' | 'more';
  enabled: boolean;
  /** Why a disabled button is disabled. */
  reason?: string;
  /** For the composite: the roles it assigns, in order. */
  composite?: string[];
}

/** Identity of a layer: its role, plus the parameter for roles that carry one. */
export function layerId(def: RoleDef | undefined, role: string, value: string | null): string {
  if (def && (def.value_kind === 'enum' || def.value_kind === 'group') && value) {
    return `${role}:${value}`;
  }
  return role;
}

function listResidues(residues: readonly PickedResidue[], limit = 4): string {
  const shown = residues.slice(0, limit).map((r) => `${r.authChain}${r.authSeq}`);
  return shown.join(' ') + (residues.length > limit ? ` (+${residues.length - limit} more)` : '');
}

export interface OptionContext {
  payload: RolesPayload | null;
  engine: string;
  picked: PickedAtoms[];
  /** Keys of the residues kept so far, or null when there is no Keep layer. */
  keep: Set<string> | null;
  /** Float groups already in use, for offering a new one. */
  floatGroups: string[];
}

/**
 * The buttons for the current pick.
 *
 * The five primary roles are always listed, disabled with a reason when they
 * cannot apply, so the vocabulary stays learnable. Everything in the More menu
 * (chemistry, secondary structure, burial, the Motif composite) is listed only
 * when the pick is valid for it: a ring-less pick never offers Pi-stack.
 */
export function roleOptions(ctx: OptionContext): RoleOption[] {
  const { payload, engine, picked, keep } = ctx;
  if (!payload) return [];
  const out: RoleOption[] = [];
  const empty = picked.length === 0;
  const outside = keep
    ? picked.filter((p) => !keep.has(residueKey(p.residue))).map((p) => p.residue)
    : [];

  const support = (def: RoleDef) => def.engines[engine] ?? { support: 'none' as const, note: '' };

  const consider = (
    def: RoleDef,
    value: string | null,
    label: string,
    tooltip: string,
    rule: AtomRule | null | undefined,
    group: 'primary' | 'more'
  ) => {
    const option: RoleOption = {
      id: value ? `${def.key}:${value}` : def.key,
      role: def.key,
      value,
      label,
      tooltip,
      colour: def.colour,
      group,
      enabled: true,
    };
    const level = support(def);
    if (level.support === 'none') {
      option.enabled = false;
      option.reason = level.note || 'The selected engine cannot take this.';
    } else if (empty) {
      option.enabled = false;
      option.reason = 'Pick residues first.';
    } else if (def.inside_keep && keep && outside.length > 0) {
      option.enabled = false;
      option.reason = `Outside Keep: ${listResidues(outside)}.`;
    } else if (rule) {
      const bad = picked.filter((p) => applyAtomRule(rule, p.residue.name, p.atoms) === null);
      if (bad.length > 0) {
        option.enabled = false;
        option.reason = `Not valid for ${listResidues(bad.map((b) => b.residue))}.`;
      }
    }
    // Chemistry and the like are offered only when valid, and never for an
    // engine that cannot take them: nothing useful to say to the user.
    if (group === 'more' && !option.enabled) return;
    out.push(option);
  };

  for (const def of payload.roles) {
    if (def.value_kind === 'enum') {
      for (const variant of def.variants) {
        consider(
          def,
          variant.value,
          variant.label,
          variant.tooltip ? `${def.tooltip}\n${variant.tooltip}` : def.tooltip,
          variant.atom_rule ?? def.atom_rule,
          def.group
        );
      }
      continue;
    }
    if (def.value_kind === 'group') {
      consider(def, DEFAULT_FLOAT_GROUP, def.label, def.tooltip, def.atom_rule, def.group);
      if (ctx.floatGroups.length > 0) {
        const next = `g${ctx.floatGroups.length + 1}`;
        consider(
          def,
          next,
          `${def.label} (new group)`,
          `${def.tooltip}\nA separate group: these residues are placed independently of the others.`,
          def.atom_rule,
          'more'
        );
      }
      continue;
    }
    consider(def, null, def.label, def.tooltip, def.atom_rule, def.group);
  }

  // The Motif composite: keep these residues, hold their coordinates and
  // sequence, and let the design choose where they sit.
  const rfd4 = payload.roles.find((r) => r.key === 'keep');
  if (rfd4 && support(rfd4).support !== 'none' && !empty) {
    const defs = MOTIF_ROLES.map((k) => payload.roles.find((r) => r.key === k));
    if (defs.every((d) => d && support(d).support !== 'none')) {
      out.push({
        id: 'motif',
        role: 'motif',
        value: null,
        label: 'Motif',
        tooltip:
          'Motif: Keep + Fix position + Fix sequence + Float. Carry these residues into a ' +
          'new scaffold, holding their shape and sequence and letting the design choose ' +
          'where they sit.',
        colour: rfd4.colour,
        group: 'more',
        enabled: true,
        composite: [...MOTIF_ROLES],
      });
    }
  }
  return out;
}

// --- The kept pieces and their layout --------------------------------------

/** A run of kept residues, adjacent in label numbering. */
export interface KeepPiece {
  chain: string;
  firstLabel: number;
  lastLabel: number;
  /** Author numbers of the first and last residue, for the reader. */
  firstAuth: number;
  lastAuth: number;
  length: number;
}

/** Kept residues grouped into the stretches the contig will name. */
export function keepPieces(residues: readonly PickedResidue[]): KeepPiece[] {
  const byChain = new Map<string, PickedResidue[]>();
  for (const r of residues) {
    const group = byChain.get(r.labelChain);
    if (group) group.push(r);
    else byChain.set(r.labelChain, [r]);
  }
  const pieces: KeepPiece[] = [];
  byChain.forEach((members, chain) => {
    const sorted = [...members].sort((a, b) => a.labelSeq - b.labelSeq);
    let run: PickedResidue[] = [];
    const flush = () => {
      if (run.length === 0) return;
      pieces.push({
        chain,
        firstLabel: run[0].labelSeq,
        lastLabel: run[run.length - 1].labelSeq,
        firstAuth: run[0].authSeq,
        lastAuth: run[run.length - 1].authSeq,
        length: run.length,
      });
      run = [];
    };
    for (const r of sorted) {
      if (run.length > 0 && r.labelSeq !== run[run.length - 1].labelSeq + 1) flush();
      run.push(r);
    }
    flush();
  });
  return pieces;
}

/** The lengths of what is designed between and around the kept pieces. */
export interface Layout {
  n_term?: [number, number];
  c_term?: [number, number];
  linkers?: Array<[number, number]>;
}

/** Whether a layout carries anything worth sending. */
export function layoutIsEmpty(layout: Layout | null | undefined): boolean {
  return !layout || (!layout.n_term && !layout.c_term && !(layout.linkers && layout.linkers.length));
}
