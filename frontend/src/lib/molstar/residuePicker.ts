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
 * Picking residues and atoms with the right mouse button.
 *
 * Mol* already gives the right button two jobs -- a click re-centres the
 * camera and a drag pans it -- so the click is taken over here (see
 * {@link withRightClickFreed}) and the drag is left alone. Mol* tells a click
 * from a drag itself, so the two coexist: a click that does not move picks,
 * a drag pans.
 *
 * What a click picks follows what was clicked, with no mode to switch. The
 * left button already shows atoms (Mol*'s focus draws the clicked residue and
 * its surroundings as ball-and-stick), so: a right-click on the ribbon picks
 * the whole residue, and a right-click on an atom in that view picks just the
 * atom. The two are told apart by the loci Mol* reports: a ribbon element is
 * a whole residue (every atom of it), an atom is one element, and a bond is
 * neither and is ignored. Shift and Ctrl always mean residues and chains.
 *
 * Picks go into one *pending* selection. Giving it a role (Hotspot, Keep, Fix
 * position, ...) stores it on the backend at once as a selection that names
 * the role, clears the pending set, and shows it as a coloured chip in the
 * role tray. The tray is the design: click a chip to edit it, remove it with
 * its x. A pending selection that never gets a role is still stored when the
 * chat sends a message ({@link ResiduePicker.commit}), so the agent can see
 * what the user pointed at.
 *
 * Which role buttons appear is decided by {@link roleOptions} against the
 * catalog the backend serves, so the viewer holds no role vocabulary of its own.
 */

import { PICK_COLOUR } from './styles';
import { PluginUIContext } from 'molstar/lib/mol-plugin-ui/context';
import { PluginSpec } from 'molstar/lib/mol-plugin/spec';
import { PluginBehaviors } from 'molstar/lib/mol-plugin/behavior';
import { DefaultPluginUISpec } from 'molstar/lib/mol-plugin-ui/spec';
import { Binding } from 'molstar/lib/mol-util/binding';
import { ButtonsType, ModifiersKeys } from 'molstar/lib/mol-util/input/input-observer';
import { Loci } from 'molstar/lib/mol-model/loci';
import { Bond, StructureElement, StructureProperties as SP } from 'molstar/lib/mol-model/structure';

import {
  picksQuery,
  polymerQuery,
  residuesByLabelQuery,
  withinDistanceQuery,
  type LabelAtom,
  type LabelResidue,
} from './queries';
import { getAtomInfo, getResidueInfo, getStructure } from './selections';
import { paintLayers, clearPaint, parseHexColour, type PaintLayer } from './residues';
import { serializeAtoms } from './selectionPayload';

import { PickSet, PickedResidue, residueKey } from './pickSet';
import {
  DEFAULT_FLOAT_GROUP,
  keepPieces,
  layerId,
  layoutIsEmpty,
  roleOptions,
  type EngineDef,
  type KeepPiece,
  type Layout,
  type PickedAtoms,
  type RoleDef,
  type RoleOption,
  type RolesPayload,
} from './roles';

/** The pending selection: what has been picked and not yet given a role. */
export interface PendingState {
  count: number;
  atomCount: number;
  /** Residues of which only some atoms are picked. */
  partialCount: number;
  description: string;
  /** True when it differs from what the backend last stored for it. */
  dirty: boolean;
}

/** A role the user has assigned, shown as a chip in the tray. */
export interface RoleLayerState {
  id: string;
  role: string;
  value: string | null;
  label: string;
  /** `#RRGGBB`. */
  colour: string;
  count: number;
  atomCount: number;
  partialCount: number;
  description: string;
}

export interface PickerState {
  pending: PendingState;
  layers: RoleLayerState[];
  options: RoleOption[];
  /** The layer being edited, whose picks are in the pending set. */
  editingId: string | null;
  /** The label of the layer being edited, for the status line. */
  editingLabel: string | null;
  engine: string;
  engines: EngineDef[];
  /** The kept stretches, when there is a Keep, and the layout between them. */
  pieces: KeepPiece[];
  layout: Layout | null;
  /** A one-line message about the last action that did nothing, if any. */
  notice: string | null;
  /** True while a role is being stored. */
  busy: boolean;
  /** A selection the agent made that can be adopted as the pending pick. */
  offered: { label: string; residueCount: number; atomCount: number } | null;
}

interface RoleLayer {
  id: string;
  role: string;
  /** Float group, enum member, or null. */
  value: string | null;
  /** A Keep layer's layout. */
  layout: Layout | null;
  picks: PickSet;
  /** Reference the backend gave the last stored set; the next one's parent. */
  committedRef: string | null;
  atomCache: { signature: string; count: number } | null;
}

const B = ButtonsType.Flag;
const Trigger = Binding.Trigger;
const M = ModifiersKeys;

/**
 * The default plugin spec with the right mouse button freed from camera focus.
 *
 * Mol*'s camera-focus behaviour treats a secondary click as "centre on this" and,
 * on empty space, as "reset the camera". Both are removed here; the left button
 * keeps both. Everything else in the default spec is unchanged.
 */
export function withRightClickFreed(): ReturnType<typeof DefaultPluginUISpec> {
  const spec = DefaultPluginUISpec();
  const freed = {
    clickCenterFocus: Binding(
      [Trigger(B.Primary, M.create()), Trigger(B.Primary, M.create({ control: true }))],
      'Camera center and focus',
      'Click element using ${triggers}'
    ),
    clickCenterFocusSelectMode: Binding(
      [Trigger(B.Primary, M.create({ control: true }))],
      'Camera center and focus (Selection Mode)',
      'Click element using ${triggers}'
    ),
    clickResetCameraOnEmpty: Binding(
      [Trigger(B.Primary, M.create()), Trigger(B.Primary, M.create({ control: true }))],
      'Reset camera focus',
      'Click on nothing using ${triggers}'
    ),
    clickResetCameraOnEmptySelectMode: Binding(
      [Trigger(B.Primary, M.create({ control: true }))],
      'Reset camera focus (Selection Mode)',
      'Click on nothing using ${triggers}'
    ),
  };
  return {
    ...spec,
    behaviors: spec.behaviors.map((behavior) =>
      behavior.transformer === PluginBehaviors.Camera.FocusLoci
        ? PluginSpec.Behavior(PluginBehaviors.Camera.FocusLoci, {
            minRadius: 8,
            extraRadius: 4,
            durationMs: 250,
            bindings: freed,
          })
        : behavior
    ),
  };
}

function residueFromLocation(loc: StructureElement.Location): PickedResidue {
  return {
    labelChain: SP.chain.label_asym_id(loc),
    labelSeq: SP.residue.label_seq_id(loc),
    authChain: SP.chain.auth_asym_id(loc),
    authSeq: SP.residue.auth_seq_id(loc),
    insCode: SP.residue.pdbx_PDB_ins_code(loc) || '',
    name: SP.atom.label_comp_id(loc),
  };
}

export class ResiduePicker {
  /** The picks that have no role yet. */
  private pending = new PickSet();
  private pendingCommittedSignature = '';
  private pendingRef: string | null = null;
  private pendingAtomCache: { signature: string; count: number } | null = null;
  private layers: RoleLayer[] = [];
  /** A layer lifted into the pending set for editing; restored on cancel. */
  private editing: RoleLayer | null = null;
  private payload: RolesPayload | null = null;
  private engine = 'rfd4';
  private sessionId: string | undefined;
  private notice: string | null = null;
  private busy = false;
  private subscription: { unsubscribe(): void } | null = null;
  private polymerCache: { structure: unknown; residues: PickedResidue[] } | null = null;
  private atomNameCache: { structure: unknown; names: Map<string, string[]> } | null = null;
  private paintQueue: Promise<void> = Promise.resolve();
  private offered: { label: string; residues: LabelResidue[]; atoms: LabelAtom[] } | null = null;

  constructor(
    private readonly plugin: PluginUIContext,
    private readonly onChange: (state: PickerState) => void
  ) {}

  /** Start listening for right-button clicks. */
  attach(): void {
    if (this.subscription) return;
    this.subscription = this.plugin.behaviors.interaction.click.subscribe((event) => {
      if (event.button !== B.Secondary) return;
      this.handleClick(event.current.loci, event.modifiers);
    });
  }

  detach(): void {
    this.subscription?.unsubscribe();
    this.subscription = null;
  }

  /** The role catalog the backend serves; without it no role buttons appear. */
  setRegistry(payload: RolesPayload): void {
    this.payload = payload;
    this.engine = payload.default_engine || this.engine;
    this.emit();
  }

  /** The session selections are stored under. */
  setSession(sessionId: string | undefined): void {
    this.sessionId = sessionId;
  }

  /** Choose the engine the design is for; roles it cannot take are greyed out. */
  setEngine(engine: string): void {
    if (this.payload && !this.payload.engines.some((e) => e.key === engine)) return;
    this.engine = engine;
    this.emit();
  }

  /**
   * Forget everything: a different structure was loaded, so the picks and
   * roles mean nothing. Roles already stored are removed on the backend too,
   * so a later "use my roles" cannot pick up residues of the old structure.
   */
  reset(): void {
    const stale = [...this.layers, ...(this.editing ? [this.editing] : [])];
    for (const layer of stale) void this.removeOnBackend(layer);
    this.layers = [];
    this.editing = null;
    this.pending.clear();
    this.pendingCommittedSignature = '';
    this.pendingRef = null;
    this.polymerCache = null;
    this.atomNameCache = null;
    this.pendingAtomCache = null;
    this.notice = null;
    this.offered = null;
    this.schedulePaint();
    this.emit();
  }

  /** Total residues picked, pending and assigned. */
  get residueCount(): number {
    return this.pending.size + this.layers.reduce((n, l) => n + l.picks.size, 0);
  }

  /** What the viewer reports to the model about the user's picks. */
  describe(): { residueCount: number; description: string } {
    const parts = this.layers.map((l) => {
      const def = this.def(l.role);
      const name = l.value && def?.value_kind !== 'layout' ? `${def?.label ?? l.role} ${l.value}` : def?.label ?? l.role;
      return (
        `${name} role (set by the user): ${l.picks.size} residues, ` +
        `${this.atomCountOf(l.picks, l)} atoms, ${l.picks.describe()}`
      );
    });
    if (this.pending.size > 0) {
      parts.push(
        `pending, no role (picked by the user): ${this.pending.size} residues, ` +
          `${this.pendingAtomCount()} atoms, ${this.pending.describe()}`
      );
    }
    const engines = this.payload?.engines ?? [];
    const label = engines.find((e) => e.key === this.engine)?.label;
    if (this.engine !== this.payload?.default_engine && label && parts.length > 0) {
      parts.push(`design engine chosen by the user: ${label}`);
    }
    return { residueCount: this.residueCount, description: parts.join('; ') };
  }

  getState(): PickerState {
    const keep = this.layers.find((l) => l.role === 'keep');
    return {
      pending: {
        count: this.pending.size,
        atomCount: this.pendingAtomCount(),
        partialCount: this.pending.partialCount,
        description: this.pending.describe(),
        dirty:
          this.pending.size > 0 && this.pending.signature() !== this.pendingCommittedSignature,
      },
      layers: this.layers.map((l) => this.layerState(l)),
      options: this.options(),
      editingId: this.editing?.id ?? null,
      editingLabel: this.editing ? this.layerState(this.editing).label : null,
      engine: this.engine,
      engines: this.payload?.engines ?? [],
      pieces: keep ? keepPieces(keep.picks.list()) : [],
      layout: keep?.layout ?? null,
      notice: this.notice,
      busy: this.busy,
      offered: this.offered
        ? {
            label: this.offered.label,
            residueCount: this.offered.residues.length + new Set(this.offered.atoms.map((a) => `${a.chain}:${a.seq}`)).size,
            atomCount: this.offered.atoms.length,
          }
        : null,
    };
  }

  /**
   * Remember a selection the agent made, so the user can take it as a pick and
   * give it a role ("Use as"). Replaces any earlier offer.
   */
  offerAgentSelection(label: string, residues: LabelResidue[], atoms: LabelAtom[]): void {
    this.offered =
      residues.length === 0 && atoms.length === 0 ? null : { label, residues, atoms };
    this.emit();
  }

  /** Adopt the offered selection into the pending picks. */
  useOffered(): boolean {
    const offer = this.offered;
    if (!offer || this.busy) return false;
    const byKey = new Map(this.polymerResidues().map((r) => [`${r.labelChain}:${r.labelSeq}`, r]));
    let adopted = 0;
    for (const { chain, seq } of offer.residues) {
      const residue = byKey.get(`${chain}:${seq}`);
      if (!residue) continue;
      this.pending.adopt(residue, null, []);
      adopted++;
    }
    const grouped = new Map<string, string[]>();
    for (const { chain, seq, name } of offer.atoms) {
      const key = `${chain}:${seq}`;
      const names = grouped.get(key);
      if (names) names.push(name);
      else grouped.set(key, [name]);
    }
    grouped.forEach((names, key) => {
      const residue = byKey.get(key);
      if (!residue) return;
      this.pending.adopt(residue, names, this.atomNames(residue));
      adopted++;
    });
    this.notice =
      adopted === 0 ? 'None of the agent\'s selection is a protein residue in this structure.' : null;
    if (adopted > 0) this.offered = null;
    this.schedulePaint();
    this.emit();
    return adopted > 0;
  }

  /** The buttons for the current pick. */
  options(): RoleOption[] {
    const keep = this.layers.find((l) => l.role === 'keep');
    return roleOptions({
      payload: this.payload,
      engine: this.engine,
      picked: this.pickedAtoms(),
      keep: keep ? new Set(keep.picks.list().map(residueKey)) : null,
      floatGroups: this.layers.filter((l) => l.role === 'float').map((l) => l.value ?? ''),
    });
  }

  clear(): void {
    this.pending.clear();
    this.notice = null;
    this.schedulePaint();
    this.emit();
  }

  /** Grow the pending selection by every polymer residue within ``radius`` angstroms. */
  expand(radius: number): void {
    if (this.pending.size === 0) return;
    // Measured from the atoms actually picked, so a single chosen atom grows a
    // shell around that atom rather than around its whole residue.
    const near = getResidueInfo(
      this.plugin,
      withinDistanceQuery(this.queryOf(this.pending), { radius, wholeResidues: true })
    )
      .filter((r) => r.entityType === 'polymer' && Number.isFinite(r.seqNumber) && r.seqNumber > 0)
      .map((r) => this.fromInfo(r));
    const added = this.pending.addAll(near);
    this.notice = added === 0 ? `Nothing new within ${radius} Å.` : null;
    this.schedulePaint();
    this.emit();
  }

  // --- Roles ---------------------------------------------------------------

  /**
   * Give the pending selection a role, storing it on the backend.
   *
   * Picking again for a role that already has a layer adds to it; editing a
   * layer (see {@link editLayer}) replaces it. The composite Motif option
   * assigns four roles at once. On failure the pending selection is kept, so
   * the user can try again.
   *
   * @returns True if every role was stored.
   */
  async commitRole(optionId: string): Promise<boolean> {
    if (this.busy) return false;
    const option = this.options().find((o) => o.id === optionId);
    if (!option) return false;
    if (!option.enabled) {
      this.notice = option.reason ?? null;
      this.emit();
      return false;
    }
    const parts = option.composite
      ? option.composite.map((role) => ({ role, value: this.defaultValue(role) }))
      : [{ role: option.role, value: option.value }];

    this.busy = true;
    this.notice = null;
    this.emit();
    const editing = this.editing;
    const ids: string[] = [];
    try {
      for (const part of parts) {
        const stored = await this.storeRole(part.role, part.value, this.pending, editing);
        if (!stored) {
          this.notice = `Could not save ${option.label}. Try again.`;
          return false;
        }
        ids.push(stored.id);
      }
      // Re-roling an edited layer retires the old role.
      if (editing && !ids.includes(editing.id)) await this.removeOnBackend(editing);
      this.editing = null;
      this.pending.clear();
      this.pendingCommittedSignature = '';
      this.pendingRef = null;
      return true;
    } finally {
      this.busy = false;
      this.schedulePaint();
      this.emit();
    }
  }

  /** Lift a layer back into the pending set to change it. */
  editLayer(id: string): void {
    if (this.busy) return;
    const layer = this.layers.find((l) => l.id === id);
    if (!layer) return;
    // Anything pending now is set aside: editing replaces it.
    if (this.editing) this.layers.push(this.editing);
    this.layers = this.layers.filter((l) => l.id !== id);
    this.editing = layer;
    this.pending = layer.picks.clone();
    this.pendingCommittedSignature = this.pending.signature();
    this.pendingRef = null;
    this.notice = null;
    this.schedulePaint();
    this.emit();
  }

  /** Put an edited layer back as it was, discarding the changes. */
  cancelEdit(): void {
    if (!this.editing) return;
    this.layers.push(this.editing);
    this.editing = null;
    this.pending.clear();
    this.pendingCommittedSignature = '';
    this.schedulePaint();
    this.emit();
  }

  /** Remove a role from the design: stored as an empty selection naming the role. */
  async removeLayer(id: string): Promise<boolean> {
    if (this.busy) return false;
    const layer = this.editing?.id === id ? this.editing : this.layers.find((l) => l.id === id);
    if (!layer) return false;
    this.busy = true;
    this.emit();
    try {
      if (!(await this.removeOnBackend(layer))) {
        this.notice = `Could not remove ${this.def(layer.role)?.label ?? layer.role}. Try again.`;
        return false;
      }
      if (this.editing === layer) {
        this.editing = null;
        this.pending.clear();
      } else {
        this.layers = this.layers.filter((l) => l !== layer);
      }
      this.notice = null;
      return true;
    } finally {
      this.busy = false;
      this.schedulePaint();
      this.emit();
    }
  }

  /** Change the lengths of what is designed around the kept pieces. */
  async setLayout(layout: Layout | null): Promise<boolean> {
    const keep = this.layers.find((l) => l.role === 'keep');
    if (!keep || this.busy) return false;
    this.busy = true;
    this.emit();
    try {
      const previous = keep.layout;
      keep.layout = layoutIsEmpty(layout) ? null : layout;
      const ref = await this.post(this.roleBody('keep', null, keep.picks, keep.layout, keep.committedRef));
      if (!ref) {
        keep.layout = previous;
        this.notice = 'Could not save the layout. Try again.';
        return false;
      }
      keep.committedRef = ref;
      this.notice = null;
      return true;
    } finally {
      this.busy = false;
      this.emit();
    }
  }

  /**
   * Store the pending selection as a plain selection with no role.
   *
   * What the chat does just before it sends a message, so the agent can see
   * what the user pointed at even if they never chose a role for it. Role
   * layers are already stored; this only covers what is still pending.
   *
   * @returns References of the selections stored by this call.
   */
  async commit(sessionId?: string): Promise<string[]> {
    if (sessionId) this.sessionId = sessionId;
    const signature = this.pending.signature();
    if (this.pending.size === 0 || signature === this.pendingCommittedSignature) return [];
    const body = this.selectionBody(this.pending, {
      name: 'pending',
      parent: this.pendingRef,
    });
    if (!body) return [];
    const ref = await this.post(body);
    if (!ref) return [];
    this.pendingRef = ref;
    this.pendingCommittedSignature = signature;
    this.emit();
    return [ref];
  }

  // --- Storing -------------------------------------------------------------

  private defaultValue(role: string): string | null {
    return this.def(role)?.value_kind === 'group' ? DEFAULT_FLOAT_GROUP : null;
  }

  private def(role: string): RoleDef | undefined {
    return this.payload?.roles.find((r) => r.key === role);
  }

  private async storeRole(
    role: string,
    value: string | null,
    picks: PickSet,
    editing: RoleLayer | null
  ): Promise<RoleLayer | null> {
    const def = this.def(role);
    const id = layerId(def, role, value);
    const replacing = editing && editing.id === id ? editing : null;
    const existing = replacing ?? this.layers.find((l) => l.id === id) ?? null;

    // Adding to a role merges; editing replaces.
    const merged = picks.clone();
    if (existing && !replacing) {
      const base = existing.picks.clone();
      base.merge(merged);
      merged.clear();
      merged.merge(base);
    }
    const layout = existing?.layout ?? null;
    const ref = await this.post(this.roleBody(role, value, merged, layout, existing?.committedRef ?? null));
    if (!ref) return null;

    const layer: RoleLayer = {
      id,
      role,
      value,
      layout,
      picks: merged,
      committedRef: ref,
      atomCache: null,
    };
    this.layers = this.layers.filter((l) => l.id !== id);
    this.layers.push(layer);
    this.sortLayers();
    return layer;
  }

  private sortLayers(): void {
    const order = new Map((this.payload?.roles ?? []).map((r, i) => [r.key, i]));
    this.layers.sort((a, b) => (order.get(a.role) ?? 99) - (order.get(b.role) ?? 99));
  }

  /** Tell the backend a role is gone: an empty selection naming it. */
  private async removeOnBackend(layer: RoleLayer): Promise<boolean> {
    const ref = await this.post({
      molql: '',
      action: 'select',
      atoms: [],
      atom_count: 0,
      residue_count: 0,
      summary: 'Role removed',
      provenance: 'user',
      scope: 'residue',
      name: layer.role,
      role: layer.role,
      ...(layer.value ? { value: layer.value } : {}),
      ...(layer.committedRef ? { parent: layer.committedRef } : {}),
    });
    return ref !== null;
  }

  private roleBody(
    role: string,
    value: string | null,
    picks: PickSet,
    layout: Layout | null,
    parent: string | null
  ): Record<string, unknown> | null {
    const def = this.def(role);
    const body = this.selectionBody(picks, { name: role, parent });
    if (!body) return null;
    body.role = role;
    if (def?.value_kind === 'layout') {
      if (layout && !layoutIsEmpty(layout)) body.value = layout;
    } else if (value) {
      body.value = value;
    }
    return body;
  }

  private selectionBody(
    picks: PickSet,
    meta: { name: string; parent: string | null }
  ): Record<string, unknown> | null {
    const atoms = getAtomInfo(this.plugin, this.queryOf(picks));
    if (atoms.length === 0) return null;
    return {
      molql: '',
      action: 'select',
      atoms: serializeAtoms(atoms),
      atom_count: atoms.length,
      residue_count: picks.size,
      summary: `Selected ${atoms.length} atoms in ${picks.size} residues`,
      provenance: 'user',
      // "atom" when some residue holds only the atoms picked in it, so the
      // backend compares atoms rather than residues when this is combined.
      scope: picks.partialCount > 0 ? 'atom' : 'residue',
      name: meta.name,
      ...(meta.parent ? { parent: meta.parent } : {}),
    };
  }

  /** POST a selection; the reference it was given, or null on any failure. */
  private async post(body: Record<string, unknown> | null): Promise<string | null> {
    if (!body) return null;
    const url = this.sessionId
      ? `/api/selection?session_id=${encodeURIComponent(this.sessionId)}`
      : '/api/selection';
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        console.error(`Failed to store ${String(body.name)}: HTTP ${response.status}`);
        return null;
      }
      const result = (await response.json()) as { reference?: string };
      return result.reference ?? null;
    } catch (err) {
      console.error(`Failed to store ${String(body.name)}:`, err);
      return null;
    }
  }

  // --- Clicks --------------------------------------------------------------

  private handleClick(loci: Loci, modifiers: ModifiersKeys): void {
    // A bond is between two atoms and picks neither; say so rather than
    // silently doing nothing, because bonds are thick targets in ball-and-stick.
    if (Bond.isLoci(loci)) {
      this.notice = 'Click an atom, not a bond.';
      this.emit();
      return;
    }
    if (!StructureElement.Loci.is(loci) || StructureElement.Loci.isEmpty(loci)) return;

    const location = StructureElement.Loci.getFirstLocation(loci);
    if (!location) return;
    const residue = residueFromLocation(location);

    if (SP.entity.type(location) !== 'polymer' || !(residue.labelSeq > 0)) {
      this.notice = 'Only protein residues and atoms can be picked.';
      this.emit();
      return;
    }

    const layer = { picks: this.pending };
    this.notice = null;
    const residueLevel = modifiers.control || modifiers.meta || modifiers.shift;
    if (modifiers.control || modifiers.meta) {
      layer.picks.toggleChain(this.chainResidues(residue.labelChain));
    } else if (modifiers.shift) {
      layer.picks.extendTo(residue, this.chainResidues(residue.labelChain));
    } else if (!residueLevel && StructureElement.Loci.size(loci) === 1) {
      // One element is one atom: a click on a ball-and-stick atom.
      layer.picks.toggleAtom(residue, SP.atom.label_atom_id(location), this.atomNames(residue));
    } else {
      layer.picks.toggle(residue);
    }
    this.schedulePaint();
    this.emit();
  }

  /** The expression selecting a set's picks: whole residues plus chosen atoms. */
  private queryOf(picks: PickSet) {
    return picksQuery(
      picks.wholeResidues().map((r) => ({ chain: r.labelChain, seq: r.labelSeq })),
      picks.chosenAtoms()
    );
  }

  /** Atoms a set contributes, counted once per signature. */
  private atomCountOf(picks: PickSet, layer: RoleLayer): number {
    if (picks.size === 0) return 0;
    const signature = picks.signature();
    if (layer.atomCache?.signature === signature) return layer.atomCache.count;
    const count = getAtomInfo(this.plugin, this.queryOf(picks)).length;
    layer.atomCache = { signature, count };
    return count;
  }

  private pendingAtomCount(): number {
    if (this.pending.size === 0) return 0;
    const signature = this.pending.signature();
    if (this.pendingAtomCache?.signature === signature) return this.pendingAtomCache.count;
    const count = getAtomInfo(this.plugin, this.queryOf(this.pending)).length;
    this.pendingAtomCache = { signature, count };
    return count;
  }

  private layerState(l: RoleLayer): RoleLayerState {
    const def = this.def(l.role);
    const base = def?.label ?? l.role;
    const label = l.value && def && def.value_kind !== 'layout'
      ? `${def.variants.find((v) => v.value === l.value)?.label ?? base}${
          def.value_kind === 'group' && l.value !== DEFAULT_FLOAT_GROUP ? ` ${l.value}` : ''
        }`
      : base;
    return {
      id: l.id,
      role: l.role,
      value: l.value,
      label,
      colour: def?.colour ?? '#ff2e93',
      count: l.picks.size,
      atomCount: this.atomCountOf(l.picks, l),
      partialCount: l.picks.partialCount,
      description: l.picks.describe(),
    };
  }

  /** The pending picks with the atoms each contributes, for deciding which roles apply. */
  private pickedAtoms(): PickedAtoms[] {
    return this.pending.entries().map(({ residue, atoms }) => ({
      residue,
      atoms: atoms ?? this.atomNames(residue),
    }));
  }

  /** Names of every atom of a residue, unique (alternate locations repeat a name). */
  private atomNames(residue: PickedResidue): string[] {
    const structure = getStructure(this.plugin);
    if (!this.atomNameCache || this.atomNameCache.structure !== structure) {
      this.atomNameCache = { structure, names: new Map() };
    }
    const key = residueKey(residue);
    let names = this.atomNameCache.names.get(key);
    if (!names) {
      const atoms = getAtomInfo(
        this.plugin,
        residuesByLabelQuery([{ chain: residue.labelChain, seq: residue.labelSeq }])
      );
      names = Array.from(new Set(atoms.map((a) => a.name)));
      this.atomNameCache.names.set(key, names);
    }
    return names;
  }

  private polymerResidues(): PickedResidue[] {
    const structure = getStructure(this.plugin);
    if (this.polymerCache && this.polymerCache.structure === structure) {
      return this.polymerCache.residues;
    }
    const residues = getResidueInfo(this.plugin, polymerQuery())
      .filter((r) => Number.isFinite(r.seqNumber) && r.seqNumber > 0)
      .map((r) => this.fromInfo(r));
    this.polymerCache = { structure, residues };
    return residues;
  }

  private chainResidues(labelChain: string): PickedResidue[] {
    return this.polymerResidues().filter((r) => r.labelChain === labelChain);
  }

  private fromInfo(r: {
    chainId: string;
    seqNumber: number;
    authChainId: string;
    authSeqNumber: number;
    insCode?: string;
    name: string;
  }): PickedResidue {
    return {
      labelChain: r.chainId,
      labelSeq: r.seqNumber,
      authChain: r.authChainId,
      authSeq: r.authSeqNumber,
      insCode: r.insCode ?? '',
      name: r.name,
    };
  }

  private schedulePaint(): void {
    const toPaint = (picks: PickSet, colour: number): PaintLayer => ({
      residues: picks.wholeResidues().map((r) => ({ chain: r.labelChain, seq: r.labelSeq })),
      atoms: picks.chosenAtoms(),
      colour: colour as unknown as PaintLayer['colour'],
    });
    // Roles first, pending last, so what is being picked is never hidden.
    const layers: PaintLayer[] = [
      ...this.layers.map((l) =>
        toPaint(l.picks, parseHexColour(this.def(l.role)?.colour) ?? PICK_COLOUR)
      ),
      toPaint(this.pending, PICK_COLOUR),
    ];
    const anyPicks = layers.some((l) => l.residues.length > 0 || (l.atoms?.length ?? 0) > 0);
    this.paintQueue = this.paintQueue
      .then(() => (anyPicks ? paintLayers(this.plugin, layers) : clearPaint(this.plugin)))
      .catch((err) => console.error('Failed to paint picks:', err));
  }

  private emit(): void {
    this.onChange(this.getState());
  }
}
