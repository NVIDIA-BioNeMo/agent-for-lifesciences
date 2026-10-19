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

// Which role buttons a pick is offered. The catalog is a copy of what the
// backend serves (a backend test keeps the copy in sync).
import test from 'node:test';
import assert from 'node:assert/strict';

import registry from './design-roles.fixture.json';
import { residueKey, type PickedResidue } from '../src/lib/molstar/pickSet';
import {
  applyAtomRule,
  keepPieces,
  layerId,
  roleOptions,
  type OptionContext,
  type RoleDef,
  type RolesPayload,
} from '../src/lib/molstar/roles';

const payload = registry as unknown as RolesPayload;
const role = (key: string) => payload.roles.find((r) => r.key === key) as RoleDef;

function residue(name: string, seq: number, chain = 'A'): PickedResidue {
  return { labelChain: chain, labelSeq: seq, authChain: chain, authSeq: seq + 16, insCode: '', name };
}

const BACKBONE = ['N', 'CA', 'C', 'O', 'CB'];
const PHE_RING = ['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'];

function context(
  picked: Array<{ residue: PickedResidue; atoms: string[] }>,
  overrides: Partial<OptionContext> = {}
): OptionContext {
  return { payload, engine: 'rfd4', picked, keep: null, floatGroups: [], ...overrides };
}

const ids = (options: ReturnType<typeof roleOptions>, group?: 'primary' | 'more') =>
  options.filter((o) => !group || o.group === group).map((o) => o.id);

test('the five primary roles are always listed, disabled until something is picked', () => {
  const options = roleOptions(context([]));
  assert.deepEqual(ids(options, 'primary'), ['hotspot', 'keep', 'fix_position', 'fix_sequence', 'float:motif']);
  assert.ok(options.every((o) => !o.enabled && o.reason === 'Pick residues first.'));
  assert.deepEqual(ids(options, 'more'), [], 'nothing optional is offered for an empty pick');
});

test('with residues picked the primary roles are enabled and wear their plain labels', () => {
  const options = roleOptions(context([{ residue: residue('ALA', 5), atoms: BACKBONE }]));
  const primary = options.filter((o) => o.group === 'primary');
  assert.deepEqual(primary.map((o) => o.label), ['Hotspot', 'Keep', 'Fix position', 'Fix sequence', 'Float']);
  assert.ok(primary.every((o) => o.enabled));
  assert.match(primary[0].tooltip, /C_HOT/, 'the RFD4 term is in the tooltip');
});

test('Pi-stack is offered for a complete ring and never for a pick without one', () => {
  const ring = roleOptions(context([{ residue: residue('PHE', 5), atoms: [...BACKBONE, ...PHE_RING] }]));
  assert.ok(ids(ring).includes('pistack'));
  const none = roleOptions(context([{ residue: residue('ALA', 5), atoms: BACKBONE }]));
  assert.ok(!ids(none).includes('pistack'));
  const half = roleOptions(context([{ residue: residue('PHE', 5), atoms: ['CG', 'CD1'] }]));
  assert.ok(!ids(half).includes('pistack'));
});

test('H-bond buttons follow the atoms picked', () => {
  const hydroxyl = roleOptions(context([{ residue: residue('TYR', 5), atoms: ['OH'] }]));
  assert.deepEqual(
    ids(hydroxyl).filter((id) => id.startsWith('hbond')),
    ['hbond:DONOR', 'hbond:ACCEPTOR', 'hbond:BOTH']
  );
  const carbonyl = roleOptions(context([{ residue: residue('ALA', 5), atoms: ['O'] }]));
  assert.deepEqual(ids(carbonyl).filter((id) => id.startsWith('hbond')), ['hbond:ACCEPTOR']);
});

test('salt-bridge charge follows the residue', () => {
  const asp = roleOptions(context([{ residue: residue('ASP', 5), atoms: ['CG', 'OD1', 'OD2'] }]));
  assert.deepEqual(ids(asp).filter((id) => id.startsWith('salt')), ['salt:NEGATIVE']);
  const lys = roleOptions(context([{ residue: residue('LYS', 5), atoms: ['NZ'] }]));
  assert.deepEqual(ids(lys).filter((id) => id.startsWith('salt')), ['salt:POSITIVE']);
});

test('burial and secondary structure are offered for any residue pick, under More', () => {
  const options = roleOptions(context([{ residue: residue('ALA', 5), atoms: BACKBONE }]));
  assert.ok(ids(options, 'more').includes('rasa:BURIED'));
  assert.ok(ids(options, 'more').includes('secondary_structure:HELIX'));
  assert.ok(!ids(options).includes('secondary_structure:STRAND'), 'the NIM has no STRAND');
});

test('the Motif composite assigns four roles and is offered under More', () => {
  const options = roleOptions(context([{ residue: residue('ALA', 5), atoms: BACKBONE }]));
  const motif = options.find((o) => o.id === 'motif');
  assert.deepEqual(motif?.composite, ['keep', 'fix_position', 'fix_sequence', 'float']);
  assert.equal(motif?.group, 'more');
});

test('a residue outside Keep cannot be a hotspot, and the reason names it', () => {
  const keep = new Set([residueKey(residue('ALA', 5))]);
  const options = roleOptions(
    context([{ residue: residue('ALA', 9), atoms: BACKBONE }], { keep })
  );
  const hotspot = options.find((o) => o.id === 'hotspot')!;
  assert.equal(hotspot.enabled, false);
  assert.match(hotspot.reason ?? '', /Outside Keep: A25/);
  assert.equal(options.find((o) => o.id === 'keep')!.enabled, true, 'Keep itself is never outside Keep');
});

test('RFdiffusion greys out Float and offers no chemistry', () => {
  const options = roleOptions(
    context([{ residue: residue('TYR', 5), atoms: ['OH'] }], { engine: 'rfdiffusion1' })
  );
  const float = options.find((o) => o.id === 'float:motif')!;
  assert.equal(float.enabled, false);
  assert.match(float.reason ?? '', /RFD4/);
  assert.ok(!ids(options).some((id) => id.startsWith('hbond') || id.startsWith('rasa') || id === 'motif'));
  assert.ok(options.find((o) => o.id === 'hotspot')!.enabled);
});

test('a second Float group is offered only once the first exists', () => {
  const pick = [{ residue: residue('ALA', 5), atoms: BACKBONE }];
  assert.ok(!ids(roleOptions(context(pick))).includes('float:g2'));
  const options = roleOptions(context(pick, { floatGroups: ['motif'] }));
  assert.ok(ids(options, 'more').includes('float:g2'));
  assert.ok(ids(roleOptions(context(pick, { floatGroups: ['motif', 'g2'] }))).includes('float:g3'));
});

test('a layer is identified by its role and, for roles with a parameter, the parameter', () => {
  assert.equal(layerId(role('hotspot'), 'hotspot', null), 'hotspot');
  assert.equal(layerId(role('rasa'), 'rasa', 'BURIED'), 'rasa:BURIED');
  assert.equal(layerId(role('float'), 'float', 'g2'), 'float:g2');
  assert.equal(layerId(role('keep'), 'keep', 'ignored'), 'keep');
});

test('applyAtomRule narrows and refuses as the backend does', () => {
  const ring = role('pistack').atom_rule;
  assert.deepEqual(applyAtomRule(ring, 'PHE', [...BACKBONE, ...PHE_RING]), PHE_RING);
  assert.equal(applyAtomRule(ring, 'ALA', BACKBONE), null);
  const donor = role('hbond').variants.find((v) => v.value === 'DONOR')!.atom_rule;
  assert.deepEqual(applyAtomRule(donor, 'SER', ['N', 'CA', 'OG']), ['N', 'OG']);
  assert.equal(applyAtomRule(donor, 'PRO', ['N', 'CA']), null);
  assert.deepEqual(applyAtomRule(null, 'ALA', ['CA']), ['CA']);
});

test('kept residues group into the stretches the contig will name', () => {
  const kept = [5, 6, 7, 20, 21].map((n) => residue('ALA', n));
  const pieces = keepPieces(kept);
  assert.deepEqual(pieces.map((p) => [p.chain, p.firstLabel, p.lastLabel, p.length]), [
    ['A', 5, 7, 3],
    ['A', 20, 21, 2],
  ]);
  assert.equal(pieces[0].firstAuth, 21);
});
