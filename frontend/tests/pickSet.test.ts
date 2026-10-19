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
import { StructureElement, StructureProperties as SP } from 'molstar/lib/mol-model/structure';
import { StructureSelection } from 'molstar/lib/mol-model/structure';

import { PickSet, type PickedResidue } from '../src/lib/molstar/pickSet';
import { residuesByLabelQuery, runQuery } from '../src/lib/molstar/queries';
import { CIF } from './fixture';
import { loadStructure } from './harness';


function res(seq: number, chain = 'A', auth = seq + 16, name = 'ALA', ins = ''): PickedResidue {
  return { labelChain: chain, labelSeq: seq, authChain: chain, authSeq: auth, insCode: ins, name };
}

const chainA = [1, 2, 3, 4, 5, 6, 7, 8].map((n) => res(n));

test('toggle adds then removes', () => {
  const s = new PickSet();
  assert.equal(s.toggle(res(3)), true);
  assert.equal(s.toggle(res(3)), false);
  assert.equal(s.size, 0);
});

test('extendTo spans from the last pick in either direction and only adds', () => {
  const s = new PickSet();
  s.toggle(res(6));
  assert.equal(s.extendTo(res(3), chainA), 3);
  assert.deepEqual(s.list().map((r) => r.labelSeq).sort(), [3, 4, 5, 6]);
  // a second extend adds, never removes
  assert.equal(s.extendTo(res(8), chainA), 2);
  assert.equal(s.size, 6);
});

test('extendTo with no previous pick, or one on another chain, takes only the target', () => {
  const s = new PickSet();
  assert.equal(s.extendTo(res(4), chainA), 1);
  const t = new PickSet();
  t.toggle(res(1, 'B'));
  assert.equal(t.extendTo(res(4), chainA), 1);
});

test('toggleChain takes a chain, then drops it', () => {
  const s = new PickSet();
  s.toggle(res(2));
  assert.equal(s.toggleChain(chainA), true);
  assert.equal(s.size, 8);
  assert.equal(s.toggleChain(chainA), false);
  assert.equal(s.size, 0);
});

test('signature ignores order and notices change', () => {
  const a = new PickSet();
  a.toggle(res(1));
  a.toggle(res(2));
  const b = new PickSet();
  b.toggle(res(2));
  b.toggle(res(1));
  assert.equal(a.signature(), b.signature());
  b.toggle(res(3));
  assert.notEqual(a.signature(), b.signature());
});

test('describe lists author numbers by author chain and caps the listing', () => {
  const s = new PickSet();
  s.addAll([res(40, 'A', 56, 'TYR'), res(99, 'A', 115, 'MET'), res(10, 'B', 52, 'SER', 'A')]);
  assert.equal(s.describe(), 'A: Y56 M115; B: S52A');
  assert.equal(s.describe(2), 'A: Y56 M115 (+1 more)');
});

test('residuesByLabelQuery selects by label ids, not author ids', async () => {
  const structure = await loadStructure(CIF);
  // Label seq 2 on chain A is the Tyr whose author number is 18.
  const selection = runQuery(residuesByLabelQuery([{ chain: 'A', seq: 2 }]), structure);
  const loci = StructureSelection.toLociWithSourceUnits(selection);
  const loc = StructureElement.Loci.getFirstLocation(loci)!;
  assert.equal(SP.atom.label_comp_id(loc), 'TYR');
  assert.equal(SP.residue.auth_seq_id(loc), 18);
  assert.equal(StructureElement.Loci.size(loci), 6);
});

test('residuesByLabelQuery handles several chains and an empty list', async () => {
  const structure = await loadStructure(CIF);
  const two = runQuery(
    residuesByLabelQuery([
      { chain: 'A', seq: 1 },
      { chain: 'B', seq: 2 },
    ]),
    structure
  );
  assert.equal(StructureElement.Loci.size(StructureSelection.toLociWithSourceUnits(two)), 4);
  const none = runQuery(residuesByLabelQuery([]), structure);
  assert.equal(StructureElement.Loci.size(StructureSelection.toLociWithSourceUnits(none)), 0);
});

const allTyr = ['N', 'CA', 'CB', 'CG', 'CZ', 'OH'];

test('toggleAtom starts a partial residue and tracks its atoms', () => {
  const s = new PickSet();
  assert.equal(s.toggleAtom(res(40, 'A', 56, 'TYR'), 'OH', allTyr), true);
  assert.equal(s.size, 1);
  assert.equal(s.partialCount, 1);
  assert.equal(s.isWhole(res(40)), false);
  assert.deepEqual(s.atomsOf(res(40)), ['OH']);
  assert.deepEqual(s.chosenAtoms(), [{ chain: 'A', seq: 40, name: 'OH' }]);
  assert.deepEqual(s.wholeResidues(), []);
});

test('toggleAtom removes the residue when its last atom goes', () => {
  const s = new PickSet();
  const tyr = res(40, 'A', 56, 'TYR');
  s.toggleAtom(tyr, 'OH', allTyr);
  assert.equal(s.toggleAtom(tyr, 'OH', allTyr), false);
  assert.equal(s.size, 0);
});

test('toggleAtom on a whole residue keeps every other atom', () => {
  const s = new PickSet();
  const tyr = res(40, 'A', 56, 'TYR');
  s.toggle(tyr);
  s.toggleAtom(tyr, 'OH', allTyr);
  assert.equal(s.isWhole(tyr), false);
  assert.deepEqual(s.atomsOf(tyr).sort(), ['CA', 'CB', 'CG', 'CZ', 'N']);
});

test('every atom picked is the same state as the whole residue', () => {
  const a = new PickSet();
  const tyr = res(40, 'A', 56, 'TYR');
  allTyr.forEach((n) => a.toggleAtom(tyr, n, allTyr));
  const b = new PickSet();
  b.toggle(tyr);
  assert.equal(a.isWhole(tyr), true);
  assert.equal(a.signature(), b.signature());
});

test('toggle completes a partial residue and then removes it', () => {
  const s = new PickSet();
  const tyr = res(40, 'A', 56, 'TYR');
  s.toggleAtom(tyr, 'OH', allTyr);
  assert.equal(s.toggle(tyr), true);
  assert.equal(s.isWhole(tyr), true);
  assert.equal(s.toggle(tyr), false);
  assert.equal(s.size, 0);
});

test('extendTo and toggleChain take whole residues over partial ones', () => {
  const s = new PickSet();
  s.toggleAtom(res(3, 'A', 19, 'ALA'), 'CA', ['N', 'CA', 'C', 'O', 'CB']);
  s.toggle(res(1));
  assert.equal(s.extendTo(res(5), chainA), 4); // 2, 3 (completed), 4, 5 -- 1 was already whole
  assert.equal(s.partialCount, 0);
  const t = new PickSet();
  t.toggleAtom(res(3), 'CA', ['N', 'CA']);
  assert.equal(t.toggleChain(chainA), true);
  assert.equal(t.partialCount, 0);
});

test('addAll leaves residues that are already picked, partial or not, as they are', () => {
  const s = new PickSet();
  s.toggleAtom(res(3), 'CA', ['N', 'CA']);
  assert.equal(s.addAll([res(3), res(4)]), 1);
  assert.equal(s.partialCount, 1);
});

test('signature tells atoms apart and ignores the order they were picked in', () => {
  const tyr = res(40, 'A', 56, 'TYR');
  const a = new PickSet();
  a.toggleAtom(tyr, 'OH', allTyr);
  a.toggleAtom(tyr, 'CZ', allTyr);
  const b = new PickSet();
  b.toggleAtom(tyr, 'CZ', allTyr);
  b.toggleAtom(tyr, 'OH', allTyr);
  assert.equal(a.signature(), b.signature());
  b.toggleAtom(tyr, 'CB', allTyr);
  assert.notEqual(a.signature(), b.signature());
});

test('describe names the chosen atoms and caps a long list', () => {
  const s = new PickSet();
  const tyr = res(40, 'A', 56, 'TYR');
  s.toggleAtom(tyr, 'OH', allTyr);
  s.toggleAtom(tyr, 'CZ', allTyr);
  s.toggle(res(99, 'A', 115, 'MET'));
  assert.equal(s.describe(), 'A: Y56[OH,CZ] M115');
  ['CB', 'CG', 'CA'].forEach((n) => s.toggleAtom(tyr, n, allTyr));
  assert.equal(s.describe(), 'A: Y56[OH,CZ,CB,CG,+1] M115');
});
