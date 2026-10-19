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
import { StructureElement, StructureProperties as SP, StructureSelection } from 'molstar/lib/mol-model/structure';

import { atomsByLabelQuery, picksQuery, residuesByLabelQuery, runQuery } from '../src/lib/molstar/queries';
import { showAtoms, showResidues } from '../src/lib/molstar/residues';
import { CIF } from './fixture';
import { fakePlugin, loadStructure } from './harness';

/** Atom names (label ids) an expression selects, as `chain:seq:name`. */
async function select(expression: ReturnType<typeof atomsByLabelQuery>): Promise<string[]> {
  const structure = await loadStructure(CIF);
  const loci = StructureSelection.toLociWithSourceUnits(runQuery(expression, structure));
  const out: string[] = [];
  StructureElement.Loci.forEachLocation(loci, (l) => {
    out.push(`${SP.chain.label_asym_id(l)}:${SP.residue.label_seq_id(l)}:${SP.atom.label_atom_id(l)}`);
  });
  return out.sort();
}

test('atomsByLabelQuery selects only the named atoms', async () => {
  const got = await select(
    atomsByLabelQuery([
      { chain: 'A', seq: 2, name: 'OH' },
      { chain: 'A', seq: 2, name: 'CZ' },
    ])
  );
  assert.deepEqual(got, ['A:2:CZ', 'A:2:OH']);
});

test('atoms with different names in different residues stay with their own residue', async () => {
  // OH belongs to residue 2 and SD to residue 3; neither may leak across.
  const got = await select(
    atomsByLabelQuery([
      { chain: 'A', seq: 2, name: 'OH' },
      { chain: 'A', seq: 3, name: 'SD' },
    ])
  );
  assert.deepEqual(got, ['A:2:OH', 'A:3:SD']);
});

test('residues sharing a name set are tested together without selecting extra atoms', async () => {
  const got = await select(
    atomsByLabelQuery([
      { chain: 'A', seq: 2, name: 'CA' },
      { chain: 'A', seq: 3, name: 'CA' },
      { chain: 'B', seq: 1, name: 'CA' },
    ])
  );
  assert.deepEqual(got, ['A:2:CA', 'A:3:CA', 'B:1:CA']);
});

test('an atom name the residue lacks selects nothing', async () => {
  assert.deepEqual(await select(atomsByLabelQuery([{ chain: 'A', seq: 4, name: 'OH' }])), []);
});

test('an empty atom list matches nothing, not everything', async () => {
  assert.deepEqual(await select(atomsByLabelQuery([])), []);
});

test('picksQuery mixes whole residues and chosen atoms', async () => {
  const got = await select(
    picksQuery([{ chain: 'A', seq: 1 }], [{ chain: 'A', seq: 2, name: 'OH' }])
  );
  assert.deepEqual(got, ['A:1:CA', 'A:1:N', 'A:2:OH']);
});

test('picksQuery with no atoms is the residue query', async () => {
  const a = await select(picksQuery([{ chain: 'A', seq: 3 }], []));
  const b = await select(residuesByLabelQuery([{ chain: 'A', seq: 3 }]));
  assert.deepEqual(a, b);
});

test('showAtoms selects, optionally focuses, and counts what it found', async () => {
  const structure = await loadStructure(CIF);
  const { plugin, shown } = fakePlugin(structure);
  const counts = showAtoms(
    plugin,
    [{ chain: 'A', seq: 1 }, { chain: 'A', seq: 99 }],
    [
      { chain: 'A', seq: 2, name: 'OH' },
      { chain: 'A', seq: 2, name: 'NZ' },
    ],
    { focus: true }
  );
  assert.deepEqual(counts, { residues: 1, atoms: 1 }); // seq 99 and atom NZ are not there
  assert.equal(shown.selected.length, 1);
  assert.equal(shown.focused.length, 1);
});

test('showAtoms without focus leaves the camera alone', async () => {
  const structure = await loadStructure(CIF);
  const { plugin, shown } = fakePlugin(structure);
  showAtoms(plugin, [], [{ chain: 'A', seq: 2, name: 'OH' }]);
  assert.equal(shown.selected.length, 1);
  assert.equal(shown.focused.length, 0);
});

test('showResidues is showAtoms with no atoms', async () => {
  const structure = await loadStructure(CIF);
  const { plugin } = fakePlugin(structure);
  assert.equal(showResidues(plugin, [{ chain: 'A', seq: 1 }, { chain: 'B', seq: 2 }]), 2);
});
