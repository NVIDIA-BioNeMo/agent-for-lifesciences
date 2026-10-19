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
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { ButtonsType } from 'molstar/lib/mol-util/input/input-observer';

import { ResiduePicker, withRightClickFreed, type PickerState } from '../src/lib/molstar/residuePicker';
import { CIF } from './fixture';
import registry from './design-roles.fixture.json';
import type { RolesPayload } from '../src/lib/molstar/roles';
import { Bond } from 'molstar/lib/mol-model/structure';
import { atomsByLabelQuery } from '../src/lib/molstar/queries';
import { clickLoci, clickResidue, fakePlugin, loadStructure } from './harness';

const P = MS.struct.generator.atomGroups;
const Q = MS.core.rel.eq;
const prop = MS.struct.atomProperty;

/** Expression for one residue by label ids (the first atom group matching). */
function labelResidue(chain: string, seq: number) {
  return P({
    'chain-test': Q([prop.macromolecular.label_asym_id(), chain]),
    'residue-test': Q([prop.macromolecular.label_seq_id(), seq]),
  });
}

function ligand() {
  return P({ 'chain-test': Q([prop.macromolecular.label_asym_id(), 'C']) });
}

async function setup() {
  const structure = await loadStructure(CIF);
  const { plugin, click } = fakePlugin(structure);
  let last: PickerState | undefined;
  const picker = new ResiduePicker(plugin, (state) => {
    last = state;
  });
  picker.attach();
  return { structure, plugin, click, picker, state: () => last! };
}

const SECONDARY = ButtonsType.Flag.Secondary;
const PRIMARY = ButtonsType.Flag.Primary;

test('a right click picks the residue, and a second one unpicks it', async () => {
  const { structure, click, picker, state } = await setup();
  clickResidue(click, structure, labelResidue('A', 2));
  assert.equal(picker.residueCount, 1);
  assert.equal(state().pending.count, 1);
  assert.equal(state().pending.dirty, true);
  clickResidue(click, structure, labelResidue('A', 2));
  assert.equal(picker.residueCount, 0);
});

test('the left button is left to Mol*', async () => {
  const { structure, click, picker } = await setup();
  clickResidue(click, structure, labelResidue('A', 2), { button: PRIMARY });
  assert.equal(picker.residueCount, 0);
});

test('picks are described in author numbering, which is what the viewer shows', async () => {
  const { structure, click, picker } = await setup();
  clickResidue(click, structure, labelResidue('A', 2)); // Tyr, author 18
  clickResidue(click, structure, labelResidue('B', 1)); // Gly, author 5 on author chain C
  const { residueCount, description } = picker.describe();
  assert.equal(residueCount, 2);
  assert.match(description, /A: Y18/);
  assert.match(description, /C: G5/);
});

test('shift extends along the chain from the last pick', async () => {
  const { structure, click, picker } = await setup();
  clickResidue(click, structure, labelResidue('A', 1));
  clickResidue(click, structure, labelResidue('A', 4), { modifiers: { shift: true } });
  assert.equal(picker.residueCount, 4);
});

test('ctrl takes the whole chain and a second ctrl-click drops it', async () => {
  const { structure, click, picker } = await setup();
  clickResidue(click, structure, labelResidue('A', 3), { modifiers: { control: true } });
  assert.equal(picker.residueCount, 5);
  clickResidue(click, structure, labelResidue('A', 3), { modifiers: { control: true } });
  assert.equal(picker.residueCount, 0);
});

test('ligands and water are refused with a notice', async () => {
  const { structure, click, picker, state } = await setup();
  clickResidue(click, structure, ligand());
  assert.equal(picker.residueCount, 0);
  assert.match(state().notice ?? '', /protein/i);
});

test('expand adds polymer residues within the radius and never the ligand', async () => {
  const { structure, click, picker } = await setup();
  clickResidue(click, structure, labelResidue('A', 2));
  picker.expand(4.5);
  const labels = picker.describe().description;
  // Residues 1 and 3 on chain A are 4 A away; chain B's first residue is ~3 A away.
  assert.ok(picker.residueCount >= 3, labels);
  assert.doesNotMatch(labels, /LIG|D101/);
});

test('reset forgets the picks', async () => {
  const { structure, click, picker } = await setup();
  clickResidue(click, structure, labelResidue('A', 2));
  picker.reset();
  assert.equal(picker.residueCount, 0);
});

test('detach stops listening', async () => {
  const { structure, click, picker } = await setup();
  picker.detach();
  clickResidue(click, structure, labelResidue('A', 2));
  assert.equal(picker.residueCount, 0);
});

/** Expression for one atom, by label ids: what a click on a ball-and-stick atom reports. */
function atom(chain: string, seq: number, name: string) {
  return atomsByLabelQuery([{ chain, seq, name }]);
}

test('a right click on an atom picks just that atom', async () => {
  const { structure, click, picker, state } = await setup();
  clickResidue(click, structure, atom('A', 2, 'OH'));
  assert.equal(picker.residueCount, 1);
  assert.equal(picker.describe().description.includes('A: Y18[OH]'), true, picker.describe().description);
  const layer = state().pending;
  assert.deepEqual([layer.count, layer.atomCount, layer.partialCount], [1, 1, 1]);
});

test('atoms of one residue accumulate and the last one off removes the residue', async () => {
  const { structure, click, picker, state } = await setup();
  clickResidue(click, structure, atom('A', 2, 'OH'));
  clickResidue(click, structure, atom('A', 2, 'CZ'));
  assert.deepEqual([state().pending.count, state().pending.atomCount], [1, 2]);
  clickResidue(click, structure, atom('A', 2, 'OH'));
  clickResidue(click, structure, atom('A', 2, 'CZ'));
  assert.equal(picker.residueCount, 0);
});

test('an atom click on a whole residue drops just that atom', async () => {
  const { structure, click, state } = await setup();
  clickResidue(click, structure, labelResidue('A', 2)); // ribbon: all 6 atoms
  assert.equal(state().pending.atomCount, 6);
  clickResidue(click, structure, atom('A', 2, 'OH'));
  const layer = state().pending;
  assert.deepEqual([layer.count, layer.atomCount, layer.partialCount], [1, 5, 1]);
});

test('picking every atom of a residue makes it whole again', async () => {
  const { structure, click, state } = await setup();
  clickResidue(click, structure, atom('B', 1, 'N'));
  assert.equal(state().pending.partialCount, 1);
  clickResidue(click, structure, atom('B', 1, 'CA')); // Gly in the fixture has two atoms
  assert.deepEqual([state().pending.partialCount, state().pending.atomCount], [0, 2]);
});

test('a ribbon click completes a partly picked residue', async () => {
  const { structure, click, state } = await setup();
  clickResidue(click, structure, atom('A', 2, 'OH'));
  clickResidue(click, structure, labelResidue('A', 2));
  assert.deepEqual([state().pending.partialCount, state().pending.atomCount], [0, 6]);
});

test('shift and ctrl keep their residue and chain meaning on an atom', async () => {
  const { structure, click, picker } = await setup();
  clickResidue(click, structure, atom('A', 3, 'SD'), { modifiers: { control: true } });
  assert.equal(picker.residueCount, 5); // the whole of chain A
});

test('a bond click picks nothing and says so', async () => {
  const { structure, click, picker, state } = await setup();
  const unit = structure.units[0];
  const loci = Bond.Loci(structure, [
    Bond.Location(structure, unit, unit.bonds.a[0], structure, unit, unit.bonds.b[0]),
  ]);
  clickLoci(click, loci);
  assert.equal(picker.residueCount, 0);
  assert.match(state().notice ?? '', /atom, not a bond/i);
});

test('an atom of the ligand is refused', async () => {
  const { structure, click, picker, state } = await setup();
  clickResidue(click, structure, atomsByLabelQuery([{ chain: 'C', seq: 0, name: 'C1' }]));
  assert.equal(picker.residueCount, 0);
  assert.match(state().notice ?? '', /protein/i);
});

test('expand grows from the picked atom, not from its whole residue', async () => {
  const { structure, click, picker } = await setup();
  clickResidue(click, structure, atom('A', 2, 'OH')); // OH is 3 A from chain B's first residue
  picker.expand(3.5);
  const text = picker.describe().description;
  // Chain B (author C) is within reach of the OH; residue A1 is near the Tyr's N, not the OH.
  assert.match(text, /C: G5/);
  assert.doesNotMatch(text, /C17|A1\b/);
  assert.equal(picker.residueCount, 2);
});

interface Posted {
  url: string;
  body: Record<string, any>;
}

function stubFetch(responses: Array<{ ok: boolean; reference?: string }>): Posted[] {
  const posted: Posted[] = [];
  globalThis.fetch = (async (url: string, init: { body: string }) => {
    posted.push({ url, body: JSON.parse(init.body) });
    const next = responses.shift() ?? { ok: true, reference: 'selection_default' };
    return { ok: next.ok, status: next.ok ? 200 : 500, json: async () => ({ reference: next.reference }) };
  }) as unknown as typeof fetch;
  return posted;
}

test('commit posts the picks as a user selection carrying both numberings', async () => {
  const { structure, click, picker, state } = await setup();
  const posted = stubFetch([{ ok: true, reference: 'selection_aaa' }]);
  clickResidue(click, structure, labelResidue('A', 2));
  clickResidue(click, structure, labelResidue('B', 1));

  const stored = await picker.commit('sess 1');
  assert.deepEqual(stored, ['selection_aaa']);
  assert.equal(posted.length, 1);
  assert.equal(posted[0].url, '/api/selection?session_id=sess%201');
  const body = posted[0].body;
  assert.equal(body.provenance, 'user');
  assert.equal(body.scope, 'residue');
  assert.equal(body.name, 'pending');
  assert.equal(body.role, undefined);
  assert.equal(body.residue_count, 2);
  assert.equal(body.parent, undefined);
  assert.equal(body.query_id, undefined);
  assert.equal(state().pending.dirty, false);

  const atoms = body.atoms as Array<Record<string, any>>;
  assert.equal(atoms.length, 8); // Tyr has 6 atoms, Gly 2
  const tyr = atoms.find((a) => JSON.stringify(a).includes('TYR'))!;
  // Label numbering (2) and author numbering (18) are both present and differ.
  const text = JSON.stringify(tyr);
  assert.ok(text.includes('18'), text);
});

test('committing atom picks posts only the chosen atoms, as an atom-scoped selection', async () => {
  const { structure, click, picker } = await setup();
  const posted = stubFetch([{ ok: true, reference: 'selection_atoms' }]);
  clickResidue(click, structure, atom('A', 2, 'OH'));
  clickResidue(click, structure, atom('A', 2, 'CZ'));
  clickResidue(click, structure, labelResidue('B', 1)); // plus one whole residue

  await picker.commit('s');
  const body = posted[0].body;
  assert.equal(body.scope, 'atom');
  assert.equal(body.residue_count, 2);
  const names = (body.atoms as Array<Record<string, any>>).map((a) => `${a.residue.chainId}:${a.residue.seqNumber}:${a.name}`);
  assert.deepEqual(names.sort(), ['A:2:CZ', 'A:2:OH', 'B:1:CA', 'B:1:N']);
  // Both numberings travel with every atom.
  const oh = (body.atoms as Array<Record<string, any>>).find((a) => a.name === 'OH')!;
  assert.equal(oh.residue.authSeqNumber, 18);
});

test('whole-residue picks are still posted as a residue-scoped selection', async () => {
  const { structure, click, picker } = await setup();
  const posted = stubFetch([{ ok: true, reference: 'selection_whole' }]);
  clickResidue(click, structure, labelResidue('A', 2));
  await picker.commit('s');
  assert.equal(posted[0].body.scope, 'residue');
  assert.equal(posted[0].body.atoms.length, 6);
});

test('an unchanged layer is not posted twice, and an edit names the last one as parent', async () => {
  const { structure, click, picker } = await setup();
  const posted = stubFetch([
    { ok: true, reference: 'selection_one' },
    { ok: true, reference: 'selection_two' },
  ]);
  clickResidue(click, structure, labelResidue('A', 2));
  await picker.commit('s');
  await picker.commit('s');
  assert.equal(posted.length, 1);

  clickResidue(click, structure, labelResidue('A', 3));
  await picker.commit('s');
  assert.equal(posted.length, 2);
  assert.equal(posted[1].body.parent, 'selection_one');
});

test('a failed commit leaves the picks dirty so the next send retries', async () => {
  const { structure, click, picker, state } = await setup();
  const posted = stubFetch([{ ok: false }, { ok: true, reference: 'selection_ok' }]);
  const errors = console.error;
  console.error = () => {};
  try {
    clickResidue(click, structure, labelResidue('A', 2));
    assert.deepEqual(await picker.commit('s'), []);
    assert.equal(state().pending.dirty, true);
  } finally {
    console.error = errors;
  }
  assert.deepEqual(await picker.commit('s'), ['selection_ok']);
  assert.equal(posted.length, 2);
});

test('with nothing picked, commit posts nothing', async () => {
  const { picker } = await setup();
  const posted = stubFetch([]);
  assert.deepEqual(await picker.commit('s'), []);
  assert.equal(posted.length, 0);
});

test('the freed spec keeps the left button and drops the right on camera focus', async () => {
  const { Binding } = await import('molstar/lib/mol-util/binding');
  const { ModifiersKeys } = await import('molstar/lib/mol-util/input/input-observer');
  const { PluginBehaviors } = await import('molstar/lib/mol-plugin/behavior');

  const spec = withRightClickFreed();
  const focus = spec.behaviors.find((b) => b.transformer === PluginBehaviors.Camera.FocusLoci)!;
  const bindings = (focus.defaultParams as any).bindings;
  const none = ModifiersKeys.create();
  assert.ok(Binding.match(bindings.clickCenterFocus, PRIMARY, none));
  assert.ok(!Binding.match(bindings.clickCenterFocus, SECONDARY, none));
  assert.ok(!Binding.match(bindings.clickResetCameraOnEmpty, SECONDARY, none));

  // The stock spec binds the right button, which is why it needs freeing.
  const stock = (await import('molstar/lib/mol-plugin-ui/spec')).DefaultPluginUISpec();
  const stockFocus = stock.behaviors.find((b) => b.transformer === PluginBehaviors.Camera.FocusLoci)!;
  const { ParamDefinition } = await import('molstar/lib/mol-util/param-definition');
  const stockBindings = (ParamDefinition.getDefaultValues(PluginBehaviors.Camera.FocusLoci.definition.params!(undefined as any)) as any).bindings;
  assert.ok(Binding.match(stockBindings.clickCenterFocus, SECONDARY, none));
});


// --- Roles -----------------------------------------------------------------

async function setupRoles() {
  const made = await setup();
  made.picker.setRegistry(registry as unknown as RolesPayload);
  made.picker.setSession('s');
  return made;
}

const optionIds = (state: PickerState) => state.options.map((o) => o.id);

test('pressing a role stores the pick with that role, clears it and puts a chip in the tray', async () => {
  const { structure, click, picker, state } = await setupRoles();
  const posted = stubFetch([{ ok: true, reference: 'selection_hot' }]);
  clickResidue(click, structure, labelResidue('A', 2));

  assert.equal(await picker.commitRole('hotspot'), true);

  assert.equal(posted.length, 1);
  const body = posted[0].body;
  assert.equal(body.role, 'hotspot');
  assert.equal(body.name, 'hotspot');
  assert.equal(body.provenance, 'user');
  assert.equal(body.residue_count, 1);
  assert.equal(body.parent, undefined);
  assert.equal(state().pending.count, 0, 'the pending selection is cleared');
  assert.deepEqual(
    state().layers.map((l) => [l.id, l.label, l.count]),
    [['hotspot', 'Hotspot', 1]]
  );
});

test('picking again for a role adds to it, and names the stored one as parent', async () => {
  const { structure, click, picker, state } = await setupRoles();
  const posted = stubFetch([
    { ok: true, reference: 'selection_one' },
    { ok: true, reference: 'selection_two' },
  ]);
  clickResidue(click, structure, labelResidue('A', 2));
  await picker.commitRole('hotspot');
  clickResidue(click, structure, labelResidue('A', 3));
  await picker.commitRole('hotspot');

  assert.equal(posted[1].body.parent, 'selection_one');
  assert.equal(posted[1].body.residue_count, 2, 'the new pick joined the old');
  assert.equal(state().layers.length, 1);
  assert.equal(state().layers[0].count, 2);
});

test('an atom pick offers only the roles it is valid for, and the chosen one carries its value', async () => {
  const { structure, click, picker, state } = await setupRoles();
  const posted = stubFetch([{ ok: true, reference: 'selection_hb' }]);
  clickResidue(click, structure, atomsByLabelQuery([{ chain: 'A', seq: 2, name: 'OH' }]));
  assert.ok(optionIds(state()).includes('hbond:DONOR'));
  assert.ok(!optionIds(state()).includes('pistack'), 'one atom is not a ring');

  await picker.commitRole('hbond:DONOR');
  assert.equal(posted[0].body.role, 'hbond');
  assert.equal(posted[0].body.value, 'DONOR');
  assert.equal(state().layers[0].id, 'hbond:DONOR');
  assert.equal(state().layers[0].label, 'H-bond donor');
});

test('a failed store keeps the pick pending and says so', async () => {
  const { structure, click, picker, state } = await setupRoles();
  stubFetch([{ ok: false }]);
  const errors = console.error;
  console.error = () => {};
  try {
    clickResidue(click, structure, labelResidue('A', 2));
    assert.equal(await picker.commitRole('hotspot'), false);
  } finally {
    console.error = errors;
  }
  assert.equal(state().pending.count, 1);
  assert.equal(state().layers.length, 0);
  assert.match(state().notice ?? '', /Could not save Hotspot/);
});

test('a role that cannot apply is refused with its reason and posts nothing', async () => {
  const { picker, state } = await setupRoles();
  const posted = stubFetch([]);
  assert.equal(await picker.commitRole('hotspot'), false);
  assert.equal(posted.length, 0);
  assert.match(state().notice ?? '', /Pick residues first/);
});

test('with Keep set, a hotspot outside it is refused', async () => {
  const { structure, click, picker, state } = await setupRoles();
  const posted = stubFetch([{ ok: true, reference: 'selection_keep' }]);
  clickResidue(click, structure, labelResidue('A', 2));
  await picker.commitRole('keep');
  clickResidue(click, structure, labelResidue('A', 3));
  assert.equal(state().options.find((o) => o.id === 'hotspot')!.enabled, false);
  assert.equal(await picker.commitRole('hotspot'), false);
  assert.equal(posted.length, 1);
});

test('editing a role lifts its picks back, and committing replaces rather than merges', async () => {
  const { structure, click, picker, state } = await setupRoles();
  const posted = stubFetch([
    { ok: true, reference: 'selection_a' },
    { ok: true, reference: 'selection_b' },
  ]);
  clickResidue(click, structure, labelResidue('A', 2));
  clickResidue(click, structure, labelResidue('A', 3));
  await picker.commitRole('hotspot');

  picker.editLayer('hotspot');
  assert.equal(state().editingId, 'hotspot');
  assert.equal(state().pending.count, 2);
  assert.equal(state().layers.length, 0, 'the layer is out of the tray while it is edited');

  clickResidue(click, structure, labelResidue('A', 3)); // drop Met
  await picker.commitRole('hotspot');
  assert.equal(posted[1].body.residue_count, 1, 'replaced, not merged');
  assert.equal(posted[1].body.parent, 'selection_a');
  assert.equal(state().editingId, null);
  assert.equal(state().layers[0].count, 1);
});

test('cancelling an edit puts the role back as it was', async () => {
  const { structure, click, picker, state } = await setupRoles();
  stubFetch([{ ok: true, reference: 'selection_a' }]);
  clickResidue(click, structure, labelResidue('A', 2));
  await picker.commitRole('hotspot');
  picker.editLayer('hotspot');
  clickResidue(click, structure, labelResidue('A', 3));
  picker.cancelEdit();
  assert.equal(state().layers[0].count, 1);
  assert.equal(state().pending.count, 0);
});

test('re-roling an edited layer removes the old role on the backend', async () => {
  const { structure, click, picker, state } = await setupRoles();
  const posted = stubFetch([
    { ok: true, reference: 'selection_hot' },
    { ok: true, reference: 'selection_keep' },
    { ok: true, reference: 'selection_gone' },
  ]);
  clickResidue(click, structure, labelResidue('A', 2));
  await picker.commitRole('hotspot');
  picker.editLayer('hotspot');
  await picker.commitRole('keep');

  assert.deepEqual(posted.map((p) => p.body.role), ['hotspot', 'keep', 'hotspot']);
  assert.equal(posted[2].body.atom_count, 0, 'an empty selection removes the role');
  assert.equal(posted[2].body.parent, 'selection_hot');
  assert.deepEqual(state().layers.map((l) => l.id), ['keep']);
});

test('removing a role stores an empty selection naming it and drops the chip', async () => {
  const { structure, click, picker, state } = await setupRoles();
  const posted = stubFetch([
    { ok: true, reference: 'selection_hot' },
    { ok: true, reference: 'selection_removed' },
  ]);
  clickResidue(click, structure, labelResidue('A', 2));
  await picker.commitRole('hotspot');
  assert.equal(await picker.removeLayer('hotspot'), true);
  assert.equal(posted[1].body.role, 'hotspot');
  assert.equal(posted[1].body.atoms.length, 0);
  assert.equal(posted[1].body.parent, 'selection_hot');
  assert.equal(state().layers.length, 0);
});

test('a role that fails to remove stays in the tray', async () => {
  const { structure, click, picker, state } = await setupRoles();
  stubFetch([{ ok: true, reference: 'selection_hot' }, { ok: false }]);
  const errors = console.error;
  console.error = () => {};
  try {
    clickResidue(click, structure, labelResidue('A', 2));
    await picker.commitRole('hotspot');
    assert.equal(await picker.removeLayer('hotspot'), false);
  } finally {
    console.error = errors;
  }
  assert.equal(state().layers.length, 1);
});

test('Motif assigns Keep, Fix position, Fix sequence and Float together', async () => {
  const { structure, click, picker, state } = await setupRoles();
  const posted = stubFetch([]);
  clickResidue(click, structure, labelResidue('A', 2));
  clickResidue(click, structure, labelResidue('A', 3));
  assert.equal(await picker.commitRole('motif'), true);
  assert.deepEqual(posted.map((p) => p.body.role), ['keep', 'fix_position', 'fix_sequence', 'float']);
  assert.equal(posted[3].body.value, 'motif');
  assert.deepEqual(state().layers.map((l) => l.id), ['keep', 'fix_position', 'fix_sequence', 'float:motif']);
  assert.equal(state().pending.count, 0);
});

test('the kept pieces and layout are shown, and changing the layout re-stores Keep with it', async () => {
  const { structure, click, picker, state } = await setupRoles();
  const posted = stubFetch([
    { ok: true, reference: 'selection_keep' },
    { ok: true, reference: 'selection_keep2' },
  ]);
  clickResidue(click, structure, labelResidue('A', 2));
  clickResidue(click, structure, labelResidue('A', 4)); // not adjacent to residue 2
  await picker.commitRole('keep');
  assert.deepEqual(state().pieces.map((p) => [p.firstLabel, p.lastLabel]), [[2, 2], [4, 4]]);

  assert.equal(await picker.setLayout({ n_term: [10, 20], linkers: [[5, 8]] }), true);
  assert.deepEqual(posted[1].body.value, { n_term: [10, 20], linkers: [[5, 8]] });
  assert.equal(posted[1].body.parent, 'selection_keep');
  assert.deepEqual(state().layout, { n_term: [10, 20], linkers: [[5, 8]] });
});

test('the engine choice greys out what it cannot take and is told to the model', async () => {
  const { structure, click, picker, state } = await setupRoles();
  clickResidue(click, structure, labelResidue('A', 2));
  picker.setEngine('rfdiffusion1');
  const float = state().options.find((o) => o.id === 'float:motif')!;
  assert.equal(float.enabled, false);
  assert.match(picker.describe().description, /design engine chosen by the user: RFdiffusion/);
  picker.setEngine('not-an-engine');
  assert.equal(state().engine, 'rfdiffusion1');
});

test('the model is told each role and what is still pending with no role', async () => {
  const { structure, click, picker } = await setupRoles();
  stubFetch([]);
  clickResidue(click, structure, labelResidue('A', 2));
  await picker.commitRole('hotspot');
  clickResidue(click, structure, labelResidue('A', 3));
  const { residueCount, description } = picker.describe();
  assert.equal(residueCount, 2);
  assert.match(description, /Hotspot role \(set by the user\): 1 residues/);
  assert.match(description, /pending, no role \(picked by the user\): 1 residues/);
});

test('with no role chosen, sending stores the pending picks as a plain selection', async () => {
  const { structure, click, picker } = await setupRoles();
  const posted = stubFetch([{ ok: true, reference: 'selection_pending' }]);
  clickResidue(click, structure, labelResidue('A', 2));
  assert.deepEqual(await picker.commit('s'), ['selection_pending']);
  assert.equal(posted[0].body.role, undefined);
  assert.equal(posted[0].body.name, 'pending');
});

test('a new structure removes the stored roles so they cannot leak into the next design', async () => {
  const { structure, click, picker, state } = await setupRoles();
  const posted = stubFetch([{ ok: true, reference: 'selection_hot' }, { ok: true, reference: 'selection_off' }]);
  clickResidue(click, structure, labelResidue('A', 2));
  await picker.commitRole('hotspot');
  picker.reset();
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(state().layers.length, 0);
  assert.equal(posted[1].body.role, 'hotspot');
  assert.equal(posted[1].body.atom_count, 0);
});

test('a selection the assistant made can be adopted as the pick, then given a role', async () => {
  const { picker, state } = await setupRoles();
  const posted = stubFetch([{ ok: true, reference: 'selection_agent_hot' }]);
  picker.offerAgentSelection('selection_probe', [{ chain: 'A', seq: 2 }], []);
  assert.deepEqual(state().offered, { label: 'selection_probe', residueCount: 1, atomCount: 0 });

  assert.equal(picker.useOffered(), true);
  assert.equal(state().offered, null);
  assert.equal(state().pending.count, 1);
  assert.ok(optionIds(state()).includes('hotspot'));

  await picker.commitRole('hotspot');
  assert.equal(posted[0].body.provenance, 'user', 'the user is the one who gave it the role');
  assert.equal(posted[0].body.role, 'hotspot');
});

test('adopting atoms of a residue keeps them partial unless they are all of it', async () => {
  const { picker, state } = await setupRoles();
  picker.offerAgentSelection('probe', [], [{ chain: 'A', seq: 2, name: 'OH' }]);
  picker.useOffered();
  assert.equal(state().pending.partialCount, 1);
  assert.equal(state().pending.atomCount, 1);

  picker.offerAgentSelection(
    'probe',
    [],
    ['N', 'CA', 'CB', 'CG', 'CZ'].map((name) => ({ chain: 'A', seq: 2, name }))
  );
  picker.useOffered();
  assert.equal(state().pending.partialCount, 0, 'with every atom the residue is whole');
});

test('adopting never discards what is already picked', async () => {
  const { structure, click, picker, state } = await setupRoles();
  clickResidue(click, structure, labelResidue('A', 3));
  picker.offerAgentSelection('probe', [{ chain: 'A', seq: 2 }], []);
  picker.useOffered();
  assert.equal(state().pending.count, 2);
});

test('an offer with nothing pickable in it says so and stays offered', async () => {
  const { picker, state } = await setupRoles();
  picker.offerAgentSelection('probe', [{ chain: 'Z', seq: 99 }], []);
  assert.equal(picker.useOffered(), false);
  assert.match(state().notice ?? '', /None of the agent/);
  assert.notEqual(state().offered, null);
});

test('an empty offer clears the earlier one, and a new structure drops it', async () => {
  const { picker, state } = await setupRoles();
  picker.offerAgentSelection('probe', [{ chain: 'A', seq: 2 }], []);
  picker.offerAgentSelection('probe', [], []);
  assert.equal(state().offered, null);
  picker.offerAgentSelection('probe', [{ chain: 'A', seq: 2 }], []);
  picker.reset();
  assert.equal(state().offered, null);
});
