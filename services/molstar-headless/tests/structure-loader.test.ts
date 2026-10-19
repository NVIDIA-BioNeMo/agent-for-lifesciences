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
 * Format coverage for the headless loader.
 *
 * The point of these is parity with the browser: the service must answer a
 * query about a ligand loaded from SDF, because `visualize_sdf` /
 * `visualize_3d` put exactly that in the user's viewer and `structure_query`
 * is expected to work there. Before MOL/SDF support the service raised a PDB
 * parse error and the caller saw "Selected 0 atoms", which is a silently wrong
 * answer rather than a visible failure -- so each case here asserts a real atom
 * count, not merely that parsing did not throw.
 */

import test from 'node:test';
import assert from 'node:assert/strict';

import { detectStructureFormat, parseStructure } from '../src/structure-loader';
import { executeMolQLQuery } from '../src/molql-engine';

/** Three heavy atoms of ethanol, V2000, as `visualize_sdf` would deliver it. */
const ETHANOL_SDF = `ethanol
     RDKit          3D

  3  2  0  0  0  0  0  0  0  0999 V2000
   -0.9000    0.1000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.3000   -0.5000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    1.4000    0.4000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
M  END
$$$$
`;

const TWO_ATOM_PDB = `ATOM      1  N   ALA A   1      -0.966   0.493   1.500  1.00  0.00           N
ATOM      2  CA  ALA A   1       0.257   0.418   0.692  1.00  0.00           C
END
`;

test('an SDF ligand is detected as MOL and parsed', async () => {
  assert.equal(detectStructureFormat(ETHANOL_SDF), 'mol');

  const structure = await parseStructure(ETHANOL_SDF);
  assert.equal(structure.elementCount, 3);
});

test('a title line that looks like a PDB record does not shadow the counts line', () => {
  // MOL titles are free text; RDKit writes the molecule name, which for a
  // benchmark ligand can be anything at all.
  assert.equal(detectStructureFormat(ETHANOL_SDF.replace('ethanol', 'ATOM')), 'mol');
});

test('PDB and mmCIF detection are unchanged', () => {
  assert.equal(detectStructureFormat(TWO_ATOM_PDB), 'pdb');
  assert.equal(detectStructureFormat('data_boltz\n_entry.id boltz\n'), 'mmcif');
  // A REMARK mentioning V2000 far from the top must not read as MOL.
  const decoy = `${'REMARK   1 nothing\n'.repeat(8)}REMARK   2 written by V2000\n${TWO_ATOM_PDB}`;
  assert.equal(detectStructureFormat(decoy), 'pdb');
});

test('MolQL runs against an SDF-loaded ligand', async () => {
  const all = await executeMolQLQuery(ETHANOL_SDF, '(sel.atom.all)');
  assert.equal(all.atom_count, 3);
});

test('MolQL element selection on an SDF ligand returns the right atoms', async () => {
  const result = await executeMolQLQuery(
    ETHANOL_SDF,
    '(sel.atom.atoms (= atom.el _O))',
  );
  assert.equal(result.atom_count, 1);
  assert.equal(result.atoms[0].element.toUpperCase(), 'O');
  assert.deepEqual(result.atoms[0].coords.map((c) => Math.round(c * 10) / 10), [1.4, 0.4, 0]);
});

test('a query the browser could not answer either fails loudly', async () => {
  // The caller distinguishes "matched nothing" from "could not run"; a parse
  // failure must raise rather than come back as a confident zero.
  await assert.rejects(() => executeMolQLQuery(ETHANOL_SDF, '(sel.atom.nonsense)'));
});

/**
 * Two residues of an mmCIF whose author numbering runs 16 ahead of its label
 * numbering, as 5O45's does. MolQL's `atom.resno` matches the AUTHOR number
 * while the label number is what AtomWorks/RFD4 resolve `res_id` against, so a
 * payload that reports only one of them forces the backend to guess which.
 */
const AUTH_OFFSET_CIF = `data_AUTH
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM 1 N N . CYS A 1 24 ? 1.0 1.0 1.0 40 CYS A N 1
ATOM 2 C CA . CYS A 1 24 ? 2.0 1.0 1.0 40 CYS A CA 1
ATOM 3 N N . TYR A 1 40 ? 3.0 1.0 1.0 56 TYR A N 1
ATOM 4 C CA . TYR A 1 40 ? 4.0 1.0 1.0 56 TYR A CA 1
#
`;

test('a pick by author number reports both numbering systems', async () => {
  // atom.resno is author numbering: 56 is the Tyr, whose label number is 40.
  const result = await executeMolQLQuery(
    AUTH_OFFSET_CIF,
    '(sel.atom.atoms (and (= atom.chain A) (= atom.resno 56) (= atom.name CA)))',
  );
  assert.equal(result.atom_count, 1);
  const residue = result.atoms[0].residue;
  assert.equal(residue.name, 'TYR');
  assert.equal(residue.authSeqNumber, 56);
  assert.equal(residue.authChainId, 'A');
  assert.equal(residue.seqNumber, 40);
  assert.equal(residue.chainId, 'A');
});
