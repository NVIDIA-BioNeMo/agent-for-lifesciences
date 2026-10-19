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
 * Headless MolQL query engine.
 *
 * Uses the same molstar libraries as the browser frontend to parse MolQL
 * S-expressions, load structures, execute queries, and extract atom data.
 * Guarantees identical query semantics between production (browser) and
 * training (headless).
 */

import { Structure, StructureSelection, StructureElement, StructureProperties as SP, QueryContext } from 'molstar/lib/mol-model/structure';
import { Vec3 } from 'molstar/lib/mol-math/linear-algebra';
import { parseStructure } from './structure-loader';

// MolQL / mol-script imports
import { compile } from 'molstar/lib/mol-script/runtime/query/compiler';
import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { Script } from 'molstar/lib/mol-script/script';
import { Loci } from 'molstar/lib/mol-model/loci';

export interface AtomInfo {
  name: string;
  element: string;
  id: number;
  coords: [number, number, number];
  residue: {
    name: string;
    /** label_seq_id: the numbering AtomWorks/RFD4 use for res_id. */
    seqNumber: number;
    /** label_asym_id */
    chainId: string;
    /** auth_seq_id: what the viewer shows and MolQL atom.resno matches. */
    authSeqNumber: number;
    /** auth_asym_id: MolQL atom.chain. */
    authChainId: string;
    insCode: string;
    entityType: string;
  };
}

export interface QueryResult {
  atoms: AtomInfo[];
  atom_count: number;
  residue_count: number;
  summary: string;
}

/**
 * Execute a MolQL S-expression query against a structure string.
 *
 * The parameter keeps its `pdbData` name because it carries the request's
 * `pdb_data` field, which stayed as it was when mmCIF support was added; the
 * format is detected from the content.
 *
 * Mirrors the frontend's queryStructure() + getAtomInfo() pipeline from
 * frontend/src/lib/molstar/selections.ts.
 */
export async function executeMolQLQuery(
  pdbData: string,
  molql: string,
): Promise<QueryResult> {
  // 1. Parse the structure text, whichever format it is in
  const structure = await parseStructure(pdbData);

  // 2. Parse MolQL S-expression into an Expression, then compile to a query
  const script = Script(molql, 'mol-script');
  const expression = Script.toExpression(script);
  const query = compile<StructureSelection>(expression);

  // 3. Execute query
  const selection = query(new QueryContext(structure));
  const loci = StructureSelection.toLociWithSourceUnits(selection);

  if (Loci.isEmpty(loci)) {
    return {
      atoms: [],
      atom_count: 0,
      residue_count: 0,
      summary: 'No atoms matched the query',
    };
  }

  // 4. Extract atom data (mirrors frontend getAtomInfo)
  const atoms: AtomInfo[] = [];
  const residueSet = new Set<string>();

  StructureElement.Loci.forEachLocation(loci, (loc: StructureElement.Location) => {
    const pos = loc.unit.conformation.position(loc.element, Vec3());

    const chainId = SP.chain.label_asym_id(loc);
    const seqNumber = SP.residue.label_seq_id(loc);
    const insCode = SP.residue.pdbx_PDB_ins_code(loc) || '';
    const residueKey = `${chainId}:${seqNumber}:${insCode}`;
    residueSet.add(residueKey);

    atoms.push({
      name: SP.atom.label_atom_id(loc),
      element: SP.atom.type_symbol(loc),
      id: SP.atom.id(loc),
      coords: [pos[0], pos[1], pos[2]],
      residue: {
        name: SP.atom.label_comp_id(loc),
        seqNumber,
        chainId,
        authSeqNumber: SP.residue.auth_seq_id(loc),
        authChainId: SP.chain.auth_asym_id(loc),
        insCode,
        entityType: SP.entity.type(loc),
      },
    });
  });

  const residue_count = residueSet.size;
  const summary = `Selected ${atoms.length} atoms in ${residue_count} residues`;

  return {
    atoms,
    atom_count: atoms.length,
    residue_count,
    summary,
  };
}
