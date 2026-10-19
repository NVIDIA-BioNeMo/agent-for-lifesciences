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
 * MolQL query building and compilation utilities.
 * 
 * Provides programmatic access to Molstar's query language for selecting
 * atoms, residues, and chains based on structural criteria.
 */

import { MolScriptBuilder as MS } from 'molstar/lib/mol-script/language/builder';
import { compile } from 'molstar/lib/mol-script/runtime/query/compiler';
import { StructureQuery, StructureSelection, QueryContext } from 'molstar/lib/mol-model/structure';
import { Structure } from 'molstar/lib/mol-model/structure';
import type { Expression } from 'molstar/lib/mol-script/language/expression';

import type {
  ChainSelector,
  ResidueSelector,
  AtomSelector,
  EntityType,
  DistanceQueryOptions,
} from './types';

/**
 * Build a MolScript expression for selecting chains.
 */
export function chainQuery(selector: ChainSelector): Expression {
  const tests: Expression[] = [];

  if (selector.chainId) {
    tests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.label_asym_id(),
        selector.chainId,
      ])
    );
  }

  if (selector.authChainId) {
    tests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.auth_asym_id(),
        selector.authChainId,
      ])
    );
  }

  if (tests.length === 0) {
    // Select all chains
    return MS.struct.generator.atomGroups({
      'chain-test': MS.core.logic.and(tests),
    });
  }

  return MS.struct.generator.atomGroups({
    'chain-test': tests.length === 1 ? tests[0] : MS.core.logic.and(tests),
  });
}

/**
 * Build a MolScript expression for selecting residues.
 */
export function residueQuery(selector: ResidueSelector, chainSelector?: ChainSelector): Expression {
  const tests: Expression[] = [];

  if (selector.name) {
    tests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.label_comp_id(),
        selector.name.toUpperCase(),
      ])
    );
  }

  if (selector.seqNumber !== undefined) {
    tests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.label_seq_id(),
        selector.seqNumber,
      ])
    );
  }

  if (selector.seqRange) {
    const [start, end] = selector.seqRange;
    tests.push(
      MS.core.rel.inRange([
        MS.struct.atomProperty.macromolecular.label_seq_id(),
        start,
        end,
      ])
    );
  }

  if (selector.insCode) {
    tests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.pdbx_PDB_ins_code(),
        selector.insCode,
      ])
    );
  }

  const chainTests: Expression[] = [];
  if (chainSelector?.chainId) {
    chainTests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.label_asym_id(),
        chainSelector.chainId,
      ])
    );
  }
  if (chainSelector?.authChainId) {
    chainTests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.auth_asym_id(),
        chainSelector.authChainId,
      ])
    );
  }

  const params: Record<string, Expression> = {};
  if (tests.length > 0) {
    params['residue-test'] = tests.length === 1 ? tests[0] : MS.core.logic.and(tests);
  }
  if (chainTests.length > 0) {
    params['chain-test'] = chainTests.length === 1 ? chainTests[0] : MS.core.logic.and(chainTests);
  }

  return MS.struct.generator.atomGroups(params);
}

/**
 * Build a MolScript expression for selecting atoms.
 */
export function atomQuery(
  selector: AtomSelector,
  residueSelector?: ResidueSelector,
  chainSelector?: ChainSelector
): Expression {
  const tests: Expression[] = [];

  if (selector.name) {
    tests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.label_atom_id(),
        selector.name,
      ])
    );
  }

  if (selector.element) {
    tests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.core.elementSymbol(),
        selector.element.toUpperCase(),
      ])
    );
  }

  if (selector.id !== undefined) {
    tests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.core.atomKey(),
        selector.id,
      ])
    );
  }

  const residueTests: Expression[] = [];
  if (residueSelector?.name) {
    residueTests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.label_comp_id(),
        residueSelector.name.toUpperCase(),
      ])
    );
  }
  if (residueSelector?.seqNumber !== undefined) {
    residueTests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.label_seq_id(),
        residueSelector.seqNumber,
      ])
    );
  }
  if (residueSelector?.seqRange) {
    const [start, end] = residueSelector.seqRange;
    residueTests.push(
      MS.core.rel.inRange([
        MS.struct.atomProperty.macromolecular.label_seq_id(),
        start,
        end,
      ])
    );
  }

  const chainTests: Expression[] = [];
  if (chainSelector?.chainId) {
    chainTests.push(
      MS.core.rel.eq([
        MS.struct.atomProperty.macromolecular.label_asym_id(),
        chainSelector.chainId,
      ])
    );
  }

  const params: Record<string, Expression> = {};
  if (tests.length > 0) {
    params['atom-test'] = tests.length === 1 ? tests[0] : MS.core.logic.and(tests);
  }
  if (residueTests.length > 0) {
    params['residue-test'] = residueTests.length === 1 ? residueTests[0] : MS.core.logic.and(residueTests);
  }
  if (chainTests.length > 0) {
    params['chain-test'] = chainTests.length === 1 ? chainTests[0] : MS.core.logic.and(chainTests);
  }

  return MS.struct.generator.atomGroups(params);
}

/**
 * Build a MolScript expression for selecting by entity type.
 */
export function entityTypeQuery(entityType: EntityType): Expression {
  return MS.struct.generator.atomGroups({
    'entity-test': MS.core.rel.eq([
      MS.struct.atomProperty.macromolecular.entityType(),
      entityType,
    ]),
  });
}

/**
 * Build a MolScript expression for selecting ligands (non-polymer entities).
 */
export function ligandQuery(): Expression {
  return entityTypeQuery('non-polymer');
}

/**
 * Build a MolScript expression for selecting water molecules.
 */
export function waterQuery(): Expression {
  return entityTypeQuery('water');
}

/**
 * Build a MolScript expression for selecting protein/nucleic acid (polymer entities).
 */
export function polymerQuery(): Expression {
  return entityTypeQuery('polymer');
}

/**
 * Build a MolScript expression for selecting residues within a distance of another selection.
 * 
 * This is the core query for "select all residues within X angstroms of the ligand".
 */
export function withinDistanceQuery(
  targetExpression: Expression,
  options: DistanceQueryOptions
): Expression {
  const { radius, wholeResidues = true } = options;

  return MS.struct.modifier.includeSurroundings({
    0: targetExpression,
    radius,
    'as-whole-residues': wholeResidues,
  });
}

/**
 * Build a query for residues within distance of a ligand.
 * 
 * Example: residuesNearLigand(5.0) - all residues within 5Å of any ligand
 */
export function residuesNearLigandQuery(radiusAngstroms: number): Expression {
  return withinDistanceQuery(ligandQuery(), {
    radius: radiusAngstroms,
    wholeResidues: true,
  });
}

/**
 * Build a query for residues within distance of a specific chain.
 */
export function residuesNearChainQuery(
  chainId: string,
  radiusAngstroms: number
): Expression {
  return withinDistanceQuery(chainQuery({ chainId }), {
    radius: radiusAngstroms,
    wholeResidues: true,
  });
}

/**
 * Build a query for selecting backbone atoms only.
 */
export function backboneQuery(): Expression {
  return MS.struct.generator.atomGroups({
    'atom-test': MS.core.set.has([
      MS.set('N', 'CA', 'C', 'O'),
      MS.struct.atomProperty.macromolecular.label_atom_id(),
    ]),
    'entity-test': MS.core.rel.eq([
      MS.struct.atomProperty.macromolecular.entityType(),
      'polymer',
    ]),
  });
}

/**
 * Build a query for selecting side chain atoms only.
 */
export function sidechainQuery(): Expression {
  return MS.struct.generator.atomGroups({
    'atom-test': MS.core.logic.not([
      MS.core.set.has([
        MS.set('N', 'CA', 'C', 'O'),
        MS.struct.atomProperty.macromolecular.label_atom_id(),
      ]),
    ]),
    'entity-test': MS.core.rel.eq([
      MS.struct.atomProperty.macromolecular.entityType(),
      'polymer',
    ]),
  });
}

/**
 * Build a query for selecting alpha carbons only.
 */
export function alphaCarbonQuery(): Expression {
  return MS.struct.generator.atomGroups({
    'atom-test': MS.core.rel.eq([
      MS.struct.atomProperty.macromolecular.label_atom_id(),
      'CA',
    ]),
    'entity-test': MS.core.rel.eq([
      MS.struct.atomProperty.macromolecular.entityType(),
      'polymer',
    ]),
  });
}

/**
 * Combine multiple queries with AND logic.
 */
export function intersectionQuery(...expressions: Expression[]): Expression {
  if (expressions.length === 1) return expressions[0];
  
  return MS.struct.combinator.intersect(
    expressions.map((expr, i) => ({ [i]: expr })).reduce((a, b) => ({ ...a, ...b }), {})
  );
}

/**
 * Combine multiple queries with OR logic.
 */
export function unionQuery(...expressions: Expression[]): Expression {
  if (expressions.length === 1) return expressions[0];
  
  // merge takes a list of queries, not a dictionary.
  return MS.struct.combinator.merge(expressions);
}

/**
 * Compile a MolScript expression into an executable query.
 */
export function compileQuery(expression: Expression): StructureQuery {
  return compile<StructureSelection>(expression);
}

/**
 * Execute a compiled query against a structure.
 */
export function executeQuery(
  query: StructureQuery,
  structure: Structure
): StructureSelection {
  return query(new QueryContext(structure));
}

/**
 * Convenience function to build and execute a query in one step.
 */
export function runQuery(
  expression: Expression,
  structure: Structure
): StructureSelection {
  const query = compileQuery(expression);
  return executeQuery(query, structure);
}


/** A residue named by its label ids, which is how the backend addresses residues. */
export interface LabelResidue {
  chain: string;
  seq: number;
}

/**
 * Build a MolScript expression for a list of residues given by label ids.
 *
 * Label ids (label_asym_id / label_seq_id) are what the backend's selection
 * artifacts carry for every residue, so this needs no MolQL text on the wire
 * and no parsing. It matches whole residues, one group per residue.
 */
export function residuesByLabelQuery(residues: LabelResidue[]): Expression {
  const byChain = new Map<string, number[]>();
  for (const { chain, seq } of residues) {
    const seqs = byChain.get(chain);
    if (seqs) seqs.push(seq);
    else byChain.set(chain, [seq]);
  }

  const perChain: Expression[] = [];
  byChain.forEach((seqs, chain) => {
    perChain.push(
      MS.core.logic.and([
        MS.core.rel.eq([
          MS.struct.atomProperty.macromolecular.label_asym_id(),
          chain,
        ]),
        MS.core.set.has([
          MS.core.type.set(seqs),
          MS.struct.atomProperty.macromolecular.label_seq_id(),
        ]),
      ])
    );
  });

  return MS.struct.generator.atomGroups({
    // An empty list must match nothing, not everything.
    'residue-test': perChain.length === 0
      ? false
      : perChain.length === 1
        ? perChain[0]
        : MS.core.logic.or(perChain),
    'group-by': MS.struct.atomProperty.macromolecular.residueKey(),
  });
}


/** One atom named by its label ids and label atom name. */
export interface LabelAtom {
  chain: string;
  seq: number;
  name: string;
}

/**
 * Build a MolScript expression for individual atoms given by label ids.
 *
 * Atoms are grouped by chain and by the exact set of names chosen, and each
 * group becomes one test: ``chain = C and seq in {...} and atom name in {...}``.
 * That keeps the expression the size of the number of distinct choices rather
 * than the number of atoms, and is exact: two residues in a chain are only
 * tested together when they were given the same atom names. Each atom is its
 * own group.
 */
export function atomsByLabelQuery(atoms: LabelAtom[]): Expression {
  // chain -> residue seq -> names
  const residues = new Map<string, Map<number, Set<string>>>();
  for (const { chain, seq, name } of atoms) {
    let bySeq = residues.get(chain);
    if (!bySeq) residues.set(chain, (bySeq = new Map()));
    let names = bySeq.get(seq);
    if (!names) bySeq.set(seq, (names = new Set()));
    names.add(name);
  }

  const tests: Expression[] = [];
  residues.forEach((bySeq, chain) => {
    // name-set signature -> the residues that share it
    const groups = new Map<string, { names: string[]; seqs: number[] }>();
    bySeq.forEach((names, seq) => {
      const sorted = Array.from(names).sort();
      const signature = sorted.join('\u0000');
      const group = groups.get(signature);
      if (group) group.seqs.push(seq);
      else groups.set(signature, { names: sorted, seqs: [seq] });
    });
    groups.forEach(({ names, seqs }) => {
      tests.push(
        MS.core.logic.and([
          MS.core.rel.eq([MS.struct.atomProperty.macromolecular.label_asym_id(), chain]),
          MS.core.set.has([
            MS.core.type.set(seqs),
            MS.struct.atomProperty.macromolecular.label_seq_id(),
          ]),
          MS.core.set.has([
            MS.core.type.set(names),
            MS.struct.atomProperty.macromolecular.label_atom_id(),
          ]),
        ])
      );
    });
  });

  return MS.struct.generator.atomGroups({
    // An empty list must match nothing, not everything.
    'atom-test': tests.length === 0 ? false : tests.length === 1 ? tests[0] : MS.core.logic.or(tests),
  });
}

/**
 * Whole residues and individual atoms in one expression.
 *
 * What a pick set or a ``show_selection`` event describes: some residues taken
 * whole, some taking only chosen atoms. Either list may be empty.
 */
export function picksQuery(residues: LabelResidue[], atoms: LabelAtom[]): Expression {
  if (atoms.length === 0) return residuesByLabelQuery(residues);
  if (residues.length === 0) return atomsByLabelQuery(atoms);
  return unionQuery(residuesByLabelQuery(residues), atomsByLabelQuery(atoms));
}
