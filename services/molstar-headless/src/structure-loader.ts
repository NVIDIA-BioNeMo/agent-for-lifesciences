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
 * Structure file loading utilities for headless Molstar.
 *
 * Converts PDB, mmCIF or MOL/SDF text into a Molstar Structure object using the
 * same mol-io and mol-model-formats pipeline as the browser plugin.
 *
 * All three formats are handled because the service has to answer any query the
 * browser could answer. mmCIF is there because Boltz-2 emits mmCIF and nothing
 * else. MOL/SDF is there because `visualize_sdf` and `visualize_3d` put a
 * ligand in the viewer and `structure_query` is expected to work on it -- with
 * only PDB and mmCIF a ligand-only viewer silently answered every query with
 * zero atoms, which is not what a browser does. The format is detected from the
 * content rather than declared, which keeps the service's `{pdb_data, molql}`
 * HTTP contract unchanged.
 */

import { Structure } from 'molstar/lib/mol-model/structure';
import { trajectoryFromPDB } from 'molstar/lib/mol-model-formats/structure/pdb';
import { trajectoryFromMmCIF } from 'molstar/lib/mol-model-formats/structure/mmcif';
import { trajectoryFromMol } from 'molstar/lib/mol-model-formats/structure/mol';
import { parsePDB as parsePDBFile } from 'molstar/lib/mol-io/reader/pdb/parser';
import { parseMol } from 'molstar/lib/mol-io/reader/mol/parser';
import { CIF } from 'molstar/lib/mol-io/reader/cif';
import { Trajectory } from 'molstar/lib/mol-model/structure';
import { Task } from 'molstar/lib/mol-task';

export type StructureFormat = 'pdb' | 'mmcif' | 'mol';

/** The V2000/V3000 tag sits at the end of a MOL counts line, which is line 4. */
const MOL_COUNTS_LINE = /^.{0,40}?\sV[23]000\s*$/;

/**
 * Guess whether structure text is mmCIF, MOL/SDF or PDB.
 *
 * An mmCIF file opens with a `data_` block header, optionally preceded by
 * comments or blank lines; PDB has no such line. Sniffing the first
 * non-comment line is enough to tell them apart, and is more robust than
 * looking for `_atom_site` which can appear far into a large file.
 *
 * MOL/SDF cannot be told apart from its first line, which is a free-text title
 * and may be empty. Its counts line is the fourth and ends in `V2000` or
 * `V3000`, so that is what is looked for -- within the first few lines only, so
 * a PDB that happens to mention V2000 in a REMARK is not misread.
 */
export function detectStructureFormat(data: string): StructureFormat {
  // A few hundred characters covers any realistic run of leading comments.
  const head = data.slice(0, 4096).split('\n');

  for (let i = 0; i < Math.min(head.length, 6); i++) {
    if (MOL_COUNTS_LINE.test(head[i])) return 'mol';
  }

  for (const line of head) {
    const trimmed = line.trim();
    if (trimmed === '' || trimmed.startsWith('#')) continue;
    return trimmed.startsWith('data_') ? 'mmcif' : 'pdb';
  }
  return 'pdb';
}

async function trajectoryFromPdbText(data: string): Promise<Trajectory> {
  const parsed = await Task.resolveInContext(parsePDBFile(data, ''));
  if (parsed.isError) {
    throw new Error(`PDB parsing failed: ${parsed.message}`);
  }
  return Task.resolveInContext(trajectoryFromPDB(parsed.result));
}

async function trajectoryFromMmcifText(data: string): Promise<Trajectory> {
  const parsed = await Task.resolveInContext(CIF.parse(data));
  if (parsed.isError) {
    throw new Error(`mmCIF parsing failed: ${parsed.message}`);
  }
  const file = parsed.result;
  if (file.blocks.length === 0) {
    throw new Error('mmCIF parsing produced no data blocks');
  }
  return Task.resolveInContext(trajectoryFromMmCIF(file.blocks[0], file));
}

/**
 * Parse MOL/SDF text the way `MolstarViewer.loadSdf` does.
 *
 * The frontend calls `parseTrajectory(data, 'mol')`, so this uses the same
 * `parseMol` + `trajectoryFromMol` pair rather than the SDF provider. The
 * consequence is shared with the browser: a multi-record SDF contributes only
 * its first record. Matching the browser matters more than reading every
 * record, because a divergence here would make the headless service answer
 * queries the browser answers differently.
 */
async function trajectoryFromMolText(data: string): Promise<Trajectory> {
  const parsed = await Task.resolveInContext(parseMol(data));
  if (parsed.isError) {
    throw new Error(`MOL/SDF parsing failed: ${parsed.message}`);
  }
  return Task.resolveInContext(trajectoryFromMol(parsed.result));
}

/**
 * Parse a PDB, mmCIF or MOL/SDF string into a Molstar Structure.
 *
 * Pipeline: text -> format-specific reader -> Trajectory (models) -> Structure
 */
export async function parseStructure(
  data: string,
  format?: StructureFormat,
): Promise<Structure> {
  const resolved = format ?? detectStructureFormat(data);

  let trajectory: Trajectory;
  if (resolved === 'mmcif') {
    trajectory = await trajectoryFromMmcifText(data);
  } else if (resolved === 'mol') {
    trajectory = await trajectoryFromMolText(data);
  } else {
    trajectory = await trajectoryFromPdbText(data);
  }

  if (trajectory.frameCount === 0) {
    throw new Error(`${resolved} parsing produced no frames/models`);
  }

  const model = await Task.resolveInContext(trajectory.getFrameAtIndex(0));
  return Structure.ofModel(model);
}
