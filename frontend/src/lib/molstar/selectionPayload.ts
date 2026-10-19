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
 * Wire format for selections POSTed to ``/api/selection``.
 *
 * One definition shared by every poster (agent ``structure_query`` events in
 * App and TrainingMode, and user-made selections), so the two numbering
 * systems can never be dropped by one path and kept by another.
 *
 * ``seqNumber`` / ``chainId`` are LABEL numbering (what RFD4 resolves
 * ``res_id`` against). ``authSeqNumber`` / ``authChainId`` are what the
 * viewer shows the user and what MolQL ``atom.resno`` / ``atom.chain`` match.
 * The backend needs both: it never has to guess which one a number is.
 */

import type { AtomInfo } from "./types";

/** Atoms are capped so a whole-structure query cannot produce a huge POST. */
export const MAX_SELECTION_ATOMS = 10000;

export interface SelectionAtomPayload {
  name: string;
  element: string;
  id: number;
  coords: [number, number, number];
  residue: {
    name: string;
    seqNumber: number;
    chainId: string;
    authSeqNumber: number;
    authChainId: string;
    insCode: string;
    entityType: string;
  };
}

/** Serialize viewer atoms into the backend's ``AtomData`` shape. */
export function serializeAtoms(atoms: AtomInfo[]): SelectionAtomPayload[] {
  return atoms.slice(0, MAX_SELECTION_ATOMS).map((atom) => ({
    name: atom.name,
    element: atom.element,
    id: atom.id,
    coords: atom.coords,
    residue: {
      name: atom.residue.name,
      seqNumber: atom.residue.seqNumber,
      chainId: atom.residue.chainId,
      authSeqNumber: atom.residue.authSeqNumber,
      authChainId: atom.residue.authChainId,
      insCode: atom.residue.insCode || "",
      entityType: atom.residue.entityType,
    },
  }));
}
