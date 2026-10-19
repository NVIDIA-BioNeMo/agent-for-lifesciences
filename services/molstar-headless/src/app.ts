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
 * Headless Molstar MolQL query service.
 *
 * Single-purpose HTTP service: receives a PDB structure + MolQL query,
 * executes the query using the same molstar libraries as the browser,
 * and returns atom-level selection data.
 *
 * Endpoints:
 *   POST /query   - Execute MolQL against a PDB structure
 *   GET  /health  - Health check
 */

import express from 'express';
import { executeMolQLQuery } from './molql-engine';

const app = express();
const PORT = parseInt(process.env.MOLSTAR_PORT || '8020', 10);
const MAX_BODY_SIZE = '50mb';

app.use(express.json({ limit: MAX_BODY_SIZE }));

interface QueryRequest {
  pdb_data: string;
  molql: string;
}

app.post('/query', async (req, res) => {
  const { pdb_data, molql } = req.body as QueryRequest;

  if (!pdb_data || !molql) {
    res.status(400).json({
      error: 'Both pdb_data and molql are required',
    });
    return;
  }

  try {
    const result = await executeMolQLQuery(pdb_data, molql);
    res.json(result);
  } catch (err: any) {
    const message = err?.message || String(err);
    console.error(`[molstar-headless] Query error: ${message}`);
    res.status(422).json({
      error: `MolQL query failed: ${message}`,
      atoms: [],
      atom_count: 0,
      residue_count: 0,
      summary: `Query error: ${message}`,
    });
  }
});

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', service: 'molstar-headless' });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`[molstar-headless] Listening on 0.0.0.0:${PORT}`);
});
