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

// Bundles the TS tests through esbuild (already present via Vite) and hands the
// result to node's own test runner, so the frontend gains no test-framework
// dependency. The tests exercise real Mol* structures, not mocks of them.
import { build } from 'esbuild';
import { spawnSync } from 'node:child_process';
import { readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const entries = readdirSync(here)
  .filter((f) => f.endsWith('.test.ts'))
  .map((f) => path.join(here, f));

const outdir = path.join(tmpdir(), 'aevom-frontend-tests');

await build({
  entryPoints: entries,
  bundle: true,
  platform: 'node',
  target: 'node20',
  outdir,
  format: 'cjs',
  outExtension: { '.js': '.cjs' },
  external: ['node:test', 'node:assert/strict'],
  logLevel: 'error',
});

const files = entries.map((f) => path.join(outdir, path.basename(f).replace(/\.ts$/, '.cjs')));
const result = spawnSync(process.execPath, ['--test', ...files], { stdio: 'inherit' });
process.exit(result.status ?? 1);
