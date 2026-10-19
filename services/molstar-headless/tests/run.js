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

// Bundles the TS tests through esbuild (same bundler the service ships with)
// and hands the result to node's test runner. Keeps the service free of a
// separate test-framework dependency.
const { build } = require('esbuild');
const { spawnSync } = require('node:child_process');
const path = require('node:path');

// Not under dist/, which the Dockerfile copies into the runtime image.
const out = path.join(require('node:os').tmpdir(), 'molstar-headless-tests.js');

build({
  entryPoints: [path.join(__dirname, 'structure-loader.test.ts')],
  bundle: true,
  platform: 'node',
  target: 'node20',
  outfile: out,
  format: 'cjs',
  external: ['node:test', 'node:assert/strict'],
}).then(() => {
  const r = spawnSync(process.execPath, ['--test', out], { stdio: 'inherit' });
  process.exit(r.status ?? 1);
});
