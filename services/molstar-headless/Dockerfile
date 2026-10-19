# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FROM node:20-slim AS build

WORKDIR /app
COPY package.json ./
RUN npm install
COPY tsconfig.json ./
COPY src/ ./src/
COPY tests/ ./tests/
# Format handling is the service's contract with the browser, and a regression
# there is silent -- the caller sees zero atoms, not an error. Running the tests
# at build time makes that a build failure instead.
RUN npm test
# `npm run build` bundles to a single CJS file via esbuild. Plain `tsc` emits
# extensionless ESM imports that Node refuses to resolve at runtime.
RUN npm run build

FROM node:20-slim
WORKDIR /app
COPY --from=build /app/dist ./dist
COPY --from=build /app/node_modules ./node_modules
COPY package.json ./

ENV MOLSTAR_PORT=8020
EXPOSE 8020

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD node -e "const http=require('http');http.get('http://localhost:8020/health',r=>{process.exit(r.statusCode===200?0:1)}).on('error',()=>process.exit(1))"

CMD ["node", "dist/app.js"]
