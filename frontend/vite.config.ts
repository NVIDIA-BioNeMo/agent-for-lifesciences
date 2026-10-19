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

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Determine API target based on environment
// In Docker: VITE_API_URL=http://backend:8000
// Local dev: defaults to http://localhost:8000
const apiTarget = process.env.VITE_API_URL || "http://localhost:8000";

// Version from git tags (injected at build time via VERSION env var)
// The Makefile runs `git describe --tags --always` and exports VERSION
const version = process.env.VERSION || "0.0.0-dev";

export default defineConfig({
  plugins: [react()],
  define: {
    // Inject VERSION_NUMBER as a global constant
    "import.meta.env.VERSION_NUMBER": JSON.stringify(version),
  },
  server: {
    host: "0.0.0.0", // Allow external connections (needed for Docker)
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
    // Enable HMR with polling for Docker volume mounts
    // Higher interval needed for devcontainer + Docker-outside-of-Docker setups
    watch: {
      usePolling: true,
      interval: 1000,
    },
  },
});
