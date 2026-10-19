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
 * Sessions are minted by the backend, never by the browser: POST
 * /api/session/create binds the new session to this browser's owner cookie,
 * and every session-scoped endpoint rejects a session the cookie does not own.
 */
export async function createSession(): Promise<string> {
  const res = await fetch("/api/session/create", {
    method: "POST",
    credentials: "same-origin",
  });
  if (!res.ok) throw new Error(`Session create failed: ${res.status}`);
  const data: { session_id: string } = await res.json();
  return data.session_id;
}
