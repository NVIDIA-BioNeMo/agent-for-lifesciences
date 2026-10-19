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
 * Parsing and failure reporting for agent-issued MolQL queries.
 *
 * A query can fail at three points: the text does not parse (`[1, 2]` is not
 * mol-script), the expression parses but does not compile (`in` is not a MolQL
 * function), or it compiles and throws while running. All three must reach the
 * backend as a selection POST, or its pending op waits out a 10 s timeout and
 * the model is told the viewer is not answering.
 *
 * The POST carries `atom_count: 0` and a summary starting with
 * MOLQL_ERROR_PREFIX. The backend uses that prefix, which must match
 * backend/api/selections/format.py, to tell a query that never ran from one
 * that ran and matched nothing.
 */

import type { Expression } from "molstar/lib/mol-script/language/expression";
import { Script } from "molstar/lib/mol-script/script";

export const MOLQL_ERROR_PREFIX = "MolQL error:";

export type ParsedMolQL =
  | { expression: Expression; error?: undefined }
  | { expression?: undefined; error: string };

/** The message of a thrown value, bounded so it stays readable as tool output. */
export function describeError(err: unknown): string {
  const message = err instanceof Error ? err.message : String(err);
  return message.slice(0, 500);
}

/**
 * Parse a MolQL S-expression string into a Molstar Expression.
 * Uses Molstar's built-in mol-script parser.
 */
export function parseMolQL(molql: string): ParsedMolQL {
  try {
    const script = Script(molql, "mol-script");
    return { expression: Script.toExpression(script) };
  } catch (err) {
    console.error("[MolQL Parse Error]", err);
    return { error: describeError(err) };
  }
}

/**
 * POST a failed query to /api/selection so the backend stops waiting and the
 * model is told why the query failed.
 */
export function postMolQLFailure(
  selectionUrl: string,
  queryId: string,
  molql: string,
  action: string,
  detail: string,
): Promise<void> {
  return fetch(selectionUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query_id: queryId,
      molql,
      action,
      atoms: [],
      atom_count: 0,
      residue_count: 0,
      summary: `${MOLQL_ERROR_PREFIX} ${detail}`,
    }),
  })
    .then(() => undefined)
    .catch((err) => console.error("Error posting MolQL failure:", err));
}
