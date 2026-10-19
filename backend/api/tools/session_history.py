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

"""Search what this session has already measured and produced.

The state envelope shows the twenty most recent artifacts and the forty most
recent findings, then says "and N more". Raising those caps only moves the
cliff -- the envelope has to stay small or it eats the budget it is supposed
to protect. What removes the cliff is giving the model a way to reach past it.

Without this, the failure mode is not an error. The agent simply re-docks a
ligand it docked thirty iterations ago, because the number scrolled out of
the envelope and re-running is the only recovery it has.
"""

import logging
from typing import Any, Optional

from .base import BaseTool, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)

# Enough to answer "what did I dock against this receptor" without turning a
# search result into the context problem it exists to solve.
DEFAULT_LIMIT = 15
MAX_LIMIT = 50


def _format_findings(findings: list[dict[str, Any]]) -> list[str]:
    lines = []
    for finding in findings:
        prefix = f"T{finding.get('turn_seq', 1)}.{finding.get('iteration', 0)}"
        summary = finding.get("summary") or finding.get("tool_name") or ""
        line = f"{prefix} {summary}".rstrip()
        if not finding.get("success", True):
            line += " [failed]"
        lines.append(line)
    return lines


def _format_artifacts(artifacts: list[dict[str, Any]]) -> list[str]:
    lines = []
    for artifact in artifacts:
        reference = artifact.get("reference", "unknown")
        data_type = artifact.get("data_type", "unknown")
        description = artifact.get("description") or ""
        line = f"{reference} ({data_type})"
        if description:
            line += f": {description}"
        lines.append(line)
    return lines


class SearchSessionHistoryTool(BaseTool):
    """Look up earlier results and artifacts from this session."""

    @property
    def name(self) -> str:
        return "search_session_history"

    @property
    def summary_template(self) -> str:
        return "Searching session history..."

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY

    @property
    def is_async(self) -> bool:
        return False

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_session_history",
                "description": """Search everything this session has already \
computed: past tool results with their numeric outcomes, and stored data \
artifacts with their descriptions.

Use this tool when:
- [STORAGE STATE] or [SESSION LOG] says "and N more" and you need one of them
- You need a number from earlier in the session that is no longer visible
- You want to check whether you have already run something before running it
- You need the reference for a structure or pose produced many steps ago

Do NOT re-run a docking, fold, or analysis to recover a result you already \
have. Search for it first -- it is far cheaper and the number is exact.

Examples:
- search_session_history(query="affinity") -> every docking result so far
- search_session_history(tool="vina_dock") -> every docking run
- search_session_history(query="kinase", data_type="pdb") -> matching structures
""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": (
                                "Free text matched against result summaries, "
                                "tool names, and artifact descriptions. Omit "
                                "to list everything matching the filters."
                            ),
                        },
                        "tool": {
                            "type": "string",
                            "description": (
                                "Restrict results to one tool, e.g. "
                                "'vina_dock' or 'openfold3'."
                            ),
                        },
                        "data_type": {
                            "type": "string",
                            "description": (
                                "Restrict artifacts to one type, e.g. 'pdb', "
                                "'sdf', 'text'."
                            ),
                        },
                        "limit": {
                            "type": "integer",
                            "description": (
                                f"Maximum matches per section "
                                f"(default {DEFAULT_LIMIT}, max {MAX_LIMIT})."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Search the findings ledger and the artifact store.

        Args:
            arguments: ``query``, ``tool``, ``data_type``, and ``limit``, all
                optional.

        Returns:
            A ToolResult listing matching results and artifacts.
        """
        query: Optional[str] = (arguments.get("query") or "").strip() or None
        tool_name: Optional[str] = (arguments.get("tool") or "").strip() or None
        data_type: Optional[str] = (arguments.get("data_type") or "").strip() or None

        try:
            limit = int(arguments.get("limit") or DEFAULT_LIMIT)
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT
        limit = max(1, min(limit, MAX_LIMIT))

        findings = await self._search_findings(query, tool_name, limit)
        artifacts = await self._search_artifacts(query, data_type, limit, tool_name)

        return self._render(query, tool_name, data_type, findings, artifacts)

    async def _search_findings(
        self, query: Optional[str], tool_name: Optional[str], limit: int
    ) -> list[dict[str, Any]]:
        from api.conversations import (
            get_findings_source,
            get_recorder,
            search_findings_in_memory,
        )

        recorder = get_recorder()
        if recorder is not None:
            try:
                return await recorder.search_findings(
                    query=query, tool_name=tool_name, limit=limit
                )
            except Exception:
                logger.warning("Findings search failed", exc_info=True)
                return []

        # No durable record for this request: a training rollout keeps its
        # ledger in process instead. Searching it here is what stops this tool
        # from existing only in deployment, which would leave the policy
        # trained never to reach for it.
        source = get_findings_source()
        if source is None:
            return []

        try:
            return search_findings_in_memory(
                source(), query=query, tool_name=tool_name, limit=limit
            )
        except Exception:
            logger.warning("In-memory findings search failed", exc_info=True)
            return []

    async def _search_artifacts(
        self,
        query: Optional[str],
        data_type: Optional[str],
        limit: int,
        tool_name: Optional[str],
    ) -> list[dict[str, Any]]:
        # A tool filter is a question about results, not about storage.
        # Answering it with the whole artifact list would bury the one line
        # that was asked for -- the exact failure the state envelope's cap
        # exists to prevent.
        if tool_name and not query and not data_type:
            return []

        storage = self._storage
        if storage is None or not hasattr(storage, "search_artifacts"):
            return []

        try:
            return await storage.search_artifacts(
                query=query, data_type=data_type, limit=limit
            )
        except Exception:
            logger.warning("Artifact search failed", exc_info=True)
            return []

    def _render(
        self,
        query: Optional[str],
        tool_name: Optional[str],
        data_type: Optional[str],
        findings: list[dict[str, Any]],
        artifacts: list[dict[str, Any]],
    ) -> ToolResult:
        criteria = [
            part
            for part in (
                f'query="{query}"' if query else "",
                f"tool={tool_name}" if tool_name else "",
                f"data_type={data_type}" if data_type else "",
            )
            if part
        ]
        described = ", ".join(criteria) if criteria else "everything"

        if not findings and not artifacts:
            return ToolResult.text(
                f"No session history matches {described}. "
                f"Nothing has been computed or stored under those terms yet, "
                f"so this is work still to do rather than work to recover.",
                metadata={"match_count": 0},
            )

        sections = [f"Session history matching {described}:"]
        if findings:
            sections.append(
                "\n".join(["", "RESULTS", *_format_findings(findings)])
            )
        if artifacts:
            sections.append(
                "\n".join(["", "ARTIFACTS", *_format_artifacts(artifacts)])
            )

        return ToolResult.text(
            "\n".join(sections),
            metadata={
                "match_count": len(findings) + len(artifacts),
                "finding_count": len(findings),
                "artifact_count": len(artifacts),
            },
        )
