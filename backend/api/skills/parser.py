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

"""Parsing and validation for Aevom skills.

A skill is a ``SKILL.md`` file (YAML frontmatter + markdown body) plus an
optional set of bundled read-only reference files. This module turns a raw
bundle of files (dropped directory, unzipped archive, or fetched GitHub
subtree) into a validated :class:`ParsedSkill`.

Security notes:
- Paths are rejected if they are absolute or contain ``..`` segments
  (defends against path traversal / zip-slip).
- Only ``.md`` and ``.txt`` resource files are kept (v1 is read-only, no
  executable payloads). Other files are silently skipped.
- Per-file and total size caps bound the amount of data a single install
  can push into storage.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

import yaml

from .contract import SkillContract, SkillContractError, parse_contract

# --- Limits -----------------------------------------------------------------

ALLOWED_RESOURCE_EXTENSIONS = {".md", ".txt"}
MAX_SKILL_FILES = 64
MAX_FILE_BYTES = 1_000_000  # 1 MB per file
MAX_TOTAL_BYTES = 5_000_000  # 5 MB per skill

SKILL_FILENAME = "skill.md"

_FRONTMATTER_RE = re.compile(
    r"^\ufeff?---\s*\n(?P<meta>.*?)\n---\s*\n?(?P<body>.*)$",
    re.DOTALL,
)


class SkillParseError(ValueError):
    """Raised when a skill bundle is malformed or fails validation."""


@dataclass
class ParsedResource:
    """A single bundled reference file, path relative to the skill root."""

    path: str
    content: bytes


@dataclass
class ParsedSkill:
    """A fully parsed and validated skill, ready to be stored."""

    name: str
    description: str
    body: str
    allowed_tools: list[str] = field(default_factory=list)
    resources: list[ParsedResource] = field(default_factory=list)
    # Present only when the author declared an `endpoints:` block. Declared
    # contracts are trusted and pre-confirmed; otherwise the install flow may
    # attach an LLM-extracted (unconfirmed) contract later.
    contract: Optional[SkillContract] = None


def split_frontmatter(text: str) -> tuple[dict, str]:
    """Split a ``SKILL.md`` string into (frontmatter dict, markdown body).

    Raises:
        SkillParseError: If no YAML frontmatter block is present or the
            frontmatter is not a mapping.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise SkillParseError(
            "SKILL.md is missing a YAML frontmatter block "
            "(expected '---' delimited header with name and description)."
        )
    try:
        meta = yaml.safe_load(match.group("meta")) or {}
    except yaml.YAMLError as exc:
        raise SkillParseError(f"Invalid YAML frontmatter: {exc}") from exc
    if not isinstance(meta, dict):
        raise SkillParseError("SKILL.md frontmatter must be a YAML mapping.")
    return meta, match.group("body").strip()


def _normalize_allowed_tools(value: object) -> list[str]:
    """Coerce the ``allowed_tools`` frontmatter field into a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    raise SkillParseError("'allowed_tools' must be a list or comma-separated string.")


def _validate_path(path: str) -> str:
    """Return a normalized posix path, rejecting traversal / absolute paths."""
    normalized = path.replace("\\", "/").strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        raise SkillParseError("Encountered an empty file path in the skill bundle.")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise SkillParseError(f"Absolute paths are not allowed: '{path}'")
    if any(segment == ".." for segment in normalized.split("/")):
        raise SkillParseError(f"Path traversal is not allowed: '{path}'")
    return normalized


def parse_skill_markdown(
    skill_md_text: str,
    resources: Iterable[ParsedResource] | None = None,
) -> ParsedSkill:
    """Parse a single ``SKILL.md`` text plus already-resolved resources."""
    meta, body = split_frontmatter(skill_md_text)

    name = meta.get("name")
    description = meta.get("description")
    if not name or not str(name).strip():
        raise SkillParseError("SKILL.md frontmatter is missing required field 'name'.")
    if not description or not str(description).strip():
        raise SkillParseError(
            "SKILL.md frontmatter is missing required field 'description'."
        )

    return ParsedSkill(
        name=str(name).strip(),
        description=str(description).strip(),
        body=body,
        allowed_tools=_normalize_allowed_tools(meta.get("allowed_tools")),
        resources=list(resources or []),
        contract=_parse_declared_contract(meta),
    )


def _parse_declared_contract(meta: dict) -> Optional[SkillContract]:
    """Build a contract from an ``endpoints:`` (or ``contract:``) frontmatter block.

    A declared contract is authored deliberately by the skill author, so it is
    marked confirmed immediately -- unlike an LLM-extracted one, which needs the
    user to approve its request shape and outbound host.

    Raises:
        SkillParseError: If the declared block is malformed.
    """
    raw = meta.get("endpoints")
    if raw is None:
        raw = meta.get("contract")
    if raw is None:
        return None

    try:
        contract = parse_contract(raw, source="declared")
    except SkillContractError as exc:
        raise SkillParseError(f"Invalid 'endpoints' block in SKILL.md: {exc}") from exc

    return contract.model_copy(update={"confirmed": True})


def parse_skill_bundle(files: Iterable[tuple[str, bytes]]) -> ParsedSkill:
    """Parse a bundle of files into a validated :class:`ParsedSkill`.

    Locates the shallowest ``SKILL.md``, treats its directory as the skill
    root, and keeps sibling/descendant ``.md``/``.txt`` files as resources
    with paths relative to that root.

    Args:
        files: Iterable of ``(path, content_bytes)`` pairs. Paths may include
            an enclosing top-level directory (e.g. ``my-skill/SKILL.md``).

    Raises:
        SkillParseError: On validation failure (no SKILL.md, traversal, size).
    """
    normalized: list[tuple[str, bytes]] = []
    for path, content in files:
        normalized.append((_validate_path(path), content))

    if len(normalized) > MAX_SKILL_FILES:
        raise SkillParseError(
            f"Skill bundle has too many files "
            f"({len(normalized)} > {MAX_SKILL_FILES})."
        )

    # Find the shallowest SKILL.md (fewest path segments).
    skill_candidates = [
        (path, content)
        for path, content in normalized
        if path.rsplit("/", 1)[-1].lower() == SKILL_FILENAME
    ]
    if not skill_candidates:
        raise SkillParseError("No SKILL.md found in the uploaded skill bundle.")
    skill_candidates.sort(key=lambda pc: pc[0].count("/"))
    skill_path, skill_content = skill_candidates[0]

    if len(skill_content) > MAX_FILE_BYTES:
        raise SkillParseError("SKILL.md exceeds the maximum allowed size.")

    root = skill_path.rsplit("/", 1)[0] if "/" in skill_path else ""
    root_prefix = f"{root}/" if root else ""

    try:
        skill_md_text = skill_content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SkillParseError("SKILL.md must be UTF-8 encoded text.") from exc

    total_bytes = len(skill_content)
    resources: list[ParsedResource] = []
    seen_rel_paths: set[str] = set()
    for path, content in normalized:
        if path == skill_path:
            continue
        if root_prefix and not path.startswith(root_prefix):
            # File lives outside the detected skill root; ignore it.
            continue
        rel_path = path[len(root_prefix):] if root_prefix else path
        if not rel_path or rel_path in seen_rel_paths:
            continue
        ext = "." + rel_path.rsplit(".", 1)[-1].lower() if "." in rel_path else ""
        if ext not in ALLOWED_RESOURCE_EXTENSIONS:
            # v1 keeps only read-only text references; skip everything else.
            continue
        if len(content) > MAX_FILE_BYTES:
            raise SkillParseError(
                f"Resource '{rel_path}' exceeds the maximum allowed file size."
            )
        total_bytes += len(content)
        if total_bytes > MAX_TOTAL_BYTES:
            raise SkillParseError("Skill bundle exceeds the maximum total size.")
        seen_rel_paths.add(rel_path)
        resources.append(ParsedResource(path=rel_path, content=content))

    return parse_skill_markdown(skill_md_text, resources)
