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

"""Global registry of curated first-party skills.

Curated skills come from two places, both baked into the image so every API pod
has an identical copy (contrast with user-installed skills, which live in Redis +
Postgres). This mirrors ``_register_default_tools()`` in ``api.tools.registry``
for builtin tools.

1. **Aevom's own skills**, under ``backend/skills/*/SKILL.md``. Every
   subdirectory is loaded.
2. **NIM skills from the pinned bionemo-agent-toolkit submodule**, whose prose is
   maintained upstream and must stay byte-identical. These are loaded from the
   allowlist in :mod:`api.skills.binding`, *not* by scanning the submodule, so
   that moving the pin cannot silently add a skill whose tool Aevom does not
   have. See that module for why they carry a binding instead of a contract.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .binding import (
    LIBRARY_SKILLS,
    NIM_SKILL_BINDINGS,
    SkillBinding,
    ToolAnnouncement,
    library_skills_root,
    nim_skills_root,
    resolve_announcement,
)
from .contract import SkillContract
from .parser import SkillParseError, parse_skill_bundle

logger = logging.getLogger(__name__)

# Default location of curated skills: backend/skills (sibling of backend/api).
_DEFAULT_SKILLS_DIR = Path(__file__).resolve().parents[2] / "skills"

BUILTIN_SOURCE = "builtin"

# Files loaded from an upstream-shaped skill bundle. The NIM bundles also ship an
# ``evals/`` tree -- a Harbor eval harness with Dockerfiles, PDB fixtures and its
# own ``instruction.md``. That is test infrastructure for the toolkit, not
# reference material for a model, and ``instruction.md`` would otherwise be
# offered to the model as a loadable resource. Library skills follow the same
# layout so that first-party prose can be swapped for a submodule without
# touching this reader.
_SKILL_FILE = "SKILL.md"
_REFERENCE_DIR = "references"


@dataclass
class SkillDefinition:
    """A curated skill fully loaded into memory (body + text resources)."""

    id: str
    name: str
    description: str
    body: str
    allowed_tools: list[str] = field(default_factory=list)
    resources: dict[str, str] = field(default_factory=dict)  # rel path -> text
    source: str = BUILTIN_SOURCE
    # Declared in frontmatter and therefore pre-confirmed; builtin skills ship
    # in the image, so there is no user to review an extracted contract.
    contract: Optional[SkillContract] = None
    # Set for skills whose work is already served by a builtin tool. Mutually
    # exclusive with `contract` in practice: a binding announces tools that
    # already exist, a contract asks for tools to be synthesized.
    binding: Optional[SkillBinding] = None

    @property
    def announcement(self) -> Optional[ToolAnnouncement]:
        """Tools to announce to the model when it reads this skill."""
        return resolve_announcement(self.id, self.binding, self.contract)


class SkillRegistry:
    """In-memory registry of curated builtin skills keyed by skill id."""

    def __init__(self) -> None:
        self._skills: dict[str, SkillDefinition] = {}

    def load_builtin_skills(self, base_dir: Optional[Path | str] = None) -> None:
        """Discover and load curated skills from ``base_dir``, then bound NIM skills.

        Idempotent: clears and reloads the registry on each call. Each skill
        directory is loaded independently so one malformed skill does not
        block the others.
        """
        directory = Path(base_dir) if base_dir is not None else Path(
            os.environ.get("AEVOM_SKILLS_DIR", str(_DEFAULT_SKILLS_DIR))
        )
        self._skills.clear()

        if directory.is_dir():
            for entry in sorted(directory.iterdir()):
                if not entry.is_dir():
                    continue
                if not (entry / _SKILL_FILE).is_file():
                    continue
                self._try_load(entry.name, lambda e=entry: self._load_one(e))
        else:
            logger.info("No curated skills directory found at %s", directory)

        self._load_bound_nim_skills()
        self._load_bound_library_skills()

        logger.info("Curated skill registry initialized with %d skills", len(self._skills))

    def _try_load(self, label: str, load) -> None:
        """Run one skill load, keeping a failure from taking down the others."""
        try:
            definition = load()
        except Exception as exc:  # noqa: BLE001 - isolate per-skill failures
            logger.warning("Failed to load curated skill '%s': %s", label, exc)
            return
        self._skills[definition.id] = definition
        logger.info(
            "Loaded curated skill '%s' (%d resources)",
            definition.id,
            len(definition.resources),
        )

    def _load_bound_nim_skills(self) -> None:
        """Load exactly the NIM skills named in the binding allowlist.

        Driven by the binding table rather than by listing the submodule: the
        submodule carries skills for NIMs Aevom has no tool for, and a skill with
        no tool behind it would tell the model to hand-build HTTP requests to an
        endpoint it was never authorized to reach.
        """
        root = nim_skills_root()
        if root is None:
            # A source checkout without `git submodule update --init`. The NIM
            # tools still work; the model just loses the prose explaining them.
            logger.warning(
                "bionemo-agent-toolkit submodule is not present, so %d NIM skills "
                "were not loaded. Run 'make submodules'.",
                len(NIM_SKILL_BINDINGS),
            )
            return

        for binding in NIM_SKILL_BINDINGS:
            skill_dir = root / binding.skill_id
            if not (skill_dir / _SKILL_FILE).is_file():
                # The pin moved and a directory was renamed or removed upstream.
                logger.warning(
                    "Skill binding '%s' names tools %s but no SKILL.md exists at %s",
                    binding.skill_id,
                    ", ".join(binding.tool_names),
                    skill_dir,
                )
                continue
            self._try_load(
                binding.skill_id,
                lambda b=binding, d=skill_dir: self._load_one(
                    d, binding=b, upstream_layout=True
                ),
            )

    def _load_bound_library_skills(self) -> None:
        """Load exactly the library skills named in the allowlist.

        Mirrors ``_load_bound_nim_skills``, with one difference: an entry may
        have no binding, because most of these skills document a sandbox package
        that has no builtin tool behind it.
        """
        root = library_skills_root()
        if root is None:
            logger.warning(
                "No library skills directory found, so %d library skills were "
                "not loaded.",
                len(LIBRARY_SKILLS),
            )
            return

        for entry in LIBRARY_SKILLS:
            skill_dir = root / entry.skill_id
            if not (skill_dir / _SKILL_FILE).is_file():
                logger.warning(
                    "Library skill '%s' is allowlisted but no SKILL.md exists at %s",
                    entry.skill_id,
                    skill_dir,
                )
                continue
            self._try_load(
                entry.skill_id,
                lambda e=entry, d=skill_dir: self._load_one(
                    d, binding=e.binding, upstream_layout=True
                ),
            )

    def _load_one(
        self,
        skill_dir: Path,
        binding: Optional[SkillBinding] = None,
        upstream_layout: bool = False,
    ) -> SkillDefinition:
        # Which reader to use follows the bundle's layout, not whether a binding
        # exists: library skills use the upstream layout but most carry no tool.
        files = (
            self._read_upstream_bundle(skill_dir)
            if upstream_layout
            else self._read_bundle(skill_dir)
        )
        if not files:
            raise SkillParseError(f"Skill directory '{skill_dir.name}' is empty.")

        parsed = parse_skill_bundle(files)
        resources = {
            resource.path: resource.content.decode("utf-8", errors="replace")
            for resource in parsed.resources
        }
        return SkillDefinition(
            id=skill_dir.name,
            name=parsed.name,
            description=parsed.description,
            body=parsed.body,
            allowed_tools=parsed.allowed_tools,
            resources=resources,
            source=BUILTIN_SOURCE,
            contract=parsed.contract,
            binding=binding,
        )

    @staticmethod
    def _read_bundle(skill_dir: Path) -> list[tuple[str, bytes]]:
        """Every file under a first-party skill directory."""
        return [
            (path.relative_to(skill_dir).as_posix(), path.read_bytes())
            for path in sorted(skill_dir.rglob("*"))
            if path.is_file()
        ]

    @staticmethod
    def _read_upstream_bundle(skill_dir: Path) -> list[tuple[str, bytes]]:
        """``SKILL.md`` plus ``references/`` from an upstream-shaped bundle.

        Restricted to the two paths the toolkit documents as skill content, so
        the ``evals/`` harness stays out of the resource list the model sees.
        """
        files = [(_SKILL_FILE, (skill_dir / _SKILL_FILE).read_bytes())]
        for path in sorted((skill_dir / _REFERENCE_DIR).glob("*.md")):
            files.append((f"{_REFERENCE_DIR}/{path.name}", path.read_bytes()))
        return files

    def list(self) -> list[SkillDefinition]:
        """Return all loaded curated skills."""
        return list(self._skills.values())

    def get(self, skill_id: str) -> Optional[SkillDefinition]:
        """Return a curated skill by id, or None."""
        return self._skills.get(skill_id)

    def __contains__(self, skill_id: str) -> bool:
        return skill_id in self._skills

    def __len__(self) -> int:
        return len(self._skills)


_skill_registry: Optional[SkillRegistry] = None


def get_skill_registry() -> SkillRegistry:
    """Get the global curated-skill registry singleton."""
    global _skill_registry
    if _skill_registry is None:
        _skill_registry = SkillRegistry()
        logger.info("Created curated skill registry")
    return _skill_registry
