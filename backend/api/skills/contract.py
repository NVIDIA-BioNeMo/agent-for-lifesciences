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

"""Data-type contract for a skill's HTTP endpoints.

A skill is prose, not a schema, so nothing about it is callable on its own. A
*contract* is the missing machine-readable part: it says which request fields
carry Aevom first-class data (SMILES, PDB, sequence, MSA, ...) and where the
interesting values live in the response. Given a contract, ``endpoint_tool``
synthesizes a real tool whose parameters are storage references, so the model
never handles raw molecular data.

A contract arrives one of two ways:

- **declared**: an ``endpoints:`` block in the ``SKILL.md`` frontmatter. Trusted
  and auto-confirmed, since the skill author wrote it deliberately.
- **llm**: extracted from the skill's prose at install time by ``extract``.
  Starts unconfirmed and must be reviewed by the user, because it both guesses
  a request shape and names an outbound host.

These models are Pydantic (the rest of ``api.skills`` uses dataclasses) because
the input is LLM-generated or hand-edited in the UI, so validation is doing real
work rather than documenting types.

``json_path`` uses a deliberately tiny subset of JSONPath -- dot-separated keys
with optional ``[i]`` indices, e.g. ``$.structures[0].structure``. Skill
endpoints address plain JSON bodies; supporting filters or wildcards would add
an evaluator with no caller.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from api.storage.datatypes import DataType

# Dot/index path: "$.a.b[0].c", "a.b", "$.results[2]".
_JSON_PATH_RE = re.compile(r"^\$?\.?[A-Za-z_][\w\-]*(?:\[\d+\]|\.[A-Za-z_][\w\-]*)*$")

# Tool names reach the LLM as OpenAI function names, which must be identifier-safe.
_TOOL_SUFFIX_RE = re.compile(r"^[a-z][a-z0-9_]{0,39}$")

ALLOWED_METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")

# Bounds mirroring tools.http_request so a contract cannot request a longer
# timeout than the shared HTTP layer permits.
MIN_TIMEOUT_SECONDS = 1
MAX_TIMEOUT_SECONDS = 180


class SkillContractError(ValueError):
    """Raised when a declared or extracted contract fails validation."""


def validate_json_path(path: str) -> str:
    """Normalize and validate a mini-JSONPath, returning it without the ``$.``.

    Raises:
        ValueError: If the path is empty or uses unsupported syntax.
    """
    raw = (path or "").strip()
    if not raw:
        raise ValueError("json_path must not be empty")
    if not _JSON_PATH_RE.match(raw):
        raise ValueError(
            f"Unsupported json_path '{path}'. Use dot/index paths only, "
            "e.g. '$.structures[0].structure'."
        )
    return raw.lstrip("$").lstrip(".")


def split_json_path(path: str) -> list[str | int]:
    """Split a validated mini-JSONPath into key/index tokens.

    ``"structures[0].structure"`` -> ``["structures", 0, "structure"]``
    """
    tokens: list[str | int] = []
    for segment in validate_json_path(path).split("."):
        name, *indices = re.split(r"\[(\d+)\]", segment)
        if name:
            tokens.append(name)
        tokens.extend(int(i) for i in indices if i != "")
    return tokens


class SkillInputParam(BaseModel):
    """A request field that carries Aevom first-class data.

    The synthesized tool exposes this as ``{name}_ref`` (or ``{name}_{type}_ref``)
    and resolves the reference to raw content before sending the request.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Logical field name, used to derive the ref param.")
    data_type: DataType = Field(..., description="Aevom storage type this field carries.")
    json_path: str = Field(
        default="",
        description="Where to place the value in the request body. Defaults to '$.{name}'.",
    )
    required: bool = True
    encoding: Literal["raw", "base64"] = Field(
        default="raw",
        description="'base64' for endpoints expecting base64-encoded payloads.",
    )
    description: str = ""

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        name = (value or "").strip()
        if not name:
            raise ValueError("input name must not be empty")
        return name

    @model_validator(mode="after")
    def _default_json_path(self) -> "SkillInputParam":
        # Default the body location to the field's own name, which is the shape
        # nearly every JSON API uses.
        object.__setattr__(self, "json_path", validate_json_path(self.json_path or self.name))
        return self

    @property
    def ref_param_name(self) -> str:
        """Name of the ``*_ref`` parameter the LLM fills in.

        Matches ``api.mcp.adapter._ref_param_name`` so refs look the same
        whether they came from an MCP tool or a skill.
        """
        lower = self.name.lower()
        if lower.endswith(self.data_type.value):
            return f"{self.name}_ref"
        return f"{self.name}_{self.data_type.value}_ref"


class SkillScalarParam(BaseModel):
    """A plain (non-reference) request field the model fills directly.

    Knobs like ``num_steps`` or ``recycles`` carry no heavy data, so they pass
    through to the LLM-facing schema unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    type: Literal["string", "integer", "number", "boolean"] = "string"
    json_path: str = ""
    required: bool = False
    description: str = ""
    enum: Optional[list[Any]] = None
    default: Optional[Any] = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        name = (value or "").strip()
        if not name:
            raise ValueError("scalar param name must not be empty")
        return name

    @model_validator(mode="after")
    def _default_json_path(self) -> "SkillScalarParam":
        object.__setattr__(self, "json_path", validate_json_path(self.json_path or self.name))
        return self


class SkillOutputField(BaseModel):
    """A response field worth persisting as its own artifact."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Short label used in the result message.")
    json_path: str = Field(..., description="Where the value lives in the response body.")
    data_type: DataType = Field(
        ...,
        description="Aevom storage type. Use 'array' for numeric matrices, 'json' for structured blobs.",
    )
    description: str = ""
    each: bool = Field(
        default=False,
        description=(
            "The path resolves to a list, and every element is its own artifact of "
            "data_type. Docking endpoints return ranked poses this way; without "
            "this only '$.poses[0]' could be claimed and the rest would be lost."
        ),
    )

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        name = (value or "").strip()
        if not name:
            raise ValueError("output name must not be empty")
        return name

    @field_validator("json_path")
    @classmethod
    def _check_json_path(cls, value: str) -> str:
        return validate_json_path(value)


class SkillEndpoint(BaseModel):
    """One callable HTTP endpoint belonging to a skill."""

    model_config = ConfigDict(extra="forbid")

    tool_suffix: str = Field(
        ...,
        description="Suffix for the synthesized tool name: skill_{skill_id}_{tool_suffix}.",
    )
    summary: str = Field(..., description="What the endpoint does; becomes the tool description.")
    url: str
    method: Literal["GET", "POST", "PUT", "PATCH", "DELETE"] = "POST"
    secret_name: Optional[str] = Field(
        default=None,
        description="Name of a stored API key to send as a Bearer token.",
    )
    auth_header: Optional[str] = None
    auth_prefix: Optional[str] = None
    body_template: dict[str, Any] = Field(
        default_factory=dict,
        description="Static request body; typed inputs are merged in at their json_paths.",
    )
    inputs: list[SkillInputParam] = Field(default_factory=list)
    scalar_params: list[SkillScalarParam] = Field(default_factory=list)
    outputs: list[SkillOutputField] = Field(default_factory=list)
    capture_unmapped: bool = Field(
        default=True,
        description=(
            "Persist response fields no declared output claimed. Array-like values "
            "become 'array' artifacts loadable in python_execute; this is what keeps "
            "auxiliary scores (PAE, pLDDT) from being discarded."
        ),
    )
    timeout_seconds: Optional[int] = Field(
        default=None, ge=MIN_TIMEOUT_SECONDS, le=MAX_TIMEOUT_SECONDS
    )

    @field_validator("tool_suffix")
    @classmethod
    def _check_suffix(cls, value: str) -> str:
        suffix = (value or "").strip().lower().replace("-", "_")
        if not _TOOL_SUFFIX_RE.match(suffix):
            raise ValueError(
                f"Invalid tool_suffix '{value}'. Use lowercase letters, digits and "
                "underscores, starting with a letter (max 40 chars)."
            )
        return suffix

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        url = (value or "").strip()
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Endpoint url must be an http(s) URL, got '{value}'.")
        return url

    @field_validator("summary")
    @classmethod
    def _check_summary(cls, value: str) -> str:
        summary = (value or "").strip()
        if not summary:
            raise ValueError("Endpoint summary must not be empty")
        return summary

    @model_validator(mode="after")
    def _check_unique_names(self) -> "SkillEndpoint":
        ref_names = [i.ref_param_name for i in self.inputs]
        scalar_names = [s.name for s in self.scalar_params]
        all_names = ref_names + scalar_names
        duplicates = {n for n in all_names if all_names.count(n) > 1}
        if duplicates:
            raise ValueError(
                f"Endpoint '{self.tool_suffix}' has colliding parameter names: {sorted(duplicates)}"
            )
        output_names = [o.name for o in self.outputs]
        dup_outputs = {n for n in output_names if output_names.count(n) > 1}
        if dup_outputs:
            raise ValueError(
                f"Endpoint '{self.tool_suffix}' has duplicate output names: {sorted(dup_outputs)}"
            )
        return self

    @property
    def host(self) -> str:
        """Hostname this endpoint calls, used for the install-time consent gate."""
        from urllib.parse import urlparse

        return (urlparse(self.url).hostname or "").lower()

    def tool_name(self, skill_id: str) -> str:
        """Full synthesized tool name for a given skill id."""
        return f"{skill_id}_{self.tool_suffix}"


class SkillContract(BaseModel):
    """The full data-type contract attached to a skill.

    ``confirmed`` gates synthesis. An LLM-extracted contract is unconfirmed
    until the user approves it in the install UI, because approving it also
    authorizes outbound requests to :attr:`hosts`.
    """

    model_config = ConfigDict(extra="forbid")

    endpoints: list[SkillEndpoint] = Field(default_factory=list)
    detected_types: list[DataType] = Field(
        default_factory=list,
        description="Types the regex detector saw in the skill prose (informational).",
    )
    source: Literal["declared", "llm", "none"] = "none"
    confirmed: bool = False

    @model_validator(mode="after")
    def _check_unique_suffixes(self) -> "SkillContract":
        suffixes = [e.tool_suffix for e in self.endpoints]
        duplicates = {s for s in suffixes if suffixes.count(s) > 1}
        if duplicates:
            raise ValueError(f"Duplicate endpoint tool_suffix values: {sorted(duplicates)}")
        return self

    @property
    def hosts(self) -> list[str]:
        """Distinct hosts the contract would authorize, for the consent gate."""
        seen: list[str] = []
        for endpoint in self.endpoints:
            host = endpoint.host
            if host and host not in seen:
                seen.append(host)
        return seen

    @property
    def is_actionable(self) -> bool:
        """Whether this contract can produce synthesized tools right now."""
        return self.confirmed and bool(self.endpoints)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the Redis catalog (enums become plain strings)."""
        return self.model_dump(mode="json")


def parse_contract(data: Any, *, source: Optional[str] = None) -> SkillContract:
    """Validate raw contract data into a :class:`SkillContract`.

    Args:
        data: A mapping (full contract), a list (bare ``endpoints``), or None.
        source: Overrides ``source`` when the caller knows the provenance.

    Raises:
        SkillContractError: If validation fails.
    """
    if data is None:
        return SkillContract(source="none")

    # Frontmatter commonly declares just the endpoint list.
    payload: dict[str, Any]
    if isinstance(data, list):
        payload = {"endpoints": data}
    elif isinstance(data, dict):
        payload = dict(data)
    else:
        raise SkillContractError(
            "Skill contract must be a mapping or a list of endpoints."
        )

    if source is not None:
        payload["source"] = source

    try:
        contract = SkillContract.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surfaced as a 400 to the caller
        raise SkillContractError(str(exc)) from exc

    return contract
