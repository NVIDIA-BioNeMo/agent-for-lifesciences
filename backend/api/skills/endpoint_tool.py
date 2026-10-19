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

"""Per-skill tools synthesized from a skill's data-type contract.

A skill on its own is prose: it tells the model to call ``http_request`` with a
hand-built JSON body containing raw sequences or SMILES. That defeats the
reference pattern -- the heavy data has to pass through the model's context in
both directions.

Given a confirmed :class:`~api.skills.contract.SkillContract`, this module
synthesizes a real tool per endpoint. The tool is the skill-side analogue of
:class:`~api.mcp.adapter.AdaptedMCPToolWrapper`:

- its parameters are ``*_ref`` strings annotated with ``x-aevom-reference``,
- refs are resolved to raw content server-side and injected into the request
  body at the declared ``json_path``,
- declared response fields are stored as typed artifacts and returned as refs,
- response fields nothing claimed are salvaged: numeric ones become ``array``
  artifacts loadable in ``python_execute``, so auxiliary scores (PAE, pLDDT) are
  neither discarded nor dumped into the context.

Outbound requests go through ``api.tools.http_common``, so these tools inherit
exactly the allowlist, SSRF, no-redirect and size-cap behavior of
``http_request``.
"""

from __future__ import annotations

import base64
import copy
import io
import json
import logging
from typing import Any, Optional

from api.storage import ArtifactNotFoundError
from api.storage.datatypes import DataType, friendly_name, is_text_type
from api.tools.base import BaseTool, DataIOCategory, ToolResult, ToolType
from api.tools.http_common import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_INLINE_BYTES,
    HttpToolError,
    clamp_timeout,
    http_request_tool_enabled,
    resolve_auth_header,
    send_request,
    validate_target,
)

from .contract import SkillContract, SkillEndpoint, split_json_path

logger = logging.getLogger(__name__)

# Numeric leaves shorter than this are left inline in the summary; storing a
# 3-element list as an artifact costs the model a round trip to learn nothing.
MIN_ARRAY_ELEMENTS = 8

# Guard against a pathological response tree exhausting the walk.
MAX_UNMAPPED_FIELDS = 24

# Scalars small enough to show the model directly in the result summary.
MAX_INLINE_SCALAR_CHARS = 200

# Cap on elements stored from an `each` output. DiffDock allows num_poses=100,
# and a hundred reference lines would swamp the result message the refs exist to
# keep small. Lists arrive rank-ordered, so the truncated tail is the least
# interesting end.
MAX_LIST_ITEMS = 10


def get_json_path(payload: Any, path: str) -> tuple[bool, Any]:
    """Read a mini-JSONPath out of a parsed JSON body.

    Returns ``(found, value)`` rather than raising, because a missing field in a
    third-party response is an expected outcome we report to the model, not a bug.
    """
    current = payload
    for token in split_json_path(path):
        if isinstance(token, int):
            if not isinstance(current, (list, tuple)) or token >= len(current):
                return False, None
            current = current[token]
        else:
            if not isinstance(current, dict) or token not in current:
                return False, None
            current = current[token]
    return True, current


def _grow_list(container: list, index: int, path: str, filler: Any) -> None:
    """Extend a list so ``index`` is writable, appending at most one new slot.

    Request bodies commonly nest under a single-element list (``$.polymers[0].sequence``)
    that no body_template declared, so appending the next position is expected.
    A jump past the end is not: that means the contract's index does not line up
    with the template, and silently padding with nulls would send a malformed body.
    """
    if index < len(container):
        return
    if index == len(container):
        container.append(filler)
        return
    raise HttpToolError(
        f"Cannot write to '{path}': index [{index}] skips past the end of a "
        f"{len(container)}-element list. Declare the surrounding structure in "
        "the endpoint's body_template."
    )


def set_json_path(body: dict[str, Any], path: str, value: Any) -> None:
    """Write a value into a nested dict/list body, creating containers as needed.

    Missing dicts are created, and lists grow by appending, so a contract can
    address ``$.polymers[0].sequence`` against an empty body.
    """
    tokens = split_json_path(path)
    current: Any = body

    for position, token in enumerate(tokens[:-1]):
        next_token = tokens[position + 1]
        filler: Any = [] if isinstance(next_token, int) else {}

        if isinstance(token, int):
            if not isinstance(current, list):
                raise HttpToolError(
                    f"Cannot write to '{path}': expected a list at index [{token}]."
                )
            _grow_list(current, token, path, filler)
            if not isinstance(current[token], (dict, list)):
                current[token] = filler
            current = current[token]
            continue

        if not isinstance(current, dict):
            raise HttpToolError(f"Cannot write to '{path}': '{token}' is not an object.")
        if not isinstance(current.get(token), (dict, list)):
            current[token] = filler
        current = current[token]

    last = tokens[-1]
    if isinstance(last, int):
        if not isinstance(current, list):
            raise HttpToolError(f"Cannot write to '{path}': parent is not a list.")
        _grow_list(current, last, path, None)
        current[last] = value
    else:
        if not isinstance(current, dict):
            raise HttpToolError(f"Cannot write to '{path}': parent is not an object.")
        current[last] = value


def _is_numeric_scalar(value: Any) -> bool:
    # bool is an int subclass, but a flag is not array data.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _numeric_leaf_count(value: Any, budget: int = 100_000) -> int:
    """Count numeric leaves in a nested list, or 0 if it is not uniformly numeric.

    Used to decide whether an unmapped field is worth persisting as an ``array``.
    """
    if _is_numeric_scalar(value):
        return 1
    if not isinstance(value, list) or not value:
        return 0
    total = 0
    for item in value:
        count = _numeric_leaf_count(item, budget - total)
        if count == 0:
            return 0
        total += count
        if total > budget:
            return total
    return total


def encode_array(value: Any) -> Optional[bytes]:
    """Encode a nested numeric list as a compressed ``.npz`` payload.

    Returns None when the value is not a rectangular numeric array, which is the
    signal to fall back to storing it as JSON.
    """
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a declared dependency
        logger.warning("numpy unavailable; cannot store array artifact")
        return None

    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return None
    if arr.dtype == object or not np.issubdtype(arr.dtype, np.number):
        return None

    buffer = io.BytesIO()
    np.savez_compressed(buffer, data=arr)
    return buffer.getvalue()


class SkillEndpointTool(BaseTool):
    """A tool synthesized from one endpoint of a skill contract.

    Attributes:
        skill_id: Owning skill, used to namespace the tool name.
        endpoint: The validated endpoint definition driving schema and execution.
    """

    def __init__(self, skill_id: str, skill_name: str, endpoint: SkillEndpoint):
        self.skill_id = skill_id
        self.skill_name = skill_name
        self.endpoint = endpoint

    @property
    def name(self) -> str:
        return self.endpoint.tool_name(self.skill_id)

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def summary_template(self) -> str:
        return f"Running {self.skill_name}: {self.endpoint.summary}..."

    @property
    def data_io_category(self) -> DataIOCategory:
        # Always at least OUTPUT_STORES: even with no declared outputs we persist
        # a large response body, which needs an injected storage service.
        if self.endpoint.inputs:
            return DataIOCategory.BOTH
        return DataIOCategory.OUTPUT_STORES

    @property
    def is_async(self) -> bool:
        return False

    @property
    def synthesized(self) -> bool:
        """Marks this tool as generated from a contract rather than hand-written."""
        return True

    def get_schema(self) -> dict[str, Any]:
        """Build the LLM-facing schema: reference params plus declared scalars."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in self.endpoint.inputs:
            ref_name = param.ref_param_name
            type_desc = friendly_name(param.data_type)
            description = param.description or (
                f"Reference to stored {type_desc}"
            )
            properties[ref_name] = {
                "type": "string",
                "description": (
                    f"{description} (e.g. '{param.data_type.value}_abc123'). "
                    "Use a reference returned by a previous tool call or by "
                    "python_execute with aevom.store_*(). Do not paste raw data."
                ),
                "x-aevom-reference": {"data_type": param.data_type.value},
            }
            if param.required:
                required.append(ref_name)

        for scalar in self.endpoint.scalar_params:
            prop: dict[str, Any] = {"type": scalar.type}
            if scalar.description:
                prop["description"] = scalar.description
            if scalar.enum:
                prop["enum"] = scalar.enum
            if scalar.default is not None:
                prop["default"] = scalar.default
            properties[scalar.name] = prop
            if scalar.required:
                required.append(scalar.name)

        # The contract's key name is only ever a guess: the extractor reads the
        # env-var name the docs happen to use (NGC_API_KEY) which need not match
        # what the user stored the key as (NVIDIA_API_KEY). Baked in with no
        # override, the auth error's "retry with one of those" is unfollowable
        # and the tool is permanently dead. A declared scalar wins the name.
        if self.endpoint.secret_name and "secret_name" not in properties:
            properties["secret_name"] = {
                "type": "string",
                "description": (
                    f"Optional. Defaults to '{self.endpoint.secret_name}'. Set this "
                    "only to retry after an authentication error that listed the "
                    "available key names, passing one of those. This is a key "
                    "NAME, never a key value."
                ),
            }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self._schema_description(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }

    def _schema_description(self) -> str:
        instead = (
            "Use this instead of http_request for this endpoint: it"
            if http_request_tool_enabled()
            else "It"
        )
        lines = [
            f"{self.endpoint.summary}",
            "",
            f"Provided by the '{self.skill_name}' skill. {instead} takes data "
            "references directly, builds the request body, and stores each output "
            "as a typed reference.",
        ]
        if self.endpoint.outputs:
            described = ", ".join(
                f"{o.name} (ranked list of {friendly_name(o.data_type)})"
                if o.each
                else f"{o.name} ({friendly_name(o.data_type)})"
                for o in self.endpoint.outputs
            )
            lines += ["", f"Returns references to: {described}."]
        if self.endpoint.capture_unmapped:
            lines.append(
                "Additional numeric fields in the response (e.g. confidence "
                "matrices) are stored as 'array' references you can load in "
                "python_execute with aevom.load()."
            )
        if self.endpoint.secret_name:
            lines += [
                "",
                f"Authentication uses the stored key '{self.endpoint.secret_name}'. "
                "If no such key is stored the error lists the names that are, and "
                "you can retry passing one of them as `secret_name`.",
            ]
        return "\n".join(lines)

    def _secret_name(self, arguments: dict[str, Any]) -> Optional[str]:
        """The stored key name to authenticate with, honouring a caller override.

        Only endpoints that declare a secret can be overridden, so a model
        cannot attach a stored key to an endpoint whose contract never asked for
        authentication.
        """
        if not self.endpoint.secret_name:
            return None
        override = arguments.get("secret_name")
        if isinstance(override, str) and override.strip():
            return override.strip()
        return self.endpoint.secret_name

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        try:
            body = await self._build_body(arguments)
            method, host = validate_target(self.endpoint.url, self.endpoint.method)
            headers: dict[str, str] = {}
            secret_value = await resolve_auth_header(
                self._session_id,
                self._secret_name(arguments),
                headers,
                auth_header=self.endpoint.auth_header,
                auth_prefix=self.endpoint.auth_prefix,
            )
            response = await send_request(
                method,
                self.endpoint.url,
                headers=headers,
                json_body=body if method != "GET" else None,
                params=body if method == "GET" else None,
                timeout=clamp_timeout(
                    self.endpoint.timeout_seconds or DEFAULT_TIMEOUT_SECONDS
                ),
                secret_value=secret_value,
            )
        except HttpToolError as exc:
            return ToolResult.error(str(exc))

        return await self._handle_response(response, host)

    async def _build_body(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Resolve reference arguments and assemble the outgoing request body.

        Raises:
            HttpToolError: If a required ref is missing, unresolvable, or the
                contract's json_path cannot be written.
        """
        body = copy.deepcopy(self.endpoint.body_template)

        for param in self.endpoint.inputs:
            ref = arguments.get(param.ref_param_name)
            if ref in (None, ""):
                if param.required:
                    raise HttpToolError(
                        f"'{param.ref_param_name}' is required. Pass a stored "
                        f"{friendly_name(param.data_type)} reference."
                    )
                continue

            raw = await self._resolve_ref(str(ref), param.data_type)
            if param.encoding == "base64":
                value: Any = base64.b64encode(raw).decode("ascii")
            else:
                value = raw.decode("utf-8", errors="replace")
            set_json_path(body, param.json_path, value)

        for scalar in self.endpoint.scalar_params:
            if scalar.name in arguments and arguments[scalar.name] is not None:
                set_json_path(body, scalar.json_path, arguments[scalar.name])
            elif scalar.default is not None:
                set_json_path(body, scalar.json_path, scalar.default)
            elif scalar.required:
                raise HttpToolError(f"'{scalar.name}' is required.")

        return body

    async def _resolve_ref(self, ref: str, data_type: DataType) -> bytes:
        """Fetch artifact content for a reference argument."""
        if self._storage is None:
            raise HttpToolError(
                f"Storage service not available; cannot resolve reference '{ref}'."
            )
        try:
            return await self._storage.retrieve(ref)
        except ArtifactNotFoundError as exc:
            raise HttpToolError(
                f"No stored data found for reference '{ref}'. Pass a valid "
                f"{friendly_name(data_type)} reference from a previous tool call."
            ) from exc
        except Exception as exc:  # noqa: BLE001 - reported to the model
            logger.exception("Failed to retrieve %s for skill tool %s", ref, self.name)
            raise HttpToolError(
                f"Failed to retrieve data for reference '{ref}': {exc}"
            ) from exc

    async def _handle_response(self, response: Any, host: str) -> ToolResult:
        """Turn a response into stored references plus a model-facing summary."""
        status = response.status_code
        body_bytes = response.content
        text = body_bytes.decode("utf-8", errors="replace")

        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            payload = None

        if status >= 400:
            preview = text[:MAX_INLINE_BYTES]
            return ToolResult.error(
                f"{self.name} failed: HTTP {status} from {host}. Response: {preview}"
            )

        if payload is None:
            return await self._store_opaque_body(body_bytes, status, host)

        stored: list[dict[str, Any]] = []
        missing: list[str] = []
        truncated: list[str] = []
        for field in self.endpoint.outputs:
            found, value = get_json_path(payload, field.json_path)
            if not found or value is None:
                missing.append(f"{field.name} (${field.json_path})")
                continue

            if field.each and isinstance(value, list):
                if len(value) > MAX_LIST_ITEMS:
                    truncated.append(f"{field.name} ({len(value)} -> {MAX_LIST_ITEMS})")
                for index, item in enumerate(value[:MAX_LIST_ITEMS], start=1):
                    if item is None:
                        continue
                    entry = await self._store_output(
                        f"{field.name} {index}",
                        item,
                        field.data_type,
                        field.description,
                    )
                    if entry:
                        stored.append(entry)
                continue

            entry = await self._store_output(
                field.name, value, field.data_type, field.description
            )
            if entry:
                stored.append(entry)

        # The salvage walk only visits top-level keys, so only single-token
        # output paths can claim one. A nested claimed path leaves its root
        # visible, which is harmless: a container of objects is not array data
        # and gets skipped.
        claimed = {
            f.json_path
            for f in self.endpoint.outputs
            if len(split_json_path(f.json_path)) == 1
        }
        extras: list[dict[str, Any]] = []
        inline: dict[str, Any] = {}
        if self.endpoint.capture_unmapped:
            extras, inline = await self._capture_unmapped(payload, claimed)

        return self._summarize(
            status, host, stored, extras, inline, missing, truncated
        )

    async def _store_output(
        self,
        name: str,
        value: Any,
        data_type: DataType,
        description: str,
    ) -> Optional[dict[str, Any]]:
        """Persist one declared output field as a typed artifact."""
        if data_type == DataType.ARRAY:
            content = encode_array(value)
            stored_type = DataType.ARRAY
            if content is None:
                # Declared as an array but not rectangular/numeric: keep the data
                # as JSON rather than losing it.
                content = json.dumps(value).encode("utf-8")
                stored_type = DataType.JSON
        elif data_type == DataType.JSON or not is_text_type(data_type):
            content = (
                value.encode("utf-8")
                if isinstance(value, str)
                else json.dumps(value).encode("utf-8")
            )
            stored_type = data_type if isinstance(value, str) else DataType.JSON
        else:
            if not isinstance(value, str):
                logger.warning(
                    "Skill %s output '%s' declared %s but got %s; storing as JSON",
                    self.name,
                    name,
                    data_type.value,
                    type(value).__name__,
                )
                content = json.dumps(value).encode("utf-8")
                stored_type = DataType.JSON
            else:
                content = value.encode("utf-8")
                stored_type = data_type

        ref = await self._store(
            stored_type,
            content,
            field_name=name,
            description=description or f"{name} from {self.skill_name}",
        )
        if ref is None:
            return None
        return {
            "name": name,
            "ref": ref,
            "data_type": stored_type.value,
            "bytes": len(content),
        }

    async def _capture_unmapped(
        self,
        payload: Any,
        claimed: set[str],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Salvage response fields no declared output claimed.

        Numeric arrays become ``array`` artifacts; small scalars are returned
        inline so the model can read them without a round trip. Everything else
        is skipped -- the full body is still available via the raw reference.
        """
        extras: list[dict[str, Any]] = []
        inline: dict[str, Any] = {}
        if not isinstance(payload, dict):
            return extras, inline

        for key, value in payload.items():
            if key in claimed or len(extras) >= MAX_UNMAPPED_FIELDS:
                continue

            if _is_numeric_scalar(value) or isinstance(value, bool):
                inline[key] = value
                continue
            if isinstance(value, str):
                if len(value) <= MAX_INLINE_SCALAR_CHARS:
                    inline[key] = value
                continue

            if _numeric_leaf_count(value) >= MIN_ARRAY_ELEMENTS:
                content = encode_array(value)
                if content is not None:
                    ref = await self._store(
                        DataType.ARRAY,
                        content,
                        field_name=key,
                        description=f"{key} from {self.skill_name}",
                    )
                    if ref:
                        extras.append(
                            {"name": key, "ref": ref, "data_type": DataType.ARRAY.value}
                        )

        return extras, inline

    async def _store_opaque_body(
        self, body_bytes: bytes, status: int, host: str
    ) -> ToolResult:
        """Store a non-JSON response body so nothing is silently dropped."""
        if len(body_bytes) <= MAX_INLINE_BYTES:
            return ToolResult.json_data(
                {
                    "status_code": status,
                    "body": body_bytes.decode("utf-8", errors="replace"),
                }
            )
        ref = await self._store(
            DataType.JSON,
            body_bytes,
            field_name="response",
            description=f"Non-JSON response from {host}",
        )
        if ref is None:
            preview = body_bytes[:MAX_INLINE_BYTES].decode("utf-8", errors="replace")
            return ToolResult.json_data(
                {"status_code": status, "truncated": True, "body_preview": preview}
            )
        return ToolResult.data_reference(
            ref,
            DataType.JSON.value,
            f"{self.name} returned a non-JSON body ({len(body_bytes)} bytes), stored "
            f"as '{ref}'. Load it with python_execute (aevom.load('{ref}')).",
        )

    async def _store(
        self,
        data_type: DataType,
        content: bytes,
        *,
        field_name: str,
        description: str,
    ) -> Optional[str]:
        """Store one artifact, returning None (not raising) on failure.

        A storage failure should degrade the result, not fail the whole call --
        the endpoint has already run and may not be idempotent.
        """
        if self._storage is None:
            logger.warning(
                "Storage unavailable; dropping '%s' output from %s", field_name, self.name
            )
            return None
        try:
            return await self._storage.store(
                data_type=data_type.value,
                content=content,
                metadata={
                    "source": "skill_endpoint",
                    "skill_id": self.skill_id,
                    "skill_name": self.skill_name,
                    "tool": self.name,
                    "field": field_name,
                },
                description=description[:256],
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to store %s output for %s: %s", field_name, self.name, exc)
            return None

    def _summarize(
        self,
        status: int,
        host: str,
        stored: list[dict[str, Any]],
        extras: list[dict[str, Any]],
        inline: dict[str, Any],
        missing: list[str],
        truncated: Optional[list[str]] = None,
    ) -> ToolResult:
        """Build the model-facing result: what was stored, and how to use it."""
        lines = [f"{self.endpoint.summary} — HTTP {status} from {host}."]

        if stored:
            lines.append("")
            lines.append("Stored outputs:")
            for entry in stored:
                lines.append(
                    f"  - {entry['name']}: '{entry['ref']}' "
                    f"({friendly_name(entry['data_type'])})"
                )
            lines.append(
                "Pass these references to other tools that accept the same data type."
            )

        if extras:
            lines.append("")
            lines.append("Additional numeric outputs stored as arrays:")
            for entry in extras:
                lines.append(f"  - {entry['name']}: '{entry['ref']}'")
            lines.append(
                "Load these in python_execute with aevom.load('<ref>') to inspect or plot."
            )

        if inline:
            lines.append("")
            lines.append(f"Other response values: {json.dumps(inline)[:1000]}")

        if truncated:
            lines.append("")
            lines.append(
                "Only the first " + str(MAX_LIST_ITEMS) + " items were stored for: "
                + ", ".join(truncated)
                + ". Lists are rank-ordered, so these are the best-scoring ones."
            )

        if missing:
            lines.append("")
            lines.append(
                "Expected but not present in the response: " + ", ".join(missing) + "."
            )

        message = "\n".join(lines)
        all_refs = stored + extras

        if not all_refs:
            return ToolResult.json_data(
                {
                    "status_code": status,
                    "message": message,
                    "values": inline,
                }
            )

        primary = all_refs[0]
        return ToolResult.data_reference(
            reference=primary["ref"],
            data_type=primary["data_type"],
            message=message,
            metadata={
                "skill_id": self.skill_id,
                "tool": self.name,
                "outputs": {e["name"]: e["ref"] for e in all_refs},
                "all_references": [e["ref"] for e in all_refs],
                "values": inline,
                "status_code": status,
            },
        )


def synthesize_skill_tools(
    skill_id: str,
    skill_name: str,
    contract: Optional[SkillContract],
) -> list[SkillEndpointTool]:
    """Build the tools for a skill, or nothing if its contract is not actionable.

    An unconfirmed contract yields no tools: confirming is also how the user
    authorizes outbound requests to the contract's hosts.
    """
    if contract is None or not contract.is_actionable:
        return []
    return [
        SkillEndpointTool(skill_id, skill_name, endpoint)
        for endpoint in contract.endpoints
    ]
