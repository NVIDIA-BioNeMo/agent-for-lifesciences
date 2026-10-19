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

"""Load the OpenShell sandbox policy for python_execute.

The policy is authored as YAML in the schema OpenShell documents
(``deploy/openshell/pyexec-policy.yaml``) and sent with every CreateSandbox
call. The gateway API takes the ``openshell.sandbox.v1.SandboxPolicy`` proto,
which mirrors the YAML except for two spellings: ``filesystem_policy`` is
``filesystem``, and the string enums (``enforcement: enforce``) are proto enum
names (``NETWORK_ENFORCEMENT_MODE_ENFORCE``).
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

_REPO_POLICY_PATH = Path(__file__).resolve().parents[3] / "deploy" / "openshell" / "pyexec-policy.yaml"
# Where backend/Dockerfile copies it.
_IMAGE_POLICY_PATH = Path("/app/deploy/openshell/pyexec-policy.yaml")
DEFAULT_POLICY_PATH = _REPO_POLICY_PATH if _REPO_POLICY_PATH.exists() else _IMAGE_POLICY_PATH

_PLACEHOLDER = re.compile(r"\$\{([A-Z0-9_]+)\}")

_ENUM_PREFIXES = {
    "enforcement": "NETWORK_ENFORCEMENT_MODE_",
    "access": "NETWORK_ACCESS_PRESET_",
    "tls": "NETWORK_TLS_MODE_",
}


class PolicyError(ValueError):
    """The policy file is missing, unparseable, or references an unset variable."""


def render_policy_text(text: str, env: Mapping[str, str]) -> str:
    """Substitute ``${VAR}`` placeholders, failing on any that are unset.

    An unset variable would otherwise produce a policy naming a host like
    ``${OPENSHELL_BACKEND_HOST}``, which the gateway accepts and which silently
    blocks every sandbox callback.
    """
    missing = sorted({m for m in _PLACEHOLDER.findall(text) if not env.get(m)})
    if missing:
        raise PolicyError(f"sandbox policy needs these environment variables: {', '.join(missing)}")
    return _PLACEHOLDER.sub(lambda m: env[m.group(1)], text)


def load_policy_document(
    path: str | os.PathLike[str] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Read, render and parse the policy YAML into a plain dict."""
    policy_path = Path(path) if path else DEFAULT_POLICY_PATH
    try:
        text = policy_path.read_text()
    except OSError as exc:
        raise PolicyError(f"cannot read sandbox policy {policy_path}: {exc}") from exc
    rendered = render_policy_text(text, os.environ if env is None else env)
    try:
        document = yaml.safe_load(rendered)
    except yaml.YAMLError as exc:
        raise PolicyError(f"sandbox policy {policy_path} is not valid YAML: {exc}") from exc
    if not isinstance(document, dict) or document.get("version") != 1:
        raise PolicyError(f"sandbox policy {policy_path} must be a mapping with version: 1")
    return document


def _enum_name(field: str, value: str) -> str:
    return _ENUM_PREFIXES[field] + value.strip().upper().replace("-", "_")


def policy_to_proto_dict(document: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the YAML schema into the field names of the SandboxPolicy proto."""
    doc = copy.deepcopy(dict(document))
    out: dict[str, Any] = {"version": doc.pop("version")}
    if "filesystem_policy" in doc:
        out["filesystem"] = doc.pop("filesystem_policy")
    for section in ("landlock", "process", "network_middlewares"):
        if section in doc:
            out[section] = doc.pop(section)
    if "network_policies" in doc:
        rules = {}
        for key, rule in (doc.pop("network_policies") or {}).items():
            rule = dict(rule)
            rule.setdefault("name", key)
            endpoints = []
            for endpoint in rule.get("endpoints", []):
                endpoint = dict(endpoint)
                for field in _ENUM_PREFIXES:
                    if field in endpoint:
                        endpoint[field] = _enum_name(field, endpoint[field])
                endpoints.append(endpoint)
            rule["endpoints"] = endpoints
            rules[key] = rule
        out["network_policies"] = rules
    if doc:
        raise PolicyError(f"unknown top-level policy sections: {', '.join(sorted(doc))}")
    return out


def build_policy_proto(document: Mapping[str, Any]):
    """Build an ``openshell.sandbox.v1.SandboxPolicy`` from a parsed policy document."""
    from google.protobuf import json_format
    from openshell._proto import sandbox_pb2

    policy = sandbox_pb2.SandboxPolicy()
    try:
        json_format.ParseDict(policy_to_proto_dict(document), policy)
    except json_format.ParseError as exc:
        raise PolicyError(f"sandbox policy does not match the OpenShell schema: {exc}") from exc
    return policy
