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

"""Tests for loading the python_execute OpenShell policy.

The checked-in policy is loaded for real, so a typo in it fails here rather
than at the first CreateSandbox call. Conversion to the protobuf is checked
only as far as the field names and enum spellings, because the openshell
package is not installed for the native backend suite.
"""

import fnmatch
import re
from pathlib import Path

import pytest

from api.sandbox.openshell_policy import (
    DEFAULT_POLICY_PATH,
    PolicyError,
    load_policy_document,
    policy_to_proto_dict,
    render_policy_text,
)

ENV = {"OPENSHELL_BACKEND_HOST": "192.168.249.250", "OPENSHELL_BACKEND_PORT": "8000"}

AEVOM_CLIENT = Path(__file__).resolve().parents[3] / "services" / "pyexec" / "app" / "aevom_client.py"
_CALLBACK_URL = re.compile(r'f"\{self\._backend_url\}(/api/sandbox/[^"]*)"')


def client_callback_routes(source: str) -> set[tuple[str, str]]:
    """(method, path) of every backend callback in the in-sandbox aevom client.

    A request is a POST when ``method="POST"`` appears between its URL and the
    ``urlopen`` that sends it; f-string fields become a literal path segment.
    """
    routes = set()
    for match in _CALLBACK_URL.finditer(source):
        call = source[match.end() : source.find("urlopen(", match.end())]
        method = "POST" if 'method="POST"' in call else "GET"
        routes.add((method, re.sub(r"\{[^}]*\}", "x", match.group(1))))
    return routes


@pytest.fixture
def document():
    return load_policy_document(DEFAULT_POLICY_PATH, env=ENV)


class TestRendering:
    def test_placeholders_are_substituted(self):
        assert render_policy_text("host: ${OPENSHELL_BACKEND_HOST}", ENV) == "host: 192.168.249.250"

    def test_an_unset_placeholder_is_an_error(self):
        with pytest.raises(PolicyError, match="OPENSHELL_BACKEND_HOST"):
            render_policy_text("host: ${OPENSHELL_BACKEND_HOST}", {})

    def test_a_missing_file_is_an_error(self, tmp_path):
        with pytest.raises(PolicyError, match="cannot read"):
            load_policy_document(tmp_path / "absent.yaml", env=ENV)

    def test_the_version_is_required(self, tmp_path):
        path = tmp_path / "p.yaml"
        path.write_text("network_policies: {}\n")
        with pytest.raises(PolicyError, match="version"):
            load_policy_document(path, env=ENV)


class TestCheckedInPolicy:
    def test_the_backend_callback_is_pinned_to_the_backend(self, document):
        endpoint = document["network_policies"]["aevom_backend"]["endpoints"][0]
        assert (endpoint["host"], endpoint["port"]) == ("192.168.249.250", 8000)

    @pytest.mark.skipif(not AEVOM_CLIENT.exists(), reason="services/pyexec is not in this checkout")
    def test_every_callback_route_the_client_uses_is_allowed(self, document):
        endpoint = document["network_policies"]["aevom_backend"]["endpoints"][0]
        rules = [(r["allow"]["method"], r["allow"]["path"]) for r in endpoint["rules"]]
        routes = client_callback_routes(AEVOM_CLIENT.read_text())
        assert ("POST", "/api/sandbox/selection") in routes
        denied = sorted(
            (method, path)
            for method, path in routes
            if not any(method == m and fnmatch.fnmatchcase(path, p) for m, p in rules)
        )
        assert denied == []

    def test_route_discovery_reads_methods_and_fields(self):
        source = (
            'url = f"{self._backend_url}/api/sandbox/data/{reference}"\n'
            "req = urllib.request.Request(url)\nurllib.request.urlopen(req)\n"
            'url = f"{self._backend_url}/api/sandbox/store"\n'
            'req = urllib.request.Request(url, data=b"", method="POST")\nurllib.request.urlopen(req)\n'
        )
        assert client_callback_routes(source) == {("GET", "/api/sandbox/data/x"), ("POST", "/api/sandbox/store")}

    def test_external_egress_is_limited_to_pubchem_and_rcsb_search(self, document):
        hosts = {
            e["host"]
            for name, rule in document["network_policies"].items()
            if name != "aevom_backend"
            for e in rule["endpoints"]
        }
        assert hosts == {"pubchem.ncbi.nlm.nih.gov", "search.rcsb.org"}

    def test_the_application_code_is_read_only(self, document):
        fs = document["filesystem_policy"]
        assert "/app" in fs["read_only"]
        assert "/app" not in fs["read_write"]
        assert set(fs["read_write"]) <= {"/tmp", "/sandbox", "/dev/null"}

    def test_landlock_is_required(self, document):
        assert document["landlock"]["compatibility"] == "hard_requirement"


class TestProtoMapping:
    def test_field_names_follow_the_proto(self, document):
        proto = policy_to_proto_dict(document)
        assert "filesystem" in proto and "filesystem_policy" not in proto

    def test_a_rule_without_a_name_is_named_after_its_key(self):
        proto = policy_to_proto_dict({"version": 1, "network_policies": {"x": {"endpoints": []}}})
        assert proto["network_policies"]["x"]["name"] == "x"

    def test_enums_use_proto_names(self):
        proto = policy_to_proto_dict(
            {
                "version": 1,
                "network_policies": {
                    "x": {"endpoints": [{"host": "h", "port": 443, "enforcement": "enforce", "access": "read-only"}]}
                },
            }
        )
        endpoint = proto["network_policies"]["x"]["endpoints"][0]
        assert endpoint["enforcement"] == "NETWORK_ENFORCEMENT_MODE_ENFORCE"
        assert endpoint["access"] == "NETWORK_ACCESS_PRESET_READ_ONLY"

    def test_unknown_sections_are_rejected(self):
        with pytest.raises(PolicyError, match="inference"):
            policy_to_proto_dict({"version": 1, "inference": {}})
