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

"""Dependency and license inventory of the conformer worker environment.

Run by build_env.sh with the worker interpreter after the runtime prefix is
final. Writes licenses.json and fails if a package that belongs only in the
Vina parent (Meeko, Open Babel, PLIP and similar) is present.

Usage: python -I inventory.py <prefix> <output.json>
"""

import json
import re
import sys
from importlib import metadata
from pathlib import Path

FORBIDDEN = {"meeko", "openbabel", "openbabel-wheel", "plip", "pdbfixer", "openmm", "vina", "pymol", "pymol-open-source"}
COPYLEFT = re.compile(r"\b(A?GPL|LGPL|MPL|EPL|CDDL|EUPL|OSL|copyleft)", re.IGNORECASE)


def _pip_license(dist: metadata.Distribution) -> str:
    meta = dist.metadata
    expression = meta.get("License-Expression")
    if expression:
        return expression
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    declared = (meta.get("License") or "").strip()
    if declared and len(declared) < 120:
        return "; ".join([declared, *classifiers]) if classifiers else declared
    return "; ".join(classifiers) or "UNKNOWN"


def main() -> int:
    prefix, output = Path(sys.argv[1]), Path(sys.argv[2])
    entries = []
    conda_names = set()
    for record in sorted((prefix / "conda-meta").glob("*.json")):
        data = json.loads(record.read_text())
        conda_names.add(data["name"].lower())
        entries.append(
            {
                "source": "conda-forge",
                "name": data["name"],
                "version": data["version"],
                "build": data.get("build", ""),
                "license": data.get("license") or "UNKNOWN",
            }
        )
    for dist in sorted(metadata.distributions(), key=lambda d: d.metadata["Name"].lower()):
        installer = (dist.read_text("INSTALLER") or "").strip()
        name = dist.metadata["Name"]
        if installer != "pip" and name.lower() in conda_names:
            continue
        entries.append(
            {
                "source": "pip" if installer == "pip" else f"python-dist ({installer or 'unknown installer'})",
                "name": name,
                "version": dist.version,
                "build": "",
                "license": _pip_license(dist),
            }
        )
    for entry in entries:
        entry["copyleft_flag"] = bool(COPYLEFT.search(entry["license"]))

    forbidden = sorted({e["name"] for e in entries if e["name"].lower() in FORBIDDEN})
    output.write_text(json.dumps({"prefix": str(prefix), "packages": entries}, indent=2) + "\n")

    flagged = [e for e in entries if e["copyleft_flag"]]
    print(f"inventory: {len(entries)} packages, {len(flagged)} flagged for copyleft review:")
    for e in flagged:
        print(f"  {e['source']:<12} {e['name']}=={e['version']}  [{e['license']}]")
    if forbidden:
        print(f"inventory: FORBIDDEN packages present: {forbidden}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
