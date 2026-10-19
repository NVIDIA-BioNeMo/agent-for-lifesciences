# Access and analyze Aevom results

Download the molecules, structures, and other artifacts produced in an Aevom
session, then analyze them with ordinary Python scripts. This guide replaces the
Jupyter notebook; no notebook server or kernel is required.

## 1. Choose the session containing your results

1. In Aevom, run a query that produces stored artifacts, such as fetching a
   molecule's 3D structure. Results appear in the storage strip.
2. Open the **top-right menu** (the **Open tool manager** button). In the
   **Tools** panel, find **Session ID** and click its copy icon.
3. Use the URL of that same Aevom deployment and the copied ID below.

Copy the ID before refreshing the page or sending a message with a different
model, since those actions create a new session. `POST /api/session/create`
also creates a new, empty session; it does not recover previous results.
Keep your session ID private: it scopes access to your stored artifacts.

The deployment must enable `AEVOM_FEAT_STORAGE_ACCESS_API=true` for downloads.
`GET /api/info` reports this as `features.storage_access_api`. If the Session ID
section is hidden or this flag is false, ask the deployment owner to enable
storage access. Metadata can still appear in the storage strip when downloads
are disabled.

## 2. Set up Python and the connection

From the repository root, create an environment and install the download client:

```bash
python -m venv .venv-results
source .venv-results/bin/activate
python -m pip install requests
export AEVOM_URL="http://localhost:8080"
export AEVOM_SESSION_ID="paste-your-existing-session-uuid"
```

In Windows PowerShell, use:

```powershell
python -m venv .venv-results
.\.venv-results\Scripts\Activate.ps1
python -m pip install requests
$env:AEVOM_URL = "http://localhost:8080"
$env:AEVOM_SESSION_ID = "paste-your-existing-session-uuid"
```

For a remote deployment, replace the URL with its HTTPS address, such as
`https://aevom.example.com`. Python requests keep certificate verification
enabled. If your deployment uses a private CA, configure its trusted CA bundle
with `REQUESTS_CA_BUNDLE` using the path supplied by your administrator.

These examples use Aevom's session-scoped API; any additional access requirements
at your deployment's ingress still apply.

## 3. List and download results

Save the following as `download_results.py` in your working directory. It lists
the session's recent artifacts, saves the JSON export with metadata, and writes
each exported artifact to a file under `aevom-results/`.

```python
import base64
import json
import os
import re
from pathlib import Path
from urllib.parse import quote

import requests

BASE_URL = os.environ.get("AEVOM_URL", "http://localhost:8080").rstrip("/")
SESSION_ID = os.environ.get("AEVOM_SESSION_ID", "").strip()
if not SESSION_ID:
    raise ValueError("Set AEVOM_SESSION_ID to the ID copied from your Aevom session.")

# Text types supported by the storage API. Other types are base64-encoded.
TEXT_TYPES = {
    "smiles", "safe", "sdf", "mol", "mol2", "pdb", "pdbqt", "mmcif", "cif", "pdbx",
    "annotated_cif", "sequence", "dna", "rna", "msa", "fasta", "selection",
    "design_spec", "json", "csv", "text", "skill_file",
}


def get_json(path, **params):
    response = requests.get(
        f"{BASE_URL}{path}",
        params={"session_id": SESSION_ID, **params},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def fetch_single_artifact(reference):
    # Use the bare reference from the listing, e.g. sdf_a1b2c3d4.
    return get_json(f"/api/storage/artifact/{quote(reference, safe='')}")


def main():
    listing = get_json("/api/storage/artifacts")["artifacts"]
    for item in listing:
        print(item["reference"], item["data_type"], item.get("description") or "")

    if not get_json("/api/info")["features"]["storage_access_api"]:
        raise RuntimeError("Downloads are disabled. Ask the deployment owner to enable storage access.")

    # Optional: set AEVOM_DATA_TYPE=sdf (or pdb, mmcif, etc.) before running.
    data_type = os.environ.get("AEVOM_DATA_TYPE", "").strip()
    filters = {"data_type": data_type} if data_type else {}
    export = get_json("/api/storage/export", **filters)
    output = Path("aevom-results")
    output.mkdir(exist_ok=True)
    (output / "export.json").write_text(json.dumps(export, indent=2), encoding="utf-8")

    for item in export["artifacts"]:
        kind = item["data_type"].strip().lower()
        reference = item["reference"]
        # Keep server-provided names inside the output directory.
        if not all(re.fullmatch(r"[A-Za-z0-9_]+", name) for name in (reference, kind)):
            raise ValueError("Artifact reference/type contains unsupported filename characters.")
        suffix = "npz" if kind == "array" else kind
        target = output / f"{reference}.{suffix}"
        if kind in TEXT_TYPES:
            target.write_text(item["content"], encoding="utf-8")
        else:
            target.write_bytes(base64.b64decode(item["content"], validate=True))
    print(f"Downloaded {export['count']} artifacts to {output.resolve()}")
    if export["count"] == 0:
        print("Check the session ID and filter, or generate results in Aevom first.")


if __name__ == "__main__":
    main()
```

Run it:

```bash
python download_results.py
```

The metadata listing contains at most the **newest 50** artifacts. An export
contains at most the **newest 100**, optionally filtered by `data_type`.
`count` is the number returned, not the total stored. These endpoints do not
expose pagination. For an older artifact whose reference you already know, use
`fetch_single_artifact(reference)` from the script. Export each relevant type
separately if the mixed-type export fills the 100-artifact limit; this still
does not retrieve more than 100 artifacts of a single type.

Each run replaces `aevom-results/export.json` and files with the same name.
Use a separate working directory for each session or export snapshot.

### Download a ZIP with curl

For SDF files, Bash users can also run this without Python:

```bash
curl --fail-with-body --get "$AEVOM_URL/api/storage/export" \
  --data-urlencode "session_id=$AEVOM_SESSION_ID" \
  --data-urlencode "data_type=sdf" \
  --data-urlencode "format=zip" \
  --output aevom-sdf.zip
```

Use `curl.exe` in PowerShell and its environment-variable/line-continuation
syntax. This ZIP example also works for other text types such as `pdb` or
`mmcif`. For binary PNG or array data, use the Python download script: the
current ZIP endpoint writes their base64 content as text, whereas the script
decodes it into usable files. Keep `export.json` when you need descriptions and
metadata as well as the raw files.

## 4. Analyze SDF molecules and export a property table

Install the optional analysis dependencies from the repository root:

```bash
python -m pip install -r examples/storage-access/requirements.txt
```

After downloading your results, save the following as `analyze_results.py` in
the same working directory. It reads the saved export, processes **every
molecule record** in each SDF, calculates descriptors, and produces:

- `aevom-results/aevom_properties.csv`: descriptors for all valid molecules.
- `aevom-results/aevom_lipinski_filter.csv`: molecules passing the four shown thresholds.
- `aevom-results/aevom_molecules.sdf`: combined valid molecules with their existing coordinates.
- `aevom-results/aevom_molecules.svg`: a 2D preview of up to 12 molecules.

```python
import io
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw

output = Path("aevom-results")
artifacts = json.loads((output / "export.json").read_text(encoding="utf-8"))["artifacts"]
molecules, rows = [], []

for artifact in artifacts:
    if artifact["data_type"] != "sdf":
        continue
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(artifact["content"].encode("utf-8")), removeHs=False,
    )
    for record_number, mol in enumerate(supplier, start=1):
        if mol is None:
            print(f"Skipping invalid SDF record {artifact['reference']}:{record_number}")
            continue
        name = f"{artifact['reference']}:{record_number}"
        mol.SetProp("_Name", name)
        mol.SetProp("Description", artifact.get("description") or "")
        mw, logp = Descriptors.MolWt(mol), Descriptors.MolLogP(mol)
        hbd, hba = Descriptors.NumHDonors(mol), Descriptors.NumHAcceptors(mol)
        rows.append({
            "Reference": artifact["reference"], "Record": record_number,
            "Description": mol.GetProp("Description"),
            "MW": round(mw, 2), "LogP": round(logp, 2), "HBD": hbd, "HBA": hba,
            "TPSA": round(Descriptors.TPSA(mol), 2),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "Rings": Descriptors.RingCount(mol), "SMILES": Chem.MolToSmiles(mol),
            "Lipinski": mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10,
        })
        molecules.append(mol)

if molecules:
    table = pd.DataFrame(rows)
    table.to_csv(output / "aevom_properties.csv", index=False)
    table.loc[table["Lipinski"]].to_csv(output / "aevom_lipinski_filter.csv", index=False)
    with Chem.SDWriter(str(output / "aevom_molecules.sdf")) as writer:
        for mol in molecules:
            writer.write(mol)
    # Make depiction copies so the exported molecules retain their coordinates.
    previews = [Chem.Mol(mol) for mol in molecules[:12]]
    for mol in previews:
        AllChem.Compute2DCoords(mol)
    svg = Draw.MolsToGridImage(
        previews, molsPerRow=3, subImgSize=(350, 300),
        legends=[mol.GetProp("_Name") for mol in previews], useSVG=True,
    )
    (output / "aevom_molecules.svg").write_text(svg, encoding="utf-8")
    print(table.to_string(index=False))
else:
    print("No valid SDF molecules in this export. No analysis files were written.")
```

```bash
python analyze_results.py
```

The Lipinski column is a descriptor filter; it does not establish binding
affinity or suitability as a drug. The original downloaded SDF files remain
available alongside the combined SDF and 2D preview. If a later export is empty,
the script leaves any earlier analysis files untouched; use a new working
directory when comparing separate exports.

## 5. Inspect PDB structures

The download script saves PDB artifacts as `<reference>.pdb`. Save this as
`inspect_pdb.py` alongside the other scripts to summarize PDB records:

```python
from pathlib import Path

paths = sorted(Path("aevom-results").glob("*.pdb"))
for path in paths:
    lines = path.read_text(encoding="utf-8").splitlines()
    atoms = sum(line.startswith("ATOM  ") for line in lines)
    heteroatoms = sum(line.startswith("HETATM") for line in lines)
    print(f"{path.name}: {atoms} ATOM records, {heteroatoms} HETATM records")
if not paths:
    print("No PDB files in this export; check for mmCIF files or export data_type=pdb.")
```

```bash
python inspect_pdb.py
```

These are line counts, not unique atom counts: alternate locations and multiple
models can repeat atoms. mmCIF structures are saved as `.mmcif` or `.cif` and
need an mmCIF-aware viewer or parser.

## API reference and troubleshooting

All three endpoints require `session_id` as a query parameter.

| Endpoint | Result |
| --- | --- |
| `GET /api/storage/artifacts` | Recent artifact metadata, up to 50 entries. |
| `GET /api/storage/artifact/{reference}` | One artifact's `reference`, `data_type`, `content`, and `metadata`. |
| `GET /api/storage/export` | Up to 100 artifacts with content, descriptions, and metadata; accepts `data_type` and `format=zip`. |

JSON responses represent text content as a UTF-8 string and binary content as
base64. A JSON artifact's `content` is itself a string: use
`json.loads(artifact["content"])` when you want its parsed value.

| Symptom | What to check |
| --- | --- |
| Empty list/export | Use the original session ID and deployment URL; confirm artifacts were generated and the type filter matches. |
| HTTP 403 | Supply `session_id`; a single-artifact request must use the session that owns it. Check any ingress access requirements. |
| HTTP 404 for content/export | Check `features.storage_access_api` in `/api/info`; for a single artifact, also check its reference still exists. |
| Certificate verification error | Configure the deployment's trusted CA bundle with `REQUESTS_CA_BUNDLE`. |
| Connection refused or timeout | Confirm the deployment URL is reachable from the computer running Python. |
| Missing RDKit/pandas import | Install the optional requirements in the same Python environment used to run the script. |

Artifacts are session working data, not an archive. Export important results
before a backend restart, which recreates ephemeral storage tables, or before
the artifacts expire.
