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

"""Python execution tool for molecular analysis.

This tool enables the LLM agent to execute Python code in a sandboxed
environment with access to scientific libraries (numpy, scipy, rdkit)
and selection data from structure queries.

Supports the Aevom data store integration, allowing code to load stored
molecular data (PDB, SDF, selections) via the pre-loaded `aevom` client.
"""

import json
import logging
import os
import re
from typing import Any

import httpx
from opentelemetry import trace

from api.sandbox import get_session_manager
from api.sandbox.openshell_manager import (
    SandboxUnavailableError,
    get_openshell_manager,
    openshell_enabled,
)
from api.selections import get_latest_selection, get_latest_nonempty_selection
from .base import ContainerTool, DataIOCategory, ToolResult

logger = logging.getLogger(__name__)

# How the OpenShell sandbox reports a blocked operation. A direct socket
# connect fails with a bare EACCES (the workload has no route except the
# policy proxy); the proxy answers a denied request or CONNECT with a 403
# naming the reason; Landlock fails a write with EACCES and the path.
_NETWORK_DENIAL = re.compile(
    r"\[Errno 13\] Permission denied(?!:)|Tunnel connection failed: 403"
    r"|policy_denied|ssrf_denied|X-OpenShell-Policy"
)
_FILESYSTEM_DENIAL = re.compile(r"\[Errno 13\] Permission denied: '")

SANDBOX_NETWORK_NOTE = (
    "\n\n[The sandbox policy blocked a network request. python_execute can "
    "reach the Aevom backend (through the aevom client), PubChem and RCSB "
    "search, and nothing else. Use the dedicated tools for other services.]"
)
SANDBOX_FILESYSTEM_NOTE = (
    "\n\n[The sandbox policy blocked a filesystem write. Only /tmp and the "
    "working directory are writable; use aevom.store() to keep results.]"
)


def sandbox_denial_note(text: str) -> str:
    """Explain an OpenShell policy denial found in execution output, if any."""
    notes = ""
    if _NETWORK_DENIAL.search(text):
        notes += SANDBOX_NETWORK_NOTE
    if _FILESYSTEM_DENIAL.search(text):
        notes += SANDBOX_FILESYSTEM_NOTE
    return notes

# Pattern to detect data references in code (e.g., pdb_abc12345, annotated_cif_919e0b2e)
# Reference format: {type}_{hex_id} where hex_id is 8 characters and the type
# may itself contain underscores
DATA_REF_PATTERN = re.compile(r"\b([a-z]+(?:_[a-z]+)*_[a-f0-9]{8})\b")

# Execution budgets, in seconds. The default covers descriptor panels and
# similarity matrices over a few hundred molecules; the ceiling leaves room for
# the expensive-but-bounded work the agent does reach for, such as an MCS over a
# congeneric series or a clustering of an enumerated library, which routinely
# ran past the old 10s default. Training rollouts get longer still because a
# timeout there is a wasted episode rather than a retryable turn. All three must
# stay at or below the pyexec service's own ceiling, which rejects anything
# higher outright.
DEFAULT_TIMEOUT_SECONDS = 30
MAX_TIMEOUT_SECONDS = 120
TRAINING_MAX_TIMEOUT_SECONDS = 180

# Upper bound on how many of a session's artifacts are auto-granted to the
# sandbox (newest first). Sessions are normally far smaller than this; the cap
# only exists so a pathological session can't build an unbounded allow-list.
MAX_SESSION_REFS = 500

# Patterns that indicate a line of sandbox output is an error message.
# Used by _output_indicates_failure() to detect cases where the sandbox
# reported success (no uncaught exception) but the code's try/except blocks
# caught and printed errors -- meaning the code did not achieve its goal.
_ERROR_LINE_PATTERNS = [
    re.compile(r"^error[:\s]", re.IGNORECASE),
    re.compile(r"^failed to ", re.IGNORECASE),
    re.compile(r"connection refused", re.IGNORECASE),
    re.compile(r"not defined", re.IGNORECASE),
    re.compile(r"has no attribute", re.IGNORECASE),
    re.compile(r"not found", re.IGNORECASE),
    re.compile(r"^traceback ", re.IGNORECASE),
    re.compile(r"not accessible in this session", re.IGNORECASE),
    re.compile(r"did not match C\+\+ signature", re.IGNORECASE),
]


def _output_indicates_failure(output: str) -> bool:
    """Check if sandbox output consists entirely of error messages.

    When the LLM writes code with try/except that catches errors and prints
    them, the sandbox reports success=True (no uncaught exception) but the
    output is all error messages. This heuristic detects that pattern.

    Only triggers when >=50% of *top-level* lines match error patterns AND
    there is no substantive non-error output, so legitimate code that prints
    some diagnostic errors alongside real results is not affected.

    Indented lines (e.g. traceback continuations, function signatures) are
    treated as continuations of the preceding line and excluded from the
    ratio calculation.
    """
    raw_lines = output.strip().splitlines()
    if not raw_lines:
        return False

    # Separate top-level lines from indented continuations.
    top_level = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue
        if line[0] in (" ", "\t"):
            continue  # indented continuation — skip for ratio
        top_level.append(stripped)

    if not top_level:
        return False

    error_count = sum(
        1 for line in top_level
        if any(p.search(line) for p in _ERROR_LINE_PATTERNS)
    )
    return error_count > 0 and error_count >= len(top_level) * 0.5


# A molecule written out as a string literal rather than produced by a program.
# Deliberately crude: the backend image has no RDKit, so this recognises the
# shape of a structure, not a valid one -- which is the point, since the ones
# that matter most are the invalid ones.
_STRUCTURE_LITERAL = re.compile(r"""['"]([^'"\n]{12,})['"]""")

# Outside square brackets SMILES admits only the organic subset, so a literal
# containing any other letter is a path, a reference or a column name rather
# than a structure. H is admitted even though a bare H is never valid SMILES:
# writing one is the commonest way these strings are broken, and a detector
# that discards the broken ones would be blind to what it exists to find.
_ORGANIC_SUBSET_CHARS = frozenset("BCNOPSFIlrbcnopsH")
_STRUCTURE_PUNCTUATION = frozenset("0123456789()=#-+\\/@.%*$:~[]")

# At or above this many structures in one block, the code is writing out a
# series rather than naming a starting point, and the response says so. Five,
# measured: across 2113 sandbox calls in 90 benchmark runs it fires on 17% of
# them while covering 77% of every structure written out by hand. Three would
# cover 84% but fire on 22%, which is close enough to always that it would stop
# being read.
HANDWRITTEN_STRUCTURES = 5


def _handwritten_structures(code: str) -> int:
    """How many string literals in *code* look like molecules."""
    found = 0
    for literal in _STRUCTURE_LITERAL.findall(code):
        # A ring closure or a branch. Without one of these it is a word.
        if not any(c.isdigit() or c == "(" for c in literal):
            continue
        depth = 0
        for character in literal:
            if character == "[":
                depth += 1
            elif character == "]":
                depth = max(0, depth - 1)
            elif depth == 0 and (
                character not in _ORGANIC_SUBSET_CHARS
                and character not in _STRUCTURE_PUNCTUATION
            ):
                break
        else:
            found += 1
    return found


class PythonExecuteTool(ContainerTool):
    """Execute Python code to analyze molecular selection data.

    This tool runs Python code in a secure sandbox with:
    - numpy (as np)
    - scipy (via import)
    - rdkit (via import)
    - math (pre-loaded)
    - Selection data injected as 'selection' variable

    Use this for computing distances, angles, properties, and
    other analysis that requires custom calculations.
    """

    @property
    def name(self) -> str:
        return "python_execute"

    @property
    def summary_template(self) -> str:
        return "Executing Python analysis code..."

    @property
    def service_url(self) -> str:
        """URL of the PyExec service container."""
        return os.environ.get("PYEXEC_SERVICE_URL", "http://pyexec:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        """This tool retrieves selection data from storage."""
        return DataIOCategory.INPUT_RETRIEVES

    @property
    def is_async(self) -> bool:
        """Code execution has a timeout but is generally fast."""
        return False

    @property
    def max_response_tokens(self) -> int:
        """Room for a table of candidates, but not for a coordinate dump.

        Sandbox output is arbitrary by nature -- a single ``print`` of an array
        can run to hundreds of kilobytes -- so this stays below
        DEFAULT_MAX_RESPONSE_TOKENS, the rule for tools whose output is not
        bounded by their input. It is well above the previous 800, which was
        smaller than a legitimate answer for the analog-enumeration work this
        tool does: fifty molecules with their descriptors was being truncated
        mid-table, and the model does not reliably follow the pointer to the
        stored overflow.

        For a panel too big to fit, prefer ``molecule_gates`` -- it writes a CSV
        artifact instead of returning the rows inline.
        """
        return 1800

    def get_schema(self) -> dict[str, Any]:
        """Get OpenAI-compatible function schema."""
        return {
            "type": "function",
            "function": {
                "name": "python_execute",
                "description": """Execute Python code in a secure sandbox for computation and analysis.

**Do NOT use python_execute for these — use the dedicated tools instead:**
- Searching PubChem → search_pubchem
- Getting molecular properties → get_pubchem_properties
- Downloading SDF structures → download_pubchem_sdf
- Looking up CCD codes → lookup_ccd_code
- Visualizing SDF molecules → visualize_sdf

**Use this tool when you need to:**
- Compute distances between atoms
- Calculate angles or dihedral angles
- Analyze geometric relationships in a selection
- Perform statistical analysis on atom properties
- Run custom calculations on coordinates
- Load and analyze stored molecular data (PDB, SDF, sequences)
- RDKit fragment analysis or custom cheminformatics

**Available in the execution environment:**
- `numpy` (also as `np`) - array operations, linear algebra
- `math` - mathematical functions
- `scipy` (via import) - scientific computing
- `rdkit` (via import) - molecular analysis
- `Bio` (BioPython, via import) - PDB parsing, sequences, neighbour search
- `biotite` (via import) - fast structure parsing, superimposition, SASA
- `openbabel` (via import) - format conversion, protonation, partial charges
- `sklearn` (via import) - clustering, PCA, regression over descriptors
- `safe` (via import) - SAFE fragment notation (`safe.encode` / `safe.decode`)
- `aevom` - client for loading/storing data, PubChem search, and visualization

Each of these has a skill with worked examples; call `load_skill` when you need
more than a one-liner.

**NOT AVAILABLE (no network access):**
- requests, urllib.request, http.client, socket, os, subprocess, sys
- sklearn.datasets (downloads and reads files; everything else in sklearn works)
- biotite.database (fetches from RCSB/UniProt) and biotite.application (runs
  external programs); the rest of biotite works
- PLIP is not installed here — use the `plip_interactions` tool

**There are no files in the sandbox.** `open()` is rejected, so every library
must be driven from strings: load with `aevom.load_pdb(ref)`, then hand the text
to `openbabel`'s `ReadString`, or wrap it in `io.StringIO` for BioPython and
biotite. Path-based readers have nothing to point at.

**Parsing structures with biotite — pass `model=1`.**
`PDBFile.get_structure()` (and the pdbx/mmCIF equivalent) returns an
`AtomArrayStack` — one entry per model — *even for a single-model file*, so an
atom-length boolean mask indexes the wrong axis and raises
`IndexError: boolean index did not match indexed array along axis 0; size of
axis is 1`. Ask for a flat `AtomArray` with `model=1`, and request the B-factor
column (this is where `openfold3`/`boltz2` put per-residue pLDDT) explicitly —
it is dropped otherwise:
```python
arr = pdb.PDBFile.read(io.StringIO(aevom.load_pdb(ref))).get_structure(
    model=1, extra_fields=["b_factor"])   # AtomArray, with arr.b_factor
```

**In-place assignment works only on `list`, `dict` and `set`.** `arr[0] = x`,
`arr[:] = x` and `obj.attr = x` all raise `Write operations are restricted`,
including on numpy arrays. Build new values instead — `arr + shift`,
`np.where(...)`, `np.concatenate(...)` — or accumulate into a list and call
`np.array()` at the end. Augmented assignment to a name you own (`a += 1`) is
fine, as is any method call that mutates internally (`mol.SetProp(...)`).

**Augmented assignment on an *item or slice* is rejected for every container**
— `counts[k] += 1`, `arr[i] += x`, `lst[i] += 1` all fail (a compile-time
restriction), even though plain `d[k] = v` is allowed. To tally into a dict use
`collections.Counter`, or `counts[k] = counts.get(k, 0) + 1`; to collect unique
things use a `set` comprehension; otherwise rebuild the value rather than
mutating it in place.

**Cheminformatics beyond the basics** — all of RDKit is importable, including
the parts that are awkward to find:
```python
from rdkit.Chem.Scaffolds import MurckoScaffold   # scaffold extraction
from rdkit.Chem import rdFMCS                     # maximum common substructure
from rdkit.Chem import FilterCatalog              # PAINS / Brenk / NIH alerts
from rdkit.Chem import ChemicalFeatures, RDConfig # pharmacophore features
from rdkit.Chem import BRICS                      # retrosynthetic fragments

# Synthetic accessibility needs a helper: its scorer lives outside the RDKit
# package and reaching it requires sys, which the sandbox does not allow.
sa = aevom.sa_score("CC(=O)Oc1ccccc1C(=O)O")  # 1 (easy) to 10 (hard)
```

**Analogues of a lead, with `aevom.enumerate_analogs`:**
```python
# Every single-edit analogue: swap one substituent, or decorate an aromatic CH.
# `similarity_min` is enforced per product, so the bound is a guarantee rather
# than a hope -- which is what makes this the right tool when a task says the
# answer must stay within some Tanimoto of the molecule you started from.
analogs = aevom.enumerate_analogs(lead_smiles, similarity_min=0.7)
print(len(analogs))            # a few hundred from one lead, typically

# Chemistry of your own in mind? Pass the substituents. Do not write out the
# molecules: splicing a group into a ring by hand is where SMILES breaks, and
# broken ones vanish into `if mol is None: continue` without a count. RDKit
# does the attaching here, so an unparseable product is not reachable.
analogs = aevom.enumerate_analogs(
    lead_smiles, r_groups=["C#C", "SC(F)(F)F", "CN(C)S(=O)(=O)", "N1CCOCC1"],
    similarity_min=0.7,
)

# Filter locally before spending an oracle call on anything. Every
# drug-likeness bound is RDKit, so hundreds of molecules cost a second here and
# nothing at all in tool calls.
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
keep = [s for s in analogs
        if Descriptors.MolWt(Chem.MolFromSmiles(s)) < 600
        and -1 <= Crippen.MolLogP(Chem.MolFromSmiles(s)) <= 5]
```
Enumerate and filter here, then score the survivors in ONE batched
`admet_predict` call, which takes up to 256 molecules for the price of one.
**Aim for a couple of hundred survivors, not tens.** Hand-writing a handful of
analogues one at a time is the common way a property-optimisation task is lost:
the molecule that clears the threshold is usually in the enumeration, and
usually was not one of the six somebody thought of. Hit rates in analogue
screening run at a few percent, so twenty candidates is a coin flip on whether
a satisfying molecule is even present. `r_groups` defaults to forty common substituents and can
be replaced; `decorate=False` keeps the heavy-atom count fixed. It does not
edit scaffolds -- no ring-heteroatom walks, no positional isomers -- so those
still have to be written by hand.
For a full descriptor-and-alerts panel over a list of molecules, prefer the
`molecule_gates` tool — it applies drug-likeness thresholds and reports which
ones each molecule fails. Use Python here when you need the intermediate
values, a custom threshold, or a calculation that tool does not do.

**Loading and storing data with `aevom`:**
```python
# Every artifact in the current session is already loadable -- you do NOT need
# to declare refs anywhere first. To act on all of them, enumerate the storage:
for ref in sorted(aevom.allowed_refs):      # every stored reference, e.g. 'sdf_..'
    if ref.startswith("sdf_"):
        mol = aevom.load_rdkit_mol(ref)     # loads lazily, only when accessed
        print(ref, aevom.canonical_smiles(mol))

# Load a PDB structure
pdb_data = aevom.load("pdb_abc12345")
coords = aevom.load_coords("pdb_abc12345")

# Load SDF as RDKit molecule
mol = aevom.load_rdkit_mol("sdf_xyz789ab")
from rdkit.Chem import Descriptors
mw = Descriptors.MolWt(mol)

# Store modified molecule
sdf = aevom.mol_to_sdf(mol)
new_ref = aevom.store_sdf(sdf, {"name": "modified"})

# Get canonical SMILES from molecule
smiles = aevom.canonical_smiles(mol)

# Store a smiles string
new_ref = aevom.store_smiles(smiles, {"name": "modified"})

# SAFE fragment notation, for generative models like GenMol and SAFE-GPT.
# (For a plain conversion prefer the smiles_to_safe / safe_to_smiles tools;
# reach for Python when you need to inspect or edit the fragments.)
safe_str = aevom.safe_from_smiles("CC(Cc1ccc(cc1)C(C(=O)O)C)C")
fragments = safe_str.split(".")
masked = aevom.scaffold_to_safe("C1CC(=O)NC1", 10, 15)  # scaffold + [*{10-15}]
safe_ref = aevom.store_safe(masked, description="Lactam scaffold, masked")
smiles = aevom.smiles_from_safe(aevom.load_safe("safe_abc12345"))

# Load a numeric array (e.g. a PAE / pLDDT matrix returned by a skill tool
# as an 'array_*' reference) and analyze it here rather than reading it as text
# A SMILES is not legible: OC(F)(F)c1ccccc1 and C(F)(F)Oc1ccccc1 differ only in
# which atom carries the ring, and one of them is a gem-fluorohydrin. Read the
# graph rather than the string when a structure matters.
d = aevom.describe_molecule(smiles)
print(d["formula"], d["groups"], d["rings"], d["unstable"])

rows = aevom.load_table("csv_abc12345")   # a stored results table, numbers parsed
keep = [r for r in rows if r["CYP3A4_Veith"] < 0.5]   # keys are the file's headers
# There is no `csv` or `pandas` here; load_table is how a stored table is read.

pae = aevom.load_array("array_abc12345")
print(pae.shape, pae.mean(), pae.max())
sub_ref = aevom.store_array(pae[:120, 120:], description="interface PAE")
```

**Pocket interaction-energy maps with `aevom.probe_grid`:**
```python
# Where would a hydrophobe, a donor and an acceptor most want to sit?
# Same computation as the `probe_grid` tool, but you get the array instead of
# a summary -- use this when you want to cluster the wells, intersect the
# families, or score positions of your own.
grid = aevom.probe_grid("pdb_abc12345", center=[12.4, -3.1, 25.6])
phobe, donor, acceptor = grid          # default probes C_H, N_D, O_A

out = aevom.probe_grid("pdb_abc12345", center=[12.4, -3.1, 25.6],
                       size=[24, 24, 24], return_details=True)
origin, spacing = out["origin"], out["spacing"]
# grid point (i, j, k) sits at origin + spacing * (i, j, k)
```
Use the `probe_grid` tool instead when a ranked list of sites is all you need.

**Filtering linked fragments with `aevom.constrained_pose`:**
```python
# Can each candidate hold the fragments where they were posed? Same check the
# `fragment_link` tool runs, over as many candidates as you like -- the rejects
# never reach the transcript.
rows = aevom.constrained_pose(["sdf_frag0001", "sdf_frag0002"], candidates,
                              pocket_ref="pdb_abc12345")
kept = sorted((r for r in rows if r["verified"]), key=lambda r: r["strain"])
print(len(kept), "of", len(rows), "hold both poses")
for r in kept[:5]:
    print(r["smiles"], round(r["strain"], 1), r["min_protein_distance"])
```
Read strain and protein contact, not RMSD: the fragment atoms were pinned, so
the fit to the given poses is near zero by construction.

**Visualizing molecules in MolStar:**
```python
# Visualize RDKit molecule
mol = aevom.load_rdkit_mol("sdf_abc12345")
aevom.visualize(mol, label="My Molecule")

# Visualize SDF content directly
aevom.visualize(sdf_content, label="Aspirin")

# Create and visualize from SMILES
mol = aevom.mol_from_smiles("CCO")
mol = aevom.embed_mol_3d(mol)  # Generate 3D coords
aevom.visualize(mol, label="Ethanol")
```

**Searching PubChem:**
```python
# Search for a molecule by name
result = aevom.search_pubchem("Ibuprofen")
if result:
    print(f"CID: {result['cid']}")
    print(f"SMILES: {result['isomeric_smiles']}")
    
    # Get molecular properties
    props = aevom.get_pubchem_properties(result['cid'])
    print(f"MW: {props['molecular_weight']}")
    print(f"H-bond donors: {props['h_bond_donor_count']}")
    
    # Download 3D structure - returns a reference directly!
    sdf_ref = aevom.download_pubchem_sdf(result['cid'])
    print(f"Stored as: {sdf_ref}")  # e.g., "sdf_abc12345"
    
    # The reference can be used directly with tools like vina_dock
    # Or load for visualization/analysis:
    mol = aevom.load_rdkit_mol(sdf_ref)
    aevom.visualize(mol, label="Ibuprofen")
```

**Looking up CCD codes for OpenFold3:**
```python
# Look up PDB Chemical Component Dictionary code for a drug
ccd_code = aevom.lookup_ccd_code("imatinib")
print(f"CCD code: {ccd_code}")  # Output: STI

# This is essential for openfold3 protein-ligand predictions
# Use the returned code in: {"type": "ligand", "ccd_codes": ccd_code}
```

**When selection_ref is provided**, the `selection` variable contains:
```python
selection = {
    "query_id": "...",
    "molql": "...",  # The MolQL query that created this selection
    "atoms": [
        {
            "name": "CA",       # Atom name (str)
            "element": "C",     # Element symbol (str)
            "id": 123,          # Atom ID (int)
            "coords": [x, y, z], # 3D coordinates — LIST, not tuple!
            "residue": {
                "name": "ALA",      # Residue name (str)
                "seqNumber": 45,    # Sequence number (int), LABEL numbering (label_seq_id)
                "chainId": "A",     # Chain ID (str), LABEL numbering (label_asym_id)
                "authSeqNumber": 61,  # Author numbering: what the viewer shows and MolQL atom.resno matches
                "authChainId": "A",   # Author chain ID. Absent on selections stored by older versions
                "entityType": "polymer"  # or "non-polymer"
            }
        },
        ...
    ],
    "atom_count": 54,
    "residue_count": 12,
    "summary": "Selected 54 atoms in 12 residues",
    "provenance": "user",  # "user" = residues the person picked by hand; "agent" = made by a tool
    "name": "epitope"       # Present only when the selection was named
}
```

`selection` (and `aevom.load_selection(ref)`) is a dict with residue helpers, so you do not have to group atoms yourself:
```python
for r in selection.residues():   # each residue once, in order
    # r["chain"], r["number"]            author numbering (what the viewer shows)
    # r["label_chain"], r["label_seq"]   label numbering (what design tools call res_id)
    # r["name"], r["ins_code"], r["entity_type"], r["author_known"]
    # r["atoms"]    names of the atoms this selection holds in the residue
    # r["partial"]  True when that is only some of a standard residue's atoms
    ...
numbers = selection.residue_numbers("A")   # author residue numbers in chain A
```
To turn residues you computed into a selection other tools can use (shown to the user, used as a design hotspot), store them:
```python
ref = aevom.store_selection("pdb_abc12345", [56, "100-120"], chain="A", name="epitope")
```
Numbers are author numbering unless you pass `numbering="label"`.

A selection can hold chosen atoms rather than whole residues (the user right-clicks atoms in the viewer; select_residues takes `atoms`). Check `r["partial"]` before assuming a residue is complete. To store atoms, name them per residue or for all of them, e.g. `aevom.store_selection(ref, ["56:OH", "115:CB,CG"], chain="A")` or `aevom.store_selection(ref, [56, 115], chain="A", atoms="sidechain")` (also `backbone`, or a list of names).

**WARNING**: `coords` is a list `[x, y, z]` and CANNOT be used as a dict key or in sets (unhashable). Use string keys for grouping, e.g. `f"{atom['residue']['name']}{atom['residue']['seqNumber']}"`. Residue fields (`name`, `seqNumber`, `chainId`) are scalars and safe to use as keys.

**To return a result**, assign it to a variable named `result`:
```python
import numpy as np
coords = np.array([a['coords'] for a in selection['atoms']])
result = {"center_of_mass": coords.mean(axis=0).tolist()}
```

**Output is captured via print statements.** The final output includes both
printed text and the `result` variable if set.

**Examples:**

1. Distance between first two atoms:
```python
import numpy as np
a1, a2 = selection['atoms'][:2]
dist = np.linalg.norm(np.array(a1['coords']) - np.array(a2['coords']))
print(f"Distance: {dist:.2f} Angstroms")
```

2. Center of mass:
```python
import numpy as np
coords = np.array([a['coords'] for a in selection['atoms']])
com = coords.mean(axis=0)
print(f"Center of mass: ({com[0]:.2f}, {com[1]:.2f}, {com[2]:.2f})")
```

3. Load and analyze a predicted structure:
```python
# Load PDB from structure prediction
coords = aevom.load_coords("pdb_abc12345")
center = coords.mean(axis=0)
print(f"Center of mass: {center}")
print(f"Structure has {len(coords)} atoms")
```""",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute. Use print() for output and assign to 'result' for structured return values.",
                        },
                        "selection_ref": {
                            "type": "string",
                            "description": "Reference to selection data (e.g., 'selection_abc12345') from [SELECTION STATE]. ALWAYS pass this when your code uses the `selection` variable — check [SELECTION STATE] for available references. Auto-injection from the latest selection is a fallback only and may pick the wrong selection.",
                        },
                        "data_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional and usually unnecessary. The sandbox is automatically granted access to EVERY artifact stored in the current session, so `aevom.load('pdb_abc12345')` works for any stored ref without listing it here. You can also enumerate your storage from inside the code via `aevom.allowed_refs` (a set of every stored reference). Only pass this to force access to a specific ref.",
                        },
                        "timeout_seconds": {
                            "type": "integer",
                            "description": (
                                f"Maximum execution time in seconds "
                                f"(1-{MAX_TIMEOUT_SECONDS}, default: "
                                f"{DEFAULT_TIMEOUT_SECONDS}). Raise it for an "
                                f"MCS over many molecules or clustering a large "
                                f"library."
                            ),
                        },
                    },
                    "required": ["code"],
                },
            },
        }

    def _detect_refs_in_code(self, code: str) -> set[str]:
        """Detect data references in Python code.

        Scans the code for patterns like 'pdb_abc12345', 'sdf_xyz789ab', etc.

        Args:
            code: Python code string.

        Returns:
            Set of detected reference IDs.
        """
        matches = DATA_REF_PATTERN.findall(code)
        return set(matches)

    async def _validate_refs(self, refs: set[str]) -> set[str]:
        """Validate that references exist in storage.

        Args:
            refs: Set of reference IDs to validate.

        Returns:
            Set of valid (existing) references.
        """
        if not self._storage or not refs:
            return set()

        valid_refs = set()
        for ref in refs:
            try:
                if await self._storage.exists(ref):
                    valid_refs.add(ref)
                else:
                    logger.warning(f"Reference not found in storage: {ref}")
            except Exception as e:
                logger.warning(f"Error checking reference {ref}: {e}")

        return valid_refs

    async def _list_session_refs(self, session_id: str | None) -> set[str]:
        """Return every artifact reference stored under ``session_id``.

        Used to scope the sandbox to the whole session's storage so
        ``aevom.load()`` works for any stored ref and ``aevom.allowed_refs``
        enumerates it. Returns an empty set (never raises) when storage is
        unavailable, the session is unknown, or the listing fails, so a listing
        problem degrades to the old narrow-scope behaviour rather than breaking
        execution. Capped at ``MAX_SESSION_REFS`` newest artifacts.
        """
        if not self._storage or not session_id:
            return set()
        try:
            # SessionScopedStorage filters by its own session and takes no
            # session_id argument; the guard above keeps an unscoped service out.
            artifacts = await self._storage.list_artifacts(limit=MAX_SESSION_REFS)
        except Exception as e:
            logger.warning(f"Failed to list session refs for sandbox scope: {e}")
            return set()
        refs = {a["reference"] for a in artifacts if a.get("reference")}
        if len(refs) >= MAX_SESSION_REFS:
            logger.warning(
                "Session %s has >= %d artifacts; sandbox scope capped to the "
                "newest %d. Older refs remain loadable only if named explicitly.",
                session_id, MAX_SESSION_REFS, MAX_SESSION_REFS,
            )
        return refs

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute Python code in the sandbox.

        Args:
            arguments: Dict containing 'code', optional 'selection_ref',
                      'data_refs', and 'timeout_seconds'.

        Returns:
            ToolResult with execution output or error.
        """
        code = arguments.get("code", "")
        selection_ref = arguments.get("selection_ref")
        data_refs = arguments.get("data_refs", [])
        timeout_seconds = arguments.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)

        if not code or not code.strip():
            return ToolResult.error("Python code is required")

        is_training = os.environ.get("AEVOM_TRAINING_RUN_ID") is not None
        max_timeout = TRAINING_MAX_TIMEOUT_SECONDS if is_training else MAX_TIMEOUT_SECONDS
        timeout_seconds = max(1, min(max_timeout, int(timeout_seconds)))

        # Build execution context
        context = {}

        # Auto-inject latest non-empty selection if code uses 'selection' but no ref provided
        # We specifically use get_latest_nonempty_selection() to avoid injecting empty
        # selections that would cause errors like "IndexError: invalid index to scalar"
        # Determine session_id for selection lookup
        _sel_session_id = getattr(self._storage, "session_id", None) or "default"
        from api.redis import get_redis as _get_redis
        _redis = await _get_redis()
        if not selection_ref and "selection" in code:
            latest = await get_latest_nonempty_selection(_redis, _sel_session_id)
            if latest:
                selection_ref = latest.reference
                logger.info(
                    f"Auto-injecting latest non-empty selection: {selection_ref} "
                    f"({latest.atom_count} atoms, {latest.residue_count} residues)"
                )
            else:
                # Check if there was a selection but it was empty
                any_selection = await get_latest_selection(_redis, _sel_session_id)
                if any_selection:
                    logger.warning(
                        f"Latest selection {any_selection.reference} is empty "
                        f"(0 atoms), not auto-injecting. Code may fail if it "
                        f"expects 'selection' variable."
                    )

        # Retrieve selection data if reference provided (or auto-injected)
        if selection_ref:
            if not self._storage:
                return ToolResult.error(
                    "Storage service not available - cannot retrieve selection data"
                )

            try:
                selection_data = await self._storage.retrieve(selection_ref)
                context["selection"] = json.loads(selection_data.decode("utf-8"))
                logger.info(
                    f"Retrieved selection data: {selection_ref} "
                    f"({context['selection'].get('atom_count', 0)} atoms)"
                )
            except Exception as e:
                logger.warning(f"Failed to retrieve selection {selection_ref}: {e}")
                return ToolResult.error(
                    f"Failed to retrieve selection data '{selection_ref}': {e}"
                )

        # Collect all data references for sandbox context
        # 1. From explicit data_refs parameter
        all_refs = set(data_refs) if data_refs else set()

        # 2. Auto-detect refs in code (if aevom is used)
        if "aevom" in code:
            detected_refs = self._detect_refs_in_code(code)
            all_refs.update(detected_refs)
            if detected_refs:
                logger.info(f"Auto-detected data references in code: {detected_refs}")

        # 3. Include selection_ref if provided
        if selection_ref:
            all_refs.add(selection_ref)

        # Build sandbox context for aevom client
        # We need a sandbox context when:
        # 1. Code uses 'aevom' (for visualization, storage, RDKit operations)
        # 2. Code has data refs to access
        sandbox_context = None
        needs_aevom = "aevom" in code

        if (needs_aevom or all_refs) and self._storage:
            # Validate any explicitly-named / code-detected refs.
            valid_refs = await self._validate_refs(all_refs) if all_refs else set()

            # Get client session_id if storage is session-scoped
            # This enables sandbox-stored artifacts to appear in the client's storage strip
            client_session_id = getattr(self._storage, "session_id", None)

            # Grant the sandbox access to *every* artifact in this session, not
            # just the refs that happen to appear literally in the code. The old
            # behaviour (regex-detect refs in the code + explicit data_refs) meant
            # `aevom.load(ref)` raised AccessDeniedError for any ref the model did
            # not spell out verbatim, and there was no way to *enumerate* storage
            # from inside the sandbox at all -- the model had to copy refs out of
            # the [STORAGE STATE] prompt block by hand. Listing the session here
            # makes `aevom.load(any_session_ref)` just work and makes
            # `aevom.allowed_refs` a usable inventory. Refs come straight from the
            # session listing so they are known to exist -- no per-ref exists()
            # round-trip needed. Access stays scoped to this session: a ref from
            # another session is still denied, both here and server-side in the
            # sandbox data route.
            session_refs = await self._list_session_refs(client_session_id)
            valid_refs |= session_refs

            # Create sandbox session - always create if aevom is used
            session_manager = get_session_manager()
            session = await session_manager.create_session(
                allowed_refs=valid_refs,
                ttl_minutes=None,  # No expiry for simplicity
                client_session_id=client_session_id,
            )

            # Get backend URL for pyexec to call back
            # The pyexec container accesses backend via docker network
            backend_url = os.environ.get("SANDBOX_BACKEND_URL", "http://backend:8000")

            sandbox_context = {
                "session_token": session.id,
                "backend_url": backend_url,
                "allowed_refs": list(valid_refs),
            }

            logger.info(
                f"Created sandbox session {session.id[:8]}... "
                f"with {len(valid_refs)} allowed refs "
                f"({len(session_refs)} from session listing, aevom needed: "
                f"{needs_aevom}, client_session: "
                f"{client_session_id[:8] if client_session_id else None}...)"
            )

        # Track the session for later cleanup/visualization retrieval
        session_token = (
            sandbox_context.get("session_token") if sandbox_context else None
        )

        # Under OpenShell each chat session has its own sandbox. There is no
        # fallback to the shared pyexec service: if the sandbox is unavailable
        # the call fails rather than running ungoverned.
        governed = openshell_enabled()
        route: dict[str, Any] = {}
        if governed:
            chat_session_id = getattr(self._storage, "session_id", None) or "default"
            try:
                endpoint = await get_openshell_manager().get_endpoint(chat_session_id)
            except SandboxUnavailableError as e:
                return ToolResult.error(f"Python sandbox unavailable: {e}")
            except Exception as e:
                logger.exception("Could not resolve the OpenShell sandbox")
                return ToolResult.error(f"Python sandbox unavailable: {e}")
            tool_span = trace.get_current_span()
            tool_span.set_attribute("sandbox.name", endpoint.name)
            tool_span.set_attribute("sandbox.wait_ms", round(endpoint.wait_ms, 1))
            tool_span.set_attribute("sandbox.cold_start", endpoint.cold_start)
            route = {
                "base_url": endpoint.base_url,
                "headers": endpoint.headers,
                "client_kwargs": endpoint.httpx_kwargs,
            }

        try:
            # Build request to pyexec service
            request_data = {
                "code": code,
                "context": context,
                "timeout_seconds": timeout_seconds,
            }

            # Add sandbox context if we have refs to access
            if sandbox_context:
                request_data["sandbox_context"] = sandbox_context

            response = await self._call_service(
                endpoint="/execute",
                method="POST",
                json_data=request_data,
                timeout=float(timeout_seconds + 10),
                retries=2 if is_training else 0,
                **route,
            )

            success = response.get("success", False)
            output = response.get("output", "")
            error = response.get("error")
            result_value = response.get("result")

            # Check for pending visualization events from the session
            visualization_events = []
            if session_token:
                try:
                    visualization_events = await session_manager.get_and_clear_visualization_events(session_token)
                    if visualization_events:
                        logger.info(
                            f"Retrieved {len(visualization_events)} visualization events "
                            f"from session {session_token[:8]}..."
                        )
                except Exception as e:
                    logger.warning(f"Failed to retrieve visualization events: {e}")

            if success:
                # Heuristic: if pyexec reports success but the output consists
                # entirely of error messages (common when code uses try/except
                # to catch and print errors), treat as a failure.
                if output and not result_value and _output_indicates_failure(output):
                    return ToolResult.error(
                        f"Code ran but produced only errors:\n{output}"
                        + (sandbox_denial_note(output) if governed else "")
                    )

                # Build response text
                response_parts = []

                if output:
                    response_parts.append(f"Output:\n{output}")

                if result_value is not None:
                    result_str = json.dumps(result_value, indent=2, default=str)
                    response_parts.append(f"Result:\n{result_str}")

                if not response_parts:
                    response_parts.append("Code executed successfully (no output)")

                # Molecules written out here are worth checking, not worth
                # avoiding. Measured over 10,300 screened candidates across 300
                # benchmark runs, the ones the model wrote by hand satisfied the
                # full constraint set 16.6% of the time against 6.1% for
                # enumerated analogues -- 2.7x, and independently 2.4x, 2.5x and
                # 3.2x in the three arms. Directed design beats a blind one-cut
                # sweep; this is the productive channel, and an earlier version
                # of this note steered away from it.
                #
                # What it lacks is a check. Of 2759 hand-written structures
                # 11.5% did not parse, and the loop that builds them almost
                # always ends `if mol is None: continue`, so the loss leaves no
                # trace and no count. Said here rather than in a docstring
                # because this is where it is happening.
                #
                # Silent when the same block generates its candidates: a run
                # that calls the enumerator and also carries a long list of
                # literals is re-pasting molecules it has already tested,
                # because there is nowhere else to keep them.
                handwritten = (
                    0 if "enumerate_analogs" in code
                    else _handwritten_structures(code)
                )
                if handwritten >= HANDWRITTEN_STRUCTURES:
                    response_parts.append(
                        f"\n[{handwritten} molecules here are written out as "
                        "string literals. That is the more productive way to "
                        "generate candidates -- hand-written structures clear a "
                        "full constraint set about 2.7x as often as enumerated "
                        "analogues -- but about one in nine does not parse, and "
                        "`if mol is None: continue` drops those without saying "
                        "so. aevom.mol_from_smiles(s) raises instead, naming the "
                        "mistake; aevom.describe_molecule(s) lists the groups "
                        "actually present, which is how a structure that is not "
                        "the one you meant gets caught before it is scored.]"
                    )
                if handwritten >= HANDWRITTEN_STRUCTURES:
                    response_parts.append(
                        f"\n[{handwritten} molecules in this code are written "
                        "out as string literals. In a measured sample about 45% "
                        "of hand-written structures were unusable -- a fifth "
                        "would not parse at all, and most of the rest had "
                        "drifted off the scaffold -- and the failures are "
                        "silent. aevom.enumerate_analogs(lead, r_groups=[...]) "
                        "attaches substituents with RDKit, so pass the "
                        "substituents you have in mind and let it write the "
                        "molecules: it cannot produce an unparseable one, and "
                        "similarity_min keeps the rest on scaffold.]"
                    )

                # If there are visualization events, mention them
                if visualization_events:
                    viz_count = len(visualization_events)
                    response_parts.append(
                        f"\n[{viz_count} visualization(s) triggered in MolStar viewer]"
                    )

                # Create result with visualization events in metadata
                result = ToolResult.text("\n\n".join(response_parts))
                if visualization_events:
                    result.metadata["visualization_events"] = visualization_events

                return result
            else:
                error_msg = error or "Unknown execution error"
                response_text = f"Execution failed: {error_msg}"
                if output:
                    response_text += f"\n\nPartial output:\n{output}"
                if governed:
                    response_text += sandbox_denial_note(f"{error_msg}\n{output}")
                return ToolResult.error(response_text)

        except httpx.TimeoutException as e:
            logger.exception("Failed to call pyexec service")
            where = "Sandbox" if governed else "The pyexec service"
            return ToolResult.error(
                f"{where} did not respond within {timeout_seconds + 10}s "
                f"({type(e).__name__}). Code that runs too long is stopped at its "
                f"own {timeout_seconds}s limit and reported as such, so this was "
                "most likely a stalled connection rather than slow code. The code "
                "may or may not have run."
            )
        except Exception as e:
            logger.exception("Failed to call pyexec service")
            return ToolResult.error(f"Failed to execute code: {str(e) or type(e).__name__}")
