---
name: openbabel
description: >
  Use OpenBabel inside the python_execute sandbox for chemical file-format
  conversion and structure preparation: translating between PDB, PDBQT, MOL2,
  SDF, SMILES and XYZ, adding hydrogens at a chosen pH, assigning Gasteiger or
  MMFF94 partial charges, perceiving bonds from coordinates, and detecting
  hydrogen-bond donors and acceptors. Invoke when RDKit refuses a structure,
  when a receptor needs protonation or charges, or when a format has no other
  reader. Out of scope: docking (vina_dock, which prepares its own receptor),
  interaction profiling (plip_interactions), descriptor panels (rdkit).
license: GPL-2.0
---

# OpenBabel in the Aevom sandbox

`openbabel` is installed (from the `openbabel-wheel` distribution) and
allowlisted. Its niche is the messy middle: formats and structures that RDKit's
stricter parsers reject, and the protonation and charge assignment that docking
preparation needs.

## Running this in Aevom

Every OpenBabel tutorial you have seen uses files. None of that works here.

- **`ReadString` and `WriteString`, never `ReadFile`/`WriteFile`.** There is no
  filesystem, and `open()` is rejected. Load with `aevom.load(ref)` and pass the
  text straight in.
- Pybel's `readfile()` is likewise unavailable; `pybel.readstring(fmt, text)` is
  the equivalent.
- **The `obabel` command line does not exist here** — no `subprocess`, no shell.
  Everything is the Python API.
- The two import spellings are `from openbabel import openbabel` (the low-level
  SWIG bindings, conventionally aliased `ob`) and `from openbabel import pybel`
  (the ergonomic wrapper). Bare `import openbabel` gives you the namespace
  package, not the module.
- **OpenBabel writes warnings to stderr** during perception on real structures.
  They are usually benign; check the returned object rather than the noise.

## Format conversion

```python
from openbabel import openbabel as ob

conv = ob.OBConversion()
conv.SetInAndOutFormats("pdb", "mol2")
mol = ob.OBMol()
conv.ReadString(mol, aevom.load(pdb_ref))
mol2_text = conv.WriteString(mol)
result = {"atoms": mol.NumAtoms(), "bonds": mol.NumBonds()}
```

`SetInAndOutFormats` takes the format extensions without a dot. The pairs that
come up most are `pdb`→`pdbqt` (docking input), `pdb`→`mol2` (typed atoms and
charges), `sdf`→`smi` (canonical SMILES) and `xyz`→`sdf` (bond perception from
raw coordinates).

`ReadString` returns `False` when parsing fails. Check it — an unchecked failure
leaves you holding an empty `OBMol` that silently produces empty output.

## Protonation at a pH

```python
from openbabel import openbabel as ob

conv = ob.OBConversion()
conv.SetInAndOutFormats("pdb", "pdb")
mol = ob.OBMol()
conv.ReadString(mol, aevom.load(pdb_ref))

mol.DeleteHydrogens()
mol.AddHydrogens(False, True, 7.4)   # polaronly=False, correctForPH=True, pH
result = {"n_atoms": mol.NumAtoms(), "pdb": conv.WriteString(mol)}
```

The three arguments to `AddHydrogens` are `polaronly`, `correctForPH` and `pH`.
Passing `polaronly=True` adds only hydrogens on N, O and S, which is what you
want for a docking receptor; the full set is needed for forcefield work.
`DeleteHydrogens()` first makes the operation idempotent — otherwise re-running
on an already-protonated structure stacks a second set of hydrogens on top.

## Partial charges

```python
from openbabel import openbabel as ob

charger = ob.OBChargeModel.FindType("gasteiger")
charger.ComputeCharges(mol)
result = [mol.GetAtom(i + 1).GetPartialCharge() for i in range(mol.NumAtoms())]
```

`GetAtom` is **1-indexed**; iterating `range(mol.NumAtoms())` directly and
passing the raw index silently skips atom 1 and runs off the end. `"gasteiger"`
is the fast empirical model used by AutoDock preparation; `"mmff94"` and
`"eem"` are also available.

## Donor and acceptor perception

```python
from openbabel import openbabel as ob

donors, acceptors = [], []
for i in range(1, mol.NumAtoms() + 1):
    atom = mol.GetAtom(i)
    if atom.IsHbondDonor():
        donors.append(i)
    if atom.IsHbondAcceptor():
        acceptors.append(i)
result = {"donors": donors, "acceptors": acceptors}
```

Treat this as a starting point on proteins, not as ground truth. OpenBabel's
perception works from valence and hybridisation with no knowledge of residue
chemistry, so it mistypes cases a template would get right — a proline backbone
nitrogen, for instance, has no hydrogen and cannot donate, and is not an
acceptor either, but generic perception has been observed to call it one.
Histidine tautomers are the other common failure. Where donor/acceptor
assignment is the answer rather than an intermediate, use `plip_interactions`
or the `probe_grid` tool, both of which apply residue templates.

## Pybel, for short scripts

```python
from openbabel import pybel

mol = pybel.readstring("smi", "CC(=O)Oc1ccccc1C(=O)O")
mol.addh()
mol.make3D(forcefield="mmff94", steps=250)
result = {"formula": mol.formula, "mw": mol.molwt, "sdf": mol.write("sdf")}
```

`make3D` generates and minimises a conformer, which is convenient but slow —
budget roughly a second per drug-sized molecule, and prefer RDKit's
`EmbedMultipleConfs` when you need many conformers or reproducible ones.
