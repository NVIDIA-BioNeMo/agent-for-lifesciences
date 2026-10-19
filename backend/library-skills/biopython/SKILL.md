---
name: biopython
description: >
  Use BioPython inside the python_execute sandbox for protein structure work:
  parsing PDB text into a Structure/Model/Chain/Residue/Atom hierarchy, radius
  and neighbour searches, distances and angles, residue selection, B-factors and
  occupancies, sequence extraction, and superimposition. Invoke for pocket
  residue listing, contact analysis, chain and hetero-record inspection, or
  geometric measurements on a receptor. Out of scope: small-molecule
  cheminformatics (rdkit), file-format conversion and protonation (openbabel),
  interaction typing (plip_interactions).
license: BSD-3-Clause
---

# BioPython in the Aevom sandbox

`Bio` is installed and allowlisted. It is the default choice for walking a
protein structure by residue and chain, and for neighbour searches.

## Running this in Aevom

- **Parse from a string, not a path.** `PDBParser.get_structure` accepts a
  file-like object, so wrap the text: `StringIO(pdb_text)`. `io` is allowlisted
  for exactly this.
- **Get the text from a reference**: `pdb_text = aevom.load(pdb_ref)`. There are
  no files in the sandbox to open.
- **Silence the parser warnings.** Real PDB entries trigger
  `PDBConstructionWarning` for discontinuous chains and missing elements; pass
  `QUIET=True` unless you are specifically auditing the file.
- `Bio.Entrez`, `Bio.PDB.PDBList` and anything else that downloads will fail —
  there is no network. Structures come in as references.
- **`atom.coord = new_xyz` is rejected**, along with every other attribute or
  array assignment; only `list`, `dict` and `set` are writable. Use the setter
  methods (`atom.set_coord(...)`) or `Superimposer.apply()`, which mutate
  internally.
- **Write back through a buffer.** `PDBIO.save()` accepts a file-like object, so
  `io_.save(StringIO())` works; take `.getvalue()` and hand it to
  `aevom.store_pdb(text, meta)`. Passing a filename does not work.
- No `os`, `sys`, `subprocess`, or `open()`.

## Parsing and walking the hierarchy

```python
from io import StringIO
from Bio.PDB import PDBParser

structure = PDBParser(QUIET=True).get_structure("rec", StringIO(aevom.load(pdb_ref)))
model = structure[0]

result = {
    "chains": [c.id for c in model],
    "n_residues": sum(1 for r in model.get_residues() if r.id[0] == " "),
    "n_atoms": sum(1 for _ in model.get_atoms()),
}
```

The hierarchy is Structure → Model → Chain → Residue → Atom. A residue's `id` is
the tuple `(hetflag, resseq, icode)`: `hetflag` is `" "` for standard amino
acids, `"W"` for water and `"H_XXX"` for a hetero-residue such as a ligand or
cofactor. Filtering on `r.id[0] == " "` is how you drop waters and ligands.

## Pocket residues around a point

```python
from io import StringIO
import numpy as np
from Bio.PDB import PDBParser, NeighborSearch, Selection

structure = PDBParser(QUIET=True).get_structure("rec", StringIO(aevom.load(pdb_ref)))
atoms = Selection.unfold_entities(structure[0], "A")
ns = NeighborSearch(atoms)

center = np.array([12.4, -3.1, 25.6])
near = ns.search(center, 10.0, level="R")
result = sorted(
    f"{r.get_parent().id}:{r.get_resname()}{r.id[1]}"
    for r in near
    if r.id[0] == " "
)
```

`NeighborSearch` builds a KD-tree once and then answers queries in log time.
Use it rather than a nested loop over atoms — on a 5,000-atom receptor the
difference is seconds versus minutes. `level=` selects what comes back: `"A"`
atoms, `"R"` residues, `"C"` chains.

## Contacts between a ligand and its receptor

```python
from io import StringIO
from Bio.PDB import PDBParser, NeighborSearch

structure = PDBParser(QUIET=True).get_structure("cx", StringIO(aevom.load(complex_ref)))
model = structure[0]

ligand = [r for r in model.get_residues() if r.id[0].startswith("H_")][0]
protein_atoms = [a for a in model.get_atoms() if a.get_parent().id[0] == " "]
ns = NeighborSearch(protein_atoms)

contacts = {}
for atom in ligand:
    for partner in ns.search(atom.coord, 4.0, level="A"):
        res = partner.get_parent()
        key = f"{res.get_parent().id}:{res.get_resname()}{res.id[1]}"
        d = float(atom - partner)
        contacts[key] = min(contacts.get(key, 99.0), round(d, 2))
result = dict(sorted(contacts.items(), key=lambda kv: kv[1]))
```

`atom_a - atom_b` is overloaded to give the distance in ångström. This reports
proximity only; for hydrogen bonds, pi-stacking and salt bridges — which need
angles and protonation, not just distance — use the `plip_interactions` tool.

## Geometry

```python
import numpy as np
from Bio.PDB import vectors

angle = vectors.calc_angle(a1.get_vector(), a2.get_vector(), a3.get_vector())
dihedral = vectors.calc_dihedral(
    a1.get_vector(), a2.get_vector(), a3.get_vector(), a4.get_vector()
)
result = {"angle_deg": np.degrees(angle), "dihedral_deg": np.degrees(dihedral)}
```

Both return **radians**.

## Sequence from a structure

```python
from Bio.PDB.Polypeptide import PPBuilder

ppb = PPBuilder()
result = {
    chain.id: "".join(str(pp.get_sequence()) for pp in ppb.build_peptides(chain))
    for chain in structure[0]
}
```

`PPBuilder` breaks the chain at gaps, which is why the peptides are joined. If
you need gap positions, iterate the peptides instead of joining them.

## Superimposing two structures

```python
from Bio.PDB import Superimposer

fixed = [r["CA"] for r in ref_chain if "CA" in r]
moving = [r["CA"] for r in alt_chain if "CA" in r]

sup = Superimposer()
sup.set_atoms(fixed, moving)
sup.apply(list(alt_chain.get_atoms()))
result = {"rmsd": round(sup.rms, 3)}
```

`set_atoms` requires the two lists to be equal length and in correspondence, so
select on a shared residue numbering before calling it. For structures that need
alignment first, biotite's `superimpose_homologs` handles the correspondence for
you.

## Handing a structure back

```python
from io import StringIO
from Bio.PDB import PDBIO

writer = PDBIO()
writer.set_structure(structure)
buf = StringIO()
writer.save(buf)
result = aevom.store_pdb(buf.getvalue(), {"name": "aligned_receptor"})
```

Store rather than printing: a reference costs a few tokens where a receptor's
PDB text costs tens of thousands.
