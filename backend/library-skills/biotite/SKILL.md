---
name: biotite
description: >
  Use biotite inside the python_execute sandbox for fast array-based structural
  biology: parsing PDB and mmCIF into a vectorised AtomArray, boolean-mask
  selection of chains, residues and elements, cell-list neighbour queries,
  superimposition including sequence-aware homolog alignment, RMSD and RMSF,
  solvent-accessible surface area, secondary structure assignment, and
  hydrogen-bond detection. Invoke for whole-structure numeric work, ensemble
  comparison, burial and exposure calculations, or when BioPython's per-object
  hierarchy is too slow. Out of scope: fetching structures (no network),
  running external programs, small-molecule cheminformatics (rdkit).
license: BSD-3-Clause
---

# biotite in the Aevom sandbox

`biotite` is installed and allowlisted. Where BioPython gives you a tree of
Python objects, biotite gives you an `AtomArray`: parallel numpy arrays of
coordinates and annotations. That makes whole-structure arithmetic both shorter
and roughly an order of magnitude faster.

## Running this in Aevom

- **Two submodules are blocked** and importing either raises `ImportError`:
  - `biotite.database` — `fetch()` against RCSB, UniProt, Entrez and PubChem.
    There is no network. Structures arrive as references; load with
    `aevom.load(ref)`.
  - `biotite.application` — wrappers that launch DSSP, MUSCLE, BLAST and other
    external binaries. None are installed and the sandbox does not spawn
    processes. For secondary structure use `annotate_sse`, which is pure Python
    and needs no DSSP.
- **Parse from `StringIO`, not a path.** `PDBFile.read()` accepts a file-like
  object; `io` is allowlisted for this.
- **You cannot assign into an array or an attribute.** `arr.coord = new_coord`
  and `arr.coord[0] = x` both raise `Write operations are restricted`; only
  `list`, `dict` and `set` are writable. Build new values instead:
  `struc.translate(arr, vector)` and `struc.rotate(arr, angles)` return a
  transformed copy, whole-array arithmetic (`arr.coord + shift`) works, and
  augmented assignment on a name you own (`coords += shift`) works. This is the
  single most common way biotite code written elsewhere fails here.
- Everything else in `biotite.structure`, `biotite.structure.io` and
  `biotite.sequence` is available.
- No `os`, `sys`, `subprocess`, or `open()`.

## Parsing

```python
from io import StringIO
import biotite.structure.io.pdb as pdb

arr = pdb.PDBFile.read(StringIO(aevom.load(pdb_ref))).get_structure(model=1)
result = {"n_atoms": arr.array_length(), "chains": list(set(arr.chain_id))}
```

Pass `model=1` for a single `AtomArray`. Omitting it on a multi-model file
returns an `AtomArrayStack` — a 3D coordinate array over models — which is what
you want for an NMR ensemble or a trajectory, and a surprise otherwise. For
mmCIF, swap in `biotite.structure.io.pdbx.CIFFile` and
`pdbx.get_structure(file, model=1)`.

## Per-residue confidence (pLDDT) and B-factors

`get_structure` **drops the B-factor column unless you ask for it**, so a
default parse then `arr.b_factor` raises `AttributeError` /
`Annotation category 'b_factor' is not existing`. Request it with
`extra_fields`. Predicted structures from `openfold3` and `boltz2` carry their
**per-residue pLDDT in the B-factor column**:

```python
from io import StringIO
import biotite.structure.io.pdb as pdb

arr = pdb.PDBFile.read(StringIO(aevom.load(pred_pdb_ref))).get_structure(
    model=1, extra_fields=["b_factor"])
ca = arr[arr.atom_name == "CA"]                 # one value per residue
plddt = ca.b_factor
low = ca.res_id[plddt < 70]                      # low-confidence residues
result = {"mean_plddt": round(float(plddt.mean()), 1),
          "low_conf_residues": low.tolist()}
```

`extra_fields` also takes `"occupancy"` and `"charge"`; the pdbx (mmCIF) reader
accepts the same argument. Note the `openfold3` / `boltz2` tool result already
prints the **overall** pLDDT/pTM — read that straight from the tool output, and
only parse the structure when you need the per-residue profile (which regions
are uncertain).

## Selection by boolean mask

```python
import numpy as np
import biotite.structure as struc

protein = arr[struc.filter_amino_acids(arr)]
backbone = arr[np.isin(arr.atom_name, ["N", "CA", "C", "O"])]
chain_a = arr[arr.chain_id == "A"]
ligand = arr[~struc.filter_amino_acids(arr) & ~struc.filter_solvent(arr)]

center = np.array([12.4, -3.1, 25.6])
pocket = arr[np.linalg.norm(arr.coord - center, axis=1) < 10.0]
result = {"pocket_atoms": pocket.array_length()}
```

Annotations (`chain_id`, `res_id`, `res_name`, `atom_name`, `element`,
`hetero`) are numpy arrays of the same length as the structure, so masks compose
with `&`, `|` and `~`. This is the whole idiom — most biotite code is one mask
and one vectorised operation. `b_factor`, `occupancy` and `charge` are **not**
loaded by default — request them via `extra_fields` (see above).

## Neighbours

```python
import biotite.structure as struc

cell_list = struc.CellList(protein, cell_size=5.0)
neighbour_idx = cell_list.get_atoms(ligand.coord, radius=4.5)
contacted = np.unique(neighbour_idx[neighbour_idx != -1])
result = sorted(
    set(zip(protein.res_id[contacted].tolist(), protein.res_name[contacted].tolist()))
)
```

`get_atoms` returns a padded 2D index array with `-1` in the empty slots, so
filter those out before indexing. Set `cell_size` near the query radius.

## Superimposition and RMSD

```python
import biotite.structure as struc

fitted, transform = struc.superimpose(ref_arr, mobile_arr)
result = {"rmsd": float(struc.rmsd(ref_arr, fitted))}
```

`superimpose` requires the two arrays to correspond atom-for-atom. When they do
not — different constructs, missing loops, a homolog — use
`superimpose_homologs`, which aligns the sequences first and superimposes on the
matched residues. That is the function to reach for when overlaying related
co-crystals. It returns **four** values, not two:

```python
fitted, transform, ref_idx, mobile_idx = struc.superimpose_homologs(ref_arr, mobile_arr)
result = {"n_matched": len(ref_idx), "rmsd": float(struc.rmsd(ref_arr[ref_idx], fitted[mobile_idx]))}
```

`ref_idx` and `mobile_idx` are the anchor atoms it actually matched. Compute RMSD
over those, not over the whole array — the unmatched residues are exactly the
ones that do not correspond.

For an ensemble, `struc.rmsf(struc.average(stack), stack)` gives per-atom
fluctuation.

## Surface area and burial

```python
import biotite.structure as struc

sasa_atom = struc.sasa(protein, vdw_radii="Single", point_number=200)
sasa_res = struc.apply_residue_wise(protein, sasa_atom, np.nansum)
result = dict(zip(protein.res_id[struc.get_residue_starts(protein)].tolist(),
                  np.round(sasa_res, 1).tolist()))
```

`sasa` returns `NaN` for atoms excluded from the calculation, so aggregate with
`np.nansum` rather than `np.sum`. `apply_residue_wise` is the general
residue-level reduction and works with any function over the per-atom values.

## Secondary structure and hydrogen bonds

```python
import biotite.structure as struc

sse = struc.annotate_sse(protein)          # 'a' helix, 'b' sheet, 'c' coil
triplets = struc.hbond(protein)            # (donor_H, donor, acceptor) indices
result = {
    "helix_fraction": round(float((sse == "a").mean()), 3),
    "n_hbonds": len(triplets),
}
```

`annotate_sse` implements P-SEA and returns one character per **residue**, not
per atom. `hbond` uses a geometric distance-and-angle criterion and needs
explicit hydrogens present — on a structure without them it returns nothing,
which looks like a result but is not. Protonate with OpenBabel first, or use
`plip_interactions`, which handles protonation itself.
