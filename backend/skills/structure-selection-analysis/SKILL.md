---
name: Structure Selection & Neighbor Analysis
description: Select and analyze parts of a protein structure -- residues near a site, a binding-site/neighbor shell within N angstroms of a residue or ligand, atoms of a chain, etc. -- and report which residues or atoms are involved. Use whenever a task asks for residues/atoms "near", "within N A of", "surrounding", "in the pocket of", or "in contact with" something, or asks to analyze a viewer selection in Python. Covers the structure_query -> python_execute workflow and the correct sandbox-safe patterns.
allowed_tools:
  - fetch_and_clean_pdb
  - structure_query
  - python_execute
  - visualize_pdb
---

# Structure Selection & Neighbor Analysis

Find the residues/atoms near a site and report them, reliably and in few steps.
The two ways this goes wrong are (1) reaching for a hand-rolled distance
calculation when the viewer can do the selection, and (2) writing sandbox code
that trips over RestrictedPython. This skill gives the one right path for each.

## When to use

- "which residues are within 5 A of residue X / the ligand / a metal"
- "atoms/residues surrounding / in contact with / lining the pocket of ..."
- analyzing an existing viewer selection in `python_execute`

## Preferred path: let the viewer do the selection

`structure_query` runs a MolQL selection **in the viewer** and returns the
atom/residue counts plus a `selection_*` reference. Prefer it for spatial
selections -- the viewer computes the geometry, so you never write a distance
loop.

```text
1. fetch_and_clean_pdb(pdb_id="1CRN")          -> pdb_abc12345
2. structure_query(
     pdb_ref="pdb_abc12345",
     molql="(sel.atom.include-surroundings (sel.atom.res (and (= atom.chain A) (= atom.resno 44))) :radius 5 :as-whole-residues true)")
   -> "Selected 67 atoms across 13 residues. Reference: selection_def678"
```

MolQL idioms (see the `structure_query` schema for the full cheatsheet):

- Shell around a **specific residue**: `(sel.atom.include-surroundings (sel.atom.res (and (= atom.chain A) (= atom.resno 44))) :radius 5 :as-whole-residues true)`
- Shell around the **ligand**: `(sel.atom.include-surroundings (sel.atom.res (= atom.entity-type non-polymer)) :radius 5 :as-whole-residues true)`

The count comes back in the tool result, but the residue **names** do not --
to list them, hand the `selection_*` ref to `python_execute` (next section).

## Listing residues from a selection in python_execute

Pass the `selection_ref`; the sandbox injects a `selection` dict
(`selection['atoms']`, each atom with `residue.name/seqNumber/chainId` and a
`coords` list). Collect unique residues with a **set comprehension** -- do not
count into a dict (see pitfalls):

```python
# python_execute(selection_ref="selection_def678", code=...)
residues = sorted(
    {(a["residue"]["chainId"], a["residue"]["seqNumber"], a["residue"]["name"])
     for a in selection["atoms"]},
    key=lambda r: (r[0], r[1]),
)
print(f"{selection['atom_count']} atoms across {len(residues)} residues")
for chain, seq, name in residues:
    print(f"  {name} {chain}{seq}")
result = {"residues": [f"{n}{c}{s}" for c, s, n in residues]}
```

## Fallback: compute the shell in python (no viewer, or a custom cutoff)

If `structure_query` is unavailable or times out ("waiting for the viewer"), or
you need a cutoff the viewer can't express, compute it yourself -- but load the
structure as plain atom dicts with **`aevom.load_pdb_structure`** (each atom has
`chain_id`, `residue_num`, `residue_name`, `name`, `coords`). This sidesteps
biotite's parsing gotchas entirely.

```python
import numpy as np
from scipy.spatial.distance import cdist

atoms = aevom.load_pdb_structure("pdb_abc12345")["atoms"]
coords = np.array([a["coords"] for a in atoms])

center_mask = np.array([a["chain_id"] == "A" and a["residue_num"] == 44 for a in atoms])
min_dist = cdist(coords, coords[center_mask]).min(axis=1)     # nearest center atom
in_shell = min_dist <= 5.0

residues = sorted(
    {(atoms[i]["chain_id"], atoms[i]["residue_num"], atoms[i]["residue_name"])
     for i in np.where(in_shell)[0]},
    key=lambda r: (r[0], r[1]),
)
print(f"{int(in_shell.sum())} atoms across {len(residues)} residues")
result = {"residues": [f"{n}{c}{i}" for c, i, n in residues]}
```

## Pitfalls that break these runs

- **Do not accumulate into a dict/list item.** `counts[key] += 1`,
  `arr[i] += x`, `obj.attr += x` all raise
  *"Augmented assignment of object items and slices is not allowed"* in the
  sandbox. For "which residues", you only need the **set** of residue keys, so
  the set comprehension above avoids counting entirely. If you truly need counts,
  use `collections.Counter(keys)`.
- **`coords` is a list `[x, y, z]`** -- unhashable, so never put it in a set or
  use it as a dict key. Key on `(chainId, seqNumber, name)` instead.
- **Don't hand-parse with biotite when you can avoid it.** If you do use it,
  `PDBFile.get_structure()` returns an `AtomArrayStack` -- pass `model=1` for a
  flat `AtomArray`, and `extra_fields=["b_factor"]` if you need B-factors.
- **Prefer the viewer over a distance loop.** Reach for the python fallback only
  when the viewer selection isn't available.
