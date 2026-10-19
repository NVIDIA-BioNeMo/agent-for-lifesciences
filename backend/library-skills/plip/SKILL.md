---
name: plip
description: >
  Profile the non-covalent interactions in a protein-ligand complex — hydrogen
  bonds, hydrophobic contacts, pi-stacking, pi-cation, salt bridges, halogen
  bonds, water bridges and metal coordination — using PLIP's published geometric
  criteria. Invoke to state a pharmacophore hypothesis from a structure, to
  decide which contacts a new scaffold must keep, to explain why one pose scores
  better than another, or to check that an analog retains a key interaction. Out
  of scope: binding affinity (vina_dock, boltz2), proximity-only contact lists
  (biopython), descriptor panels (rdkit).
license: GPL-2.0
---

# PLIP in Aevom

PLIP answers a question docking scores cannot: not *how tightly* a ligand binds,
but *how* — which residue, which atom, which kind of interaction, and with what
geometry.

## Running this in Aevom

**Use the `plip_interactions` tool. Do not write PLIP code.**

The PLIP package is deliberately not installed in the `python_execute` sandbox,
so every `import plip` in the documentation below will fail. Two reasons, both
practical:

- PLIP's API is file-based throughout — `PDBComplex.load_pdb(path)` and report
  writers that take output directories. The sandbox has no filesystem and
  rejects `open()`.
- PLIP's protonation step writes to a fixed path, so two analyses running at
  once race each other. The service that backs the tool serialises this; a
  sandbox copy would not.

The tool takes a `structure_ref` for a complex that already contains the ligand
as a hetero record, and returns residue-level interaction records. It handles
protonation, ligand detection and the settings that make results reproducible.

```
plip_interactions(structure_ref="pdb_a1b2c3d4")
```

Ignore, in any PLIP documentation you have seen: the `plipcmd` command line,
`from plip.structure.preparation import PDBComplex`, `PDBComplex.load_pdb`,
`BindingSiteReport`, and every instruction to read a PDB from a path or write a
report to disk.

## What comes back, and how to read it

Each interaction names a **protein residue**, a **ligand atom**, and a **type**.
The geometry is the point — the types are not interchangeable, and each carries
a different implication for what a replacement scaffold must preserve.

- **Hydrogen bond.** Reported with donor, acceptor, distance and angle. PLIP
  requires the D–H···A angle to be reasonable, not merely that two heteroatoms
  are close, which is why its hydrogen bonds are a much better pharmacophore
  hypothesis than a distance cutoff. Note which partner is the donor: swapping
  donor for acceptor at the same position loses the interaction.
- **Hydrophobic contact.** Carbon-to-carbon proximity. Individually weak and
  numerous, so read them as a shape complementarity signal in aggregate rather
  than as individual anchors.
- **Pi-stacking.** Parallel or T-shaped, with an offset. Requires an aromatic
  ring on both sides; the geometry constrains ring placement tightly.
- **Pi-cation.** An aromatic ring against a charged group, usually Lys, Arg or a
  protonated ligand amine.
- **Salt bridge.** Formal charges of opposite sign in range. Strong and
  directional in the sense that it pins a charged group to a specific residue,
  and pH-dependent — a salt bridge to a histidine may not survive a change in
  protonation state.
- **Halogen bond.** A sigma-hole interaction from a ligand Cl, Br or I to a
  backbone or sidechain acceptor. Easy to lose by replacing a halogen with a
  same-size but non-polarisable group.
- **Water bridge.** Ligand and protein both hydrogen-bond to the same ordered
  water. Real, but the water must be present in the input structure; a
  docked or predicted pose usually has none, so an empty water-bridge list says
  more about the model than about the binding site.
- **Metal coordination.** Ligand donor atoms to a bound Zn, Mg, Fe or Mn.
  Usually the dominant term when present.

## Choosing what to profile

The input must be a **complex** — receptor and ligand in one structure, with the
ligand as a hetero record. That is what an experimental co-crystal gives you and
what `vina_dock` and `boltz2` return.

Profiling both a reference co-crystal and a new pose, then comparing the
interaction sets, is the usual move: it turns "the score got worse" into "the
hinge hydrogen bond is gone". When a designed analog must keep a contact, check
it explicitly rather than inferring it from the score.

An empty or near-empty result usually means the ligand was not recognised as a
hetero record. Confirm the complex actually contains one — `biopython` can list
hetero residues in a couple of lines — before concluding there are no
interactions.
