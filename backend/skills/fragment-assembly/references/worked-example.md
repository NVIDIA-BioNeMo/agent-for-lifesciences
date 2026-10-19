# One case, end to end

BRAF kinase with sorafenib (PDB 1uwh), decomposed the way SMDD-Bench builds a
Fragment Assembly task: the ligand is cut at two BRICS bonds, the two end pieces
are handed over with their crystal poses, and the bridge between them is hidden.
Every number below is what the tools actually returned; the fixtures are in
`services/rdkit/tests/data/`.

**What you are given.** Two fragments, `FC(F)(F)c1ccccc1Cl` and
`CNC(=O)c1cc(O)ccn1`, each with coordinates, plus the pocket. What is hidden is
the urea-anilide bridge, 10 heavy atoms of which 7 lie on the path between the
attachment points.

## Step 1: the geometry

```
fragment_exit_vectors(fragment_sdf_refs=[frag0, frag1], pocket_pdb_ref=pocket)
```

The fragments' closest heavy atoms are 9.42 Å apart. They spend 1 rotatable bond
between them, so joining them leaves **7 of the 10 allowed**.

Top-ranked pair: atom 8 of fragment 0 to atom 1 of fragment 1, anchors 9.42 Å
apart, exit vectors 94° apart with the first pointing squarely at its partner
(alignment 0.89) and the second obliquely (0.39). Tube clearance 3.18 Å — a
straight bridge is not in the wall. Recommended path length **6-10 atoms**, mask
window **6-17**.

That is the true attachment pair. It came first of five, and the true bridge's
7 path atoms sit inside the recommended window.

The pair below it in the ranking is instructive: atoms 6 and 6, anchors 9.79 Å,
better mutual alignment (0.81 / 0.76), but **tube clearance 1.24 Å** — a straight
bridge there is inside the protein.

**The rotatable-bond arithmetic.** A saturated chain long enough to span 9.42 Å
would cost 7 rotatable bonds, exactly the headroom, leaving nothing. Only rigid
bridges are affordable, and the library offers two that reach that far:
`phenyl_bicyclopentane` (9.33 Å, 3 RB) and `phenyl_piperazine` (10.06 Å, 3 RB).

## Step 2: candidates and verification

```
fragment_link(fragment_sdf_refs=[frag0, frag1], pocket_pdb_ref=pocket)
```

Ten candidates, two per pair across the top five pairs. Three verify — both
candidates on the true pair, and (as a control) the true ligand itself:

| molecule | strain | protein contact | verified |
| --- | --- | --- | --- |
| true ligand (sorafenib) | 1.3 | 2.64 Å | yes |
| `phenyl_piperazine` bridge on the true pair | 2.8 | 2.24 Å | yes |
| `phenyl_bicyclopentane` bridge on the true pair | 4.7 | 2.20 Å | yes |
| `phenyl_bicyclopentane` on pair 2 | 27.7 | 1.28 Å | no — 2 atoms in the wall |
| `phenyl_piperazine` on pair 2 | 11.6 | 1.12 Å | no — 4 atoms in the wall |
| `bis_alkynyl_phenyl` on pair 5 | 36.2 | 1.09 Å | no — 2 atoms in the wall |

Read the pattern rather than the rows. **The exit-vector pair decides the
outcome, not the linker.** Both library bridges work on the true pair and both
fail on the runner-up, and they fail for the reason the geometry predicted: the
tube through that pair is 1.24 Å wide. Strain tracks the same thing — single
digits on the good pair, tens on the bad one, because holding the fragments in
place while threading a bridge through the protein is what costs energy.

The seventh row is worth noting too: `bis_alkynyl_phenyl` was the zero-RB entry,
the cheapest bridge in the library, and it still failed. Cheap in rotatable bonds
does not mean geometrically right.

## Step 3: the property panel

```
molecule_gates(smiles=[...])
```

| molecule | MW | logP | RB | verdict |
| --- | --- | --- | --- | --- |
| `phenyl_piperazine` bridge | 490.9 | 4.90 | 5 | passes |
| `phenyl_bicyclopentane` bridge | 472.9 | 6.03 | 5 | fails logP |
| true ligand (sorafenib) | 464.8 | 5.55 | 5 | fails logP |

Note where that leaves you. Both surviving candidates hold the poses; one of them
also clears the panel, and **the crystal ligand does not** — sorafenib is a real
drug that sits outside the default logP bound. `fragment_link` ranks gate-passers
first, so the piperazine bridge comes out on top, which is the right answer for a
task scored on the panel even though it is not the molecule nature chose.

This is also the case for reading the panel failure correctly. The bicyclopentane
bridge is 6.03 logP against a bound of 5. That is not a geometry problem and
shortening the linker will not fix it: the remedy is a more polar bridge of the
same span, which is exactly what the piperazine is.

## Step 4: rescore and confirm

```
fragment_link(..., candidate_smiles=[the two survivors], rescore_with_vina=true)
boltz2(... the single best ...)
```

Local Vina minimisation from each verified pose gives a comparable affinity
without moving the pose. Then one co-fold, on one molecule. If it comes back with
a fragment more than 2 Å out, the pair is what to revisit — not the linker.

## What this case teaches

1. The top-ranked pair was right, and the numbers said why before any molecule
   existed: alignment high, tube clear.
2. Failures clustered by pair, not by linker. Choosing where to attach is the
   decision that matters.
3. The library alone produced a passing molecule. No generative model was
   involved, and it took milliseconds.
4. The crystal ligand is not automatically the target. It failed the panel; a
   library bridge did not.
