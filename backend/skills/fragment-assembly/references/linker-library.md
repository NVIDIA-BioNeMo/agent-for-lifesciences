# The measured linker library

Every number here was measured, not asserted. Each entry is capped with a
benzene ring at both ends, embedded with ETKDG (4 conformers for a rigid entry,
24 for a flexible one), minimised with MMFF, and then two things are read off:

- **span** — the distance between the two cap carbons, minimum and maximum over
  the conformers. This is the *anchor-to-anchor* distance, which is exactly what
  `fragment_exit_vectors` reports for a pair of fragment sites, so the two
  numbers compare directly.
- **RB** — rotatable bonds of the capped molecule. Phenyl caps rather than
  methyl caps, because RDKit does not count a bond to a terminal atom: a
  methyl-capped chain under-reports its own two attachment bonds, and those are
  precisely the bonds that Table 6 charges you for.

**atoms** is the heavy atoms the linker contributes; **path** is how many of them
lie on the shortest route between the anchors. They differ for anything with a
ring, and the difference is why a mask window derived from a distance has to be
widened before it is handed to GenMol.

## The table, shortest span first

| linker | SMILES | span (Å) | RB | atoms | path | family | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `direct_bond` | `[*:1][*:2]` | 1.5-1.5 | 1 | 0 | 0 | bond | Fragments bonded directly; only reachable for anchors under ~2 Å apart |
| `ether` | `[*:1]O[*:2]` | 2.4-2.4 | 2 | 1 | 1 | chain | One-atom bridge with an acceptor; shortens the span versus CH2 |
| `methylene` | `[*:1]C[*:2]` | 2.5-2.5 | 2 | 1 | 1 | chain | Shortest flexible bridge |
| `sulfonamide` | `[*:1]S(=O)(=O)N[*:2]` | 3.1-3.4 | 3 | 4 | 2 | sulfonamide | Tetrahedral sulfur, strongly polar, adds TPSA |
| `ester` | `[*:1]C(=O)O[*:2]` | 3.6-3.6 | 2 | 3 | 2 | amide | Planar but metabolically labile |
| `reverse_amide` | `[*:1]NC(=O)[*:2]` | 3.8-3.8 | 2 | 3 | 2 | amide | Amide with the carbonyl pointing the other way |
| `amide` | `[*:1]C(=O)N[*:2]` | 3.8-3.8 | 2 | 3 | 2 | amide | Planar, directional, donor plus acceptor; the workhorse bridge |
| `ethylene` | `[*:1]CC[*:2]` | 3.2-3.9 | 3 | 2 | 2 | chain | Two-carbon chain |
| `alkyne` | `[*:1]C#C[*:2]` | 4.1-4.1 | 0 | 2 | 2 | rigid_rod | Linear and rigid; spends no rotatable bonds on the span itself |
| `oxadiazole` | `[*:1]c1nnc([*:2])o1` | 4.8-4.8 | 2 | 5 | 3 | heteroaryl | Rigid amide bioisostere |
| `triazole` | `[*:1]c1cn([*:2])nn1` | 5.0-5.0 | 2 | 5 | 3 | heteroaryl | Click chemistry product; rigid, planar, three acceptors |
| `urea` | `[*:1]NC(=O)N[*:2]` | 5.0-5.0 | 2 | 4 | 3 | amide | Two donors and an acceptor, planar |
| `bicyclopentane_1_3` | `[*:1]C12CC([*:2])(C1)C2` | 5.0-5.0 | 2 | 5 | 3 | saturated_ring | Bicyclo[1.1.1]pentane; para-phenyl span with no aromatic ring |
| `phenyl_1_3` | `[*:1]c1cccc([*:2])c1` | 5.0-5.0 | 2 | 6 | 3 | aryl | Meta-phenylene; bends the span by about 60 degrees |
| `propylene` | `[*:1]CCC[*:2]` | 3.8-5.0 | 4 | 3 | 3 | chain | Three-carbon chain |
| `piperidine_1_4` | `[*:1]N1CCC([*:2])CC1` | 5.6-5.6 | 2 | 6 | 4 | saturated_ring | One basic nitrogen, sp3 exit at the 4-position |
| `pyrimidine_2_5` | `[*:1]c1cnc([*:2])nc1` | 5.7-5.7 | 2 | 6 | 4 | heteroaryl | Two acceptors, rigid; lowers logP versus phenyl |
| `piperazine_1_4` | `[*:1]N1CCN([*:2])CC1` | 5.7-5.7 | 2 | 6 | 4 | saturated_ring | Chair-locked span with a basic nitrogen; common solubilising bridge |
| `pyridine_2_5` | `[*:1]c1ccc([*:2])nc1` | 5.7-5.7 | 2 | 6 | 4 | heteroaryl | Para-phenylene span with an acceptor nitrogen |
| `phenyl_1_4` | `[*:1]c1ccc([*:2])cc1` | 5.8-5.8 | 2 | 6 | 4 | aryl | Para-phenylene; the canonical rigid span |
| `cyclohexane_1_4` | `[*:1]C1CCC([*:2])CC1` | 5.2-5.9 | 2 | 6 | 4 | saturated_ring | Aliphatic phenyl bioisostere, trans-diequatorial span |
| `peg2` | `[*:1]OCCO[*:2]` | 4.9-6.0 | 5 | 4 | 4 | chain | Flexible and solubilising; expensive in rotatable bonds |
| `butylene` | `[*:1]CCCC[*:2]` | 4.9-6.3 | 5 | 4 | 4 | chain | Four-carbon chain; four rotatable bonds is most of a tight budget |
| `phenyl_ether` | `[*:1]c1ccc(O[*:2])cc1` | 6.5-6.5 | 3 | 7 | 5 | composite | Aryl ether; one rotatable bond buys a long span |
| `phenyl_methylene` | `[*:1]c1ccc(C[*:2])cc1` | 6.6-6.6 | 3 | 7 | 5 | composite | Benzyl bridge; one flexible joint at the far end |
| `alkynyl_bicyclopentane` | `[*:1]C#CC12CC([*:2])(C1)C2` | 7.5-7.5 | 1 | 7 | 5 | rigid_rod | Alkyne plus a bicyclopentane; rigid, aliphatic, near 7.5 Å |
| `phenyl_amide` | `[*:1]c1ccc(C(=O)N[*:2])cc1` | 7.9-7.9 | 3 | 9 | 6 | composite | Anilide with the ring on the other side |
| `amide_phenyl` | `[*:1]C(=O)Nc1ccc([*:2])cc1` | 8.1-8.1 | 3 | 9 | 6 | composite | Anilide; long, rigid, directional |
| `alkynyl_phenyl` | `[*:1]C#Cc1ccc([*:2])cc1` | 8.4-8.4 | 1 | 8 | 6 | rigid_rod | Rigid rod near 8 Å; for spans no chain can reach inside the RB budget |
| `phenyl_bicyclopentane` | `[*:1]c1ccc(C23CC([*:2])(C2)C3)cc1` | 9.3-9.3 | 3 | 11 | 7 | rigid_rod | Fills the 9-10 Å band that no chain can reach affordably |
| `phenyl_piperazine` | `[*:1]c1ccc(N2CCN([*:2])CC2)cc1` | 10.1-10.1 | 3 | 12 | 8 | composite | Aryl-piperazine; long, soluble, only one rotatable joint |
| `biphenyl` | `[*:1]c1ccc(-c2ccc([*:2])cc2)cc1` | 10.1-10.1 | 3 | 12 | 8 | rigid_rod | Rigid rod near 10 Å; heavy, and pushes MW and logP |
| `bis_alkynyl_phenyl` | `[*:1]C#Cc1ccc(C#C[*:2])cc1` | 10.9-10.9 | 0 | 10 | 8 | rigid_rod | Longest rigid entry, near 11 Å, at zero rotatable-bond cost |
| `phenyl_amide_phenyl` | `[*:1]c1ccc(C(=O)Nc2ccc([*:2])cc2)cc1` | 12.2-12.2 | 4 | 15 | 10 | composite | Twelve-atom directional span; check molecular weight before using |

`fragment_exit_vectors` and `fragment_link` already filter this table by span and
by your remaining rotatable-bond budget, and return one representative per
family. The table is here for the cases where you want to argue with their
choice.

## What the table is for

**A rigid entry costs less per ångström than a chain.** Look at the 8 Å band:
`alkynyl_phenyl` reaches it for 1 rotatable bond, while no chain reaches it at
all — `butylene` tops out near 6.3 Å for 5 bonds. That is the whole
rotatable-bond argument in two rows, and it is why the passing answer in this
task is nearly always rigid.

**The zero-cost entries are `alkyne` and `bis_alkynyl_phenyl`.** Between them
they cover 4.1 Å and 10.9 Å at no rotatable-bond cost at all. If the budget is
genuinely exhausted, start there.

**Span ranges are narrow for rigid entries and wide for flexible ones**, and that
is real rather than a sampling artefact: a para-phenylene holds 5.8 Å and nothing
else, while `butylene` can be anywhere from 4.9 to 6.3 Å depending on how it
coils. A flexible entry that *can* reach your span will also spend entropy
getting there, which shows up downstream as strain.

**Angle matters as much as length.** `phenyl_1_3` and `bicyclopentane_1_3` both
span 5.0 Å, but the meta-phenylene bends the axis about 60°. When the exit
vectors point at each other, take the linear entry; when they are splayed, the
bent one is what fits.

**Composite entries push molecular weight.** `phenyl_amide_phenyl` is 15 heavy
atoms. Two fragments of 12 atoms each plus that bridge is already near 500 Da
before substituents, so check `molecule_gates` before committing to the long end
of the table.

## Adding your own bridge

Nothing stops you writing candidate SMILES by hand and passing them to
`fragment_link` as `candidate_smiles` — the verification does not care where a
molecule came from. The library is the fast, measured floor, not a fence.
