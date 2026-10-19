# Reading the geometry

What `fragment_exit_vectors` computes, what each number predicts, and how the
atom-count windows are derived. Read this when a pair's numbers are not obviously
good or obviously bad.

## Substitutable sites

A **site** is a heavy atom carrying a hydrogen — the position where a substituent
could be attached. The fragments arrive valence-complete with no marked
attachment points, so choosing the site is part of the problem, not given.

Each site has an **outward vector**: the direction a substituent would grow,
taken from the hydrogen's position when the SDF carries explicit hydrogens and
from the heavy-neighbour centroid otherwise. Both give the same answer for a
well-formed fragment.

**Free depth** is how far that vector travels before a protein heavy atom comes
within 3.0 Å. It is a ray march at 0.25 Å steps, capped at 8 Å. This is the
number that separates a real growth vector from one that only looks exposed in
two dimensions: a site pointing straight into the wall has a free depth near
zero and cannot carry anything, however unhindered the SMILES makes it appear.

A site with zero free depth scores zero. That is deliberate — no weighting of
the other terms should rescue a direction that is blocked.

## Pair descriptors

For two sites on different fragments:

- **anchor distance** — between the two heavy atoms. This is what sizes the
  linker, and what the library table's span column is measured against.
- **exit-vector angle** — between the two outward vectors. Near 180° the vectors
  point away from each other and no bridge is comfortable; small angles with good
  alignment mean the two sites face one another.
- **alignment** — how squarely each vector points at its partner, as a cosine.
  Two sites can be the right distance apart and still be the wrong pair if both
  vectors point sideways.
- **tube clearance** — the closest protein approach along the straight segment
  between the anchors, sampled at 0.25 Å steps with 1.5 Å skipped at each end.
  The ends are skipped because the anchors are fragment atoms already in van der
  Waals contact with the protein; including them would report every buried pair
  as blocked. Low clearance means a straight bridge is inside the wall and the
  linker has to go around, which costs atoms and usually rotatable bonds.

The pair score combines facing, proximity, clearance and depth. It exists only to
order pairs so the true attachment pair of a known complex lands near the top;
nothing downstream reads the number. On 46 PDBBind complexes decomposed the way
SMDD-Bench builds its tasks, the true pair is in the top 5 in 93% of them and the
top 10 in 98%. **So the top pair is usually right and occasionally not.** If
nothing verifies against it, try the next one rather than enumerating harder
against the first.

## Sizing the linker

A bridge of `n` heavy atoms makes `n + 1` bonds, each contributing between about
1.1 Å (a zig-zag chain projected onto the axis) and 1.5 Å (extended chain or
rigid rod). With a one-atom cushion on each side:

```
n_min = ceil(d / 1.5) - 1        n_max = floor(d / 1.1) + 2
```

The cushions are load-bearing. Real crystal linkers in those 46 complexes span
0.77 to 1.26 Å per bond, median 1.06 — more coiled than the taut estimate. The
window above corresponds to roughly 0.8 to 1.35 Å per bond, and every one of the
46 true linker lengths falls inside it.

**This window counts atoms on the path**, not atoms in the bridge. A
para-phenylene puts four atoms on the path and carries six.

## Sizing a GenMol mask

`[*{m-n}]` bounds the size of the *generated fragment*, which is not the number
of atoms it places between the anchors. Every ring-bearing entry in the library
carries between 1.5 and 1.7 times as many atoms as it puts on the path. So the
mask ceiling is the path ceiling scaled:

```
mask_min = n_min                 mask_max = ceil(1.7 * n_max)
```

Taking the mask straight from the path window forbids exactly the rigid bridges a
long span needs — the ceiling would exclude a phenylene while asking for a
phenylene-length span. The tools already apply this scaling; the formula is here
so a hand-written mask matches.

## Rotatable-bond headroom

Table 6 caps the whole molecule at 10 rotatable bonds. The fragments have already
spent some, and joining them adds two more bonds before any linker exists:

```
headroom = 10 - RB(fragments) - 2
```

A saturated chain of `n` atoms costs `n + 1`. At `n = 6` that is 7 bonds, which
for most fragment pairs is more than the headroom — so a chain is not merely
inelegant, it is arithmetically excluded, and the tool says so. That is the
moment to take a rigid entry from the library table.

## Failure modes and what each one means

| what you see | what it means | what to change |
| --- | --- | --- |
| rejected as "too short to span these poses", with an atom count | the bridge cannot reach, by arithmetic over the bond path — no embedding was attempted | lengthen the bridge by at least the atoms it names |
| embedding fails for every candidate | no molecule of that topology can bridge those poses | the pair, or the length window |
| embedding succeeds, strain in the hundreds | the bridge is being bent to reach | a longer or more flexible linker, or a different pair |
| closest protein contact under 2 Å | the linker is inside the wall | a bent bridge, or a pair with better tube clearance |
| everything verifies but fails Table 6 | geometry is right, chemistry is wrong | what the linker is made of — never its length |
| the true-looking candidate fails only on rotatable bonds | you picked a chain | a rigid entry of the same span |

## One fragment

There is no pair, so there is no anchor distance and no tube. Sites are ranked by
free depth alone, unless you pass `target_points` — typically `probe_grid`
minima — in which case a vector pointing at a hotspot outranks one pointing at
empty space. The same mask machinery applies with the mask at the end of the
template rather than between two fragments.
