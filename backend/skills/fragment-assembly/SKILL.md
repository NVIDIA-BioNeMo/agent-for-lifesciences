---
name: Fragment Assembly
description: Join two posed fragments into one molecule that keeps them where they sit, or grow a single posed fragment into open pocket space. Use when the task supplies fragment SDFs with coordinates plus a pocket and asks for a single molecule containing them -- "link these fragments", "connect fragment_0 and fragment_1", "grow this fragment", fragment linking, fragment merging, or scaffold assembly from crystallographic fragment hits.
allowed_tools:
  - fragment_exit_vectors
  - fragment_link
  - genmol_generate_molecules
  - molecule_gates
  - probe_grid
  - python_execute
  - vina_dock
  - boltz2
---

# Fragment Assembly

You are given fragment SDFs that carry coordinates and a pocket they sit in. You
have to return one molecule that contains every fragment, holds each of them in
three dimensions roughly where it was handed to you, and binds the target at
least as well as the reference ligand the task was built from.

Those are three gates, and they fail in opposite directions. A molecule
containing both fragments is easy and almost always wrong on the second: a
linker one bond too short pulls the fragments together; three bonds too long
lets them drift; the right length hung off the wrong carbon points the bridge
into the protein wall. All three are valid molecules, all three contain both
fragments as disjoint substructures, and all three fail.

The third gate is the one that is quietly lost. A fragment is a starting point,
not an answer -- it binds weakly by definition, which is why it was handed to
you as a fragment. Returning it barely decorated satisfies every gate you can
check offline and fails the one that decides the task.

`fragment_link` answers the geometry question offline, before any oracle is
spent. Use it. Do not submit a molecule you have not put through it.

## Most of these tasks are growth, not linking

One fragment and a pocket is the common case, and it is a different problem: no
bridge to size, no exit-vector pair to choose, nothing to hold together. What
you are being asked is how much molecule the pocket will accept and where to put
it. Read the input first and pick the workflow to match -- the linking machinery
below has nothing to say about a single fragment, and reaching for it anyway is
how a growth task ends as a decorated fragment.

## Workflow

Two fragments, the common case:

```
1. fragment_link(fragment_sdf_refs=[...], pocket_pdb_ref=...)
       -- with no candidates it picks the exit vectors, sizes the bridge,
          builds candidates from the measured linker library and verifies them
2. read the survivors: strain, protein contact, Table 6 verdict
3. fragment_link(..., candidate_smiles=[...], rescore_with_vina=true)
       -- your own edits, once you know what the geometry wants
4. boltz2 on the single best survivor, with the pocket constraint
```

When the library has nothing that spans the gap, add the generator:

```
1. fragment_exit_vectors(fragment_sdf_refs=[...], pocket_pdb_ref=...)
       -- ranked pairs, the linker atom-count window, a safe_ref per pair
2. genmol_generate_molecules(safe_ref=<the best pair's template>)
3. fragment_link(..., candidate_csv_ref=<from step 2>)
```

or, equivalently and in one call, `fragment_link(..., generate=true)`.

One fragment is a growth problem, not a linking one:

```
1. probe_grid(receptor_pdb_ref=..., center=<pocket centre>)
       -- where a hydrophobe, a donor and an acceptor would each want to sit,
          and how much unfilled volume there is to aim at
2. fragment_exit_vectors(fragment_sdf_refs=[one], pocket_pdb_ref=...,
                         target_points=<the best minima>)
3. genmol_generate_molecules on the growth template -- ask for substituents that
       reach the minima from step 1, not for the smallest thing that attaches
4. fragment_link to verify the fragment stays put, then molecule_gates
5. boltz2 on the best few, and read the affinity, not just the probability
6. grow again toward whatever probe minima are still unoccupied, until the
       affinity stops improving or a gate starts failing
```

Step 6 is the loop, and skipping it is the single most common way this task is
lost. One pass of growth almost never reaches the reference ligand's potency.

With hundreds of candidates -- a generated library, a systematic scan -- run the
loop in `python_execute` with `aevom.constrained_pose()` instead, so the rejects
never reach this conversation.

## Reading the numbers

**RMSD is not a result.** The fragment atoms were pinned to their given
coordinates before minimisation, so a fit near zero says the constraint was
applied, nothing more. Three things carry signal:

- **Embedding failure.** A candidate that cannot be built in 3D at all is the
  clearest answer you will get: that topology cannot bridge those poses. Do not
  retry it with a different seed; change the linker.
- **Strain**, in kcal/mol above the molecule's own free minimum. This is what
  holding the fragments in place costs. Tens are normal; a few hundred means the
  molecule is being bent into a shape it does not want, and a co-fold will let
  it relax and move the fragments.
- **Closest protein contact.** Under about 2 Å the linker is inside the wall.
  The remedy is a different exit-vector pair or a bridge with a different shape,
  not a shorter one.

**A property failure and a geometry failure need opposite fixes.** A candidate
that holds its poses but breaks Table 6 is geometrically right and chemically
wrong -- change what the linker is made of, not how long it is. Shortening it
will break the geometry too.

## Potency is a stated threshold, so treat it as one

The task states the affinity you have to beat, as a line like
`affinity_pred_value < -1.34 (lower is better)`. It is a hard target with a
number on it, exactly like the weight ceiling, and it is the one most often left
unmet while every other gate reads green. Quote it to yourself early and check
each candidate against it, or the gates you can compute offline will crowd it
out.

**Use the headroom.** The property limits are the shape of the answer. A
molecular-weight ceiling near 600 with a fragment near 200 is telling you that
roughly two thirds of the molecule is still yours to build; an answer near 250 Da
has used almost none of it and will lose on affinity no matter how well the
fragment sits. Rotatable bonds, TPSA and logP are the real budget -- spend them
close to their limits rather than banking them. A molecule that passes every gate
with room to spare has usually left its potency on the table.

**Read the affinity, not the probability.** `boltz2` reports
`binding_probability` alongside `pic50` and `log_ic50_um`. The probability
saturates -- it will read comfortably above 0.7 for a weak binder that fills the
pocket at all -- so it tells you the molecule is in the right place, not that it
is good. `log_ic50_um` is the number the threshold is written in; drive it down.
Doing so is iterative: co-fold, keep what helped, grow again. A single co-fold
reports a number; three tell you which direction is working.

**Judge the pool, not just the pick.** Choosing the best of several candidates
is worth nothing if all of them are fragment-sized -- the winner of that
comparison still misses the affinity threshold by a log unit or more. Before
selecting, check that at least one candidate is a serious attempt at the
threshold. If none is, the answer is to build better candidates, not to rank the
ones you have.

Growth costs pose fidelity if you are careless, so check both together. A
substituent that improves affinity while pulling the fragment out of position has
traded one gate for another.

## The rotatable-bond gate decides most of these

Fragment pairs in this task sit at least 6 Å apart. Spanning that with an alkyl
chain costs five or six rotatable bonds against a budget of ten, and the
fragments themselves have usually spent several already. The passing answer is
almost always rigid: an alkyne, a para-phenylene, an amide, a bicyclopentane, a
biphenyl. `references/linker-library.md` is the measured table -- span,
rotatable-bond cost and path length for every entry -- and both tools rank by
that cost, so take what they offer before inventing a chain.

## Strategy

*This section is the decision procedure. Follow it in order.*

1. **Measure before you enumerate.** Get the exit-vector geometry first. The
   anchor distance, the angle between the vectors, and the clearance of the tube
   between them determine which bridges are even possible. Enumerating before
   measuring produces candidates that fail for a reason you could have known.

2. **Commit to one exit-vector pair, but check the runner-up before you spend an
   oracle.** The top-ranked pair is right most of the time, not always. If
   nothing verifies against it, the pair is the thing to change -- re-running the
   same pair with more candidates repeats a mistake more expensively.

3. **Try the library before the generator.** It costs nothing, answers in
   milliseconds, and every entry is a motif a medicinal chemist would recognise
   with a span that was measured rather than asserted. Reach for GenMol when the
   span falls between library entries, when the geometry needs a shape the
   library does not carry, or when the library's candidates all fail Table 6.

4. **Widen the mask before you widen the search.** If generated molecules keep
   failing the pose check on length, the atom-count window is the wrong window --
   move it, do not generate more molecules inside it.

5. **Rescore locally, never globally.** The pose you built is the answer being
   tested. A global docking search will find a different pose, score that, and
   report it as though it were yours. `rescore_with_vina=true` does local
   minimisation from the pose in hand, which is the question you are asking.

6. **Spend co-folds on survivors, and spend more than one.** The oracle is the
   only thing that settles the pose and affinity gates, and it is budgeted at
   several calls rather than one. A candidate that fails strain, clash or Table 6
   offline will not pass it, so never spend one on those. But do not stop at a
   single call either: the first co-fold tells you where you stand and the next
   two tell you whether growing helped. If a fragment comes back out of pose, go
   back to step 2 rather than co-folding the next candidate down.

7. **Stop on the affinity, not on the checklist.** Every offline gate passing is
   what a decorated fragment looks like, so it is not a reason to submit. Stop
   when growth has stopped improving `pic50`, when the property budget is
   genuinely spent, or when the co-fold budget is out -- whichever comes first.

8. **Submit what you verified.** The most common recorded failure in this task is
   not chemistry: it is enumerating something that works and then submitting
   something else. Every survivor is written to the candidate ledger with its
   numbers; use `best_candidate` to pick, and state the SMILES you are
   submitting alongside the numbers that justified it.

## Hard rules

- Never submit a molecule that has not been through `fragment_link`.
- Never report an RMSD near zero as evidence the molecule is good.
- Never submit a fragment-sized molecule. If your answer's weight is close to the
  fragments you were given and far below the ceiling you were allowed, you have
  not done the task yet.
- Never treat a high `binding_probability` as potency. Quote `pic50`.
- Never pass the fragment SDF or the pose SDF as text into a tool call; these
  tools exchange references.
- If no candidate verifies, say so and say which gate they failed. A reported
  dead end with a diagnosis is worth more than a molecule that passes gates 1-4
  and will fail the co-fold.

## References

Load these when you need them, with
`load_skill_resource(skill_id="fragment-assembly", path="...")`:

- `references/linker-library.md` -- the measured library: span range, rotatable-
  bond cost, heavy atoms and path length for every entry, and how to read it.
- `references/geometry-cookbook.md` -- what the geometry numbers mean, how the
  atom-count window is derived, mask sizing for GenMol, and the failure modes
  each descriptor predicts.
- `references/worked-example.md` -- one case end to end, with the actual calls
  and the numbers they returned.
