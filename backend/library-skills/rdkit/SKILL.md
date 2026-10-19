---
name: rdkit
description: >
  Use RDKit inside the python_execute sandbox for cheminformatics: parsing and
  canonicalising SMILES, computing physicochemical descriptors, Morgan
  fingerprints and Tanimoto similarity, substructure and SMARTS matching,
  Murcko scaffolds, BRICS fragmentation, maximum common substructure,
  pharmacophore features, structural alert catalogs, conformer generation and
  constrained embedding. Invoke for descriptor panels, similarity searches,
  scaffold analysis, fragment enumeration, SMARTS induction, or any custom
  molecule calculation. Out of scope: docking (vina_dock), interaction
  profiling (plip_interactions), protein structure parsing (biopython,
  biotite), format conversion of macromolecules (openbabel).
license: BSD-3-Clause
---

# RDKit in the Aevom sandbox

RDKit is installed in the `python_execute` sandbox and the whole package is
importable. Use it when you need intermediate values, a custom threshold, or a
calculation no tool covers.

## Running this in Aevom

Everything in this section overrides habits from RDKit documentation written for
a normal Python process.

- **There is no filesystem.** `open()` is rejected, and so is every path-based
  reader: use `Chem.MolFromPDBBlock`, `Chem.MolFromMolBlock` and
  `Chem.SDMolSupplier`-free string parsing instead of `MolFromPDBFile` or
  `SDMolSupplier`. Note that source text containing `File(` is refused outright
  by the pattern scan, so `MolFromPDBFile(` will not even compile.
- **No `os`, `sys`, `subprocess`, `requests`.** Anything reached through
  `sys.path` manipulation — most notably the SA-score contrib module — is
  unavailable. Use `aevom.sa_score(smiles)` instead.
- **Inputs arrive as references.** `aevom.load_rdkit_mol(ref)` gives a molecule
  directly; `aevom.load(ref)` gives raw text you can parse yourself.
- **`obj.attr = x` and `array[i] = x` are rejected**; only `list`, `dict` and
  `set` are writable. RDKit is barely affected because its mutators are method
  calls (`mol.SetProp`, `conf.SetAtomPosition`, `RWMol.AddBond`), but numpy
  post-processing of coordinates has to build new arrays rather than assign
  into them.
- **Assign to `result`** to return structured data, and `print()` for anything
  the user should read. The default timeout is 30 s, maximum 120 s.
- **Prefer `molecule_gates`** over hand-rolling a drug-likeness panel. Its
  thresholds are the ones your results are judged against.

## Descriptors and drug-likeness

```python
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, QED

mol = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
result = {
    "mw": Descriptors.MolWt(mol),
    "logp": Crippen.MolLogP(mol),
    "tpsa": rdMolDescriptors.CalcTPSA(mol),
    "hbd": rdMolDescriptors.CalcNumHBD(mol),
    "hba": rdMolDescriptors.CalcNumHBA(mol),
    "rotb": rdMolDescriptors.CalcNumRotatableBonds(mol),
    "qed": QED.qed(mol),
    "sa": aevom.sa_score("CC(=O)Oc1ccccc1C(=O)O"),
}
```

Always check `mol is None` before using it. `MolFromSmiles` returns `None` on a
parse failure rather than raising, so an unchecked mol turns a bad input into a
confusing `AttributeError` several lines later.

## Similarity and fingerprints

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ref = gen.GetFingerprint(Chem.MolFromSmiles("c1ccccc1C(=O)NC"))
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in candidates]
result = [round(DataStructs.TanimotoSimilarity(ref, fp), 3) for fp in fps]
```

`rdFingerprintGenerator` is the current API. `AllChem.GetMorganFingerprintAsBitVect`
still works but is deprecated and prints a warning.

## Scaffolds, fragments and common substructure

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import BRICS, rdFMCS

mol = Chem.MolFromSmiles(smiles)
scaffold = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
fragments = sorted(BRICS.BRICSDecompose(mol))

mols = [Chem.MolFromSmiles(s) for s in series]
mcs = rdFMCS.FindMCS(mols, timeout=10, completeRingsOnly=True)
result = {"scaffold": scaffold, "brics": fragments, "mcs": mcs.smartsString}
```

Always pass `timeout=` to `FindMCS`. It is worst-case exponential, and without a
bound a dozen flexible molecules can exhaust the sandbox's execution budget and
return nothing at all.

## SMARTS matching

```python
from rdkit import Chem

patt = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")   # carboxylic acid
mol = Chem.MolFromSmiles(smiles)
hits = mol.GetSubstructMatches(patt)
result = {"n_matches": len(hits), "atom_indices": [list(h) for h in hits]}
```

`MolFromSmarts` also returns `None` for an invalid pattern, which is the usual
cause of a "matches everything" or "matches nothing" surprise. Check it.

## Structural alerts

```python
from rdkit import Chem
from rdkit.Chem import FilterCatalog

params = FilterCatalog.FilterCatalogParams()
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
catalog = FilterCatalog.FilterCatalog(params)

mol = Chem.MolFromSmiles(smiles)
result = [entry.GetDescription() for entry in catalog.GetMatches(mol)]
```

## Pharmacophore features

```python
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures

factory = ChemicalFeatures.BuildFeatureFactory(
    RDConfig.RDDataDir + "/BaseFeatures.fdef"
)
mol = Chem.MolFromSmiles(smiles)
feats = factory.GetFeaturesForMol(mol)
result = [(f.GetFamily(), list(f.GetAtomIds())) for f in feats]
```

`RDConfig.RDDataDir` is a package data path, not user file access, so it is
allowed. Families are `Donor`, `Acceptor`, `Hydrophobe`, `LumpedHydrophobe`,
`Aromatic`, `PosIonizable`, `NegIonizable` and `ZnBinder`.

Two traps when running this over a **protein** rather than a small molecule:

- `MolFromPDBBlock(..., sanitize=False)` leaves ring perception uninitialised and
  the feature factory raises `Pre-condition Violation: RingInfo not initialized`.
  Either sanitize, or call `Chem.FastFindRings(mol)` first.
- With `sanitize=True`, a whole protein often fails to parse and returns `None`.
  Crop to the pocket — a shell of residues around the site of interest — and it
  parses. That is also much faster.

## Conformers and constrained embedding

```python
from rdkit import Chem
from rdkit.Chem import AllChem

mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
cids = AllChem.EmbedMultipleConfs(mol, numConfs=20, randomSeed=0xF00D)
AllChem.MMFFOptimizeMoleculeConfs(mol)
result = {"n_confs": len(cids)}
```

`AddHs` before embedding and `RemoveHs` after, or the geometry will be wrong.
Set `randomSeed` so a rerun reproduces the same conformers. For growing a
molecule while holding a known fragment fixed, `AllChem.ConstrainedEmbed(mol,
core)` keeps the core coordinates.

## Handing molecules back

```python
sdf = aevom.mol_to_sdf(mol)
result = aevom.store_sdf(sdf, {"name": "analog_3"})
```

Store rather than printing structures: a reference costs a few tokens and a
multi-molecule SDF block costs thousands.
