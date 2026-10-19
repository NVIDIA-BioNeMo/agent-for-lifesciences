# Reference: report template and property interpretation

This file is Level 3 material -- load it only when you are ready to write the
final property report.

## Report template

```
**<Preferred name>** (PubChem CID <cid>)

- Formula: <molecular_formula>
- Molecular weight: <mw> g/mol
- Canonical SMILES: <smiles>
- XLogP: <xlogp>
- H-bond donors / acceptors: <hbd> / <hba>

<one-sentence interpretation>
```

Only include a property line when the value is available; drop the line
otherwise rather than printing "N/A".

## Interpreting the numbers

- **Molecular weight**: > 500 g/mol trends against oral bioavailability
  (Lipinski). Note this only if the user cares about drug-likeness.
- **XLogP**: a rough lipophilicity estimate. > 5 is quite lipophilic;
  < 0 is hydrophilic.
- **H-bond donors/acceptors**: Lipinski's rule-of-five flags > 5 donors or
  > 10 acceptors. Mention only when relevant to a drug-likeness question.

## Common pitfalls

- Salts and hydrates: PubChem may return a salt form. If the user asked about
  the free base/acid, say which form you reported.
- Tautomers and stereochemistry: the canonical SMILES may not capture the
  stereochemistry the user has in mind; flag this if it matters.
