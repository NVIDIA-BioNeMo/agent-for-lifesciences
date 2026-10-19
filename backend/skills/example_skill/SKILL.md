---
name: Small-Molecule Property Report
description: Produce a concise property report for a small molecule named by the user (by common name or SMILES), including identity, key physicochemical properties, and a 3D visualization. Use when the user asks to "look up", "characterize", "profile", or "report properties" for a small molecule.
allowed_tools:
  - search_pubchem
  - get_pubchem_properties
  - download_pubchem_sdf
  - visualize_sdf
---

# Small-Molecule Property Report

This skill orchestrates the built-in PubChem tools to build a short, reliable
property report for a single small molecule. Follow the steps in order and do
not skip identity resolution -- reporting properties for the wrong compound is
the most common failure mode.

## When to use

- The user names a small molecule (e.g. "aspirin", "caffeine") or provides a
  SMILES string and asks for its properties, identity, or a characterization.
- Not for proteins, complexes, or docking tasks.

## Workflow

1. **Resolve identity.** Call `search_pubchem` with the user's term to obtain
   the canonical compound (CID + preferred name). If several candidates come
   back, pick the exact-name match and state which one you chose.
2. **Fetch properties.** Call `get_pubchem_properties` for the resolved
   compound. Report at minimum: molecular formula, molecular weight, canonical
   SMILES, and (when available) XLogP and hydrogen-bond donor/acceptor counts.
3. **Get 3D coordinates and visualize.** Call `download_pubchem_sdf` to obtain
   an SDF reference, then `visualize_sdf` with that reference so the structure
   appears in the viewer. Reference the artifact by id -- never paste raw SDF
   text into your reply.
4. **Summarize.** Present a compact report: identity line, then a short bullet
   list of properties, then a one-sentence interpretation.

## Output format

See `reference.md` for the exact report template and property-interpretation
guidance. Load it with
`load_skill_resource(skill_id="example_skill", path="reference.md")` before
writing the final summary.

## Hard rules

- Always resolve identity before reporting properties.
- If identity cannot be resolved confidently, say so and ask the user to
  clarify rather than guessing.
- Keep raw structure data out of the chat; pass artifact references between
  tools.
