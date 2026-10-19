# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Join posed fragments into whole molecules that keep the poses.

The task this exists for is SMDD-Bench's Fragment Assembly, the floor of that
benchmark: GPT-5.4 solved 1 of 60, and four of seven models solved none. The
failure is specific and it is not a chemistry failure -- agents never enumerate a
molecule that passes at all. Five gates decide it, and the one that bites is the
last: after co-folding, every fragment has to sit within 2.0 A of the pose it was
handed. Topological validity is cheap and most candidates have it. A linker one
bond too short, three bonds too long, or hung off the wrong carbon passes gates 1
to 4 and fails gate 5 -- and the oracle that says so costs a co-fold each time.

So this tool runs the loop that answers the geometry question offline:

    exit-vector geometry  ->  candidates  ->  constrained pose  ->  Table 6
                                          ->  local Vina rescoring

Candidates come from wherever the caller has them: GenMol filling a mask this
tool placed, a measured linker library needing no model at all, or a list the
model wrote itself. Every one of them is embedded with its fragment atoms pinned
to the coordinates they arrived with, minimised, and measured -- fit residual,
strain, closest approach to the protein. What comes back is the survivors, in
order, with their poses stored as references and a row in the candidate ledger.

The loop is deterministic, which is why it is a tool and not a subgraph: each
step's output fully determines the next step's input, so a step handed to the
model is a step it can only get wrong. The judgment above the loop -- which pair
to commit to, when to widen the mask, when to spend a Boltz-2 call -- stays with
the model.
"""

import logging
import os
from typing import Any, Optional

from api.storage import (
    ArtifactNotFoundError,
    InvalidReferenceError,
    format_storage_message,
)

from .base import ContainerTool, DataIOCategory, ToolResult
from .candidates import store_candidate
from .genmol import generate_from_safe

logger = logging.getLogger(__name__)

MAX_FRAGMENTS = 4
MAX_CANDIDATES = 200

# The service chunks nothing, so a batch is one HTTP call and the timeout has to
# cover every candidate in it: a restrained embedding plus two minimisations,
# tens of milliseconds to a second each, times four seeds.
VERIFY_TIMEOUT = 600.0
VERIFY_CHUNK = 32
GEOMETRY_TIMEOUT = 120.0
DOCK_TIMEOUT = 600.0

DEFAULT_GENERATE = 60
MAX_GENERATE = 400

# Poses stored as references, best first. Every survivor's numbers are reported;
# this is how many get an SDF the model can dock, view or hand to Boltz-2.
DEFAULT_STORE_POSES = 5


class FragmentLinkTool(ContainerTool):
    """Build and verify molecules that hold their fragments in place.

    Data Flow:
        Input: fragment SDF refs, pocket PDB ref, and either candidate SMILES or
            a request to generate them
        Output: stores each surviving pose as an ``sdf`` reference, writes the
            candidate ledger, returns the ranked table inline
    """

    @property
    def name(self) -> str:
        return "fragment_link"

    @property
    def summary_template(self) -> str:
        return "Linking fragments..."

    def format_summary(self, arguments: dict) -> str:
        given = len(arguments.get("candidate_smiles") or [])
        if given:
            return f"Verifying {given} linked candidates against the fragment poses..."
        return "Generating and verifying fragment linkers..."

    @property
    def service_url(self) -> str:
        return os.environ.get("RDKIT_SERVICE_URL", "http://rdkit:8000")

    @property
    def vina_service_url(self) -> str:
        return os.environ.get("VINA_SERVICE_URL", "http://vina:8000")

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.BOTH

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "fragment_link",
                "description": (
                    "Join fragments into whole molecules that keep the fragments "
                    "where they were posed, and verify that they do.\n\n"
                    "Give it the fragment SDFs and the pocket, and either your "
                    "own candidate SMILES or nothing -- with no candidates it "
                    "picks exit vectors, sizes the linker and enumerates them "
                    "itself, from the measured linker library and, if asked, from "
                    "GenMol.\n\n"
                    "Every candidate is embedded with its fragment atoms pinned "
                    "to the coordinates they arrived with, minimised with those "
                    "atoms fixed, and measured three ways: how far each fragment "
                    "ended up from its given pose, how much strain holding them "
                    "there costs, and how close the molecule comes to the "
                    "protein. Candidates that cannot be built in three dimensions "
                    "at all are rejected outright -- that is the cheapest and most "
                    "informative answer there is.\n\n"
                    "WHEN TO USE THIS: any task that hands you posed fragments "
                    "and asks for one molecule. It is the difference between "
                    "guessing and knowing, and it costs no oracle calls.\n\n"
                    "WHEN NOT TO USE THIS: to dock a molecule with no fragments "
                    "to hold -- that is vina_dock. To decorate a scaffold whose "
                    "pose does not matter -- that is genmol_generate_molecules.\n\n"
                    "OUTPUT: the survivors in order, each with its fit to the "
                    "given poses, its strain, its closest protein contact, its "
                    "Table 6 verdict and (if asked) a Vina score from local "
                    "minimisation of the pose just built. Poses are stored as "
                    "`sdf` references. Rejected candidates are reported with the "
                    "gate they failed and why, because that is what tells you "
                    "whether to change the linker length or the exit vector.\n\n"
                    "AN RMSD NEAR ZERO IS BY CONSTRUCTION, not a result: the "
                    "fragments were pinned there. The signals are strain, protein "
                    "contact, and whether the embedding succeeded at all."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fragment_sdf_refs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": MAX_FRAGMENTS,
                            "description": (
                                "References to the fragment SDFs, each carrying "
                                "the pose that has to be kept."
                            ),
                            "x-aevom-reference": {"data_type": "sdf"},
                        },
                        "pocket_pdb_ref": {
                            "type": "string",
                            "description": (
                                "Reference to the pocket PDB. Without it no clash "
                                "check runs and a molecule threaded through the "
                                "protein wall will verify."
                            ),
                            "x-aevom-reference": {"data_type": "pdb"},
                        },
                        "candidate_smiles": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_CANDIDATES,
                            "description": (
                                "Molecules to verify. Omit to have the tool "
                                "enumerate its own from the linker library."
                            ),
                        },
                        "candidate_csv_ref": {
                            "type": "string",
                            "description": (
                                "Reference to a CSV with a `smiles` column, such "
                                "as the one genmol_generate_molecules stores. Its "
                                "molecules are verified alongside candidate_smiles."
                            ),
                            "x-aevom-reference": {"data_type": "csv"},
                        },
                        "generate": {
                            "type": "boolean",
                            "description": (
                                "Also fill the best pair's mask with GenMol. "
                                "Default false; the linker library needs no model "
                                "and is the sensible first pass."
                            ),
                        },
                        "num_generate": {
                            "type": "integer",
                            "description": (
                                f"Molecules to ask GenMol for (default "
                                f"{DEFAULT_GENERATE}, max {MAX_GENERATE}). Only "
                                "used when generate is true."
                            ),
                        },
                        "rb_max": {
                            "type": "integer",
                            "description": (
                                "Rotatable-bond ceiling, default 10 (Table 6). "
                                "Used both to size linkers and to gate survivors."
                            ),
                        },
                        "strain_max": {
                            "type": "number",
                            "description": (
                                "Strain ceiling in kcal/mol above the molecule's "
                                "own free minimum. Defaults to the calibrated "
                                "value; raise it to see near misses."
                            ),
                        },
                        "rmsd_max": {
                            "type": "number",
                            "description": (
                                "How far a fragment may end up from its given "
                                "pose, in angstrom (default 1.0). The benchmark "
                                "gate is 2.0 after co-folding, so this is the "
                                "stricter offline proxy."
                            ),
                        },
                        "rescore_with_vina": {
                            "type": "boolean",
                            "description": (
                                "Locally minimise each surviving pose with Vina "
                                "and report its affinity. Default false. Costs a "
                                "few seconds per survivor and is worth it once "
                                "the list is short."
                            ),
                        },
                        "max_survivors": {
                            "type": "integer",
                            "description": (
                                f"How many survivors get a stored pose (default "
                                f"{DEFAULT_STORE_POSES}). All of them are "
                                "reported either way."
                            ),
                        },
                    },
                    "required": ["fragment_sdf_refs"],
                },
            },
        }

    # ------------------------------------------------------------------ #
    # Inputs
    # ------------------------------------------------------------------ #

    async def _fetch(self, reference: str, kind: str) -> str:
        try:
            content = await self._storage.retrieve(reference)
        except InvalidReferenceError:
            raise ValueError(
                f"Invalid {kind} reference: '{reference}'. Expected something like "
                f"'{kind}_abc12345'."
            )
        except ArtifactNotFoundError:
            raise ValueError(
                f"No {kind} found for '{reference}'. The reference is well formed "
                "but nothing is stored under it."
            )
        return content.decode("utf-8", errors="replace")

    @staticmethod
    def _smiles_from_csv(text: str) -> list[str]:
        """The `smiles` column of a stored table, in order."""
        import csv
        import io

        rows = list(csv.DictReader(io.StringIO(text)))
        return [
            str(row["smiles"]).strip()
            for row in rows
            if row.get("smiles") and str(row["smiles"]).strip()
        ]

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #

    async def execute(self, arguments: dict, context: dict = None) -> ToolResult:
        fragment_refs = arguments.get("fragment_sdf_refs") or []
        pocket_ref = arguments.get("pocket_pdb_ref") or ""

        if not fragment_refs:
            return ToolResult.error(
                "fragment_sdf_refs is required: the fragment SDFs whose poses have "
                "to be kept."
            )
        if len(fragment_refs) > MAX_FRAGMENTS:
            return ToolResult.error(
                f"At most {MAX_FRAGMENTS} fragments; got {len(fragment_refs)}."
            )
        if self._storage is None:
            return ToolResult.error(
                "Storage service not available. Cannot retrieve the fragments."
            )

        try:
            fragment_sdfs = [
                await self._fetch(reference, "sdf") for reference in fragment_refs
            ]
            pocket_pdb = await self._fetch(pocket_ref, "pdb") if pocket_ref else ""
            csv_ref = arguments.get("candidate_csv_ref") or ""
            from_csv = (
                self._smiles_from_csv(await self._fetch(csv_ref, "csv"))
                if csv_ref
                else []
            )
        except ValueError as exc:
            return ToolResult.error(str(exc))

        rb_max = int(arguments.get("rb_max") or 10)
        notes: list[str] = []

        candidates = [
            str(smiles).strip()
            for smiles in (arguments.get("candidate_smiles") or [])
            if str(smiles).strip()
        ] + from_csv
        provenance = {smiles: "supplied" for smiles in candidates}

        # No candidates means the caller wants the tool to find its own, which is
        # the whole loop: geometry, then the library, then optionally GenMol.
        geometry: Optional[dict] = None
        if not candidates or arguments.get("generate"):
            geometry, error = await self._geometry(fragment_sdfs, pocket_pdb, rb_max)
            if error:
                return ToolResult.error(error)

        if geometry and not candidates:
            for candidate in geometry.get("candidates") or []:
                provenance.setdefault(candidate["smiles"], f"library:{candidate['linker']}")
            candidates.extend(
                candidate["smiles"] for candidate in geometry.get("candidates") or []
            )
            notes.append(
                f"{len(candidates)} candidate(s) assembled from the linker library."
            )

        if geometry and arguments.get("generate"):
            generated, note = await self._generate(geometry, arguments)
            for smiles in generated:
                provenance.setdefault(smiles, "genmol")
            candidates.extend(generated)
            notes.append(note)

        # Deduplicate while keeping the order candidates arrived in, so the
        # library's suggestions stay ahead of the generated ones.
        seen: set[str] = set()
        candidates = [
            smiles for smiles in candidates
            if not (smiles in seen or seen.add(smiles))
        ][:MAX_CANDIDATES]

        if not candidates:
            return ToolResult.error(
                "No candidates to verify. Either pass candidate_smiles, or give a "
                "pocket and two fragments the library can bridge, or set "
                "generate=true. "
                + (
                    "The geometry step found no library bridge for this span: "
                    + "; ".join(
                        warning
                        for pair in (geometry.get("pairs") or [])[:1]
                        for warning in pair.get("warnings") or []
                    )
                    if geometry
                    else ""
                )
            )

        results, error = await self._verify(
            candidates, fragment_sdfs, pocket_pdb, arguments
        )
        if error:
            return ToolResult.error(error)

        survivors = [result for result in results if result.get("verified")]
        gates = await self._gates([result["smiles"] for result in survivors], rb_max)
        survivors = self._apply_gates(survivors, gates)

        if arguments.get("rescore_with_vina") and pocket_pdb:
            await self._rescore(survivors, pocket_pdb)

        survivors.sort(key=self._rank_key)
        stored = await self._store_poses(
            survivors, int(arguments.get("max_survivors") or DEFAULT_STORE_POSES),
            fragment_refs,
        )
        await self._record(survivors, provenance)

        metadata = {
            "source": "fragment_link",
            "fragment_refs": list(fragment_refs),
            "pocket_ref": pocket_ref or None,
            "num_candidates": len(candidates),
            "num_verified": len(survivors),
            "results": results,
            "pose_refs": stored,
            "outputs": stored,
            "all_references": list(stored.values()),
            "geometry": (geometry or {}).get("pairs"),
        }
        message = self._format_message(results, survivors, stored, notes, geometry)

        if stored:
            return ToolResult.data_reference(
                reference=next(iter(stored.values())),
                data_type="sdf",
                message=message,
                metadata=metadata,
            )
        return ToolResult.text(message, metadata=metadata)

    async def _geometry(
        self, fragment_sdfs: list[str], pocket_pdb: str, rb_max: int
    ) -> tuple[Optional[dict], Optional[str]]:
        try:
            response = await self._call_service(
                endpoint="/link-geometry",
                method="POST",
                json_data={
                    "fragment_sdfs": fragment_sdfs,
                    "pocket_pdb": pocket_pdb,
                    "rb_max": rb_max,
                    "max_pairs": 3,
                    "assemble_candidates": True,
                    "max_candidates_per_pair": 8,
                },
                timeout=GEOMETRY_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
            logger.exception("Link geometry request failed")
            return None, f"Exit-vector geometry failed: {exc}"
        if not response.get("success"):
            return None, f"Exit-vector geometry failed: {response.get('error')}"
        return response, None

    async def _generate(self, geometry: dict, arguments: dict) -> tuple[list[str], str]:
        """Fill the best pair's mask with GenMol, reporting rather than raising."""
        pairs = geometry.get("pairs") or []
        template = next(
            (
                (pair.get("templates") or {}).get("safe")
                for pair in pairs
                if (pair.get("templates") or {}).get("safe")
            ),
            None,
        )
        if not template:
            return [], (
                "GenMol was not called: the geometry step could not build a SAFE "
                "template for any pair."
            )

        count = min(int(arguments.get("num_generate") or DEFAULT_GENERATE), MAX_GENERATE)
        ranked, error = await generate_from_safe(template, num_molecules=count)
        if error:
            return [], f"GenMol did not contribute: {error}"
        return [smiles for smiles, _score in ranked], (
            f"GenMol returned {len(ranked)} molecule(s) for the best pair's mask."
        )

    async def _verify(
        self,
        candidates: list[str],
        fragment_sdfs: list[str],
        pocket_pdb: str,
        arguments: dict,
    ) -> tuple[list[dict], Optional[str]]:
        """Constrained-pose verification, chunked to the service's batch limit."""
        payload: dict[str, Any] = {
            "fragment_sdfs": fragment_sdfs,
            "pocket_pdb": pocket_pdb,
            "include_poses": True,
        }
        if arguments.get("strain_max") is not None:
            payload["strain_max"] = float(arguments["strain_max"])
        if arguments.get("rmsd_max") is not None:
            payload["rmsd_max"] = float(arguments["rmsd_max"])

        results: list[dict] = []
        for start in range(0, len(candidates), VERIFY_CHUNK):
            chunk = candidates[start : start + VERIFY_CHUNK]
            try:
                response = await self._call_service(
                    endpoint="/constrained-pose",
                    method="POST",
                    json_data={**payload, "smiles": chunk},
                    timeout=VERIFY_TIMEOUT,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the model as text
                logger.exception("Constrained pose request failed")
                return [], f"Pose verification failed: {exc}"
            if not response.get("success"):
                return [], f"Pose verification failed: {response.get('error')}"
            results.extend(response.get("results") or [])
        return results, None

    async def _gates(self, smiles: list[str], rb_max: int) -> dict[str, dict]:
        """Table 6 verdicts, keyed by the SMILES they were asked about."""
        if not smiles:
            return {}
        try:
            response = await self._call_service(
                endpoint="/molecule-gates",
                method="POST",
                json_data={"smiles": smiles, "thresholds": {"rotatable_bonds": rb_max}},
                timeout=GEOMETRY_TIMEOUT,
            )
        except Exception as exc:  # noqa: BLE001 - a missing panel is not fatal
            logger.warning("Gate evaluation failed: %s", exc)
            return {}
        if not response.get("success"):
            logger.warning("Gate evaluation failed: %s", response.get("error"))
            return {}
        return {result["smiles"]: result for result in response.get("results") or []}

    @staticmethod
    def _apply_gates(survivors: list[dict], gates: dict[str, dict]) -> list[dict]:
        """Attach each survivor's property panel, keeping the ones that pass.

        A candidate that holds its poses but breaks the property panel is still
        reported -- with what it broke -- because the remedy differs: the linker
        is geometrically right and chemically wrong, so shortening it is exactly
        the wrong move.
        """
        for survivor in survivors:
            panel = gates.get(survivor["smiles"])
            if panel is None:
                continue
            survivor["properties"] = panel.get("properties")
            survivor["gates_passed"] = panel.get("passed")
            survivor["gate_failures"] = panel.get("failures") or []
        return survivors

    async def _rescore(self, survivors: list[dict], pocket_pdb: str) -> None:
        """Local Vina minimisation from each pose just built.

        Local rather than a search: the pose is the answer being tested, and a
        global search would score a different one and report it as this one's.
        """
        for survivor in survivors:
            if not survivor.get("pose_sdf"):
                continue
            try:
                response = await self._call_service_at(
                    self.vina_service_url,
                    endpoint="/dock-auto",
                    json_data={
                        "receptor_pdb": pocket_pdb,
                        "ligand_sdf": survivor["pose_sdf"],
                        **self._box(survivor["pose_sdf"]),
                        "score_mode": "local",
                    },
                    timeout=DOCK_TIMEOUT,
                )
            except Exception as exc:  # noqa: BLE001 - a score is a bonus, not the answer
                logger.warning("Vina rescoring failed: %s", exc)
                survivor["vina_error"] = str(exc)
                continue
            if response.get("success"):
                survivor["vina_affinity"] = response.get("best_affinity")
            else:
                survivor["vina_error"] = response.get("error")

    async def _call_service_at(
        self, base_url: str, endpoint: str, json_data: dict, timeout: float
    ) -> dict:
        """Call a service other than this tool's own.

        ``ContainerTool`` binds one ``service_url``; the verification lives in
        rdkit and the rescoring in vina, so the second one is reached explicitly
        rather than by giving this tool two identities.
        """
        import httpx

        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(f"{base_url}{endpoint}", json=json_data)
            response.raise_for_status()
            return response.json()

    @staticmethod
    def _box(pose_sdf: str, padding: float = 8.0) -> dict[str, float]:
        """A search box around the pose, since local mode still needs one."""
        coordinates = []
        for line in pose_sdf.splitlines():
            parts = line.split()
            # An SDF atom line is three decimals then an element symbol. The
            # counts line ("31 33 0 0 ...") also has four-plus fields and parses
            # as numbers, so the decimal point and the symbol are what tell them
            # apart.
            if len(parts) < 4 or not parts[3].isalpha():
                continue
            if not all("." in part for part in parts[:3]):
                continue
            try:
                coordinates.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue
        if not coordinates:
            return {
                "center_x": 0.0, "center_y": 0.0, "center_z": 0.0,
                "size_x": 20.0, "size_y": 20.0, "size_z": 20.0,
            }
        lows = [min(row[axis] for row in coordinates) for axis in range(3)]
        highs = [max(row[axis] for row in coordinates) for axis in range(3)]
        box = {}
        for axis, name in enumerate("xyz"):
            box[f"center_{name}"] = round((lows[axis] + highs[axis]) / 2, 3)
            box[f"size_{name}"] = round(
                min(max(highs[axis] - lows[axis] + 2 * padding, 12.0), 35.0), 3
            )
        return box

    async def _store_poses(
        self, survivors: list[dict], limit: int, fragment_refs: list[str]
    ) -> dict[str, str]:
        stored: dict[str, str] = {}
        for survivor in survivors[: max(limit, 0)]:
            if not survivor.get("pose_sdf"):
                continue
            reference = await self._storage.store(
                data_type="sdf",
                content=survivor["pose_sdf"].encode("utf-8"),
                metadata={
                    "source": "fragment_link",
                    "smiles": survivor["smiles"],
                    "max_fragment_rmsd": survivor.get("max_fragment_rmsd"),
                    "strain": survivor.get("strain"),
                    "min_protein_distance": survivor.get("min_protein_distance"),
                    "vina_affinity": survivor.get("vina_affinity"),
                    "fragment_refs": list(fragment_refs),
                },
                description=f"Constrained pose for {survivor['smiles']}",
            )
            stored[survivor["smiles"]] = reference
            survivor["pose_ref"] = reference
        return stored

    async def _record(self, survivors: list[dict], provenance: dict[str, str]) -> None:
        """File every survivor in the candidate ledger.

        Automatic because the selection failure this whole tool exists to fix is
        partly a bookkeeping failure: agents enumerate something that works and
        then submit something else.
        """
        for survivor in survivors:
            scores: dict[str, float] = {}
            for key, name in (
                ("max_fragment_rmsd", "pose_fit_rmsd_angstrom"),
                ("strain", "strain_kcal_mol"),
                ("min_protein_distance", "protein_contact_angstrom"),
                ("vina_affinity", "affinity_kcal_mol"),
            ):
                if survivor.get(key) is not None:
                    scores[name] = float(survivor[key])
            try:
                await store_candidate(
                    smiles=survivor["smiles"],
                    origin=f"fragment_link:{provenance.get(survivor['smiles'], 'unknown')}",
                    scores=scores,
                    passed=survivor.get("gates_passed"),
                    notes="holds both fragment poses in a constrained embedding",
                )
            except Exception as exc:  # noqa: BLE001 - the ledger is not the answer
                logger.warning("Could not record candidate: %s", exc)

    @staticmethod
    def _rank_key(survivor: dict) -> tuple:
        """Property panel first, then affinity, then how cheaply the pose is held.

        Affinity leads among gate-passing molecules because that is what the task
        is finally judged on; strain breaks ties, since between two equally scored
        molecules the one straining less to hold its fragments is the one more
        likely to survive a co-fold.
        """
        return (
            not survivor.get("gates_passed", True),
            survivor.get("vina_affinity") if survivor.get("vina_affinity") is not None else 0.0,
            survivor.get("strain") if survivor.get("strain") is not None else 0.0,
        )

    # ------------------------------------------------------------------ #
    # Reporting
    # ------------------------------------------------------------------ #

    @staticmethod
    def _format_message(
        results: list[dict],
        survivors: list[dict],
        stored: dict[str, str],
        notes: list[str],
        geometry: Optional[dict],
    ) -> str:
        lines: list[str] = list(notes)
        lines.append(
            f"{len(survivors)} of {len(results)} candidate(s) hold both fragment "
            "poses."
        )

        if geometry and geometry.get("pairs"):
            best = geometry["pairs"][0]
            lines.append(
                f"Best exit-vector pair: atom {best['atom_a']} to atom "
                f"{best['atom_b']}, {best['anchor_distance']} A apart, wanting "
                f"{best['linker_atoms_min']}-{best['linker_atoms_max']} atoms on the "
                f"path. Rotatable-bond headroom: {geometry.get('rb_headroom')}."
            )

        if survivors:
            lines.append("\nSurvivors, best first:")
            for survivor in survivors:
                parts = [
                    f"fit {survivor['max_fragment_rmsd']:.2f} A"
                    if survivor.get("max_fragment_rmsd") is not None
                    else "fit unknown"
                ]
                if survivor.get("strain") is not None:
                    parts.append(f"strain {survivor['strain']:.1f} kcal/mol")
                if survivor.get("min_protein_distance") is not None:
                    parts.append(
                        f"closest protein contact {survivor['min_protein_distance']:.2f} A"
                    )
                if survivor.get("vina_affinity") is not None:
                    parts.append(f"Vina {survivor['vina_affinity']:.2f} kcal/mol")
                if survivor.get("num_linker_atoms") is not None:
                    parts.append(f"{survivor['num_linker_atoms']} linker atoms")
                lines.append(f"- {survivor['smiles']}: " + ", ".join(parts))
                if survivor.get("gates_passed") is False:
                    lines.append(
                        "  Fails the property panel: "
                        + "; ".join(survivor.get("gate_failures") or [])
                        + ". The geometry is right and the chemistry is not, so "
                        "change what the linker is made of rather than how long "
                        "it is."
                    )
                if survivor.get("pose_ref"):
                    lines.append(f"  Pose stored as {survivor['pose_ref']}")

        rejected = [result for result in results if not result.get("verified")]
        if rejected:
            lines.append(f"\nRejected ({len(rejected)}):")
            by_gate: dict[str, list[dict]] = {}
            for result in rejected:
                by_gate.setdefault(result.get("failed_gate") or "unknown", []).append(result)
            for gate, group in sorted(by_gate.items()):
                lines.append(f"- {gate}: {len(group)}")
                for result in group[:3]:
                    lines.append(f"  {result['smiles']}: {result.get('reason')}")
                if len(group) > 3:
                    lines.append(f"  ... and {len(group) - 3} more")

        if stored:
            lines.append("")
            lines.append(
                format_storage_message(
                    data_type="sdf",
                    reference=next(iter(stored.values())),
                    source_description="constrained pose of the best candidate",
                    accepting_tools=["vina_dock", "plip_interactions", "boltz2"],
                )
            )
            if len(stored) > 1:
                lines.append(
                    "Other poses: "
                    + ", ".join(f"{ref} ({smiles})" for smiles, ref in list(stored.items())[1:])
                )

        lines.append(
            "\nThe fit to the given poses is near zero by construction -- the "
            "fragment atoms were pinned there -- so read the strain and the "
            "protein contact, not the fit. Survivors are in the candidate ledger; "
            "confirm the best one with a co-fold before submitting it."
        )
        return "\n".join(lines)
