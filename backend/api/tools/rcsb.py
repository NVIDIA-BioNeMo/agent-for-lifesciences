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

"""RCSB PDB API tools for retrieving protein structures.

This module provides two tools for the AI agent to retrieve and prepare
protein structures from RCSB PDB:

1. search_rcsb - A proxy over the RCSB Search API that passes the agent's
   query through verbatim and enriches the hits with per-entry facts
2. fetch_and_clean_pdb - Download biological assembly and remove water molecules

The two-tool design allows the agent to reason about which structure to use
based on method, resolution, ligand presence (apo vs holo), and other factors.

The search tool applies no filters of its own. It used to hard-code
"X-RAY DIFFRACTION", resolution <= 3.0 A and organism "Homo sapiens", which
made whole classes of target invisible: TRPV1 has 84 entries in the PDB, 80 of
them cryo-EM, and those filters left 2. Deciding what counts as a usable
structure is the agent's job, so the filters are now the agent's to write --
and the injected experimental-method facet is what tells it what a narrow
filter is costing.
"""

import copy
import gzip
import io
import json
import logging
import re
import tempfile
import time
from typing import Any

import httpx
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Atom import DisorderedAtom

from api.storage import format_storage_message

from .base import BaseTool, DataIOCategory, ToolResult, ToolType

logger = logging.getLogger(__name__)

# RCSB endpoints. The Search API returns identifiers only; every descriptive
# field (title, method, resolution, ligands) comes from the Data API.
SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DATA_GRAPHQL_URL = "https://data.rcsb.org/graphql"
SEARCH_SCHEMA_URL = "https://search.rcsb.org/rcsbsearch/v2/metadata/schema"

HTTP_TIMEOUT = 30.0

# Hits returned to the model per call. RCSB itself defaults to 10 and allows
# 10,000; the ceiling here is a context budget, not an API limit. `paginate.start`
# is passed through untouched so the agent can page past it.
DEFAULT_ROWS = 10
MAX_ROWS = 25

# Enrichment is only meaningful for identifiers that name a whole entry.
ENTRY_RETURN_TYPE = "entry"

RETURN_TYPES = (
    "entry",
    "assembly",
    "polymer_entity",
    "non_polymer_entity",
    "polymer_instance",
    "mol_definition",
)

METHOD_FACET_NAME = "experimental_method"
METHOD_FACET_ATTRIBUTE = "rcsb_entry_info.experimental_method"

# Legacy 4-character codes, the extended `pdb_0000xxxx` form, and computed
# structure model IDs (`AF_...`, `MA_...`), which the Data API also resolves.
# Anything else is not an entry identifier and is not worth a lookup.
_ENTRY_ID_RE = re.compile(r"^[A-Za-z0-9_]{4,32}$")

# Fields pulled for each hit. Kept deliberately small: this text goes into the
# model's context, and a wide selection would crowd out the hits themselves.
_ENTRY_FACTS_QUERY = """query($ids: [String!]!) {
  entries(entry_ids: $ids) {
    rcsb_id
    struct { title }
    rcsb_entry_info {
      resolution_combined
      experimental_method
      structure_determination_methodology
    }
    rcsb_accession_info { initial_release_date }
    nonpolymer_entities {
      nonpolymer_comp { chem_comp { id name formula_weight } }
    }
    rcsb_binding_affinity { comp_id type value unit }
  }
}"""

_ENTRY_TITLE_QUERY = """query($ids: [String!]!) {
  entries(entry_ids: $ids) { rcsb_id struct { title } }
}"""

# Per-hit truncation limits. These bound the worst case rather than the typical
# one: a ligand's `id` is the actionable handle, and its name only has to be
# recognizable enough to tell a drug-like molecule from a lipid or detergent.
_MAX_TITLE_CHARS = 80
_MAX_LIGAND_NAME_CHARS = 36
_MAX_LIGANDS_PER_ENTRY = 4
_MAX_AFFINITIES_PER_ENTRY = 2

# A full page of ligand-rich entries exceeds the 2,000-token default, and
# truncating it is worse than it sounds: the budget keeps the head and tail of
# the text, so a list of candidates loses its middle — silently, and precisely
# where the agent is choosing from. The caps above keep the worst case under
# this ceiling; test_a_full_page_fits_the_response_budget pins that down.
#
# The ceiling is only approached when the agent asks for all 25 rows and every
# hit is ligand-dense. A default page of 10 real hits costs about 1,200 tokens.
_MAX_RESPONSE_TOKENS = 5000


def _as_json_lines(records: list[dict[str, Any]]) -> str:
    """Render one compact JSON object per line.

    Indented JSON costs about a third more tokens for the same content, which
    is the difference between a full page of hits fitting the budget and not.
    """
    return "\n".join(json.dumps(record, separators=(",", ":")) for record in records)

# Common ions, buffers, and crystallization additives to exclude from "drug-like" ligands
# These are frequently present but not relevant for docking candidate selection
COMMON_NON_DRUG_LIGANDS = {
    # Ions
    'CA', 'MG', 'ZN', 'FE', 'MN', 'CU', 'NA', 'K', 'CL', 'BR', 'I',
    'NI', 'CO', 'CD', 'HG', 'PB', 'SR', 'BA', 'LI', 'CS', 'FE2', 'FE3',
    # Common buffers/additives
    'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'ACT', 'DMS', 'MPD',
    'TRS', 'MES', 'EPE', 'BME', 'CIT', 'IPA', 'ACE', 'FMT',
    'SCN', 'NO3', 'IOD', 'AZI',
    # Crystallization additives and PEG variants
    'PGE', 'PG4', 'PE4', 'P6G', '1PE', '2PE', 'PG0', 'PE8',
    # Water and common solvents
    'HOH', 'DOD', 'EOH', 'MOH',
}


# Metal ions that may be catalytically important and should optionally be kept
CATALYTIC_METAL_IONS = {
    'ZN', 'MG', 'CA', 'FE', 'MN', 'CU', 'NI', 'CO', 'FE2', 'FE3'
}

# Water molecules to always remove
WATER_RESIDUES = {'HOH', 'DOD'}


class RcsbSearchError(Exception):
    """An RCSB response the agent can act on.

    RCSB's own 4xx messages name the offending attribute or operator, so they
    are relayed verbatim rather than replaced with wording of our own.
    """


def _clamp_rows(value: Any) -> int:
    """Bound the page size, treating a non-positive request as unspecified.

    RCSB reads rows=0 as "facets only", but a model that emits it almost always
    meant "some rows"; return_counts covers the deliberate case.
    """
    try:
        rows = int(value)
    except (TypeError, ValueError):
        return DEFAULT_ROWS
    if rows < 1:
        return DEFAULT_ROWS
    return min(rows, MAX_ROWS)


def _clamp_start(value: Any) -> int:
    """Bound the page offset. RCSB rejects start + rows beyond 10,000."""
    try:
        start = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, min(start, 10_000 - MAX_ROWS))


def _apply_request_guards(request_options: Any) -> dict[str, Any]:
    """Bound a model-supplied `request_options` to what fits in context.

    Everything not named here passes through, so `sort`, `group_by` and
    `results_content_type` reach RCSB exactly as written.
    """
    options = copy.deepcopy(request_options) if isinstance(request_options, dict) else {}

    # `return_all_hits` ignores `paginate` and can return 10,000 identifiers.
    options.pop("return_all_hits", None)

    counts_only = bool(options.get("return_counts"))

    if not counts_only:
        paginate = options.get("paginate")
        paginate = dict(paginate) if isinstance(paginate, dict) else {}
        paginate["rows"] = _clamp_rows(paginate.get("rows"))
        paginate["start"] = _clamp_start(paginate.get("start"))
        options["paginate"] = paginate

        # Grouped results carry their members inside `group_set`, which the
        # compact verbosity flattens away. Only force compact when the response
        # is a plain list of identifiers.
        if options.get("group_by_return_type") != "groups":
            options["results_verbosity"] = "compact"

        # The method breakdown of the *whole* match set, not just this page. An
        # agent that filters to X-ray sees here how many cryo-EM entries that
        # cost it. Skipped when the agent asked for facets of its own, and
        # pointless alongside return_counts, which suppresses facets.
        if not options.get("facets"):
            options["facets"] = [
                {
                    "name": METHOD_FACET_NAME,
                    "aggregation_type": "terms",
                    "attribute": METHOD_FACET_ATTRIBUTE,
                }
            ]

    return options


async def _post_json(url: str, payload: dict[str, Any]) -> tuple[int, Any]:
    """POST JSON and return the status with the decoded body (None if empty)."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        response = await client.post(
            url, json=payload, headers={"Content-Type": "application/json"}
        )

    body: Any = None
    if response.content:
        try:
            body = response.json()
        except ValueError:
            body = None
    return response.status_code, body


async def _run_search(payload: dict[str, Any]) -> dict[str, Any] | None:
    """Execute a Search API query.

    Returns None for HTTP 204, which RCSB uses for "processed, no hits" -- an
    outcome, not a failure.
    """
    try:
        status, body = await _post_json(SEARCH_URL, payload)
    except httpx.HTTPError as exc:
        raise RcsbSearchError(f"Could not reach the RCSB Search API: {exc}") from exc

    if status == 204:
        return None

    if status >= 400:
        detail = ""
        if isinstance(body, dict):
            detail = str(body.get("message") or "").strip()
        raise RcsbSearchError(
            f"RCSB Search API rejected the query (HTTP {status})"
            + (f": {detail}" if detail else ".")
        )

    if not isinstance(body, dict):
        raise RcsbSearchError("RCSB Search API returned a response that was not JSON.")

    return body


async def _graphql(query: str, ids: list[str]) -> list[dict[str, Any]]:
    """Run a Data API GraphQL query over a batch of entry IDs."""
    status, body = await _post_json(DATA_GRAPHQL_URL, {"query": query, "variables": {"ids": ids}})

    if status >= 400 or not isinstance(body, dict):
        raise RcsbSearchError(f"RCSB Data API returned HTTP {status}.")

    errors = body.get("errors")
    if errors:
        first = errors[0].get("message") if isinstance(errors[0], dict) else str(errors[0])
        raise RcsbSearchError(f"RCSB Data API error: {first}")

    entries = (body.get("data") or {}).get("entries") or []
    return [entry for entry in entries if isinstance(entry, dict)]


def _first_resolution(entry_info: dict[str, Any]) -> float | None:
    """Return the entry's resolution, or None when it has none.

    NMR entries have no resolution at all. That is a fact about the entry, not
    a reason to drop it, so None is reported as None.
    """
    values = entry_info.get("resolution_combined") or []
    if not isinstance(values, list) or not values:
        return None
    try:
        return round(float(values[0]), 2)
    except (TypeError, ValueError):
        return None


def _drug_like_ligands(nonpolymer_entities: Any) -> list[dict[str, Any]]:
    """Extract ligands worth docking against, dropping ions and buffers."""
    ligands: list[dict[str, Any]] = []
    for npe in nonpolymer_entities or []:
        if not isinstance(npe, dict):
            continue
        chem_comp = ((npe.get("nonpolymer_comp") or {}).get("chem_comp")) or {}
        lig_id = (chem_comp.get("id") or "").strip()
        if not lig_id or lig_id.upper() in COMMON_NON_DRUG_LIGANDS:
            continue
        ligand: dict[str, Any] = {"id": lig_id, "name": (chem_comp.get("name") or "")[:_MAX_LIGAND_NAME_CHARS]}
        weight = chem_comp.get("formula_weight")
        if isinstance(weight, (int, float)):
            ligand["formula_weight"] = round(float(weight), 1)
        ligands.append(ligand)
        if len(ligands) >= _MAX_LIGANDS_PER_ENTRY:
            break
    return ligands


def _binding_affinities(raw: Any) -> list[dict[str, Any]]:
    """Summarize BindingDB measurements attached to the entry, if any."""
    affinities: list[dict[str, Any]] = []
    for item in raw or []:
        if not isinstance(item, dict) or item.get("value") is None:
            continue
        affinities.append(
            {
                "comp_id": item.get("comp_id"),
                "type": item.get("type"),
                "value": item.get("value"),
                "unit": item.get("unit"),
            }
        )
        if len(affinities) >= _MAX_AFFINITIES_PER_ENTRY:
            break
    return affinities


def _entry_facts(entry: dict[str, Any]) -> dict[str, Any]:
    """Flatten one Data API entry into the record the model sees."""
    entry_info = entry.get("rcsb_entry_info") or {}
    ligands = _drug_like_ligands(entry.get("nonpolymer_entities"))
    release = ((entry.get("rcsb_accession_info") or {}).get("initial_release_date") or "")

    facts: dict[str, Any] = {
        "pdb_id": entry.get("rcsb_id", ""),
        "title": ((entry.get("struct") or {}).get("title") or "")[:_MAX_TITLE_CHARS],
        "experimental_method": entry_info.get("experimental_method"),
        "resolution": _first_resolution(entry_info),
        "released": release[:10],
        "is_holo": bool(ligands),
        "ligands": ligands,
    }

    methodology = entry_info.get("structure_determination_methodology")
    if methodology and methodology != "experimental":
        facts["methodology"] = methodology

    affinities = _binding_affinities(entry.get("rcsb_binding_affinity"))
    if affinities:
        facts["binding_affinity"] = affinities

    return facts


async def _enrich_entries(pdb_ids: list[str]) -> list[dict[str, Any]]:
    """Attach per-entry facts to search hits, preserving the search order.

    The Data API answers in its own order, so results are re-keyed by ID. An
    entry the Data API does not know about still appears, as a bare ID.
    """
    queryable = [pdb_id for pdb_id in pdb_ids if _ENTRY_ID_RE.match(pdb_id)]
    if not queryable:
        return [{"pdb_id": pdb_id} for pdb_id in pdb_ids]

    entries = await _graphql(_ENTRY_FACTS_QUERY, queryable)
    by_id = {}
    for entry in entries:
        rcsb_id = entry.get("rcsb_id")
        if rcsb_id:
            by_id[str(rcsb_id).upper()] = _entry_facts(entry)

    return [by_id.get(pdb_id.upper(), {"pdb_id": pdb_id}) for pdb_id in pdb_ids]


def _result_identifiers(result_set: Any) -> list[str]:
    """Pull identifiers from a result set of either verbosity shape."""
    identifiers: list[str] = []
    for hit in result_set or []:
        if isinstance(hit, str):
            identifiers.append(hit)
        elif isinstance(hit, dict) and hit.get("identifier"):
            identifiers.append(str(hit["identifier"]))
    return identifiers


def _format_facets(facets: Any) -> str:
    """Render facet buckets as one line per facet."""
    lines: list[str] = []
    for facet in facets or []:
        if not isinstance(facet, dict):
            continue
        buckets = [b for b in (facet.get("buckets") or []) if isinstance(b, dict)]
        if not buckets:
            continue
        rendered = ", ".join(
            f"{bucket.get('label')}: {bucket.get('population')}" for bucket in buckets[:12]
        )
        lines.append(f"{facet.get('name', 'facet')} — {rendered}")
    return "\n".join(lines)


# Flattened search schema, populated on first use. The schema is ~1.4 MB and
# describes ~1900 attributes, so it is fetched lazily rather than at import:
# blocking application startup on a network call is the problem the previous
# rcsbapi-based implementation had to work around.
_ATTRIBUTE_INDEX: list[dict[str, Any]] | None = None
_ATTRIBUTE_INDEX_RETRY_AFTER = 0.0
_ATTRIBUTE_INDEX_RETRY_SECONDS = 300.0
_SCHEMA_TIMEOUT = 10.0
_SCALAR_TYPES = {"string", "number", "integer", "boolean", "date"}
_MAX_ATTRIBUTE_MATCHES = 25


def _flatten_schema(node: Any, path: str = "") -> list[dict[str, Any]]:
    """Walk the search schema into a flat list of dotted attribute paths.

    Arrays are transparent: an attribute inside an array of objects is
    addressed by the same dotted path as a plain nested object would be.
    """
    if not isinstance(node, dict):
        return []

    found: list[dict[str, Any]] = []

    properties = node.get("properties")
    if isinstance(properties, dict):
        for key, value in properties.items():
            found.extend(_flatten_schema(value, f"{path}.{key}" if path else key))

    if node.get("type") == "array" and isinstance(node.get("items"), dict):
        found.extend(_flatten_schema(node["items"], path))

    if path and node.get("type") in _SCALAR_TYPES:
        entry: dict[str, Any] = {"attribute": path, "type": node["type"]}
        description = node.get("description")
        if isinstance(description, str) and description.strip():
            entry["description"] = " ".join(description.split())[:160]
        enum = node.get("enum")
        if isinstance(enum, list) and enum:
            entry["allowed_values"] = [str(value) for value in enum[:20]]
        found.append(entry)

    return found


async def _attribute_index() -> list[dict[str, Any]] | None:
    """Return the flattened attribute index, or None if it cannot be fetched.

    Discovery is a convenience on top of a tool that works without it, so a
    failure degrades to None instead of failing the call. Failures are cached
    briefly so a network outage does not add a timeout to every request.
    """
    global _ATTRIBUTE_INDEX, _ATTRIBUTE_INDEX_RETRY_AFTER

    if _ATTRIBUTE_INDEX is not None:
        return _ATTRIBUTE_INDEX

    if time.monotonic() < _ATTRIBUTE_INDEX_RETRY_AFTER:
        return None

    try:
        async with httpx.AsyncClient(timeout=_SCHEMA_TIMEOUT) as client:
            response = await client.get(SEARCH_SCHEMA_URL)
        response.raise_for_status()
        schema = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Could not fetch the RCSB search schema: %s", exc)
        _ATTRIBUTE_INDEX_RETRY_AFTER = time.monotonic() + _ATTRIBUTE_INDEX_RETRY_SECONDS
        return None

    _ATTRIBUTE_INDEX = _flatten_schema(schema)
    logger.info("Indexed %d RCSB search attributes", len(_ATTRIBUTE_INDEX))
    return _ATTRIBUTE_INDEX


def _match_attributes(index: list[dict[str, Any]], needle: str) -> list[dict[str, Any]]:
    """Rank attributes matching `needle` in their path, then their description."""
    lowered = needle.lower()
    by_path = [item for item in index if lowered in item["attribute"].lower()]
    by_description = [
        item
        for item in index
        if lowered not in item["attribute"].lower()
        and lowered in item.get("description", "").lower()
    ]
    return (by_path + by_description)[:_MAX_ATTRIBUTE_MATCHES]


def _first_query_value(node: Any, depth: int = 0) -> str | None:
    """Find the first human-recognizable value in a query tree, for the UI."""
    if depth > 6 or not isinstance(node, dict):
        return None

    value = (node.get("parameters") or {}).get("value")
    if isinstance(value, str) and value.strip():
        return value.strip()[:40]
    if isinstance(value, (int, float)):
        return None

    for child in node.get("nodes") or []:
        found = _first_query_value(child, depth + 1)
        if found:
            return found
    return None


def remove_alternate_conformations(structure):
    """Remove non-primary alternate conformations using Biopython's native tools.
    
    Many PDB structures contain alternate conformations (altloc A, B, etc.) where
    crystallographers have modeled multiple positions for flexible atoms. When
    visualizing or analyzing structures, this can appear as "duplicate" atoms.
    
    This function uses Biopython's DisorderedAtom.disordered_remove() to keep only
    the currently selected conformation (by default, the one with highest occupancy)
    and removes all others. It also clears the altloc indicator so the output is clean.
    
    Args:
        structure: A Biopython Structure object (modified in place)
        
    Returns:
        int: Number of alternate conformations removed
    """
    removed_count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue.get_list():
                    if isinstance(atom, DisorderedAtom) and atom.is_disordered():
                        altlocs = list(atom.disordered_get_id_list())
                        if len(altlocs) > 1:
                            selected = atom.get_altloc()
                            # Remove all non-selected alternate conformations
                            for altloc in altlocs:
                                if altloc != selected:
                                    atom.disordered_remove(altloc)
                                    removed_count += 1
                            # Clear the altloc indicator on the remaining atom
                            atom.selected_child.altloc = ' '
    return removed_count


class CleanForDockingSelect(Select):
    """Biopython Select class that cleans PDB structures for docking or visualization.
    
    Always removes:
    - Water molecules (HOH, DOD) - interfere with docking calculations
    - Crystallization additives and buffers (GOL, EDO, PEG, etc.)
    
    Optionally removes (controlled by keep_ligands):
    - Drug-like ligands - removed for docking, kept for visualization/reference
    
    Optionally keeps (controlled by keep_metals):
    - Catalytically important metal ions (ZN, MG, etc.)
    
    Note: Alternate conformations are handled separately by remove_alternate_conformations()
    which should be called on the structure before saving.
    """
    
    def __init__(self, keep_metals: bool = False, keep_ligands: bool = False):
        """Initialize the selector.
        
        Args:
            keep_metals: If True, keeps catalytically important metal ions.
                        Default False for standard docking preparation.
            keep_ligands: If True, keeps drug-like ligands (non-buffer/non-ion HETATM).
                         Default False for docking preparation (ligands must be removed).
                         Set True to preserve native ligands for visualization or 
                         reference pose comparison.
        """
        self.keep_metals = keep_metals
        self.keep_ligands = keep_ligands
    
    def accept_residue(self, residue):
        """Accept residues based on configuration."""
        resname = residue.get_resname().strip()
        hetflag = residue.id[0]
        
        # Always keep standard amino acid residues (ATOM records)
        if hetflag == ' ':
            return True
        
        # Remove water (flagged as 'W' or HETATM with HOH/DOD)
        if hetflag == 'W' or resname in WATER_RESIDUES:
            return False
        
        # Handle hetero residues (ligands, ions, additives)
        if hetflag.startswith('H_'):
            # Check if it's a catalytically important metal ion
            if resname in CATALYTIC_METAL_IONS:
                return self.keep_metals
            
            # Check if it's a common buffer/additive (always remove these)
            if resname in COMMON_NON_DRUG_LIGANDS:
                return False
            
            # It's a drug-like ligand - keep or remove based on keep_ligands
            return self.keep_ligands
        
        return True


# Keep old class name for backwards compatibility, but use new logic
class NonWaterSelect(CleanForDockingSelect):
    """Backwards-compatible alias - now removes water AND ligands."""
    pass


class SearchRcsbTool(BaseTool):
    """Proxy the RCSB Search API, passing the agent's query through verbatim.

    The agent writes the RCSB query DSL directly, so any attribute and operator
    the Search API supports is reachable without a code change here. This tool
    adds no filters: what counts as a usable structure depends on the task, and
    the previous hard-coded X-ray/resolution/organism filters answered that
    question wrongly for every cryo-EM target.

    Hits are enriched from the Data API, because the Search API returns
    identifiers only. Results come back as a flat ranked list -- no apo/holo
    bucketing, no de-duplication -- alongside `total_count` and a breakdown of
    the match set by experimental method.

    Data Flow:
        Input: RCSB query object, return_type, request_options (text)
        Output: JSON list of hits with per-entry facts (text, no storage needed)
    """

    @property
    def name(self) -> str:
        return "search_rcsb"

    @property
    def summary_template(self) -> str:
        return "Searching RCSB PDB..."

    def format_summary(self, arguments: dict[str, Any]) -> str:
        """Name the thing being searched for, when the query makes it obvious.

        The query is an arbitrarily nested object, so there is no argument to
        interpolate. The first terminal value in it is usually the gene symbol,
        accession or free text the user would recognize.
        """
        value = _first_query_value(arguments.get("query"))
        if value:
            return f"Searching RCSB PDB for {value}..."
        return self.summary_template

    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL

    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.TEXT_ONLY

    @property
    def max_response_tokens(self) -> int:
        return _MAX_RESPONSE_TOKENS

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "search_rcsb",
                "description": (
                    "Search the RCSB PDB using the RCSB Search API query language. "
                    "You write the query; this tool adds NO filters of its own. Each hit "
                    "comes back with its experimental method, resolution, release date, "
                    "ligands and is_holo flag, plus the total match count and a breakdown "
                    "of the whole match set by experimental method.\n\n"
                    "**DO NOT filter by experimental method unless the task requires it.** "
                    "Most large or membrane proteins are solved only by cryo-EM: TRPV1 has "
                    "84 PDB entries, 80 of them cryo-EM. Search unfiltered first, read the "
                    "experimental_method breakdown, and narrow only if you must.\n\n"
                    "**QUERY SHAPE.** A terminal node matches one attribute:\n"
                    '{\"type\": \"terminal\", \"service\": \"text\", \"parameters\": '
                    '{\"attribute\": \"<attr>\", \"operator\": \"<op>\", \"value\": <val>}}\n'
                    "A group node combines them:\n"
                    '{\"type\": \"group\", \"logical_operator\": \"and\", \"nodes\": [ ... ]}\n'
                    "Add \"negation\": true to invert a terminal; \"case_sensitive\": true for "
                    "exact_match (useful for gene symbols, which are upper-case in human).\n\n"
                    "**SERVICES.** text (structure attributes), text_chem (chemical component "
                    "attributes), full_text (unstructured, use {\"value\": \"free text\"} with no "
                    "attribute), sequence, seqmotif, structure, strucmotif, chemical.\n\n"
                    "**OPERATORS.** exact_match, in (value is a list), contains_words, "
                    "contains_phrase, equals, greater, less, greater_or_equal, less_or_equal, "
                    "range ({\"from\": x, \"to\": y, \"include_lower\": true}), exists (no value).\n\n"
                    "**KEY ATTRIBUTES.**\n"
                    "- rcsb_entity_source_organism.rcsb_gene_name.value — gene symbol, e.g. 'EGFR'\n"
                    "- rcsb_entity_source_organism.taxonomy_lineage.name — organism, e.g. 'Homo sapiens'\n"
                    "- rcsb_entity_source_organism.taxonomy_lineage.id — taxon ID as a string, e.g. '9606'\n"
                    "- rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers"
                    ".database_accession — UniProt accession, e.g. 'P00533'\n"
                    "- rcsb_entry_info.experimental_method — one of: X-ray, EM, NMR, Neutron, "
                    "Multiple methods, Integrative, Other\n"
                    "- exptl.method — finer-grained: X-RAY DIFFRACTION, ELECTRON MICROSCOPY, "
                    "SOLUTION NMR, SOLID-STATE NMR, NEUTRON DIFFRACTION, ELECTRON CRYSTALLOGRAPHY\n"
                    "- rcsb_entry_info.resolution_combined — Angstroms; absent for NMR, so any "
                    "resolution filter silently excludes NMR entries\n"
                    "- rcsb_entry_info.structure_determination_methodology — experimental, "
                    "integrative, computational. Computed models are excluded from results "
                    'unless request_options.results_content_type includes "computational"\n'
                    "- rcsb_accession_info.initial_release_date — ISO date, e.g. '2020-01-01'\n"
                    "- rcsb_nonpolymer_entity_annotation.comp_id — bound ligand CCD code, e.g. 'STI'\n"
                    "- rcsb_entry_info.nonpolymer_entity_count — >0 selects entries with ligands\n"
                    "- rcsb_binding_affinity.value — measured affinity from BindingDB\n"
                    "- rcsb_entry_info.polymer_entity_count_protein — protein chain count\n"
                    "- rcsb_struct_symmetry.symbol, struct.title, rcsb_entry_info.assembly_count\n"
                    "That list is the common subset of roughly 1900 searchable attributes. "
                    "To search for others by keyword, call this tool with `find_attributes` "
                    "instead of `query` — e.g. find_attributes='membrane' or 'symmetry'. "
                    "An attribute or operator RCSB does not accept is not a dead end either: "
                    "it replies naming exactly what it rejected, so you can correct and retry.\n\n"
                    "**EXAMPLE 1 — every human EGFR structure, method-agnostic, best resolution first:**\n"
                    '{\"query\": {\"type\": \"group\", \"logical_operator\": \"and\", \"nodes\": ['
                    '{\"type\": \"terminal\", \"service\": \"text\", \"parameters\": {\"attribute\": '
                    '\"rcsb_entity_source_organism.rcsb_gene_name.value\", \"operator\": \"exact_match\", '
                    '\"value\": \"EGFR\", \"case_sensitive\": true}}, {\"type\": \"terminal\", \"service\": '
                    '\"text\", \"parameters\": {\"attribute\": '
                    '\"rcsb_entity_source_organism.taxonomy_lineage.name\", \"operator\": \"exact_match\", '
                    '\"value\": \"Homo sapiens\"}}]}, \"request_options\": {\"sort\": [{\"sort_by\": '
                    '\"rcsb_entry_info.resolution_combined\", \"direction\": \"asc\"}]}}\n\n'
                    "**EXAMPLE 2 — cryo-EM TRPV1 entries with a bound ligand, better than 3.5 A:**\n"
                    '{\"query\": {\"type\": \"group\", \"logical_operator\": \"and\", \"nodes\": ['
                    '{\"type\": \"terminal\", \"service\": \"text\", \"parameters\": {\"attribute\": '
                    '\"rcsb_entity_source_organism.rcsb_gene_name.value\", \"operator\": \"exact_match\", '
                    '\"value\": \"TRPV1\"}}, {\"type\": \"terminal\", \"service\": \"text\", \"parameters\": '
                    '{\"attribute\": \"rcsb_entry_info.experimental_method\", \"operator\": \"exact_match\", '
                    '\"value\": \"EM\"}}, {\"type\": \"terminal\", \"service\": \"text\", \"parameters\": '
                    '{\"attribute\": \"rcsb_entry_info.resolution_combined\", \"operator\": '
                    '\"less_or_equal\", \"value\": 3.5}}, {\"type\": \"terminal\", \"service\": \"text\", '
                    '\"parameters\": {\"attribute\": \"rcsb_entry_info.nonpolymer_entity_count\", '
                    '\"operator\": \"greater\", \"value\": 0}}]}}\n\n'
                    "Pass a chosen pdb_id to fetch_and_clean_pdb to download it. For docking, "
                    "prefer a hit with is_holo=true: fetch_and_clean_pdb derives binding site "
                    "coordinates from the ligand it removes."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "object",
                            "description": (
                                "An RCSB Search API query node, passed through verbatim. Either a "
                                "terminal node (type/service/parameters) or a group node "
                                "(type/logical_operator/nodes). See the examples above. Required "
                                "unless you are using find_attributes."
                            ),
                        },
                        "find_attributes": {
                            "type": "string",
                            "description": (
                                "Instead of searching, look up searchable attributes whose name or "
                                "description contains this keyword, and return their types and "
                                "allowed values. Use this when you need to filter on something the "
                                "key attribute list above does not cover. Pass this OR query, not both."
                            ),
                        },
                        "return_type": {
                            "type": "string",
                            "enum": list(RETURN_TYPES),
                            "description": (
                                "Granularity of the returned identifiers. Default 'entry', which "
                                "returns PDB IDs and is the only type enriched with per-entry "
                                "facts; the others return identifiers only. 'polymer_entity' gives "
                                "PDBID_N, 'assembly' gives PDBID-N, 'polymer_instance' gives "
                                "PDBID.CHAIN, 'mol_definition' gives ligand CCD codes."
                            ),
                        },
                        "request_options": {
                            "type": "object",
                            "description": (
                                "RCSB request_options, passed through verbatim. Supports:\n"
                                '- paginate: {\"start\": 0, \"rows\": 10} — rows is capped at '
                                f"{MAX_ROWS} (default {DEFAULT_ROWS}); page with start\n"
                                '- sort: [{\"sort_by\": \"<attribute>\", \"direction\": \"asc\"}] — '
                                "sorts server-side over the whole match set, not just this page. "
                                "Default is relevance score\n"
                                '- facets: [{\"name\": \"...\", \"aggregation_type\": \"terms\", '
                                '\"attribute\": \"...\"}] — counts across the match set. If you omit '
                                "this, a breakdown by experimental method is added for you\n"
                                '- results_content_type: [\"experimental\", \"computational\"] — add '
                                '\"computational\" to include AlphaFold DB / ModelArchive models\n'
                                '- group_by / group_by_return_type — de-duplicate by sequence '
                                "identity or UniProt accession\n"
                                "- return_counts: true — total count only, no hits"
                            ),
                        },
                    },
                    # Not ["query"]: find_attributes is the other valid way to
                    # call this tool. execute() enforces that one of them is set.
                    "required": [],
                },
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Run the agent's query against the RCSB Search API and enrich the hits."""
        needle = (arguments.get("find_attributes") or "").strip()
        if needle:
            return await self._find_attributes(needle)

        query = arguments.get("query")
        if not isinstance(query, dict) or not query:
            return ToolResult.error(
                "A 'query' object is required: an RCSB Search API terminal node "
                '{"type": "terminal", "service": "text", "parameters": {...}} or a '
                'group node {"type": "group", "logical_operator": "and", "nodes": [...]}. '
                "To look up attribute names instead, pass find_attributes."
            )

        return_type = (arguments.get("return_type") or ENTRY_RETURN_TYPE).strip()
        if return_type not in RETURN_TYPES:
            return ToolResult.error(
                f"Unknown return_type '{return_type}'. Valid values: {', '.join(RETURN_TYPES)}."
            )

        request_options = _apply_request_guards(arguments.get("request_options"))
        payload = {
            "query": query,
            "return_type": return_type,
            "request_options": request_options,
        }

        try:
            result = await _run_search(payload)
        except RcsbSearchError as exc:
            logger.warning("RCSB search failed: %s", exc)
            return ToolResult.error(str(exc))
        except Exception as exc:
            logger.exception("Failed to search RCSB PDB")
            return ToolResult.error(f"Failed to search RCSB PDB: {exc}")

        if result is None:
            return ToolResult.text(
                "No structures matched (total_count: 0). The query is valid but nothing "
                "in the PDB satisfies it. Loosen the most restrictive clause — an "
                "experimental method or resolution filter is the usual culprit — and retry."
            )

        total_count = result.get("total_count", 0)

        # Grouped-as-groups responses carry their own nested shape; report it
        # as-is rather than flattening away the grouping the agent asked for.
        if "group_set" in result:
            return self._format_groups(result, total_count)

        if request_options.get("return_counts"):
            return ToolResult.text(
                f"total_count: {total_count} entries match this query in the PDB.",
                metadata={"total_count": total_count},
            )

        identifiers = _result_identifiers(result.get("result_set"))
        if not identifiers:
            return ToolResult.text(
                f"total_count: {total_count}, but this page returned no hits. "
                "Lower paginate.start to page back into the result set."
            )

        hits: list[dict[str, Any]]
        enrichment_note = ""
        if return_type == ENTRY_RETURN_TYPE:
            try:
                hits = await _enrich_entries(identifiers)
            except (RcsbSearchError, httpx.HTTPError) as exc:
                # The search itself succeeded. Identifiers are still useful, so
                # hand them over rather than discarding a good result.
                logger.warning("RCSB Data API enrichment failed: %s", exc)
                hits = [{"pdb_id": pdb_id} for pdb_id in identifiers]
                enrichment_note = (
                    "\n\nNote: per-entry details are unavailable (the RCSB Data API did not "
                    f"respond: {exc}). The PDB IDs above are still valid."
                )
        else:
            hits = [{"identifier": identifier} for identifier in identifiers]

        return ToolResult.text(
            self._format_hits(
                hits=hits,
                total_count=total_count,
                shown=len(identifiers),
                start=(request_options.get("paginate") or {}).get("start", 0),
                return_type=return_type,
                facets=result.get("facets"),
                sorted_by_attribute=bool(request_options.get("sort")),
                subject=_first_query_value(query),
            )
            + enrichment_note,
            metadata={
                "total_count": total_count,
                "returned": len(identifiers),
                "return_type": return_type,
                "identifiers": identifiers,
            },
        )

    async def _find_attributes(self, needle: str) -> ToolResult:
        """Answer an attribute-discovery request from the live search schema."""
        index = await _attribute_index()
        if index is None:
            return ToolResult.text(
                "Attribute lookup is unavailable right now (the RCSB search schema could "
                "not be fetched). Use the attributes listed in this tool's description, or "
                "try a query and read RCSB's error if it rejects the attribute."
            )

        matches = _match_attributes(index, needle)
        if not matches:
            return ToolResult.text(
                f"No searchable attribute mentions '{needle}'. Try a broader keyword — "
                "'resolution', 'organism', 'ligand', 'symmetry', 'method' — or search the "
                f"{len(index)} attributes by a word from the annotation you have in mind."
            )

        return ToolResult.text(
            f"{len(matches)} searchable attributes matching '{needle}':\n\n"
            f"{json.dumps(matches, indent=2)}\n\n"
            "Use one as the 'attribute' of a terminal node. Numeric attributes take "
            "equals/greater/less/range; string attributes take exact_match/in/contains_words. "
            "Where allowed_values is listed, the value must be one of them.",
            metadata={"matched": len(matches), "indexed": len(index)},
        )

    def _format_hits(
        self,
        hits: list[dict[str, Any]],
        total_count: int,
        shown: int,
        start: int,
        return_type: str,
        facets: Any,
        sorted_by_attribute: bool,
        subject: str | None,
    ) -> str:
        """Render the result header, facet breakdown and hit list.

        The header names the search subject because the query is a nested
        object: the findings ledger drops arguments too long to be a handle, so
        without this a recorded search says nothing about what was searched for.
        """
        header = f"Searched for '{subject}'. " if subject else ""
        header += f"total_count: {total_count} — showing {shown}"
        if start:
            header += f" starting at {start}"
        if start + shown < total_count:
            header += f" (page on with request_options.paginate.start={start + shown}"
            # Without a sort these are the highest-scoring hits, not the best by
            # any property, so paging is the only way to see the rest. With one,
            # this page really is the top of the whole match set.
            header += (
                ")"
                if sorted_by_attribute
                else "; these are the top hits by relevance score, so add "
                "request_options.sort to rank the whole match set by resolution or date)"
            )
        header += f". return_type: {return_type}."

        facet_text = _format_facets(facets)
        if facet_text:
            header += f"\n\nMatch set breakdown:\n{facet_text}"

        body = _as_json_lines(hits)

        if return_type != ENTRY_RETURN_TYPE:
            return (
                f"{header}\n\nOne JSON object per line:\n{body}\n\n"
                "These identifiers are not plain PDB IDs, so no per-entry details were "
                "fetched. Re-run with return_type='entry' for method, resolution and ligands."
            )

        return (
            f"{header}\n\nOne JSON object per line:\n{body}\n\n"
            "resolution is null for entries that have none (NMR, and some integrative "
            "models) — that is a property of the method, not a missing value. "
            "Pass a pdb_id to fetch_and_clean_pdb to download it."
        )

    def _format_groups(self, result: dict[str, Any], total_count: int) -> ToolResult:
        """Render a grouped response, preserving group membership."""
        groups = []
        for group in result.get("group_set") or []:
            if not isinstance(group, dict):
                continue
            members = _result_identifiers(group.get("result_set"))
            representative = (group.get("representative") or {}).get("identifier")
            groups.append(
                {
                    "group_id": group.get("identifier"),
                    "count": group.get("count", len(members)),
                    "representative": representative,
                    "members": members[:MAX_ROWS],
                }
            )

        group_by = result.get("group_by") or {}
        header = (
            f"total_count: {total_count} hits in {result.get('group_by_count', len(groups))} "
            f"groups (aggregation_method: {group_by.get('aggregation_method', 'unknown')}). "
            f"Showing {len(groups)} groups."
        )

        return ToolResult.text(
            f"{header}\n\nOne JSON object per line:\n{_as_json_lines(groups)}\n\n"
            "Each group is a set of redundant hits. Take one representative per group for a "
            "non-redundant set, or set group_by_return_type='representatives' to get that "
            "directly with full per-entry details.",
            metadata={"total_count": total_count, "group_count": len(groups)},
        )


class FetchAndCleanPdbTool(BaseTool):
    """Fetch and clean a PDB structure for docking.
    
    This tool downloads the biological assembly of a PDB structure (not just
    the asymmetric unit) and removes water molecules to prepare it for docking.
    The cleaned structure is stored and a reference is returned.
    
    The biological assembly is important because:
    - The asymmetric unit may not represent the functional oligomeric state
    - Docking to the wrong assembly could give misleading results
    
    Water removal is standard docking preparation:
    - Water molecules interfere with docking calculations
    - Some explicit waters can be kept for specialized analyses
    
    Data Flow:
        Input: PDB ID, assembly number (text)
        Output: Stores cleaned PDB, returns reference
    """
    
    @property
    def name(self) -> str:
        return "fetch_and_clean_pdb"
    
    @property
    def summary_template(self) -> str:
        return "Fetching and cleaning PDB structure {pdb_id}..."
    
    @property
    def tool_type(self) -> ToolType:
        return ToolType.LOCAL
    
    @property
    def data_io_category(self) -> DataIOCategory:
        return DataIOCategory.OUTPUT_STORES
    
    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "fetch_and_clean_pdb",
                "description": (
                    "Download a PDB structure from RCSB and prepare it for use. "
                    "Downloads the biological assembly (not asymmetric unit), removes "
                    "water molecules and crystallization additives. By default, also "
                    "removes drug-like ligands (for docking preparation).\n\n"
                    "**TWO MODES:**\n"
                    "1. **Docking mode (default, keep_ligands=false):** Removes all ligands. "
                    "Returns BINDING SITE COORDINATES from removed ligands for use with vina_dock.\n"
                    "2. **Holo mode (keep_ligands=true):** Keeps drug-like ligands in place. "
                    "Use this to visualize the native ligand pose, compare docked poses to "
                    "crystal poses, or study protein-ligand interactions.\n\n"
                    "**CRITICAL FOR DOCKING:** When keep_ligands=false (default), this tool "
                    "returns binding site coordinates ONLY if the structure had drug-like "
                    "ligands that were removed. For docking, ALWAYS use a HOLO structure.\n\n"
                    "**TIP FOR VISUALIZATION:** Set keep_ligands=true to see how the native "
                    "ligand binds. This is useful for understanding binding modes before docking.\n\n"
                    "Use this after search_rcsb to download a selected structure. "
                    "The returned reference can be passed to visualize_pdb or vina_dock."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pdb_id": {
                            "type": "string",
                            "description": (
                                "4-character PDB ID (e.g., '4I22', '2ITY'). "
                                "Can be obtained from search_rcsb."
                            ),
                        },
                        "assembly_id": {
                            "type": "integer",
                            "description": (
                                "Biological assembly number. Default is 1 (the most "
                                "common biological assembly). Use higher numbers only "
                                "if you need a specific assembly."
                            ),
                        },
                        "keep_ligands": {
                            "type": "boolean",
                            "description": (
                                "Whether to keep drug-like ligands in the structure. "
                                "Default is false (remove ligands for docking preparation).\n\n"
                                "Set to TRUE when you want to:\n"
                                "- Visualize the native ligand bound to the protein\n"
                                "- Compare a docked pose to the crystal pose\n"
                                "- Study protein-ligand interactions in the native state\n\n"
                                "Set to FALSE (default) when you want to:\n"
                                "- Prepare a receptor for docking with vina_dock\n"
                                "- Get binding site coordinates from the removed ligand\n\n"
                                "Note: Water and crystallization additives (buffers, ions) "
                                "are always removed regardless of this setting."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Short description of the structure for reference tracking "
                                "(e.g., 'COX-2 receptor cleaned for docking'). "
                                "This helps identify the stored structure later."
                            ),
                        },
                    },
                    "required": ["pdb_id"],
                },
            },
        }
    
    async def _get_entry_title(self, pdb_id: str) -> str:
        """Look up the entry title, returning "" if it cannot be determined.

        The title is descriptive metadata only, so a failure here must not stop
        the download that follows.
        """
        try:
            entries = await _graphql(_ENTRY_TITLE_QUERY, [pdb_id])
        except (RcsbSearchError, httpx.HTTPError) as exc:
            logger.warning("Could not fetch title for %s: %s", pdb_id, exc)
            return ""
        if not entries:
            return ""
        return (entries[0].get("struct") or {}).get("title") or ""
    
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Fetch and clean a PDB structure."""
        pdb_id = arguments.get("pdb_id", "").strip().upper()
        assembly_id = arguments.get("assembly_id", 1)
        keep_ligands = arguments.get("keep_ligands", False)
        description = arguments.get("description")
        
        if not pdb_id or len(pdb_id) != 4:
            return ToolResult.error("Valid 4-character PDB ID is required")
        
        if self._storage is None:
            return ToolResult.error("Storage service not available")
        
        try:
            ligand_mode = "keeping ligands" if keep_ligands else "removing ligands"
            logger.info(f"Fetching PDB {pdb_id} assembly {assembly_id} ({ligand_mode})")
            
            title = await self._get_entry_title(pdb_id)
            
            # Download biological assembly
            # Format: {pdb_id}.pdb{assembly_id}.gz for PDB format biological assembly
            assembly_url = f"https://files.rcsb.org/download/{pdb_id}.pdb{assembly_id}.gz"
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(assembly_url)
                
                if response.status_code == 404:
                    # Try alternative URL format for CIF
                    cif_url = f"https://files.rcsb.org/download/{pdb_id}-assembly{assembly_id}.cif.gz"
                    response = await client.get(cif_url)
                    
                    if response.status_code == 404:
                        return ToolResult.error(
                            f"PDB structure {pdb_id} not found or assembly {assembly_id} "
                            f"does not exist. Try assembly_id=1 or verify the PDB ID."
                        )
                
                response.raise_for_status()
            
            # Decompress
            pdb_content = gzip.decompress(response.content).decode('utf-8')
            
            logger.info(f"Downloaded {len(pdb_content)} bytes for {pdb_id}")
            
            # Parse and clean with Biopython
            parser = PDBParser(QUIET=True)
            
            # Write to temp file for parsing (PDBParser needs a file)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                f.write(pdb_content)
                temp_input = f.name
            
            try:
                structure = parser.get_structure(pdb_id, temp_input)
            finally:
                import os
                os.unlink(temp_input)
            
            # Remove alternate conformations (altloc A, B, etc.) to keep only primary
            # This prevents "duplicate" atoms in visualization and analysis
            altlocs_removed = remove_alternate_conformations(structure)
            if altlocs_removed > 0:
                logger.info(f"Removed {altlocs_removed} alternate conformations from {pdb_id}")
            
            # Count waters, ligands, and atoms before cleaning
            # Also extract center of mass for significant ligands (useful for docking)
            water_count = 0
            ligand_count = 0
            ligand_names = []
            ligand_binding_sites = {}  # Dict of ligand centers for docking guidance
            total_atoms_before = 0
            chains = set()
            
            for model in structure:
                for chain in model:
                    chains.add(chain.id)
                    for residue in chain:
                        total_atoms_before += len(residue)
                        resname = residue.get_resname().strip()
                        hetflag = residue.id[0]
                        resnum = residue.id[1]
                        chain_id = chain.id
                        
                        if resname in WATER_RESIDUES or hetflag == 'W':
                            water_count += 1
                        elif hetflag.startswith('H_'):
                            # It's a ligand or ion
                            ligand_count += 1
                            
                            # For significant ligands (not ions/buffers), compute center of mass
                            if resname not in COMMON_NON_DRUG_LIGANDS:
                                ligand_names.append(resname)
                                
                                # Compute center of mass for this ligand
                                coords = [atom.coord for atom in residue]
                                if coords:
                                    coords_array = np.array(coords)
                                    center = coords_array.mean(axis=0)
                                    
                                    lig_id = f"{resname}_{chain_id}_{resnum}"
                                    ligand_binding_sites[lig_id] = {
                                        "name": resname,
                                        "chain": chain_id,
                                        "resnum": resnum,
                                        "center_x": round(float(center[0]), 2),
                                        "center_y": round(float(center[1]), 2),
                                        "center_z": round(float(center[2]), 2),
                                        "num_atoms": len(coords)
                                    }
            
            # Write cleaned structure
            io_out = PDBIO()
            io_out.set_structure(structure)
            
            output_buffer = io.StringIO()
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                temp_output = f.name
            
            try:
                # Use selector configured based on keep_ligands setting
                selector = CleanForDockingSelect(keep_ligands=keep_ligands)
                io_out.save(temp_output, selector)
                with open(temp_output, 'r') as f:
                    cleaned_content = f.read()
            finally:
                import os
                os.unlink(temp_output)
            
            # Count atoms after cleaning
            parser_clean = PDBParser(QUIET=True)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as f:
                f.write(cleaned_content)
                temp_clean = f.name
            
            try:
                structure_clean = parser_clean.get_structure(f"{pdb_id}_clean", temp_clean)
                total_atoms_after = sum(
                    len(residue)
                    for model in structure_clean
                    for chain in model
                    for residue in chain
                )
            finally:
                import os
                os.unlink(temp_clean)
            
            # Store the cleaned structure
            metadata = {
                "pdb_id": pdb_id,
                "assembly_id": assembly_id,
                "title": title[:200] if title else "",
                "chains": list(chains),
                "water_molecules_removed": water_count,
                "keep_ligands": keep_ligands,
                "ligands_removed": 0 if keep_ligands else ligand_count,
                "ligands_kept": ligand_count if keep_ligands else 0,
                "ligand_names": list(set(ligand_names)),
                "ligand_binding_sites": ligand_binding_sites,  # Centers of mass for docking
                "atoms_before_cleaning": total_atoms_before,
                "atoms_after_cleaning": total_atoms_after,
            }
            
            reference = await self._storage.store(
                data_type="pdb",
                content=cleaned_content.encode("utf-8"),
                metadata=metadata,
                description=description,
            )
            
            mode_str = "holo (ligands kept)" if keep_ligands else "docking (ligands removed)"
            logger.info(f"Stored PDB {pdb_id} as {reference} [{mode_str}]")
            
            # Build response message
            message = format_storage_message(
                data_type="pdb",
                reference=reference,
                source_description=f"{pdb_id} (assembly {assembly_id})",
                accepting_tools=["visualize_pdb", "vina_dock"],
            )
            
            # Build detailed message based on mode
            mode_description = "with ligands preserved (holo mode)" if keep_ligands else "for docking (ligands removed)"
            
            message += (
                f"\n\nStructure details ({mode_description}):\n"
                f"- PDB ID: {pdb_id}\n"
                f"- Assembly: {assembly_id}\n"
                f"- Title: {title[:80]}{'...' if len(title) > 80 else ''}\n"
                f"- Chains: {', '.join(sorted(chains))}\n"
                f"- Water molecules removed: {water_count}\n"
            )
            
            if keep_ligands:
                # Holo mode - ligands were kept
                if ligand_names:
                    message += f"- Ligands KEPT: {', '.join(set(ligand_names))}\n"
                message += f"- Atoms after cleaning: {total_atoms_after:,} (protein + ligands)"
                
                # Provide ligand location info for reference
                if ligand_binding_sites:
                    message += "\n\n" + "=" * 50
                    message += "\n**NATIVE LIGAND LOCATIONS (for reference)**"
                    message += "\n" + "=" * 50
                    message += "\nLigand center-of-mass coordinates:\n"
                    
                    for lig_id, info in ligand_binding_sites.items():
                        message += (
                            f"\n{info['name']} (chain {info['chain']}, residue {info['resnum']}):\n"
                            f"  center: ({info['center_x']}, {info['center_y']}, {info['center_z']})\n"
                            f"  atoms: {info['num_atoms']}\n"
                        )
                    
                    message += (
                        "\n**NOTE:** To dock new ligands to this structure, fetch it again "
                        "with keep_ligands=false to remove native ligands and get binding "
                        "site coordinates for vina_dock."
                    )
            else:
                # Docking mode - ligands were removed
                if ligand_names:
                    message += f"- Ligands REMOVED: {', '.join(set(ligand_names))}\n"
                message += f"- Atoms after cleaning: {total_atoms_after:,} (protein only)"
                
                # Add binding site information prominently if ligands were removed
                if ligand_binding_sites:
                    message += "\n\n" + "=" * 50
                    message += "\n**BINDING SITE COORDINATES FOR DOCKING**"
                    message += "\n" + "=" * 50
                    message += "\nUse these EXACT values in vina_dock:\n"
                    
                    for lig_id, info in ligand_binding_sites.items():
                        message += (
                            f"\n{info['name']} binding site (chain {info['chain']}, residue {info['resnum']}):\n"
                            f"  center_x = {info['center_x']}\n"
                            f"  center_y = {info['center_y']}\n"
                            f"  center_z = {info['center_z']}\n"
                        )
                    
                    # Add explicit guidance
                    first_site = list(ligand_binding_sites.values())[0]
                    message += (
                        f"\n**RECOMMENDED:** For vina_dock, use the {first_site['name']} binding site:\n"
                        f"  vina_dock(..., center_x={first_site['center_x']}, center_y={first_site['center_y']}, center_z={first_site['center_z']})"
                    )
            
            return ToolResult.data_reference(
                reference=reference,
                data_type="pdb",
                message=message,
                metadata=metadata,
            )
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching PDB {pdb_id}: {e}")
            return ToolResult.error(
                f"Failed to download PDB {pdb_id}: HTTP {e.response.status_code}"
            )
        except Exception as e:
            logger.exception(f"Failed to fetch and clean PDB {pdb_id}")
            return ToolResult.error(f"Failed to fetch and clean PDB: {str(e)}")
